"""Rule-based vs Ollama sentiment benchmark. **Skeleton -- no model calls yet.**

The adoption rule (CLAUDE.md 4) is: keep the rule-based scorer unless a local
model is *measurably* better. "Measurably" needs a labelled set and a metric,
which is what this file is. It runs end to end today against the rule-based
scorer alone; the Ollama column stays empty until
:class:`~nfl.data.sentiment.ollama_scorer.OllamaScorer` is implemented and
explicitly enabled.

Labelled set format (JSONL, one object per line)::

    {"text": "Mahomes ruled out with ankle injury", "team": "KC",
     "label_score": -0.8, "label_category": "injury"}

Metrics reported per scorer:

* ``mae`` -- mean absolute error against ``label_score``. The headline number.
* ``sign_accuracy`` -- fraction where the sign matches. Direction is what the
  downstream feature actually consumes, so a scorer can win on MAE and still
  lose here.
* ``category_accuracy`` -- exact category match, on labelled examples only.
* ``market_leak_rate`` -- fraction of market-labelled examples given a non-zero
  score. Must be 0.0 for any scorer to be adoptable (ADR 0001).
* ``seconds_per_document`` -- the rule-based scorer is ~free; a local 8B model
  is not, and a pregame refresh scores thousands of documents.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nfl.data.sentiment.base import Document, DocumentScore, DocumentScorer
from nfl.data.sentiment.news_rules import RuleBasedScorer
from nfl.data.sentiment.ollama_scorer import OllamaScorer

log = logging.getLogger(__name__)

#: A scorer must not beat the incumbent by less than this on MAE to replace it.
#: Ties go to the rule-based scorer: it is deterministic, auditable and free.
ADOPTION_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class LabeledExample:
    text: str
    label_score: float
    label_category: str | None = None
    team: str | None = None
    example_id: str = ""

    def to_document(self) -> Document:
        return Document(text=self.text, team=self.team, doc_id=self.example_id, source="benchmark")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    scorer: str
    version: str
    n: int
    mae: float
    sign_accuracy: float
    category_accuracy: float | None
    market_leak_rate: float | None
    seconds_per_document: float
    errors: int = 0
    per_example: tuple[tuple[str, float, float], ...] = field(default=())

    def summary(self) -> str:
        category = "n/a" if self.category_accuracy is None else f"{self.category_accuracy:.3f}"
        return (
            f"{self.scorer} v{self.version}: n={self.n} mae={self.mae:.4f} "
            f"sign={self.sign_accuracy:.3f} category={category} "
            f"{self.seconds_per_document * 1000:.2f} ms/doc"
        )


def load_labeled_set(path: Path) -> list[LabeledExample]:
    """Read a JSONL labelled set. Malformed lines are skipped with a warning."""
    examples: list[LabeledExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                examples.append(
                    LabeledExample(
                        text=str(record["text"]),
                        label_score=float(record["label_score"]),
                        label_category=record.get("label_category"),
                        team=record.get("team"),
                        example_id=str(record.get("id", lineno)),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("benchmark: skipping %s:%d (%s)", path, lineno, exc)
    return examples


def evaluate(scorer: DocumentScorer, examples: Sequence[LabeledExample]) -> BenchmarkResult:
    """Score every example and compute the metrics. No I/O beyond the scorer."""
    if not examples:
        raise ValueError("labelled set is empty")

    scores: list[DocumentScore] = []
    errors = 0
    started = time.perf_counter()
    for example in examples:
        try:
            scores.append(scorer.score_document(example.to_document()))
        except Exception as exc:  # noqa: BLE001 - a flaky scorer must not abort the run
            log.warning("benchmark: %s failed on %r (%s)", scorer.name, example.example_id, exc)
            scores.append(DocumentScore(score=0.0, category="neutral", confidence=0.0))
            errors += 1
    elapsed = time.perf_counter() - started

    absolute = [abs(s.score - e.label_score) for s, e in zip(scores, examples, strict=True)]
    signs = [
        _sign(s.score) == _sign(e.label_score) for s, e in zip(scores, examples, strict=True)
    ]
    labelled = [(s, e) for s, e in zip(scores, examples, strict=True) if e.label_category]
    category_accuracy = (
        sum(1 for s, e in labelled if s.category == e.label_category) / len(labelled)
        if labelled
        else None
    )
    market = [(s, e) for s, e in labelled if e.label_category == "market"]
    market_leak_rate = (
        sum(1 for s, _ in market if abs(s.score) > 1e-9) / len(market) if market else None
    )

    return BenchmarkResult(
        scorer=scorer.name,
        version=scorer.version,
        n=len(examples),
        mae=sum(absolute) / len(absolute),
        sign_accuracy=sum(signs) / len(signs),
        category_accuracy=category_accuracy,
        market_leak_rate=market_leak_rate,
        seconds_per_document=elapsed / len(examples),
        errors=errors,
        per_example=tuple(
            (e.example_id, e.label_score, s.score) for e, s in zip(examples, scores, strict=True)
        ),
    )


def _sign(value: float, band: float = 0.05) -> int:
    if value > band:
        return 1
    if value < -band:
        return -1
    return 0


@dataclass(frozen=True, slots=True)
class Comparison:
    rules: BenchmarkResult
    ollama: BenchmarkResult | None
    verdict: str

    def report(self) -> str:
        lines = [self.rules.summary()]
        if self.ollama is not None:
            lines.append(self.ollama.summary())
        lines.append(f"verdict: {self.verdict}")
        return "\n".join(lines)


def compare(
    examples: Sequence[LabeledExample],
    *,
    rules: DocumentScorer | None = None,
    ollama: DocumentScorer | None = None,
) -> Comparison:
    """Run both scorers and apply the adoption rule.

    The Ollama side is skipped -- not failed -- when it is disabled, which is the
    default and the state this file ships in.
    """
    rules_scorer: DocumentScorer = rules or RuleBasedScorer()
    rules_result = evaluate(rules_scorer, examples)

    ollama_scorer: DocumentScorer = ollama or OllamaScorer()
    if not ollama_scorer.enabled:
        return Comparison(
            rules=rules_result,
            ollama=None,
            verdict="keep rules (ollama scorer disabled; set NFL_ENABLE_OLLAMA=1 to benchmark)",
        )

    ollama_result = evaluate(ollama_scorer, examples)
    return Comparison(rules=rules_result, ollama=ollama_result, verdict=verdict(rules_result, ollama_result))


def verdict(rules: BenchmarkResult, ollama: BenchmarkResult) -> str:
    """Adoption rule: Ollama must clear :data:`ADOPTION_MARGIN` and leak nothing."""
    if ollama.market_leak_rate:
        return "keep rules (ollama scored market-labelled documents non-zero -- ADR 0001)"
    if ollama.errors:
        return f"keep rules (ollama errored on {ollama.errors}/{ollama.n} documents)"
    improvement = rules.mae - ollama.mae
    if improvement > ADOPTION_MARGIN:
        return f"ollama better by {improvement:.4f} MAE (> {ADOPTION_MARGIN} margin) -- candidate"
    return f"keep rules (ollama improvement {improvement:.4f} <= {ADOPTION_MARGIN} margin)"


def run(path: Path) -> Comparison:
    """Load a labelled set from ``path`` and compare the scorers."""
    return compare(load_labeled_set(path))


__all__ = [
    "ADOPTION_MARGIN",
    "BenchmarkResult",
    "Comparison",
    "LabeledExample",
    "compare",
    "evaluate",
    "load_labeled_set",
    "run",
    "verdict",
]
