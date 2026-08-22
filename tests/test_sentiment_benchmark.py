"""Benchmark skeleton + Ollama stub. No model is ever called here."""

from __future__ import annotations

import pytest

from nfl.data.sentiment import benchmark, ollama_scorer
from nfl.data.sentiment.base import Document, DocumentScore
from nfl.data.sentiment.news_rules import RuleBasedScorer

LABELS = "\n".join(
    [
        '{"id": "1", "text": "QB out for the season with a torn ACL", "team": "KC", '
        '"label_score": -0.9, "label_category": "injury"}',
        '{"id": "2", "text": "WR cleared to play Sunday", "team": "KC", '
        '"label_score": 0.6, "label_category": "return"}',
        '{"id": "3", "text": "Chiefs open as a 7.5-point favorite", "team": "KC", '
        '"label_score": 0.0, "label_category": "market"}',
        "# a comment line",
        "",
        "{not json}",
    ]
)


@pytest.fixture
def labeled_path(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(LABELS, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Labelled set
# --------------------------------------------------------------------------- #


def test_load_labeled_set_skips_junk_lines(labeled_path):
    examples = benchmark.load_labeled_set(labeled_path)
    assert len(examples) == 3
    assert examples[0].label_category == "injury"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def test_evaluate_rule_based_scorer(labeled_path):
    result = benchmark.evaluate(RuleBasedScorer(), benchmark.load_labeled_set(labeled_path))

    assert result.scorer == "rules"
    assert result.n == 3
    assert 0.0 <= result.mae <= 2.0
    assert result.sign_accuracy == 1.0
    assert result.errors == 0
    assert result.seconds_per_document >= 0.0
    assert "rules" in result.summary()


def test_market_leak_rate_is_zero_for_the_rule_based_scorer(labeled_path):
    result = benchmark.evaluate(RuleBasedScorer(), benchmark.load_labeled_set(labeled_path))
    assert result.market_leak_rate == 0.0


def test_evaluate_rejects_an_empty_set():
    with pytest.raises(ValueError, match="empty"):
        benchmark.evaluate(RuleBasedScorer(), [])


def test_a_failing_scorer_is_counted_not_fatal(labeled_path):
    class Exploding:
        name = "boom"
        version = "0"
        enabled = True

        def score_document(self, document: Document) -> DocumentScore:
            raise RuntimeError("model died")

        def score_documents(self, documents):  # pragma: no cover - unused
            return [self.score_document(d) for d in documents]

    result = benchmark.evaluate(Exploding(), benchmark.load_labeled_set(labeled_path))
    assert result.errors == 3


# --------------------------------------------------------------------------- #
# Comparison / adoption rule
# --------------------------------------------------------------------------- #


def test_compare_skips_ollama_when_disabled(labeled_path, monkeypatch):
    monkeypatch.delenv(ollama_scorer.ENABLE_ENV, raising=False)
    comparison = benchmark.compare(benchmark.load_labeled_set(labeled_path))

    assert comparison.ollama is None
    assert "keep rules" in comparison.verdict
    assert "keep rules" in comparison.report()


def _result(mae: float, **kwargs) -> benchmark.BenchmarkResult:
    defaults = dict(
        scorer="ollama",
        version="0",
        n=10,
        mae=mae,
        sign_accuracy=0.9,
        category_accuracy=0.9,
        market_leak_rate=0.0,
        seconds_per_document=0.5,
        errors=0,
    )
    defaults.update(kwargs)
    return benchmark.BenchmarkResult(**defaults)


def test_ties_go_to_the_rule_based_scorer():
    rules = _result(0.20, scorer="rules")
    assert "keep rules" in benchmark.verdict(rules, _result(0.19))
    assert "keep rules" in benchmark.verdict(rules, _result(0.20))


def test_clear_win_makes_ollama_a_candidate():
    assert "candidate" in benchmark.verdict(_result(0.30, scorer="rules"), _result(0.10))


def test_market_leak_disqualifies_regardless_of_accuracy():
    verdict = benchmark.verdict(_result(0.90, scorer="rules"), _result(0.01, market_leak_rate=0.5))
    assert "ADR 0001" in verdict


def test_errors_disqualify():
    verdict = benchmark.verdict(_result(0.90, scorer="rules"), _result(0.01, errors=2))
    assert "keep rules" in verdict


def test_run_loads_and_compares(labeled_path, monkeypatch):
    monkeypatch.delenv(ollama_scorer.ENABLE_ENV, raising=False)
    assert benchmark.run(labeled_path).rules.n == 3


# --------------------------------------------------------------------------- #
# Ollama stub
# --------------------------------------------------------------------------- #


def test_ollama_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ollama_scorer.ENABLE_ENV, raising=False)
    scorer = ollama_scorer.OllamaScorer()
    assert scorer.enabled is False
    with pytest.raises(ollama_scorer.OllamaDisabled):
        scorer.score_document(Document(text="anything"))


def test_enabled_scorer_is_still_a_stub(monkeypatch):
    monkeypatch.setenv(ollama_scorer.ENABLE_ENV, "1")
    scorer = ollama_scorer.OllamaScorer()
    assert scorer.enabled is True
    with pytest.raises(NotImplementedError):
        scorer.score_document(Document(text="anything"))


def test_non_local_host_is_refused(monkeypatch):
    monkeypatch.setenv(ollama_scorer.ENABLE_ENV, "1")
    monkeypatch.setenv(ollama_scorer.HOST_ENV, "https://ollama.example.com")
    with pytest.raises(ollama_scorer.OllamaNotLocal):
        ollama_scorer.OllamaScorer()


def test_localhost_is_accepted(monkeypatch):
    monkeypatch.setenv(ollama_scorer.ENABLE_ENV, "1")
    monkeypatch.setenv(ollama_scorer.HOST_ENV, "http://127.0.0.1:11434")
    assert ollama_scorer.OllamaScorer().host == "http://127.0.0.1:11434"


def test_prompt_and_payload_are_pure(monkeypatch):
    monkeypatch.delenv(ollama_scorer.ENABLE_ENV, raising=False)
    scorer = ollama_scorer.OllamaScorer(model="llama3.1:8b")
    document = Document(text="QB ruled out", team="KC")

    prompt = scorer.build_prompt(document)
    assert "QB ruled out" in prompt and "KC" in prompt

    payload = scorer.request_payload(document)
    assert payload["model"] == "llama3.1:8b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0


def test_parse_response_handles_prose_around_json():
    score = ollama_scorer.OllamaScorer.parse_response(
        'Sure! {"score": -0.7, "category": "injury", "confidence": 0.8} Hope that helps.'
    )
    assert score.score == -0.7
    assert score.category == "injury"


def test_parse_response_degrades_gracefully():
    score = ollama_scorer.OllamaScorer.parse_response("I refuse to answer.")
    assert score == DocumentScore(score=0.0, category="neutral", confidence=0.0)


def test_parse_response_rejects_unknown_category_and_clamps():
    score = ollama_scorer.OllamaScorer.parse_response(
        '{"score": -9.0, "category": "vibes", "confidence": 5}'
    )
    assert score.category == "neutral"
    assert score.score == -1.0
    assert score.confidence == 1.0


def test_parse_response_forces_market_documents_to_zero():
    score = ollama_scorer.OllamaScorer.parse_response(
        '{"score": -0.8, "category": "market", "confidence": 0.9}'
    )
    assert score.score == 0.0
