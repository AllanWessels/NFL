"""Local-only Ollama sentiment scorer. **Stub: disabled by default, no calls.**

CLAUDE.md hard rule 4: sentiment is rule-based by default; a *local* Ollama
model may be benchmarked against it and adopted only if the rule-based scorer is
measurably worse. Nothing leaves the machine, ever -- this module talks only to
``http://localhost:11434`` and refuses any other host.

Nothing here calls a model yet. :class:`OllamaScorer` implements the same
:class:`~nfl.data.sentiment.base.DocumentScorer` interface as
:class:`~nfl.data.sentiment.news_rules.RuleBasedScorer` so that
``benchmark.py`` can hold both, and raises :class:`OllamaDisabled` unless it is
explicitly turned on. The prompt and the parse step are sketched below so the
shape of the eventual implementation is reviewable now.

To enable later::

    NFL_ENABLE_OLLAMA=1 NFL_OLLAMA_MODEL=llama3.1:8b
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from nfl.data.sentiment.base import CATEGORIES, Category, Document, DocumentScore, clamp

log = logging.getLogger(__name__)

SCORER_NAME = "ollama"
SCORER_VERSION = "0.0.1-stub"

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"
ENABLE_ENV = "NFL_ENABLE_OLLAMA"
MODEL_ENV = "NFL_OLLAMA_MODEL"
HOST_ENV = "NFL_OLLAMA_HOST"

#: Only these hosts are ever acceptable. A remote Ollama would be a cloud LLM
#: with extra steps, which the project forbids outright.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

PROMPT_TEMPLATE = """\
You are scoring one NFL news headline for its effect on the named team.

Return ONLY compact JSON: {{"score": <float -1..1>, "category": "<one of {categories}>", \
"confidence": <float 0..1>}}

score: -1 = clearly bad for the team, 0 = neutral, +1 = clearly good.
Never score betting lines, spreads or odds; use category "market" and score 0.

team: {team}
headline: {text}
"""


class OllamaDisabled(RuntimeError):
    """Raised when the Ollama scorer is used while disabled (the default)."""


class OllamaNotLocal(RuntimeError):
    """Raised when a non-local host is configured. Never relaxed."""


def is_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_host() -> str:
    host = os.environ.get(HOST_ENV, DEFAULT_HOST).strip() or DEFAULT_HOST
    hostname = urlparse(host).hostname
    if hostname not in LOCAL_HOSTS:
        raise OllamaNotLocal(
            f"refusing non-local Ollama host {host!r}: no cloud LLMs (CLAUDE.md rule 4)"
        )
    return host


@dataclass
class OllamaScorer:
    """Same interface as the rule-based scorer; not implemented yet.

    ``enabled`` is False unless ``NFL_ENABLE_OLLAMA`` is set, so importing this
    module and constructing the scorer is always safe and never touches a socket.
    """

    model: str = ""
    host: str = ""
    temperature: float = 0.0
    timeout: float = 60.0
    name: str = SCORER_NAME
    version: str = SCORER_VERSION

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        self.host = self.host or (resolve_host() if is_enabled() else DEFAULT_HOST)

    @property
    def enabled(self) -> bool:
        return is_enabled()

    # -- interface -------------------------------------------------------- #

    def score_document(self, document: Document) -> DocumentScore:
        self._require_enabled()
        raise NotImplementedError(  # pragma: no cover - stub
            "OllamaScorer.score_document is a stub; implement the /api/generate call "
            "and parse_response() before enabling."
        )

    def score_documents(self, documents: list[Document]) -> list[DocumentScore]:
        return [self.score_document(document) for document in documents]

    # -- pieces the real implementation will use -------------------------- #

    def build_prompt(self, document: Document) -> str:
        """Render the scoring prompt. Pure, testable, no I/O."""
        return PROMPT_TEMPLATE.format(
            categories="|".join(CATEGORIES),
            team=document.team or "unknown",
            text=document.text,
        )

    def request_payload(self, document: Document) -> dict[str, Any]:
        """The body the real implementation will POST to ``/api/generate``."""
        return {
            "model": self.model,
            "prompt": self.build_prompt(document),
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

    @staticmethod
    def parse_response(raw: str) -> DocumentScore:
        """Parse the model's JSON reply, defensively.

        A local 8B model will occasionally emit prose around the JSON or a
        category outside the enum; both degrade to a neutral, zero-confidence
        score rather than raising, so one bad generation cannot poison a batch.
        """
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            payload = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            log.warning("ollama: unparseable response %r", raw[:120])
            return DocumentScore(score=0.0, category="neutral", confidence=0.0)

        category_raw = str(payload.get("category", "neutral"))
        category: Category = category_raw if category_raw in CATEGORIES else "neutral"  # type: ignore[assignment]
        try:
            score = clamp(float(payload.get("score", 0.0)))
            confidence = clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            return DocumentScore(score=0.0, category=category, confidence=0.0)
        if category == "market":
            score = 0.0  # ADR 0001, enforced regardless of what the model said
        return DocumentScore(score=score, category=category, confidence=confidence)

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise OllamaDisabled(
                f"Ollama scorer is disabled; set {ENABLE_ENV}=1 to opt in "
                "(local models only -- see CLAUDE.md rule 4)"
            )
        resolve_host()


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MODEL",
    "ENABLE_ENV",
    "LOCAL_HOSTS",
    "PROMPT_TEMPLATE",
    "SCORER_NAME",
    "SCORER_VERSION",
    "OllamaDisabled",
    "OllamaNotLocal",
    "OllamaScorer",
    "is_enabled",
    "resolve_host",
]
