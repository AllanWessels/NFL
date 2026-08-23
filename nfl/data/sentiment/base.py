"""Shared types for sentiment scorers.

Every scorer -- the rule-based default and the (disabled by default, local-only)
Ollama one -- implements :class:`DocumentScorer`, so ``benchmark.py`` can swap
them without knowing which is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

#: Coarse reason a document moved the score. ``market`` documents are scored
#: 0.0 on purpose: line-movement chatter is market data in disguise (ADR 0001).
Category = Literal[
    "injury",
    "return",
    "suspension",
    "trade",
    "coaching_change",
    "qb_change",
    "weather",
    "discord",
    "contract",
    "market",
    "neutral",
]

CATEGORIES: tuple[Category, ...] = (
    "injury",
    "return",
    "suspension",
    "trade",
    "coaching_change",
    "qb_change",
    "weather",
    "discord",
    "contract",
    "market",
    "neutral",
)


@dataclass(frozen=True, slots=True)
class Document:
    """One scoreable text item (a headline, or an injury-report comment)."""

    text: str
    published_at: datetime | None = None
    team: str | None = None
    entity_id: str | None = None
    source: str = ""
    doc_id: str = ""


@dataclass(frozen=True, slots=True)
class DocumentScore:
    """A scorer's verdict on one document.

    ``score`` is signed and clamped to ``[-1, 1]`` from the perspective of the
    subject team: negative = bad news for that team.
    """

    score: float
    category: Category
    confidence: float = 0.0
    matched_terms: tuple[str, ...] = ()
    negated: bool = False
    flags: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class DocumentScorer(Protocol):
    """The interface both the rule-based and Ollama scorers satisfy."""

    name: str
    version: str

    @property
    def enabled(self) -> bool:
        """False when the scorer cannot run here (no model, disabled by env)."""
        ...

    def score_document(self, document: Document) -> DocumentScore: ...

    def score_documents(self, documents: list[Document]) -> list[DocumentScore]: ...


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


__all__ = [
    "CATEGORIES",
    "Category",
    "Document",
    "DocumentScore",
    "DocumentScorer",
    "clamp",
]
