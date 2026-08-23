"""Rule-based sentiment over injury reports and news headlines.

No cloud LLMs (CLAUDE.md rule 4). The default scorer is deterministic and
lexicon-based; :mod:`nfl.data.sentiment.ollama_scorer` is a local-only stub,
disabled unless explicitly enabled, benchmarked by
:mod:`nfl.data.sentiment.benchmark`.

Unlike :mod:`nfl.data.market`, sentiment **is** a legitimate feature source --
but market-derived sentiment is not: headlines about lines, spreads and odds are
scored 0.0 and categorised ``market`` (ADR 0001).
"""

from __future__ import annotations

from nfl.data.sentiment.base import Document, DocumentScore, DocumentScorer

__all__ = [
    "Document",
    "DocumentScore",
    "DocumentScorer",
    "benchmark",
    "injury_rules",
    "news_rules",
    "ollama_scorer",
    "sources",
]
