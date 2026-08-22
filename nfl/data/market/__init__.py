"""Market data adapters. **Evaluation / ledger only — never a model input.**

Everything in this package emits :class:`nfl.data.schemas.Market` rows, which
carry ``market_only: Literal[True]``. See ``docs/ADR/0001-no-market-inputs.md``.
Feature builders in ``nfl/features/`` must never import from here.
"""

from __future__ import annotations

__all__ = ["closing_lines", "kalshi"]
