"""The market-column blocklist — the mechanical half of ADR 0001.

``docs/ADR/0001-no-market-inputs.md`` forbids market data from reaching any
model input. ``nfl.data.schemas.MARKET_FIELD_NAMES`` enforces that for rows that
pass through the :class:`~nfl.data.schemas.Market` pydantic model. This module
enforces it for the *raw vendor column names* that arrive from nflverse, which
never see that schema: they are simply columns on a play-by-play or schedule
frame, and several of them do not look like market data at all.

Three families are blocked:

1. **Direct market quotes.** ``spread_line``, ``total_line``, the moneylines and
   the American odds columns. Obvious, and present in ``schedules``;
   ``spread_line``/``total_line`` are additionally denormalised onto every
   play-by-play row.
2. **Spread-derived model outputs.** ``vegas_wp``, ``vegas_home_wp``,
   ``vegas_wpa``, ``vegas_home_wpa``. These are nflfastR win-probability
   columns *conditioned on the closing spread* — they are the closing line
   wearing a lab coat, and they are null exactly when ``spread_line`` is null.
   This is the easiest leak in the project to miss.
3. **Third-party win-probability model outputs.** ``wp``, ``def_wp``,
   ``home_wp``, ``away_wp``, ``wpa`` and the whole ``*_wpa`` family. These are
   not spread-derived, so they are not market data, but they are another
   model's estimate of *our own target*. ADR 0001 rule 3 and CLAUDE.md hard
   rule 2 make them a baseline to beat, never a feature. They are blocked by
   the same mechanism because the failure mode is identical: a model handed one
   of these columns learns to copy it and reports a fraudulent score.

Use :func:`assert_no_market_columns` in any feature builder and in its tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "MARKET_COLUMNS",
    "MARKET_QUOTE_COLUMNS",
    "MARKET_DERIVED_WP_COLUMNS",
    "THIRD_PARTY_WP_COLUMNS",
    "MarketColumnLeak",
    "assert_no_market_columns",
    "market_columns_in",
    "drop_market_columns",
]


#: Betting quotes as they appear in nflverse ``schedules`` / ``pbp``.
MARKET_QUOTE_COLUMNS: frozenset[str] = frozenset(
    {
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    }
)

#: nflfastR win-probability columns conditioned on the closing spread.
MARKET_DERIVED_WP_COLUMNS: frozenset[str] = frozenset(
    {
        "vegas_wp",
        "vegas_home_wp",
        "vegas_wpa",
        "vegas_home_wpa",
    }
)

#: Third-party win-probability model outputs: a baseline to beat, not a feature.
THIRD_PARTY_WP_COLUMNS: frozenset[str] = frozenset(
    {
        "wp",
        "def_wp",
        "home_wp",
        "away_wp",
        "home_wp_post",
        "away_wp_post",
        "wpa",
        "air_wpa",
        "yac_wpa",
        "comp_air_wpa",
        "comp_yac_wpa",
        "total_home_rush_wpa",
        "total_away_rush_wpa",
        "total_home_pass_wpa",
        "total_away_pass_wpa",
        "total_home_comp_air_wpa",
        "total_away_comp_air_wpa",
        "total_home_comp_yac_wpa",
        "total_away_comp_yac_wpa",
        "total_home_raw_air_wpa",
        "total_away_raw_air_wpa",
        "total_home_raw_yac_wpa",
        "total_away_raw_yac_wpa",
    }
)

#: Every column name that must never appear in a feature frame.
MARKET_COLUMNS: frozenset[str] = (
    MARKET_QUOTE_COLUMNS | MARKET_DERIVED_WP_COLUMNS | THIRD_PARTY_WP_COLUMNS
)

#: Suffixes/substrings that catch future vendor columns the explicit lists miss.
#: ``_wpa``/``vegas_`` renames appear in nflverse from time to time; a new column
#: matching one of these is blocked on sight rather than silently allowed.
_BLOCKED_PATTERNS: tuple[str, ...] = ("vegas_", "moneyline", "spread_odds")


class MarketColumnLeak(AssertionError):
    """Raised when a frame destined for a model carries market columns."""

    def __init__(self, offenders: Iterable[str], *, context: str | None = None) -> None:
        cols = sorted(offenders)
        where = f" in {context}" if context else ""
        super().__init__(
            f"market/third-party-model columns must never reach a feature frame"
            f"{where}: {cols}. See docs/ADR/0001-no-market-inputs.md."
        )
        self.offenders = cols


def _column_names(df: Any) -> list[str]:
    """Best-effort column extraction for polars/pandas frames and plain iterables."""
    collect_schema = getattr(df, "collect_schema", None)
    if callable(collect_schema):  # polars LazyFrame: avoids a schema-resolve warning
        return [str(c) for c in collect_schema().names()]
    columns = getattr(df, "columns", None)
    if columns is not None:
        return [str(c) for c in columns]
    if isinstance(df, dict):
        return [str(c) for c in df]
    if isinstance(df, Iterable):
        return [str(c) for c in df]
    raise TypeError(f"cannot determine column names from {type(df)!r}")


def market_columns_in(df: Any) -> list[str]:
    """Return the blocked column names present on ``df`` (sorted, may be empty)."""
    names = _column_names(df)
    offenders = {n for n in names if n in MARKET_COLUMNS}
    offenders |= {n for n in names if any(p in n for p in _BLOCKED_PATTERNS)}
    return sorted(offenders)


def assert_no_market_columns(df: Any, *, context: str | None = None) -> None:
    """Raise :class:`MarketColumnLeak` if ``df`` carries any blocked column.

    Accepts anything column-shaped: a ``polars.DataFrame`` or ``LazyFrame``, a
    pandas frame, a mapping, or a bare iterable of column names.
    """
    offenders = market_columns_in(df)
    if offenders:
        raise MarketColumnLeak(offenders, context=context)


def drop_market_columns(df: Any) -> Any:
    """Return ``df`` without its blocked columns. Ingest-side convenience."""
    offenders = market_columns_in(df)
    return df.drop(offenders) if offenders else df
