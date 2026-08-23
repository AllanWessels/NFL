"""A deliberately tiny play-level feature set: the fallback the pipeline can run on.

``nfl.features.ingame`` is the real in-game builder. This module exists so that
assembly, the season splits and the baseline model are runnable *today*, with a
feature set small enough to read in one screen and verify by hand — score,
clock, possession, down/distance and timeouts, all strictly pre-snap. It is also
the reference implementation of the Phase 4 builder contract: keys, prefix,
market gate, no future information.

Two traps are handled here explicitly, both from ``docs/DATA_MODEL.md`` §2.2:

* ``total_home_score`` / ``total_away_score`` in nflverse are **post-play** —
  the touchdown's points are already on the row that scored them. Every score
  feature is therefore lagged one play within the game.
* nflverse ships the timeout columns in two flavours depending on the vintage
  and the slice (``home_``/``away_`` vs ``posteam_``/``defteam_``). Whichever
  pair is present is used, and the other is derived from it.

Prefix: ``bi__``. Nulls are left as nulls (GBDT-friendly).
"""

from __future__ import annotations

import polars as pl

from nfl.data.market_blocklist import assert_no_market_columns

__all__ = ["FEATURE_DOCS", "PREFIX", "build_baseline_inputs"]

PREFIX = "bi__"

FEATURE_DOCS: dict[str, str] = {
    "bi__score_diff_home": "Pre-snap home score minus away score (lagged one play).",
    "bi__abs_score_diff": "Absolute pre-snap score margin.",
    "bi__score_diff_posteam": "Pre-snap margin from the possessing team's perspective.",
    "bi__home_score": "Pre-snap home points.",
    "bi__away_score": "Pre-snap away points.",
    "bi__is_one_score_game": "1 when the pre-snap margin is 8 or fewer points.",
    "bi__possessions_to_tie": "Scoring drives the trailing team needs, ceil(|margin| / 8).",
    "bi__game_seconds_remaining": "Seconds left in regulation (0 during overtime).",
    "bi__half_seconds_remaining": "Seconds left in the current half.",
    "bi__quarter_seconds_remaining": "Seconds left in the current quarter.",
    "bi__elapsed_game_seconds": "Seconds of regulation already played.",
    "bi__period": "Quarter number; 5+ is overtime.",
    "bi__is_overtime": "1 when the play is in overtime.",
    "bi__is_final_two_minutes": "1 inside two minutes of either half.",
    "bi__posteam_is_home": "1 when the home team has the ball; null on non-plays.",
    "bi__down": "Down (1-4); null on kickoffs, extra points and non-plays.",
    "bi__ydstogo": "Yards needed for a first down.",
    "bi__yardline_100": "Distance to the opponent's end zone, 1-99.",
    "bi__goal_to_go": "1 when the possessing team is in a goal-to-go situation.",
    "bi__in_red_zone": "1 when the ball is inside the opponent's 20.",
    "bi__home_timeouts_remaining": "Home timeouts left, pre-snap.",
    "bi__away_timeouts_remaining": "Away timeouts left, pre-snap.",
    "bi__posteam_timeouts_remaining": "Possessing-team timeouts left, pre-snap.",
    "bi__defteam_timeouts_remaining": "Defending-team timeouts left, pre-snap.",
    "bi__timeout_diff_home": "Home timeouts minus away timeouts.",
}


def _num(pbp: pl.DataFrame, name: str) -> pl.Expr:
    """``pl.col(name)`` as Float64, or a null column when nflverse omits it."""
    if name in pbp.columns:
        return pl.col(name).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64)


def _str(pbp: pl.DataFrame, name: str) -> pl.Expr:
    if name in pbp.columns:
        return pl.col(name).cast(pl.String, strict=False)
    return pl.lit(None, dtype=pl.String)


def build_baseline_inputs(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per ``(game_id, play_id)`` of pre-snap game state.

    ``pbp`` is a raw nflverse play-by-play frame; only the handful of columns
    named below are read, so a trimmed fixture works as well as the real table.
    """
    required = {"game_id", "play_id"}
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise KeyError(f"pbp is missing required column(s) {missing}")

    frame = pbp.sort(["game_id", "play_id"])
    posteam_is_home = (
        pl.when(_str(frame, "posteam").is_null())
        .then(None)
        .otherwise(_str(frame, "posteam") == _str(frame, "home_team"))
    )

    # Score, lagged one play: nflverse totals are post-play (DATA_MODEL §2.2).
    home_pre = _num(frame, "total_home_score").shift(1).over("game_id").fill_null(0.0)
    away_pre = _num(frame, "total_away_score").shift(1).over("game_id").fill_null(0.0)
    diff_home = home_pre - away_pre

    # Timeouts arrive as either (home, away) or (posteam, defteam); derive the
    # missing pair from the one that is present.
    has_side = {"home_timeouts_remaining", "away_timeouts_remaining"} <= set(frame.columns)
    has_poss = {"posteam_timeouts_remaining", "defteam_timeouts_remaining"} <= set(frame.columns)
    if has_side:
        home_to = _num(frame, "home_timeouts_remaining")
        away_to = _num(frame, "away_timeouts_remaining")
        pos_to = pl.when(posteam_is_home).then(home_to).otherwise(away_to)
        def_to = pl.when(posteam_is_home).then(away_to).otherwise(home_to)
    elif has_poss:
        pos_to = _num(frame, "posteam_timeouts_remaining")
        def_to = _num(frame, "defteam_timeouts_remaining")
        home_to = pl.when(posteam_is_home).then(pos_to).otherwise(def_to)
        away_to = pl.when(posteam_is_home).then(def_to).otherwise(pos_to)
    else:
        home_to = away_to = pos_to = def_to = pl.lit(None, dtype=pl.Float64)

    period = _num(frame, "qtr")
    gsr = _num(frame, "game_seconds_remaining")
    half = _num(frame, "half_seconds_remaining")
    yardline = _num(frame, "yardline_100")

    out = frame.select(
        pl.col("game_id").cast(pl.String),
        pl.col("play_id").cast(pl.Int64, strict=False),
        diff_home.alias("bi__score_diff_home"),
        diff_home.abs().alias("bi__abs_score_diff"),
        pl.when(posteam_is_home)
        .then(diff_home)
        .otherwise(-diff_home)
        .alias("bi__score_diff_posteam"),
        home_pre.alias("bi__home_score"),
        away_pre.alias("bi__away_score"),
        (diff_home.abs() <= 8).cast(pl.Int8).alias("bi__is_one_score_game"),
        (diff_home.abs() / 8.0).ceil().alias("bi__possessions_to_tie"),
        gsr.alias("bi__game_seconds_remaining"),
        half.alias("bi__half_seconds_remaining"),
        _num(frame, "quarter_seconds_remaining").alias("bi__quarter_seconds_remaining"),
        (3600.0 - gsr).clip(0.0, 3600.0).alias("bi__elapsed_game_seconds"),
        period.alias("bi__period"),
        (period >= 5).cast(pl.Int8).alias("bi__is_overtime"),
        (half <= 120.0).cast(pl.Int8).alias("bi__is_final_two_minutes"),
        posteam_is_home.cast(pl.Int8).alias("bi__posteam_is_home"),
        _num(frame, "down").alias("bi__down"),
        _num(frame, "ydstogo").alias("bi__ydstogo"),
        yardline.alias("bi__yardline_100"),
        _num(frame, "goal_to_go").cast(pl.Int8, strict=False).alias("bi__goal_to_go"),
        (yardline <= 20.0).cast(pl.Int8).alias("bi__in_red_zone"),
        home_to.alias("bi__home_timeouts_remaining"),
        away_to.alias("bi__away_timeouts_remaining"),
        pos_to.alias("bi__posteam_timeouts_remaining"),
        def_to.alias("bi__defteam_timeouts_remaining"),
        (home_to - away_to).alias("bi__timeout_diff_home"),
    ).unique(subset=["game_id", "play_id"], keep="first", maintain_order=True)

    assert_no_market_columns(out, context="build_baseline_inputs")
    return out
