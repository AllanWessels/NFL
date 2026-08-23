"""Player-level roster-quality features — prefix ``pl__``.

One row per ``(game_id, team)``, every column computed **strictly as of
kickoff**: only plays from games in earlier weeks of the same season, the depth
chart / injury report published for the game's own week, and snap counts from
*earlier* weeks. Nothing here reads the game it describes.

What the module produces, in the order the sections appear below:

``player_game_stats``
    Explodes play-by-play into one row per ``(player, game)`` by joining on the
    ``*_player_id`` columns — passer, rusher, receiver, and the defensive id
    family (sacks, interceptions, tackles, QB hits, pass breakups) plus the
    kick/punt returners. Pure polars; no row loops.

``unit_strength``
    The four opponent units of ``docs/DATA_MODEL.md`` §6 — ``pass_offense``,
    ``rush_offense``, ``pass_defense``, ``rush_defense`` — rated by EPA/play
    accumulated over *prior* weeks only, then z-scored and ranked across the
    league **as of that week**. Orientation is normalised into ``weakness_z``:
    higher always means a *weaker* unit, matching the rank convention where a
    higher rank number is weaker. Getting that sign backwards silently inverts
    every adjustment, so the tests assert it.

``opponent_adjust``
    The reusable §6 implementation: rank/z → multiplier, divide, recency-weight,
    shrink toward the position-group mean. Usable by any builder, not just this
    one.

``identify_starters``
    Depth chart first (``depth_team == "1"``), prior-week snap share second,
    weekly-roster ``status``/experience last. Assigns ``QB1``, ``RB1``,
    ``WR1``-``WR3``, ``TE1``, ``OL1``-``OL5``, ``DL1``-``DL4``, ``LB1``-``LB3``,
    ``CB1``-``CB3``, ``S1``-``S2``, ``K1``, ``P1``.

``build_player``
    Team-level rollup: QB1 adjusted EPA and bio, skill-position adjusted EPA,
    OL/DL sack-rate proxies, special-teams returner, position ranks, and the
    availability block scored through :mod:`nfl.data.sentiment.injury_rules`.

``roster_from_snapshot``
    The live path. Takes a :class:`~nfl.data.live.base.LiveSnapshot` and the
    pregame table and returns the *same* columns, with the availability block
    refreshed from the snapshot's in-game injury list.

Leakage notes specific to this module:

* Opponent ranks are as-of-week (§6 leakage rule, taxonomy item 2). The
  cumulative unit ratings are ``cum_sum().shift(1)`` over a dense team-week
  grid, so week *W* sees weeks ``< W`` and nothing else.
* ``snap_counts`` is postgame truth for its own week (taxonomy item 3); it is
  only ever read for weeks strictly before the target week.
* The weekly injury report *is* pregame for its own week and is used as such;
  ``did_play`` / ``snap_share_if_played`` are never read.
* ``assert_no_market_columns`` guards the return value.

``pl__qb1_contract_years_remaining`` is emitted as an all-null column on
purpose: ``docs/DATA_MODEL.md`` §5.2 wants it, the source (an OverTheCap import
or a hand-maintained table) does not exist yet, and a documented null column is
cheaper than a schema change later.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from nfl.data.market_blocklist import assert_no_market_columns
from nfl.data.sentiment.injury_rules import (
    InjuryItem,
    availability_probability,
    canonical_status,
    load_position_impact,
    position_impact,
    score_injury_items,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nfl.data.live.base import LiveSnapshot

__all__ = [
    "PREFIX",
    "FEATURE_DOCS",
    "DEFAULT_K",
    "DEFAULT_LAMBDA",
    "DEFAULT_SHRINKAGE_GAMES",
    "N_TEAMS",
    "Z_CLIP",
    "TRAILING_GAMES",
    "UNITS",
    "SLOT_COUNTS",
    "POSITION_GROUPS",
    "position_group",
    "rank_multiplier",
    "z_multiplier",
    "adjust_stat",
    "player_game_stats",
    "player_game_rates",
    "unit_strength",
    "opponent_adjust",
    "as_of_aggregate",
    "identify_starters",
    "team_availability",
    "build_player",
    "roster_from_snapshot",
    "load_player_inputs",
]

PREFIX = "pl__"

#: §6 adjustment aggressiveness. rank 1 -> ``1 - k``; rank 32 -> ``1 + k``.
DEFAULT_K = 0.25
#: §6 recency weight ``w(g) = lambda ** weeks_ago``.
DEFAULT_LAMBDA = 0.95
#: §6 shrinkage constant ``c``, in "games worth of prior".
DEFAULT_SHRINKAGE_GAMES = 4.0
#: League size used by the rank -> multiplier map.
N_TEAMS = 32
#: §6 clip on the continuous strength z-score.
Z_CLIP = 2.5
#: Trailing-form window, in games.
TRAILING_GAMES = 8
#: Plays a unit needs before its rating is trusted; below this it is neutral.
MIN_UNIT_PLAYS = 50

UNITS: tuple[str, ...] = ("pass_offense", "rush_offense", "pass_defense", "rush_defense")

#: Starter slots per position group, in depth order.
SLOT_COUNTS: dict[str, int] = {
    "QB": 1,
    "RB": 1,
    "WR": 3,
    "TE": 1,
    "OL": 5,
    "DL": 4,
    "LB": 3,
    "CB": 3,
    "S": 2,
    "K": 1,
    "P": 1,
}

#: Raw position -> the group this module slots on. Finer than
#: ``position_impact.csv``'s ``position_group`` because CB and S are ranked
#: separately here.
POSITION_GROUPS: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "T": "OL",
    "OT": "OL",
    "LT": "OL",
    "RT": "OL",
    "G": "OL",
    "OG": "OL",
    "LG": "OL",
    "RG": "OL",
    "C": "OL",
    "OL": "OL",
    "DE": "DL",
    "EDGE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "LB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "OLB": "LB",
    "CB": "CB",
    "DB": "CB",
    "S": "S",
    "FS": "S",
    "SS": "S",
    "SAF": "S",
    "K": "K",
    "PK": "K",
    "P": "P",
    "LS": "ST",
    "KR": "ST",
    "PR": "ST",
}

#: Designations that make a player unavailable outright.
HARD_OUT_STATUSES = frozenset({"OUT", "IR", "PUP", "NFI", "SUSP", "SUSPENDED"})


def position_group(position: str | None) -> str:
    """Map a raw position abbreviation onto this module's slotting group."""
    key = (position or "").strip().upper()
    if key in POSITION_GROUPS:
        return POSITION_GROUPS[key]
    return position_impact(key).position_group or "ST"


# --------------------------------------------------------------------------- #
# §6 — rank / z to multiplier
# --------------------------------------------------------------------------- #


def rank_multiplier(rank: float, k: float = DEFAULT_K, n_teams: int = N_TEAMS) -> float:
    """``m(u, g) = 1 + k * ((r - (N+1)/2) / ((N-1)/2))`` — §6 step 1.

    Rank 1 (strongest opponent unit) gives ``1 - k``; rank ``N`` gives ``1 + k``;
    the midpoint gives exactly 1.0. Because production is *divided* by ``m``, a
    small ``m`` scales production against a strong unit **up**.
    """
    return 1.0 + k * ((rank - (n_teams + 1) / 2) / ((n_teams - 1) / 2))


def z_multiplier(weakness_z: float, k: float = DEFAULT_K, z_clip: float = Z_CLIP) -> float:
    """Continuous-strength variant of §6: ``m = 1 + k * z``.

    ``weakness_z`` is oriented so **higher means weaker**, matching the rank
    convention. See :func:`unit_strength`, which produces it.
    """
    return 1.0 + k * max(-z_clip, min(z_clip, weakness_z))


def adjust_stat(
    stat: float,
    *,
    rank: float | None = None,
    weakness_z: float | None = None,
    k: float = DEFAULT_K,
    n_teams: int = N_TEAMS,
) -> float:
    """§6 step 2: ``adj_s = s / m``. Supply exactly one of ``rank``/``weakness_z``."""
    if (rank is None) == (weakness_z is None):
        raise ValueError("pass exactly one of rank= or weakness_z=")
    if rank is not None:
        return stat / rank_multiplier(rank, k, n_teams)
    assert weakness_z is not None  # noqa: S101 - guaranteed by the check above
    return stat / z_multiplier(weakness_z, k)


def _multiplier_expr(*, use_rank: bool, k: float, n_teams: int, z_clip: float) -> pl.Expr:
    if use_rank:
        expr = 1.0 + k * ((pl.col("_rank") - (n_teams + 1) / 2) / ((n_teams - 1) / 2))
    else:
        expr = 1.0 + k * pl.col("_weakness_z").clip(-z_clip, z_clip)
    # Neutral 1.0 where the opponent unit has no rating yet, and a floor so a
    # pathological k can never flip the sign of an adjustment.
    return expr.fill_null(1.0).clip(lower_bound=0.05)


# --------------------------------------------------------------------------- #
# Play-by-play -> player-game stats
# --------------------------------------------------------------------------- #

#: Counting stats produced by :func:`player_game_stats`.
STAT_COLUMNS: tuple[str, ...] = (
    "pass_dropbacks",
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions_thrown",
    "sacks_taken",
    "pass_epa",
    "cpoe_sum",
    "cpoe_n",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "rush_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rec_epa",
    "yac",
    "sacks_made",
    "interceptions_made",
    "tackles",
    "tackles_for_loss",
    "qb_hits",
    "pass_defensed",
    "forced_fumbles",
    "kick_returns",
    "kick_return_yards",
    "punt_returns",
    "punt_return_yards",
)

_PLAYER_GAME_KEYS = ("season", "week", "game_id", "team", "opponent", "player_id")


def _num(names: Iterable[str], column: str) -> pl.Expr:
    """``pl.col(column)`` as Float64, or a null literal when the column is absent."""
    return (
        pl.col(column).cast(pl.Float64)
        if column in names
        else pl.lit(None, dtype=pl.Float64).alias(column)
    )


def _flag(names: Iterable[str], column: str) -> pl.Expr:
    """1.0 where ``column`` equals 1, else 0.0; all-zero when the column is absent."""
    if column not in names:
        return pl.lit(0.0).alias(column)
    return (pl.col(column).cast(pl.Float64).fill_null(0.0) == 1.0).cast(pl.Float64).alias(column)


def _role_frame(
    base: pl.DataFrame,
    *,
    id_column: str,
    team_column: str,
    opponent_column: str,
    stats: Mapping[str, pl.Expr],
) -> pl.DataFrame | None:
    """One player-role slice of pbp, aggregated to ``(player, game)``."""
    if id_column not in base.columns:
        return None
    sliced = base.filter(pl.col(id_column).is_not_null()).select(
        pl.col("season"),
        pl.col("week"),
        pl.col("game_id"),
        pl.col(team_column).alias("team"),
        pl.col(opponent_column).alias("opponent"),
        pl.col(id_column).cast(pl.Utf8).alias("player_id"),
        *[expr.alias(name) for name, expr in stats.items()],
    )
    if sliced.height == 0:
        return None
    return sliced.group_by(list(_PLAYER_GAME_KEYS)).agg(
        [pl.col(name).sum().alias(name) for name in stats]
    )


#: ``(pbp id column, stat it credits, credit per appearance)`` for defence.
_DEFENSIVE_ID_COLUMNS: tuple[tuple[str, str, float], ...] = (
    ("sack_player_id", "sacks_made", 1.0),
    ("half_sack_1_player_id", "sacks_made", 0.5),
    ("half_sack_2_player_id", "sacks_made", 0.5),
    ("interception_player_id", "interceptions_made", 1.0),
    ("solo_tackle_1_player_id", "tackles", 1.0),
    ("solo_tackle_2_player_id", "tackles", 1.0),
    ("assist_tackle_1_player_id", "tackles", 0.5),
    ("assist_tackle_2_player_id", "tackles", 0.5),
    ("assist_tackle_3_player_id", "tackles", 0.5),
    ("assist_tackle_4_player_id", "tackles", 0.5),
    ("tackle_for_loss_1_player_id", "tackles_for_loss", 1.0),
    ("tackle_for_loss_2_player_id", "tackles_for_loss", 1.0),
    ("qb_hit_1_player_id", "qb_hits", 1.0),
    ("qb_hit_2_player_id", "qb_hits", 1.0),
    ("pass_defense_1_player_id", "pass_defensed", 1.0),
    ("pass_defense_2_player_id", "pass_defensed", 1.0),
    ("forced_fumble_player_1_player_id", "forced_fumbles", 1.0),
    ("forced_fumble_player_2_player_id", "forced_fumbles", 1.0),
)

_PBP_COLUMNS_USED: frozenset[str] = frozenset(
    {
        "season",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "epa",
        "cpoe",
        "yards_gained",
        "return_yards",
        "yards_after_catch",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "pass_attempt",
        "rush_attempt",
        "qb_dropback",
        "sack",
        "complete_pass",
        "interception",
        "pass_touchdown",
        "rush_touchdown",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
        "kickoff_returner_player_id",
        "punt_returner_player_id",
    }
    | {name for name, _, _ in _DEFENSIVE_ID_COLUMNS}
)


def player_game_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per ``(player, game)`` of raw counting stats, from play-by-play.

    Offensive rows are attributed to ``posteam``; defensive rows and kick/punt
    returners to ``defteam`` (on a kick or punt nflfastR's ``posteam`` is the
    *kicking* team). Split sacks count 0.5 each, assisted tackles 0.5 each.
    """
    names = set(pbp.columns)
    missing = {"season", "week", "game_id", "posteam", "defteam"} - names
    if missing:
        raise ValueError(f"pbp is missing required columns: {sorted(missing)}")

    base = pbp.select(sorted(_PBP_COLUMNS_USED & names)).filter(
        pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    )
    base = base.with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        _num(names, "epa"),
        _num(names, "cpoe"),
        _flag(names, "pass_attempt"),
        _flag(names, "rush_attempt"),
        _flag(names, "qb_dropback"),
        _flag(names, "sack"),
        _flag(names, "complete_pass"),
        _flag(names, "interception"),
        _flag(names, "pass_touchdown"),
        _flag(names, "rush_touchdown"),
    )
    names = set(base.columns)

    zero, one = pl.lit(0.0), pl.lit(1.0)
    is_sack = pl.col("sack") == 1.0
    # An *official* pass attempt is a dropback that was not a sack; nflfastR
    # flags a scramble as rush_attempt, so it drops out here too.
    is_attempt = (pl.col("pass_attempt") == 1.0) & ~is_sack
    dropback = pl.max_horizontal(pl.col("qb_dropback"), pl.col("pass_attempt"))

    frames: list[pl.DataFrame | None] = [
        _role_frame(
            base,
            id_column="passer_player_id",
            team_column="posteam",
            opponent_column="defteam",
            stats={
                "pass_dropbacks": dropback,
                "pass_attempts": pl.when(is_attempt).then(one).otherwise(zero),
                "completions": pl.col("complete_pass"),
                "passing_yards": _num(names, "passing_yards").fill_null(0.0),
                "passing_tds": pl.col("pass_touchdown"),
                "interceptions_thrown": pl.col("interception"),
                "sacks_taken": pl.col("sack"),
                "pass_epa": pl.col("epa").fill_null(0.0),
                "cpoe_sum": pl.col("cpoe").fill_null(0.0),
                "cpoe_n": pl.col("cpoe").is_not_null().cast(pl.Float64),
            },
        ),
        _role_frame(
            base,
            id_column="rusher_player_id",
            team_column="posteam",
            opponent_column="defteam",
            stats={
                "rush_attempts": pl.max_horizontal(pl.col("rush_attempt"), one),
                "rushing_yards": _num(names, "rushing_yards").fill_null(0.0),
                "rushing_tds": pl.col("rush_touchdown"),
                "rush_epa": pl.col("epa").fill_null(0.0),
            },
        ),
        _role_frame(
            base,
            id_column="receiver_player_id",
            team_column="posteam",
            opponent_column="defteam",
            stats={
                "targets": pl.when(is_attempt).then(one).otherwise(zero),
                "receptions": pl.col("complete_pass"),
                "receiving_yards": _num(names, "receiving_yards").fill_null(0.0),
                "receiving_tds": pl.col("pass_touchdown"),
                "rec_epa": pl.col("epa").fill_null(0.0),
                "yac": _num(names, "yards_after_catch").fill_null(0.0),
            },
        ),
    ]
    for id_column, stat, credit in _DEFENSIVE_ID_COLUMNS:
        frames.append(
            _role_frame(
                base,
                id_column=id_column,
                team_column="defteam",
                opponent_column="posteam",
                stats={stat: pl.lit(credit)},
            )
        )
    # nflfastR is not symmetric here, verified against 2024 rosters: on a
    # kickoff ``posteam`` is the *receiving* team (986/986 returners match it),
    # on a punt it is the punting team (1477/1477 returners match ``defteam``).
    return_yards = _num(names, "return_yards").fill_null(0.0)
    frames.append(
        _role_frame(
            base,
            id_column="kickoff_returner_player_id",
            team_column="posteam",
            opponent_column="defteam",
            stats={"kick_returns": one, "kick_return_yards": return_yards},
        )
    )
    frames.append(
        _role_frame(
            base,
            id_column="punt_returner_player_id",
            team_column="defteam",
            opponent_column="posteam",
            stats={"punt_returns": one, "punt_return_yards": return_yards},
        )
    )

    present = [f for f in frames if f is not None]
    if not present:
        return _empty_player_game_stats()
    stacked = pl.concat(present, how="diagonal_relaxed")
    out = stacked.group_by(list(_PLAYER_GAME_KEYS)).agg(
        [pl.col(c).sum().alias(c) for c in STAT_COLUMNS if c in stacked.columns]
    )
    for column in STAT_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(column))
    out = out.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in STAT_COLUMNS])
    return (
        _resolve_player_team(out, base)
        .select(*_PLAYER_GAME_KEYS, *STAT_COLUMNS)
        .sort(["season", "week", "game_id", "team", "player_id"])
    )


def _resolve_player_team(stats: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """Collapse a player to one team per game.

    Possession flips *inside* a play — a receiver who tackles the defender who
    just intercepted him is credited by nflfastR while ``defteam`` still names
    his own opponent — so a handful of players end up with two rows per game,
    one per side. The row carrying more production wins, and the stats are
    re-summed onto it.
    """
    sides = pl.concat(
        [
            base.select(
                "game_id", pl.col("posteam").alias("team"), pl.col("defteam").alias("opponent")
            ),
            base.select(
                "game_id", pl.col("defteam").alias("team"), pl.col("posteam").alias("opponent")
            ),
        ]
    ).unique()
    activity = pl.sum_horizontal([pl.col(c) for c in STAT_COLUMNS])
    primary = (
        stats.with_columns(activity.alias("_activity"))
        .sort(["_activity", "team"], descending=[True, False])
        .unique(
            subset=["season", "week", "game_id", "player_id"], keep="first", maintain_order=True
        )
        .select("season", "week", "game_id", "player_id", pl.col("team").alias("_team"))
    )
    merged = (
        stats.join(primary, on=["season", "week", "game_id", "player_id"], how="left")
        .drop("team", "opponent")
        .rename({"_team": "team"})
        .group_by(["season", "week", "game_id", "team", "player_id"])
        .agg([pl.col(c).sum().alias(c) for c in STAT_COLUMNS])
    )
    return merged.join(sides, on=["game_id", "team"], how="left")


def _empty_player_game_stats() -> pl.DataFrame:
    schema: dict[str, Any] = {
        "season": pl.Int64,
        "week": pl.Int64,
        "game_id": pl.Utf8,
        "team": pl.Utf8,
        "opponent": pl.Utf8,
        "player_id": pl.Utf8,
    }
    schema.update({c: pl.Float64 for c in STAT_COLUMNS})
    return pl.DataFrame(schema=schema)


def _safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


#: Per-game rate stats added by :func:`player_game_rates`.
RATE_COLUMNS: tuple[str, ...] = (
    "epa_per_dropback",
    "rush_epa_per_play",
    "rec_epa_per_play",
    "yac_per_reception",
    "completion_pct",
    "cpoe",
    "sack_rate_taken",
    "pressures",
)


def player_game_rates(stats: pl.DataFrame) -> pl.DataFrame:
    """Per-game rate stats derived from :func:`player_game_stats` counts.

    These are what §6 adjusts — a *rate* against a strong unit is what the
    formula is about — and the per-game yardage totals sit alongside them.
    """
    return stats.with_columns(
        _safe_div(pl.col("pass_epa"), pl.col("pass_dropbacks")).alias("epa_per_dropback"),
        _safe_div(pl.col("rush_epa"), pl.col("rush_attempts")).alias("rush_epa_per_play"),
        _safe_div(pl.col("rec_epa"), pl.col("targets")).alias("rec_epa_per_play"),
        _safe_div(pl.col("yac"), pl.col("receptions")).alias("yac_per_reception"),
        _safe_div(pl.col("completions"), pl.col("pass_attempts")).alias("completion_pct"),
        _safe_div(pl.col("cpoe_sum"), pl.col("cpoe_n")).alias("cpoe"),
        _safe_div(pl.col("sacks_taken"), pl.col("pass_dropbacks")).alias("sack_rate_taken"),
        (pl.col("sacks_made") + 0.5 * pl.col("qb_hits") + 0.5 * pl.col("tackles_for_loss")).alias(
            "pressures"
        ),
    )


# --------------------------------------------------------------------------- #
# Opponent unit strength, as-of-week
# --------------------------------------------------------------------------- #


def unit_strength(
    pbp: pl.DataFrame,
    *,
    min_plays: int = MIN_UNIT_PLAYS,
    z_clip: float = Z_CLIP,
) -> pl.DataFrame:
    """As-of-week strength of every team's four units — §6's ``r``/``z`` source.

    Returns one row per ``(season, week, unit, team)`` with:

    ``epa_per_play``
        EPA per play accumulated over the season's **earlier weeks only**; null
        until ``min_plays`` plays exist.
    ``weakness_z``
        League z-score of that rating as of the week, clipped to ``±z_clip`` and
        **oriented so higher = weaker unit** (for a defence that is EPA allowed
        as-is; for an offence it is negated).
    ``rank``
        1 = strongest, ``N`` = weakest. Neutral ``(N+1)/2`` when unrated.
    """
    names = set(pbp.columns)
    missing = {"season", "week", "posteam", "defteam"} - names
    if missing:
        raise ValueError(f"pbp is missing required columns: {sorted(missing)}")

    base = pbp.select(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam"),
        pl.col("defteam"),
        _num(names, "epa"),
        _flag(names, "pass_attempt"),
        _flag(names, "rush_attempt"),
    ).filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("epa").is_not_null()
        & ((pl.col("pass_attempt") == 1.0) | (pl.col("rush_attempt") == 1.0))
    )
    kind = pl.when(pl.col("pass_attempt") == 1.0).then(pl.lit("pass")).otherwise(pl.lit("rush"))
    offense = base.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("posteam").alias("team"),
        pl.concat_str([kind, pl.lit("_offense")]).alias("unit"),
        pl.col("epa"),
    )
    defense = base.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("defteam").alias("team"),
        pl.concat_str([kind, pl.lit("_defense")]).alias("unit"),
        pl.col("epa"),
    )
    team_week = (
        pl.concat([offense, defense])
        .group_by(["season", "week", "unit", "team"])
        .agg(pl.col("epa").sum().alias("epa_sum"), pl.len().cast(pl.Float64).alias("plays"))
    )
    if team_week.height == 0:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "unit": pl.Utf8,
                "team": pl.Utf8,
                "epa_per_play": pl.Float64,
                "plays_prior": pl.Float64,
                "weakness_z": pl.Float64,
                "rank": pl.Float64,
            }
        )

    # A dense (season, week) x (team, unit) grid, so cum_sum().shift(1) really
    # means "everything before this week" even across a bye.
    weeks = team_week.select("season", "week").unique()
    entities = team_week.select("season", "unit", "team").unique()
    full = (
        weeks.join(entities, on="season", how="inner")
        .join(team_week, on=["season", "week", "unit", "team"], how="left")
        .with_columns(pl.col("epa_sum").fill_null(0.0), pl.col("plays").fill_null(0.0))
        .sort(["season", "unit", "team", "week"])
    )
    full = full.with_columns(
        pl.col("epa_sum")
        .cum_sum()
        .shift(1, fill_value=0.0)
        .over(["season", "unit", "team"])
        .alias("epa_prior"),
        pl.col("plays")
        .cum_sum()
        .shift(1, fill_value=0.0)
        .over(["season", "unit", "team"])
        .alias("plays_prior"),
    )
    full = full.with_columns(
        pl.when(pl.col("plays_prior") >= min_plays)
        .then(pl.col("epa_prior") / pl.col("plays_prior"))
        .otherwise(None)
        .alias("epa_per_play")
    )

    group = ["season", "week", "unit"]
    mean = pl.col("epa_per_play").mean().over(group)
    std = pl.col("epa_per_play").std().over(group)
    raw_z = pl.when(std > 0).then((pl.col("epa_per_play") - mean) / std).otherwise(None)
    # Defence: higher EPA allowed = weaker, so the z-score is already oriented.
    # Offence: higher EPA scored = *stronger*, so negate to keep "higher = weaker".
    sign = pl.when(pl.col("unit").str.ends_with("_defense")).then(1.0).otherwise(-1.0)
    full = full.with_columns((sign * raw_z).clip(-z_clip, z_clip).alias("weakness_z"))
    full = full.with_columns(
        pl.col("weakness_z").rank(method="average").over(group).cast(pl.Float64).alias("rank")
    )
    return full.select(
        "season",
        "week",
        "unit",
        "team",
        "epa_per_play",
        "plays_prior",
        pl.col("weakness_z").fill_null(0.0),
        pl.col("rank").fill_null((N_TEAMS + 1) / 2),
    ).sort(["season", "week", "unit", "team"])


# --------------------------------------------------------------------------- #
# §6 — opponent adjustment
# --------------------------------------------------------------------------- #


def opponent_adjust(
    stat_frame: pl.DataFrame,
    opponent_strength_frame: pl.DataFrame,
    *,
    stat_columns: Sequence[str],
    targets: pl.DataFrame | None = None,
    entity_keys: Sequence[str] = ("player_id",),
    opponent_column: str = "opponent",
    season_column: str = "season",
    week_column: str = "week",
    unit: str | None = None,
    group_column: str | None = None,
    use_rank: bool = False,
    k: float = DEFAULT_K,
    n_teams: int = N_TEAMS,
    z_clip: float = Z_CLIP,
    recency_lambda: float = DEFAULT_LAMBDA,
    shrinkage_games: float = DEFAULT_SHRINKAGE_GAMES,
    prefix: str = "adj_",
    sample_column: str = "games_in_adjustment_sample",
    factor_column: str = "opponent_adjustment_factor_mean",
) -> pl.DataFrame:
    """``docs/DATA_MODEL.md`` §6, end to end, for any entity.

    Parameters
    ----------
    stat_frame
        One row per ``(entity, game)``. Must carry ``entity_keys``, the season
        and week the game was played, an ``opponent_column`` holding the
        opposing team code, and every name in ``stat_columns``.
    opponent_strength_frame
        One row per ``(season, week, team)`` — the shape :func:`unit_strength`
        returns. ``weakness_z`` is used by default, ``rank`` when
        ``use_rank=True``. Pass ``unit=`` to select one unit from a frame that
        carries several.
    targets
        One row per ``(entity, game being predicted)``, carrying ``entity_keys``
        plus season/week. When given, steps 3-4 run and the result is one row
        per target holding the aggregated, recency-weighted, shrunk value. When
        ``None`` only steps 1-2 run and the per-game adjusted stats come back.
    group_column
        Column on ``targets`` naming the shrinkage pool (usually a position
        group). Without it the pool is every entity at that week.

    Notes
    -----
    Step 3's sample is **strictly** earlier weeks of the same season, so a
    week-*W* row cannot see week *W* or later; that is the §6 leakage rule and
    leakage-taxonomy item 2. The step-4 prior is computed from that same as-of
    sample, so it cannot look forward either.
    """
    stat_columns = list(stat_columns)
    entity_keys = list(entity_keys)
    needed = [*entity_keys, season_column, week_column, opponent_column, *stat_columns]
    absent = [c for c in needed if c not in stat_frame.columns]
    if absent:
        raise ValueError(f"stat_frame is missing columns: {absent}")

    strength = opponent_strength_frame
    if unit is not None:
        if "unit" not in strength.columns:
            raise ValueError("unit= given but opponent_strength_frame has no 'unit' column")
        strength = strength.filter(pl.col("unit") == unit)
    value_column = "rank" if use_rank else "weakness_z"
    if value_column not in strength.columns:
        raise ValueError(f"opponent_strength_frame has no {value_column!r} column")
    strength = strength.select(
        pl.col(season_column).cast(pl.Int64),
        pl.col(week_column).cast(pl.Int64),
        pl.col("team").alias(opponent_column),
        pl.col(value_column).cast(pl.Float64).alias(f"_{value_column}"),
    ).unique(subset=[season_column, week_column, opponent_column], keep="first")

    per_game = (
        stat_frame.with_columns(
            pl.col(season_column).cast(pl.Int64), pl.col(week_column).cast(pl.Int64)
        )
        .join(strength, on=[season_column, week_column, opponent_column], how="left")
        .with_columns(
            _multiplier_expr(use_rank=use_rank, k=k, n_teams=n_teams, z_clip=z_clip).alias(
                "opponent_adjustment_factor"
            )
        )
    )
    per_game = per_game.with_columns(
        [
            (pl.col(c) / pl.col("opponent_adjustment_factor")).alias(f"{prefix}{c}")
            for c in stat_columns
        ]
    ).drop(f"_{value_column}")

    if targets is None:
        return per_game

    adj_columns = [f"{prefix}{c}" for c in stat_columns]
    source = per_game.select(
        *entity_keys,
        pl.col(season_column),
        pl.col(week_column).alias("_source_week"),
        pl.col("opponent_adjustment_factor"),
        *adj_columns,
    )
    targets_indexed = targets.with_columns(
        pl.col(season_column).cast(pl.Int64), pl.col(week_column).cast(pl.Int64)
    ).with_row_index("_target_id")
    if group_column is not None and group_column not in targets_indexed.columns:
        raise ValueError(f"targets has no group column {group_column!r}")

    pairs = targets_indexed.join(source, on=[*entity_keys, season_column], how="inner").filter(
        pl.col("_source_week") < pl.col(week_column)
    )
    pairs = pairs.with_columns(
        (pl.lit(recency_lambda) ** (pl.col(week_column) - pl.col("_source_week"))).alias("_w")
    )

    def _weighted(column: str) -> pl.Expr:
        present = pl.col(column).is_not_null().cast(pl.Float64)
        return _safe_div(
            (pl.col("_w") * pl.col(column).fill_null(0.0) * present).sum(),
            (pl.col("_w") * present).sum(),
        ).alias(column)

    aggregated = pairs.group_by("_target_id").agg(
        *[_weighted(c) for c in adj_columns],
        pl.col("opponent_adjustment_factor").mean().alias(factor_column),
        pl.len().cast(pl.Float64).alias(sample_column),
    )
    pool_keys = [season_column, week_column] + ([group_column] if group_column else [])
    pool = pairs.group_by(pool_keys).agg(*[_weighted(c).alias(f"_mu_{c}") for c in adj_columns])

    out = targets_indexed.join(aggregated, on="_target_id", how="left").join(
        pool, on=pool_keys, how="left"
    )
    n = pl.col(sample_column).fill_null(0.0)
    c = pl.lit(float(shrinkage_games))
    shrunk = []
    for column in adj_columns:
        mu, raw = pl.col(f"_mu_{column}"), pl.col(column)
        blended = (n * raw.fill_null(mu) + c * mu.fill_null(raw)) / (n + c)
        shrunk.append(
            pl.when(raw.is_null() & mu.is_null()).then(None).otherwise(blended).alias(column)
        )
    return (
        out.with_columns(shrunk)
        .with_columns(pl.col(sample_column).fill_null(0.0))
        .drop(["_target_id", *[f"_mu_{c}" for c in adj_columns]])
    )


# --------------------------------------------------------------------------- #
# Generic as-of aggregation (raw, unadjusted)
# --------------------------------------------------------------------------- #


def as_of_aggregate(
    stat_frame: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    value_columns: Sequence[str],
    entity_keys: Sequence[str] = ("player_id",),
    season_column: str = "season",
    week_column: str = "week",
    trailing_games: int | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Sum ``value_columns`` over each entity's games in **strictly earlier weeks**.

    ``trailing_games`` restricts the window to the most recent *n* such games
    (the trailing-8 form features). ``games{suffix}`` carries the sample size.
    """
    value_columns = list(value_columns)
    entity_keys = list(entity_keys)
    source = stat_frame.select(
        *entity_keys,
        pl.col(season_column).cast(pl.Int64),
        pl.col(week_column).cast(pl.Int64).alias("_source_week"),
        *value_columns,
    )
    targets_indexed = targets.with_columns(
        pl.col(season_column).cast(pl.Int64), pl.col(week_column).cast(pl.Int64)
    ).with_row_index("_target_id")
    pairs = targets_indexed.join(source, on=[*entity_keys, season_column], how="inner").filter(
        pl.col("_source_week") < pl.col(week_column)
    )
    if trailing_games is not None:
        pairs = pairs.filter(
            pl.col("_source_week").rank(method="ordinal", descending=True).over("_target_id")
            <= trailing_games
        )
    aggregated = pairs.group_by("_target_id").agg(
        *[pl.col(c).sum().alias(f"{c}{suffix}") for c in value_columns],
        pl.len().cast(pl.Float64).alias(f"games{suffix}"),
    )
    return (
        targets_indexed.join(aggregated, on="_target_id", how="left")
        .with_columns(pl.col(f"games{suffix}").fill_null(0.0))
        .drop("_target_id")
    )


def _prior_totals(
    frame: pl.DataFrame,
    *,
    entity_keys: Sequence[str],
    value_columns: Sequence[str],
) -> pl.DataFrame:
    """Cumulative ``value_columns`` over **strictly earlier weeks** of the season.

    ``frame`` is one row per ``(season, week, *entity_keys)``. The dense
    ``(season, week) x entity`` grid is what makes ``cum_sum().shift(1)`` mean
    "before this week" rather than "before this entity's previous appearance",
    which matters across byes.
    """
    entity_keys = list(entity_keys)
    value_columns = list(value_columns)
    if frame.height == 0:
        return frame
    weeks = frame.select("season", "week").unique()
    entities = frame.select("season", *entity_keys).unique()
    full = (
        weeks.join(entities, on="season", how="inner")
        .join(frame, on=["season", "week", *entity_keys], how="left")
        .with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in value_columns])
        .sort(["season", *entity_keys, "week"])
    )
    return full.with_columns(
        [
            pl.col(c).cum_sum().shift(1, fill_value=0.0).over(["season", *entity_keys]).alias(c)
            for c in value_columns
        ]
    )


# --------------------------------------------------------------------------- #
# Starter identification
# --------------------------------------------------------------------------- #

#: Source priority: a published depth chart beats inferred snap share, which
#: beats "on the active roster and experienced", which beats raw pbp usage.
_SOURCE_PRIORITY = {"depth_chart": 1, "snap_counts": 2, "roster": 3, "pbp_usage": 4}

#: pbp roles that stand in for a snap count when nothing else is available.
_USAGE_COLUMNS = (
    "pass_dropbacks",
    "rush_attempts",
    "targets",
    "tackles",
    "pressures",
    "pass_defensed",
    "kick_returns",
    "punt_returns",
)


@lru_cache(maxsize=1)
def _slotting_map() -> dict[str, str]:
    """Every position this module can slot, resolved through :func:`position_group`.

    Built from ``position_impact.csv`` as well as :data:`POSITION_GROUPS`, so a
    position the reference table knows about cannot silently fall out of the
    starter pool just because it is missing from the literal map.
    """
    table = {position: position_group(position) for position in load_position_impact()}
    return {**table, **POSITION_GROUPS}


def _group_expr(column: str) -> pl.Expr:
    """Vectorised :func:`position_group`; unknown positions are not slotted."""
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .replace_strict(_slotting_map(), default=None, return_dtype=pl.Utf8)
    )


def _player_usage_as_of(stats: pl.DataFrame) -> pl.DataFrame:
    """Per ``(season, week, team, player_id)`` snap-proxy usage in earlier weeks."""
    rates = stats if "pressures" in stats.columns else player_game_rates(stats)
    available = [c for c in _USAGE_COLUMNS if c in rates.columns]
    per_week = (
        rates.group_by(["season", "week", "team", "player_id"])
        .agg([pl.col(c).sum().alias(c) for c in available])
        .with_columns(pl.sum_horizontal([pl.col(c) for c in available]).alias("usage"))
        .select("season", "week", "team", "player_id", "usage")
    )
    prior = _prior_totals(per_week, entity_keys=["team", "player_id"], value_columns=["usage"])
    return prior.select("season", "week", "team", "player_id", pl.col("usage").alias("usage_prior"))


def _depth_chart_candidates(depth_charts: pl.DataFrame) -> pl.DataFrame | None:
    names = set(depth_charts.columns)
    team_col = "club_code" if "club_code" in names else ("team" if "team" in names else None)
    id_col = "gsis_id" if "gsis_id" in names else ("player_id" if "player_id" in names else None)
    if team_col is None or id_col is None or "position" not in names:
        return None
    depth = (
        pl.col("depth_team").cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
        if "depth_team" in names
        else pl.lit(1.0)
    )
    frame = depth_charts
    if "formation" in names:
        # A special-teams row lists a gunner or returner at their *offensive*
        # position with depth 1, which would otherwise outrank the real WR1.
        # Kickers, punters and long snappers are the rows we actually want.
        specialist = (
            pl.col("position")
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in(["K", "PK", "P", "LS", "KR", "PR"])
        )
        frame = frame.filter((pl.col("formation") != "Special Teams") | specialist)
    return (
        frame.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col(team_col).cast(pl.Utf8).alias("team"),
            pl.col(id_col).cast(pl.Utf8).alias("player_id"),
            pl.col("position").cast(pl.Utf8).alias("position"),
            depth.alias("depth_rank"),
        )
        .filter(pl.col("player_id").is_not_null() & pl.col("depth_rank").is_not_null())
        .with_columns(
            pl.lit("depth_chart").alias("starter_source"), pl.lit(0.0).alias("source_usage")
        )
    )


def _snap_count_candidates(
    snap_counts: pl.DataFrame, rosters: pl.DataFrame | None
) -> pl.DataFrame | None:
    """Prior-week snap share as a depth proxy.

    ``snap_counts`` keys players by ``pfr_player_id``; the weekly roster is the
    only bridge to GSIS ids, so without it this source is unusable. Only weeks
    strictly earlier than the target are read — a week's own snap counts are
    postgame truth (leakage taxonomy item 3).
    """
    names = set(snap_counts.columns)
    if rosters is None or "pfr_player_id" not in names:
        return None
    if not {"pfr_id", "gsis_id"} <= set(rosters.columns):
        return None
    bridge = (
        rosters.select(
            pl.col("pfr_id").cast(pl.Utf8).alias("pfr_player_id"),
            pl.col("gsis_id").cast(pl.Utf8).alias("player_id"),
        )
        .drop_nulls()
        .unique(subset=["pfr_player_id"], keep="first")
    )
    share = pl.max_horizontal(
        pl.col("offense_pct").cast(pl.Float64).fill_null(0.0)
        if "offense_pct" in names
        else pl.lit(0.0),
        pl.col("defense_pct").cast(pl.Float64).fill_null(0.0)
        if "defense_pct" in names
        else pl.lit(0.0),
    )
    weekly = (
        snap_counts.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("team").cast(pl.Utf8),
            pl.col("pfr_player_id").cast(pl.Utf8),
            pl.col("position").cast(pl.Utf8).alias("position"),
            share.alias("snap_share"),
        )
        .join(bridge, on="pfr_player_id", how="inner")
        .group_by(["season", "week", "team", "player_id"])
        .agg(pl.col("position").first(), pl.col("snap_share").max())
    )
    prior = _prior_totals(
        weekly.select("season", "week", "team", "player_id", "snap_share"),
        entity_keys=["team", "player_id"],
        value_columns=["snap_share"],
    )
    positions = weekly.select("season", "team", "player_id", "position").unique(
        subset=["season", "team", "player_id"], keep="first"
    )
    return (
        prior.filter(pl.col("snap_share") > 0)
        .join(positions, on=["season", "team", "player_id"], how="left")
        .select(
            "season",
            "week",
            "team",
            "player_id",
            "position",
            pl.lit(None, dtype=pl.Float64).alias("depth_rank"),
            pl.lit("snap_counts").alias("starter_source"),
            pl.col("snap_share").alias("source_usage"),
        )
    )


def _roster_candidates(rosters: pl.DataFrame) -> pl.DataFrame | None:
    names = set(rosters.columns)
    if "gsis_id" not in names or "position" not in names or "week" not in names:
        return None
    active = (
        pl.col("status").cast(pl.Utf8).str.to_uppercase().is_in(["ACT", "A01", "ACTIVE"])
        if "status" in names
        else pl.lit(True)
    )
    experience = pl.col("years_exp").cast(pl.Float64) if "years_exp" in names else pl.lit(0.0)
    return (
        rosters.filter(active)
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("team").cast(pl.Utf8),
            pl.col("gsis_id").cast(pl.Utf8).alias("player_id"),
            pl.col("position").cast(pl.Utf8).alias("position"),
            pl.lit(None, dtype=pl.Float64).alias("depth_rank"),
            pl.lit("roster").alias("starter_source"),
            experience.fill_null(0.0).alias("source_usage"),
        )
        .filter(pl.col("player_id").is_not_null())
    )


def _pbp_candidates(stats: pl.DataFrame) -> pl.DataFrame:
    """Last-ditch starters: whoever actually took the snaps in earlier weeks.

    Position is inferred from the pbp role, so this can only produce QB, RB and
    WR — enough to keep the QB1 block alive on a pbp-only input.
    """
    per_week = stats.group_by(["season", "week", "team", "player_id"]).agg(
        [pl.col(c).sum().alias(c) for c in ("pass_dropbacks", "rush_attempts", "targets")]
    )
    prior = _prior_totals(
        per_week,
        entity_keys=["team", "player_id"],
        value_columns=["pass_dropbacks", "rush_attempts", "targets"],
    )
    role = (
        pl.when(
            (pl.col("pass_dropbacks") >= pl.col("rush_attempts"))
            & (pl.col("pass_dropbacks") >= pl.col("targets"))
            & (pl.col("pass_dropbacks") > 0)
        )
        .then(pl.lit("QB"))
        .when((pl.col("rush_attempts") >= pl.col("targets")) & (pl.col("rush_attempts") > 0))
        .then(pl.lit("RB"))
        .when(pl.col("targets") > 0)
        .then(pl.lit("WR"))
        .otherwise(None)
    )
    return (
        prior.with_columns(
            role.alias("position"),
            pl.sum_horizontal("pass_dropbacks", "rush_attempts", "targets").alias("source_usage"),
        )
        .filter(pl.col("position").is_not_null())
        .select(
            "season",
            "week",
            "team",
            "player_id",
            "position",
            pl.lit(None, dtype=pl.Float64).alias("depth_rank"),
            pl.lit("pbp_usage").alias("starter_source"),
            "source_usage",
        )
    )


_STARTER_SCHEMA: dict[str, Any] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team": pl.Utf8,
    "player_id": pl.Utf8,
    "position": pl.Utf8,
    "position_group": pl.Utf8,
    "depth_rank": pl.Float64,
    "slot_index": pl.Int64,
    "slot": pl.Utf8,
    "starter_source": pl.Utf8,
}


def identify_starters(
    team_weeks: pl.DataFrame,
    *,
    depth_charts: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    rosters: pl.DataFrame | None = None,
    stats: pl.DataFrame | None = None,
    slot_counts: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """Assign starter slots for every ``(season, week, team)`` in ``team_weeks``.

    Sources are tried in priority order — published depth chart, prior-week snap
    share, active roster, raw pbp usage — and within a source by depth chart
    position then by prior-week usage. One row per assigned slot, carrying
    ``slot`` / ``slot_index`` / ``position_group`` / ``starter_source``.
    """
    slots = dict(slot_counts or SLOT_COUNTS)
    usage_schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "team": pl.Utf8,
        "player_id": pl.Utf8,
        "usage_prior": pl.Float64,
    }
    usage = (
        _player_usage_as_of(stats)
        if stats is not None and stats.height
        else pl.DataFrame(schema=usage_schema)
    )

    candidates: list[pl.DataFrame] = []
    if depth_charts is not None and depth_charts.height:
        frame = _depth_chart_candidates(depth_charts)
        if frame is not None:
            candidates.append(frame)
    if snap_counts is not None and snap_counts.height:
        frame = _snap_count_candidates(snap_counts, rosters)
        if frame is not None:
            candidates.append(frame)
    if rosters is not None and rosters.height:
        frame = _roster_candidates(rosters)
        if frame is not None:
            candidates.append(frame)
    if stats is not None and stats.height:
        candidates.append(_pbp_candidates(stats))
    if not candidates:
        return pl.DataFrame(schema=_STARTER_SCHEMA)

    pool = (
        pl.concat(candidates, how="diagonal_relaxed")
        .join(
            team_weeks.select(
                pl.col("season").cast(pl.Int64),
                pl.col("week").cast(pl.Int64),
                pl.col("team").cast(pl.Utf8),
            ).unique(),
            on=["season", "week", "team"],
            how="inner",
        )
        .with_columns(
            _group_expr("position").alias("position_group"),
            pl.col("starter_source")
            .replace_strict(_SOURCE_PRIORITY, default=9, return_dtype=pl.Int64)
            .alias("_priority"),
            pl.col("depth_rank").cast(pl.Float64),
            pl.col("source_usage").cast(pl.Float64).fill_null(0.0),
        )
        .filter(pl.col("position_group").is_not_null())
    )
    if pool.height == 0:
        return pl.DataFrame(schema=_STARTER_SCHEMA)
    pool = pool.join(usage, on=["season", "week", "team", "player_id"], how="left").with_columns(
        pl.col("usage_prior").fill_null(0.0)
    )
    # One row per player per group: the strongest claim they have on it. The
    # sort has to survive the dedupe, hence maintain_order -- without it polars
    # keeps an arbitrary row and a depth-chart starter can lose to their own
    # roster row, which carries no depth rank.
    pool = pool.sort(
        ["_priority", "depth_rank", "source_usage", "usage_prior"],
        descending=[False, False, True, True],
        nulls_last=True,
    ).unique(
        subset=["season", "week", "team", "player_id", "position_group"],
        keep="first",
        maintain_order=True,
    )

    ordered = pool.sort(
        ["_priority", "depth_rank", "usage_prior", "source_usage", "player_id"],
        descending=[False, False, True, True, False],
        nulls_last=True,
    ).with_columns(
        (pl.int_range(pl.len()).over(["season", "week", "team", "position_group"]) + 1)
        .cast(pl.Int64)
        .alias("slot_index")
    )
    limits = pl.DataFrame(
        {
            "position_group": list(slots),
            "_limit": pl.Series([int(v) for v in slots.values()], dtype=pl.Int64),
        }
    )
    return (
        ordered.join(limits, on="position_group", how="inner")
        .filter(pl.col("slot_index") <= pl.col("_limit"))
        .with_columns(
            pl.concat_str([pl.col("position_group"), pl.col("slot_index").cast(pl.Utf8)]).alias(
                "slot"
            )
        )
        .select(list(_STARTER_SCHEMA))
        .sort(["season", "week", "team", "position_group", "slot_index"])
    )


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #

_AVAILABILITY_SCHEMA: dict[str, Any] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team": pl.Utf8,
    "player_id": pl.Utf8,
    "position": pl.Utf8,
    "report_status": pl.Utf8,
    "practice_status": pl.Utf8,
    "availability_prob": pl.Float64,
}


def _injury_columns(frame: pl.DataFrame) -> list[pl.Expr]:
    names = set(frame.columns)
    id_col = "gsis_id" if "gsis_id" in names else "player_id"
    if id_col not in names:
        raise ValueError("injuries frame has neither 'gsis_id' nor 'player_id'")
    optional = {
        "position": pl.Utf8,
        "report_status": pl.Utf8,
        "practice_status": pl.Utf8,
    }
    exprs: list[pl.Expr] = [
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.Utf8),
        pl.col(id_col).cast(pl.Utf8).alias("player_id"),
    ]
    for name, dtype in optional.items():
        exprs.append(
            pl.col(name).cast(dtype).alias(name)
            if name in names
            else pl.lit(None, dtype=dtype).alias(name)
        )
    return exprs


def player_availability(injuries: pl.DataFrame) -> pl.DataFrame:
    """``P(available)`` per weekly injury-report row, via the rule-based scorer.

    The weekly report for week *W* is published before week *W*'s kickoff, so
    reading it for the game's own week is pregame, not leakage. What *is*
    forbidden — ``did_play``, ``snap_share_if_played`` — is never touched.

    ``availability_probability`` is Python per call, but there are only a few
    dozen distinct ``(report_status, practice_status)`` pairs in a season, so it
    is evaluated once per pair and joined back.
    """
    if injuries is None or injuries.height == 0:
        return pl.DataFrame(schema=_AVAILABILITY_SCHEMA)
    frame = injuries.select(_injury_columns(injuries)).filter(pl.col("player_id").is_not_null())
    distinct = frame.select("report_status", "practice_status").unique()
    rows = [
        {
            "report_status": report,
            "practice_status": practice,
            "availability_prob": availability_probability(report, practice_fri=practice),
        }
        for report, practice in zip(
            distinct.get_column("report_status").to_list(),
            distinct.get_column("practice_status").to_list(),
            strict=True,
        )
    ]
    lookup = pl.DataFrame(
        rows,
        schema={
            "report_status": pl.Utf8,
            "practice_status": pl.Utf8,
            "availability_prob": pl.Float64,
        },
    )
    return frame.join(lookup, on=["report_status", "practice_status"], how="left").with_columns(
        pl.col("availability_prob").fill_null(1.0)
    )


_TEAM_INJURY_SCHEMA: dict[str, Any] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team": pl.Utf8,
    "injury_expected_value_lost": pl.Float64,
    "injury_qb_absence_cost": pl.Float64,
    "injury_score": pl.Float64,
    "injury_n_out": pl.Float64,
    "injury_n_doubtful": pl.Float64,
    "injury_n_questionable": pl.Float64,
    "injury_n_items": pl.Float64,
}


def _summary_row(team: str, items: Sequence[InjuryItem]) -> dict[str, Any]:
    summary = score_injury_items(items, team=team)
    return {
        "injury_expected_value_lost": summary.total_absence_cost,
        "injury_qb_absence_cost": summary.qb_absence_cost,
        "injury_score": summary.injury_sentiment_score,
        "injury_n_out": float(summary.n_out),
        "injury_n_doubtful": float(summary.n_doubtful),
        "injury_n_questionable": float(summary.n_questionable),
        "injury_n_items": float(summary.n_items),
    }


def team_availability(
    team_weeks: pl.DataFrame,
    injuries: pl.DataFrame | None,
) -> pl.DataFrame:
    """Team-level expected value lost, per ``(season, week, team)``.

    The rollup goes through :func:`nfl.data.sentiment.injury_rules.score_injury_items`,
    so the position weighting is exactly ``position_impact.csv`` and the depth
    discount is the one the sentiment pipeline already applies. The loop is over
    team-weeks (a few thousand), never over plays.
    """
    if injuries is None or injuries.height == 0:
        return pl.DataFrame(schema=_TEAM_INJURY_SCHEMA)
    wanted = team_weeks.select(
        pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), pl.col("team").cast(pl.Utf8)
    ).unique()
    scoped = (
        injuries.select(_injury_columns(injuries))
        .join(wanted, on=["season", "week", "team"], how="inner")
        .filter(pl.col("player_id").is_not_null())
    )
    if scoped.height == 0:
        return pl.DataFrame(schema=_TEAM_INJURY_SCHEMA)

    rows: list[dict[str, Any]] = []
    for (season, week, team), frame in scoped.partition_by(
        ["season", "week", "team"], as_dict=True
    ).items():
        items = [
            InjuryItem(
                player_id=row["player_id"] or "",
                team=str(team),
                position=row["position"],
                status=row["report_status"],
                practice_fri=row["practice_status"],
            )
            for row in frame.iter_rows(named=True)
        ]
        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "team": str(team),
                **_summary_row(str(team), items),
            }
        )
    scored = pl.DataFrame(rows, schema=_TEAM_INJURY_SCHEMA)
    # A team absent from a week the feed *does* cover has nobody on its report,
    # which is zero value lost -- not unknown. A week the feed does not cover at
    # all stays null, because there we genuinely do not know.
    covered = scored.select("season", "week").unique()
    value_columns = [c for c in _TEAM_INJURY_SCHEMA if c not in {"season", "week", "team"}]
    return (
        wanted.join(covered, on=["season", "week"], how="inner")
        .join(scored, on=["season", "week", "team"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in value_columns])
    )


# --------------------------------------------------------------------------- #
# Team-level as-of proxies (OL / DL / special teams)
# --------------------------------------------------------------------------- #


def team_sack_rates(pbp: pl.DataFrame) -> pl.DataFrame:
    """As-of-week sack rate allowed (OL proxy) and generated (DL proxy).

    The line units have no per-player pbp production to rank, so the team's own
    prior-weeks sack rate stands in for them, ranked across the league at the
    same week.
    """
    names = set(pbp.columns)
    base = pbp.select(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam").cast(pl.Utf8),
        pl.col("defteam").cast(pl.Utf8),
        _flag(names, "pass_attempt"),
        _flag(names, "qb_dropback"),
        _flag(names, "sack"),
    ).filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    dropback = pl.max_horizontal(pl.col("qb_dropback"), pl.col("pass_attempt"))
    base = base.filter(dropback == 1.0)
    offense = base.group_by(["season", "week", pl.col("posteam").alias("team")]).agg(
        pl.len().cast(pl.Float64).alias("dropbacks"), pl.col("sack").sum().alias("sacks_allowed")
    )
    defense = base.group_by(["season", "week", pl.col("defteam").alias("team")]).agg(
        pl.len().cast(pl.Float64).alias("dropbacks_faced"),
        pl.col("sack").sum().alias("sacks_generated"),
    )
    merged = offense.join(defense, on=["season", "week", "team"], how="full", coalesce=True)
    if merged.height == 0:
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "team": pl.Utf8,
                "sack_rate_allowed": pl.Float64,
                "sack_rate_allowed_rank": pl.Float64,
                "sack_rate_generated": pl.Float64,
                "sack_rate_generated_rank": pl.Float64,
            }
        )
    value_columns = ["dropbacks", "sacks_allowed", "dropbacks_faced", "sacks_generated"]
    prior = _prior_totals(merged, entity_keys=["team"], value_columns=value_columns)
    prior = prior.with_columns(
        _safe_div(pl.col("sacks_allowed"), pl.col("dropbacks")).alias("sack_rate_allowed"),
        _safe_div(pl.col("sacks_generated"), pl.col("dropbacks_faced")).alias(
            "sack_rate_generated"
        ),
    )
    group = ["season", "week"]
    return prior.select(
        "season",
        "week",
        "team",
        "sack_rate_allowed",
        # Rank 1 = fewest sacks allowed = best offensive line.
        pl.col("sack_rate_allowed")
        .rank(method="average")
        .over(group)
        .alias("sack_rate_allowed_rank"),
        "sack_rate_generated",
        # Rank 1 = most sacks generated = best pass rush.
        pl.col("sack_rate_generated")
        .rank(method="average", descending=True)
        .over(group)
        .alias("sack_rate_generated_rank"),
    )


def team_return_game(stats: pl.DataFrame) -> pl.DataFrame:
    """As-of-week return production and the team's primary returner."""
    per_week = stats.group_by(["season", "week", "team", "player_id"]).agg(
        [
            pl.col(c).sum().alias(c)
            for c in ("kick_returns", "kick_return_yards", "punt_returns", "punt_return_yards")
        ]
    )
    prior = _prior_totals(
        per_week,
        entity_keys=["team", "player_id"],
        value_columns=["kick_returns", "kick_return_yards", "punt_returns", "punt_return_yards"],
    ).with_columns((pl.col("kick_returns") + pl.col("punt_returns")).alias("returns"))
    team_totals = (
        prior.group_by(["season", "week", "team"])
        .agg(
            pl.col("kick_returns").sum(),
            pl.col("kick_return_yards").sum(),
            pl.col("punt_returns").sum(),
            pl.col("punt_return_yards").sum(),
        )
        .with_columns(
            _safe_div(pl.col("kick_return_yards"), pl.col("kick_returns")).alias(
                "kick_return_yards_per_return"
            ),
            _safe_div(pl.col("punt_return_yards"), pl.col("punt_returns")).alias(
                "punt_return_yards_per_return"
            ),
            (pl.col("kick_returns") + pl.col("punt_returns")).alias("returns_in_sample"),
        )
    )
    primary = (
        prior.filter(pl.col("returns") > 0)
        .sort(["returns", "player_id"], descending=[True, False])
        .unique(subset=["season", "week", "team"], keep="first", maintain_order=True)
        .select("season", "week", "team", pl.col("player_id").alias("returner_player_id"))
    )
    return team_totals.join(primary, on=["season", "week", "team"], how="left").select(
        "season",
        "week",
        "team",
        "returner_player_id",
        "kick_return_yards_per_return",
        "punt_return_yards_per_return",
        "returns_in_sample",
    )


def _prior_snap_share(
    snap_counts: pl.DataFrame | None, rosters: pl.DataFrame | None
) -> pl.DataFrame:
    """Mean snap share over **earlier** weeks, keyed by GSIS id."""
    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "team": pl.Utf8,
        "player_id": pl.Utf8,
        "snap_share_prior": pl.Float64,
    }
    if snap_counts is None or snap_counts.height == 0 or rosters is None:
        return pl.DataFrame(schema=schema)
    if _snap_count_candidates(snap_counts, rosters) is None:
        return pl.DataFrame(schema=schema)
    weekly = (
        snap_counts.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("team").cast(pl.Utf8),
            pl.col("pfr_player_id").cast(pl.Utf8),
            pl.max_horizontal(
                pl.col("offense_pct").cast(pl.Float64).fill_null(0.0),
                pl.col("defense_pct").cast(pl.Float64).fill_null(0.0),
            ).alias("share"),
        )
        .join(
            rosters.select(
                pl.col("pfr_id").cast(pl.Utf8).alias("pfr_player_id"),
                pl.col("gsis_id").cast(pl.Utf8).alias("player_id"),
            )
            .drop_nulls()
            .unique(subset=["pfr_player_id"], keep="first"),
            on="pfr_player_id",
            how="inner",
        )
        .group_by(["season", "week", "team", "player_id"])
        .agg(pl.col("share").max(), pl.lit(1.0).alias("appearances"))
    )
    prior = _prior_totals(
        weekly.select("season", "week", "team", "player_id", "share", "appearances"),
        entity_keys=["team", "player_id"],
        value_columns=["share", "appearances"],
    )
    return prior.select(
        "season",
        "week",
        "team",
        "player_id",
        _safe_div(pl.col("share"), pl.col("appearances")).alias("snap_share_prior"),
    )


def _position_weight_frame() -> pl.DataFrame:
    """``position_impact.csv`` as a joinable frame."""
    table = load_position_impact()
    return pl.DataFrame(
        {
            "position": list(table),
            "impact_weight": pl.Series([v.impact_weight for v in table.values()], dtype=pl.Float64),
            "replacement_dropoff": pl.Series(
                [v.replacement_dropoff for v in table.values()], dtype=pl.Float64
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Feature documentation — also the canonical output column order
# --------------------------------------------------------------------------- #

FEATURE_DOCS: dict[str, str] = {
    "pl__week": "Week of the game. Sample-size context for every as-of column below.",
    "pl__prior_games_in_season": "Team games already played this season, strictly before this one.",
    "pl__starters_identified": "Number of starter slots filled from depth chart / snaps / roster.",
    # ---- QB1 ---------------------------------------------------------------
    "pl__qb1_player_id": "GSIS id of the projected starting QB. Identifier, not a model input.",
    "pl__qb1_adj_epa_per_dropback": (
        "QB1 EPA per dropback, opponent-adjusted vs each opposing pass defence "
        "(DATA_MODEL §6), recency-weighted and shrunk toward the QB pool mean."
    ),
    "pl__qb1_epa_per_dropback": "QB1 raw EPA per dropback, season to date (earlier weeks only).",
    "pl__qb1_epa_per_dropback_l8": "QB1 raw EPA per dropback over their trailing 8 games.",
    "pl__qb1_passing_yards_per_game": "QB1 passing yards per game, season to date.",
    "pl__qb1_passing_yards_per_game_l8": "QB1 passing yards per game over the trailing 8 games.",
    "pl__qb1_adj_passing_yards": "QB1 per-game passing yards, opponent-adjusted per §6.",
    "pl__qb1_passing_tds_per_game": "QB1 passing touchdowns per game, season to date.",
    "pl__qb1_interceptions_per_game": "QB1 interceptions thrown per game, season to date.",
    "pl__qb1_sack_rate_taken": "QB1 sacks taken per dropback, season to date.",
    "pl__qb1_completion_pct": "QB1 completions / attempts, season to date.",
    "pl__qb1_cpoe": "QB1 mean completion percentage over expected, season to date.",
    "pl__qb1_rushing_yards_per_game": "QB1 rushing yards per game, season to date.",
    "pl__qb1_games_in_sample": "Games in QB1's as-of sample; 0 in week 1 and for a new starter.",
    "pl__qb1_position_rank": "QB1's rank among league starting QBs by adjusted EPA/dropback, as of week.",
    "pl__qb1_position_percentile": "Same ranking expressed in [0, 1]; 1 = best QB in the league.",
    "pl__qb1_age_years": "QB1 age at kickoff, in years.",
    "pl__qb1_season_number": "QB1 NFL season number, 1-indexed (rookie = 1).",
    "pl__qb1_draft_round": "QB1 draft round, when the roster feed carries one; else null.",
    "pl__qb1_draft_pick": "QB1 overall draft pick number; null if undrafted or unknown.",
    "pl__qb1_contract_years_remaining": (
        "Always null. DATA_MODEL §5.2 wants it; no contract source is ingested yet."
    ),
    "pl__qb1_is_rookie": "1 when QB1 is in their first NFL season.",
    "pl__qb1_available": "1 when QB1's rule-based availability is >= 0.5; 0 when Out/Doubtful.",
    "pl__qb1_availability_prob": (
        "P(QB1 plays) from nfl.data.sentiment.injury_rules on the pre-kickoff report."
    ),
    # ---- RB1 ---------------------------------------------------------------
    "pl__rb1_adj_epa_per_rush": "RB1 rush EPA per carry, opponent-adjusted vs opposing rush defences.",
    "pl__rb1_adj_rushing_yards": "RB1 per-game rushing yards, opponent-adjusted per §6.",
    "pl__rb1_rushing_yards_per_game": "RB1 rushing yards per game, season to date.",
    "pl__rb1_rushing_yards_per_game_l8": "RB1 rushing yards per game over the trailing 8 games.",
    "pl__rb1_rushing_tds_per_game": "RB1 rushing touchdowns per game, season to date.",
    "pl__rb1_position_rank": "RB1's rank among league starting RBs by adjusted rush EPA.",
    "pl__rb1_games_in_sample": "Games in RB1's as-of sample.",
    # ---- receivers ---------------------------------------------------------
    "pl__wr1_adj_epa_per_target": "WR1 receiving EPA per target, opponent-adjusted vs pass defences.",
    "pl__wr1_receiving_yards_per_game": "WR1 receiving yards per game, season to date.",
    "pl__wr1_adj_receiving_yards": "WR1 per-game receiving yards, opponent-adjusted per §6.",
    "pl__wr1_position_rank": "WR1's rank among league starting WRs by adjusted EPA per target.",
    "pl__wr_corps_adj_epa_per_target": "Mean adjusted EPA per target across the WR1-WR3 slots.",
    "pl__wr_corps_receiving_yards_per_game": "Combined WR1-WR3 receiving yards per game.",
    "pl__wr_corps_yac_per_reception": "Adjusted yards after catch per reception across WR1-WR3.",
    "pl__te1_adj_epa_per_target": "TE1 receiving EPA per target, opponent-adjusted.",
    "pl__te1_receiving_yards_per_game": "TE1 receiving yards per game, season to date.",
    "pl__te1_position_rank": "TE1's rank among league starting TEs by adjusted EPA per target.",
    "pl__skill_adj_epa_sum": "Sum of adjusted per-play EPA across RB1, WR1-3 and TE1.",
    "pl__skill_yards_per_game_sum": "Combined per-game rushing+receiving yards for those six slots.",
    # ---- lines and defence -------------------------------------------------
    "pl__ol_sack_rate_allowed": "Team sack rate allowed per dropback, season to date (OL proxy).",
    "pl__ol_sack_rate_allowed_rank": "League rank of that sack rate; 1 = fewest sacks allowed.",
    "pl__ol_starters_available": "OL starter slots whose availability probability is >= 0.5.",
    "pl__dl_sack_rate_generated": "Team sack rate generated per dropback faced (DL proxy).",
    "pl__dl_sack_rate_generated_rank": "League rank of that sack rate; 1 = most sacks generated.",
    "pl__dl_adj_pressures_per_game": (
        "Mean adjusted pressures (sacks + half of QB hits and TFLs) per game across DL starters."
    ),
    "pl__dl1_position_rank": "DL1's rank among league starting DL by adjusted pressures.",
    "pl__lb_adj_tackles_per_game": "Mean adjusted tackles per game across LB starters.",
    "pl__lb1_position_rank": "LB1's rank among league starting LBs by adjusted tackles.",
    "pl__cb_adj_pass_defensed_per_game": "Mean adjusted passes defensed per game across CB starters.",
    "pl__cb1_position_rank": "CB1's rank among league starting CBs by adjusted passes defensed.",
    "pl__s_adj_interceptions_per_game": "Mean adjusted interceptions per game across S starters.",
    "pl__s1_position_rank": "S1's rank among league starting safeties by adjusted ball production.",
    # ---- availability ------------------------------------------------------
    "pl__injury_expected_value_lost": (
        "Team expected value lost to injury: sum over reported players of "
        "P(absent) x position_impact.csv weight x replacement dropoff, depth-discounted. "
        "Produced by injury_rules.score_injury_items."
    ),
    "pl__injury_qb_absence_cost": "The QB share of that expected value lost.",
    "pl__injury_score": "Team injury sentiment in [-1, 1] from the same scorer; negative is worse.",
    "pl__injury_n_out": "Players designated Out / IR / PUP / suspended on the pre-kickoff report.",
    "pl__injury_n_doubtful": "Players designated Doubtful.",
    "pl__injury_n_questionable": "Players designated Questionable.",
    "pl__injury_n_items": "Rows on the team's pre-kickoff injury report.",
    "pl__starters_out_count": "Identified starters whose availability probability is < 0.5.",
    "pl__starters_expected_value_lost": (
        "Expected value lost restricted to identified starters, weighted the same way."
    ),
    "pl__snap_share_lost": ("Prior-week snap share carried by starters who are unlikely to play."),
    # ---- special teams and roster shape -----------------------------------
    "pl__st_returner_player_id": "GSIS id of the team's primary returner in earlier weeks.",
    "pl__st_kick_return_yards_per_return": "Team kickoff-return yards per return, season to date.",
    "pl__st_punt_return_yards_per_return": "Team punt-return yards per return, season to date.",
    "pl__st_returns_in_sample": "Kick + punt returns in that as-of sample.",
    "pl__starters_mean_age": "Mean age at kickoff across identified starters.",
    "pl__starters_mean_experience": "Mean NFL season number across identified starters.",
    "pl__starters_n_rookies": "Identified starters in their first NFL season.",
}

#: Output dtypes, so an input-starved build still returns the full schema.
FEATURE_DTYPES: dict[str, Any] = {
    name: (pl.Utf8 if name.endswith("_player_id") else pl.Float64) for name in FEATURE_DOCS
}

OUTPUT_COLUMNS: tuple[str, ...] = ("game_id", "team", *FEATURE_DOCS)


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


def _kickoff_expr(names: Iterable[str]) -> pl.Expr:
    """Kickoff timestamp from ``gameday`` (+ ``gametime`` when present)."""
    if "gameday" not in names:
        return pl.lit(None, dtype=pl.Datetime("us")).alias("kickoff")
    day = pl.col("gameday").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    if "gametime" not in names:
        return day.cast(pl.Datetime("us")).alias("kickoff")
    time = pl.col("gametime").cast(pl.Utf8).fill_null("12:00")
    return (
        pl.concat_str([day.cast(pl.Utf8), pl.lit(" "), time])
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=False)
        .fill_null(day.cast(pl.Datetime("us")))
        .alias("kickoff")
    )


def team_game_frame(schedules: pl.DataFrame) -> pl.DataFrame:
    """Schedules unpivoted to one row per ``(game_id, team)`` with kickoff and opponent."""
    names = set(schedules.columns)
    missing = {"game_id", "season", "week", "home_team", "away_team"} - names
    if missing:
        raise ValueError(f"schedules is missing required columns: {sorted(missing)}")
    base = schedules.select(
        pl.col("game_id").cast(pl.Utf8),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("home_team").cast(pl.Utf8),
        pl.col("away_team").cast(pl.Utf8),
        _kickoff_expr(names),
    )
    home = base.select(
        "game_id",
        "season",
        "week",
        "kickoff",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
    )
    away = base.select(
        "game_id",
        "season",
        "week",
        "kickoff",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
    )
    frame = pl.concat([home, away])
    prior_games = (
        frame.group_by(["season", "week", "team"])
        .agg(pl.len().cast(pl.Float64).alias("_games"))
        .pipe(_prior_totals, entity_keys=["team"], value_columns=["_games"])
        .select("season", "week", "team", pl.col("_games").alias("prior_games_in_season"))
    )
    return frame.join(prior_games, on=["season", "week", "team"], how="left").sort(
        ["season", "week", "game_id", "team"]
    )


def _bio_frame(rosters: pl.DataFrame | None, games: pl.DataFrame) -> pl.DataFrame:
    """Per ``(season, week, team, player_id)`` age at kickoff, experience and draft slot."""
    schema = {
        "season": pl.Int64,
        "week": pl.Int64,
        "team": pl.Utf8,
        "player_id": pl.Utf8,
        "age_years": pl.Float64,
        "season_number": pl.Float64,
        "draft_round": pl.Float64,
        "draft_pick": pl.Float64,
        "is_rookie": pl.Float64,
    }
    if rosters is None or rosters.height == 0 or "gsis_id" not in rosters.columns:
        return pl.DataFrame(schema=schema)
    names = set(rosters.columns)
    birth = (
        pl.col("birth_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        if "birth_date" in names
        else pl.lit(None, dtype=pl.Date)
    )
    experience = (
        pl.col("years_exp").cast(pl.Float64) if "years_exp" in names else pl.lit(None, pl.Float64)
    )
    # ``draft_number`` is the overall pick; nflverse weekly rosters carry no
    # round, so ``draft_round`` stays null unless the caller supplies one.
    draft_round = (
        pl.col("draft_round").cast(pl.Float64)
        if "draft_round" in names
        else pl.lit(None, pl.Float64)
    )
    draft_pick = (
        pl.col("draft_number").cast(pl.Float64)
        if "draft_number" in names
        else (
            pl.col("draft_pick").cast(pl.Float64)
            if "draft_pick" in names
            else pl.lit(None, pl.Float64)
        )
    )
    frame = rosters.select(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("team").cast(pl.Utf8),
        pl.col("gsis_id").cast(pl.Utf8).alias("player_id"),
        birth.alias("_birth_date"),
        experience.alias("_years_exp"),
        draft_round.alias("draft_round"),
        draft_pick.alias("draft_pick"),
    ).filter(pl.col("player_id").is_not_null())
    kickoffs = games.select("season", "week", "team", "kickoff").unique(
        subset=["season", "week", "team"], keep="first"
    )
    return (
        frame.join(kickoffs, on=["season", "week", "team"], how="left")
        .with_columns(
            (
                (pl.col("kickoff").cast(pl.Date) - pl.col("_birth_date")).dt.total_days() / 365.25
            ).alias("age_years"),
            (pl.col("_years_exp") + 1.0).alias("season_number"),
            (pl.col("_years_exp") == 0.0).cast(pl.Float64).alias("is_rookie"),
        )
        .select(list(schema))
        .unique(subset=["season", "week", "team", "player_id"], keep="first")
    )


#: ``(unit, [stats adjusted against it])`` — which opponent unit each stat faces.
_ADJUSTMENT_PLAN: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "pass_defense",
        (
            "epa_per_dropback",
            "passing_yards",
            "rec_epa_per_play",
            "receiving_yards",
            "yac_per_reception",
        ),
    ),
    ("rush_defense", ("rush_epa_per_play", "rushing_yards")),
    ("pass_offense", ("pressures", "tackles", "pass_defensed", "interceptions_made")),
)

#: Counting stats carried through the raw season-to-date / trailing-8 windows.
_WINDOW_COLUMNS: tuple[str, ...] = (
    "pass_dropbacks",
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions_thrown",
    "sacks_taken",
    "pass_epa",
    "cpoe_sum",
    "cpoe_n",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "rush_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rec_epa",
    "yac",
)

#: Composite each position group is ranked on, league-wide, as of the week.
_RANK_COMPOSITE: dict[str, str] = {
    "QB": "adj_epa_per_dropback",
    "RB": "adj_rush_epa_per_play",
    "WR": "adj_rec_epa_per_play",
    "TE": "adj_rec_epa_per_play",
    "DL": "adj_pressures",
    "LB": "adj_tackles",
    "CB": "adj_pass_defensed",
    "S": "adj_interceptions_made",
}


def _position_ranks(players: pl.DataFrame) -> pl.DataFrame:
    """League-wide rank of each starter within their position group, as of week."""
    composite = pl.lit(None, dtype=pl.Float64)
    for group, column in _RANK_COMPOSITE.items():
        if column in players.columns:
            composite = (
                pl.when(pl.col("position_group") == group).then(pl.col(column)).otherwise(composite)
            )
    group_keys = ["season", "week", "position_group"]
    with_composite = players.with_columns(composite.alias("_composite"))
    rank = pl.col("_composite").rank(method="min", descending=True).over(group_keys)
    n = pl.col("_composite").count().over(group_keys)
    return with_composite.with_columns(
        rank.cast(pl.Float64).alias("position_rank"),
        pl.when(n > 1)
        .then(1.0 - (rank.cast(pl.Float64) - 1.0) / (n.cast(pl.Float64) - 1.0))
        .otherwise(None)
        .alias("position_percentile"),
    ).drop("_composite")


def _slot_block(players: pl.DataFrame, slot: str, columns: Mapping[str, str]) -> pl.DataFrame:
    return (
        players.filter(pl.col("slot") == slot)
        .select(
            "season",
            "week",
            "team",
            *[pl.col(src).alias(dst) for src, dst in columns.items()],
        )
        .unique(subset=["season", "week", "team"], keep="first", maintain_order=True)
    )


def _group_block(players: pl.DataFrame, group: str, aggs: Sequence[pl.Expr]) -> pl.DataFrame:
    return (
        players.filter(pl.col("position_group") == group)
        .group_by(["season", "week", "team"])
        .agg(list(aggs))
    )


def build_player(
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    *,
    rosters: pl.DataFrame | None = None,
    depth_charts: pl.DataFrame | None = None,
    snap_counts: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    game_ids: Sequence[str] | None = None,
    k: float = DEFAULT_K,
    recency_lambda: float = DEFAULT_LAMBDA,
    shrinkage_games: float = DEFAULT_SHRINKAGE_GAMES,
    trailing_games: int = TRAILING_GAMES,
    use_rank: bool = False,
) -> pl.DataFrame:
    """Roster-quality features, one row per ``(game_id, team)``.

    ``pbp`` and ``schedules`` are required; every other input is optional and
    degrades gracefully — with pbp alone the starters fall back to prior-week
    usage, the bio and availability blocks come back null, and the column set is
    unchanged. Only games whose season appears in ``pbp`` are emitted.
    """
    games = team_game_frame(schedules)
    if game_ids is not None:
        games = games.filter(pl.col("game_id").is_in(list(game_ids)))
    seasons = pbp.select(pl.col("season").cast(pl.Int64)).unique().to_series().to_list()
    games = games.filter(pl.col("season").is_in(seasons))
    if games.height == 0:
        return _empty_output()

    stats = player_game_rates(player_game_stats(pbp))
    strength = unit_strength(pbp)
    team_weeks = games.select("season", "week", "team").unique()
    starters = identify_starters(
        team_weeks,
        depth_charts=depth_charts,
        snap_counts=snap_counts,
        rosters=rosters,
        stats=stats,
    )
    if starters.height == 0:
        return _finalise(
            games.select("game_id", "team", "prior_games_in_season", "week").rename(
                {"week": f"{PREFIX}week"}
            )
        )

    targets = starters.select("season", "week", "team", "player_id", "position_group")

    players = targets
    for unit, stat_columns in _ADJUSTMENT_PLAN:
        adjusted = opponent_adjust(
            stats,
            strength,
            stat_columns=list(stat_columns),
            targets=targets,
            entity_keys=["player_id"],
            unit=unit,
            group_column="position_group",
            use_rank=use_rank,
            k=k,
            recency_lambda=recency_lambda,
            shrinkage_games=shrinkage_games,
            sample_column=f"games_{unit}",
            factor_column=f"factor_{unit}",
        ).select(
            "season",
            "week",
            "team",
            "player_id",
            *[f"adj_{c}" for c in stat_columns],
            f"games_{unit}",
        )
        players = players.join(adjusted, on=["season", "week", "team", "player_id"], how="left")

    window_columns = [c for c in _WINDOW_COLUMNS if c in stats.columns]
    ytd = as_of_aggregate(stats, targets, value_columns=window_columns, suffix="_ytd").select(
        "season", "week", "team", "player_id", *[f"{c}_ytd" for c in window_columns], "games_ytd"
    )
    last8 = as_of_aggregate(
        stats, targets, value_columns=window_columns, trailing_games=trailing_games, suffix="_l8"
    ).select(
        "season", "week", "team", "player_id", *[f"{c}_l8" for c in window_columns], "games_l8"
    )
    players = players.join(ytd, on=["season", "week", "team", "player_id"], how="left").join(
        last8, on=["season", "week", "team", "player_id"], how="left"
    )
    players = players.with_columns(
        _safe_div(pl.col("pass_epa_ytd"), pl.col("pass_dropbacks_ytd")).alias(
            "epa_per_dropback_ytd"
        ),
        _safe_div(pl.col("pass_epa_l8"), pl.col("pass_dropbacks_l8")).alias(
            "epa_per_dropback_l8_rate"
        ),
        _safe_div(pl.col("completions_ytd"), pl.col("pass_attempts_ytd")).alias(
            "completion_pct_ytd"
        ),
        _safe_div(pl.col("cpoe_sum_ytd"), pl.col("cpoe_n_ytd")).alias("cpoe_ytd"),
        _safe_div(pl.col("sacks_taken_ytd"), pl.col("pass_dropbacks_ytd")).alias("sack_rate_ytd"),
        _safe_div(pl.col("passing_yards_ytd"), pl.col("games_ytd")).alias("passing_yards_pg"),
        _safe_div(pl.col("passing_yards_l8"), pl.col("games_l8")).alias("passing_yards_pg_l8"),
        _safe_div(pl.col("passing_tds_ytd"), pl.col("games_ytd")).alias("passing_tds_pg"),
        _safe_div(pl.col("interceptions_thrown_ytd"), pl.col("games_ytd")).alias(
            "interceptions_pg"
        ),
        _safe_div(pl.col("rushing_yards_ytd"), pl.col("games_ytd")).alias("rushing_yards_pg"),
        _safe_div(pl.col("rushing_yards_l8"), pl.col("games_l8")).alias("rushing_yards_pg_l8"),
        _safe_div(pl.col("rushing_tds_ytd"), pl.col("games_ytd")).alias("rushing_tds_pg"),
        _safe_div(pl.col("receiving_yards_ytd"), pl.col("games_ytd")).alias("receiving_yards_pg"),
    )
    players = players.join(
        starters.select("season", "week", "team", "player_id", "position", "slot", "slot_index"),
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    players = _position_ranks(players)
    players = players.join(
        _bio_frame(rosters, games), on=["season", "week", "team", "player_id"], how="left"
    )

    availability = (
        player_availability(injuries)
        if injuries is not None and injuries.height
        else pl.DataFrame(schema=_AVAILABILITY_SCHEMA)
    )
    if availability.height:
        players = players.join(
            availability.select("season", "week", "team", "player_id", "availability_prob"),
            on=["season", "week", "team", "player_id"],
            how="left",
        ).with_columns(pl.col("availability_prob").fill_null(1.0))
    else:
        players = players.with_columns(pl.lit(None, dtype=pl.Float64).alias("availability_prob"))
    players = players.join(
        _prior_snap_share(snap_counts, rosters),
        on=["season", "week", "team", "player_id"],
        how="left",
    ).join(
        _position_weight_frame().with_columns(pl.col("position").str.to_uppercase()),
        left_on=pl.col("position").cast(pl.Utf8).str.to_uppercase(),
        right_on="position",
        how="left",
    )

    out = games.select("game_id", "season", "week", "team", "prior_games_in_season")
    out = out.join(
        _slot_block(players, "QB1", _QB1_COLUMNS), on=["season", "week", "team"], how="left"
    )
    out = out.join(
        _slot_block(players, "RB1", _RB1_COLUMNS), on=["season", "week", "team"], how="left"
    )
    out = out.join(
        _slot_block(players, "WR1", _WR1_COLUMNS), on=["season", "week", "team"], how="left"
    )
    out = out.join(
        _slot_block(players, "TE1", _TE1_COLUMNS), on=["season", "week", "team"], how="left"
    )
    out = out.join(
        _slot_block(players, "DL1", {"position_rank": "dl1_position_rank"}),
        on=["season", "week", "team"],
        how="left",
    )
    out = out.join(
        _slot_block(players, "LB1", {"position_rank": "lb1_position_rank"}),
        on=["season", "week", "team"],
        how="left",
    )
    out = out.join(
        _slot_block(players, "CB1", {"position_rank": "cb1_position_rank"}),
        on=["season", "week", "team"],
        how="left",
    )
    out = out.join(
        _slot_block(players, "S1", {"position_rank": "s1_position_rank"}),
        on=["season", "week", "team"],
        how="left",
    )

    out = out.join(
        _group_block(
            players,
            "WR",
            [
                pl.col("adj_rec_epa_per_play").mean().alias("wr_corps_adj_epa_per_target"),
                pl.col("receiving_yards_pg").sum().alias("wr_corps_receiving_yards_per_game"),
                pl.col("adj_yac_per_reception").mean().alias("wr_corps_yac_per_reception"),
            ],
        ),
        on=["season", "week", "team"],
        how="left",
    )
    for group, expr, alias in (
        ("DL", pl.col("adj_pressures").mean(), "dl_adj_pressures_per_game"),
        ("LB", pl.col("adj_tackles").mean(), "lb_adj_tackles_per_game"),
        ("CB", pl.col("adj_pass_defensed").mean(), "cb_adj_pass_defensed_per_game"),
        ("S", pl.col("adj_interceptions_made").mean(), "s_adj_interceptions_per_game"),
    ):
        out = out.join(
            _group_block(players, group, [expr.alias(alias)]),
            on=["season", "week", "team"],
            how="left",
        )
    out = out.join(
        _group_block(
            players,
            "OL",
            [
                (pl.col("availability_prob") >= 0.5)
                .cast(pl.Float64)
                .sum()
                .alias("ol_starters_available")
            ],
        ),
        on=["season", "week", "team"],
        how="left",
    )

    skill = players.filter(pl.col("slot").is_in(["RB1", "WR1", "WR2", "WR3", "TE1"]))
    out = out.join(
        skill.group_by(["season", "week", "team"]).agg(
            pl.sum_horizontal(
                pl.col("adj_rush_epa_per_play").fill_null(0.0),
                pl.col("adj_rec_epa_per_play").fill_null(0.0),
            )
            .sum()
            .alias("skill_adj_epa_sum"),
            pl.sum_horizontal(
                pl.col("rushing_yards_pg").fill_null(0.0),
                pl.col("receiving_yards_pg").fill_null(0.0),
            )
            .sum()
            .alias("skill_yards_per_game_sum"),
        ),
        on=["season", "week", "team"],
        how="left",
    )

    roster_shape = players.group_by(["season", "week", "team"]).agg(
        pl.len().cast(pl.Float64).alias("starters_identified"),
        pl.col("age_years").mean().alias("starters_mean_age"),
        pl.col("season_number").mean().alias("starters_mean_experience"),
        pl.col("is_rookie").sum().alias("starters_n_rookies"),
        (pl.col("availability_prob") < 0.5).cast(pl.Float64).sum().alias("starters_out_count"),
        (
            (1.0 - pl.col("availability_prob").fill_null(1.0))
            * pl.col("impact_weight").fill_null(0.10)
            * pl.col("replacement_dropoff").fill_null(0.45)
        )
        .sum()
        .alias("starters_expected_value_lost"),
        pl.when(pl.col("availability_prob") < 0.5)
        .then(pl.col("snap_share_prior"))
        .otherwise(0.0)
        .sum()
        .alias("snap_share_lost"),
    )
    out = out.join(roster_shape, on=["season", "week", "team"], how="left")
    out = out.join(team_sack_rates(pbp), on=["season", "week", "team"], how="left")
    out = out.join(team_return_game(stats), on=["season", "week", "team"], how="left")
    # An empty availability frame still carries the schema, so the left join
    # adds the block as nulls when there is no injury feed at all.
    out = out.join(
        team_availability(team_weeks, injuries), on=["season", "week", "team"], how="left"
    )
    return _finalise(out.rename({"week": f"{PREFIX}week"}, strict=False))


#: ``player frame column -> output name (minus the pl__ prefix)`` per slot.
_QB1_COLUMNS: dict[str, str] = {
    "player_id": "qb1_player_id",
    "adj_epa_per_dropback": "qb1_adj_epa_per_dropback",
    "epa_per_dropback_ytd": "qb1_epa_per_dropback",
    "epa_per_dropback_l8_rate": "qb1_epa_per_dropback_l8",
    "passing_yards_pg": "qb1_passing_yards_per_game",
    "passing_yards_pg_l8": "qb1_passing_yards_per_game_l8",
    "adj_passing_yards": "qb1_adj_passing_yards",
    "passing_tds_pg": "qb1_passing_tds_per_game",
    "interceptions_pg": "qb1_interceptions_per_game",
    "sack_rate_ytd": "qb1_sack_rate_taken",
    "completion_pct_ytd": "qb1_completion_pct",
    "cpoe_ytd": "qb1_cpoe",
    "rushing_yards_pg": "qb1_rushing_yards_per_game",
    "games_ytd": "qb1_games_in_sample",
    "position_rank": "qb1_position_rank",
    "position_percentile": "qb1_position_percentile",
    "age_years": "qb1_age_years",
    "season_number": "qb1_season_number",
    "draft_round": "qb1_draft_round",
    "draft_pick": "qb1_draft_pick",
    "is_rookie": "qb1_is_rookie",
    "availability_prob": "qb1_availability_prob",
}

_RB1_COLUMNS: dict[str, str] = {
    "adj_rush_epa_per_play": "rb1_adj_epa_per_rush",
    "adj_rushing_yards": "rb1_adj_rushing_yards",
    "rushing_yards_pg": "rb1_rushing_yards_per_game",
    "rushing_yards_pg_l8": "rb1_rushing_yards_per_game_l8",
    "rushing_tds_pg": "rb1_rushing_tds_per_game",
    "position_rank": "rb1_position_rank",
    "games_ytd": "rb1_games_in_sample",
}

_WR1_COLUMNS: dict[str, str] = {
    "adj_rec_epa_per_play": "wr1_adj_epa_per_target",
    "receiving_yards_pg": "wr1_receiving_yards_per_game",
    "adj_receiving_yards": "wr1_adj_receiving_yards",
    "position_rank": "wr1_position_rank",
}

_TE1_COLUMNS: dict[str, str] = {
    "adj_rec_epa_per_play": "te1_adj_epa_per_target",
    "receiving_yards_pg": "te1_receiving_yards_per_game",
    "position_rank": "te1_position_rank",
}

#: Team-level columns that already carry their final (unprefixed) name.
_TEAM_PASSTHROUGH: tuple[str, ...] = (
    "prior_games_in_season",
    "starters_identified",
    "starters_mean_age",
    "starters_mean_experience",
    "starters_n_rookies",
    "starters_out_count",
    "starters_expected_value_lost",
    "snap_share_lost",
    "wr_corps_adj_epa_per_target",
    "wr_corps_receiving_yards_per_game",
    "wr_corps_yac_per_reception",
    "dl_adj_pressures_per_game",
    "dl1_position_rank",
    "lb_adj_tackles_per_game",
    "lb1_position_rank",
    "cb_adj_pass_defensed_per_game",
    "cb1_position_rank",
    "s_adj_interceptions_per_game",
    "s1_position_rank",
    "ol_starters_available",
    "skill_adj_epa_sum",
    "skill_yards_per_game_sum",
    "injury_expected_value_lost",
    "injury_qb_absence_cost",
    "injury_score",
    "injury_n_out",
    "injury_n_doubtful",
    "injury_n_questionable",
    "injury_n_items",
)

#: Team-level columns whose builder name differs from the feature name.
_TEAM_RENAMES: dict[str, str] = {
    "sack_rate_allowed": "ol_sack_rate_allowed",
    "sack_rate_allowed_rank": "ol_sack_rate_allowed_rank",
    "sack_rate_generated": "dl_sack_rate_generated",
    "sack_rate_generated_rank": "dl_sack_rate_generated_rank",
    "returner_player_id": "st_returner_player_id",
    "kick_return_yards_per_return": "st_kick_return_yards_per_return",
    "punt_return_yards_per_return": "st_punt_return_yards_per_return",
    "returns_in_sample": "st_returns_in_sample",
}


def _empty_output() -> pl.DataFrame:
    schema: dict[str, Any] = {"game_id": pl.Utf8, "team": pl.Utf8}
    schema.update(FEATURE_DTYPES)
    return pl.DataFrame(schema=schema)


def _finalise(frame: pl.DataFrame) -> pl.DataFrame:
    """Rename to ``pl__``, fill in any column the inputs could not produce, guard ADR 0001."""
    renames = {
        **{name: f"{PREFIX}{name}" for name in _TEAM_PASSTHROUGH},
        **{src: f"{PREFIX}{dst}" for src, dst in _TEAM_RENAMES.items()},
    }
    for mapping in (_QB1_COLUMNS, _RB1_COLUMNS, _WR1_COLUMNS, _TE1_COLUMNS):
        for dst in mapping.values():
            renames.setdefault(dst, f"{PREFIX}{dst}")
    frame = frame.rename({k: v for k, v in renames.items() if k in frame.columns})
    # Derived once, from the pre-kickoff report. The live path overrides it
    # before calling back in here, so an existing value always wins.
    if (
        f"{PREFIX}qb1_availability_prob" in frame.columns
        and f"{PREFIX}qb1_available" not in frame.columns
    ):
        frame = frame.with_columns(
            (pl.col(f"{PREFIX}qb1_availability_prob") >= 0.5)
            .cast(pl.Float64)
            .alias(f"{PREFIX}qb1_available")
        )
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(f"{PREFIX}qb1_contract_years_remaining")
    )
    missing = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if missing:
        frame = frame.with_columns(
            [pl.lit(None, dtype=FEATURE_DTYPES.get(c, pl.Float64)).alias(c) for c in missing]
        )
    out = frame.select(
        [
            pl.col(c).cast(FEATURE_DTYPES[c]) if c in FEATURE_DTYPES else pl.col(c)
            for c in OUTPUT_COLUMNS
        ]
    ).sort(["game_id", "team"])
    assert_no_market_columns(out, context="nfl.features.player.build_player")
    return out


# --------------------------------------------------------------------------- #
# Live path
# --------------------------------------------------------------------------- #


def roster_from_snapshot(
    snapshot: LiveSnapshot,
    pregame: pl.DataFrame | None = None,
    *,
    game_id: str | None = None,
) -> pl.DataFrame:
    """The ``build_player`` columns for a live game, refreshed from the snapshot.

    The pregame table is the base — nothing in a live feed can recompute a
    season's opponent-adjusted EPA — and the snapshot's in-game injury list
    updates the availability block on top of it.

    ESPN publishes a *current status* list that already contains the pre-kickoff
    report, so the two costs are combined with ``max`` rather than summed;
    double counting a player who appears in both would be worse than missing an
    increment. ``pl__qb1_available`` drops to 0 when a QB on the list is Out (or
    when the snapshot names the pregame QB1 explicitly).
    """
    resolved = game_id or snapshot.game_id
    if resolved is None:
        week = snapshot.week if snapshot.week is not None else 0
        resolved = f"{snapshot.season}_{week:02d}_{snapshot.away_team}_{snapshot.home_team}"
    teams = [snapshot.home_team, snapshot.away_team]

    if pregame is not None and pregame.height:
        base = pregame.filter(pl.col("game_id") == resolved)
        if base.height == 0:
            base = _blank_rows(resolved, teams)
    else:
        base = _blank_rows(resolved, teams)
    base = _finalise(base)

    live = [
        InjuryItem(
            player_id=injury.player_id or "",
            team=injury.team,
            position=injury.position,
            status=injury.status,
        )
        for injury in snapshot.injuries
        if injury.team in teams
    ]
    if not live:
        return base

    updates: list[dict[str, Any]] = []
    for team in teams:
        items = [item for item in live if item.team == team]
        if not items:
            continue
        summary = _summary_row(team, items)
        qb_out = any(
            (item.position or "").strip().upper() == "QB"
            and canonical_status(item.status) in HARD_OUT_STATUSES
            for item in items
        )
        updates.append({"team": team, "_qb_out": float(not qb_out), **summary})
    if not updates:
        return base

    schema: dict[str, Any] = {"team": pl.Utf8, "_qb_out": pl.Float64}
    schema.update(
        {c: pl.Float64 for c in _TEAM_INJURY_SCHEMA if c not in {"season", "week", "team"}}
    )
    frame = pl.DataFrame(updates, schema=schema).rename(
        {c: f"_live_{c}" for c in schema if c not in {"team", "_qb_out"}}
    )
    merged = base.join(frame, on="team", how="left")
    combined = []
    for column in ("injury_expected_value_lost", "injury_qb_absence_cost"):
        name = f"{PREFIX}{column}"
        combined.append(pl.max_horizontal(pl.col(name), pl.col(f"_live_{column}")).alias(name))
    for column in ("injury_n_out", "injury_n_doubtful", "injury_n_questionable", "injury_n_items"):
        name = f"{PREFIX}{column}"
        combined.append(pl.max_horizontal(pl.col(name), pl.col(f"_live_{column}")).alias(name))
    combined.append(
        pl.min_horizontal(pl.col(f"{PREFIX}injury_score"), pl.col("_live_injury_score")).alias(
            f"{PREFIX}injury_score"
        )
    )
    combined.append(
        pl.min_horizontal(
            pl.col(f"{PREFIX}qb1_available").fill_null(1.0), pl.col("_qb_out").fill_null(1.0)
        ).alias(f"{PREFIX}qb1_available")
    )
    return _finalise(merged.with_columns(combined).select(OUTPUT_COLUMNS))


def _blank_rows(game_id: str, teams: Sequence[str]) -> pl.DataFrame:
    frame = pl.DataFrame({"game_id": [game_id] * len(teams), "team": list(teams)})
    return frame.with_columns(
        [pl.lit(None, dtype=dtype).alias(name) for name, dtype in FEATURE_DTYPES.items()]
    )


# --------------------------------------------------------------------------- #
# Raw-data loader
# --------------------------------------------------------------------------- #

#: nflverse tables :func:`build_player` can consume, and the kwarg each fills.
INPUT_TABLES: dict[str, str] = {
    "pbp": "pbp",
    "schedules": "schedules",
    "weekly_rosters": "rosters",
    "depth_charts": "depth_charts",
    "snap_counts": "snap_counts",
    "injuries": "injuries",
}


def load_player_inputs(
    seasons: Sequence[int],
    *,
    root: Path | None = None,
    columns: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, pl.DataFrame]:
    """Read the nflverse parquet partitions ``build_player`` wants, if present.

    Missing tables are simply absent from the result; ``build_player`` handles
    that. Returns kwargs-shaped keys (``pbp``, ``schedules``, ``rosters``, ...).
    """
    from nfl.config import RAW_DIR

    base = Path(root) if root is not None else RAW_DIR / "nflverse"
    out: dict[str, pl.DataFrame] = {}
    for table, kwarg in INPUT_TABLES.items():
        paths = [p for season in seasons if (p := base / table / f"season={season}").is_dir()]
        files = [f for path in paths for f in sorted(path.glob("*.parquet"))]
        if not files:
            continue
        wanted = list(columns[table]) if columns and table in columns else None
        frames = [pl.read_parquet(f, columns=wanted) for f in files]
        out[kwarg] = pl.concat(frames, how="diagonal_relaxed")
    return out
