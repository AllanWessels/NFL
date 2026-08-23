"""In-game state features (``ig__``) — DATA_MODEL §2.1–2.8 and §2.11.

One row per ``(game_id, play_id)`` carrying the complete cumulative in-game
state **as of the moment immediately before that play is run**.

Two entry points produce the *same* columns:

``build_ingame(pbp)``
    Vectorised over an nflverse play-by-play frame. Every cumulative quantity is
    a ``cum_sum().shift(1)`` window inside ``game_id``, so a row for play *t*
    sees plays ``< t`` and nothing else. The pre-snap situation of play *t*
    itself (down, distance, yardline, timeouts, clock) is fair game — it is on
    the field before the snap.

``snapshot_to_row(snapshot)``
    The serving path. A :class:`~nfl.data.live.base.LiveSnapshot` is reshaped
    into a play-by-play-like frame plus one synthetic row for the snap that is
    *about* to happen, and pushed through the exact same expressions. Train and
    serve therefore cannot drift: there is one implementation of every feature.

Orientation
-----------
Everything is **home-perspective**. ``ig__score_diff_home`` is positive when the
home team leads; per-team quantities exist as ``*_home`` / ``*_away`` pairs;
``*_posteam`` / ``*_defteam`` views are selected from those pairs by
``ig__posteam_is_home``. Swapping the home and away teams in the input negates
every signed feature and swaps every pair — ``tests/test_ingame.py`` asserts it.

Leakage notes
-------------
* ``total_home_score`` / ``total_away_score`` in nflverse are **post-play**.
  They are lagged one play to recover the pre-snap score. This is the single
  most common leak in in-game modelling and it is handled in exactly one place
  (:func:`_with_scoring`).
* Drive-level nflverse columns (``fixed_drive_result``, ``ydsnet``,
  ``drive_play_count``) are replicated onto *every* play of a drive and so
  describe that drive's future. None of them are read. Drive outcomes, yardage
  and points are recomputed from the plays themselves and attributed to the
  drive's **last** play, which only ever feeds rows after the drive is over.
* nflverse EPA is a season-wide model output. DATA_MODEL §2.5 accepts it for
  in-game aggregation; it is summed as-is and never recomputed.
* ``wp`` / ``wpa`` / ``vegas_*`` are third-party model outputs and market
  derivatives. They are never read, and :func:`assert_no_market_columns` guards
  the returned frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from nfl.config import RAW_DIR
from nfl.data.market_blocklist import assert_no_market_columns

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nfl.data.live.base import LiveSnapshot

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_DOCS",
    "KEY_COLUMNS",
    "PREFIX",
    "SNAPSHOT_CLOCK_SENSITIVE_COLUMNS",
    "SNAPSHOT_UNKNOWABLE_COLUMNS",
    "build_ingame",
    "load_pbp_frame",
    "snapshot_history_rows",
    "snapshot_play_frame",
    "snapshot_to_row",
]

PREFIX = "ig__"
KEY_COLUMNS = ("game_id", "play_id")

_REGULATION_PERIODS = 4
_PERIOD_SECONDS = 900
_SIDES = ("home", "away")

#: Runs of this many yards or more, and passes of this many or more, are
#: "explosive" (DATA_MODEL §2.5).
_EXPLOSIVE_RUSH = 12
_EXPLOSIVE_PASS = 20

#: Pre-snap penalty types — a discipline proxy that does not depend on the play.
_PRE_SNAP_PENALTIES = (
    "False Start",
    "Delay of Game",
    "Encroachment",
    "Neutral Zone Infraction",
    "Offside on Free Kick",
    "Defensive Offside",
    "Illegal Formation",
    "Illegal Shift",
    "Illegal Motion",
    "Illegal Substitution",
    "Defensive Too Many Men on Field",
    "Offensive Too Many Men on Field",
)


# --------------------------------------------------------------------------- #
# Small expression helpers
# --------------------------------------------------------------------------- #


def _safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """``numerator / denominator``, or null when the denominator is 0/null.

    Nulls are GBDT-friendly (CLAUDE.md conventions); a smoothed 0/0 would be a
    lie about a team that has not run a third down yet.
    """
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _lag_cum(name: str) -> pl.Expr:
    """Sum of ``name`` over plays strictly before this one, within the game."""
    return pl.col(name).cum_sum().shift(1).fill_null(0.0).over("game_id")


def _window_sum(name: str, plays: int) -> pl.Expr:
    """Sum of ``name`` over the last ``plays`` plays, ending strictly before *t*."""
    running = pl.col(name).cum_sum()
    return (running.shift(1).fill_null(0.0) - running.shift(1 + plays).fill_null(0.0)).over(
        "game_id"
    )


def _pick(home: str, away: str) -> pl.Expr:
    """Select the home or away column according to who has the ball."""
    return (
        pl.when(pl.col(f"{PREFIX}posteam_is_home"))
        .then(pl.col(home))
        .when(pl.col(f"{PREFIX}posteam_is_home").not_())
        .then(pl.col(away))
        .otherwise(None)
    )


class _Source:
    """Column accessor that tolerates a trimmed play-by-play frame.

    The committed fixtures and the ESPN-shaped serving frame carry a subset of
    nflverse's ~372 columns. A missing column yields a null/false literal rather
    than an exception, so one code path serves both.
    """

    def __init__(self, columns: Iterable[str]) -> None:
        self.columns = set(columns)

    def has(self, name: str) -> bool:
        return name in self.columns

    def num(self, name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Float64) if name in self.columns else pl.lit(None, pl.Float64)

    def flag(self, name: str) -> pl.Expr:
        """A 1/0 or boolean nflverse indicator as a non-null boolean."""
        if name not in self.columns:
            return pl.lit(value=False, dtype=pl.Boolean)
        return (pl.col(name).cast(pl.Float64) == 1).fill_null(value=False)

    def maybe_flag(self, name: str) -> pl.Expr:
        """Like :meth:`flag` but null (not false) when the column is absent."""
        if name not in self.columns:
            return pl.lit(None, pl.Boolean)
        return (pl.col(name).cast(pl.Float64) == 1).fill_null(value=False)

    def text(self, name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name in self.columns else pl.lit(None, pl.Utf8)


# --------------------------------------------------------------------------- #
# Stage 1 — normalise the raw play-by-play into per-play primitives
# --------------------------------------------------------------------------- #


def _prepare(pbp: pl.DataFrame) -> pl.DataFrame:
    """Sort by game time and reduce the frame to the primitives we derive from."""
    src = _Source(pbp.columns)
    order = "order_sequence" if src.has("order_sequence") else "play_id"

    frame = pbp.select(
        pl.col("game_id").cast(pl.Utf8),
        pl.col("play_id").cast(pl.Float64).alias("play_id"),
        pl.col(order).cast(pl.Float64).alias("_order"),
        pl.col("home_team").cast(pl.Utf8).alias("home_team"),
        pl.col("away_team").cast(pl.Utf8).alias("away_team"),
        src.text("posteam").alias("posteam"),
        src.text("defteam").alias("defteam"),
        pl.col("qtr").cast(pl.Int32).alias("qtr"),
        pl.col("quarter_seconds_remaining").cast(pl.Float64).alias("qsr"),
        src.num("down").alias("down"),
        src.num("ydstogo").alias("ydstogo"),
        src.num("yardline_100").alias("yardline_100"),
        src.num("total_home_score").alias("raw_home_score"),
        src.num("total_away_score").alias("raw_away_score"),
        src.num("home_timeouts_remaining").alias("home_to"),
        src.num("away_timeouts_remaining").alias("away_to"),
        src.text("play_type").alias("play_type"),
        src.num("yards_gained").alias("yards_gained"),
        src.num("epa").alias("epa"),
        src.maybe_flag("success").alias("raw_success"),
        src.flag("complete_pass").alias("complete_pass"),
        src.flag("first_down").alias("first_down"),
        src.flag("third_down_converted").alias("third_down_converted"),
        src.flag("fourth_down_converted").alias("fourth_down_converted"),
        src.flag("sack").alias("sack"),
        src.flag("qb_hit").alias("qb_hit"),
        src.flag("tackled_for_loss").alias("tackled_for_loss"),
        src.flag("interception").alias("interception"),
        src.flag("fumble").alias("fumble"),
        src.flag("fumble_lost").alias("fumble_lost"),
        src.maybe_flag("is_turnover").alias("feed_turnover"),
        src.flag("touchdown").alias("touchdown"),
        src.text("td_team").alias("td_team"),
        src.flag("return_touchdown").alias("return_touchdown"),
        src.flag("penalty").alias("penalty"),
        src.text("penalty_team").alias("penalty_team"),
        src.num("penalty_yards").alias("penalty_yards"),
        src.text("penalty_type").alias("penalty_type"),
        src.flag("punt_attempt").alias("punt_attempt"),
        src.flag("field_goal_attempt").alias("field_goal_attempt"),
        src.text("field_goal_result").alias("field_goal_result"),
        src.num("kick_distance").alias("kick_distance"),
        src.flag("extra_point_attempt").alias("extra_point_attempt"),
        src.text("extra_point_result").alias("extra_point_result"),
        src.flag("two_point_attempt").alias("two_point_attempt"),
        src.text("two_point_conv_result").alias("two_point_conv_result"),
        src.flag("kickoff_attempt").alias("kickoff_attempt"),
        src.num("return_yards").alias("return_yards"),
        src.text("return_team").alias("return_team"),
        pl.coalesce(src.num("fixed_drive"), src.num("drive")).alias("drive_key"),
    ).sort(["game_id", "_order"])

    # "Scrimmage play" is defined from ``play_type`` alone rather than nflverse's
    # ``pass``/``rush`` indicators. Those indicators also fire on penalty-nullified
    # snaps and two-point plays, and they have no counterpart in a LiveSnapshot —
    # defining the split on ``play_type`` is what keeps the two paths identical.
    is_scrimmage = pl.col("play_type").is_in(["pass", "run"]) & pl.col("posteam").is_not_null()

    return frame.with_columns(
        pl.col("posteam").eq(pl.col("home_team")).alias("pos_is_home"),
        is_scrimmage.alias("is_scrimmage"),
        # ``goal_to_go`` is derived rather than read: the nflverse column is not
        # present in every frame we serve, and the derivation is exact.
        (pl.col("ydstogo") >= pl.col("yardline_100")).alias("goal_to_go"),
    )


def _with_clock(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.1 — clock, elapsed time and the two-minute flags."""
    qtr, qsr = pl.col("qtr"), pl.col("qsr")
    half = pl.when(qtr.is_in([1, 3])).then(qsr + _PERIOD_SECONDS).otherwise(qsr)
    game = (
        pl.when(qtr <= _REGULATION_PERIODS)
        .then(qsr + _PERIOD_SECONDS * (_REGULATION_PERIODS - qtr))
        .otherwise(qsr)
    )
    # Seconds burned since the previous play. Computed from the quarter clock so
    # that overtime periods of any length behave; a quarter change contributes
    # whatever was left on the previous quarter's clock.
    delta = (
        pl.when(pl.col("qtr") == pl.col("qtr").shift(1))
        .then(pl.col("qsr").shift(1) - pl.col("qsr"))
        .otherwise(pl.col("qsr").shift(1))
        .fill_null(0.0)
        .clip(0.0, float(_PERIOD_SECONDS))
    )

    frame = frame.with_columns(
        half.alias("hsr"),
        game.alias("gsr"),
        (qtr > _REGULATION_PERIODS).alias("is_ot"),
    )
    return frame.with_columns(delta.over("game_id").alias("dt")).with_columns(
        pl.col("dt").cum_sum().over("game_id").alias("elapsed"),
    )


def _with_scoring(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.2 — pre-snap score, and the per-play points that everything else uses.

    ``total_home_score`` is post-play, so the pre-snap score is its one-play lag.
    """
    home_pre = pl.col("raw_home_score").shift(1).fill_null(0.0).over("game_id")
    away_pre = pl.col("raw_away_score").shift(1).fill_null(0.0).over("game_id")
    frame = frame.with_columns(home_pre.alias("home_pre"), away_pre.alias("away_pre"))
    return frame.with_columns(
        (pl.col("raw_home_score") - pl.col("home_pre")).clip(0.0, None).alias("pts_home"),
        (pl.col("raw_away_score") - pl.col("away_pre")).clip(0.0, None).alias("pts_away"),
        (pl.col("home_pre") - pl.col("away_pre")).alias("diff_pre"),
    )


def _with_drives(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.3/§2.7 — drive identity, within-drive progress and completed-drive events.

    Drives are run-length groups of ``fixed_drive`` numbered from 1, matching
    :class:`~nfl.data.live.replay.HistoricalReplayFeed`'s grouping exactly so
    that the serving path lands on the same drive numbers.
    """
    key = pl.col("drive_key")
    drive_id = key.ne_missing(key.shift(1)).cum_sum().over("game_id").cast(pl.Int32)
    frame = frame.with_columns(drive_id.alias("drive_id"))

    grp = ["game_id", "drive_id"]
    # The first *scrimmage* snap of the drive fixes its starting field position.
    # Held back until that snap happens so the value is never read before it is
    # known.
    scrim_yardline = pl.when(pl.col("down").is_not_null()).then(pl.col("yardline_100"))
    is_first_scrim = scrim_yardline.is_not_null() & (
        scrim_yardline.is_not_null().cum_sum().over(grp) == 1
    )
    frame = frame.with_columns(
        pl.int_range(pl.len()).over(grp).cast(pl.Int32).alias("drive_play_index"),
        pl.when(is_first_scrim)
        .then(scrim_yardline)
        .otherwise(None)
        .forward_fill()
        .over(grp)
        .alias("drive_start_yardline"),
        pl.col("elapsed").first().over(grp).alias("drive_start_elapsed"),
        pl.when(pl.col("is_scrimmage"))
        .then(pl.col("yards_gained"))
        .otherwise(0.0)
        .fill_null(0.0)
        .cum_sum()
        .shift(1)
        .fill_null(0.0)
        .over(grp)
        .alias("drive_yards_so_far"),
        pl.col("posteam").drop_nulls().first().over(grp).alias("drive_team"),
    )

    # A drive is "over" on the last row that carries its id. Marking it there and
    # only ever consuming the mark through a lagged cumulative sum means the mark
    # is invisible to the rows inside the drive itself.
    is_drive_end = pl.col("drive_id").ne_missing(pl.col("drive_id").shift(-1)).over("game_id")
    drive_pts = {side: pl.col(f"pts_{side}").sum().over(grp) for side in _SIDES}
    drive_scrim_plays = pl.col("is_scrimmage").sum().over(grp)
    drive_had_punt = pl.col("punt_attempt").max().over(grp)
    drive_had_turnover = (
        (
            pl.col("interception")
            | pl.col("fumble_lost")
            | pl.col("feed_turnover").fill_null(value=False)
        )
        .max()
        .over(grp)
    )

    frame = frame.with_columns(is_drive_end.alias("is_drive_end"))
    for side in _SIDES:
        own = pl.col("drive_team") == pl.col(f"{side}_team")
        pts = drive_pts[side]
        end = pl.col("is_drive_end") & own
        frame = frame.with_columns(
            end.cast(pl.Float64).alias(f"ev_drive_{side}"),
            pl.when(end).then(pts).otherwise(0.0).alias(f"ev_drive_points_{side}"),
            (end & (pts > 0)).cast(pl.Float64).alias(f"ev_scoring_drive_{side}"),
            (end & (pts >= 6)).cast(pl.Float64).alias(f"ev_td_drive_{side}"),
            (end & (pts == 3)).cast(pl.Float64).alias(f"ev_fg_drive_{side}"),
            (end & drive_had_punt).cast(pl.Float64).alias(f"ev_punt_drive_{side}"),
            (end & drive_had_turnover).cast(pl.Float64).alias(f"ev_to_drive_{side}"),
            (end & drive_had_punt & (drive_scrim_plays <= 3))
            .cast(pl.Float64)
            .alias(f"ev_three_and_out_{side}"),
            pl.when(end)
            .then(pl.col("drive_start_yardline"))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias(f"ev_drive_start_{side}"),
            pl.when(end & pl.col("drive_start_yardline").is_not_null())
            .then(1.0)
            .otherwise(0.0)
            .alias(f"ev_drive_start_n_{side}"),
        )
    return frame


# --------------------------------------------------------------------------- #
# Stage 2 — per-play events, attributed to home / away
# --------------------------------------------------------------------------- #


def _event_specs() -> list[tuple[str, pl.Expr, str]]:
    """``(name, value, attribution)`` for every cumulative quantity.

    ``attribution`` is ``"off"`` (credited to the team with the ball),
    ``"def"`` (credited to the defence), ``"td"`` (the scoring team),
    ``"penalty"`` (the penalised team) or ``"return"`` (the returning team).
    """
    scrim = pl.col("is_scrimmage")
    yards = pl.col("yards_gained").fill_null(0.0)
    epa = pl.col("epa")
    is_pass = pl.col("play_type") == "pass"
    is_rush = pl.col("play_type") == "run"
    success = pl.coalesce(pl.col("raw_success"), epa > 0).fill_null(value=False)
    zero = pl.lit(0.0)

    def when(cond: pl.Expr, value: pl.Expr | None = None) -> pl.Expr:
        """1.0 where ``cond`` holds (or ``value`` there), 0.0 everywhere else."""
        return (
            pl.when(cond)
            .then(pl.lit(1.0) if value is None else value)
            .otherwise(zero)
            .fill_null(0.0)
        )

    # Net yards on pass / rush plays, from ``yards_gained`` rather than nflverse's
    # ``passing_yards`` / ``rushing_yards``: sack losses count against the pass
    # game, and a LiveSnapshot has only the play's net yardage to offer.
    pass_yards = pl.when(is_pass & scrim).then(yards).otherwise(None)
    rush_yards = pl.when(is_rush & scrim).then(yards).otherwise(None)
    explosive = scrim & (
        (is_rush & (yards >= _EXPLOSIVE_RUSH)) | (is_pass & (yards >= _EXPLOSIVE_PASS))
    )
    turnover = pl.coalesce(
        pl.when(pl.col("interception") | pl.col("fumble_lost"))
        .then(pl.lit(value=True))
        .otherwise(None),
        pl.col("feed_turnover"),
        pl.lit(value=False),
    )
    fg_made = pl.col("field_goal_attempt") & (pl.col("field_goal_result") == "made")
    fg_missed = pl.col("field_goal_attempt") & (pl.col("field_goal_result") != "made")
    pre_snap = pl.col("penalty") & pl.col("penalty_type").is_in(list(_PRE_SNAP_PENALTIES))

    return [
        ("plays", when(scrim), "off"),
        ("yards", when(scrim, yards), "off"),
        ("pass_attempts", when(scrim & is_pass), "off"),
        ("pass_completions", when(pl.col("complete_pass")), "off"),
        ("pass_yards", when(pass_yards.is_not_null(), pass_yards), "off"),
        ("rush_attempts", when(scrim & is_rush), "off"),
        ("rush_yards", when(rush_yards.is_not_null(), rush_yards), "off"),
        ("first_downs", when(pl.col("first_down")), "off"),
        ("third_down_attempts", when(scrim & (pl.col("down") == 3)), "off"),
        ("third_down_conversions", when(pl.col("third_down_converted")), "off"),
        ("fourth_down_attempts", when(scrim & (pl.col("down") == 4)), "off"),
        ("fourth_down_conversions", when(pl.col("fourth_down_converted")), "off"),
        ("red_zone_trips", pl.col("_rz_trip"), "off"),
        ("red_zone_touchdowns", when(pl.col("touchdown") & (pl.col("yardline_100") <= 20)), "td"),
        ("sacks_taken", when(pl.col("sack")), "off"),
        ("sacks_made", when(pl.col("sack")), "def"),
        ("qb_hits_taken", when(pl.col("qb_hit")), "off"),
        ("tackles_for_loss_made", when(pl.col("tackled_for_loss")), "def"),
        ("explosive_plays", when(explosive), "off"),
        ("time_of_possession_seconds", pl.col("dt").fill_null(0.0), "prev_off"),
        ("epa_offense", when(scrim & epa.is_not_null(), epa), "off"),
        ("epa_defense", when(scrim & epa.is_not_null(), epa), "def"),
        ("epa_pass", when(scrim & is_pass & epa.is_not_null(), epa), "off"),
        ("epa_rush", when(scrim & is_rush & epa.is_not_null(), epa), "off"),
        ("touchdowns", when(pl.col("touchdown")), "td"),
        ("interceptions_thrown", when(pl.col("interception")), "off"),
        ("fumbles", when(pl.col("fumble")), "off"),
        ("fumbles_lost", when(pl.col("fumble_lost")), "off"),
        ("turnovers_committed", when(turnover), "off"),
        ("turnovers_forced", when(turnover), "def"),
        (
            "defensive_touchdowns",
            when(pl.col("touchdown") & (pl.col("td_team") == pl.col("defteam"))),
            "td",
        ),
        (
            "special_teams_touchdowns",
            when(pl.col("return_touchdown") & (pl.col("kickoff_attempt") | pl.col("punt_attempt"))),
            "td",
        ),
        ("penalties", when(pl.col("penalty")), "penalty"),
        ("penalty_yards", when(pl.col("penalty"), pl.col("penalty_yards")), "penalty"),
        ("pre_snap_penalties", when(pre_snap), "penalty"),
        ("punts", when(pl.col("punt_attempt")), "off"),
        ("punt_yards", when(pl.col("punt_attempt"), pl.col("kick_distance")), "off"),
        ("field_goals_attempted", when(pl.col("field_goal_attempt")), "off"),
        ("field_goals_made", when(fg_made), "off"),
        ("field_goals_missed", when(fg_missed), "off"),
        ("extra_points_attempted", when(pl.col("extra_point_attempt")), "off"),
        (
            "extra_points_made",
            when(pl.col("extra_point_attempt") & (pl.col("extra_point_result") == "good")),
            "off",
        ),
        ("two_point_attempts", when(pl.col("two_point_attempt")), "off"),
        (
            "two_point_conversions",
            when(pl.col("two_point_attempt") & (pl.col("two_point_conv_result") == "success")),
            "off",
        ),
        ("kick_return_yards", when(pl.col("kickoff_attempt"), pl.col("return_yards")), "return"),
        ("punt_return_yards", when(pl.col("punt_attempt"), pl.col("return_yards")), "return"),
        ("successes", when(scrim & success), "off"),
        ("epa_plays", when(scrim & epa.is_not_null()), "off"),
    ]


_ATTRIBUTION_OWNER: dict[str, str] = {
    "off": "posteam",
    "def": "defteam",
    "td": "td_team",
    "penalty": "penalty_team",
    "return": "return_team",
}


def _with_events(frame: pl.DataFrame) -> pl.DataFrame:
    """Split every per-play quantity into a home column and an away column."""
    grp = ["game_id", "drive_id"]
    in_red_zone = (pl.col("yardline_100") <= 20) & pl.col("down").is_not_null()
    frame = frame.with_columns(
        (in_red_zone & (in_red_zone.cum_sum().over(grp) == 1)).cast(pl.Float64).alias("_rz_trip"),
        # Possession time is charged to whoever had the ball when the clock ran.
        pl.col("posteam").shift(1).over("game_id").alias("_prev_posteam"),
    )

    exprs: list[pl.Expr] = []
    for name, value, attribution in _event_specs():
        owner = (
            pl.col("_prev_posteam")
            if attribution == "prev_off"
            else pl.col(_ATTRIBUTION_OWNER[attribution])
        )
        for side in _SIDES:
            exprs.append(
                pl.when(owner == pl.col(f"{side}_team"))
                .then(value)
                .otherwise(0.0)
                .fill_null(0.0)
                .alias(f"ev_{name}_{side}")
            )
    return frame.with_columns(exprs)


# --------------------------------------------------------------------------- #
# Stage 3 — cumulative, lagged features
# --------------------------------------------------------------------------- #

#: Cumulative counters exposed directly, in the order they appear in the output.
_CUMULATIVE = (
    "plays",
    "yards",
    "pass_attempts",
    "pass_completions",
    "pass_yards",
    "rush_attempts",
    "rush_yards",
    "first_downs",
    "third_down_attempts",
    "third_down_conversions",
    "fourth_down_attempts",
    "fourth_down_conversions",
    "red_zone_trips",
    "red_zone_touchdowns",
    "sacks_taken",
    "sacks_made",
    "qb_hits_taken",
    "tackles_for_loss_made",
    "explosive_plays",
    "time_of_possession_seconds",
    "epa_offense",
    "epa_defense",
    "epa_pass",
    "epa_rush",
    "touchdowns",
    "interceptions_thrown",
    "fumbles",
    "fumbles_lost",
    "turnovers_committed",
    "turnovers_forced",
    "defensive_touchdowns",
    "special_teams_touchdowns",
    "penalties",
    "penalty_yards",
    "pre_snap_penalties",
    "punts",
    "punt_yards",
    "field_goals_attempted",
    "field_goals_made",
    "field_goals_missed",
    "extra_points_attempted",
    "extra_points_made",
    "two_point_attempts",
    "two_point_conversions",
    "kick_return_yards",
    "punt_return_yards",
)

#: Cumulative counters that are whole numbers. The EPA sums are not.
_FLOAT_CUMULATIVE = frozenset(
    {"epa_offense", "epa_defense", "epa_pass", "epa_rush", "punt_yards", "penalty_yards"}
)

_DRIVE_EVENTS = (
    "drive",
    "drive_points",
    "scoring_drive",
    "td_drive",
    "fg_drive",
    "punt_drive",
    "to_drive",
    "three_and_out",
    "drive_start",
    "drive_start_n",
)


def _with_cumulative(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.5–2.7 — everything that is a lagged running total."""
    exprs: list[pl.Expr] = []
    for name in _CUMULATIVE:
        for side in _SIDES:
            out = f"{PREFIX}{name}_{side}"
            value = _lag_cum(f"ev_{name}_{side}")
            exprs.append(
                value.alias(out) if name in _FLOAT_CUMULATIVE else value.cast(pl.Int32).alias(out)
            )
    for name in ("successes", "epa_plays", *_DRIVE_EVENTS):
        for side in _SIDES:
            exprs.append(_lag_cum(f"ev_{name}_{side}").alias(f"_cum_{name}_{side}"))
    frame = frame.with_columns(exprs)

    derived: list[pl.Expr] = []
    for side in _SIDES:
        other = "away" if side == "home" else "home"
        plays = pl.col(f"{PREFIX}plays_{side}")
        derived += [
            _safe_div(pl.col(f"{PREFIX}yards_{side}"), plays).alias(
                f"{PREFIX}yards_per_play_{side}"
            ),
            _safe_div(
                pl.col(f"{PREFIX}third_down_conversions_{side}"),
                pl.col(f"{PREFIX}third_down_attempts_{side}"),
            ).alias(f"{PREFIX}third_down_rate_{side}"),
            _safe_div(
                pl.col(f"{PREFIX}fourth_down_conversions_{side}"),
                pl.col(f"{PREFIX}fourth_down_attempts_{side}"),
            ).alias(f"{PREFIX}fourth_down_rate_{side}"),
            _safe_div(pl.col(f"_cum_successes_{side}"), pl.col(f"_cum_epa_plays_{side}")).alias(
                f"{PREFIX}success_rate_{side}"
            ),
            _safe_div(pl.col(f"{PREFIX}epa_offense_{side}"), plays).alias(
                f"{PREFIX}epa_per_play_{side}"
            ),
            (pl.col(f"{PREFIX}epa_offense_{side}") - pl.col(f"{PREFIX}epa_defense_{side}")).alias(
                f"{PREFIX}epa_total_{side}"
            ),
            (
                pl.col(f"{PREFIX}turnovers_forced_{side}")
                - pl.col(f"{PREFIX}turnovers_committed_{side}")
            ).alias(f"{PREFIX}turnover_margin_{side}"),
            _safe_div(
                pl.col(f"{PREFIX}punt_yards_{side}") - pl.col(f"{PREFIX}punt_return_yards_{other}"),
                pl.col(f"{PREFIX}punts_{side}"),
            ).alias(f"{PREFIX}punt_net_average_{side}"),
            pl.col(f"_cum_drive_{side}").cast(pl.Int32).alias(f"{PREFIX}drives_completed_{side}"),
            pl.col(f"_cum_scoring_drive_{side}")
            .cast(pl.Int32)
            .alias(f"{PREFIX}scoring_drives_{side}"),
            pl.col(f"_cum_td_drive_{side}").cast(pl.Int32).alias(f"{PREFIX}td_drives_{side}"),
            pl.col(f"_cum_fg_drive_{side}").cast(pl.Int32).alias(f"{PREFIX}fg_drives_{side}"),
            pl.col(f"_cum_punt_drive_{side}").cast(pl.Int32).alias(f"{PREFIX}punt_drives_{side}"),
            pl.col(f"_cum_to_drive_{side}").cast(pl.Int32).alias(f"{PREFIX}turnover_drives_{side}"),
            pl.col(f"_cum_three_and_out_{side}")
            .cast(pl.Int32)
            .alias(f"{PREFIX}three_and_outs_{side}"),
            pl.col(f"_cum_drive_points_{side}")
            .cast(pl.Int32)
            .alias(f"{PREFIX}drive_points_{side}"),
            _safe_div(pl.col(f"_cum_scoring_drive_{side}"), pl.col(f"_cum_drive_{side}")).alias(
                f"{PREFIX}scoring_drive_rate_{side}"
            ),
            _safe_div(pl.col(f"_cum_drive_points_{side}"), pl.col(f"_cum_drive_{side}")).alias(
                f"{PREFIX}points_per_drive_{side}"
            ),
            _safe_div(
                pl.col(f"_cum_drive_start_{side}"), pl.col(f"_cum_drive_start_n_{side}")
            ).alias(f"{PREFIX}average_drive_start_yardline_100_{side}"),
        ]
    return frame.with_columns(derived)


def _with_momentum(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.8 — trailing windows over the last 5 and 10 plays, and score runs."""
    exprs: list[pl.Expr] = []
    for side in _SIDES:
        exprs += [
            _window_sum(f"ev_epa_offense_{side}", 5).alias(f"{PREFIX}epa_last_5_plays_{side}"),
            _window_sum(f"ev_epa_offense_{side}", 10).alias(f"{PREFIX}epa_last_10_plays_{side}"),
            _window_sum(f"ev_yards_{side}", 5)
            .cast(pl.Int32)
            .alias(f"{PREFIX}yards_last_5_plays_{side}"),
            _window_sum(f"ev_yards_{side}", 10)
            .cast(pl.Int32)
            .alias(f"{PREFIX}yards_last_10_plays_{side}"),
            _safe_div(
                _window_sum(f"ev_successes_{side}", 10), _window_sum(f"ev_epa_plays_{side}", 10)
            ).alias(f"{PREFIX}success_rate_last_10_{side}"),
            _window_sum(f"pts_{side}", 10)
            .cast(pl.Int32)
            .alias(f"{PREFIX}points_last_10_plays_{side}"),
            _window_sum(f"ev_turnovers_committed_{side}", 10)
            .cast(pl.Int32)
            .alias(f"{PREFIX}turnovers_last_10_plays_{side}"),
        ]

    # Unanswered points: the running total of the team that scored last, signed
    # toward home, reset whenever the other team scores.
    signed = pl.col("pts_home") - pl.col("pts_away")
    sign = pl.when(signed > 0).then(1.0).when(signed < 0).then(-1.0).otherwise(None)
    frame = frame.with_columns(sign.alias("_score_sign"))
    frame = frame.with_columns(
        pl.col("_score_sign").forward_fill().over("game_id").alias("_score_sign_ff")
    )
    new_run = (
        pl.col("_score_sign").is_not_null()
        & pl.col("_score_sign").ne_missing(pl.col("_score_sign_ff").shift(1))
    ).over("game_id")
    frame = frame.with_columns(new_run.cum_sum().over("game_id").alias("_run_id"))
    streak = (pl.col("pts_home") + pl.col("pts_away")).cum_sum().over(
        ["game_id", "_run_id"]
    ) * pl.col("_score_sign_ff").fill_null(0.0)
    exprs.append(
        streak.shift(1)
        .fill_null(0.0)
        .over("game_id")
        .cast(pl.Int32)
        .alias(f"{PREFIX}unanswered_points_streak")
    )
    return frame.with_columns(exprs)


def _with_state(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.1–2.4 headline state plus the lead-change / largest-lead history."""
    diff = pl.col("diff_pre")
    lead_sign = pl.when(diff > 0).then(1.0).when(diff < 0).then(-1.0).otherwise(None)
    frame = frame.with_columns(lead_sign.forward_fill().over("game_id").alias("_lead_sign"))
    lead_change = (
        pl.col("_lead_sign").is_not_null()
        & pl.col("_lead_sign").shift(1).is_not_null()
        & (pl.col("_lead_sign") != pl.col("_lead_sign").shift(1))
    ).over("game_id")
    tie_event = ((diff == 0) & (diff.shift(1) != 0)).over("game_id")
    lead_secs = (
        pl.when(pl.col("diff_pre").shift(1) > 0).then(pl.col("dt")).otherwise(0.0).over("game_id")
    )

    frame = frame.with_columns(
        lead_change.cast(pl.Float64).alias("_ev_lead_change"),
        tie_event.cast(pl.Float64).fill_null(0.0).alias("_ev_tie"),
        lead_secs.fill_null(0.0).alias("_ev_lead_secs"),
        diff.clip(0.0, None).alias("_home_lead"),
        (-diff).clip(0.0, None).alias("_away_lead"),
    )

    pos_home = pl.col("pos_is_home")
    gsr, hsr = pl.col("gsr"), pl.col("hsr")
    abs_diff = pl.col("diff_pre").abs()
    diff_pos = pl.when(pos_home).then(diff).when(pos_home.not_()).then(-diff).otherwise(None)

    return frame.with_columns(
        pl.int_range(pl.len()).over("game_id").cast(pl.Int32).alias(f"{PREFIX}play_index"),
        pl.col("qtr").alias(f"{PREFIX}quarter"),
        pl.col("qsr").cast(pl.Int32).alias(f"{PREFIX}quarter_seconds_remaining"),
        hsr.cast(pl.Int32).alias(f"{PREFIX}half_seconds_remaining"),
        gsr.cast(pl.Int32).alias(f"{PREFIX}game_seconds_remaining"),
        pl.col("elapsed").cast(pl.Int32).alias(f"{PREFIX}elapsed_game_seconds"),
        pl.col("is_ot").alias(f"{PREFIX}is_overtime"),
        (hsr <= 120).alias(f"{PREFIX}is_two_minute_warning_half"),
        (pl.col("is_ot").not_() & (gsr <= 120)).alias(f"{PREFIX}is_final_two_minutes"),
        (pl.col("is_ot").not_() & (gsr <= 300)).alias(f"{PREFIX}is_final_five_minutes"),
        pl.col("home_pre").cast(pl.Int32).alias(f"{PREFIX}home_score"),
        pl.col("away_pre").cast(pl.Int32).alias(f"{PREFIX}away_score"),
        diff.cast(pl.Int32).alias(f"{PREFIX}score_diff_home"),
        diff_pos.cast(pl.Int32).alias(f"{PREFIX}score_diff_posteam"),
        abs_diff.cast(pl.Int32).alias(f"{PREFIX}abs_score_diff"),
        (abs_diff <= 8).alias(f"{PREFIX}is_one_score_game"),
        ((abs_diff > 8) & (abs_diff <= 16)).alias(f"{PREFIX}is_two_score_game"),
        (abs_diff / 8.0).ceil().alias(f"{PREFIX}possessions_to_tie"),
        _lag_cum("_ev_lead_change").cast(pl.Int32).alias(f"{PREFIX}lead_changes"),
        _lag_cum("_ev_tie").cast(pl.Int32).alias(f"{PREFIX}times_tied"),
        pl.col("_home_lead")
        .cum_max()
        .over("game_id")
        .cast(pl.Int32)
        .alias(f"{PREFIX}largest_lead_home"),
        pl.col("_away_lead")
        .cum_max()
        .over("game_id")
        .cast(pl.Int32)
        .alias(f"{PREFIX}largest_lead_away"),
        pl.col("_ev_lead_secs")
        .cum_sum()
        .over("game_id")
        .cast(pl.Int32)
        .alias(f"{PREFIX}seconds_leading_home"),
        pos_home.alias(f"{PREFIX}posteam_is_home"),
        pl.col("down").cast(pl.Int32).alias(f"{PREFIX}down"),
        pl.col("ydstogo").cast(pl.Int32).alias(f"{PREFIX}ydstogo"),
        pl.col("yardline_100").cast(pl.Int32).alias(f"{PREFIX}yardline_100"),
        pl.col("goal_to_go").alias(f"{PREFIX}goal_to_go"),
        (pl.col("yardline_100") <= 20).alias(f"{PREFIX}in_red_zone"),
        (pl.col("yardline_100") <= 38).alias(f"{PREFIX}in_fg_range"),
        pl.col("drive_id").alias(f"{PREFIX}drive_number"),
        pl.col("drive_play_index").alias(f"{PREFIX}drive_play_index"),
        pl.col("drive_start_yardline").cast(pl.Int32).alias(f"{PREFIX}drive_start_yardline_100"),
        pl.col("drive_yards_so_far").cast(pl.Int32).alias(f"{PREFIX}drive_yards_so_far"),
        (pl.col("elapsed") - pl.col("drive_start_elapsed"))
        .cast(pl.Int32)
        .alias(f"{PREFIX}drive_elapsed_seconds"),
        pl.col("home_to").cast(pl.Int32).alias(f"{PREFIX}home_timeouts_remaining"),
        pl.col("away_to").cast(pl.Int32).alias(f"{PREFIX}away_timeouts_remaining"),
        pl.when(pos_home)
        .then(pl.col("home_to"))
        .when(pos_home.not_())
        .then(pl.col("away_to"))
        .otherwise(None)
        .cast(pl.Int32)
        .alias(f"{PREFIX}posteam_timeouts_remaining"),
        pl.when(pos_home)
        .then(pl.col("away_to"))
        .when(pos_home.not_())
        .then(pl.col("home_to"))
        .otherwise(None)
        .cast(pl.Int32)
        .alias(f"{PREFIX}defteam_timeouts_remaining"),
    )


def _with_situational(frame: pl.DataFrame) -> pl.DataFrame:
    """§2.4 clock management and §2.11 situational flags."""
    gsr = pl.col(f"{PREFIX}game_seconds_remaining").cast(pl.Float64)
    hsr = pl.col(f"{PREFIX}half_seconds_remaining").cast(pl.Float64)
    diff_pos = pl.col(f"{PREFIX}score_diff_posteam").cast(pl.Float64)
    abs_diff = pl.col(f"{PREFIX}abs_score_diff").cast(pl.Float64)
    down = pl.col(f"{PREFIX}down").cast(pl.Float64)
    pos_to = pl.col(f"{PREFIX}posteam_timeouts_remaining").cast(pl.Float64)
    def_to = pl.col(f"{PREFIX}defteam_timeouts_remaining").cast(pl.Float64)
    in_fg = pl.col(f"{PREFIX}in_fg_range")
    ot = pl.col(f"{PREFIX}is_overtime")

    # Pace so far, used to project how many possessions are left.
    drives_so_far = (
        pl.col(f"{PREFIX}drives_completed_home") + pl.col(f"{PREFIX}drives_completed_away")
    ).cast(pl.Float64)
    elapsed = pl.col(f"{PREFIX}elapsed_game_seconds").cast(pl.Float64)
    seconds_per_drive = (
        pl.when(drives_so_far >= 2).then(elapsed / drives_so_far).otherwise(pl.lit(150.0))
    ).clip(45.0, 400.0)
    remaining_possessions = (gsr / seconds_per_drive).clip(0.0, None)
    posteam_possessions = (remaining_possessions / 2.0).ceil().clip(1.0, None)

    # Clock kill: how much time the leader can bleed off before punting the ball
    # back, allowing for the defence's timeouts.
    kneel_seconds = (4.0 - down.fill_null(1.0)).clip(0.0, 3.0) * 40.0 + 2.0
    kneel_budget = kneel_seconds - def_to.fill_null(0.0) * 40.0
    can_kneel = (diff_pos > 0) & ot.not_() & (gsr <= kneel_budget)

    frame = frame.with_columns(
        (
            (pos_to.fill_null(0.0) - def_to.fill_null(0.0))
            * (1.0 - (hsr / float(_PERIOD_SECONDS))).clip(0.0, 1.0)
        ).alias(f"{PREFIX}timeout_leverage"),
        can_kneel.alias(f"{PREFIX}can_kneel_out"),
        ((hsr <= 120) & (diff_pos <= 0)).alias(f"{PREFIX}two_minute_drill_active"),
        remaining_possessions.alias(f"{PREFIX}expected_remaining_possessions"),
    )

    must_score = -diff_pos > 8.0 * posteam_possessions
    return frame.with_columns(
        (((gsr <= 900) & (abs_diff > 16)) | ((gsr <= 1800) & (abs_diff > 24))).alias(
            f"{PREFIX}garbage_time"
        ),
        must_score.alias(f"{PREFIX}must_score_posteam"),
        (pl.col(f"{PREFIX}can_kneel_out") & (gsr <= 120)).alias(
            f"{PREFIX}victory_formation_likely"
        ),
        # A free-kick snap is the one with no down. Reading ``play_type`` here
        # would read the outcome of play *t*, which is exactly what §2 forbids.
        (down.is_null() & (diff_pos < 0) & (gsr <= 300)).alias(f"{PREFIX}onside_kick_situation"),
        ((down == 4) & ((pl.col(f"{PREFIX}ydstogo") <= 2) | must_score)).alias(
            f"{PREFIX}fourth_down_go_situation"
        ),
        (in_fg & (diff_pos == -3)).alias(f"{PREFIX}field_goal_to_tie"),
        (in_fg & (diff_pos >= -2) & (diff_pos <= 0)).alias(f"{PREFIX}field_goal_to_win"),
        ((diff_pos == -7) | (diff_pos == -8)).alias(f"{PREFIX}td_to_tie"),
        ((diff_pos >= -6) & (diff_pos <= -4)).alias(f"{PREFIX}td_to_win"),
        (diff_pos == -8).alias(f"{PREFIX}two_point_needed_after_td"),
        (hsr <= 120).alias(f"{PREFIX}end_of_half_possession"),
        ((diff_pos > 0) & (gsr <= 300) & ot.not_()).alias(f"{PREFIX}clock_kill_mode"),
        ((-diff_pos) / (8.0 * posteam_possessions))
        .clip(0.0, 3.0)
        .alias(f"{PREFIX}desperation_index"),
        ((pl.col(f"{PREFIX}quarter") >= 4) & (abs_diff <= 8)).alias(
            f"{PREFIX}fourth_quarter_within_one_score"
        ),
        (pl.col("posteam").drop_nulls().first().over("game_id") != pl.col("home_team")).alias(
            f"{PREFIX}receiving_second_half_is_home"
        ),
        (
            (pl.col(f"{PREFIX}epa_last_10_plays_home") - pl.col(f"{PREFIX}epa_last_10_plays_away"))
            / 3.0
        )
        .tanh()
        .alias(f"{PREFIX}momentum_score"),
    )


#: Home/away pairs also exposed from the possessing team's point of view.
_POSTEAM_VIEWS = (
    "epa_last_5_plays",
    "epa_last_10_plays",
    "yards_last_10_plays",
    "success_rate_last_10",
    "epa_per_play",
    "plays",
    "yards",
    "turnovers_committed",
)


def _with_posteam_views(frame: pl.DataFrame) -> pl.DataFrame:
    """Home/away pairs re-expressed from the possessing team's point of view."""
    exprs: list[pl.Expr] = []
    for name in _POSTEAM_VIEWS:
        home, away = f"{PREFIX}{name}_home", f"{PREFIX}{name}_away"
        exprs.append(_pick(home, away).alias(f"{PREFIX}{name}_posteam"))
        exprs.append(_pick(away, home).alias(f"{PREFIX}{name}_defteam"))
    return frame.with_columns(exprs)


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

_STATE_COLUMNS = (
    "play_index",
    "quarter",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "elapsed_game_seconds",
    "is_overtime",
    "is_two_minute_warning_half",
    "is_final_two_minutes",
    "is_final_five_minutes",
    "home_score",
    "away_score",
    "score_diff_home",
    "score_diff_posteam",
    "abs_score_diff",
    "is_one_score_game",
    "is_two_score_game",
    "possessions_to_tie",
    "lead_changes",
    "times_tied",
    "largest_lead_home",
    "largest_lead_away",
    "seconds_leading_home",
    "posteam_is_home",
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "in_red_zone",
    "in_fg_range",
    "drive_number",
    "drive_play_index",
    "drive_start_yardline_100",
    "drive_yards_so_far",
    "drive_elapsed_seconds",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "timeout_leverage",
    "can_kneel_out",
    "two_minute_drill_active",
)

_PER_SIDE_COLUMNS = (
    *_CUMULATIVE,
    "yards_per_play",
    "third_down_rate",
    "fourth_down_rate",
    "success_rate",
    "epa_per_play",
    "epa_total",
    "turnover_margin",
    "punt_net_average",
    "average_drive_start_yardline_100",
    "drives_completed",
    "scoring_drives",
    "scoring_drive_rate",
    "drive_points",
    "points_per_drive",
    "td_drives",
    "fg_drives",
    "punt_drives",
    "turnover_drives",
    "three_and_outs",
    "epa_last_5_plays",
    "epa_last_10_plays",
    "yards_last_5_plays",
    "yards_last_10_plays",
    "success_rate_last_10",
    "points_last_10_plays",
    "turnovers_last_10_plays",
)

_FLAG_COLUMNS = (
    "unanswered_points_streak",
    "momentum_score",
    "expected_remaining_possessions",
    "garbage_time",
    "must_score_posteam",
    "victory_formation_likely",
    "onside_kick_situation",
    "fourth_down_go_situation",
    "field_goal_to_tie",
    "field_goal_to_win",
    "td_to_tie",
    "td_to_win",
    "two_point_needed_after_td",
    "end_of_half_possession",
    "clock_kill_mode",
    "desperation_index",
    "fourth_quarter_within_one_score",
    "receiving_second_half_is_home",
)


def _feature_columns() -> tuple[str, ...]:
    cols: list[str] = list(_STATE_COLUMNS)
    for name in _PER_SIDE_COLUMNS:
        cols += [f"{name}_home", f"{name}_away"]
    for name in _POSTEAM_VIEWS:
        cols += [f"{name}_posteam", f"{name}_defteam"]
    cols += list(_FLAG_COLUMNS)
    return tuple(f"{PREFIX}{c}" for c in cols)


FEATURE_COLUMNS: tuple[str, ...] = _feature_columns()


#: Per-side features a :class:`LiveSnapshot` cannot fill. A ``LivePlay`` carries
#: the play's type, yardage, scores and coarse turnover/penalty flags. It carries
#: no EPA, no per-play statistical detail (completions, first downs, sacks), no
#: penalised team and no kick result, so features built on those are ``None`` on
#: the serving path rather than a plausible-looking zero.
_UNKNOWABLE_PER_SIDE = (
    "pass_completions",
    "first_downs",
    "third_down_conversions",
    "third_down_rate",
    "fourth_down_conversions",
    "fourth_down_rate",
    "red_zone_touchdowns",
    "sacks_taken",
    "sacks_made",
    "qb_hits_taken",
    "tackles_for_loss_made",
    "epa_offense",
    "epa_defense",
    "epa_pass",
    "epa_rush",
    "epa_total",
    "epa_per_play",
    "success_rate",
    "touchdowns",
    "interceptions_thrown",
    "fumbles",
    "fumbles_lost",
    "defensive_touchdowns",
    "special_teams_touchdowns",
    "penalties",
    "penalty_yards",
    "pre_snap_penalties",
    "punt_yards",
    "punt_net_average",
    "field_goals_made",
    "field_goals_missed",
    "extra_points_attempted",
    "extra_points_made",
    "two_point_attempts",
    "two_point_conversions",
    "kick_return_yards",
    "punt_return_yards",
    "epa_last_5_plays",
    "epa_last_10_plays",
    "success_rate_last_10",
)


def _unknowable() -> frozenset[str]:
    names: set[str] = set()
    for name in _UNKNOWABLE_PER_SIDE:
        names |= {f"{PREFIX}{name}_home", f"{PREFIX}{name}_away"}
    for name in _POSTEAM_VIEWS:
        if name in _UNKNOWABLE_PER_SIDE:
            names |= {f"{PREFIX}{name}_posteam", f"{PREFIX}{name}_defteam"}
    names.add(f"{PREFIX}momentum_score")
    return frozenset(names & set(FEATURE_COLUMNS))


SNAPSHOT_UNKNOWABLE_COLUMNS: frozenset[str] = _unknowable()

#: Features that move with the game clock. A snapshot truncated to ``Q4 05:00``
#: reports 300 seconds left; the play-by-play row for the snap that follows
#: reports whatever the clock actually read when the ball was snapped (4:58 in
#: the 2024_01_BAL_KC parity fixture). The numeric ones differ by that gap and
#: compare within a tolerance rather than exactly; the three booleans threshold
#: on it and can flip when the gap straddles the threshold. Verified by replaying
#: 1,723 snapshots across all 285 games of 2024: these are the *only* columns the
#: two paths ever disagree on — bar one drive-boundary edge case in those 1,723
#: (0.06%), where nflverse's ``fixed_drive`` grouping is not recoverable from a
#: snapshot's drive list alone.
SNAPSHOT_CLOCK_SENSITIVE_COLUMNS: frozenset[str] = frozenset(
    {
        f"{PREFIX}quarter_seconds_remaining",
        f"{PREFIX}half_seconds_remaining",
        f"{PREFIX}game_seconds_remaining",
        f"{PREFIX}elapsed_game_seconds",
        f"{PREFIX}seconds_leading_home",
        f"{PREFIX}drive_elapsed_seconds",
        f"{PREFIX}timeout_leverage",
        f"{PREFIX}expected_remaining_possessions",
        f"{PREFIX}desperation_index",
        f"{PREFIX}can_kneel_out",
        f"{PREFIX}victory_formation_likely",
    }
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_ingame(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per-play cumulative in-game state, one row per ``(game_id, play_id)``.

    ``pbp`` is an nflverse play-by-play frame, or any frame carrying a subset of
    its columns — a missing input column yields null features rather than an
    error, which is what lets the serving path reuse these expressions.
    """
    if pbp.height == 0:
        out = pl.DataFrame(
            {
                "game_id": pl.Series([], dtype=pl.Utf8),
                "play_id": pl.Series([], dtype=pl.Int64),
            }
        ).with_columns([pl.lit(None).alias(c) for c in FEATURE_COLUMNS])
        assert_no_market_columns(out, context="build_ingame")
        return out

    frame = _prepare(pbp)
    frame = _with_clock(frame)
    frame = _with_scoring(frame)
    frame = _with_drives(frame)
    frame = _with_events(frame)
    frame = _with_cumulative(frame)
    frame = _with_momentum(frame)
    frame = _with_state(frame)
    frame = _with_situational(frame)
    frame = _with_posteam_views(frame)

    out = frame.select(
        pl.col("game_id"),
        pl.col("play_id").cast(pl.Int64),
        *[pl.col(c) for c in FEATURE_COLUMNS],
    )
    assert_no_market_columns(out, context="build_ingame")
    return out


def load_pbp_frame(
    seasons: Sequence[int] | None = None, root: Path | str | None = None
) -> pl.DataFrame:
    """Read the hive-partitioned play-by-play under ``data/raw/nflverse/pbp``."""
    base = Path(root) if root is not None else RAW_DIR / "nflverse" / "pbp"
    if seasons is None:
        paths = sorted(base.glob("season=*/*.parquet"))
    else:
        paths = [p for s in seasons for p in sorted((base / f"season={s}").glob("*.parquet"))]
    if not paths:
        raise FileNotFoundError(f"no play-by-play parquet under {base}")
    frames = [pl.read_parquet(p) for p in paths]
    common = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    return pl.concat([f.select(common) for f in frames], how="vertical_relaxed")


# --------------------------------------------------------------------------- #
# Serving path
# --------------------------------------------------------------------------- #

#: ESPN and nflverse spell play types differently. Both are normalised to the
#: nflverse vocabulary so ``build_ingame`` sees one language.
_PLAY_TYPE_ALIASES: dict[str, str] = {
    "pass": "pass",
    "pass reception": "pass",
    "pass incompletion": "pass",
    "pass interception return": "pass",
    "passing touchdown": "pass",
    "sack": "pass",
    "run": "run",
    "rush": "run",
    "rushing touchdown": "run",
    "punt": "punt",
    "punt return touchdown": "punt",
    "blocked punt": "punt",
    "field goal good": "field_goal",
    "field goal missed": "field_goal",
    "blocked field goal": "field_goal",
    "field_goal": "field_goal",
    "kickoff": "kickoff",
    "kickoff return (offense)": "kickoff",
    "kickoff return touchdown": "kickoff",
    "extra point good": "extra_point",
    "extra point missed": "extra_point",
    "extra_point": "extra_point",
    "two point conversion": "two_point",
    "timeout": "no_play",
    "end period": "no_play",
    "end of half": "no_play",
    "no_play": "no_play",
    "penalty": "no_play",
    "qb kneel": "qb_kneel",
    "qb spike": "qb_spike",
}

_SNAPSHOT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game_id": pl.Utf8,
    "play_id": pl.Float64,
    "order_sequence": pl.Float64,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "posteam": pl.Utf8,
    "defteam": pl.Utf8,
    "qtr": pl.Float64,
    "quarter_seconds_remaining": pl.Float64,
    "down": pl.Float64,
    "ydstogo": pl.Float64,
    "yardline_100": pl.Float64,
    "total_home_score": pl.Float64,
    "total_away_score": pl.Float64,
    "home_timeouts_remaining": pl.Float64,
    "away_timeouts_remaining": pl.Float64,
    "play_type": pl.Utf8,
    "yards_gained": pl.Float64,
    "punt_attempt": pl.Float64,
    "field_goal_attempt": pl.Float64,
    "kickoff_attempt": pl.Float64,
    "penalty": pl.Float64,
    "is_turnover": pl.Float64,
    "fixed_drive": pl.Float64,
    # Not read by :func:`build_ingame` (``_prepare`` derives its own), but
    # :func:`nfl.features.baseline_inputs.build_baseline_inputs` reads all three
    # by name and the serving path pushes *this* frame through both builders.
    "half_seconds_remaining": pl.Float64,
    "game_seconds_remaining": pl.Float64,
    "goal_to_go": pl.Float64,
}


def _normalise_play_type(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    return _PLAY_TYPE_ALIASES.get(text, text)


def _maybe_float(value: int | float | None) -> float | None:
    return None if value is None else float(value)


def _drive_continues(snapshot: LiveSnapshot) -> bool:
    """Does the upcoming snap belong to the drive already in progress?

    Mirrors how nflverse's ``fixed_drive`` groups plays, which is what the
    play-by-play path counts drives with:

    * A snap with no offence named — a two-minute warning, a timeout, an
      administrative ``no_play`` — stays on the current drive.
    * The same offence keeps the ball, so it keeps the drive.
    * A touchdown scored from scrimmage hands the ball to the *other* team for
      the try, and nflverse keeps that try on the drive that produced the score.
      A pick-six is the case that matters: the drive is the offence's, the try is
      the defence's, and they share one ``fixed_drive``.

    Everything else — a kickoff after a try, a punt, a made field goal — starts a
    new drive.
    """
    possession = snapshot.possession
    if possession is None:
        return True
    last = snapshot.last_play
    if last is None:
        return False
    if last.team == possession:
        return True
    return last.scoring_play and _normalise_play_type(last.play_type) in ("pass", "run")


def snapshot_play_frame(snapshot: LiveSnapshot) -> pl.DataFrame:
    """A play-by-play-like frame for everything a snapshot has seen, plus the snap.

    One row per play already run, in order, then one synthetic row for the snap
    the snapshot is poised at. This is the single source both serving builders
    read: :func:`build_ingame` for the ``ig__`` block and
    :func:`~nfl.features.baseline_inputs.build_baseline_inputs` for ``bi__``, so
    the two sides of a serving row cannot drift apart the way two hand-written
    reshapes would.

    The synthetic row's *outcome* columns are null — nothing has happened yet —
    while its pre-snap situation is what the model is asked to predict from.
    """
    from nfl.data.live.base import game_seconds_remaining, half_seconds_remaining

    home, away = snapshot.home_team, snapshot.away_team

    def other(team: str | None) -> str | None:
        return None if team is None else (away if team == home else home)

    def clocks(period: int, quarter_seconds: int | None) -> tuple[float | None, float | None]:
        """``(half, game)`` seconds remaining, or ``(None, None)`` with no clock."""
        if quarter_seconds is None:
            return None, None
        return (
            float(half_seconds_remaining(int(period), int(quarter_seconds))),
            float(game_seconds_remaining(int(period), int(quarter_seconds))),
        )

    def goal_to_go(ydstogo: float | None, yardline: float | None) -> float:
        """nflverse's flag, derived. ``0`` when the row is not a snap at all.

        ``_prepare`` derives the ``ig__`` side the same way. nflverse writes
        ``0`` rather than null on administrative rows (a timeout has no yard
        line), so an unknown situation is ``0`` here too — returning null there
        was the one ``bi__`` column the two paths disagreed on.
        """
        if ydstogo is None or yardline is None:
            return 0.0
        return float(ydstogo >= yardline)

    rows: list[dict[str, Any]] = []
    for order, (drive, play) in enumerate(
        ((d, p) for d in snapshot.drives for p in d.plays), start=1
    ):
        play_type = _normalise_play_type(play.play_type)
        play_half, play_game = clocks(play.period, play.clock_seconds)
        rows.append(
            {
                "game_id": snapshot.game_id or "",
                "play_id": float(order),
                "order_sequence": float(order),
                "home_team": home,
                "away_team": away,
                "posteam": play.team,
                "defteam": other(play.team),
                "qtr": float(play.period),
                "quarter_seconds_remaining": _maybe_float(play.clock_seconds),
                "down": _maybe_float(play.start_down),
                "ydstogo": _maybe_float(play.start_distance),
                "yardline_100": _maybe_float(play.start_yards_to_endzone),
                "total_home_score": _maybe_float(play.home_score),
                "total_away_score": _maybe_float(play.away_score),
                "home_timeouts_remaining": _maybe_float(play.home_timeouts_remaining),
                "away_timeouts_remaining": _maybe_float(play.away_timeouts_remaining),
                "play_type": play_type,
                "yards_gained": _maybe_float(play.stat_yardage),
                "punt_attempt": float(play_type == "punt"),
                "field_goal_attempt": float(play_type == "field_goal"),
                "kickoff_attempt": float(play_type == "kickoff"),
                "penalty": float(play.is_penalty),
                "is_turnover": float(play.is_turnover),
                "fixed_drive": _maybe_float(drive.drive_number),
                "half_seconds_remaining": play_half,
                "game_seconds_remaining": play_game,
                "goal_to_go": goal_to_go(
                    _maybe_float(play.start_distance),
                    _maybe_float(play.start_yards_to_endzone),
                ),
            }
        )

    # The synthetic snap. Its own outcome columns are null: nothing has happened
    # yet. Its pre-snap situation is what the model is asked to predict from.
    last_drive = snapshot.drives[-1] if snapshot.drives else None
    if last_drive is not None and last_drive.drive_number is not None:
        drive_number = float(last_drive.drive_number + (0 if _drive_continues(snapshot) else 1))
    else:
        drive_number = 1.0
    order = len(rows) + 1
    rows.append(
        {
            "game_id": snapshot.game_id or "",
            "play_id": float(order),
            "order_sequence": float(order),
            "home_team": home,
            "away_team": away,
            "posteam": snapshot.possession,
            "defteam": other(snapshot.possession),
            "qtr": float(snapshot.period),
            "quarter_seconds_remaining": float(snapshot.clock_seconds),
            "down": _maybe_float(snapshot.down),
            "ydstogo": _maybe_float(snapshot.distance),
            "yardline_100": _maybe_float(snapshot.yards_to_endzone),
            "total_home_score": float(snapshot.home_score),
            "total_away_score": float(snapshot.away_score),
            "home_timeouts_remaining": _maybe_float(snapshot.home_timeouts),
            "away_timeouts_remaining": _maybe_float(snapshot.away_timeouts),
            "play_type": None,
            "yards_gained": None,
            "punt_attempt": 0.0,
            "field_goal_attempt": 0.0,
            "kickoff_attempt": 0.0,
            "penalty": 0.0,
            "is_turnover": 0.0,
            "fixed_drive": drive_number,
            "half_seconds_remaining": float(snapshot.half_seconds_remaining),
            "game_seconds_remaining": float(snapshot.game_seconds_remaining),
            "goal_to_go": goal_to_go(
                _maybe_float(snapshot.distance), _maybe_float(snapshot.yards_to_endzone)
            ),
        }
    )
    return pl.DataFrame(rows, schema=_SNAPSHOT_SCHEMA)


def snapshot_history_rows(snapshot: LiveSnapshot) -> list[dict[str, Any]]:
    """The ``ig__`` block for **every** play the snapshot has seen, plus the snap.

    Same columns and same expressions as :func:`build_ingame` — one call, over
    :func:`snapshot_play_frame`, so the running aggregates accumulate exactly as
    they do offline. The last element is :func:`snapshot_to_row`; the ones
    before it are the play history a sequence model windows over.

    Without this, ``nfl predict`` built a single row and the GRU saw a window of
    length one: a model trained on 32 plays of context served on 1, which is a
    different model. Columns in :data:`SNAPSHOT_UNKNOWABLE_COLUMNS` are ``None``
    on every row, as they are for the snap itself.
    """
    built = build_ingame(snapshot_play_frame(snapshot))
    rows = built.to_dicts()
    for row in rows:
        row["play_id"] = None
        for column in SNAPSHOT_UNKNOWABLE_COLUMNS:
            row[column] = None
    return rows


def snapshot_to_row(snapshot: LiveSnapshot) -> dict[str, Any]:
    """The ``ig__`` feature row for the snap a :class:`LiveSnapshot` is poised at.

    The last row of :func:`snapshot_history_rows`: same columns as
    :func:`build_ingame`, produced by the same expressions, with the columns in
    :data:`SNAPSHOT_UNKNOWABLE_COLUMNS` set to ``None``.

    Two honest differences from the play-by-play row for the same snap:

    * **Clock.** A truncated snapshot reports the time that was *asked for*
      (``Q4 05:00``), not the game clock of the next snap (``4:58``). The
      seconds-remaining features differ by that gap; every flag derived from
      them is unaffected unless the gap straddles a threshold. Ask for the
      snap's own clock and the gap — and every difference it causes — is zero;
      that is how the serving-parity gate samples.
    * **Timeouts.** ``HistoricalReplayFeed`` carries per-play timeouts, so they
      are exact. ESPN publishes timeouts only for "now"; a truncated ESPN
      snapshot sets ``timeouts_exact=False`` and the timeout features are then
      the feed's current values, not the values at ``at``.
    """
    return snapshot_history_rows(snapshot)[-1]


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #

_STATE_DOCS: dict[str, str] = {
    "play_index": "0-based play counter within the game.",
    "quarter": "Period number; 5+ is overtime.",
    "quarter_seconds_remaining": "Seconds left in the period, pre-snap.",
    "half_seconds_remaining": "Seconds left in the half, pre-snap.",
    "game_seconds_remaining": (
        "Seconds left in regulation; collapses to the period clock in overtime."
    ),
    "elapsed_game_seconds": "Game seconds burned so far, summed from the play clock.",
    "is_overtime": "Period > 4.",
    "is_two_minute_warning_half": "Two minutes or less left in the half.",
    "is_final_two_minutes": "Two minutes or less left in regulation.",
    "is_final_five_minutes": "Five minutes or less left in regulation.",
    "home_score": "Home score before this snap (nflverse score lagged one play).",
    "away_score": "Away score before this snap.",
    "score_diff_home": "home_score - away_score, pre-snap.",
    "score_diff_posteam": "Score differential from the possessing team's view.",
    "abs_score_diff": "Absolute score differential.",
    "is_one_score_game": "Absolute differential <= 8.",
    "is_two_score_game": "Absolute differential in (8, 16].",
    "possessions_to_tie": "ceil(|differential| / 8): scoring drives needed to level.",
    "lead_changes": "Lead changes over plays < t.",
    "times_tied": "Times the game became tied over plays < t.",
    "largest_lead_home": "Largest home lead so far.",
    "largest_lead_away": "Largest away lead so far.",
    "seconds_leading_home": "Game seconds the home team has spent in the lead.",
    "posteam_is_home": "True when the home team has the ball.",
    "down": "Down of the upcoming snap; null on kickoffs and extra points.",
    "ydstogo": "Yards to the line to gain.",
    "yardline_100": "Distance to the opponent's end zone, 1-99.",
    "goal_to_go": "Line to gain is the goal line (ydstogo >= yardline_100).",
    "in_red_zone": "yardline_100 <= 20.",
    "in_fg_range": "yardline_100 <= 38.",
    "drive_number": "1-based drive counter within the game.",
    "drive_play_index": "Plays already run on the current drive.",
    "drive_start_yardline_100": "Field position at the drive's first scrimmage snap.",
    "drive_yards_so_far": "Scrimmage yards gained on the current drive before t.",
    "drive_elapsed_seconds": "Game seconds elapsed since the drive began.",
    "home_timeouts_remaining": "Home timeouts left, pre-snap.",
    "away_timeouts_remaining": "Away timeouts left, pre-snap.",
    "posteam_timeouts_remaining": "Timeouts left for the team with the ball.",
    "defteam_timeouts_remaining": "Timeouts left for the defence.",
    "timeout_leverage": ("(posteam - defteam timeouts) scaled by how little of the half is left."),
    "can_kneel_out": (
        "The leader with the ball can drain the clock given the down and the defence's timeouts."
    ),
    "two_minute_drill_active": "Two minutes or less in the half and posteam not ahead.",
    "unanswered_points_streak": (
        "Points scored since the other team last scored, signed toward home."
    ),
    "momentum_score": "tanh of the home-minus-away EPA gap over the last 10 plays.",
    "expected_remaining_possessions": (
        "Remaining game seconds divided by the observed seconds per drive."
    ),
    "garbage_time": "Score out of reach for the time left.",
    "must_score_posteam": "Deficit exceeds 8 points per remaining posteam possession.",
    "victory_formation_likely": "can_kneel_out inside the last two minutes.",
    "onside_kick_situation": "Free-kick snap by a trailing team inside the last five minutes.",
    "fourth_down_go_situation": "Fourth down, short or must-score.",
    "field_goal_to_tie": "In range and trailing by exactly 3.",
    "field_goal_to_win": "In range and trailing by 2 or less (or tied).",
    "td_to_tie": "Trailing by 7 or 8.",
    "td_to_win": "Trailing by 4 to 6.",
    "two_point_needed_after_td": "Trailing by 8: a touchdown needs the two-pointer.",
    "end_of_half_possession": "Possession inside the final two minutes of a half.",
    "clock_kill_mode": "Leader with the ball inside the last five minutes.",
    "desperation_index": "Deficit per remaining possession, capped at 3.",
    "fourth_quarter_within_one_score": "Q4 or later with the margin inside 8.",
    "receiving_second_half_is_home": (
        "The home team receives the second-half kickoff (from the opening possession)."
    ),
}

_SIDE_DOCS: dict[str, str] = {
    "plays": "Scrimmage plays run",
    "yards": "Scrimmage yards gained",
    "pass_attempts": "Pass plays (sacks included, as nflverse counts them)",
    "pass_completions": "Completed passes",
    "pass_yards": "Net yards on pass plays (sack losses included)",
    "rush_attempts": "Rush plays",
    "rush_yards": "Net yards on rush plays",
    "first_downs": "First downs gained",
    "third_down_attempts": "Third-down snaps",
    "third_down_conversions": "Third downs converted",
    "fourth_down_attempts": "Fourth-down snaps",
    "fourth_down_conversions": "Fourth downs converted",
    "red_zone_trips": "Drives that reached the opponent 20",
    "red_zone_touchdowns": "Touchdowns scored from inside the 20",
    "sacks_taken": "Sacks allowed",
    "sacks_made": "Sacks recorded on defence",
    "qb_hits_taken": "Quarterback hits allowed",
    "tackles_for_loss_made": "Tackles for loss recorded on defence",
    "explosive_plays": f"Runs of {_EXPLOSIVE_RUSH}+ or passes of {_EXPLOSIVE_PASS}+ yards",
    "time_of_possession_seconds": "Game seconds held with the ball",
    "epa_offense": "Sum of nflverse EPA on offence",
    "epa_defense": "Sum of nflverse EPA allowed on defence",
    "epa_pass": "Sum of EPA on pass plays",
    "epa_rush": "Sum of EPA on rush plays",
    "touchdowns": "Touchdowns scored",
    "interceptions_thrown": "Interceptions thrown",
    "fumbles": "Fumbles (lost or not)",
    "fumbles_lost": "Fumbles lost",
    "turnovers_committed": "Interceptions plus fumbles lost",
    "turnovers_forced": "Turnovers taken away on defence",
    "defensive_touchdowns": "Touchdowns scored by the defence",
    "special_teams_touchdowns": "Kick or punt return touchdowns",
    "penalties": "Penalties accepted against this team",
    "penalty_yards": "Penalty yards assessed against this team",
    "pre_snap_penalties": "Pre-snap penalties (false start, offside, delay)",
    "punts": "Punts",
    "punt_yards": "Gross punt yards",
    "field_goals_attempted": "Field goals attempted",
    "field_goals_made": "Field goals made",
    "field_goals_missed": "Field goals missed or blocked",
    "extra_points_attempted": "Extra points attempted",
    "extra_points_made": "Extra points made",
    "two_point_attempts": "Two-point conversions attempted",
    "two_point_conversions": "Two-point conversions made",
    "kick_return_yards": "Kickoff return yards",
    "punt_return_yards": "Punt return yards",
    "yards_per_play": "Scrimmage yards per play",
    "third_down_rate": "Third-down conversion rate (null before any attempt)",
    "fourth_down_rate": "Fourth-down conversion rate (null before any attempt)",
    "success_rate": "Fraction of scrimmage plays with positive EPA",
    "epa_per_play": "Offensive EPA per scrimmage play",
    "epa_total": "Offensive EPA minus EPA allowed",
    "turnover_margin": "Turnovers forced minus turnovers committed",
    "punt_net_average": "Net punt average against the opponent's return yards",
    "average_drive_start_yardline_100": "Mean starting field position of completed drives",
    "drives_completed": "Drives finished before this snap",
    "scoring_drives": "Completed drives that produced points",
    "scoring_drive_rate": "Scoring drives divided by completed drives",
    "drive_points": "Points scored on completed drives",
    "points_per_drive": "Points per completed drive",
    "td_drives": "Completed drives worth 6 or more points",
    "fg_drives": "Completed drives worth exactly 3 points",
    "punt_drives": "Completed drives containing a punt",
    "turnover_drives": "Completed drives containing a turnover",
    "three_and_outs": "Punt drives of three scrimmage plays or fewer",
    "epa_last_5_plays": "Offensive EPA over the previous 5 plays of the game",
    "epa_last_10_plays": "Offensive EPA over the previous 10 plays of the game",
    "yards_last_5_plays": "Scrimmage yards over the previous 5 plays of the game",
    "yards_last_10_plays": "Scrimmage yards over the previous 10 plays of the game",
    "success_rate_last_10": "Success rate over the previous 10 plays of the game",
    "points_last_10_plays": "Points scored over the previous 10 plays of the game",
    "turnovers_last_10_plays": "Turnovers committed over the previous 10 plays",
}


def _build_feature_docs() -> dict[str, str]:
    docs = {f"{PREFIX}{name}": text for name, text in _STATE_DOCS.items()}
    for stem, text in _SIDE_DOCS.items():
        for side in _SIDES:
            docs[f"{PREFIX}{stem}_{side}"] = f"{text} by the {side} team, over plays < t."
    for stem in _POSTEAM_VIEWS:
        base = _SIDE_DOCS[stem]
        docs[f"{PREFIX}{stem}_posteam"] = f"{base}, by the team with the ball."
        docs[f"{PREFIX}{stem}_defteam"] = f"{base}, by the defending team."
    return docs


FEATURE_DOCS: dict[str, str] = _build_feature_docs()
