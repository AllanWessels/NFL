"""``tm__`` — as-of-kickoff team features, one row per ``(game_id, team)``.

Everything in this module answers one question: *what did we know about this team
the moment before this game kicked off?* Two rows come out per game — one from
each team's perspective — carrying rating, form, schedule context, coach record
and opponent-mirrored defensive splits.

**The as-of rule is structural, not a filter.** Every rolling quantity is a
``cum_sum()``/``rolling_sum()`` that has been ``shift(1)``-ed over a
franchise-major ordering by kickoff, so a game can never contribute to its own
row and later games can never contribute to earlier ones. Elo is a forward
sequential fold, which has the same property by construction. Ranks come from a
``join_asof`` against a state table keyed on ``week + 1``, so a week-``w`` rank
sees only games from weeks ``< w``. ``tests/test_features_team.py`` pins all
three with the delete-the-future invariant from ``docs/DATA_MODEL.md``.

Layout of the feature set:

``tm__elo_*``
    Elo rating per ``docs/DATA_MODEL.md`` §3 — configurable K, home-field
    advantage in rating points, a margin-of-victory multiplier, and
    between-season regression toward the league mean. ``tm__elo_pre`` is the
    rating *before* this game; ``tm__elo_opp_pre`` and ``tm__elo_diff`` give the
    matchup, and ``tm__elo_win_prob`` the model's own pre-game expectation.
``tm__power_rating*``
    An entirely in-house alternative to Elo: an exponentially weighted mean of
    prior point differential. Cheap, transparent, and different enough from Elo
    to be worth carrying alongside it.
``tm__<metric>_{season,l4,l8,prior}``
    Rolling efficiency from play-by-play, over four windows: season-to-date,
    last 4 games, last 8 games (both crossing the season boundary), and the
    whole of the prior season.
``tm__rank_*``
    1–32 league ranks as of the week of the game, computed across every
    franchise in the season — including teams on a bye that week, which is why
    they come from a state grid rather than from the week's game rows.
``tm__rest_*``, ``tm__travel_*``, ``tm__is_*``, ``tm__wx_*``
    Schedule and environment context. Travel and timezone shift are measured
    from the venue the team last *played* in (falling back to the last venue it
    hosted in), never from a lookup that reads the rest of the schedule.
``tm__coach_*``
    Joined from :mod:`nfl.data.coach_records`, which already enforces the
    exclude-the-current-game rule.

Market columns never enter: the play-by-play and schedule columns this module
reads are listed explicitly, and
:func:`~nfl.data.market_blocklist.assert_no_market_columns` guards the return.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from nfl.data.market_blocklist import assert_no_market_columns
from nfl.data.venues import resolve_games

__all__ = [
    "PREFIX",
    "DEFAULT_ELO",
    "FEATURE_DOCS",
    "EloConfig",
    "RollingConfig",
    "elo_expected",
    "elo_update",
    "compute_elo",
    "build_team_features",
    "load_pbp_frame",
    "load_schedules_frame",
    "load_weather_frame",
]

PREFIX = "tm__"

#: Ordering used by every window function in this module.
_ORDER = ["kickoff", "game_id"]

#: Relocations and vendor spelling drift. Ratings and rolling form follow the
#: *franchise*, not the three-letter code that happened to be current: the 2016
#: Rams did not restart at 1500 because the abbreviation changed. The unmapped
#: source code stays on the row as ``team``, which is the join key.
FRANCHISE_ALIASES: dict[str, str] = {
    "SD": "LAC",
    "SDG": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "RAM": "LA",
    "OAK": "LV",
    "RAI": "LV",
    "JAC": "JAX",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "WSH": "WAS",
}

#: Anything polars accepts as a dtype: an instance or the class itself.
PolarsDType = pl.DataType | type[pl.DataType]

_SCHEDULE_TZ = "America/New_York"
_EARTH_RADIUS_MILES = 3958.7613


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EloConfig:
    """Elo parameters. Defaults are the classic FiveThirtyEight NFL settings.

    ``k``
        Rating points moved by a one-unit surprise, before the margin multiplier.
    ``home_advantage``
        Rating points added to the home team's rating when forming the
        expectation. Zero at neutral sites. 55 points ≈ 2.1 points of spread.
    ``mean``
        League mean; new franchises start here and seasons regress toward it.
    ``season_regression``
        Fraction of the distance to ``mean`` given back between seasons.
    ``margin_multiplier``
        Scale the update by margin of victory, damped by the favourite's rating
        edge so blowouts by heavy favourites move ratings less.
    ``playoff_k_multiplier``
        Extra weight on postseason games.
    """

    k: float = 20.0
    home_advantage: float = 55.0
    mean: float = 1500.0
    season_regression: float = 1.0 / 3.0
    margin_multiplier: bool = True
    playoff_k_multiplier: float = 1.0


#: Module-level singletons so the dataclasses can be keyword defaults without
#: building a fresh instance on every call. Both are frozen, so sharing is safe.
DEFAULT_ELO = EloConfig()


@dataclass(frozen=True, slots=True)
class RollingConfig:
    """Windows and thresholds for the rolling form block."""

    #: Last-N-game windows. These cross the season boundary deliberately: a
    #: team's last four games in week 2 are two from this season and two from
    #: the end of the last one, which is genuinely what we knew.
    windows: tuple[int, ...] = (4, 8)
    #: Smoothing factor of the exponentially weighted point-differential rating.
    power_alpha: float = 0.25
    #: Yards that make a play "explosive", by play type.
    explosive_pass_yards: int = 20
    explosive_rush_yards: int = 10
    #: Upper bound on seconds attributed to one snap, so halftime and quarter
    #: breaks stay out of the pace estimate.
    max_seconds_per_play: int = 60


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #


def elo_expected(rating: float, opponent_rating: float) -> float:
    """Probability ``rating`` beats ``opponent_rating`` on the logistic Elo curve."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def elo_update(
    home_rating: float,
    away_rating: float,
    home_score: float,
    away_score: float,
    *,
    config: EloConfig = DEFAULT_ELO,
    neutral_site: bool = False,
    playoff: bool = False,
) -> tuple[float, float]:
    """Post-game ``(home, away)`` ratings for one completed game.

    Zero-sum by construction: whatever the home team gains, the away team loses,
    so the league mean is invariant.
    """
    adv = 0.0 if neutral_site else config.home_advantage
    expected_home = elo_expected(home_rating + adv, away_rating)
    margin = home_score - away_score
    actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

    k = config.k * (config.playoff_k_multiplier if playoff else 1.0)
    multiplier = 1.0
    if config.margin_multiplier:
        elo_diff = (home_rating + adv) - away_rating
        # From the winner's perspective: a heavy favourite winning big is less
        # informative than an underdog doing the same.
        winner_diff = elo_diff if actual_home >= 0.5 else -elo_diff
        # A tie carries no margin, but it is still information, so it is scored
        # as a one-point game rather than a zero-weight one.
        mov = max(abs(margin), 1.0)
        multiplier = math.log(mov + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))

    shift = k * multiplier * (actual_home - expected_home)
    return home_rating + shift, away_rating - shift


def compute_elo(games: pl.DataFrame, config: EloConfig = DEFAULT_ELO) -> pl.DataFrame:
    """Pre- and post-game Elo for every game, in kickoff order.

    ``games`` needs ``game_id``, ``season``, ``kickoff``, ``home_franchise``,
    ``away_franchise``, ``home_score``, ``away_score``, ``neutral_site`` and
    ``game_type``. A game with a null score is carried through at its pre-game
    ratings and leaves both alone — scheduled-but-unplayed contributes nothing.

    A forward fold, so it is leak-free by construction: the rating on a row is a
    function only of games that kicked off before it.
    """
    ordered = games.sort(_ORDER)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}

    def rating_of(team: str, season: int) -> float:
        current = ratings.get(team, config.mean)
        seen = last_season.get(team)
        if seen is not None and season > seen:
            # One regression step per season crossed, so a franchise missing
            # from the data for a year does not keep a stale rating.
            for _ in range(season - seen):
                current = config.mean + (1.0 - config.season_regression) * (current - config.mean)
        ratings[team] = current
        last_season[team] = season
        return current

    game_ids: list[str] = []
    home_pre: list[float] = []
    away_pre: list[float] = []
    home_post: list[float] = []
    away_post: list[float] = []

    for row in ordered.select(
        "game_id",
        "season",
        "home_franchise",
        "away_franchise",
        "home_score",
        "away_score",
        "neutral_site",
        "game_type",
    ).iter_rows(named=True):
        season = int(row["season"])
        home, away = row["home_franchise"], row["away_franchise"]
        h_pre = rating_of(home, season)
        a_pre = rating_of(away, season)
        home_points, away_points = row["home_score"], row["away_score"]
        if home_points is None or away_points is None:
            h_post, a_post = h_pre, a_pre
        else:
            h_post, a_post = elo_update(
                h_pre,
                a_pre,
                float(home_points),
                float(away_points),
                config=config,
                neutral_site=bool(row["neutral_site"]),
                playoff=str(row["game_type"] or "REG") != "REG",
            )
            ratings[home] = h_post
            ratings[away] = a_post
        game_ids.append(row["game_id"])
        home_pre.append(h_pre)
        away_pre.append(a_pre)
        home_post.append(h_post)
        away_post.append(a_post)

    return pl.DataFrame(
        {
            "game_id": game_ids,
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "home_elo_post": home_post,
            "away_elo_post": away_post,
        },
        schema={
            "game_id": pl.String,
            "home_elo_pre": pl.Float64,
            "away_elo_pre": pl.Float64,
            "home_elo_post": pl.Float64,
            "away_elo_post": pl.Float64,
        },
    )


# --------------------------------------------------------------------------- #
# Metric registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RateSpec:
    """A rolling rate: ``numerator / denominator``, both summed over the window."""

    name: str
    numerator: str
    denominator: str
    doc: str


#: Per-team-game counters. Everything rolling is a windowed sum of these, so
#: adding a metric means adding a counter here and a :class:`RateSpec` below —
#: never a bespoke window function that might forget the ``shift(1)``.
COUNTERS: tuple[str, ...] = (
    "games",
    "wins",
    "losses",
    "ties",
    "win_points",
    "points_for",
    "points_against",
    "point_diff",
    "off_plays",
    "off_epa",
    "off_pass_plays",
    "off_pass_epa",
    "off_rush_plays",
    "off_rush_epa",
    "off_success",
    "off_explosive",
    "off_third_att",
    "off_third_conv",
    "off_rz_trips",
    "off_rz_td",
    "off_dropbacks",
    "off_sacks_allowed",
    "off_seconds",
    "giveaways",
    "penalty_yards",
    "def_plays",
    "def_epa",
    "def_pass_plays",
    "def_pass_epa",
    "def_rush_plays",
    "def_rush_epa",
    "def_success",
    "def_explosive",
    "def_third_att",
    "def_third_conv",
    "def_rz_trips",
    "def_rz_td",
    "def_dropbacks",
    "def_sacks_made",
    "takeaways",
    "turnover_margin",
    "fg_att",
    "fg_made",
    "punts",
    "punt_net_yards",
    "st_return_epa",
    "st_epa",
    "opp_elo",
)

RATES: tuple[RateSpec, ...] = (
    RateSpec("win_pct", "win_points", "games", "Wins per game played, a tie counting half."),
    RateSpec("points_for_per_game", "points_for", "games", "Points scored per game."),
    RateSpec("points_against_per_game", "points_against", "games", "Points allowed per game."),
    RateSpec("point_diff_per_game", "point_diff", "games", "Average scoring margin."),
    RateSpec("off_epa_per_play", "off_epa", "off_plays", "Offensive EPA per scrimmage play."),
    RateSpec(
        "off_pass_epa_per_play", "off_pass_epa", "off_pass_plays", "Offensive EPA per dropback."
    ),
    RateSpec("off_rush_epa_per_play", "off_rush_epa", "off_rush_plays", "Offensive EPA per rush."),
    RateSpec(
        "off_success_rate",
        "off_success",
        "off_plays",
        "Share of offensive plays with positive EPA.",
    ),
    RateSpec(
        "off_explosive_rate",
        "off_explosive",
        "off_plays",
        "Share of offensive plays gaining 20+ pass / 10+ rush yards.",
    ),
    RateSpec("off_third_down_pct", "off_third_conv", "off_third_att", "Third downs converted."),
    RateSpec(
        "off_red_zone_td_pct",
        "off_rz_td",
        "off_rz_trips",
        "Share of drives reaching the red zone that ended in an offensive touchdown.",
    ),
    RateSpec("off_sack_rate", "off_sacks_allowed", "off_dropbacks", "Sacks taken per dropback."),
    RateSpec(
        "def_epa_per_play",
        "def_epa",
        "def_plays",
        "EPA allowed per scrimmage play (lower is better).",
    ),
    RateSpec(
        "def_pass_epa_per_play",
        "def_pass_epa",
        "def_pass_plays",
        "EPA allowed per opponent dropback.",
    ),
    RateSpec(
        "def_rush_epa_per_play", "def_rush_epa", "def_rush_plays", "EPA allowed per opponent rush."
    ),
    RateSpec(
        "def_success_rate", "def_success", "def_plays", "Share of opponent plays with positive EPA."
    ),
    RateSpec(
        "def_explosive_rate",
        "def_explosive",
        "def_plays",
        "Share of opponent plays that were explosive.",
    ),
    RateSpec(
        "def_third_down_pct",
        "def_third_conv",
        "def_third_att",
        "Opponent third-down conversion rate.",
    ),
    RateSpec(
        "def_red_zone_td_pct", "def_rz_td", "def_rz_trips", "Opponent red-zone touchdown rate."
    ),
    RateSpec(
        "def_sack_rate", "def_sacks_made", "def_dropbacks", "Sacks made per opponent dropback."
    ),
    RateSpec(
        "turnover_margin_per_game",
        "turnover_margin",
        "games",
        "Takeaways minus giveaways per game.",
    ),
    RateSpec(
        "penalty_yards_per_game",
        "penalty_yards",
        "games",
        "Penalty yards charged to the team per game.",
    ),
    RateSpec("plays_per_game", "off_plays", "games", "Offensive scrimmage plays per game."),
    RateSpec(
        "seconds_per_play",
        "off_seconds",
        "off_plays",
        "Neutral-ish pace: seconds of game clock consumed per offensive snap.",
    ),
    RateSpec("fg_pct", "fg_made", "fg_att", "Field goals made per attempt."),
    RateSpec(
        "net_punt_yards", "punt_net_yards", "punts", "Punt distance minus return yards, per punt."
    ),
    RateSpec(
        "return_epa_per_game",
        "st_return_epa",
        "games",
        "EPA gained on kick and punt returns per game.",
    ),
    RateSpec(
        "st_epa_per_game",
        "st_epa",
        "games",
        "Total special-teams EPA (kicking plus returning) per game.",
    ),
    RateSpec(
        "sos_elo_played", "opp_elo", "games", "Mean pre-game Elo of the opponents faced so far."
    ),
)

#: Rates derived from two other rates rather than from a counter pair.
_DERIVED_RATES: tuple[tuple[str, str, str, str], ...] = (
    (
        "net_epa_per_play",
        "off_epa_per_play",
        "def_epa_per_play",
        "Offensive EPA per play minus EPA allowed per play — the composite the overall rank uses.",
    ),
)

#: ``(suffix, description)`` for the four rolling windows.
_WINDOW_DOCS: dict[str, str] = {
    "season": "season to date",
    "l4": "last 4 games (crossing the season boundary)",
    "l8": "last 8 games (crossing the season boundary)",
    "prior": "the whole of the previous season",
}

#: Ranks: ``name -> (rate metric ranked, descending)``. ``descending`` means a
#: larger value earns rank 1, so defensive EPA allowed is ranked ascending.
RANK_SPECS: dict[str, tuple[str, bool]] = {
    "overall": ("net_epa_per_play", True),
    "offense": ("off_epa_per_play", True),
    "defense": ("def_epa_per_play", False),
    "special_teams": ("st_epa_per_game", True),
    "pass_offense": ("off_pass_epa_per_play", True),
    "rush_offense": ("off_rush_epa_per_play", True),
    "pass_defense": ("def_pass_epa_per_play", False),
    "rush_defense": ("def_rush_epa_per_play", False),
    "scoring_offense": ("points_for_per_game", True),
    "scoring_defense": ("points_against_per_game", False),
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _raw_dir(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    from nfl.config import RAW_DIR

    return RAW_DIR


def load_schedules_frame(root: Path | None = None) -> pl.DataFrame:
    """Read the ingested ``nflverse/schedules`` partitions."""
    directory = _raw_dir(root) / "nflverse" / "schedules"
    if not any(directory.rglob("*.parquet")):
        raise FileNotFoundError(
            f"no schedules partitions under {directory}; run "
            f"`nfl ingest nflverse --tables schedules` first"
        )
    return pl.read_parquet(str(directory / "**" / "*.parquet"), hive_partitioning=True)


def load_pbp_frame(
    seasons: Iterable[int] | None = None,
    root: Path | None = None,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read the ingested ``nflverse/pbp`` partitions, projecting only what we need.

    The full frame is 375 columns wide and several GB across 2010-2024; the
    default projection is :data:`PBP_COLUMNS`, which is also the audit trail for
    "no market column can reach this builder" — nothing outside the list is read.
    """
    directory = _raw_dir(root) / "nflverse" / "pbp"
    if not any(directory.rglob("*.parquet")):
        raise FileNotFoundError(
            f"no pbp partitions under {directory}; run `nfl ingest nflverse --tables pbp` first"
        )
    wanted = list(PBP_COLUMNS if columns is None else columns)
    lazy = pl.scan_parquet(str(directory / "**" / "*.parquet"), hive_partitioning=True)
    available = set(lazy.collect_schema().names())
    projection = [c for c in wanted if c in available]
    if seasons is not None:
        lazy = lazy.filter(pl.col("season").is_in(list(seasons)))
    return lazy.select(projection).collect()


def load_weather_frame(root: Path | None = None) -> pl.DataFrame | None:
    """Read ``data/raw/weather`` if it has been ingested, else ``None``."""
    directory = _raw_dir(root) / "weather"
    if not directory.exists() or not any(directory.rglob("weather.parquet")):
        return None
    return pl.read_parquet(str(directory / "**" / "weather.parquet"), hive_partitioning=True)


#: Every play-by-play column this module reads. Explicit so the projection is
#: auditable: no market or third-party win-probability column is on it.
PBP_COLUMNS: tuple[str, ...] = (
    "game_id",
    "play_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "penalty_team",
    "return_team",
    "td_team",
    "fixed_drive",
    "qtr",
    "down",
    "yardline_100",
    "yards_gained",
    "game_seconds_remaining",
    "epa",
    "success",
    "pass",
    "rush",
    "special",
    "sack",
    "touchdown",
    "interception",
    "fumble_lost",
    "third_down_converted",
    "third_down_failed",
    "penalty_yards",
    "field_goal_attempt",
    "field_goal_result",
    "punt_attempt",
    "kick_distance",
    "return_yards",
)

#: The subset of :data:`PBP_COLUMNS` without which nothing can be computed.
#: Everything else degrades to a null feature rather than an exception, but a
#: play-by-play frame with no ``epa`` is a mistake worth failing loudly on.
PBP_REQUIRED_COLUMNS: tuple[str, ...] = ("game_id", "play_id", "posteam", "epa")


# --------------------------------------------------------------------------- #
# Game-level normalisation
# --------------------------------------------------------------------------- #

REQUIRED_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)


def _optional(
    frame: pl.DataFrame, name: str, dtype: PolarsDType, default: object = None
) -> pl.Expr:
    """``pl.col(name)`` when the column exists, a typed literal otherwise."""
    if name in frame.columns:
        return pl.col(name).cast(dtype, strict=False).alias(name)
    return pl.lit(default, dtype=dtype).alias(name)


def _franchise(expr: pl.Expr) -> pl.Expr:
    """Map a source team code onto its franchise identity (:data:`FRANCHISE_ALIASES`)."""
    return expr.replace_strict(FRANCHISE_ALIASES, default=expr)


def _kickoff_expr(frame: pl.DataFrame) -> pl.Expr:
    """``gameday`` + ``gametime`` (US Eastern, DST-aware) as a naive-UTC timestamp.

    ``gametime`` is Eastern wall clock even for games played abroad. A missing or
    malformed value falls back to 13:00 ET, the modal kickoff slot — the same
    convention :func:`nfl.data.weather.kickoff_utc` uses.
    """
    day = pl.col("gameday").cast(pl.String).str.slice(0, 10)
    if "gametime" in frame.columns:
        raw = pl.col("gametime").cast(pl.String)
        time = (
            pl.when(raw.is_null() | (raw.str.len_chars() < 4)).then(pl.lit("13:00")).otherwise(raw)
        )
    else:
        time = pl.lit("13:00")
    return (
        pl.concat_str([day, time], separator=" ")
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone(_SCHEDULE_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .alias("kickoff")
    )


def _normalise_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Game-level frame: kickoff, franchise ids, venue geography, outcome flags."""
    missing = [c for c in REQUIRED_SCHEDULE_COLUMNS if c not in schedules.columns]
    if missing:
        raise ValueError(f"schedules frame is missing required columns: {missing}")

    games = schedules.select(
        pl.col("game_id").cast(pl.String),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("gameday"),
        pl.col("home_team").cast(pl.String),
        pl.col("away_team").cast(pl.String),
        pl.col("home_score").cast(pl.Float64),
        pl.col("away_score").cast(pl.Float64),
        _optional(schedules, "gametime", pl.String),
        _optional(schedules, "game_type", pl.String, "REG"),
        _optional(schedules, "location", pl.String),
        _optional(schedules, "div_game", pl.Int64),
        _optional(schedules, "roof", pl.String),
        _optional(schedules, "surface", pl.String),
        _optional(schedules, "stadium_id", pl.String),
        _optional(schedules, "stadium", pl.String),
        _optional(schedules, "home_coach", pl.String),
        _optional(schedules, "away_coach", pl.String),
    )
    games = games.with_columns(_kickoff_expr(games))
    games = resolve_games(games)

    margin = pl.col("home_score") - pl.col("away_score")
    played = pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    return games.with_columns(
        _franchise(pl.col("home_team")).alias("home_franchise"),
        _franchise(pl.col("away_team")).alias("away_franchise"),
        (pl.col("location").fill_null("Home") == "Neutral").alias("neutral_site"),
        played.alias("played"),
        pl.when(played).then(margin).otherwise(None).alias("margin"),
        pl.col("gameday")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False)
        .alias("game_date"),
    ).sort(_ORDER)


def _to_team_rows(games: pl.DataFrame) -> pl.DataFrame:
    """Explode each game into the two team perspectives."""
    frames = []
    for side in ("home", "away"):
        other = "away" if side == "home" else "home"
        won = pl.col("margin") > 0 if side == "home" else pl.col("margin") < 0
        lost = pl.col("margin") < 0 if side == "home" else pl.col("margin") > 0
        frames.append(
            games.select(
                pl.col("game_id"),
                pl.col("season"),
                pl.col("week"),
                pl.col("game_type"),
                pl.col("kickoff"),
                pl.col("game_date"),
                pl.col("gametime"),
                pl.col(f"{side}_team").alias("team"),
                pl.col(f"{side}_franchise").alias("franchise"),
                pl.col(f"{other}_team").alias("opponent"),
                pl.col(f"{other}_franchise").alias("opponent_franchise"),
                pl.col(f"{side}_coach").alias("coach"),
                pl.lit(side == "home").alias("is_home"),
                pl.col("neutral_site"),
                pl.col("div_game"),
                pl.col("roof"),
                pl.col("surface"),
                pl.col("venue_id"),
                pl.col("latitude"),
                pl.col("longitude"),
                pl.col("elevation_m"),
                pl.col("venue_timezone"),
                pl.col("played"),
                pl.col("margin").alias("game_margin"),
                pl.col(f"{side}_score").alias("points_for"),
                pl.col(f"{other}_score").alias("points_against"),
                pl.col(f"{side}_elo_pre").alias("elo_pre"),
                pl.col(f"{other}_elo_pre").alias("opp_elo_pre"),
                pl.col(f"{side}_elo_post").alias("elo_post"),
                (won & pl.col("played")).cast(pl.Float64).alias("wins"),
                (lost & pl.col("played")).cast(pl.Float64).alias("losses"),
                ((pl.col("margin") == 0) & pl.col("played")).cast(pl.Float64).alias("ties"),
            )
        )
    long = pl.concat(frames, how="vertical")
    diff = pl.col("points_for") - pl.col("points_against")
    return long.with_columns(
        pl.col("played").cast(pl.Float64).alias("games"),
        (pl.col("wins") + 0.5 * pl.col("ties")).alias("win_points"),
        pl.when(pl.col("played")).then(diff).otherwise(0.0).alias("point_diff"),
        pl.when(pl.col("played")).then(diff).otherwise(None).alias("margin"),
        pl.when(pl.col("played")).then(pl.col("opp_elo_pre")).otherwise(0.0).alias("opp_elo"),
        pl.col("points_for").fill_null(0.0),
        pl.col("points_against").fill_null(0.0),
    ).sort(["franchise", *_ORDER])


# --------------------------------------------------------------------------- #
# Play-by-play counters
# --------------------------------------------------------------------------- #


def _pbp_num(frame: pl.DataFrame, name: str) -> pl.Expr:
    """A 0/1 or numeric play-by-play column as a null-free float."""
    if name in frame.columns:
        return pl.col(name).cast(pl.Float64, strict=False).fill_null(0.0)
    return pl.lit(0.0, dtype=pl.Float64)


def _offense_counters(pbp: pl.DataFrame, rolling: RollingConfig) -> pl.DataFrame:
    """Per ``(game_id, team)`` offensive counters, from the possessing team's plays.

    Defensive counters are deliberately *not* computed here: for team *T* in game
    *G* they are exactly the offensive counters of *T*'s opponent in *G*, so the
    caller mirrors this frame across the matchup rather than aggregating twice.
    """
    df = pbp.sort(["game_id", "play_id"])
    is_pass = _pbp_num(df, "pass") == 1
    is_rush = _pbp_num(df, "rush") == 1
    scrimmage = (is_pass | is_rush) & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null()

    # Seconds of game clock burned by a snap: the gap to the next snap in the
    # same game, discarded when it spans a quarter or half break.
    delta = (pl.col("game_seconds_remaining") - pl.col("game_seconds_remaining").shift(-1)).over(
        "game_id"
    )

    df = df.with_columns(
        scrimmage.alias("_scrimmage"),
        is_pass.alias("_pass"),
        is_rush.alias("_rush"),
        (
            (is_pass & (pl.col("yards_gained") >= rolling.explosive_pass_yards))
            | (is_rush & (pl.col("yards_gained") >= rolling.explosive_rush_yards))
        ).alias("_explosive"),
        pl.when(scrimmage & (delta > 0) & (delta <= rolling.max_seconds_per_play))
        .then(delta)
        .otherwise(None)
        .cast(pl.Float64)
        .alias("_seconds"),
        (_pbp_num(df, "third_down_converted") + _pbp_num(df, "third_down_failed")).alias(
            "_third_att"
        ),
        _pbp_num(df, "third_down_converted").alias("_third_conv"),
        _pbp_num(df, "success").alias("_success"),
        _pbp_num(df, "sack").alias("_sack"),
        (_pbp_num(df, "interception") + _pbp_num(df, "fumble_lost")).alias("_giveaway"),
    )

    scrim = pl.col("_scrimmage")
    agg = (
        df.filter(pl.col("posteam").is_not_null())
        .group_by(["game_id", "posteam"])
        .agg(
            pl.col("epa").filter(scrim).count().cast(pl.Float64).alias("o_plays"),
            pl.col("epa").filter(scrim).sum().alias("o_epa"),
            pl.col("epa")
            .filter(scrim & pl.col("_pass"))
            .count()
            .cast(pl.Float64)
            .alias("o_pass_plays"),
            pl.col("epa").filter(scrim & pl.col("_pass")).sum().alias("o_pass_epa"),
            pl.col("epa")
            .filter(scrim & pl.col("_rush"))
            .count()
            .cast(pl.Float64)
            .alias("o_rush_plays"),
            pl.col("epa").filter(scrim & pl.col("_rush")).sum().alias("o_rush_epa"),
            pl.col("_success").filter(scrim).sum().alias("o_success"),
            pl.col("_explosive").filter(scrim).sum().cast(pl.Float64).alias("o_explosive"),
            pl.col("_third_att").filter(scrim).sum().alias("o_third_att"),
            pl.col("_third_conv").filter(scrim).sum().alias("o_third_conv"),
            pl.col("_pass").filter(scrim).sum().cast(pl.Float64).alias("o_dropbacks"),
            pl.col("_sack").filter(scrim).sum().alias("o_sacks_allowed"),
            pl.col("_seconds").sum().cast(pl.Float64).alias("o_seconds"),
            pl.col("_giveaway").sum().alias("o_giveaways"),
        )
    )
    return agg.rename({"posteam": "team"})


def _red_zone_counters(pbp: pl.DataFrame) -> pl.DataFrame:
    """Red-zone *drives*, and the ones that ended in an offensive touchdown.

    Drive-level rather than play-level, and on the *drive's* outcome rather than
    on where the touchdown was scored from. Both choices matter:

    * counted per snap, a red-zone rate rewards a team for stalling inside the
      20 and running four plays there;
    * requiring the scoring play itself to start inside the 20 drops the drive
      that reaches the 18, takes a sack back to the 27 and scores from there.
      That is a red-zone touchdown by every published definition, and excluding
      it puts the league rate around 46% instead of the true ~58%.
    """
    empty = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "team": pl.String,
            "o_rz_trips": pl.Float64,
            "o_rz_td": pl.Float64,
        }
    )
    if "fixed_drive" not in pbp.columns or "yardline_100" not in pbp.columns:
        return empty
    inside = pl.col("yardline_100") <= 20
    scored = (_pbp_num(pbp, "touchdown") == 1) & (
        pl.col("td_team") == pl.col("posteam") if "td_team" in pbp.columns else pl.lit(True)
    )
    drives = (
        pbp.filter(pl.col("posteam").is_not_null())
        .group_by(["game_id", "posteam", "fixed_drive"])
        .agg(inside.any().alias("entered"), scored.any().alias("scored"))
    )
    return (
        drives.group_by(["game_id", "posteam"])
        .agg(
            pl.col("entered").sum().cast(pl.Float64).alias("o_rz_trips"),
            (pl.col("entered") & pl.col("scored")).sum().cast(pl.Float64).alias("o_rz_td"),
        )
        .rename({"posteam": "team"})
    )


def _special_teams_counters(pbp: pl.DataFrame) -> pl.DataFrame:
    """Kicking, punting and return counters per ``(game_id, team)``.

    EPA on a special-teams play is signed from the *kicking* team's point of
    view, so the returning team's share is its negation. Summing both sides gives
    one ``st_epa`` that nets out across a return and still picks up field-goal
    value on top.
    """
    if "special" not in pbp.columns:
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "team": pl.String,
                "fg_att": pl.Float64,
                "fg_made": pl.Float64,
                "punts": pl.Float64,
                "punt_net_yards": pl.Float64,
                "st_return_epa": pl.Float64,
                "st_epa": pl.Float64,
            }
        )
    special = pl.col("special") == 1
    made = (
        (pl.col("field_goal_result") == "made").cast(pl.Float64)
        if "field_goal_result" in pbp.columns
        else pl.lit(0.0, dtype=pl.Float64)
    )
    kicking = (
        pbp.filter(special & pl.col("posteam").is_not_null())
        .group_by(["game_id", "posteam"])
        .agg(
            _pbp_num(pbp, "field_goal_attempt").sum().alias("fg_att"),
            made.sum().alias("fg_made"),
            _pbp_num(pbp, "punt_attempt").sum().alias("punts"),
            (_pbp_num(pbp, "kick_distance") - _pbp_num(pbp, "return_yards"))
            .filter(_pbp_num(pbp, "punt_attempt") == 1)
            .sum()
            .alias("punt_net_yards"),
            pl.col("epa").fill_null(0.0).sum().alias("kick_epa"),
        )
        .rename({"posteam": "team"})
    )
    if "return_team" in pbp.columns:
        returning = (
            pbp.filter(special & pl.col("return_team").is_not_null())
            .group_by(["game_id", "return_team"])
            .agg((-pl.col("epa").fill_null(0.0)).sum().alias("st_return_epa"))
            .rename({"return_team": "team"})
        )
    else:
        returning = pl.DataFrame(
            schema={"game_id": pl.String, "team": pl.String, "st_return_epa": pl.Float64}
        )
    out = kicking.join(returning, on=["game_id", "team"], how="full", coalesce=True)
    return (
        out.with_columns(pl.col("kick_epa").fill_null(0.0), pl.col("st_return_epa").fill_null(0.0))
        .with_columns((pl.col("kick_epa") + pl.col("st_return_epa")).alias("st_epa"))
        .drop("kick_epa")
    )


def _penalty_counters(pbp: pl.DataFrame) -> pl.DataFrame:
    """Penalty yards charged to a team, on offence, defence or special teams."""
    if "penalty_team" not in pbp.columns:
        return pl.DataFrame(
            schema={"game_id": pl.String, "team": pl.String, "penalty_yards": pl.Float64}
        )
    return (
        pbp.filter(pl.col("penalty_team").is_not_null())
        .group_by(["game_id", "penalty_team"])
        .agg(_pbp_num(pbp, "penalty_yards").sum().alias("penalty_yards"))
        .rename({"penalty_team": "team"})
    )


#: Offensive counter -> the defensive counter it becomes when mirrored onto the
#: opponent. This mapping *is* the defensive aggregation.
_MIRROR: dict[str, str] = {
    "o_plays": "def_plays",
    "o_epa": "def_epa",
    "o_pass_plays": "def_pass_plays",
    "o_pass_epa": "def_pass_epa",
    "o_rush_plays": "def_rush_plays",
    "o_rush_epa": "def_rush_epa",
    "o_success": "def_success",
    "o_explosive": "def_explosive",
    "o_third_att": "def_third_att",
    "o_third_conv": "def_third_conv",
    "o_rz_trips": "def_rz_trips",
    "o_rz_td": "def_rz_td",
    "o_dropbacks": "def_dropbacks",
    "o_sacks_allowed": "def_sacks_made",
    "o_giveaways": "takeaways",
}

_OWN: dict[str, str] = {
    "o_plays": "off_plays",
    "o_epa": "off_epa",
    "o_pass_plays": "off_pass_plays",
    "o_pass_epa": "off_pass_epa",
    "o_rush_plays": "off_rush_plays",
    "o_rush_epa": "off_rush_epa",
    "o_success": "off_success",
    "o_explosive": "off_explosive",
    "o_third_att": "off_third_att",
    "o_third_conv": "off_third_conv",
    "o_rz_trips": "off_rz_trips",
    "o_rz_td": "off_rz_td",
    "o_dropbacks": "off_dropbacks",
    "o_sacks_allowed": "off_sacks_allowed",
    "o_seconds": "off_seconds",
    "o_giveaways": "giveaways",
}

#: Counters the schedule supplies directly; everything else comes from pbp.
_SCHEDULE_COUNTERS = frozenset(
    {
        "games",
        "wins",
        "losses",
        "ties",
        "win_points",
        "points_for",
        "points_against",
        "point_diff",
        "opp_elo",
        "turnover_margin",
    }
)


def _attach_pbp_counters(
    long: pl.DataFrame, pbp: pl.DataFrame | None, rolling: RollingConfig
) -> pl.DataFrame:
    """Join own and opponent-mirrored play-by-play counters onto the team rows."""
    wanted = [c for c in COUNTERS if c not in _SCHEDULE_COUNTERS]
    if pbp is not None and pbp.height:
        absent_required = [c for c in PBP_REQUIRED_COLUMNS if c not in pbp.columns]
        if absent_required:
            raise ValueError(
                f"play-by-play frame is missing required columns: {absent_required}. "
                f"See nfl.features.team.PBP_COLUMNS for the full projection."
            )
    if pbp is None or pbp.height == 0:
        out = long.with_columns([pl.lit(0.0, dtype=pl.Float64).alias(c) for c in wanted])
        return out.with_columns(
            (pl.col("takeaways") - pl.col("giveaways")).alias("turnover_margin")
        )

    per_team = _offense_counters(pbp, rolling)
    per_team = per_team.join(_red_zone_counters(pbp), on=["game_id", "team"], how="left")
    per_team = per_team.join(
        _special_teams_counters(pbp), on=["game_id", "team"], how="full", coalesce=True
    )
    per_team = per_team.join(
        _penalty_counters(pbp), on=["game_id", "team"], how="full", coalesce=True
    )
    per_team = per_team.with_columns(
        [
            pl.col(c).cast(pl.Float64).fill_null(0.0)
            for c in per_team.columns
            if c not in ("game_id", "team")
        ]
    )

    st_columns = [
        c
        for c in (
            "penalty_yards",
            "fg_att",
            "fg_made",
            "punts",
            "punt_net_yards",
            "st_return_epa",
            "st_epa",
        )
        if c in per_team.columns
    ]
    own = per_team.select(
        "game_id",
        "team",
        *[pl.col(src).alias(dst) for src, dst in _OWN.items() if src in per_team.columns],
        *st_columns,
    )
    mirrored = per_team.select(
        "game_id",
        pl.col("team").alias("opponent"),
        *[pl.col(src).alias(dst) for src, dst in _MIRROR.items() if src in per_team.columns],
    )
    out = long.join(own, on=["game_id", "team"], how="left").join(
        mirrored, on=["game_id", "opponent"], how="left"
    )
    absent = [c for c in wanted if c not in out.columns]
    if absent:
        out = out.with_columns([pl.lit(0.0, dtype=pl.Float64).alias(c) for c in absent])
    out = out.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in wanted])
    return out.with_columns((pl.col("takeaways") - pl.col("giveaways")).alias("turnover_margin"))


# --------------------------------------------------------------------------- #
# Rolling windows
# --------------------------------------------------------------------------- #


def _rate_exprs(suffix: str) -> list[pl.Expr]:
    """Rate columns for one window. ``suffix=""`` reads the bare counters."""
    sfx = f"_{suffix}" if suffix else ""
    exprs: list[pl.Expr] = []
    for spec in RATES:
        den = pl.col(spec.denominator + sfx)
        num = pl.col(spec.numerator + sfx)
        # A null (not a zero) when the denominator is empty: "we have no idea"
        # and "exactly average" are different things, and GBDTs can tell them
        # apart only if we keep them apart.
        exprs.append(
            pl.when(den > 0).then(num / den).otherwise(None).cast(pl.Float64).alias(spec.name + sfx)
        )
    return exprs


def _derived_rate_exprs(suffix: str) -> list[pl.Expr]:
    sfx = f"_{suffix}" if suffix else ""
    return [
        (pl.col(left + sfx) - pl.col(right + sfx)).cast(pl.Float64).alias(name + sfx)
        for name, left, right, _doc in _DERIVED_RATES
    ]


def window_suffixes(rolling: RollingConfig) -> tuple[str, ...]:
    """``("season", "l4", "l8", "prior")`` for the default configuration."""
    return ("season", *[f"l{w}" for w in rolling.windows], "prior")


def _add_windows(long: pl.DataFrame, rolling: RollingConfig) -> pl.DataFrame:
    """Season-to-date, last-N and prior-season sums of every counter, then rates.

    ``cum_sum().shift(1)`` and ``shift(1).rolling_sum()`` over a franchise-major
    kickoff ordering are what make this as-of: the row's own game is shifted out
    before it can be added in, and nothing downstream can put it back.
    """
    counters = list(COUNTERS)
    totals = long.group_by(["franchise", "season"]).agg(
        [pl.col(c).sum().alias(c) for c in counters]
    )
    prior = totals.with_columns((pl.col("season") + 1).alias("season")).rename(
        {c: f"{c}_prior" for c in counters}
    )

    exprs: list[pl.Expr] = []
    for c in counters:
        exprs.append(
            pl.col(c)
            .cum_sum()
            .shift(1)
            .fill_null(0.0)
            .over(["franchise", "season"], order_by=_ORDER)
            .alias(f"{c}_season")
        )
        for w in rolling.windows:
            exprs.append(
                pl.col(c)
                .shift(1)
                .rolling_sum(w, min_samples=1)
                .over("franchise", order_by=_ORDER)
                .alias(f"{c}_l{w}")
            )
    out = long.with_columns(exprs).join(prior, on=["franchise", "season"], how="left")

    for suffix in window_suffixes(rolling):
        out = out.with_columns(_rate_exprs(suffix))
        out = out.with_columns(_derived_rate_exprs(suffix))
    return out


def _add_ratings(long: pl.DataFrame, rolling: RollingConfig, elo: EloConfig) -> pl.DataFrame:
    """The in-house power rating, plus the Elo columns derived from the fold."""
    out = long.with_columns(
        pl.col("margin")
        .shift(1)
        .ewm_mean(alpha=rolling.power_alpha, ignore_nulls=True)
        .over("franchise", order_by=_ORDER)
        .alias("power_rating"),
        (pl.col("elo_pre") - pl.col("elo_pre").shift(4).over("franchise", order_by=_ORDER)).alias(
            "elo_change_last_4"
        ),
        (pl.col("elo_pre") - pl.col("opp_elo_pre")).alias("elo_diff"),
    )
    advantage = pl.lit(elo.home_advantage, dtype=pl.Float64)
    own_adv = pl.when(pl.col("is_home") & ~pl.col("neutral_site")).then(advantage).otherwise(0.0)
    opp_adv = pl.when(~pl.col("is_home") & ~pl.col("neutral_site")).then(advantage).otherwise(0.0)
    out = out.with_columns(
        (
            1.0
            / (
                1.0
                + (
                    10.0
                    ** (((pl.col("opp_elo_pre") + opp_adv) - (pl.col("elo_pre") + own_adv)) / 400.0)
                )
            )
        ).alias("elo_win_prob")
    )
    opponent_view = out.select(
        "game_id",
        pl.col("team").alias("opponent"),
        pl.col("power_rating").alias("power_rating_opp"),
    )
    out = out.join(opponent_view, on=["game_id", "opponent"], how="left")
    return out.with_columns(
        (pl.col("power_rating") - pl.col("power_rating_opp")).alias("power_rating_diff")
    )


# --------------------------------------------------------------------------- #
# League ranks
# --------------------------------------------------------------------------- #

RANK_COLUMNS: tuple[str, ...] = tuple(f"rank_{name}" for name in RANK_SPECS)


def _add_ranks(long: pl.DataFrame) -> pl.DataFrame:
    """1-32 league ranks as of the week of the game.

    Ranking the teams that happen to play in week ``w`` would give a 1-to-14
    scale on a bye-heavy week and would silently change meaning week to week, so
    the ranks come from a state grid instead: every franchise in the season gets
    a row for every week, and its state is fetched with a ``join_asof`` on
    ``week + 1`` — which is exactly "cumulative through games in weeks < w".
    """
    counters = list(COUNTERS)
    state = (
        long.with_columns(
            [
                pl.col(c).cum_sum().over(["franchise", "season"], order_by=_ORDER).alias(c)
                for c in counters
            ]
        )
        .select(["franchise", "season", "week", *counters])
        .with_columns((pl.col("week") + 1).cast(pl.Int64).alias("avail_week"))
        .group_by(["franchise", "season", "avail_week"])
        .agg([pl.col(c).max() for c in counters])
    )
    weeks = long.select("season", pl.col("week").cast(pl.Int64)).unique()
    franchises = long.select("franchise", "season").unique()
    grid = (
        franchises.join(weeks, on="season", how="inner")
        .join(
            state,
            left_on=["franchise", "season", "week"],
            right_on=["franchise", "season", "avail_week"],
            how="left",
        )
        # A bye week has no state row of its own, so it inherits the last one —
        # which is precisely "cumulative through every game before this week".
        .with_columns(
            [
                pl.col(c).forward_fill().over(["franchise", "season"], order_by="week")
                for c in counters
            ]
        )
        .with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in counters])
    )
    grid = grid.with_columns(_rate_exprs("")).with_columns(_derived_rate_exprs(""))
    grid = grid.with_columns(
        [
            pl.col(metric)
            .rank("min", descending=descending)
            .cast(pl.Int32)
            .over(["season", "week"])
            .alias(f"rank_{name}")
            for name, (metric, descending) in RANK_SPECS.items()
        ]
    )
    return long.join(
        grid.select(["franchise", "season", "week", *RANK_COLUMNS]),
        on=["franchise", "season", "week"],
        how="left",
    )


# --------------------------------------------------------------------------- #
# Schedule, travel and environment context
# --------------------------------------------------------------------------- #


def _haversine_miles(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    """Great-circle distance in statute miles."""
    dlat = (lat2 - lat1).radians()
    dlon = (lon2 - lon1).radians()
    a = (dlat / 2).sin() ** 2 + lat1.radians().cos() * lat2.radians().cos() * (
        (dlon / 2).sin() ** 2
    )
    return (2 * _EARTH_RADIUS_MILES * a.sqrt().arcsin()).alias("travel_miles")


def _utc_offsets(
    kickoffs: Sequence[datetime | None], zones: Sequence[str | None]
) -> list[float | None]:
    """UTC offset in hours for each ``(naive-UTC kickoff, IANA zone)`` pair."""
    cache: dict[tuple[str, int], float] = {}
    out: list[float | None] = []
    for moment, zone in zip(kickoffs, zones, strict=True):
        if moment is None or not zone:
            out.append(None)
            continue
        key = (zone, int(moment.timestamp() // 86400))
        offset = cache.get(key)
        if offset is None:
            aware = moment.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(zone))
            delta = aware.utcoffset()
            offset = 0.0 if delta is None else delta.total_seconds() / 3600.0
            cache[key] = offset
        out.append(offset)
    return out


def _add_context(long: pl.DataFrame, rolling: RollingConfig) -> pl.DataFrame:
    """Rest, travel, timezone, kickoff slot and the home/away/neutral flags."""
    ordered = long.sort(["franchise", *_ORDER])
    prev_season = pl.col("season").shift(1).over("franchise", order_by=_ORDER)
    prev_game_date = pl.col("game_date").shift(1).over("franchise", order_by=_ORDER)
    same_season = prev_season == pl.col("season")

    def previous_played(column: str) -> pl.Expr:
        return pl.col(column).shift(1).over("franchise", order_by=_ORDER)

    def last_hosted(column: str) -> pl.Expr:
        # The venue the franchise most recently *hosted* in, carried forward.
        # Used as the origin for the first game of a season, when "where were
        # they last week" has no answer inside the season.
        hosted = (
            pl.when(pl.col("is_home") & ~pl.col("neutral_site"))
            .then(pl.col(column))
            .otherwise(None)
        )
        return hosted.shift(1).forward_fill().over("franchise", order_by=_ORDER)

    ordered = ordered.with_columns(
        # Calendar days, not clock hours: a 20:20 Thursday game after a 13:00
        # Sunday one is four days of rest, not 4.3.
        pl.when(same_season)
        .then((pl.col("game_date") - prev_game_date).dt.total_days())
        .otherwise(None)
        .cast(pl.Int64)
        .alias("rest_days"),
        *[
            pl.when(same_season)
            .then(previous_played(col))
            .otherwise(last_hosted(col))
            .alias(f"origin_{col}")
            for col in ("latitude", "longitude", "elevation_m", "venue_timezone")
        ],
    )
    ordered = ordered.with_columns(
        _haversine_miles(
            pl.col("origin_latitude"),
            pl.col("origin_longitude"),
            pl.col("latitude"),
            pl.col("longitude"),
        ),
        ((pl.col("elevation_m") - pl.col("origin_elevation_m")) * 3.280839895).alias(
            "altitude_change_feet"
        ),
        (pl.col("rest_days") <= 5).alias("short_week"),
        (pl.col("rest_days") >= 13).alias("bye_last_week"),
        (~pl.col("is_home")).cast(pl.Int64).alias("_away_flag"),
    )
    ordered = ordered.with_columns(
        pl.col("is_home")
        .cast(pl.Int64)
        .cum_sum()
        .over("franchise", order_by=_ORDER)
        .alias("_home_runs")
    )
    ordered = ordered.with_columns(
        pl.col("_away_flag")
        .cum_sum()
        .over(["franchise", "_home_runs"], order_by=_ORDER)
        .alias("consecutive_road_games"),
        pl.col("travel_miles")
        .fill_null(0.0)
        .rolling_sum_by("kickoff", "14d")
        .over("franchise")
        .alias("travel_miles_last_14_days"),
    )

    offsets = _utc_offsets(
        ordered.get_column("kickoff").to_list(), ordered.get_column("venue_timezone").to_list()
    )
    origin_offsets = _utc_offsets(
        ordered.get_column("kickoff").to_list(),
        ordered.get_column("origin_venue_timezone").to_list(),
    )
    ordered = ordered.with_columns(
        pl.Series("_venue_utc_offset", offsets, dtype=pl.Float64),
        pl.Series("_origin_utc_offset", origin_offsets, dtype=pl.Float64),
    )
    kickoff_local = pl.col("kickoff") + pl.duration(
        seconds=(pl.col("_venue_utc_offset") * 3600).cast(pl.Int64)
    )
    et_hour = pl.col("gametime").cast(pl.String).str.slice(0, 2).cast(pl.Int64, strict=False)
    ordered = ordered.with_columns(
        (pl.col("_venue_utc_offset") - pl.col("_origin_utc_offset")).alias("timezone_shift_hours"),
        kickoff_local.dt.hour().alias("kickoff_hour_local"),
        et_hour.alias("kickoff_hour_et"),
        (et_hour >= 18).alias("is_primetime"),
        pl.col("div_game").cast(pl.Boolean).alias("is_division_game"),
    )

    opponent_rest = ordered.select(
        "game_id", pl.col("team").alias("opponent"), pl.col("rest_days").alias("opp_rest_days")
    )
    ordered = ordered.join(opponent_rest, on=["game_id", "opponent"], how="left")
    return ordered.with_columns(
        (pl.col("rest_days") - pl.col("opp_rest_days")).alias("rest_advantage")
    )


# --------------------------------------------------------------------------- #
# Coach and weather joins
# --------------------------------------------------------------------------- #

_COACH_METRICS: tuple[str, ...] = (
    "career_win_pct",
    "career_games",
    "career_wins",
    "career_losses",
    "season_win_pct",
    "season_games",
)


def _add_coach(long: pl.DataFrame, coach_records: pl.DataFrame | None) -> pl.DataFrame:
    """Join per-side coach records and count the coach's tenure with this team."""
    tenure = (
        pl.when(pl.col("coach").is_not_null())
        .then(1)
        .otherwise(0)
        .cum_sum()
        .shift(1)
        .fill_null(0)
        .over(["franchise", "coach"], order_by=_ORDER)
        .cast(pl.Int64)
        .alias("coach_tenure_games")
    )
    out = long.with_columns(tenure)

    if coach_records is None:
        return out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(f"coach_{m}") for m in _COACH_METRICS]
        )

    frames = []
    for side in ("home", "away"):
        available = [m for m in _COACH_METRICS if f"{side}_coach_{m}" in coach_records.columns]
        frames.append(
            coach_records.select(
                pl.col("game_id"),
                pl.col(f"{side}_team").alias("team"),
                *[
                    pl.col(f"{side}_coach_{m}").cast(pl.Float64).alias(f"coach_{m}")
                    for m in available
                ],
            )
        )
    joined = pl.concat(frames, how="vertical")
    out = out.join(joined, on=["game_id", "team"], how="left")
    absent = [m for m in _COACH_METRICS if f"coach_{m}" not in out.columns]
    if absent:
        out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(f"coach_{m}") for m in absent])
    return out


#: Weather columns lifted onto the team row, ``source -> tm__wx_*`` suffix.
_WEATHER_COLUMNS: dict[str, tuple[str, PolarsDType]] = {
    "temperature_f": ("wx_temperature_f", pl.Float64),
    "feels_like_f": ("wx_feels_like_f", pl.Float64),
    "humidity_pct": ("wx_humidity_pct", pl.Float64),
    "wind_speed_mph": ("wx_wind_speed_mph", pl.Float64),
    "wind_gust_mph": ("wx_wind_gust_mph", pl.Float64),
    "precipitation_mm": ("wx_precipitation_mm", pl.Float64),
    "snow_depth_cm": ("wx_snow_depth_cm", pl.Float64),
    "is_indoor": ("wx_is_indoor", pl.Boolean),
    "severe_weather_flag": ("wx_severe_flag", pl.Boolean),
    "kicking_difficulty_index": ("wx_kicking_difficulty", pl.Float64),
    "passing_difficulty_index": ("wx_passing_difficulty", pl.Float64),
}


def _add_weather(long: pl.DataFrame, weather: pl.DataFrame | None) -> pl.DataFrame:
    """Join the pre-kickoff weather summary, or all-null columns when absent."""
    if weather is None or weather.height == 0:
        return long.with_columns(
            [pl.lit(None, dtype=dtype).alias(alias) for alias, dtype in _WEATHER_COLUMNS.values()]
        )
    available = {src: spec for src, spec in _WEATHER_COLUMNS.items() if src in weather.columns}
    frame = weather.select(
        pl.col("game_id"),
        *[pl.col(src).cast(dtype).alias(alias) for src, (alias, dtype) in available.items()],
    ).unique(subset=["game_id"], keep="first")
    out = long.join(frame, on="game_id", how="left")
    absent = [spec for src, spec in _WEATHER_COLUMNS.items() if src not in available]
    if absent:
        out = out.with_columns([pl.lit(None, dtype=dtype).alias(alias) for alias, dtype in absent])
    return out


# --------------------------------------------------------------------------- #
# Output shape
# --------------------------------------------------------------------------- #

#: ``internal name -> (output name without the prefix, doc)`` for everything that
#: is not part of the windowed rate family.
_SCALARS: tuple[tuple[str, str, str], ...] = (
    ("season", "season", "Season the game belongs to."),
    ("week", "week", "Week number within the season."),
    ("game_type", "game_type", "REG / WC / DIV / CON / SB."),
    (
        "franchise",
        "franchise",
        "Franchise identity, stable across relocations and abbreviation changes.",
    ),
    ("opponent", "opponent", "Opponent's team code, as it appears in schedules."),
    ("is_home", "is_home", "True when this team is the designated home team."),
    (
        "neutral_site",
        "is_neutral_site",
        "True when the game is at a neutral site (no home-field Elo bonus).",
    ),
    ("is_division_game", "is_division_game", "True when the two teams share a division."),
    ("roof", "roof", "Per-game roof state from schedules: dome / outdoors / closed / open."),
    ("surface", "surface", "Playing surface from schedules."),
    (
        "venue_id",
        "venue_id",
        "Resolved venue id (see nfl.data.venues, which corrects neutral-site tagging).",
    ),
    ("elo_pre", "elo_pre", "This team's Elo rating before kickoff."),
    ("opp_elo_pre", "elo_opp_pre", "Opponent's Elo rating before kickoff."),
    (
        "elo_diff",
        "elo_diff",
        "tm__elo_pre minus tm__elo_opp_pre; home-field is not folded in here.",
    ),
    (
        "elo_win_prob",
        "elo_win_prob",
        "Our own Elo's pre-game win probability, home-field adjusted. Sums to 1 across the two rows of a game.",
    ),
    (
        "elo_change_last_4",
        "elo_change_last_4",
        "Elo movement over this team's previous four games — recent form as the rating sees it.",
    ),
    (
        "power_rating",
        "power_rating",
        "In-house rating: exponentially weighted mean of prior scoring margin.",
    ),
    ("power_rating_opp", "power_rating_opp", "The opponent's power rating going into this game."),
    ("power_rating_diff", "power_rating_diff", "Power rating minus the opponent's."),
    ("games_season", "games_played", "Completed games this season before this one."),
    ("wins_season", "wins_to_date", "Wins this season before this game."),
    ("losses_season", "losses_to_date", "Losses this season before this game."),
    ("ties_season", "ties_to_date", "Ties this season before this game."),
    (
        "rest_days",
        "rest_days",
        "Days between the previous game's kickoff and this one; null for a team's first game of the season.",
    ),
    ("opp_rest_days", "opp_rest_days", "The same for the opponent."),
    ("rest_advantage", "rest_advantage", "tm__rest_days minus tm__opp_rest_days."),
    ("short_week", "short_week", "Rest of 5 days or fewer — the Thursday-game flag."),
    (
        "bye_last_week",
        "bye_last_week",
        "Rest of 13 days or more, i.e. a bye (or equivalent) immediately before.",
    ),
    (
        "consecutive_road_games",
        "consecutive_road_games",
        "Length of the current road trip, including this game.",
    ),
    (
        "travel_miles",
        "travel_miles",
        "Great-circle miles from the venue the team last played in (or last hosted in, at the start of a season).",
    ),
    (
        "travel_miles_last_14_days",
        "travel_miles_last_14_days",
        "Sum of tm__travel_miles over trips in the trailing 14 days, this one included.",
    ),
    (
        "timezone_shift_hours",
        "timezone_shift_hours",
        "Signed UTC-offset change from the origin venue to this one, at this kickoff.",
    ),
    (
        "altitude_change_feet",
        "altitude_change_feet",
        "Venue elevation minus the origin venue's, in feet. Denver.",
    ),
    (
        "kickoff_hour_local",
        "kickoff_hour_local",
        "Kickoff hour in the venue's local timezone, 0-23.",
    ),
    (
        "kickoff_hour_et",
        "kickoff_hour_et",
        "Kickoff hour on the Eastern clock the league schedules in.",
    ),
    (
        "is_primetime",
        "is_primetime",
        "Eastern kickoff at 18:00 or later — the national-window flag.",
    ),
    (
        "coach_tenure_games",
        "coach_tenure_games",
        "Games this head coach has coached this franchise before this one.",
    ),
    (
        "coach_career_win_pct",
        "coach_career_win_pct",
        "Head coach's career win percentage, excluding this game (nfl.data.coach_records).",
    ),
    ("coach_career_games", "coach_career_games", "Head coach's career games, excluding this game."),
    ("coach_career_wins", "coach_career_wins", "Head coach's career wins, excluding this game."),
    (
        "coach_career_losses",
        "coach_career_losses",
        "Head coach's career losses, excluding this game.",
    ),
    (
        "coach_season_win_pct",
        "coach_season_win_pct",
        "Head coach's win percentage this season, excluding this game.",
    ),
    (
        "coach_season_games",
        "coach_season_games",
        "Head coach's games this season, excluding this game.",
    ),
)

_WEATHER_DOCS: dict[str, str] = {
    "wx_temperature_f": "Temperature at kickoff. Null when weather has not been ingested.",
    "wx_feels_like_f": "Apparent temperature at kickoff.",
    "wx_humidity_pct": "Relative humidity at kickoff.",
    "wx_wind_speed_mph": "Sustained wind at kickoff.",
    "wx_wind_gust_mph": "Gust speed at kickoff.",
    "wx_precipitation_mm": "Precipitation in the kickoff hour.",
    "wx_snow_depth_cm": "Snow on the ground at kickoff.",
    "wx_is_indoor": "True when the roof neutralises weather entirely.",
    "wx_severe_flag": "Composite severe-conditions flag from nfl.data.weather.",
    "wx_kicking_difficulty": "Wind/temperature/precipitation composite, kicking.",
    "wx_passing_difficulty": "The same, oriented at the passing game.",
}

_RANK_DOCS: dict[str, str] = {
    "rank_overall": "1-32 league rank on net EPA per play, as of this week.",
    "rank_offense": "1-32 rank on offensive EPA per play.",
    "rank_defense": "1-32 rank on EPA allowed per play (1 = fewest allowed).",
    "rank_special_teams": "1-32 rank on special-teams EPA per game.",
    "rank_pass_offense": "1-32 rank on offensive EPA per dropback.",
    "rank_rush_offense": "1-32 rank on offensive EPA per rush.",
    "rank_pass_defense": "1-32 rank on EPA allowed per opponent dropback.",
    "rank_rush_defense": "1-32 rank on EPA allowed per opponent rush.",
    "rank_scoring_offense": "1-32 rank on points scored per game.",
    "rank_scoring_defense": "1-32 rank on points allowed per game.",
}


def _output_plan(rolling: RollingConfig) -> list[tuple[str, str, str]]:
    """``(source column, output column, doc)`` for the whole feature set."""
    plan: list[tuple[str, str, str]] = [
        (src, f"{PREFIX}{name}", doc) for src, name, doc in _SCALARS
    ]
    for name in RANK_COLUMNS:
        plan.append((name, f"{PREFIX}{name}", _RANK_DOCS[name]))
    for alias, _dtype in _WEATHER_COLUMNS.values():
        plan.append((alias, f"{PREFIX}{alias}", _WEATHER_DOCS[alias]))
    for suffix in window_suffixes(rolling):
        window = _WINDOW_DOCS.get(suffix, suffix)
        for spec in RATES:
            src = f"{spec.name}_{suffix}"
            plan.append((src, f"{PREFIX}{src}", f"{spec.doc} Measured over {window}."))
        for name, _left, _right, doc in _DERIVED_RATES:
            src = f"{name}_{suffix}"
            plan.append((src, f"{PREFIX}{src}", f"{doc} Measured over {window}."))
    return plan


#: Every emitted feature, documented. Keys are the full ``tm__`` column names.
FEATURE_DOCS: dict[str, str] = {out: doc for _src, out, doc in _output_plan(RollingConfig())}


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_team_features(
    schedules: pl.DataFrame,
    pbp: pl.DataFrame | None = None,
    *,
    coach_records: pl.DataFrame | None = None,
    weather: pl.DataFrame | None = None,
    elo: EloConfig | None = None,
    rolling: RollingConfig | None = None,
) -> pl.DataFrame:
    """One row per ``(game_id, team)`` of strictly as-of-kickoff team features.

    ``schedules``
        The nflverse games frame. Pass **every** season you have: Elo, career
        coach records and the prior-season block are all cumulative, so slicing
        the input to one season silently restarts them.
    ``pbp``
        Play-by-play, at least the columns in :data:`PBP_COLUMNS`. Optional —
        without it the rolling efficiency block comes back null and the
        schedule/rating/context blocks are unaffected.
    ``coach_records``
        Output of :func:`nfl.data.coach_records.build_coach_records`. Built from
        ``schedules`` automatically when omitted and the coach columns are there.
    ``weather``
        Output of the weather ingest. Omitted or ``None`` leaves ``tm__wx_*``
        null, which is the normal state before ``nfl ingest weather`` has run.

    Every returned column outside the two key columns carries the ``tm__``
    prefix and is documented in :data:`FEATURE_DOCS`.
    """
    elo = elo or EloConfig()
    rolling = rolling or RollingConfig()

    games = _normalise_games(schedules)
    games = games.join(compute_elo(games, elo), on="game_id", how="left")

    if coach_records is None and {"home_coach", "away_coach"} <= set(schedules.columns):
        from nfl.data.coach_records import build_coach_records

        try:
            coach_records = build_coach_records(schedules)
        except ValueError:  # schedules lacks a column coach_records needs
            coach_records = None

    long = _to_team_rows(games)
    long = _attach_pbp_counters(long, pbp, rolling)
    long = _add_windows(long, rolling)
    long = _add_ratings(long, rolling, elo)
    long = _add_ranks(long)
    long = _add_context(long, rolling)
    long = _add_coach(long, coach_records)
    long = _add_weather(long, weather)

    plan = _output_plan(rolling)
    missing = [src for src, _out, _doc in plan if src not in long.columns]
    if missing:  # pragma: no cover - a registry/implementation mismatch
        raise RuntimeError(f"feature plan references columns that were never built: {missing}")

    out = long.select(
        pl.col("game_id"),
        pl.col("team"),
        *[pl.col(src).alias(name) for src, name, _doc in plan],
    ).sort(["game_id", "team"])
    assert_no_market_columns(out, context="nfl.features.team")
    return out
