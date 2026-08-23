"""Feed-agnostic contract for in-game snapshots.

Two adapters implement :class:`LiveFeed`:

* :mod:`nfl.data.live.espn` -- the live wire (ESPN site + core APIs).
* :mod:`nfl.data.live.replay` -- nflverse play-by-play replayed at a chosen
  ``(quarter, clock)``.

They exist as a pair on purpose. Everything downstream of ``/predict`` sees a
:class:`LiveSnapshot` and nothing else, so a model trained on replayed history
is served exactly the field semantics it was trained on. That parity is
enforced by ``tests/test_live_parity.py``; if you add a field here, populate it
in *both* adapters or leave it ``None`` in both.

Nothing in this module is a feature. :class:`LiveSnapshot` is the raw
observation; ``nfl/features`` turns it into a
:class:`~nfl.data.schemas.GameState`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date as date_type
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from nfl.config import RAW_DIR

__all__ = [
    "ESPN_TEAM_ID_TO_ABBR",
    "REGULATION_PERIODS",
    "AtTime",
    "GameRef",
    "LiveDrive",
    "LiveFeed",
    "LiveInjury",
    "LivePlay",
    "LiveSnapshot",
    "LiveVenue",
    "LiveWeather",
    "TeamRecord",
    "clock_to_seconds",
    "espn_to_nflverse_team",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "load_schedule_frame",
    "nflverse_to_espn_team",
    "parse_at",
    "read_parquet_dir",
    "resolve_espn_event_id",
    "seconds_to_clock",
    "truncate_snapshot",
]

REGULATION_PERIODS = 4
PERIOD_SECONDS = 900


# --------------------------------------------------------------------------- #
# Team identity
# --------------------------------------------------------------------------- #

#: ESPN numeric team id -> ESPN abbreviation. Core-API ``$ref``s identify teams
#: only by id, so this is the only way to join them to anything else.
ESPN_TEAM_ID_TO_ABBR: dict[str, str] = {
    "1": "ATL",
    "2": "BUF",
    "3": "CHI",
    "4": "CIN",
    "5": "CLE",
    "6": "DAL",
    "7": "DEN",
    "8": "DET",
    "9": "GB",
    "10": "TEN",
    "11": "IND",
    "12": "KC",
    "13": "LV",
    "14": "LAR",
    "15": "MIA",
    "16": "MIN",
    "17": "NE",
    "18": "NO",
    "19": "NYG",
    "20": "NYJ",
    "21": "PHI",
    "22": "ARI",
    "23": "PIT",
    "24": "LAC",
    "25": "SF",
    "26": "SEA",
    "27": "TB",
    "28": "WSH",
    "29": "CAR",
    "30": "JAX",
    "33": "BAL",
    "34": "HOU",
}

#: The only two abbreviations where ESPN and nflverse disagree, verified across
#: every season 2002-2026 by diffing ``nfldata/data/teams.csv``.
_ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
_NFLVERSE_TO_ESPN = {v: k for k, v in _ESPN_TO_NFLVERSE.items()}


def espn_to_nflverse_team(abbr: str | None) -> str | None:
    """Map an ESPN abbreviation to the nflverse one (``WSH``->``WAS``, ``LAR``->``LA``)."""
    if abbr is None:
        return None
    return _ESPN_TO_NFLVERSE.get(abbr, abbr)


def nflverse_to_espn_team(abbr: str | None) -> str | None:
    """Map an nflverse abbreviation to the ESPN one (``WAS``->``WSH``, ``LA``->``LAR``)."""
    if abbr is None:
        return None
    return _NFLVERSE_TO_ESPN.get(abbr, abbr)


def espn_team_abbr_from_id(team_id: str | int | None) -> str | None:
    """Resolve an ESPN numeric team id to an nflverse abbreviation."""
    if team_id is None:
        return None
    return espn_to_nflverse_team(ESPN_TEAM_ID_TO_ABBR.get(str(team_id)))


# --------------------------------------------------------------------------- #
# Clock helpers
# --------------------------------------------------------------------------- #


def clock_to_seconds(display: str | None) -> int | None:
    """``"5:06"`` -> ``306``. Tolerates ``"05:06"``, ``"0:04.5"`` and ``None``."""
    if display is None:
        return None
    text = str(display).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return int(float(parts[0]))
        minutes = int(parts[0])
        seconds = float(parts[1])
    except ValueError:
        return None
    return minutes * 60 + int(seconds)


def seconds_to_clock(seconds: int | None) -> str:
    """``306`` -> ``"5:06"``."""
    if seconds is None:
        return "--:--"
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def half_seconds_remaining(period: int, quarter_seconds: int) -> int:
    """Seconds left in the current half. Overtime has no second half."""
    if period > REGULATION_PERIODS:
        return quarter_seconds
    if period in (1, 3):
        return quarter_seconds + PERIOD_SECONDS
    return quarter_seconds


def game_seconds_remaining(period: int, quarter_seconds: int) -> int:
    """Seconds left in regulation. In overtime this collapses to the period clock,
    matching nflverse's ``game_seconds_remaining``."""
    if period > REGULATION_PERIODS:
        return quarter_seconds
    return quarter_seconds + PERIOD_SECONDS * (REGULATION_PERIODS - period)


# --------------------------------------------------------------------------- #
# Snapshot time ("--at")
# --------------------------------------------------------------------------- #


class AtTime(BaseModel):
    """A point in game time: ``(period, seconds remaining in that period)``."""

    model_config = ConfigDict(frozen=True)

    quarter: int = Field(ge=1, le=10)
    clock_seconds: int = Field(ge=0, le=PERIOD_SECONDS)

    @property
    def order_key(self) -> tuple[int, int]:
        """Monotonic in game time. Period lengths differ (OT is 10 or 15 minutes),
        so compare ``(period, -clock)`` rather than elapsed seconds."""
        return (self.quarter, -self.clock_seconds)

    def __str__(self) -> str:
        label = "OT" if self.quarter > REGULATION_PERIODS else f"Q{self.quarter}"
        return f"{label} {seconds_to_clock(self.clock_seconds)}"


_AT_RE = re.compile(
    r"^\s*(?:(?P<ot>OT)(?P<otnum>\d*)|Q?(?P<q>\d+))\s*(?:[\s,]\s*(?P<clock>[\d:.]+))?\s*$",
    re.IGNORECASE,
)


def parse_at(value: str | AtTime | None) -> AtTime | None:
    """Parse ``"Q4 05:00"`` / ``"4 5:00"`` / ``"OT 10:00"`` / ``"Q4"``.

    ``None`` means *latest* -- the feed's current state, untruncated.
    """
    if value is None or isinstance(value, AtTime):
        return value
    match = _AT_RE.match(value)
    if match is None:
        raise ValueError(f"cannot parse --at {value!r}; expected e.g. 'Q4 05:00' or 'OT 10:00'")
    if match.group("ot"):
        quarter = REGULATION_PERIODS + int(match.group("otnum") or 1)
    else:
        quarter = int(match.group("q"))
    clock = match.group("clock")
    if clock is None:
        seconds = PERIOD_SECONDS if quarter <= REGULATION_PERIODS else 600
    else:
        seconds = clock_to_seconds(clock) or 0
    return AtTime(quarter=quarter, clock_seconds=min(seconds, PERIOD_SECONDS))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class LiveBase(BaseModel):
    """Immutable, no silent extra fields -- same posture as ``nfl.data.schemas``."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class GameRef(LiveBase):
    """Enough to address one game in either feed."""

    espn_event_id: str | None = None
    nflverse_game_id: str | None = None
    season: int | None = None
    season_type: str | None = None
    week: int | None = None
    game_date: date_type | None = None
    kickoff_utc: datetime | None = None
    home_team: str | None = None
    away_team: str | None = None
    state: str | None = Field(default=None, description="'pre' | 'in' | 'post'.")
    completed: bool | None = None
    short_name: str | None = None
    venue_name: str | None = None
    neutral_site: bool = False

    @property
    def label(self) -> str:
        return self.nflverse_game_id or self.short_name or self.espn_event_id or "<unknown game>"


class TeamRecord(LiveBase):
    """Season record as of this game."""

    team: str
    overall: str | None = None
    home: str | None = None
    road: str | None = None
    wins: int | None = None
    losses: int | None = None
    ties: int | None = None


class LiveVenue(LiveBase):
    venue_id: str | None = None
    name: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    indoor: bool | None = None
    grass: bool | None = None
    roof: str | None = Field(default=None, description="nflverse per-game roof, when known.")
    surface: str | None = None


class LiveWeather(LiveBase):
    """Whatever the feed publishes. Never fabricated: dome games carry ``None``."""

    display_value: str | None = None
    temperature_f: int | None = None
    high_temperature_f: int | None = None
    wind_mph: int | None = None
    condition_id: str | None = None


class LiveInjury(LiveBase):
    """One injury attached to this game."""

    team: str
    player_name: str
    player_id: str | None = None
    position: str | None = None
    status: str | None = None
    detail: str | None = None
    comment: str | None = None
    reported_at: datetime | None = None


class LivePlay(LiveBase):
    """One play. Scores are *after* the play; down/distance are *before* it.

    That split is what both adapters agree on: ESPN publishes post-play scores
    on the play object and pre-snap ``start`` fields, and nflverse's
    ``total_*_score`` / ``down`` / ``ydstogo`` columns line up the same way.
    """

    play_id: str
    sequence: int | None = None
    drive_number: int | None = None
    period: int
    clock_seconds: int | None = None
    clock_display: str | None = None
    wallclock: datetime | None = Field(
        default=None, description="Real-world timestamp; drives the staleness check."
    )
    team: str | None = Field(default=None, description="Offense (nflverse abbreviation).")
    play_type: str | None = None
    text: str | None = None
    home_score: int | None = Field(default=None, description="Home score AFTER this play.")
    away_score: int | None = Field(default=None, description="Away score AFTER this play.")
    scoring_play: bool = False
    is_penalty: bool = False
    is_turnover: bool = False
    stat_yardage: int | None = None
    start_down: int | None = None
    start_distance: int | None = None
    start_yards_to_endzone: int | None = None
    end_yards_to_endzone: int | None = None
    home_timeouts_remaining: int | None = Field(
        default=None, description="Pre-snap; ESPN does not publish per-play timeouts."
    )
    away_timeouts_remaining: int | None = None

    @property
    def order_key(self) -> tuple[int, int]:
        return (self.period, -(self.clock_seconds if self.clock_seconds is not None else 0))


class LiveDrive(LiveBase):
    drive_number: int | None = None
    team: str | None = None
    description: str | None = None
    result: str | None = None
    display_result: str | None = None
    yards: int | None = None
    offensive_plays: int | None = None
    is_score: bool = False
    start_period: int | None = None
    start_clock_display: str | None = None
    end_period: int | None = None
    end_clock_display: str | None = None
    time_elapsed: str | None = None
    plays: list[LivePlay] = Field(default_factory=list)


class LiveSnapshot(LiveBase):
    """The complete observation of a game at one moment.

    ``at`` is set when the snapshot was truncated to a requested game time; the
    reported ``period``/``clock_seconds`` are then exactly the requested ones and
    ``down``/``distance``/``yards_to_endzone`` describe the snap that is *about*
    to happen.
    """

    source: str
    fetched_at: datetime
    at: AtTime | None = None

    # Identity
    game_id: str | None = None
    espn_event_id: str | None = None
    season: int | None = None
    season_type: str | None = None
    week: int | None = None
    kickoff_utc: datetime | None = None
    home_team: str
    away_team: str

    # Status
    state: str = Field(description="'pre' | 'in' | 'post'.")
    completed: bool = False
    status_detail: str | None = None

    # Clock
    period: int = 1
    clock_seconds: int = PERIOD_SECONDS
    clock_display: str | None = None

    # Score
    home_score: int = 0
    away_score: int = 0

    # Possession / field position
    possession: str | None = None
    posteam_is_home: bool | None = None
    down: int | None = None
    distance: int | None = None
    yards_to_endzone: int | None = None
    is_red_zone: bool | None = None
    down_distance_text: str | None = None

    # Timeouts
    home_timeouts: int | None = None
    away_timeouts: int | None = None
    timeouts_exact: bool = Field(
        default=True,
        description=(
            "False when timeouts are the feed's current value rather than the value at "
            "``at``. ESPN publishes timeouts only for 'now', so a truncated ESPN snapshot "
            "cannot know them."
        ),
    )

    # Context
    drives: list[LiveDrive] = Field(default_factory=list)
    injuries: list[LiveInjury] = Field(default_factory=list)
    venue: LiveVenue | None = None
    weather: LiveWeather | None = None
    home_record: TeamRecord | None = None
    away_record: TeamRecord | None = None

    # ----------------------------------------------------------------- #

    @property
    def plays(self) -> list[LivePlay]:
        """Every play so far, in order. ESPN nests plays inside drives and has no
        top-level array; so do we."""
        return [play for drive in self.drives for play in drive.plays]

    @property
    def play_count(self) -> int:
        return sum(len(drive.plays) for drive in self.drives)

    @property
    def last_play(self) -> LivePlay | None:
        for drive in reversed(self.drives):
            if drive.plays:
                return drive.plays[-1]
        return None

    @property
    def last_play_wallclock(self) -> datetime | None:
        for drive in reversed(self.drives):
            for play in reversed(drive.plays):
                if play.wallclock is not None:
                    return play.wallclock
        return None

    @property
    def quarter_seconds_remaining(self) -> int:
        return self.clock_seconds

    @property
    def half_seconds_remaining(self) -> int:
        return half_seconds_remaining(self.period, self.clock_seconds)

    @property
    def game_seconds_remaining(self) -> int:
        return game_seconds_remaining(self.period, self.clock_seconds)

    @property
    def is_overtime(self) -> bool:
        return self.period > REGULATION_PERIODS

    @property
    def score_differential_home(self) -> int:
        return self.home_score - self.away_score

    def staleness(self, now: datetime | None = None) -> timedelta | None:
        """Wall-clock age of the newest play, or ``None`` if no play carries one.

        The feed's own ``fetched_at`` says nothing about whether ESPN's CDN
        handed us a cached page: only the play stream advancing does.
        """
        wallclock = self.last_play_wallclock
        if wallclock is None:
            return None
        now = now or datetime.now(tz=wallclock.tzinfo)
        return now - wallclock

    def is_stale(self, max_age: timedelta = timedelta(minutes=2), now: datetime | None = None) -> bool:
        """True if a *live* game's play stream has not advanced within ``max_age``."""
        if self.state != "in":
            return False
        age = self.staleness(now)
        return age is not None and age > max_age


# --------------------------------------------------------------------------- #
# Truncation -- the shared "rewind to (quarter, clock)" rule
# --------------------------------------------------------------------------- #


def truncate_snapshot(snapshot: LiveSnapshot, at: AtTime) -> LiveSnapshot:
    """Rewind ``snapshot`` to ``at``, keeping only plays that had already started.

    A play counts as *before* ``at`` when its own ``(period, clock)`` is strictly
    earlier in game time. The resulting situation (down / distance / yards to
    endzone / possession) is read off the **next** play's pre-snap ``start``,
    which is precisely the snap the model is asked to predict from -- and is the
    field group where ESPN and nflverse agree exactly.
    """
    cutoff = at.order_key
    kept: list[LiveDrive] = []
    next_play: LivePlay | None = None
    for drive in snapshot.drives:
        before = [p for p in drive.plays if p.order_key < cutoff]
        after = [p for p in drive.plays if p.order_key >= cutoff]
        if next_play is None and after:
            next_play = after[0]
        if not before:
            continue
        if after:
            # This drive is still in progress at ``at``. Its result, yardage and
            # play count are outcomes of plays that have not happened -- carrying
            # them forward would be textbook leakage (CLAUDE.md rule 3).
            kept.append(
                drive.model_copy(
                    update={
                        "plays": before,
                        "result": None,
                        "display_result": None,
                        "description": None,
                        "yards": None,
                        "offensive_plays": len(before),
                        "is_score": False,
                        "end_period": before[-1].period,
                        "end_clock_display": before[-1].clock_display,
                        "time_elapsed": None,
                    }
                )
            )
        else:
            kept.append(drive.model_copy(update={"plays": before}))

    last = None
    for drive in reversed(kept):
        if drive.plays:
            last = drive.plays[-1]
            break

    home_score = last.home_score if last and last.home_score is not None else 0
    away_score = last.away_score if last and last.away_score is not None else 0

    possession = next_play.team if next_play else None
    posteam_is_home = None if possession is None else possession == snapshot.home_team

    home_timeouts: int | None
    away_timeouts: int | None
    if next_play is not None and next_play.home_timeouts_remaining is not None:
        home_timeouts = next_play.home_timeouts_remaining
        away_timeouts = next_play.away_timeouts_remaining
        timeouts_exact = True
    else:
        home_timeouts = snapshot.home_timeouts
        away_timeouts = snapshot.away_timeouts
        timeouts_exact = next_play is None and snapshot.timeouts_exact

    yards_to_endzone = next_play.start_yards_to_endzone if next_play else None

    return snapshot.model_copy(
        update={
            "at": at,
            "period": at.quarter,
            "clock_seconds": at.clock_seconds,
            "clock_display": seconds_to_clock(at.clock_seconds),
            "state": "in" if next_play is not None else snapshot.state,
            "completed": False if next_play is not None else snapshot.completed,
            "status_detail": str(at) if next_play is not None else snapshot.status_detail,
            "home_score": home_score,
            "away_score": away_score,
            "possession": possession,
            "posteam_is_home": posteam_is_home,
            "down": next_play.start_down if next_play else None,
            "distance": next_play.start_distance if next_play else None,
            "yards_to_endzone": yards_to_endzone,
            "is_red_zone": None if yards_to_endzone is None else yards_to_endzone <= 20,
            "down_distance_text": None,
            "home_timeouts": home_timeouts,
            "away_timeouts": away_timeouts,
            "timeouts_exact": timeouts_exact,
            "drives": kept,
            "injuries": _visible_injuries(snapshot.injuries, last),
        }
    )


def _visible_injuries(injuries: list[LiveInjury], last_play: LivePlay | None) -> list[LiveInjury]:
    """Leakage gate: an injury reported after the last visible play was not
    knowable yet. Injuries with no timestamp stay -- they are pregame reports."""
    cutoff = last_play.wallclock if last_play else None
    if cutoff is None or cutoff.tzinfo is None:
        return list(injuries)
    return [
        i
        for i in injuries
        if i.reported_at is None or i.reported_at.tzinfo is None or i.reported_at <= cutoff
    ]


# --------------------------------------------------------------------------- #
# Schedules -- the nflverse <-> ESPN crosswalk
# --------------------------------------------------------------------------- #

_SCHEDULE_COLUMNS = ("game_id", "espn", "season", "game_type", "week", "gameday", "gametime")


def read_parquet_dir(root: Path, required: tuple[str, ...] = ()) -> pl.DataFrame | None:
    """Concatenate every parquet under ``root`` that carries ``required`` columns.

    The column filter matters because a directory handed in by a caller may hold
    more than one table; silently vertical-stacking schedules onto play-by-play
    produces a frame that is wrong in a hard-to-notice way.
    """
    files = sorted(root.rglob("*.parquet"))
    frames = [pl.read_parquet(f) for f in files]
    frames = [f for f in frames if all(c in f.columns for c in required)]
    if not frames:
        return None
    common = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    return pl.concat([f.select(common) for f in frames], how="vertical_relaxed")


@lru_cache(maxsize=4)
def _load_schedules_cached(path: str | None) -> pl.DataFrame:
    if path is not None:
        target = Path(path)
        frame = (
            read_parquet_dir(target, ("game_id", "espn"))
            if target.is_dir()
            else pl.read_parquet(target)
        )
        if frame is None:
            raise FileNotFoundError(f"no schedules parquet under {target}")
        return frame

    local = RAW_DIR / "nflverse" / "schedules"
    if local.exists():
        frame = read_parquet_dir(local, ("game_id", "espn"))
        if frame is not None:
            return frame

    import nflreadpy  # imported lazily: a network dependency, never at import time

    return nflreadpy.load_schedules()


def load_schedule_frame(source: pl.DataFrame | str | Path | None = None) -> pl.DataFrame:
    """Return the nflverse schedules frame.

    ``source`` may be an already-loaded frame, a parquet file, a directory of
    parquet files, or ``None`` -- in which case ``data/raw/nflverse/schedules``
    is used if it exists and ``nflreadpy.load_schedules()` otherwise. The ingest
    module for nflverse is deliberately not imported: this works with or without
    it.
    """
    if isinstance(source, pl.DataFrame):
        return source
    return _load_schedules_cached(None if source is None else str(Path(source)))


def resolve_espn_event_id(
    nflverse_game_id: str, schedules: pl.DataFrame | str | Path | None = None
) -> str:
    """``2024_01_BAL_KC`` -> ``"401671789"`` via the schedules ``espn`` column.

    The crosswalk is authoritative; we never construct the id from team names.
    """
    frame = load_schedule_frame(schedules)
    row = frame.filter(pl.col("game_id") == nflverse_game_id)
    if row.height == 0:
        raise KeyError(f"game_id {nflverse_game_id!r} not in schedules")
    espn = row.get_column("espn").item(0)
    if espn is None or str(espn).strip() in ("", "None", "nan"):
        raise KeyError(f"schedules row for {nflverse_game_id!r} has no espn id yet")
    return str(espn).split(".")[0]


def schedule_row(
    nflverse_game_id: str, schedules: pl.DataFrame | str | Path | None = None
) -> dict[str, object]:
    """The schedules row for a game, as a plain dict."""
    frame = load_schedule_frame(schedules)
    row = frame.filter(pl.col("game_id") == nflverse_game_id)
    if row.height == 0:
        raise KeyError(f"game_id {nflverse_game_id!r} not in schedules")
    return row.row(0, named=True)


def team_record_as_of(
    team: str,
    season: int,
    week: int,
    schedules: pl.DataFrame | str | Path | None = None,
) -> TeamRecord:
    """Regular-season record from *completed prior weeks only* (leakage gate)."""
    frame = load_schedule_frame(schedules)
    prior = frame.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        & pl.col("result").is_not_null()
    )
    wins = losses = ties = 0
    for row in prior.iter_rows(named=True):
        result = row["result"]
        margin = result if row["home_team"] == team else -result
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
    overall = f"{wins}-{losses}" if ties == 0 else f"{wins}-{losses}-{ties}"
    return TeamRecord(team=team, overall=overall, wins=wins, losses=losses, ties=ties)


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


class LiveFeed(ABC):
    """Anything that can hand back a :class:`LiveSnapshot` for a game."""

    source: ClassVar[str] = "live"

    def __init__(self, schedules: pl.DataFrame | str | Path | None = None) -> None:
        self._schedules = schedules

    # -- required -------------------------------------------------------- #

    @abstractmethod
    def list_games(self, date: date_type) -> list[GameRef]:
        """Every game on ``date``."""

    @abstractmethod
    def snapshot(self, game_ref: GameRef | str, at: AtTime | str | None = None) -> LiveSnapshot:
        """The state of ``game_ref`` at ``at`` (``None`` = latest)."""

    # -- shared ---------------------------------------------------------- #

    @property
    def schedules(self) -> pl.DataFrame:
        return load_schedule_frame(self._schedules)

    def resolve_game_id(self, nflverse_game_id: str) -> str:
        """``2024_01_BAL_KC`` -> ESPN event id."""
        return resolve_espn_event_id(nflverse_game_id, self._schedules)

    def game_ref(self, game_id: str) -> GameRef:
        """Build a :class:`GameRef` from an nflverse game id (or an ESPN event id)."""
        if not re.match(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$", game_id):
            return GameRef(espn_event_id=str(game_id))
        row = schedule_row(game_id, self._schedules)
        espn = row.get("espn")
        gameday = row.get("gameday")
        if isinstance(gameday, str):
            gameday = date_type.fromisoformat(gameday)
        return GameRef(
            espn_event_id=None if espn in (None, "") else str(espn).split(".")[0],
            nflverse_game_id=game_id,
            season=row.get("season"),
            season_type=row.get("game_type"),
            week=row.get("week"),
            game_date=gameday,
            home_team=row.get("home_team"),
            away_team=row.get("away_team"),
            short_name=f"{row.get('away_team')} @ {row.get('home_team')}",
            venue_name=row.get("stadium"),
        )
