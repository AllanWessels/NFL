"""Replay a finished game from nflverse play-by-play as if it were live.

This is the train/serve parity bridge. ``/predict 2024_01_BAL_KC --at "Q4 05:00"``
runs the *same* inference path as a Sunday-afternoon prediction because both
end at a :class:`~nfl.data.live.base.LiveSnapshot`; only the adapter differs.

Field semantics are matched to :mod:`nfl.data.live.espn` deliberately:

* play scores are **post-play** -- nflverse ``total_home_score`` /
  ``total_away_score`` already are, and so is ESPN's ``homeScore`` / ``awayScore``;
* ``down`` / ``ydstogo`` / ``yardline_100`` are **pre-snap**, matching a play's
  ESPN ``start`` block;
* truncation and the resulting situation come from the shared
  :func:`~nfl.data.live.base.truncate_snapshot`, not from a second implementation.

The one honest asymmetry: nflverse publishes per-play timeouts and ESPN does
not, so a truncated replay snapshot carries ``timeouts_exact=True`` where a
truncated ESPN snapshot carries ``False``.
"""

from __future__ import annotations

import re
from datetime import UTC, date as date_type, datetime
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from nfl.config import RAW_DIR
from nfl.data.live.base import (
    AtTime,
    GameRef,
    LiveDrive,
    LiveFeed,
    LivePlay,
    LiveSnapshot,
    LiveVenue,
    LiveWeather,
    clock_to_seconds,
    parse_at,
    TeamRecord,
    read_parquet_dir,
    schedule_row,
    seconds_to_clock,
    team_record_as_of,
    truncate_snapshot,
)

__all__ = ["HistoricalReplayFeed"]

_YARD_LINE_RE = re.compile(r"^\s*([A-Z]{2,3})\s+(\d{1,2})\s*$")


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and value != value:  # NaN
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any) -> bool:
    return _int(value) == 1


def _time_of_day(value: Any) -> datetime | None:
    """nflverse ``time_of_day`` is ``"2024-09-06T03:27:26.537Z"`` (UTC), 2011+."""
    text = _str(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _yards_to_endzone(yard_line: str | None, posteam: str | None) -> int | None:
    """``"KC 32"`` -> 32 if KC is on defense, 68 if KC has the ball."""
    text = _str(yard_line)
    if text is None or posteam is None:
        return None
    if text in ("50", "MID 50"):
        return 50
    match = _YARD_LINE_RE.match(text)
    if match is None:
        return None
    side, number = match.group(1), int(match.group(2))
    return 100 - number if side == posteam else number


class HistoricalReplayFeed(LiveFeed):
    """A :class:`LiveFeed` backed by nflverse play-by-play parquet."""

    source: ClassVar[str] = "replay"

    def __init__(
        self,
        pbp: pl.DataFrame | str | Path | None = None,
        schedules: pl.DataFrame | str | Path | None = None,
    ) -> None:
        super().__init__(schedules=schedules)
        self._pbp = pbp
        self._cache: dict[int | None, pl.DataFrame] = {}

    # -- data access ----------------------------------------------------- #

    def _load_pbp(self, season: int | None) -> pl.DataFrame:
        if isinstance(self._pbp, pl.DataFrame):
            return self._pbp
        if season in self._cache:
            return self._cache[season]

        frame = self._read_local(Path(self._pbp) if self._pbp is not None else None, season)
        if frame is None:
            if self._pbp is not None:
                raise FileNotFoundError(f"no play-by-play parquet under {self._pbp}")
            import nflreadpy  # lazy: network dependency

            frame = nflreadpy.load_pbp([season] if season is not None else None)

        self._cache[season] = frame
        return frame

    @staticmethod
    def _read_local(target: Path | None, season: int | None) -> pl.DataFrame | None:
        """Read play-by-play from an explicit path, or from the hive-partitioned
        ``data/raw/nflverse/pbp/season=YYYY/`` layout. ``None`` if there is none."""
        if target is None:
            target = RAW_DIR / "nflverse" / "pbp"
            if not target.exists():
                return None
        if target.is_file():
            return pl.read_parquet(target)
        if season is not None:
            partition = target / f"season={season}"
            if partition.is_dir():
                return read_parquet_dir(partition, ("game_id", "play_id"))
        return read_parquet_dir(target, ("game_id", "play_id"))

    def _game_frame(self, game_id: str, season: int | None) -> pl.DataFrame:
        frame = self._load_pbp(season).filter(pl.col("game_id") == game_id)
        if frame.height == 0:
            raise KeyError(f"no play-by-play rows for {game_id!r}")
        sort_key = "order_sequence" if "order_sequence" in frame.columns else "play_id"
        return frame.sort(sort_key, nulls_last=False)

    # -- LiveFeed -------------------------------------------------------- #

    def list_games(self, date: date_type) -> list[GameRef]:
        frame = self.schedules
        gameday = pl.col("gameday")
        if frame.schema["gameday"] == pl.Utf8:
            gameday = gameday.str.to_date()
        rows = frame.filter(gameday == date)
        return [
            GameRef(
                espn_event_id=None if row.get("espn") in (None, "") else str(row["espn"]).split(".")[0],
                nflverse_game_id=row["game_id"],
                season=row.get("season"),
                season_type=row.get("game_type"),
                week=row.get("week"),
                game_date=date,
                home_team=row.get("home_team"),
                away_team=row.get("away_team"),
                state="post" if row.get("result") is not None else "pre",
                completed=row.get("result") is not None,
                short_name=f"{row.get('away_team')} @ {row.get('home_team')}",
                venue_name=row.get("stadium"),
            )
            for row in rows.iter_rows(named=True)
        ]

    def snapshot(self, game_ref: GameRef | str, at: AtTime | str | None = None) -> LiveSnapshot:
        ref = self.game_ref(game_ref) if isinstance(game_ref, str) else game_ref
        game_id = ref.nflverse_game_id
        if game_id is None:
            raise ValueError("HistoricalReplayFeed needs an nflverse game_id")

        frame = self._game_frame(game_id, ref.season)
        snapshot = self._build(game_id, ref, frame)
        at_time = parse_at(at)
        return truncate_snapshot(snapshot, at_time) if at_time else snapshot

    # -- construction ---------------------------------------------------- #

    def _build(self, game_id: str, ref: GameRef, frame: pl.DataFrame) -> LiveSnapshot:
        rows = frame.to_dicts()
        first, last = rows[0], rows[-1]
        home_team = _str(first.get("home_team")) or ref.home_team or "HOME"
        away_team = _str(first.get("away_team")) or ref.away_team or "AWAY"
        season = _int(first.get("season")) or ref.season
        week = _int(first.get("week")) or ref.week

        drives = self._build_drives(rows)
        last_play = drives[-1].plays[-1] if drives and drives[-1].plays else None

        try:
            schedule = schedule_row(game_id, self._schedules)
        except (KeyError, FileNotFoundError, ImportError):
            schedule = {}

        home_score = _int(schedule.get("home_score"))
        away_score = _int(schedule.get("away_score"))
        if home_score is None and last_play is not None:
            home_score = last_play.home_score
        if away_score is None and last_play is not None:
            away_score = last_play.away_score

        period = _int(last.get("qtr")) or 4
        clock_seconds = _int(last.get("quarter_seconds_remaining")) or 0

        records: tuple[TeamRecord | None, TeamRecord | None] = (None, None)
        if season is not None and week is not None:
            try:
                records = (
                    team_record_as_of(home_team, season, week, self._schedules),
                    team_record_as_of(away_team, season, week, self._schedules),
                )
            except (KeyError, FileNotFoundError, ImportError):
                records = (None, None)

        return LiveSnapshot(
            source=self.source,
            fetched_at=datetime.now(tz=UTC),
            game_id=game_id,
            espn_event_id=ref.espn_event_id
            or (None if schedule.get("espn") in (None, "") else str(schedule["espn"]).split(".")[0]),
            season=season,
            season_type=_str(first.get("season_type")) or ref.season_type,
            week=week,
            kickoff_utc=ref.kickoff_utc,
            home_team=home_team,
            away_team=away_team,
            # A replayed game is over unless the caller rewinds it; truncation
            # flips this back to "in".
            state="post",
            completed=True,
            status_detail="Final",
            period=period,
            clock_seconds=clock_seconds,
            clock_display=seconds_to_clock(clock_seconds),
            home_score=home_score or 0,
            away_score=away_score or 0,
            possession=None,
            posteam_is_home=None,
            down=None,
            distance=None,
            yards_to_endzone=None,
            is_red_zone=None,
            down_distance_text=None,
            home_timeouts=_int(last.get("home_timeouts_remaining")),
            away_timeouts=_int(last.get("away_timeouts_remaining")),
            timeouts_exact=True,
            drives=drives,
            # nflverse play-by-play carries no in-game injury feed. Left empty
            # rather than faked, so the field means the same thing in both feeds.
            injuries=[],
            venue=self._build_venue(first, schedule),
            weather=self._build_weather(first, schedule),
            home_record=records[0],
            away_record=records[1],
        )

    def _build_drives(self, rows: list[dict[str, Any]]) -> list[LiveDrive]:
        groups: list[tuple[Any, list[dict[str, Any]]]] = []
        for row in rows:
            key = row.get("fixed_drive")
            if key is None:
                key = row.get("drive")
            if groups and groups[-1][0] == key:
                groups[-1][1].append(row)
            else:
                groups.append((key, [row]))

        drives: list[LiveDrive] = []
        for number, (_, group) in enumerate(groups, start=1):
            team = next((_str(r.get("posteam")) for r in group if _str(r.get("posteam"))), None)
            head, tail = group[0], group[-1]
            start_yards = _int(head.get("yardline_100"))
            end_yards = _int(tail.get("yardline_100"))
            drives.append(
                LiveDrive(
                    drive_number=number,
                    team=team,
                    description=_drive_description(group),
                    result=_str(tail.get("fixed_drive_result")),
                    display_result=_str(tail.get("fixed_drive_result")),
                    yards=_int(tail.get("ydsnet"))
                    if tail.get("ydsnet") is not None
                    else (
                        None
                        if start_yards is None or end_yards is None
                        else start_yards - end_yards
                    ),
                    offensive_plays=_int(tail.get("drive_play_count")) or len(group),
                    is_score=any(_bool(r.get("sp")) for r in group),
                    start_period=_int(head.get("qtr")),
                    start_clock_display=_str(head.get("time")),
                    end_period=_int(tail.get("qtr")),
                    end_clock_display=_str(tail.get("time")),
                    time_elapsed=_str(tail.get("drive_time_of_possession")),
                    plays=[self._build_play(r, number) for r in group],
                )
            )
        return drives

    def _build_play(self, row: dict[str, Any], drive_number: int) -> LivePlay:
        posteam = _str(row.get("posteam"))
        yardline_100 = _int(row.get("yardline_100"))
        end_yards = _yards_to_endzone(_str(row.get("end_yard_line")), posteam)
        if end_yards is None and yardline_100 is not None:
            gained = _int(row.get("yards_gained")) or 0
            end_yards = max(0, min(100, yardline_100 - gained))
        clock_display = _str(row.get("time"))
        return LivePlay(
            play_id=str(_int(row.get("play_id"))),
            sequence=_int(row.get("order_sequence")) or _int(row.get("play_id")),
            drive_number=drive_number,
            period=_int(row.get("qtr")) or 1,
            clock_seconds=_int(row.get("quarter_seconds_remaining"))
            if row.get("quarter_seconds_remaining") is not None
            else clock_to_seconds(clock_display),
            clock_display=clock_display,
            wallclock=_time_of_day(row.get("time_of_day")),
            team=posteam,
            play_type=_str(row.get("play_type")),
            text=_str(row.get("desc")),
            home_score=_int(row.get("total_home_score")),
            away_score=_int(row.get("total_away_score")),
            scoring_play=_bool(row.get("sp")),
            is_penalty=_bool(row.get("penalty")),
            is_turnover=_bool(row.get("interception")) or _bool(row.get("fumble_lost")),
            stat_yardage=_int(row.get("yards_gained")),
            start_down=_int(row.get("down")),
            start_distance=_int(row.get("ydstogo")),
            start_yards_to_endzone=yardline_100,
            end_yards_to_endzone=end_yards,
            home_timeouts_remaining=_int(row.get("home_timeouts_remaining")),
            away_timeouts_remaining=_int(row.get("away_timeouts_remaining")),
        )

    def _build_venue(self, row: dict[str, Any], schedule: dict[str, Any]) -> LiveVenue | None:
        name = _str(row.get("stadium")) or _str(schedule.get("stadium"))
        venue_id = _str(row.get("stadium_id")) or _str(schedule.get("stadium_id"))
        roof = _str(row.get("roof")) or _str(schedule.get("roof"))
        surface = _str(row.get("surface")) or _str(schedule.get("surface"))
        if not any((name, venue_id, roof, surface)):
            return None
        return LiveVenue(
            venue_id=venue_id,
            name=name,
            indoor=None if roof is None else roof in ("dome", "closed"),
            grass=None if surface is None else surface == "grass",
            roof=roof,
            surface=surface,
        )

    def _build_weather(self, row: dict[str, Any], schedule: dict[str, Any]) -> LiveWeather | None:
        display = _str(row.get("weather"))
        temp = _int(row.get("temp")) if row.get("temp") is not None else _int(schedule.get("temp"))
        wind = _int(row.get("wind")) if row.get("wind") is not None else _int(schedule.get("wind"))
        if display is None and temp is None and wind is None:
            return None
        return LiveWeather(display_value=display, temperature_f=temp, wind_mph=wind)


def _drive_description(group: list[dict[str, Any]]) -> str | None:
    """``"11 plays, 70 yards, 6:46"`` -- ESPN's format, rebuilt from nflverse."""
    tail = group[-1]
    plays = _int(tail.get("drive_play_count")) or len(group)
    yards = _int(tail.get("ydsnet"))
    top = _str(tail.get("drive_time_of_possession"))
    parts = [f"{plays} play{'s' if plays != 1 else ''}"]
    if yards is not None:
        parts.append(f"{yards} yards")
    if top:
        parts.append(top)
    return ", ".join(parts)
