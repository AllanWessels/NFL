"""ESPN live feed.

Three endpoints, chosen because between them they cover everything a snapshot
needs and nothing else does:

``site .../scoreboard?dates=YYYYMMDD``
    The whole slate in one call -- ids, kickoff, status, score, venue, records,
    and (live games only) ``competitions[0].situation``.
``site .../summary?event=<id>``
    Drives with **nested** plays (there is no top-level ``plays`` array),
    per-play ``wallclock``, venue, and the game-scoped injury blocks.
``core .../competitions/<id>/situation``
    The only place timeouts are published anywhere in ESPN's API, and unlike the
    scoreboard's ``situation`` it survives the end of the game.

Every optional block (``situation``, ``odds``, ``weather``) vanishes outside the
live window, so every read goes through ``.get()``.
"""

from __future__ import annotations

import random
import re
import time
from datetime import UTC, date as date_type, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import polars as pl

from nfl.data.live.base import (
    AtTime,
    GameRef,
    LiveDrive,
    LiveFeed,
    LiveInjury,
    LivePlay,
    LiveSnapshot,
    LiveVenue,
    LiveWeather,
    TeamRecord,
    clock_to_seconds,
    espn_team_abbr_from_id,
    espn_to_nflverse_team,
    parse_at,
    truncate_snapshot,
)

__all__ = ["CORE_BASE", "SITE_BASE", "EspnLiveFeed"]

SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

USER_AGENT = "nfl-winprob/0.1 (research; contact via repository)"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_TEAM_REF_RE = re.compile(r"/teams/(\d+)")


def _as_int(value: Any) -> int | None:
    """ESPN ships scores as strings and yardages as floats."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime | None:
    """ESPN timestamps are UTC and sometimes drop the seconds: ``2024-11-10T14:30Z``."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _season_type(value: Any) -> str | None:
    """ESPN's ``season.type``: 1 preseason, 2 regular, 3 post."""
    return {1: "PRE", 2: "REG", 3: "POST"}.get(_as_int(value) or 0)


def _team_id_from(value: Any) -> str | None:
    """``"25"``, ``{"id": "25"}`` or ``{"$ref": ".../teams/25?..."}`` -> ``"25"``."""
    if value is None:
        return None
    if isinstance(value, str | int):
        return str(value)
    if isinstance(value, dict):
        if value.get("id") is not None:
            return str(value["id"])
        ref = value.get("$ref")
        if ref:
            match = _TEAM_REF_RE.search(str(ref))
            if match:
                return match.group(1)
    return None


class EspnLiveFeed(LiveFeed):
    """Read-only ESPN adapter. No key, no auth, be polite."""

    source: ClassVar[str] = "espn"

    def __init__(
        self,
        schedules: pl.DataFrame | str | Path | None = None,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        super().__init__(schedules=schedules)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._backoff = backoff

    # -- plumbing -------------------------------------------------------- #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EspnLiveFeed:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """GET with bounded retries. Returns ``None`` on 404 -- several of these
        endpoints legitimately do not exist for a given game."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.get(url, params=params, headers=DEFAULT_HEADERS)
            except httpx.TransportError as exc:  # network flake
                last_error = exc
            else:
                if response.status_code == 404:
                    return None
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    return response.json()
                last_error = httpx.HTTPStatusError(
                    f"{response.status_code} from {url}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries - 1:
                time.sleep(self._backoff * (2**attempt) * (1 + random.random() * 0.1))  # noqa: S311
        assert last_error is not None
        raise last_error

    # -- LiveFeed -------------------------------------------------------- #

    def list_games(self, date: date_type) -> list[GameRef]:
        payload = self._get(f"{SITE_BASE}/scoreboard", {"dates": date.strftime("%Y%m%d")}) or {}
        return [self._game_ref_from_event(event) for event in payload.get("events", [])]

    def snapshot(self, game_ref: GameRef | str, at: AtTime | str | None = None) -> LiveSnapshot:
        ref = self.game_ref(game_ref) if isinstance(game_ref, str) else game_ref
        event_id = ref.espn_event_id
        if event_id is None:
            if ref.nflverse_game_id is None:
                raise ValueError("GameRef carries neither an ESPN event id nor an nflverse id")
            event_id = self.resolve_game_id(ref.nflverse_game_id)

        summary = self._get(f"{SITE_BASE}/summary", {"event": event_id})
        if summary is None:
            raise KeyError(f"ESPN has no summary for event {event_id}")
        situation = self._get(
            f"{CORE_BASE}/events/{event_id}/competitions/{event_id}/situation",
            {"lang": "en", "region": "us"},
        )

        snapshot = self._build(event_id, ref, summary, situation)
        at_time = parse_at(at)
        return truncate_snapshot(snapshot, at_time) if at_time else snapshot

    # -- parsing --------------------------------------------------------- #

    def _game_ref_from_event(self, event: dict[str, Any]) -> GameRef:
        competition = (event.get("competitions") or [{}])[0]
        competitors = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        status = (event.get("status") or {}).get("type", {})
        kickoff = _as_datetime(event.get("date"))
        season = (event.get("season") or {}).get("year")
        season_type = (event.get("season") or {}).get("type")
        home = espn_to_nflverse_team((competitors.get("home", {}).get("team") or {}).get("abbreviation"))
        away = espn_to_nflverse_team((competitors.get("away", {}).get("team") or {}).get("abbreviation"))
        return GameRef(
            espn_event_id=str(event.get("id")),
            nflverse_game_id=self._nflverse_id_for(str(event.get("id"))),
            season=season,
            season_type=_season_type(season_type),
            week=(event.get("week") or {}).get("number"),
            game_date=kickoff.date() if kickoff else None,
            kickoff_utc=kickoff,
            home_team=home,
            away_team=away,
            state=status.get("state"),
            completed=status.get("completed"),
            short_name=event.get("shortName"),
            venue_name=(competition.get("venue") or {}).get("fullName"),
            neutral_site=bool(competition.get("neutralSite", False)),
        )

    def _nflverse_id_for(self, espn_event_id: str) -> str | None:
        """Reverse crosswalk. Best-effort: no schedules on disk is not an error."""
        try:
            frame = self.schedules
        except Exception:  # noqa: BLE001 -- offline is a normal state here
            return None
        if "espn" not in frame.columns:
            return None
        match = frame.filter(pl.col("espn").cast(pl.Utf8) == espn_event_id)
        if match.height == 0:
            return None
        return str(match.get_column("game_id").item(0))

    def _build(
        self,
        event_id: str,
        ref: GameRef,
        summary: dict[str, Any],
        situation: dict[str, Any] | None,
    ) -> LiveSnapshot:
        header = summary.get("header") or {}
        competition = (header.get("competitions") or [{}])[0]
        competitors = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        home = competitors.get("home", {})
        away = competitors.get("away", {})

        home_team = espn_to_nflverse_team((home.get("team") or {}).get("abbreviation")) or "HOME"
        away_team = espn_to_nflverse_team((away.get("team") or {}).get("abbreviation")) or "AWAY"

        status = competition.get("status") or {}
        status_type = status.get("type") or {}
        period = int(status.get("period") or 0) or 1
        clock_seconds = _as_int(status.get("clock"))
        if clock_seconds is None:
            clock_seconds = clock_to_seconds(status.get("displayClock")) or 0

        drives = self._parse_drives(summary.get("drives") or {}, home_team, away_team)
        last_play = None
        for drive in reversed(drives):
            if drive.plays:
                last_play = drive.plays[-1]
                break

        # A completed game's status block often zeroes the clock; the last play is
        # the honest answer for "where did this game end".
        if status_type.get("state") == "post" and last_play is not None:
            period = last_play.period
            clock_seconds = last_play.clock_seconds or 0

        home_score = _as_int(home.get("score"))
        away_score = _as_int(away.get("score"))
        if home_score is None and last_play is not None:
            home_score = last_play.home_score
        if away_score is None and last_play is not None:
            away_score = last_play.away_score

        possession, down, distance, yards_to_endzone, red_zone, dd_text = self._parse_situation(
            competition, situation, last_play
        )

        situation = situation or {}
        return LiveSnapshot(
            source=self.source,
            fetched_at=datetime.now(tz=UTC),
            game_id=ref.nflverse_game_id or self._nflverse_id_for(event_id),
            espn_event_id=event_id,
            season=(header.get("season") or {}).get("year"),
            season_type=_season_type((header.get("season") or {}).get("type")),
            week=(header.get("week") if isinstance(header.get("week"), int) else None),
            kickoff_utc=_as_datetime(competition.get("date")),
            home_team=home_team,
            away_team=away_team,
            state=status_type.get("state") or "pre",
            completed=bool(status_type.get("completed", False)),
            status_detail=status_type.get("detail") or status_type.get("shortDetail"),
            period=period,
            clock_seconds=clock_seconds,
            clock_display=status.get("displayClock"),
            home_score=home_score or 0,
            away_score=away_score or 0,
            possession=possession,
            posteam_is_home=None if possession is None else possession == home_team,
            down=down,
            distance=distance,
            yards_to_endzone=yards_to_endzone,
            is_red_zone=red_zone,
            down_distance_text=dd_text,
            home_timeouts=_as_int(situation.get("homeTimeouts")),
            away_timeouts=_as_int(situation.get("awayTimeouts")),
            timeouts_exact=situation.get("homeTimeouts") is not None,
            drives=drives,
            injuries=self._parse_injuries(summary.get("injuries") or []),
            venue=self._parse_venue(summary.get("gameInfo") or {}),
            weather=parse_weather((summary.get("gameInfo") or {}).get("weather")),
            home_record=_parse_record(home_team, home),
            away_record=_parse_record(away_team, away),
        )

    def _parse_situation(
        self,
        competition: dict[str, Any],
        situation: dict[str, Any] | None,
        last_play: LivePlay | None,
    ) -> tuple[str | None, int | None, int | None, int | None, bool | None, str | None]:
        """Current down / distance / possession.

        The scoreboard's ``situation`` is richest but exists only mid-game; the
        core endpoint persists but drops ``possession``; the last play always
        knows whose ball it was. Prefer in that order.
        """
        board = competition.get("situation") or {}
        core = situation or {}

        possession = espn_team_abbr_from_id(
            _team_id_from(board.get("possession"))
            or _team_id_from(core.get("possession"))
            or _team_id_from((board.get("lastPlay") or {}).get("team"))
        )
        if possession is None and last_play is not None:
            possession = last_play.team

        down = _as_int(board.get("down")) or _as_int(core.get("down"))
        distance = _as_int(board.get("distance"))
        if distance is None:
            distance = _as_int(core.get("distance"))

        # In the ``situation`` object -- unlike a play's ``start`` -- ``yardLine``
        # is already measured toward the opponent's end zone.
        yards = _as_int(board.get("yardsToEndzone"))
        if yards is None:
            yards = _as_int(board.get("yardLine"))
        if yards is None:
            yards = _as_int(core.get("yardLine"))

        red_zone = board.get("isRedZone")
        if red_zone is None:
            red_zone = core.get("isRedZone")

        return (
            possession,
            down,
            distance,
            yards,
            None if red_zone is None else bool(red_zone),
            board.get("downDistanceText"),
        )

    def _parse_drives(
        self, drives: dict[str, Any], home_team: str, away_team: str
    ) -> list[LiveDrive]:
        del home_team, away_team
        parsed: list[LiveDrive] = []
        raw = list(drives.get("previous") or [])
        current = drives.get("current")
        if current:
            raw.append(current)
        for index, drive in enumerate(raw, start=1):
            parsed.append(self._parse_drive(drive, index))
        return parsed

    def _parse_drive(self, drive: dict[str, Any], number: int) -> LiveDrive:
        team = espn_to_nflverse_team((drive.get("team") or {}).get("abbreviation"))
        start = drive.get("start") or {}
        end = drive.get("end") or {}
        return LiveDrive(
            drive_number=number,
            team=team,
            description=drive.get("description"),
            result=drive.get("result") or drive.get("shortDisplayResult"),
            display_result=drive.get("displayResult"),
            yards=_as_int(drive.get("yards")),
            offensive_plays=_as_int(drive.get("offensivePlays")),
            is_score=bool(drive.get("isScore", False)),
            start_period=(start.get("period") or {}).get("number"),
            start_clock_display=(start.get("clock") or {}).get("displayValue"),
            end_period=(end.get("period") or {}).get("number"),
            end_clock_display=(end.get("clock") or {}).get("displayValue"),
            time_elapsed=(drive.get("timeElapsed") or {}).get("displayValue"),
            plays=[self._parse_play(p, number, team) for p in drive.get("plays") or []],
        )

    def _parse_play(self, play: dict[str, Any], drive_number: int, drive_team: str | None) -> LivePlay:
        start = play.get("start") or {}
        end = play.get("end") or {}
        clock_display = (play.get("clock") or {}).get("displayValue")
        team = espn_team_abbr_from_id(_team_id_from(start.get("team"))) or drive_team
        return LivePlay(
            play_id=str(play.get("id")),
            sequence=_as_int(play.get("sequenceNumber")),
            drive_number=drive_number,
            period=int((play.get("period") or {}).get("number") or 0) or 1,
            clock_seconds=clock_to_seconds(clock_display),
            clock_display=clock_display,
            wallclock=_as_datetime(play.get("wallclock")),
            team=team,
            play_type=(play.get("type") or {}).get("text"),
            text=play.get("text"),
            home_score=_as_int(play.get("homeScore")),
            away_score=_as_int(play.get("awayScore")),
            scoring_play=bool(play.get("scoringPlay", False)),
            is_penalty=bool(play.get("isPenalty", False)),
            is_turnover=bool(play.get("isTurnover", False)),
            stat_yardage=_as_int(play.get("statYardage")),
            start_down=_as_int(start.get("down")),
            start_distance=_as_int(start.get("distance")),
            start_yards_to_endzone=_as_int(start.get("yardsToEndzone")),
            end_yards_to_endzone=_as_int(end.get("yardsToEndzone")),
        )

    def _parse_injuries(self, blocks: list[dict[str, Any]]) -> list[LiveInjury]:
        """``summary.injuries`` is one block per team -- a free by-product of the
        summary call, and the right scope for 'who is hurt in *this* game'."""
        out: list[LiveInjury] = []
        for block in blocks:
            team = espn_to_nflverse_team((block.get("team") or {}).get("abbreviation"))
            if team is None:
                continue
            for item in block.get("injuries") or []:
                athlete = item.get("athlete") or {}
                name = athlete.get("displayName") or athlete.get("fullName")
                if not name:
                    continue
                out.append(
                    LiveInjury(
                        team=team,
                        player_name=name,
                        player_id=None if athlete.get("id") is None else str(athlete["id"]),
                        position=(athlete.get("position") or {}).get("abbreviation"),
                        status=item.get("status") or (item.get("type") or {}).get("description"),
                        detail=(item.get("details") or {}).get("type"),
                        comment=item.get("shortComment"),
                        reported_at=_as_datetime(item.get("date")),
                    )
                )
        return out

    def _parse_venue(self, game_info: dict[str, Any]) -> LiveVenue | None:
        venue = game_info.get("venue")
        if not venue:
            return None
        address = venue.get("address") or {}
        return LiveVenue(
            venue_id=None if venue.get("id") is None else str(venue["id"]),
            name=venue.get("fullName"),
            city=address.get("city"),
            state=address.get("state"),
            country=address.get("country"),
            indoor=venue.get("indoor"),
            grass=venue.get("grass"),
        )


def parse_weather(weather: dict[str, Any] | None) -> LiveWeather | None:
    """ESPN sometimes swaps ``displayValue`` and ``conditionId`` (observed live:
    ``{"displayValue": "2", "conditionId": "Mostly sunny"}``). Un-swap when the
    numeric one is clearly the id."""
    if not weather:
        return None
    display = weather.get("displayValue")
    condition = weather.get("conditionId")
    if display is not None and str(display).isdigit() and condition is not None and not str(condition).isdigit():
        display, condition = condition, display
    return LiveWeather(
        display_value=None if display is None else str(display),
        temperature_f=_as_int(weather.get("temperature")),
        high_temperature_f=_as_int(weather.get("highTemperature")),
        wind_mph=_as_int(weather.get("windSpeed") or weather.get("wind")),
        condition_id=None if condition is None else str(condition),
    )


def _parse_record(team: str, competitor: dict[str, Any]) -> TeamRecord | None:
    """``record`` on the summary header, ``records`` on the scoreboard."""
    entries = competitor.get("record") or competitor.get("records") or []
    if isinstance(entries, dict):
        entries = [entries]
    by_type = {str(e.get("type") or e.get("name") or "").lower(): e.get("summary") for e in entries}
    overall = by_type.get("total") or by_type.get("overall") or by_type.get("any")
    if overall is None and not by_type:
        return None
    wins = losses = ties = None
    if overall:
        parts = str(overall).split("-")
        if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
            wins, losses = int(parts[0]), int(parts[1])
            ties = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return TeamRecord(
        team=team,
        overall=overall,
        home=by_type.get("home"),
        road=by_type.get("road") or by_type.get("away"),
        wins=wins,
        losses=losses,
        ties=ties,
    )
