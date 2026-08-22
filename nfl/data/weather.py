"""Open-Meteo weather ingest for NFL games.

Two endpoints, one output schema:

* **ERA5 archive** (``archive-api.open-meteo.com/v1/archive``) -- gap-free hourly
  reanalysis back to 1940, roughly a five-day publication lag. This is the training-set
  source, and rows it produces carry ``is_forecast=False``.
* **Forecast** (``api.open-meteo.com/v1/forecast``) -- up to 16 days ahead, for games
  that have not been played (or are inside the archive's lag). Rows carry
  ``is_forecast=True``.

``is_forecast`` is not cosmetic: training on ERA5 hindcasts and serving on forecasts is a
train/serve skew, and the flag is what lets an evaluation notice it.

**Roof gating.** The per-game nflverse ``roof`` column (``outdoors`` / ``open`` /
``closed`` / ``dome``) is the authority, not a hardcoded dome list -- retractables such as
AT&T, NRG, State Farm and Lucas Oil are ``closed`` some weeks and ``open`` others. Indoor
games get a row with ``is_indoor=True`` and no HTTP call at all. When
:func:`nfl.data.venues.resolve_venue` had to override the venue (the neutral-site
``stadium_id`` bug), the game's ``roof`` came from the *wrong* stadium and is discarded in
favour of the resolved venue's ``roof_type``.

Units are requested to match :class:`nfl.data.schemas.Weather` directly: Fahrenheit, mph,
millimetres. ``snow_depth`` arrives in metres and is converted to centimetres.

All requests go out with ``timezone=UTC`` so no DST arithmetic is needed anywhere;
``gameday`` + ``gametime`` (US Eastern in nflverse schedules) is converted to UTC once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from nfl.config import RAW_DIR

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

SOURCE_ARCHIVE = "openmeteo.era5"
SOURCE_FORECAST = "openmeteo.forecast"
SCHEMA_VERSION = "1"

#: nflverse schedule ``gametime`` is US Eastern wall clock, including for games abroad.
SCHEDULE_TZ = ZoneInfo("America/New_York")

#: Hourly variables. Modern spellings only -- Open-Meteo mirrors whatever spelling you
#: ask for into the response keys, and mixing legacy (``windspeed_10m``) with modern
#: (``wind_speed_10m``) silently yields half-empty rows.
ARCHIVE_HOURLY: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
)
FORECAST_HOURLY: tuple[str, ...] = (*ARCHIVE_HOURLY, "precipitation_probability", "visibility")

#: nflverse ``roof`` values that mean the field was enclosed at kickoff.
INDOOR_ROOFS = frozenset({"dome", "closed"})
OUTDOOR_ROOFS = frozenset({"outdoors", "open"})

#: Venue ``roof_type`` -> the nflverse ``roof`` value to assume when the game's own
#: ``roof`` is unusable. ``retractable`` maps to ``open``: whether the roof was shut for a
#: one-off neutral-site game abroad is not knowable from the feed, so we fetch the weather
#: and leave ``roof_closed`` null rather than silently dropping the observation.
_VENUE_ROOF_TO_GAME_ROOF = {"dome": "dome", "open": "outdoors", "retractable": "open"}

#: ERA5 publication lag. A game inside this window is not in the archive yet.
ARCHIVE_LAG_DAYS = 6

#: Open-Meteo's forecast ceiling. A game further out than this has no forecast to fetch,
#: so it is reported rather than requested -- most of a season is beyond the horizon in
#: September, and requesting it anyway would waste one call per game.
FORECAST_MAX_DAYS = 16

#: Hours of hourly data summarised per game: kickoff hour plus the rest of a ~3h game.
GAME_WINDOW_HOURS = 4

# WMO 4677 weather codes, grouped for :class:`nfl.data.schemas.PrecipitationType`.
_SNOW_CODES = frozenset({71, 73, 75, 77, 85, 86})
_SLEET_CODES = frozenset({56, 57, 66, 67})
_RAIN_CODES = frozenset({51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99})
_THUNDER_CODES = frozenset({95, 96, 99})


class OpenMeteoError(RuntimeError):
    """A non-retryable Open-Meteo failure, or retries exhausted."""


# --------------------------------------------------------------------------- #
# Kickoff timing
# --------------------------------------------------------------------------- #


def kickoff_utc(
    gameday: str | date | datetime,
    gametime: str | None,
    tz: ZoneInfo = SCHEDULE_TZ,
) -> datetime:
    """Combine nflverse ``gameday`` and ``gametime`` into an aware UTC timestamp.

    ``gametime`` is US Eastern wall clock even for games played abroad. A missing or
    malformed ``gametime`` falls back to 13:00 Eastern, the modal kickoff slot.
    """
    if isinstance(gameday, datetime):
        day = gameday.date()
    elif isinstance(gameday, date):
        day = gameday
    else:
        day = date.fromisoformat(str(gameday)[:10])

    hour, minute = 13, 0
    if gametime:
        parts = str(gametime).strip().split(":")
        try:
            hour, minute = int(parts[0]), int(parts[1])
        except (IndexError, ValueError):
            log.warning("unparseable gametime %r; defaulting to 13:00 ET", gametime)
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
    return local.astimezone(UTC)


def _floor_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------- #
# Rate limiting and HTTP
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Sliding-window limiter.

    Open-Meteo's free tier allows ~600 requests/minute and 10 000/day without a key. The
    default here is deliberately well under that: a full 2010-2025 backfill is only ~4 500
    calls, so there is nothing to gain from crowding the ceiling.
    """

    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.max_per_minute <= 0:
            return
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.max_per_minute:
                sleep_for = 60.0 - (now - self._calls[0]) + 0.01
                if sleep_for > 0:
                    log.debug("rate limit reached; sleeping %.2fs", sleep_for)
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


class OpenMeteoClient:
    """Thin Open-Meteo wrapper: rate limiting, retries with backoff, on-disk cache.

    The cache stores whole hourly *days*, so re-deriving a different in-game moment
    (halftime, the start of Q4) never costs another request.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        cache_dir: Path | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.5,
        timeout: float = 30.0,
        sleep: Any = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout, headers={"User-Agent": "nfl-wp/0.1"})
        self._owns_client = client is None
        self.cache_dir = cache_dir
        self.rate_limiter = rate_limiter or RateLimiter()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self.request_count = 0
        self.cache_hits = 0

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- cache ------------------------------------------------------------- #

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        blob = json.dumps([url, sorted(params.items())], sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    # -- request ----------------------------------------------------------- #

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with cache, rate limit and retry. Returns the decoded payload."""
        cache_path = self._cache_path(url, params)
        if cache_path is not None and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text())

        payload = self._get_uncached(url, params)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
        return payload

    def _get_uncached(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            self.request_count += 1
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:  # transport-level: worth retrying
                last_error = exc
                self._backoff(attempt, None)
                continue

            if response.status_code == 200:
                payload = response.json()
                if payload.get("error"):
                    raise OpenMeteoError(f"Open-Meteo error: {payload.get('reason', payload)}")
                return payload

            if response.status_code == 429 or response.status_code >= 500:
                last_error = OpenMeteoError(f"HTTP {response.status_code}: {response.text[:200]}")
                self._backoff(attempt, response.headers.get("Retry-After"))
                continue

            # 4xx other than 429 will not get better by asking again.
            raise OpenMeteoError(f"HTTP {response.status_code}: {response.text[:200]}")

        raise OpenMeteoError(f"exhausted {self.max_retries} attempts for {url}") from last_error

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = self.backoff_base**attempt
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        log.debug("retrying Open-Meteo in %.2fs (attempt %d)", delay, attempt + 1)
        self._sleep(delay)

    # -- endpoints ---------------------------------------------------------- #

    def archive(
        self, latitude: float, longitude: float, start: date, end: date | None = None
    ) -> dict[str, Any]:
        """ERA5 reanalysis for a date range (inclusive)."""
        return self.get(
            ARCHIVE_URL,
            {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "start_date": start.isoformat(),
                "end_date": (end or start).isoformat(),
                "hourly": ",".join(ARCHIVE_HOURLY),
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "mm",
            },
        )

    def forecast(
        self, latitude: float, longitude: float, *, forecast_days: int = 7
    ) -> dict[str, Any]:
        """Forecast for the next ``forecast_days`` days (max 16)."""
        return self.get(
            FORECAST_URL,
            {
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "hourly": ",".join(FORECAST_HOURLY),
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "mm",
                "forecast_days": min(max(forecast_days, 1), 16),
            },
        )


# --------------------------------------------------------------------------- #
# Payload -> per-game row
# --------------------------------------------------------------------------- #


def hourly_frame(payload: dict[str, Any]) -> pl.DataFrame:
    """Turn an Open-Meteo payload's ``hourly`` block into a frame with UTC timestamps."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    data: dict[str, Any] = {
        "time_utc": [datetime.fromisoformat(t).replace(tzinfo=UTC) for t in times]
    }
    for key, values in hourly.items():
        if key == "time":
            continue
        data[key] = list(values)
    return pl.DataFrame(data)


def _series(payload: dict[str, Any], key: str) -> list[Any]:
    return list((payload.get("hourly") or {}).get(key) or [])


def _time_index(payload: dict[str, Any], moment: datetime) -> int | None:
    """Index of ``moment`` (floored to the hour) in the payload's hourly time axis."""
    times = _series(payload, "time")
    target = _floor_hour(moment.astimezone(UTC)).strftime("%Y-%m-%dT%H:%M")
    try:
        return times.index(target)
    except ValueError:
        return None


def _clean(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _window_values(payload: dict[str, Any], key: str, start: int, hours: int) -> list[float]:
    values = _series(payload, key)[start : start + hours]
    return [v for v in (_clean(x) for x in values) if v is not None]


def _precipitation_type(code: float | None, rain: float | None, snow: float | None) -> str:
    rain = rain or 0.0
    snow = snow or 0.0
    if rain > 0.0 and snow > 0.0:
        return "mix"
    if code is not None:
        icode = int(code)
        if icode in _SLEET_CODES:
            return "sleet"
        if icode in _SNOW_CODES:
            return "snow"
        if icode in _RAIN_CODES:
            return "rain"
    if snow > 0.0:
        return "snow"
    if rain > 0.0:
        return "rain"
    return "none"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _kicking_difficulty(
    wind: float | None, gust: float | None, precip: float | None, temp: float | None
) -> float:
    """0 (benign) to 1 (miserable) for placekicking. Heuristic, not a fitted model."""
    wind = wind or 0.0
    gust = gust if gust is not None else wind
    precip = precip or 0.0
    temp = 60.0 if temp is None else temp
    score = (
        0.45 * _clip01(wind / 25.0)
        + 0.20 * _clip01(gust / 45.0)
        + 0.20 * _clip01(precip / 5.0)
        + 0.15 * _clip01((45.0 - temp) / 45.0)
    )
    return round(_clip01(score), 4)


def _passing_difficulty(
    wind: float | None, gust: float | None, precip: float | None, temp: float | None
) -> float:
    """0 to 1 for the passing game. Heuristic, not a fitted model."""
    wind = wind or 0.0
    gust = gust if gust is not None else wind
    precip = precip or 0.0
    temp = 60.0 if temp is None else temp
    score = (
        0.40 * _clip01(wind / 20.0)
        + 0.15 * _clip01(gust / 40.0)
        + 0.30 * _clip01(precip / 4.0)
        + 0.15 * _clip01((40.0 - temp) / 50.0)
    )
    return round(_clip01(score), 4)


def _is_severe(
    gust: float | None,
    precip: float | None,
    temp: float | None,
    snowfall: float | None,
    code: float | None,
) -> bool:
    if gust is not None and gust >= 40.0:
        return True
    if precip is not None and precip >= 7.6:  # >= 0.3 in/hr
        return True
    if temp is not None and (temp <= 10.0 or temp >= 100.0):
        return True
    if snowfall is not None and snowfall >= 2.5:  # cm in the kickoff hour
        return True
    return bool(code is not None and int(code) in _THUNDER_CODES)


def summarize_payload(
    payload: dict[str, Any],
    kickoff: datetime,
    *,
    window_hours: int = GAME_WINDOW_HOURS,
) -> dict[str, Any]:
    """Reduce an hourly payload to kickoff-hour values plus game-window aggregates.

    Returns ``{}`` when the kickoff hour is not present in the payload's time axis.
    """
    index = _time_index(payload, kickoff)
    if index is None:
        return {}

    def at(key: str) -> float | None:
        values = _series(payload, key)
        return _clean(values[index]) if index < len(values) else None

    def window(key: str) -> list[float]:
        return _window_values(payload, key, index, window_hours)

    temperature = at("temperature_2m")
    wind = at("wind_speed_10m")
    gust = at("wind_gusts_10m")
    precip = at("precipitation")
    rain = at("rain")
    snowfall = at("snowfall")
    code = at("weather_code")
    snow_depth_m = at("snow_depth")

    window_temp = window("temperature_2m")
    window_wind = window("wind_speed_10m")
    window_gust = window("wind_gusts_10m")
    window_precip = window("precipitation")
    window_snow = window("snowfall")

    return {
        "observed_at_utc": _floor_hour(kickoff.astimezone(UTC)).replace(tzinfo=None),
        "temperature_f": temperature,
        "feels_like_f": at("apparent_temperature"),
        "humidity_pct": at("relative_humidity_2m"),
        "dew_point_f": at("dew_point_2m"),
        "wind_speed_mph": wind,
        "wind_gust_mph": gust,
        "wind_direction_deg": at("wind_direction_10m"),
        "precipitation_mm": precip,
        "precipitation_type": _precipitation_type(code, rain, snowfall),
        "precipitation_probability_pct": at("precipitation_probability"),
        "rain_mm": rain,
        "snowfall_cm": snowfall,
        "snow_depth_cm": None if snow_depth_m is None else round(snow_depth_m * 100.0, 2),
        "cloud_cover_pct": at("cloud_cover"),
        "visibility_m": at("visibility"),
        "pressure_hpa": at("pressure_msl"),
        "surface_pressure_hpa": at("surface_pressure"),
        "weather_code": None if code is None else int(code),
        "window_hours": window_hours,
        "window_temperature_f_mean": _mean(window_temp),
        "window_wind_speed_mph_mean": _mean(window_wind),
        "window_wind_gust_mph_max": max(window_gust) if window_gust else None,
        "window_precipitation_mm_sum": sum(window_precip) if window_precip else None,
        "window_snowfall_cm_sum": sum(window_snow) if window_snow else None,
        "severe_weather_flag": _is_severe(gust, precip, temperature, snowfall, code),
        "kicking_difficulty_index": _kicking_difficulty(wind, gust, precip, temperature),
        "passing_difficulty_index": _passing_difficulty(wind, gust, precip, temperature),
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


# --------------------------------------------------------------------------- #
# Roof gating
# --------------------------------------------------------------------------- #


def effective_roof(game_roof: str | None, venue_roof_type: str | None, overridden: bool) -> str:
    """The roof state to gate on.

    The per-game nflverse ``roof`` normally wins -- it is what makes retractables correct
    week to week. It is discarded in two cases: when it is blank, and when the venue had
    to be overridden (the neutral-site bug means the roof describes the *designated home
    team's* stadium, not the one the game was played in).
    """
    roof = (game_roof or "").strip().lower()
    if roof and not overridden:
        return roof
    if venue_roof_type:
        return _VENUE_ROOF_TO_GAME_ROOF.get(venue_roof_type, "outdoors")
    return roof or "outdoors"


def roof_flags(roof: str) -> tuple[bool, bool | None]:
    """``(is_indoor, roof_closed)`` for an nflverse roof value."""
    if roof in INDOOR_ROOFS:
        return True, True
    if roof in OUTDOOR_ROOFS:
        return False, False
    return False, None


# --------------------------------------------------------------------------- #
# Per-game fetch
# --------------------------------------------------------------------------- #

#: Output column order and dtypes. Declared explicitly so an all-indoor or empty batch
#: still round-trips through parquet with the same schema as a full one.
WEATHER_SCHEMA: dict[str, Any] = {
    "game_id": pl.String,
    "season": pl.Int32,
    "venue_id": pl.String,
    "venue_name": pl.String,
    "latitude": pl.Float64,
    "longitude": pl.Float64,
    "elevation_m": pl.Float64,
    "kickoff_utc": pl.Datetime("us"),
    "observed_at_utc": pl.Datetime("us"),
    "is_forecast": pl.Boolean,
    "roof": pl.String,
    "is_indoor": pl.Boolean,
    "roof_closed": pl.Boolean,
    "venue_overridden": pl.Boolean,
    "temperature_f": pl.Float64,
    "feels_like_f": pl.Float64,
    "humidity_pct": pl.Float64,
    "dew_point_f": pl.Float64,
    "wind_speed_mph": pl.Float64,
    "wind_gust_mph": pl.Float64,
    "wind_direction_deg": pl.Float64,
    "wind_relative_to_field_deg": pl.Float64,
    "precipitation_mm": pl.Float64,
    "precipitation_type": pl.String,
    "precipitation_probability_pct": pl.Float64,
    "rain_mm": pl.Float64,
    "snowfall_cm": pl.Float64,
    "snow_depth_cm": pl.Float64,
    "cloud_cover_pct": pl.Float64,
    "visibility_m": pl.Float64,
    "pressure_hpa": pl.Float64,
    "surface_pressure_hpa": pl.Float64,
    "weather_code": pl.Int32,
    "window_hours": pl.Int32,
    "window_temperature_f_mean": pl.Float64,
    "window_wind_speed_mph_mean": pl.Float64,
    "window_wind_gust_mph_max": pl.Float64,
    "window_precipitation_mm_sum": pl.Float64,
    "window_snowfall_cm_sum": pl.Float64,
    "severe_weather_flag": pl.Boolean,
    "kicking_difficulty_index": pl.Float64,
    "passing_difficulty_index": pl.Float64,
    "fetch_status": pl.String,
    "source": pl.String,
    "schema_version": pl.String,
    "ingested_at": pl.Datetime("us"),
}

REQUIRED_GAME_COLUMNS = ("game_id", "gameday")


def _season_of(row: dict[str, Any]) -> int:
    season = row.get("season")
    if season is not None:
        return int(season)
    game_id = str(row.get("game_id") or "")
    if game_id[:4].isdigit():
        return int(game_id[:4])
    return date.fromisoformat(str(row["gameday"])[:10]).year


def _use_forecast(kickoff: datetime, mode: str, today: date) -> bool:
    if mode == "archive":
        return False
    if mode == "forecast":
        return True
    return kickoff.date() > today - timedelta(days=ARCHIVE_LAG_DAYS)


def fetch_weather(
    games: pl.DataFrame,
    *,
    client: OpenMeteoClient | None = None,
    mode: str = "auto",
    window_hours: int = GAME_WINDOW_HOURS,
    forecast_days: int = 16,
    today: date | None = None,
    hourly_out: list[pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Fetch one weather row per game.

    ``games`` needs ``game_id`` and ``gameday``; ``gametime``, ``season``, ``stadium_id``,
    ``stadium``, ``location`` and ``roof`` are all used when present. Venue resolution
    (and therefore the neutral-site fix) happens here via
    :func:`nfl.data.venues.resolve_games`, so the caller does not have to pre-join.

    ``mode`` is ``auto`` (archive for games older than the ERA5 lag, forecast otherwise),
    ``archive`` or ``forecast``. Indoor games never hit the network.

    Pass a list as ``hourly_out`` to collect the full hourly frames alongside the summary.
    """
    from nfl.data.venues import resolve_games

    for column in REQUIRED_GAME_COLUMNS:
        if column not in games.columns:
            raise ValueError(f"games frame is missing required column {column!r}")
    if mode not in ("auto", "archive", "forecast"):
        raise ValueError(f"mode must be auto|archive|forecast, got {mode!r}")

    today = today or datetime.now(UTC).date()
    resolved = resolve_games(games)
    owns_client = client is None
    client = client or OpenMeteoClient()
    ingested_at = datetime.now(UTC).replace(tzinfo=None)
    rows: list[dict[str, Any]] = []

    try:
        for row in resolved.iter_rows(named=True):
            rows.append(
                _weather_row(
                    row,
                    client=client,
                    mode=mode,
                    window_hours=window_hours,
                    forecast_days=forecast_days,
                    today=today,
                    ingested_at=ingested_at,
                    hourly_out=hourly_out,
                )
            )
    finally:
        if owns_client:
            client.close()

    return _to_frame(rows)


def _weather_row(
    row: dict[str, Any],
    *,
    client: OpenMeteoClient,
    mode: str,
    window_hours: int,
    forecast_days: int,
    today: date,
    ingested_at: datetime,
    hourly_out: list[pl.DataFrame] | None,
) -> dict[str, Any]:
    game_id = row["game_id"]
    kickoff = kickoff_utc(row["gameday"], row.get("gametime"))
    overridden = bool(row.get("venue_overridden"))
    roof = effective_roof(row.get("roof"), row.get("venue_roof_type"), overridden)
    is_indoor, roof_closed = roof_flags(roof)
    is_forecast = _use_forecast(kickoff, mode, today)

    base: dict[str, Any] = {
        "game_id": game_id,
        "season": _season_of(row),
        "venue_id": row.get("venue_id"),
        "venue_name": row.get("venue_name"),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "elevation_m": row.get("elevation_m"),
        "kickoff_utc": kickoff.replace(tzinfo=None),
        "observed_at_utc": _floor_hour(kickoff).replace(tzinfo=None),
        "is_forecast": is_forecast,
        "roof": roof,
        "is_indoor": is_indoor,
        "roof_closed": roof_closed,
        "venue_overridden": overridden,
        "precipitation_type": "none",
        "window_hours": window_hours,
        "severe_weather_flag": False,
        "kicking_difficulty_index": 0.0,
        "passing_difficulty_index": 0.0,
        "source": SOURCE_FORECAST if is_forecast else SOURCE_ARCHIVE,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": ingested_at,
        "fetch_status": "ok",
    }

    if is_indoor:
        # Climate controlled: nflverse temp/wind are null by construction and an ERA5
        # lookup would describe the car park, not the field.
        base["fetch_status"] = "skipped_indoor"
        return base

    if row.get("latitude") is None or row.get("longitude") is None:
        log.warning("%s: no venue coordinates; skipping weather fetch", game_id)
        base["fetch_status"] = "no_venue"
        return base

    latitude = float(row["latitude"])
    longitude = float(row["longitude"])
    days_ahead = (kickoff.date() - today).days + 1
    if is_forecast and days_ahead > min(forecast_days, FORECAST_MAX_DAYS):
        log.info("%s: kickoff is %d days out; beyond the forecast horizon", game_id, days_ahead - 1)
        base["fetch_status"] = "beyond_forecast_horizon"
        return base

    try:
        if is_forecast:
            # Ask for exactly the days needed: a shorter range is a smaller response and
            # a more cacheable key than always requesting the full 16.
            wanted = min(max(days_ahead, 1), forecast_days, FORECAST_MAX_DAYS)
            payload = client.forecast(latitude, longitude, forecast_days=wanted)
        else:
            payload = client.archive(latitude, longitude, kickoff.date(), kickoff.date())
    except OpenMeteoError as exc:
        log.warning("%s: Open-Meteo fetch failed: %s", game_id, exc)
        base["fetch_status"] = "error"
        return base

    if hourly_out is not None:
        frame = hourly_frame(payload)
        if frame.height:
            hourly_out.append(frame.with_columns(pl.lit(game_id).alias("game_id")))

    summary = summarize_payload(payload, kickoff, window_hours=window_hours)
    if not summary:
        log.warning("%s: kickoff hour %s absent from payload", game_id, kickoff.isoformat())
        base["fetch_status"] = "hour_missing"
        return base

    base.update(summary)
    return base


def _to_frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    rows = list(rows)
    if not rows:
        return pl.DataFrame(schema=WEATHER_SCHEMA)
    data = {name: [row.get(name) for row in rows] for name in WEATHER_SCHEMA}
    return pl.DataFrame(data, schema=WEATHER_SCHEMA)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

WEATHER_DIR = RAW_DIR / "weather"
DEFAULT_GAMES_PARQUET = RAW_DIR / "nflverse" / "schedules" / "*.parquet"


#: Sentinel for ``ingest_weather(cache_dir=...)`` so that an explicit ``None`` means
#: "no payload cache" rather than falling through to the default location.
CACHE_DIR_DEFAULT = "auto"


def season_path(season: int, root: Path | None = None) -> Path:
    """``data/raw/weather/season=YYYY/weather.parquet``."""
    return (root or WEATHER_DIR) / f"season={season}" / "weather.parquet"


def hourly_path(season: int, root: Path | None = None) -> Path:
    """Raw hourly series kept beside the summary so re-derivation needs no refetch."""
    return (root or WEATHER_DIR) / f"season={season}" / "hourly.parquet"


def parse_seasons(spec: str | Iterable[int]) -> list[int]:
    """Parse ``"2010-2024"`` / ``"2021,2023"`` / ``"2024"`` into a sorted season list."""
    if not isinstance(spec, str):
        return sorted({int(s) for s in spec})
    seasons: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = (int(part) for part in chunk.split("-", 1))
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(chunk))
    return sorted(seasons)


def load_games(
    games_parquet: str | Path = DEFAULT_GAMES_PARQUET, seasons: Sequence[int] | None = None
) -> pl.DataFrame:
    """Read a games frame from parquet (glob allowed) and optionally filter by season.

    Deliberately schema-light: it reads whatever the nflverse ingest wrote and only
    insists on the columns :func:`fetch_weather` actually needs.
    """
    frame = pl.read_parquet(str(games_parquet))
    if "season" not in frame.columns and "game_id" in frame.columns:
        frame = frame.with_columns(
            pl.col("game_id").str.slice(0, 4).cast(pl.Int32, strict=False).alias("season")
        )
    if seasons is not None and "season" in frame.columns:
        frame = frame.filter(pl.col("season").is_in(list(seasons)))
    return frame


def _seasons_in(frame: pl.DataFrame) -> list[int]:
    if "season" in frame.columns:
        return sorted({int(s) for s in frame.get_column("season").drop_nulls().to_list()})
    return sorted({int(str(g)[:4]) for g in frame.get_column("game_id").to_list()})


def _iter_season_frames(games: pl.DataFrame) -> Iterator[tuple[int, pl.DataFrame]]:
    if "season" not in games.columns:
        games = games.with_columns(
            pl.col("game_id").str.slice(0, 4).cast(pl.Int32, strict=False).alias("season")
        )
    for season in _seasons_in(games):
        yield season, games.filter(pl.col("season") == season)


def ingest_weather(
    seasons: str | Iterable[int] | None = None,
    *,
    games: pl.DataFrame | None = None,
    games_parquet: str | Path = DEFAULT_GAMES_PARQUET,
    root: Path | None = None,
    cache_dir: Path | None | str = CACHE_DIR_DEFAULT,
    client: OpenMeteoClient | None = None,
    mode: str = "auto",
    force: bool = False,
    window_hours: int = GAME_WINDOW_HOURS,
    max_per_minute: int = 120,
) -> list[Path]:
    """Fetch and persist weather, one parquet per season. Idempotent unless ``force``.

    Already-present ``game_id``s are skipped and merged with the new rows, so a run that
    is interrupted (or a season that gains games) costs only the missing requests.

    ``force`` re-fetches from the API, bypassing the on-disk payload cache as well as the
    parquet -- otherwise a bad cached response could never be corrected. Pass
    ``cache_dir=None`` to disable the payload cache outright.
    """
    season_list = parse_seasons(seasons) if seasons is not None else None
    if games is None:
        games = load_games(games_parquet, season_list)
    elif season_list is not None and "season" in games.columns:
        games = games.filter(pl.col("season").is_in(season_list))

    root = root or WEATHER_DIR
    resolved_cache: Path | None
    if cache_dir == CACHE_DIR_DEFAULT:
        resolved_cache = root / "_cache"
    else:
        resolved_cache = Path(cache_dir) if cache_dir is not None else None
    if force:
        resolved_cache = None
    owns_client = client is None
    client = client or OpenMeteoClient(
        cache_dir=resolved_cache, rate_limiter=RateLimiter(max_per_minute=max_per_minute)
    )
    written: list[Path] = []

    try:
        for season, season_games in _iter_season_frames(games):
            out_path = season_path(season, root)
            existing: pl.DataFrame | None = None
            if out_path.exists() and not force:
                existing = pl.read_parquet(out_path)
                done = set(existing.get_column("game_id").to_list())
                season_games = season_games.filter(~pl.col("game_id").is_in(list(done)))
                if season_games.height == 0:
                    log.info("season %d: already complete (%d games)", season, existing.height)
                    written.append(out_path)
                    continue

            hourly_frames: list[pl.DataFrame] = []
            fresh = fetch_weather(
                season_games,
                client=client,
                mode=mode,
                window_hours=window_hours,
                hourly_out=hourly_frames,
            )
            combined = (
                pl.concat([existing, fresh], how="vertical_relaxed")
                if existing is not None
                else fresh
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined.sort("game_id").write_parquet(out_path)
            written.append(out_path)

            if hourly_frames:
                _write_hourly(hourly_frames, hourly_path(season, root), force=force)
            log.info("season %d: wrote %d rows to %s", season, combined.height, out_path)
    finally:
        if owns_client:
            client.close()

    return written


def _write_hourly(frames: list[pl.DataFrame], path: Path, *, force: bool) -> None:
    fresh = pl.concat(frames, how="diagonal_relaxed")
    if path.exists() and not force:
        try:
            fresh = pl.concat([pl.read_parquet(path), fresh], how="diagonal_relaxed").unique(
                subset=["game_id", "time_utc"], keep="last"
            )
        except Exception:  # pragma: no cover - corrupt cache should not fail the ingest
            log.warning("could not merge existing hourly parquet at %s; overwriting", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_parquet(path)
