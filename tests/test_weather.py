"""Open-Meteo weather ingest. All HTTP is mocked with respx -- no live network."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
import respx

from nfl.data import weather as wx
from nfl.data.schemas import Weather

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _payload(
    day: str = "2024-11-24",
    *,
    latitude: float = 42.7738,
    longitude: float = -78.7870,
    forecast: bool = False,
    **overrides: list[Any],
) -> dict[str, Any]:
    """A 24-hour Open-Meteo payload shaped exactly like the live one.

    Every hourly array is index-aligned with ``time``; overrides replace a whole array.
    """
    times = [f"{day}T{hour:02d}:00" for hour in range(24)]
    hourly: dict[str, list[Any]] = {
        "time": times,
        "temperature_2m": [30.0 + h for h in range(24)],
        "apparent_temperature": [25.0 + h for h in range(24)],
        "relative_humidity_2m": [60.0] * 24,
        "dew_point_2m": [20.0] * 24,
        "precipitation": [0.0] * 24,
        "rain": [0.0] * 24,
        "snowfall": [0.0] * 24,
        "snow_depth": [0.0] * 24,
        "weather_code": [0] * 24,
        "cloud_cover": [40.0] * 24,
        "pressure_msl": [1015.0] * 24,
        "surface_pressure": [1010.0] * 24,
        "wind_speed_10m": [10.0] * 24,
        "wind_gusts_10m": [18.0] * 24,
        "wind_direction_10m": [270.0] * 24,
    }
    if forecast:
        hourly["precipitation_probability"] = [10.0] * 24
        hourly["visibility"] = [24000.0] * 24
    hourly.update(overrides)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "utc_offset_seconds": 0,
        "elevation": 235.0,
        "hourly_units": {"temperature_2m": "°F", "wind_speed_10m": "mph"},
        "hourly": hourly,
    }


def _games(**overrides: list[Any]) -> pl.DataFrame:
    data: dict[str, list[Any]] = {
        "game_id": ["2024_12_BUF_KC"],
        "season": [2024],
        "gameday": ["2024-11-24"],
        "gametime": ["16:25"],
        "stadium_id": ["BUF00"],
        "stadium": ["Highmark Stadium"],
        "location": ["Home"],
        "roof": ["outdoors"],
    }
    data.update(overrides)
    return pl.DataFrame(data)


@pytest.fixture
def archive_route():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        yield route


# --------------------------------------------------------------------------- #
# Kickoff timing
# --------------------------------------------------------------------------- #


def test_kickoff_converts_eastern_to_utc_in_daylight_time() -> None:
    # September: EDT is UTC-4, so 13:00 ET -> 17:00 UTC.
    assert wx.kickoff_utc("2024-09-08", "13:00") == datetime(2024, 9, 8, 17, 0, tzinfo=UTC)


def test_kickoff_converts_eastern_to_utc_in_standard_time() -> None:
    # December: EST is UTC-5, so 13:00 ET -> 18:00 UTC. Requesting timezone=UTC from
    # Open-Meteo means this is the only DST arithmetic in the module.
    assert wx.kickoff_utc("2024-12-08", "13:00") == datetime(2024, 12, 8, 18, 0, tzinfo=UTC)


def test_kickoff_of_a_london_game_is_still_eastern_wall_clock() -> None:
    """nflverse stores 09:30 ET for the London morning slot, not 14:30 local."""
    assert wx.kickoff_utc("2024-10-20", "09:30") == datetime(2024, 10, 20, 13, 30, tzinfo=UTC)


def test_kickoff_accepts_date_objects_and_datetimes() -> None:
    assert wx.kickoff_utc(date(2024, 12, 8), "13:00").hour == 18
    assert wx.kickoff_utc(datetime(2024, 12, 8, 9, 9), "13:00").hour == 18


@pytest.mark.parametrize("bad", [None, "", "nonsense", "13"])
def test_missing_gametime_falls_back_to_one_pm_eastern(bad: str | None) -> None:
    assert wx.kickoff_utc("2024-12-08", bad) == datetime(2024, 12, 8, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Roof gating
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("roof", "indoor", "closed"),
    [
        ("outdoors", False, False),
        ("open", False, False),
        ("closed", True, True),
        ("dome", True, True),
        ("", False, None),
    ],
)
def test_roof_flags(roof: str, indoor: bool, closed: bool | None) -> None:
    assert wx.roof_flags(roof) == (indoor, closed)


def test_per_game_roof_wins_for_retractables() -> None:
    """A retractable is closed some weeks and open others; the game row decides."""
    assert wx.effective_roof("closed", "retractable", False) == "closed"
    assert wx.effective_roof("open", "retractable", False) == "open"


def test_blank_game_roof_falls_back_to_venue() -> None:
    assert wx.effective_roof("", "dome", False) == "dome"
    assert wx.effective_roof(None, "open", False) == "outdoors"


def test_overridden_venue_discards_the_stale_game_roof() -> None:
    """2026_07_PIT_NO is roof="dome" because nflverse copied the Superdome's roof;
    the game is actually at Stade de France, which is open to the sky."""
    assert wx.effective_roof("dome", "open", True) == "outdoors"


def test_overridden_retractable_is_fetched_not_dropped() -> None:
    """Whether the Bernabeu roof was shut is unknowable from the feed, so fetch anyway."""
    assert wx.effective_roof("outdoors", "retractable", True) == "open"
    assert wx.roof_flags("open") == (False, False)


@pytest.mark.parametrize("roof", ["dome", "closed"])
def test_indoor_games_make_no_http_request(roof: str) -> None:
    with respx.mock(assert_all_called=False) as mock:
        archive = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        out = wx.fetch_weather(
            _games(roof=[roof], stadium_id=["DET00"], stadium=["Ford Field"]),
            mode="archive",
        )
    assert archive.call_count == 0
    row = out.row(0, named=True)
    assert row["is_indoor"] is True
    assert row["roof_closed"] is True
    assert row["fetch_status"] == "skipped_indoor"
    assert row["temperature_f"] is None
    assert row["kicking_difficulty_index"] == 0.0


# --------------------------------------------------------------------------- #
# Payload summarisation
# --------------------------------------------------------------------------- #


def test_summary_reads_the_kickoff_hour_not_the_first_hour() -> None:
    kickoff = datetime(2024, 11, 24, 21, 25, tzinfo=UTC)  # 16:25 ET
    summary = wx.summarize_payload(_payload(), kickoff)
    # temperature_2m[h] == 30 + h, so hour 21 must give 51.
    assert summary["temperature_f"] == 51.0
    assert summary["observed_at_utc"] == datetime(2024, 11, 24, 21, 0)


def test_summary_window_aggregates_span_the_game() -> None:
    kickoff = datetime(2024, 11, 24, 20, 0, tzinfo=UTC)
    summary = wx.summarize_payload(
        _payload(precipitation=[1.0] * 24, wind_gusts_10m=[float(h) for h in range(24)]),
        kickoff,
        window_hours=4,
    )
    assert summary["window_hours"] == 4
    assert summary["window_temperature_f_mean"] == pytest.approx((50 + 51 + 52 + 53) / 4)
    assert summary["window_precipitation_mm_sum"] == pytest.approx(4.0)
    assert summary["window_wind_gust_mph_max"] == pytest.approx(23.0)


def test_summary_returns_empty_when_kickoff_hour_is_absent() -> None:
    assert wx.summarize_payload(_payload("2024-11-24"), datetime(2030, 1, 1, tzinfo=UTC)) == {}


def test_snow_depth_converts_metres_to_centimetres() -> None:
    summary = wx.summarize_payload(
        _payload(snow_depth=[0.15] * 24), datetime(2024, 11, 24, 18, 0, tzinfo=UTC)
    )
    assert summary["snow_depth_cm"] == pytest.approx(15.0)


def test_nulls_in_the_hourly_array_survive_as_none() -> None:
    summary = wx.summarize_payload(
        _payload(wind_gusts_10m=[None] * 24), datetime(2024, 11, 24, 18, 0, tzinfo=UTC)
    )
    assert summary["wind_gust_mph"] is None


@pytest.mark.parametrize(
    ("code", "rain", "snow", "expected"),
    [
        (0, 0.0, 0.0, "none"),
        (45, 0.0, 0.0, "none"),  # fog is not precipitation
        (61, 2.0, 0.0, "rain"),
        (81, 5.0, 0.0, "rain"),
        (73, 0.0, 3.0, "snow"),
        (86, 0.0, 4.0, "snow"),
        (66, 1.0, 0.0, "sleet"),
        (57, 0.5, 0.0, "sleet"),
        (71, 1.0, 2.0, "mix"),  # both falling at once beats the code
    ],
)
def test_precipitation_type_from_wmo_code(
    code: int, rain: float, snow: float, expected: str
) -> None:
    summary = wx.summarize_payload(
        _payload(weather_code=[code] * 24, rain=[rain] * 24, snowfall=[snow] * 24),
        datetime(2024, 11, 24, 18, 0, tzinfo=UTC),
    )
    assert summary["precipitation_type"] == expected


@pytest.mark.parametrize(
    ("overrides", "severe"),
    [
        ({}, False),
        ({"wind_gusts_10m": [45.0] * 24}, True),
        ({"precipitation": [9.0] * 24}, True),
        ({"temperature_2m": [5.0] * 24}, True),
        ({"snowfall": [4.0] * 24}, True),
        ({"weather_code": [95] * 24}, True),
    ],
)
def test_severe_weather_flag(overrides: dict[str, list[Any]], severe: bool) -> None:
    summary = wx.summarize_payload(_payload(**overrides), datetime(2024, 11, 24, 18, 0, tzinfo=UTC))
    assert summary["severe_weather_flag"] is severe


def test_difficulty_indices_are_bounded_and_monotone() -> None:
    calm = wx.summarize_payload(
        _payload(wind_speed_10m=[1.0] * 24, wind_gusts_10m=[2.0] * 24),
        datetime(2024, 11, 24, 18, 0, tzinfo=UTC),
    )
    gale = wx.summarize_payload(
        _payload(
            wind_speed_10m=[35.0] * 24,
            wind_gusts_10m=[60.0] * 24,
            precipitation=[8.0] * 24,
            temperature_2m=[5.0] * 24,
        ),
        datetime(2024, 11, 24, 18, 0, tzinfo=UTC),
    )
    for key in ("kicking_difficulty_index", "passing_difficulty_index"):
        assert 0.0 <= calm[key] <= 1.0
        assert 0.0 <= gale[key] <= 1.0
        assert gale[key] > calm[key]


# --------------------------------------------------------------------------- #
# HTTP client behaviour
# --------------------------------------------------------------------------- #


def test_archive_request_asks_for_the_units_the_schema_stores(archive_route) -> None:
    with wx.OpenMeteoClient() as client:
        client.archive(42.7738, -78.7870, date(2024, 11, 24))
    params = archive_route.calls.last.request.url.params
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"
    assert params["precipitation_unit"] == "mm"
    assert params["timezone"] == "UTC"
    assert params["start_date"] == params["end_date"] == "2024-11-24"


def test_request_uses_modern_variable_spellings_only(archive_route) -> None:
    """Open-Meteo mirrors legacy spellings into response keys; mixing them loses data."""
    with wx.OpenMeteoClient() as client:
        client.archive(42.7738, -78.7870, date(2024, 11, 24))
    hourly = archive_route.calls.last.request.url.params["hourly"]
    assert "wind_speed_10m" in hourly and "windspeed_10m" not in hourly
    assert "weather_code" in hourly and "weathercode" not in hourly
    assert "relative_humidity_2m" in hourly and "relativehumidity_2m" not in hourly


def test_forecast_clamps_days_to_the_api_maximum() -> None:
    with respx.mock as mock:
        route = mock.get(wx.FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_payload(forecast=True))
        )
        with wx.OpenMeteoClient() as client:
            client.forecast(0.0, 0.0, forecast_days=99)
    assert route.calls.last.request.url.params["forecast_days"] == "16"


def test_retries_on_429_then_succeeds() -> None:
    slept: list[float] = []
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}, text="slow down"),
                httpx.Response(200, json=_payload()),
            ]
        )
        client = wx.OpenMeteoClient(sleep=slept.append)
        payload = client.archive(0.0, 0.0, date(2024, 11, 24))
    assert payload["hourly"]["time"]
    assert client.request_count == 2
    assert slept == [2.0]  # Retry-After is honoured over the backoff curve


def test_retries_on_server_error_then_gives_up() -> None:
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(503, text="down"))
        client = wx.OpenMeteoClient(max_retries=3, sleep=lambda _: None)
        with pytest.raises(wx.OpenMeteoError, match="exhausted 3 attempts"):
            client.archive(0.0, 0.0, date(2024, 11, 24))
    assert client.request_count == 3


def test_client_errors_are_not_retried() -> None:
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(400, text="bad latitude"))
        client = wx.OpenMeteoClient(sleep=lambda _: None)
        with pytest.raises(wx.OpenMeteoError, match="HTTP 400"):
            client.archive(999.0, 0.0, date(2024, 11, 24))
    assert route.call_count == 1


def test_json_error_body_raises() -> None:
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(
            return_value=httpx.Response(200, json={"error": True, "reason": "no data"})
        )
        with (
            wx.OpenMeteoClient(sleep=lambda _: None) as client,
            pytest.raises(wx.OpenMeteoError, match="no data"),
        ):
            client.archive(0.0, 0.0, date(2024, 11, 24))


def test_disk_cache_prevents_a_second_request(tmp_path: Path) -> None:
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        with wx.OpenMeteoClient(cache_dir=tmp_path) as client:
            first = client.archive(42.7738, -78.7870, date(2024, 11, 24))
            second = client.archive(42.7738, -78.7870, date(2024, 11, 24))
    assert route.call_count == 1
    assert client.cache_hits == 1
    assert first == second


def test_rate_limiter_sleeps_once_the_window_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(wx.time, "sleep", slept.append)
    monkeypatch.setattr(wx.time, "monotonic", lambda: 100.0)  # time never advances
    limiter = wx.RateLimiter(max_per_minute=2)
    for _ in range(3):
        limiter.acquire()
    assert len(slept) == 1
    assert slept[0] == pytest.approx(60.0, abs=0.1)


def test_rate_limiter_disabled_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(wx.time, "sleep", slept.append)
    limiter = wx.RateLimiter(max_per_minute=0)
    for _ in range(50):
        limiter.acquire()
    assert slept == []


# --------------------------------------------------------------------------- #
# fetch_weather
# --------------------------------------------------------------------------- #


def test_fetch_weather_end_to_end(archive_route) -> None:
    out = wx.fetch_weather(_games(), mode="archive")
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["game_id"] == "2024_12_BUF_KC"
    assert row["venue_id"] == "BUF00"
    assert row["is_forecast"] is False
    assert row["source"] == wx.SOURCE_ARCHIVE
    assert row["is_indoor"] is False
    assert row["fetch_status"] == "ok"
    assert row["kickoff_utc"] == datetime(2024, 11, 24, 21, 25)
    assert row["observed_at_utc"] == datetime(2024, 11, 24, 21, 0)
    assert row["temperature_f"] == 51.0
    assert row["wind_speed_mph"] == 10.0
    assert row["schema_version"] == wx.SCHEMA_VERSION
    assert row["ingested_at"] is not None


def test_fetch_weather_queries_the_resolved_venue_not_the_stale_stadium_id() -> None:
    """2025_04_MIN_PIT is tagged Acrisure Stadium but was played at Croke Park.
    Fetching Pittsburgh's weather for a game in Dublin is the whole bug."""
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(
            return_value=httpx.Response(200, json=_payload("2025-09-28"))
        )
        out = wx.fetch_weather(
            _games(
                game_id=["2025_04_MIN_PIT"],
                season=[2025],
                gameday=["2025-09-28"],
                gametime=["09:30"],
                stadium_id=["PIT00"],
                stadium=["Acrisure Stadium"],
                location=["Neutral"],
                roof=["outdoors"],
            ),
            mode="archive",
        )
    params = route.calls.last.request.url.params
    assert float(params["latitude"]) == pytest.approx(53.3607, abs=1e-3)  # Dublin
    assert float(params["longitude"]) == pytest.approx(-6.2512, abs=1e-3)
    row = out.row(0, named=True)
    assert row["venue_id"] == "DUB00"
    assert row["venue_overridden"] is True


def test_mode_auto_picks_forecast_for_upcoming_games() -> None:
    with respx.mock as mock:
        archive = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        forecast = mock.get(wx.FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_payload("2024-11-24", forecast=True))
        )
        out = wx.fetch_weather(_games(), mode="auto", today=date(2024, 11, 22))
    assert forecast.call_count == 1
    assert archive.call_count == 0
    row = out.row(0, named=True)
    assert row["is_forecast"] is True
    assert row["source"] == wx.SOURCE_FORECAST
    assert row["precipitation_probability_pct"] == 10.0
    assert row["visibility_m"] == 24000.0


def test_mode_auto_picks_archive_for_settled_games() -> None:
    with respx.mock as mock:
        archive = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        forecast = mock.get(wx.FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_payload(forecast=True))
        )
        out = wx.fetch_weather(_games(), mode="auto", today=date(2025, 1, 1))
    assert archive.call_count == 1
    assert forecast.call_count == 0
    assert out.row(0, named=True)["is_forecast"] is False


def test_forecast_requests_only_as_many_days_as_the_game_needs() -> None:
    with respx.mock as mock:
        route = mock.get(wx.FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_payload(forecast=True))
        )
        wx.fetch_weather(_games(), mode="forecast", today=date(2024, 11, 22))
    assert route.calls.last.request.url.params["forecast_days"] == "3"


def test_games_beyond_the_forecast_horizon_are_not_requested() -> None:
    """In September most of the season is >16 days out; requesting it burns quota
    for a response that cannot contain the kickoff hour."""
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(wx.FORECAST_URL).mock(
            return_value=httpx.Response(200, json=_payload(forecast=True))
        )
        out = wx.fetch_weather(_games(), mode="forecast", today=date(2024, 9, 1))
    assert route.call_count == 0
    row = out.row(0, named=True)
    assert row["fetch_status"] == "beyond_forecast_horizon"
    assert row["is_forecast"] is True
    assert row["temperature_f"] is None


def test_a_failed_fetch_yields_a_row_rather_than_killing_the_batch() -> None:
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(500, text="boom"))
        out = wx.fetch_weather(
            _games(), mode="archive", client=wx.OpenMeteoClient(max_retries=2, sleep=lambda _: None)
        )
    row = out.row(0, named=True)
    assert row["fetch_status"] == "error"
    assert row["temperature_f"] is None
    assert row["game_id"] == "2024_12_BUF_KC"


def test_unresolvable_venue_is_recorded_not_fetched() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        out = wx.fetch_weather(
            _games(stadium_id=["ZZZ99"], stadium=["Nowhere Municipal"]), mode="archive"
        )
    assert route.call_count == 0
    assert out.row(0, named=True)["fetch_status"] == "no_venue"


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="gameday"):
        wx.fetch_weather(pl.DataFrame({"game_id": ["2024_01_AA_BB"]}))


def test_bad_mode_raises() -> None:
    with pytest.raises(ValueError, match="auto|archive|forecast"):
        wx.fetch_weather(_games(), mode="sideways")


def test_empty_input_keeps_the_output_schema() -> None:
    out = wx.fetch_weather(_games().head(0), mode="archive")
    assert out.height == 0
    assert list(out.columns) == list(wx.WEATHER_SCHEMA)


def test_hourly_series_can_be_collected_for_rederivation(archive_route) -> None:
    collected: list[pl.DataFrame] = []
    wx.fetch_weather(_games(), mode="archive", hourly_out=collected)
    assert len(collected) == 1
    assert collected[0].height == 24
    assert "time_utc" in collected[0].columns
    assert collected[0].get_column("game_id").unique().to_list() == ["2024_12_BUF_KC"]


def test_output_rows_validate_against_the_weather_schema(archive_route) -> None:
    out = wx.fetch_weather(_games(), mode="archive")
    fields = set(Weather.model_fields)
    row = {k: v for k, v in out.row(0, named=True).items() if k in fields}
    weather = Weather(**row)
    assert weather.game_id == "2024_12_BUF_KC"
    assert weather.is_forecast is False
    assert weather.temperature_f == 51.0
    # Every Weather field the ingest is expected to populate must exist in the frame.
    assert fields - set(out.columns) == set()


# --------------------------------------------------------------------------- #
# Season parsing and ingest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2024", [2024]),
        ("2010-2013", [2010, 2011, 2012, 2013]),
        ("2011,2013", [2011, 2013]),
        ("2011,2013-2015", [2011, 2013, 2014, 2015]),
        ("2013-2011", []),
        ([2024, 2020, 2024], [2020, 2024]),
    ],
)
def test_parse_seasons(spec: str | list[int], expected: list[int]) -> None:
    assert wx.parse_seasons(spec) == expected


def test_ingest_writes_hive_partitioned_parquet(tmp_path: Path) -> None:
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        paths = wx.ingest_weather(
            games=_games(), root=tmp_path, cache_dir=tmp_path / "cache", mode="archive"
        )
    assert paths == [tmp_path / "season=2024" / "weather.parquet"]
    frame = pl.read_parquet(paths[0])
    assert frame.get_column("game_id").to_list() == ["2024_12_BUF_KC"]
    assert (tmp_path / "season=2024" / "hourly.parquet").exists()


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        wx.ingest_weather(games=_games(), root=tmp_path, cache_dir=None, mode="archive")
        assert route.call_count == 1
        wx.ingest_weather(games=_games(), root=tmp_path, cache_dir=None, mode="archive")
        assert route.call_count == 1, "second run must not refetch a completed season"


def test_ingest_fetches_only_the_games_it_is_missing(tmp_path: Path) -> None:
    two_games = _games(
        game_id=["2024_12_BUF_KC", "2024_13_NE_BUF"],
        season=[2024, 2024],
        gameday=["2024-11-24", "2024-12-01"],
        gametime=["16:25", "13:00"],
        stadium_id=["BUF00", "BUF00"],
        stadium=["Highmark Stadium", "Highmark Stadium"],
        location=["Home", "Home"],
        roof=["outdoors", "outdoors"],
    )
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(
            side_effect=lambda request: httpx.Response(
                200, json=_payload(request.url.params["start_date"])
            )
        )
        wx.ingest_weather(games=two_games.head(1), root=tmp_path, cache_dir=None, mode="archive")
        assert route.call_count == 1
        wx.ingest_weather(games=two_games, root=tmp_path, cache_dir=None, mode="archive")
        assert route.call_count == 2, "only the new game should be fetched"

    frame = pl.read_parquet(wx.season_path(2024, tmp_path))
    assert sorted(frame.get_column("game_id").to_list()) == ["2024_12_BUF_KC", "2024_13_NE_BUF"]


def test_ingest_force_refetches(tmp_path: Path) -> None:
    with respx.mock as mock:
        route = mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        wx.ingest_weather(games=_games(), root=tmp_path, cache_dir=None, mode="archive")
        wx.ingest_weather(games=_games(), root=tmp_path, cache_dir=None, mode="archive", force=True)
    assert route.call_count == 2


def test_ingest_defaults_the_payload_cache_under_the_root(tmp_path: Path) -> None:
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        wx.ingest_weather(games=_games(), root=tmp_path, mode="archive")
    assert list((tmp_path / "_cache").glob("*.json")), "default run should cache payloads"


def test_ingest_cache_dir_none_really_disables_the_cache(tmp_path: Path) -> None:
    """An explicit ``None`` must mean 'no cache', not 'use the default location'."""
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=_payload()))
        wx.ingest_weather(games=_games(), root=tmp_path, cache_dir=None, mode="archive")
    assert not (tmp_path / "_cache").exists()


def test_ingest_splits_seasons_into_separate_partitions(tmp_path: Path) -> None:
    games = _games(
        game_id=["2023_12_BUF_KC", "2024_12_BUF_KC"],
        season=[2023, 2024],
        gameday=["2023-11-26", "2024-11-24"],
        gametime=["16:25", "16:25"],
        stadium_id=["BUF00", "BUF00"],
        stadium=["Highmark Stadium", "Highmark Stadium"],
        location=["Home", "Home"],
        roof=["outdoors", "outdoors"],
    )
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(
            side_effect=lambda request: httpx.Response(
                200, json=_payload(request.url.params["start_date"])
            )
        )
        paths = wx.ingest_weather(games=games, root=tmp_path, cache_dir=None, mode="archive")
    assert [p.parent.name for p in paths] == ["season=2023", "season=2024"]


def test_ingest_filters_by_season(tmp_path: Path) -> None:
    games = _games(
        game_id=["2023_12_BUF_KC", "2024_12_BUF_KC"],
        season=[2023, 2024],
        gameday=["2023-11-26", "2024-11-24"],
        gametime=["16:25", "16:25"],
        stadium_id=["BUF00", "BUF00"],
        stadium=["Highmark Stadium", "Highmark Stadium"],
        location=["Home", "Home"],
        roof=["outdoors", "outdoors"],
    )
    with respx.mock as mock:
        mock.get(wx.ARCHIVE_URL).mock(
            side_effect=lambda request: httpx.Response(
                200, json=_payload(request.url.params["start_date"])
            )
        )
        paths = wx.ingest_weather(
            "2024", games=games, root=tmp_path, cache_dir=None, mode="archive"
        )
    assert [p.parent.name for p in paths] == ["season=2024"]


def test_load_games_derives_season_from_game_id(tmp_path: Path) -> None:
    path = tmp_path / "schedules.parquet"
    _games().drop("season").write_parquet(path)
    frame = wx.load_games(path)
    assert frame.get_column("season").to_list() == [2024]
    assert wx.load_games(path, seasons=[2023]).height == 0
