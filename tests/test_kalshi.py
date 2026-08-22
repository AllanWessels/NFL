"""Kalshi client tests. No network: every HTTP call is respx-mocked."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import polars as pl
import pytest
import respx

from nfl.data.market import kalshi
from nfl.data.schemas import MARKET_FIELD_NAMES, Market

BASE = kalshi.KALSHI_BASE_URL
CHI = "KXNFLGAME-26AUG29CHITEN-CHI"
TEN = "KXNFLGAME-26AUG29CHITEN-TEN"


@pytest.fixture
def schedule() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2026_01_CHI_TEN", "2026_01_KC_DEN"],
            "season": [2026, 2026],
            "week": [1, 1],
            "gameday": ["2026-08-29", "2026-09-06"],
            "away_team": ["CHI", "KC"],
            "home_team": ["TEN", "DEN"],
        }
    )


# --------------------------------------------------------------------------- #
# Price parsing
# --------------------------------------------------------------------------- #


def test_parse_dollars_returns_decimal_not_float():
    parsed = kalshi.parse_dollars("0.5900")
    assert isinstance(parsed, Decimal)
    assert parsed == Decimal("0.59")


@pytest.mark.parametrize("value", [None, "", "not-a-price"])
def test_parse_dollars_tolerates_junk(value):
    assert kalshi.parse_dollars(value) is None


def test_quote_prefers_dollar_strings_and_fp_volume(kalshi_events):
    payload = kalshi_events["events"][0]["markets"][0]
    quote = kalshi.quote_from_payload(payload)

    assert quote.team == "CHI"
    assert quote.yes_bid == Decimal("0.59")
    assert quote.yes_ask == Decimal("0.64")
    # `volume` is null in the 2026 schema; the _fp string is authoritative.
    assert quote.volume == Decimal("91.51")
    assert quote.open_interest == Decimal("71.51")
    assert quote.yes_mid == Decimal("0.615")
    assert quote.bid_ask_spread == Decimal("0.05")


def test_legacy_integer_cent_fields_still_parse():
    quote = kalshi.quote_from_payload(
        {"ticker": "X-Y-KC", "event_ticker": "X-Y", "yes_bid": 42, "yes_ask": 45}
    )
    assert quote.yes_bid == Decimal("42")


# --------------------------------------------------------------------------- #
# Liquidity filter
# --------------------------------------------------------------------------- #


def test_liquidity_filter_rejects_wide_book(kalshi_events):
    tight = kalshi.quote_from_payload(kalshi_events["events"][0]["markets"][0])
    wide = kalshi.quote_from_payload(kalshi_events["events"][0]["markets"][1])

    assert tight.is_liquid  # 5c spread, 71 OI
    assert wide.bid_ask_spread == Decimal("0.08")
    assert not wide.is_liquid  # 8c > 6c gate


def test_liquidity_filter_rejects_thin_open_interest():
    quote = kalshi.quote_from_payload(
        {
            "ticker": CHI,
            "event_ticker": "KXNFLGAME-26AUG29CHITEN",
            "yes_bid_dollars": "0.50",
            "yes_ask_dollars": "0.52",
            "open_interest_fp": "3.00",
        }
    )
    assert not quote.is_liquid


# --------------------------------------------------------------------------- #
# Orderbook
# --------------------------------------------------------------------------- #


def test_orderbook_derives_yes_ask_from_best_no_bid(kalshi_orderbook):
    book = kalshi.orderbook_from_payload(TEN, kalshi_orderbook)

    assert book.best_yes_bid == Decimal("0.36")
    assert book.best_no_bid == Decimal("0.56")
    # No ask ladder exists: yes_ask = 1 - best no bid.
    assert book.yes_ask == Decimal("0.44")
    assert book.bid_ask_spread == Decimal("0.08")
    assert book.depth("yes") == Decimal("1575.00")


# --------------------------------------------------------------------------- #
# Ticker / schedule mapping
# --------------------------------------------------------------------------- #


def test_parse_event_ticker():
    day, away, home = kalshi.parse_event_ticker("KXNFLGAME-26AUG29CHITEN")
    assert (day.year, day.month, day.day) == (2026, 8, 29)
    assert (away, home) == ("CHI", "TEN")


def test_parse_event_ticker_normalizes_espn_abbreviations():
    _, away, home = kalshi.parse_event_ticker("KXNFLGAME-26SEP14WSHLAR")
    assert (away, home) == ("WAS", "LA")


def test_parse_game_id():
    key = kalshi.parse_game_id("2024_18_KC_DEN")
    assert (key.season, key.week, key.away_team, key.home_team) == (2024, 18, "KC", "DEN")


def test_parse_game_id_rejects_junk():
    with pytest.raises(ValueError, match="game_id"):
        kalshi.parse_game_id("not-a-game")


def test_match_event_tolerates_local_kickoff_date_drift(kalshi_events, schedule):
    event = kalshi.GameEvent(
        event_ticker="KXNFLGAME-26AUG29CHITEN",
        quotes=tuple(kalshi.quote_from_payload(m) for m in kalshi_events["events"][0]["markets"]),
    )
    # occurrence_datetime is 2026-08-30T01:00Z; nflverse gameday is 2026-08-29.
    matched = kalshi.match_event_to_game(event, kalshi.schedule_keys(schedule))
    assert matched is not None
    assert matched.game_id == "2026_01_CHI_TEN"


def test_match_event_returns_none_for_unknown_teams(schedule):
    event = kalshi.GameEvent(event_ticker="KXNFLGAME-26AUG29XXXYYY")
    assert kalshi.match_event_to_game(event, kalshi.schedule_keys(schedule)) is None


def test_season_for_january_belongs_to_prior_season():
    assert kalshi.season_for(datetime(2025, 1, 12, tzinfo=UTC)) == 2024
    assert kalshi.season_for(datetime(2024, 9, 5, tzinfo=UTC)) == 2024


# --------------------------------------------------------------------------- #
# Sibling normalisation
# --------------------------------------------------------------------------- #


def test_sibling_mids_are_normalized_to_sum_to_one(kalshi_events):
    quotes = tuple(kalshi.quote_from_payload(m) for m in kalshi_events["events"][0]["markets"])
    event = kalshi.GameEvent(event_ticker="KXNFLGAME-26AUG29CHITEN", quotes=quotes)

    raw_sum = quotes[0].yes_mid + quotes[1].yes_mid
    assert raw_sum != Decimal(1)  # 0.615 + 0.40 = 1.015

    probs = kalshi.normalize_sibling_probabilities(event)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["CHI"] > probs["TEN"]


# --------------------------------------------------------------------------- #
# Market rows
# --------------------------------------------------------------------------- #


def _event(kalshi_events) -> kalshi.GameEvent:
    return kalshi.GameEvent(
        event_ticker="KXNFLGAME-26AUG29CHITEN",
        quotes=tuple(kalshi.quote_from_payload(m) for m in kalshi_events["events"][0]["markets"]),
    )


def test_market_row_is_tagged_market_only(kalshi_events):
    row = kalshi.market_row_from_event(
        _event(kalshi_events), game_id="2026_01_CHI_TEN", home_team="TEN", away_team="CHI"
    )
    assert isinstance(row, Market)
    assert row.market_only is True
    assert row.book == "kalshi"
    # Home is TEN, the underdog here.
    assert row.implied_prob_home == pytest.approx(0.3941, abs=1e-3)
    assert row.implied_prob_home + row.implied_prob_away == pytest.approx(1.0)
    assert row.kalshi_ticker == TEN


def test_market_only_cannot_be_falsified(kalshi_events):
    with pytest.raises(ValueError):
        Market(source="x", game_id="g", observed_at_utc=datetime.now(UTC), market_only=False)


def test_market_collapsed_gate(kalshi_events):
    row = kalshi.market_row_from_event(
        _event(kalshi_events), game_id="2026_01_CHI_TEN", home_team="TEN", away_team="CHI"
    )
    assert row.market_collapsed is False


def test_liquidity_flag_drops_row_when_requested(kalshi_events):
    # Home market (TEN) has an 8c spread, so the row is dropped.
    row = kalshi.market_row_from_event(
        _event(kalshi_events),
        game_id="2026_01_CHI_TEN",
        home_team="TEN",
        away_team="CHI",
        require_liquidity=True,
    )
    assert row is None


def test_candles_become_market_rows(kalshi_candlesticks):
    candles = [kalshi.candle_from_payload(TEN, bar) for bar in kalshi_candlesticks["candlesticks"]]
    rows = kalshi.market_rows_from_candles(candles, game_id="2026_01_CHI_TEN", ticker=TEN)

    assert len(rows) == 2
    assert all(row.market_only is True for row in rows)
    assert rows[0].observed_at_utc.year == 2025 or rows[0].observed_at_utc.year == 2026
    assert rows[0].implied_prob_home == pytest.approx(0.41, abs=1e-6)
    assert rows[0].implied_prob_away == pytest.approx(0.59, abs=1e-6)


def test_candle_ms_timestamp_is_not_read_as_seconds(kalshi_candlesticks):
    candle = kalshi.candle_from_payload(TEN, kalshi_candlesticks["candlesticks"][0])
    assert candle.end_ts.year > 2020


def test_to_frame_columns_are_all_market_tagged(kalshi_events):
    row = kalshi.market_row_from_event(
        _event(kalshi_events), game_id="2026_01_CHI_TEN", home_team="TEN", away_team="CHI"
    )
    frame = kalshi.to_frame([row])
    assert frame.height == 1
    assert frame["market_only"][0] is True
    assert {"kalshi_yes_price", "implied_prob_home"} <= set(frame.columns)
    assert {"kalshi_yes_price", "implied_prob_home"} <= MARKET_FIELD_NAMES


# --------------------------------------------------------------------------- #
# HTTP client (respx)
# --------------------------------------------------------------------------- #


@respx.mock
def test_discover_game_events(kalshi_events):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=kalshi_events))
    with kalshi.KalshiClient() as client:
        events = client.discover_game_events()

    assert len(events) == 1
    assert sorted(events[0].teams) == ["CHI", "TEN"]
    assert events[0].kickoff == datetime(2026, 8, 30, 1, 0, tzinfo=UTC)


@respx.mock
def test_no_auth_headers_sent_without_env(kalshi_events, monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    route = respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=kalshi_events))
    with kalshi.KalshiClient() as client:
        client.list_events()

    sent = route.calls[0].request
    assert "KALSHI-ACCESS-KEY" not in sent.headers


@respx.mock
def test_client_retries_on_500_then_succeeds(kalshi_events):
    respx.get(f"{BASE}/events").mock(
        side_effect=[
            httpx.Response(500, json={}),
            httpx.Response(200, json=kalshi_events),
        ]
    )
    with kalshi.KalshiClient(backoff=0.0) as client:
        assert len(client.list_events()) == 1


@respx.mock
def test_client_raises_on_persistent_500():
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(500, json={}))
    with kalshi.KalshiClient(backoff=0.0) as client, pytest.raises(kalshi.KalshiError):
        client.list_events()


@respx.mock
def test_client_raises_on_404_without_retrying():
    route = respx.get(f"{BASE}/markets/nope").mock(return_value=httpx.Response(404, text="nope"))
    with kalshi.KalshiClient(backoff=0.0) as client, pytest.raises(kalshi.KalshiError):
        client.get_market("nope")
    assert route.call_count == 1


@respx.mock
def test_events_pagination_follows_cursor(kalshi_events):
    page_two = {"events": [], "cursor": ""}
    first = dict(kalshi_events, cursor="abc")
    respx.get(f"{BASE}/events").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=page_two)]
    )
    with kalshi.KalshiClient() as client:
        assert len(client.list_events()) == 1


@respx.mock
def test_snapshot_game_end_to_end(kalshi_events, kalshi_orderbook, schedule, tmp_path):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=kalshi_events))
    respx.get(url__regex=rf"{BASE}/markets/.*/orderbook").mock(
        return_value=httpx.Response(200, json=kalshi_orderbook)
    )
    with kalshi.KalshiClient() as client:
        row = kalshi.snapshot_game("2026_01_CHI_TEN", schedule=schedule, client=client)

    assert row is not None
    assert row.game_id == "2026_01_CHI_TEN"
    assert row.market_only is True

    path = kalshi.write_rows([row], season=2026, root=tmp_path)
    assert path.parent.name == "season=2026"
    assert pl.read_parquet(path).height == 1


@respx.mock
def test_snapshot_returns_none_when_game_has_no_market(schedule):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json={"events": []}))
    respx.get(f"{BASE}/markets").mock(return_value=httpx.Response(200, json={"markets": []}))
    with kalshi.KalshiClient() as client:
        assert kalshi.snapshot_game("2026_01_KC_DEN", schedule=schedule, client=client) is None


@respx.mock
def test_discover_falls_back_to_markets_when_events_lack_nested_markets(kalshi_events):
    flat = [dict(m) for m in kalshi_events["events"][0]["markets"]]
    respx.get(f"{BASE}/events").mock(
        return_value=httpx.Response(
            200, json={"events": [{"event_ticker": "KXNFLGAME-26AUG29CHITEN", "markets": []}]}
        )
    )
    respx.get(f"{BASE}/markets").mock(return_value=httpx.Response(200, json={"markets": flat}))
    with kalshi.KalshiClient() as client:
        events = client.discover_game_events()

    assert len(events) == 1
    assert sorted(events[0].teams) == ["CHI", "TEN"]


@respx.mock
def test_history_game_end_to_end(kalshi_events, kalshi_candlesticks, schedule):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=kalshi_events))
    respx.get(url__regex=rf"{BASE}/series/.*/candlesticks").mock(
        return_value=httpx.Response(200, json=kalshi_candlesticks)
    )
    with kalshi.KalshiClient() as client:
        rows = kalshi.history_game("2026_01_CHI_TEN", schedule=schedule, client=client)

    assert len(rows) == 2
    assert rows[0].kalshi_ticker == TEN
    assert rows[0].observed_at_utc < rows[1].observed_at_utc


@respx.mock
def test_ingest_writes_partitioned_parquet(kalshi_events, schedule, tmp_path):
    respx.get(f"{BASE}/events").mock(return_value=httpx.Response(200, json=kalshi_events))
    with kalshi.KalshiClient() as client:
        path = kalshi.ingest_kalshi(
            ["2026_01_CHI_TEN"], schedule=schedule, client=client, root=tmp_path
        )

    assert path.parent == tmp_path / "market" / "kalshi" / "season=2026"
    frame = pl.read_parquet(path)
    assert frame["game_id"].to_list() == ["2026_01_CHI_TEN"]
    assert frame["market_only"].to_list() == [True]
