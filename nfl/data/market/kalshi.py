"""Kalshi NFL game-market client. **EVALUATION / LEDGER ONLY (ADR 0001).**

Nothing here may ever reach a feature vector. Rows are emitted as
:class:`nfl.data.schemas.Market`, which is structurally tagged
``market_only=True``.

Facts this module is built around (verified 2026-08-22, see the Phase 3 research
note ``research_live_feeds.md`` §7):

* Base URL ``https://api.elections.kalshi.com/trade-api/v2``. **Read-only market
  data needs no authentication.** An API key is supported for later (private
  endpoints) via ``KALSHI_API_KEY`` + ``KALSHI_PRIVATE_KEY_PATH`` but is never
  required by anything in this file.
* Series ``KXNFLGAME``; event ``KXNFLGAME-{YY}{MON}{DD}{AWAY}{HOME}``; market
  ``{EVENT}-{TEAM}``. Each game is a mutually-exclusive event with **two**
  sibling markets, one per team.
* **2026 schema change:** prices are dollar *strings* (``yes_bid_dollars``,
  ``last_price_dollars``, ...), not integer cents. ``volume`` is ``None``; the
  ``*_fp`` fixed-point strings are authoritative. Everything is parsed with
  :class:`decimal.Decimal`, never ``float``, until the final schema coercion.
* The orderbook carries **bids only** on both sides, so
  ``yes_ask = 1 - best_no_bid``.
* Sibling mid-prices do not sum to 1.00 (vig + thin books), so probabilities are
  normalised across the event before anything is compared to ``P(home)``.
* NFL books are wide and thin: rows are flagged by a liquidity filter
  (``spread <= 0.06`` and open interest above a floor) and callers should drop
  the failures before computing edge.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from nfl import config
from nfl.data.schemas import Market

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE_URL = "https://external-api.demo.kalshi.co/trade-api/v2"
NFL_GAME_SERIES = "KXNFLGAME"

SOURCE = "kalshi.game"
BOOK = "kalshi"
SCHEMA_VERSION = "1"

#: Wider than this and the "market" is noise, not a price (research §7).
MAX_BID_ASK_SPREAD = Decimal("0.06")
#: Contracts of open interest below which we do not trust the mid.
MIN_OPEN_INTEREST = Decimal("50")

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}  # fmt: skip

#: Kalshi/ESPN abbreviations that differ from nflverse's.
TEAM_ALIASES: dict[str, str] = {
    "WSH": "WAS",
    "LAR": "LA",
    "JAC": "JAX",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SFO": "SF",
    "TAM": "TB",
    "NWE": "NE",
    "NOR": "NO",
    "KAN": "KC",
    "GNB": "GB",
}

#: nflverse team abbreviations (current + historical), used to split the team
#: pair out of an event ticker suffix when no schedule frame is available.
NFLVERSE_TEAMS: frozenset[str] = frozenset(
    """ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV
    MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS OAK SD STL""".split()
)


class KalshiError(RuntimeError):
    """Any non-recoverable failure talking to Kalshi."""


# --------------------------------------------------------------------------- #
# Price parsing -- Decimal everywhere
# --------------------------------------------------------------------------- #


def parse_dollars(value: Any) -> Decimal | None:
    """Parse a 2026-style dollar/fixed-point string into :class:`Decimal`.

    Tolerates ``None``, ``""`` and numeric input. Never returns a ``float``.
    """
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # pragma: no cover - defensive
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        log.warning("kalshi: unparseable price %r", value)
        return None


def _first_dollars(payload: Mapping[str, Any], *keys: str) -> Decimal | None:
    """First parseable value among ``keys`` (2026 names first, legacy after)."""
    for key in keys:
        if key in payload:
            parsed = parse_dollars(payload[key])
            if parsed is not None:
                return parsed
    return None


def _as_float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _as_int(value: Decimal | None) -> int | None:
    return None if value is None else int(value)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        log.warning("kalshi: unparseable timestamp %r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def normalize_team(abbr: str | None) -> str | None:
    """Map a Kalshi/ESPN abbreviation onto the nflverse one."""
    if not abbr:
        return None
    upper = abbr.strip().upper()
    return TEAM_ALIASES.get(upper, upper)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Quote:
    """One market (= one team's "wins" contract) at one instant."""

    ticker: str
    event_ticker: str
    team: str | None
    status: str = ""
    result: str = ""
    yes_bid: Decimal | None = None
    yes_ask: Decimal | None = None
    no_bid: Decimal | None = None
    no_ask: Decimal | None = None
    last_price: Decimal | None = None
    volume: Decimal | None = None
    open_interest: Decimal | None = None
    liquidity: Decimal | None = None
    close_time: datetime | None = None
    occurrence_datetime: datetime | None = None
    title: str = ""

    @property
    def yes_mid(self) -> Decimal | None:
        """Mid of the yes bid/ask; falls back to last trade, then a one-sided book."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        if self.last_price is not None:
            return self.last_price
        return self.yes_bid if self.yes_bid is not None else self.yes_ask

    @property
    def bid_ask_spread(self) -> Decimal | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def is_liquid(self) -> bool:
        """Liquidity gate: tight enough spread and enough open interest."""
        spread = self.bid_ask_spread
        if spread is None or spread > MAX_BID_ASK_SPREAD:
            return False
        if self.open_interest is None or self.open_interest < MIN_OPEN_INTEREST:
            return False
        return True


@dataclass(frozen=True, slots=True)
class OrderBook:
    """Kalshi orderbook: **bids only** on both sides."""

    ticker: str
    yes_bids: tuple[tuple[Decimal, Decimal], ...] = ()
    no_bids: tuple[tuple[Decimal, Decimal], ...] = ()

    @property
    def best_yes_bid(self) -> Decimal | None:
        return max((price for price, _ in self.yes_bids), default=None)

    @property
    def best_no_bid(self) -> Decimal | None:
        return max((price for price, _ in self.no_bids), default=None)

    @property
    def yes_ask(self) -> Decimal | None:
        """``1 - best_no_bid`` -- there is no separate ask ladder."""
        best_no = self.best_no_bid
        return None if best_no is None else Decimal(1) - best_no

    @property
    def bid_ask_spread(self) -> Decimal | None:
        bid, ask = self.best_yes_bid, self.yes_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: str = "yes") -> Decimal:
        rows = self.yes_bids if side == "yes" else self.no_bids
        return sum((size for _, size in rows), Decimal(0))


@dataclass(frozen=True, slots=True)
class GameEvent:
    """A Kalshi event (one NFL game) plus its two sibling markets."""

    event_ticker: str
    title: str = ""
    quotes: tuple[Quote, ...] = ()

    @property
    def teams(self) -> tuple[str, ...]:
        return tuple(q.team for q in self.quotes if q.team)

    @property
    def kickoff(self) -> datetime | None:
        for quote in self.quotes:
            if quote.occurrence_datetime is not None:
                return quote.occurrence_datetime
        return None

    def quote_for(self, team: str) -> Quote | None:
        target = normalize_team(team)
        for quote in self.quotes:
            if quote.team == target:
                return quote
        return None


@dataclass(frozen=True, slots=True)
class Candle:
    """One candlestick bar for a single market."""

    ticker: str
    end_ts: datetime
    price_close: Decimal | None = None
    price_open: Decimal | None = None
    price_high: Decimal | None = None
    price_low: Decimal | None = None
    price_mean: Decimal | None = None
    yes_bid_close: Decimal | None = None
    yes_ask_close: Decimal | None = None
    volume: Decimal | None = None
    open_interest: Decimal | None = None

    @property
    def mid(self) -> Decimal | None:
        if self.yes_bid_close is not None and self.yes_ask_close is not None:
            return (self.yes_bid_close + self.yes_ask_close) / 2
        return self.price_close if self.price_close is not None else self.price_mean

    @property
    def bid_ask_spread(self) -> Decimal | None:
        if self.yes_bid_close is None or self.yes_ask_close is None:
            return None
        return self.yes_ask_close - self.yes_bid_close


@dataclass(slots=True)
class GameKey:
    """The nflverse identity of a game, parsed from its ``game_id``."""

    game_id: str
    season: int
    week: int
    away_team: str
    home_team: str
    kickoff_date: date | None = None

    @property
    def team_set(self) -> frozenset[str]:
        return frozenset({self.away_team, self.home_team})


# --------------------------------------------------------------------------- #
# Ticker / game_id mapping
# --------------------------------------------------------------------------- #


def parse_game_id(game_id: str) -> GameKey:
    """``2024_01_BAL_KC`` -> :class:`GameKey`."""
    parts = game_id.split("_")
    if len(parts) != 4:
        raise ValueError(f"not an nflverse game_id: {game_id!r}")
    season, week, away, home = parts
    try:
        return GameKey(
            game_id=game_id,
            season=int(season),
            week=int(week),
            away_team=normalize_team(away) or away,
            home_team=normalize_team(home) or home,
        )
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"not an nflverse game_id: {game_id!r}") from exc


def parse_event_ticker(event_ticker: str) -> tuple[date | None, str | None, str | None]:
    """``KXNFLGAME-26AUG29CHITEN`` -> ``(date(2026, 8, 29), "CHI", "TEN")``.

    Best-effort only: the suffix is ``{AWAY}{HOME}`` with no delimiter, so the
    split is resolved against :data:`NFLVERSE_TEAMS`. Ambiguous or unknown
    suffixes yield ``None`` teams -- callers should prefer matching on
    ``occurrence_datetime`` plus the market tickers' team suffixes.
    """
    suffix = event_ticker.split("-", 1)[-1]
    if len(suffix) < 7:
        return None, None, None
    yy, mon, dd = suffix[0:2], suffix[2:5], suffix[5:7]
    kickoff: date | None = None
    if yy.isdigit() and dd.isdigit() and mon.upper() in _MONTHS:
        try:
            kickoff = date(2000 + int(yy), _MONTHS[mon.upper()], int(dd))
        except ValueError:
            kickoff = None
    teams_blob = suffix[7:].upper()
    candidates = [
        (teams_blob[:i], teams_blob[i:])
        for i in range(2, len(teams_blob) - 1)
        if normalize_team(teams_blob[:i]) in NFLVERSE_TEAMS
        and normalize_team(teams_blob[i:]) in NFLVERSE_TEAMS
    ]
    if len(candidates) != 1:
        return kickoff, None, None
    away, home = candidates[0]
    return kickoff, normalize_team(away), normalize_team(home)


def schedule_keys(schedule: pl.DataFrame) -> list[GameKey]:
    """Build :class:`GameKey` rows from an nflverse schedule frame."""
    needed = {"game_id", "season", "week", "away_team", "home_team"}
    missing = needed - set(schedule.columns)
    if missing:
        raise KeyError(f"schedule frame missing columns: {sorted(missing)}")
    keys: list[GameKey] = []
    has_gameday = "gameday" in schedule.columns
    for row in schedule.iter_rows(named=True):
        gameday = row.get("gameday") if has_gameday else None
        if isinstance(gameday, datetime):
            gameday = gameday.date()
        elif isinstance(gameday, str):
            try:
                gameday = date.fromisoformat(gameday[:10])
            except ValueError:
                gameday = None
        elif not isinstance(gameday, date):
            gameday = None
        keys.append(
            GameKey(
                game_id=str(row["game_id"]),
                season=int(row["season"]),
                week=int(row["week"]),
                away_team=normalize_team(str(row["away_team"])) or str(row["away_team"]),
                home_team=normalize_team(str(row["home_team"])) or str(row["home_team"]),
                kickoff_date=gameday,
            )
        )
    return keys


def match_event_to_game(
    event: GameEvent,
    keys: Sequence[GameKey],
    *,
    date_tolerance_days: int = 1,
) -> GameKey | None:
    """Match a Kalshi event to a scheduled game on ``{teams}`` + kickoff date.

    The event ticker's date is the *local* kickoff date, which drifts from
    nflverse ``gameday`` for late kickoffs, hence the tolerance window.
    """
    teams = frozenset(event.teams)
    if len(teams) != 2:
        ticker_date, away, home = parse_event_ticker(event.event_ticker)
        if away and home:
            teams = frozenset({away, home})
        else:
            return None
    kickoff = event.kickoff
    kickoff_date = kickoff.date() if kickoff else parse_event_ticker(event.event_ticker)[0]

    best: GameKey | None = None
    for key in keys:
        if key.team_set != teams:
            continue
        if kickoff_date is None or key.kickoff_date is None:
            best = best or key
            continue
        if abs((key.kickoff_date - kickoff_date).days) <= date_tolerance_days:
            return key
    return best


def build_game_id_index(
    events: Iterable[GameEvent], schedule: pl.DataFrame
) -> dict[str, GameEvent]:
    """``{game_id: GameEvent}`` for every event that maps onto the schedule."""
    keys = schedule_keys(schedule)
    index: dict[str, GameEvent] = {}
    for event in events:
        key = match_event_to_game(event, keys)
        if key is None:
            log.info("kalshi: no schedule match for event %s", event.event_ticker)
            continue
        index[key.game_id] = event
    return index


def season_for(when: datetime | date | None, fallback: int | None = None) -> int | None:
    """NFL season a date belongs to (Jan/Feb belong to the prior season)."""
    if when is None:
        return fallback
    day = when.date() if isinstance(when, datetime) else when
    return day.year - 1 if day.month < 3 else day.year


# --------------------------------------------------------------------------- #
# Payload -> value object
# --------------------------------------------------------------------------- #


def _team_from_ticker(ticker: str, event_ticker: str) -> str | None:
    if event_ticker and ticker.startswith(f"{event_ticker}-"):
        return normalize_team(ticker[len(event_ticker) + 1 :])
    tail = ticker.rsplit("-", 1)[-1]
    return normalize_team(tail) if normalize_team(tail) in NFLVERSE_TEAMS else None


def quote_from_payload(payload: Mapping[str, Any]) -> Quote:
    """Parse a ``/markets/{ticker}`` (or nested ``markets[]``) payload."""
    ticker = str(payload.get("ticker", ""))
    event_ticker = str(payload.get("event_ticker", ""))
    return Quote(
        ticker=ticker,
        event_ticker=event_ticker,
        team=_team_from_ticker(ticker, event_ticker),
        status=str(payload.get("status", "") or ""),
        result=str(payload.get("result", "") or ""),
        yes_bid=_first_dollars(payload, "yes_bid_dollars", "yes_bid"),
        yes_ask=_first_dollars(payload, "yes_ask_dollars", "yes_ask"),
        no_bid=_first_dollars(payload, "no_bid_dollars", "no_bid"),
        no_ask=_first_dollars(payload, "no_ask_dollars", "no_ask"),
        last_price=_first_dollars(payload, "last_price_dollars", "last_price"),
        volume=_first_dollars(payload, "volume_fp", "volume"),
        open_interest=_first_dollars(payload, "open_interest_fp", "open_interest"),
        liquidity=_first_dollars(payload, "liquidity_dollars", "liquidity"),
        close_time=_parse_ts(payload.get("close_time")),
        occurrence_datetime=_parse_ts(payload.get("occurrence_datetime")),
        title=str(payload.get("title", "") or ""),
    )


def orderbook_from_payload(ticker: str, payload: Mapping[str, Any]) -> OrderBook:
    """Parse ``/markets/{ticker}/orderbook``; handles 2026 and legacy shapes."""
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}

    def _ladder(*keys: str) -> tuple[tuple[Decimal, Decimal], ...]:
        for key in keys:
            rows = book.get(key)
            if not rows:
                continue
            parsed: list[tuple[Decimal, Decimal]] = []
            for row in rows:
                if not row or len(row) < 2:
                    continue
                price, size = parse_dollars(row[0]), parse_dollars(row[1])
                if price is not None and size is not None:
                    parsed.append((price, size))
            if parsed:
                return tuple(parsed)
        return ()

    return OrderBook(
        ticker=ticker,
        yes_bids=_ladder("yes_dollars", "yes"),
        no_bids=_ladder("no_dollars", "no"),
    )


def candle_from_payload(ticker: str, payload: Mapping[str, Any]) -> Candle | None:
    """Parse one candlestick bar. ``end_period_ts`` is epoch **milliseconds**."""
    raw_ts = payload.get("end_period_ts")
    if raw_ts is None:
        return None
    ts_value = int(raw_ts)
    # Kalshi documents ms; tolerate seconds so a schema flip does not silently
    # produce 1970 timestamps.
    seconds = ts_value / 1000 if ts_value > 10_000_000_000 else ts_value
    price = payload.get("price") or {}
    yes_bid = payload.get("yes_bid") or {}
    yes_ask = payload.get("yes_ask") or {}
    return Candle(
        ticker=ticker,
        end_ts=datetime.fromtimestamp(seconds, tz=UTC),
        price_close=_first_dollars(price, "close_dollars", "close"),
        price_open=_first_dollars(price, "open_dollars", "open"),
        price_high=_first_dollars(price, "high_dollars", "high"),
        price_low=_first_dollars(price, "low_dollars", "low"),
        price_mean=_first_dollars(price, "mean_dollars", "mean"),
        yes_bid_close=_first_dollars(yes_bid, "close_dollars", "close"),
        yes_ask_close=_first_dollars(yes_ask, "close_dollars", "close"),
        volume=_first_dollars(payload, "volume_fp", "volume"),
        open_interest=_first_dollars(payload, "open_interest_fp", "open_interest"),
    )


# --------------------------------------------------------------------------- #
# Sibling normalisation
# --------------------------------------------------------------------------- #


def normalize_sibling_probabilities(event: GameEvent) -> dict[str, Decimal]:
    """Vig-removed ``P(win)`` per team, normalised across the event's siblings.

    Sibling mids do not sum to 1.00 (0.5425 + 0.40 in the reference sample), so
    every mid is divided by their sum. Returns ``{}`` when no mid is available.
    """
    mids = {q.team: q.yes_mid for q in event.quotes if q.team and q.yes_mid is not None}
    total = sum(mids.values(), Decimal(0))
    if not mids or total <= 0:
        return {}
    return {team: mid / total for team, mid in mids.items()}


def _clamp01(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(min(max(value, Decimal(0)), Decimal(1)))


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def _auth_headers(method: str, path: str) -> dict[str, str]:
    """RSA-PSS headers for private endpoints. Unused by market-data reads.

    Signature is ``base64(RSA-PSS-SHA256("{ts_ms}{METHOD}{path}"))``; the query
    string is excluded from signing. Requires ``cryptography``, which is not a
    project dependency -- import is deliberately local and lazy.
    """
    key_id = os.environ.get("KALSHI_API_KEY")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not key_id or not key_path:
        return {}
    try:  # pragma: no cover - optional dependency, no live auth in tests
        import base64

        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:  # pragma: no cover
        raise KalshiError(
            "Kalshi API key configured but 'cryptography' is not installed; "
            "market-data reads do not need auth -- unset KALSHI_API_KEY."
        ) from exc
    ts_ms = str(int(time.time() * 1000))  # pragma: no cover
    private_key = serialization.load_pem_private_key(  # pragma: no cover
        Path(key_path).read_bytes(), password=None
    )
    signature = private_key.sign(  # pragma: no cover
        f"{ts_ms}{method.upper()}{path}".encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256(),
    )
    return {  # pragma: no cover
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


@dataclass
class KalshiClient:
    """Thin read-only Kalshi client. No auth required for anything here."""

    base_url: str = KALSHI_BASE_URL
    timeout: float = 15.0
    max_retries: int = 3
    backoff: float = 0.5
    client: httpx.Client | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout, headers={"Accept": "application/json"})
            self._owns_client = True

    def __enter__(self) -> KalshiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self.client is not None:
            self.client.close()

    # -- transport -------------------------------------------------------- #

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        assert self.client is not None
        url = f"{self.base_url}{path}"
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.get(url, params=clean, headers=_auth_headers("GET", path))
            except httpx.HTTPError as exc:
                last_exc = exc
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_exc = KalshiError(f"{response.status_code} from {url}")
                else:
                    if response.status_code >= 400:
                        raise KalshiError(f"{response.status_code} from {url}: {response.text[:200]}")
                    return response.json()
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff * (2**attempt))
        raise KalshiError(f"GET {url} failed after {self.max_retries} attempts") from last_exc

    def _paginate(
        self, path: str, key: str, params: Mapping[str, Any] | None = None, *, max_pages: int = 20
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        for _ in range(max_pages):
            page = self._get(path, {**(params or {}), "cursor": cursor})
            yield from page.get(key, []) or []
            cursor = page.get("cursor") or None
            if not cursor:
                return

    # -- endpoints -------------------------------------------------------- #

    def list_events(
        self, *, series_ticker: str = NFL_GAME_SERIES, status: str | None = "open", limit: int = 200
    ) -> list[dict[str, Any]]:
        return list(
            self._paginate(
                "/events",
                "events",
                {"series_ticker": series_ticker, "status": status, "limit": limit},
            )
        )

    def get_event(self, event_ticker: str, *, with_nested_markets: bool = True) -> dict[str, Any]:
        return self._get(
            f"/events/{event_ticker}",
            {"with_nested_markets": str(with_nested_markets).lower()},
        )

    def list_markets(
        self,
        *,
        series_ticker: str | None = NFL_GAME_SERIES,
        event_ticker: str | None = None,
        status: str | None = "open",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return list(
            self._paginate(
                "/markets",
                "markets",
                {
                    "series_ticker": series_ticker,
                    "event_ticker": event_ticker,
                    "status": status,
                    "limit": limit,
                },
            )
        )

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str, *, depth: int = 10) -> OrderBook:
        payload = self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        return orderbook_from_payload(ticker, payload)

    def get_candlesticks(
        self,
        ticker: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        period_interval: int = 60,
        series_ticker: str = NFL_GAME_SERIES,
    ) -> list[Candle]:
        """Candlestick history. ``period_interval`` is 1 | 60 | 1440 minutes."""
        payload = self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        bars = [candle_from_payload(ticker, bar) for bar in payload.get("candlesticks", []) or []]
        return [bar for bar in bars if bar is not None]

    # -- composed --------------------------------------------------------- #

    def discover_game_events(
        self, *, status: str | None = "open", limit: int = 200
    ) -> list[GameEvent]:
        """List ``KXNFLGAME`` events with their sibling markets attached.

        Falls back to ``/markets`` grouping when ``/events`` returns no nested
        markets, so one call shape is enough either way.
        """
        events = self.list_events(status=status, limit=limit)
        by_ticker: dict[str, list[Quote]] = {}
        titles: dict[str, str] = {}
        needs_markets = False
        for payload in events:
            ticker = str(payload.get("event_ticker", ""))
            titles[ticker] = str(payload.get("title", "") or "")
            nested = payload.get("markets") or []
            if not nested:
                needs_markets = True
            by_ticker[ticker] = [quote_from_payload(m) for m in nested]
        if needs_markets or not by_ticker:
            for payload in self.list_markets(status=status, limit=limit):
                quote = quote_from_payload(payload)
                if not quote.event_ticker:
                    continue
                bucket = by_ticker.setdefault(quote.event_ticker, [])
                if not any(q.ticker == quote.ticker for q in bucket):
                    bucket.append(quote)
        return [
            GameEvent(event_ticker=ticker, title=titles.get(ticker, ""), quotes=tuple(quotes))
            for ticker, quotes in by_ticker.items()
            if quotes
        ]

    def find_event_for_game(
        self, game_id: str, schedule: pl.DataFrame, *, status: str | None = None
    ) -> GameEvent | None:
        """Locate the Kalshi event for one nflverse ``game_id``."""
        key = parse_game_id(game_id)
        for event in self.discover_game_events(status=status):
            matched = match_event_to_game(event, schedule_keys(schedule))
            if matched is not None and matched.game_id == game_id:
                return event
        log.info("kalshi: no open event found for %s (%s)", game_id, key.team_set)
        return None


# --------------------------------------------------------------------------- #
# Market rows
# --------------------------------------------------------------------------- #


def market_row_from_event(
    event: GameEvent,
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    observed_at: datetime | None = None,
    require_liquidity: bool = False,
) -> Market | None:
    """Collapse an event's sibling markets into one :class:`Market` row.

    ``implied_prob_home`` is the *normalised* home mid, so the two sides sum to
    1.0 exactly. ``kalshi_yes_price`` / ``kalshi_no_price`` are the raw,
    un-normalised sibling mids.
    """
    home = event.quote_for(home_team)
    away = event.quote_for(away_team)
    if home is None and away is None:
        return None
    probs = normalize_sibling_probabilities(event)
    anchor = home or away
    assert anchor is not None
    if require_liquidity and not anchor.is_liquid:
        log.info("kalshi: %s dropped by liquidity filter", anchor.ticker)
        return None

    home_key = normalize_team(home_team)
    away_key = normalize_team(away_team)
    implied_home = probs.get(home_key) if home_key else None
    implied_away = probs.get(away_key) if away_key else None
    if implied_home is None and implied_away is not None:
        implied_home = Decimal(1) - implied_away
    if implied_away is None and implied_home is not None:
        implied_away = Decimal(1) - implied_home

    return Market(
        source=SOURCE,
        ingested_at=datetime.now(UTC),
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        observed_at_utc=observed_at or datetime.now(UTC),
        book=BOOK,
        implied_prob_home=_clamp01(implied_home),
        implied_prob_away=_clamp01(implied_away),
        kalshi_ticker=home.ticker if home else (away.ticker if away else None),
        kalshi_yes_price=_clamp01(home.yes_mid) if home else None,
        kalshi_no_price=_clamp01(away.yes_mid) if away else None,
        kalshi_volume=_as_int(anchor.volume),
        kalshi_open_interest=_as_int(anchor.open_interest),
        bid_ask_spread=_as_float(anchor.bid_ask_spread),
    )


def market_rows_from_candles(
    candles: Sequence[Candle],
    *,
    game_id: str,
    ticker: str,
    require_liquidity: bool = False,
) -> list[Market]:
    """One :class:`Market` row per candlestick bar of the *home* market.

    Only one side has history, so ``implied_prob_away = 1 - implied_prob_home``
    without cross-sibling normalisation; treat these as coarser than snapshots.
    """
    rows: list[Market] = []
    now = datetime.now(UTC)
    for candle in candles:
        mid = candle.mid
        if mid is None:
            continue
        spread = candle.bid_ask_spread
        if require_liquidity and (spread is None or spread > MAX_BID_ASK_SPREAD):
            continue
        implied_home = _clamp01(mid)
        rows.append(
            Market(
                source=f"{SOURCE}.candles",
                ingested_at=now,
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                observed_at_utc=candle.end_ts,
                book=BOOK,
                implied_prob_home=implied_home,
                implied_prob_away=None if implied_home is None else 1.0 - implied_home,
                kalshi_ticker=ticker,
                kalshi_yes_price=implied_home,
                kalshi_no_price=None if implied_home is None else 1.0 - implied_home,
                kalshi_volume=_as_int(candle.volume),
                kalshi_open_interest=_as_int(candle.open_interest),
                bid_ask_spread=_as_float(spread),
            )
        )
    return rows


def to_frame(rows: Sequence[Market]) -> pl.DataFrame:
    """:class:`Market` rows -> polars frame (``market_collapsed`` included)."""
    if not rows:
        return pl.DataFrame(schema={"game_id": pl.Utf8, "market_only": pl.Boolean})
    records = []
    for row in rows:
        record = row.model_dump()
        record["market_collapsed"] = row.market_collapsed
        records.append(record)
    return pl.DataFrame(records, infer_schema_length=None)


# --------------------------------------------------------------------------- #
# Public API: snapshot / history / ingest
# --------------------------------------------------------------------------- #


def snapshot_game(
    game_id: str,
    *,
    schedule: pl.DataFrame,
    client: KalshiClient | None = None,
    require_liquidity: bool = False,
    with_orderbook: bool = True,
) -> Market | None:
    """Current Kalshi mid for one game as a :class:`Market` row."""
    key = parse_game_id(game_id)
    owned = client is None
    client = client or KalshiClient()
    try:
        event = client.find_event_for_game(game_id, schedule)
        if event is None:
            return None
        quotes = list(event.quotes)
        if with_orderbook:
            refreshed: list[Quote] = []
            for quote in quotes:
                if quote.yes_bid is not None and quote.yes_ask is not None:
                    refreshed.append(quote)
                    continue
                book = client.get_orderbook(quote.ticker)
                refreshed.append(
                    replace(
                        quote,
                        yes_bid=book.best_yes_bid or quote.yes_bid,
                        yes_ask=book.yes_ask or quote.yes_ask,
                        no_bid=book.best_no_bid or quote.no_bid,
                    )
                )
            event = GameEvent(
                event_ticker=event.event_ticker, title=event.title, quotes=tuple(refreshed)
            )
        return market_row_from_event(
            event,
            game_id=game_id,
            home_team=key.home_team,
            away_team=key.away_team,
            require_liquidity=require_liquidity,
        )
    finally:
        if owned:
            client.close()


def history_game(
    game_id: str,
    *,
    schedule: pl.DataFrame,
    client: KalshiClient | None = None,
    period_interval: int = 60,
    start_ts: int | None = None,
    end_ts: int | None = None,
    require_liquidity: bool = False,
) -> list[Market]:
    """Candlestick price history for the home team's market."""
    key = parse_game_id(game_id)
    owned = client is None
    client = client or KalshiClient()
    try:
        event = client.find_event_for_game(game_id, schedule)
        if event is None:
            return []
        home = event.quote_for(key.home_team)
        if home is None:
            log.info("kalshi: no home-team market in %s", event.event_ticker)
            return []
        candles = client.get_candlesticks(
            home.ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=period_interval,
        )
        return market_rows_from_candles(
            candles, game_id=game_id, ticker=home.ticker, require_liquidity=require_liquidity
        )
    finally:
        if owned:
            client.close()


def fetch_kalshi(
    game_ids: Sequence[str],
    *,
    schedule: pl.DataFrame,
    client: KalshiClient | None = None,
    require_liquidity: bool = False,
) -> pl.DataFrame:
    """Snapshot several games at once -> polars frame of :class:`Market` rows."""
    owned = client is None
    client = client or KalshiClient()
    try:
        events = client.discover_game_events(status=None)
        index = build_game_id_index(events, schedule)
        rows: list[Market] = []
        for game_id in game_ids:
            event = index.get(game_id)
            if event is None:
                log.info("kalshi: no market for %s", game_id)
                continue
            key = parse_game_id(game_id)
            row = market_row_from_event(
                event,
                game_id=game_id,
                home_team=key.home_team,
                away_team=key.away_team,
                require_liquidity=require_liquidity,
            )
            if row is not None:
                rows.append(row)
        return to_frame(rows)
    finally:
        if owned:
            client.close()


def output_dir(season: int, root: Path | None = None) -> Path:
    base = root if root is not None else config.RAW_DIR
    return base / "market" / "kalshi" / f"season={season}"


def write_rows(rows: Sequence[Market], *, season: int, root: Path | None = None) -> Path:
    """Write :class:`Market` rows to ``data/raw/market/kalshi/season=YYYY/``."""
    directory = output_dir(season, root)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"kalshi_{stamp}.parquet"
    to_frame(rows).write_parquet(path)
    return path


def ingest_kalshi(
    game_ids: Sequence[str],
    *,
    schedule: pl.DataFrame,
    season: int | None = None,
    client: KalshiClient | None = None,
    root: Path | None = None,
    require_liquidity: bool = False,
) -> Path:
    """Snapshot ``game_ids`` and persist them. Returns the parquet path."""
    if not game_ids:
        raise ValueError("ingest_kalshi requires at least one game_id")
    resolved_season = season or parse_game_id(game_ids[0]).season
    frame = fetch_kalshi(
        game_ids, schedule=schedule, client=client, require_liquidity=require_liquidity
    )
    directory = output_dir(resolved_season, root)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"kalshi_{stamp}.parquet"
    frame.write_parquet(path)
    log.info("kalshi: wrote %d rows to %s", frame.height, path)
    return path


def resolution_window(kickoff: datetime, *, hours_before: int = 48, hours_after: int = 6) -> tuple[int, int]:
    """Epoch-second ``(start_ts, end_ts)`` bracketing a kickoff, for candlesticks."""
    start = kickoff - timedelta(hours=hours_before)
    end = kickoff + timedelta(hours=hours_after)
    return int(start.timestamp()), int(end.timestamp())


__all__ = [
    "BOOK",
    "KALSHI_BASE_URL",
    "KALSHI_DEMO_BASE_URL",
    "MAX_BID_ASK_SPREAD",
    "MIN_OPEN_INTEREST",
    "NFL_GAME_SERIES",
    "Candle",
    "GameEvent",
    "GameKey",
    "KalshiClient",
    "KalshiError",
    "OrderBook",
    "Quote",
    "build_game_id_index",
    "candle_from_payload",
    "fetch_kalshi",
    "history_game",
    "ingest_kalshi",
    "market_row_from_event",
    "market_rows_from_candles",
    "match_event_to_game",
    "normalize_sibling_probabilities",
    "normalize_team",
    "orderbook_from_payload",
    "output_dir",
    "parse_dollars",
    "parse_event_ticker",
    "parse_game_id",
    "quote_from_payload",
    "resolution_window",
    "schedule_keys",
    "season_for",
    "snapshot_game",
    "to_frame",
    "write_rows",
]
