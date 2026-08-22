"""Closing lines from the nflverse schedule. **EVALUATION ONLY (ADR 0001).**

nflverse's schedule/games table carries the sportsbook columns
(``spread_line``, ``total_line``, ``home_moneyline``, ``away_moneyline``,
``home_spread_odds`` ...). This module is a *thin reader*: it locates whatever
schedule file the nflverse ingest has already written, pulls only the market
columns out of it, and emits :class:`nfl.data.schemas.Market` rows.

It deliberately does **not** import the nflverse ingest module -- it reads the
files off disk, so the two can land independently. If no file is present, point
it at one explicitly with ``path=``.

Sign convention: ``spread_home`` follows nflverse, where ``spread_line`` is the
**home team's expected margin** (positive = home favoured). It is *not* the
sportsbook's signed handicap. Since the schedule only publishes the closing
number, ``spread_home`` and ``spread_close_home`` are the same value and the
``*_open`` fields stay ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path

import polars as pl

from nfl import config
from nfl.data.schemas import Market

log = logging.getLogger(__name__)

SOURCE = "nflverse.schedules"
BOOK = "consensus"
SCHEMA_VERSION = "1"

#: Where the nflverse ingest may have parked the schedule, in preference order.
SCHEDULE_GLOBS: tuple[str, ...] = (
    "nflverse/market/*.parquet",
    "nflverse/market/**/*.parquet",
    "nflverse/schedules/**/*.parquet",
    "nflverse/schedule/**/*.parquet",
    "nflverse/games/**/*.parquet",
    "nflverse/**/schedules*.parquet",
    "nflverse/**/games*.parquet",
    "nflverse/**/schedules*.csv",
    "nflverse/**/games*.csv",
)

MARKET_COLUMNS: tuple[str, ...] = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
)

_ID_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
)


class ClosingLinesNotFound(FileNotFoundError):
    """No nflverse schedule file on disk and none supplied."""


# --------------------------------------------------------------------------- #
# Locating / loading
# --------------------------------------------------------------------------- #


def find_schedule_file(root: Path | None = None) -> Path | None:
    """First schedule-ish file under ``data/raw/`` that carries market columns."""
    base = root if root is not None else config.RAW_DIR
    if not base.exists():
        return None
    for pattern in SCHEDULE_GLOBS:
        matches = sorted(base.glob(pattern))
        for match in matches:
            if match.is_file():
                return match
    return None


def _read_any(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10_000, try_parse_dates=False)
    return pl.read_parquet(path)


def load_schedule(
    path: Path | None = None,
    *,
    root: Path | None = None,
    seasons: Iterable[int] | None = None,
) -> pl.DataFrame:
    """Load the nflverse schedule frame (id columns + whatever market columns exist).

    Also the schedule source for :mod:`nfl.data.market.kalshi`'s game mapping.
    """
    resolved = path or find_schedule_file(root)
    if resolved is None:
        raise ClosingLinesNotFound(
            "no nflverse schedule parquet found under "
            f"{root if root is not None else config.RAW_DIR}; pass path= explicitly"
        )
    frame = _read_any(Path(resolved))
    keep = [c for c in (*_ID_COLUMNS, *MARKET_COLUMNS) if c in frame.columns]
    missing_ids = {"game_id", "season", "home_team", "away_team"} - set(frame.columns)
    if missing_ids:
        raise KeyError(f"{resolved} is not an nflverse schedule (missing {sorted(missing_ids)})")
    frame = frame.select(keep)
    if seasons is not None:
        wanted = list(seasons)
        if wanted:
            frame = frame.filter(pl.col("season").is_in(wanted))
    return frame


# --------------------------------------------------------------------------- #
# Odds maths
# --------------------------------------------------------------------------- #


def american_to_probability(odds: float | int | None) -> float | None:
    """American moneyline -> raw (vig-inclusive) implied probability."""
    if odds is None:
        return None
    value = float(odds)
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return -value / (-value + 100.0)


def devig(p_home: float | None, p_away: float | None) -> tuple[float | None, float | None]:
    """Proportional (multiplicative) vig removal across the two sides."""
    if p_home is None and p_away is None:
        return None, None
    if p_home is None:
        return (None, None) if p_away is None else (1.0 - p_away, p_away)
    if p_away is None:
        return p_home, 1.0 - p_home
    total = p_home + p_away
    if total <= 0:
        return None, None
    return p_home / total, p_away / total


def _kickoff(row: dict[str, object]) -> datetime:
    """Best-effort kickoff timestamp; the closing line is observed at kickoff."""
    season = _i(row.get("season")) or 1970
    fallback = date(season, 9, 1)  # season opener-ish, when gameday is unusable

    gameday = row.get("gameday")
    if isinstance(gameday, datetime):
        day = gameday.date()
    elif isinstance(gameday, date):
        day = gameday
    elif isinstance(gameday, str) and gameday:
        try:
            day = date.fromisoformat(gameday[:10])
        except ValueError:
            day = fallback
    else:
        day = fallback

    gametime = row.get("gametime")
    clock = time(13, 0)
    if isinstance(gametime, time):
        clock = gametime
    elif isinstance(gametime, str) and ":" in gametime:
        try:
            hour, minute = gametime.split(":")[:2]
            clock = time(int(hour), int(minute))
        except ValueError:
            pass
    return datetime.combine(day, clock, tzinfo=UTC)


def _f(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # drop NaN


def _i(value: object) -> int | None:
    out = _f(value)
    return None if out is None else int(out)


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def market_rows(frame: pl.DataFrame) -> list[Market]:
    """Schedule frame -> :class:`Market` rows, one per game with any line."""
    rows: list[Market] = []
    now = datetime.now(UTC)
    for row in frame.iter_rows(named=True):
        ml_home = _i(row.get("home_moneyline"))
        ml_away = _i(row.get("away_moneyline"))
        spread = _f(row.get("spread_line"))
        total = _f(row.get("total_line"))
        if ml_home is None and ml_away is None and spread is None and total is None:
            continue
        p_home, p_away = devig(
            american_to_probability(ml_home), american_to_probability(ml_away)
        )
        rows.append(
            Market(
                source=SOURCE,
                ingested_at=now,
                schema_version=SCHEMA_VERSION,
                game_id=str(row["game_id"]),
                observed_at_utc=_kickoff(row),
                book=BOOK,
                spread_home=spread,
                spread_close_home=spread,
                total_close=total,
                moneyline_home=ml_home,
                moneyline_away=ml_away,
                implied_prob_home=p_home,
                implied_prob_away=p_away,
            )
        )
    return rows


def to_frame(rows: Sequence[Market]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"game_id": pl.Utf8, "market_only": pl.Boolean})
    records = []
    for row in rows:
        record = row.model_dump()
        record["market_collapsed"] = row.market_collapsed
        records.append(record)
    return pl.DataFrame(records, infer_schema_length=None)


def fetch_closing_lines(
    seasons: Iterable[int] | None = None,
    *,
    path: Path | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Read closing lines for ``seasons`` as a frame of :class:`Market` rows."""
    schedule = load_schedule(path, root=root, seasons=seasons)
    return to_frame(market_rows(schedule))


def output_dir(season: int, root: Path | None = None) -> Path:
    base = root if root is not None else config.RAW_DIR
    return base / "market" / "closing_lines" / f"season={season}"


def ingest_closing_lines(
    seasons: Iterable[int],
    *,
    path: Path | None = None,
    root: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Write one parquet per season. Idempotent: skips existing unless ``force``."""
    written: list[Path] = []
    for season in seasons:
        directory = output_dir(season, root)
        target = directory / "closing_lines.parquet"
        if target.exists() and not force:
            log.info("closing_lines: %s exists, skipping", target)
            written.append(target)
            continue
        frame = fetch_closing_lines([season], path=path, root=root)
        directory.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        log.info("closing_lines: wrote %d rows to %s", frame.height, target)
        written.append(target)
    return written


__all__ = [
    "BOOK",
    "MARKET_COLUMNS",
    "SOURCE",
    "ClosingLinesNotFound",
    "american_to_probability",
    "devig",
    "fetch_closing_lines",
    "find_schedule_file",
    "ingest_closing_lines",
    "load_schedule",
    "market_rows",
    "output_dir",
    "to_frame",
]
