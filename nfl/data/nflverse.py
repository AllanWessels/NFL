"""nflverse ingestion — play-by-play, schedules and the weekly report tables.

Everything here goes through :mod:`nflreadpy` (``nfl_data_py`` is archived and
must not be used), which returns ``polars`` frames read from the nflverse
GitHub release parquet assets. The one exception is ESPN QBR: nflreadpy 0.1.5
ships no loader for it, so it is fetched from its release URL directly.

Layout, per ``docs/ADR/0002-storage-parquet-duckdb.md`` and the Phase 3
conventions::

    data/raw/nflverse/<table>/season=YYYY/part-0.parquet

``season`` is kept inside the file as well as in the path. DuckDB and polars
both tolerate the duplicate and it keeps a single file self-describing.

ADR 0001 is enforced structurally at *ingest* time, not left to the feature
builder: :func:`_split_market` lifts the eight betting columns out of
``schedules`` into a physically separate ``market`` table tagged
``market_only = True``. A schedules parquet that still carried them would be one
careless ``select *`` away from contaminating a model.

Public surface:

* :func:`fetch_nflverse` — ``(table, seasons) -> polars.DataFrame``
* :func:`ingest_nflverse` — ``(seasons, ...) -> Path`` (writes parquet)
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl.config import RAW_DIR, SEASON_START
from nfl.data.market_blocklist import MARKET_QUOTE_COLUMNS

__all__ = [
    "SOURCE",
    "TABLES",
    "TABLE_NAMES",
    "IngestResult",
    "configure_cache",
    "ingested_at_now",
    "fetch_nflverse",
    "ingest_nflverse",
    "parse_seasons",
    "raw_table_dir",
    "season_partition_path",
]

SOURCE = "nflverse"
SCHEMA_VERSION = "1"

#: ESPN QBR has no nflreadpy loader; hit the release asset directly.
QBR_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.parquet"
)

_HTTP_TIMEOUT = 120.0


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def configure_cache(cache_dir: Path | None = None) -> Path:
    """Point nflreadpy at a persistent on-disk cache under ``DATA_DIR/cache``.

    nflreadpy defaults to an in-memory cache, which means every new process
    re-downloads the ~300 MB of play-by-play parquet. Called for its side effect
    at the top of every ingest.
    """
    from nfl.config import DATA_DIR

    target = Path(cache_dir) if cache_dir is not None else DATA_DIR / "cache" / "nflreadpy"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["NFLREADPY_CACHE"] = "filesystem"
    os.environ["NFLREADPY_CACHE_DIR"] = str(target)
    # The config object is instantiated at import time, so the env vars alone
    # are not enough for an already-imported nflreadpy.
    from nflreadpy.config import CacheMode, update_config

    # update_config() is a bare setattr with no coercion, so pass the enum and a
    # real Path rather than the strings the env vars would have been parsed from.
    update_config(cache_mode=CacheMode.FILESYSTEM, cache_dir=target)
    return target


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_pbp(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_pbp(seasons)


def _load_schedules(seasons: list[int]) -> pl.DataFrame:
    """Schedules with the betting columns removed — they live in ``market``."""
    import nflreadpy as nr

    df = nr.load_schedules(seasons)
    df = df.filter(pl.col("season").is_in(seasons))
    return df.drop([c for c in df.columns if c in MARKET_QUOTE_COLUMNS])


def _load_market(seasons: list[int]) -> pl.DataFrame:
    """The betting columns lifted out of ``schedules``, tagged per ADR 0001."""
    import nflreadpy as nr

    df = nr.load_schedules(seasons).filter(pl.col("season").is_in(seasons))
    return _split_market(df)


def _load_weekly_rosters(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_rosters_weekly(seasons)


def _load_injuries(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_injuries(seasons)


def _load_depth_charts(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_depth_charts(seasons)


def _load_snap_counts(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_snap_counts(seasons)


def _load_participation(seasons: list[int]) -> pl.DataFrame:
    import nflreadpy as nr

    return nr.load_participation(seasons)


def _load_ngs(seasons: list[int]) -> pl.DataFrame:
    """Next Gen Stats: three stat types stacked with a ``stat_type`` discriminator."""
    import nflreadpy as nr

    return _stack_variants(
        lambda variant: nr.load_nextgen_stats(seasons, stat_type=variant),
        variants=("passing", "receiving", "rushing"),
        column="stat_type",
    )


def _load_pfr_advstats(seasons: list[int]) -> pl.DataFrame:
    """PFR weekly advanced stats: four stat types stacked.

    ``summary_level="season"`` silently ignores ``seasons=``, so only the weekly
    level is ingested here.
    """
    import nflreadpy as nr

    return _stack_variants(
        lambda variant: nr.load_pfr_advstats(seasons, stat_type=variant, summary_level="week"),
        variants=("pass", "rush", "rec", "def"),
        column="stat_type",
    )


def _load_qbr(seasons: list[int]) -> pl.DataFrame:
    """ESPN weekly QBR, fetched from the release asset (no nflreadpy loader)."""
    import httpx

    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                response = client.get(QBR_URL)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 2:
                    raise
        else:  # pragma: no cover - the loop always breaks or raises
            raise RuntimeError("unreachable")
    df = pl.read_parquet(io.BytesIO(response.content))
    return df.filter(pl.col("season").is_in(seasons))


def _stack_variants(
    load: Callable[[str], pl.DataFrame],
    *,
    variants: Sequence[str],
    column: str,
) -> pl.DataFrame:
    """Load several vendor variants of one dataset and stack them with a tag column."""
    frames: list[pl.DataFrame] = []
    for variant in variants:
        frame = load(variant)
        if frame.height == 0:
            continue
        frames.append(frame.with_columns(pl.lit(variant, dtype=pl.String).alias(column)))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# --------------------------------------------------------------------------- #
# Market split (ADR 0001)
# --------------------------------------------------------------------------- #

#: Identity columns carried alongside the quotes so the market table is joinable.
_MARKET_KEYS: tuple[str, ...] = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "away_team",
    "home_team",
)


def _split_market(schedules: pl.DataFrame) -> pl.DataFrame:
    """Build the ``market`` table from a raw schedules frame.

    The lines in ``games.parquet`` are a *closing* market consensus with no
    observation timestamp of their own, so ``observed_at_utc`` is set to
    kickoff, which is what "closing" means, and ``line_type`` records the
    approximation. Raw vendor column names are preserved: this is the raw layer,
    and mapping onto :class:`nfl.data.schemas.Market` is a processed-layer job.
    """
    keys = [c for c in _MARKET_KEYS if c in schedules.columns]
    quotes = [c for c in schedules.columns if c in MARKET_QUOTE_COLUMNS]
    out = schedules.select(keys + quotes)
    return out.with_columns(
        _kickoff_utc(out).alias("observed_at_utc"),
        pl.lit(True, dtype=pl.Boolean).alias("market_only"),
        pl.lit("consensus", dtype=pl.String).alias("book"),
        pl.lit("close", dtype=pl.String).alias("line_type"),
    )


def _kickoff_utc(df: pl.DataFrame) -> pl.Expr:
    """``gameday`` + ``gametime`` (US Eastern, DST-aware) as a naive-UTC timestamp."""
    if "gameday" not in df.columns or "gametime" not in df.columns:
        return pl.lit(None, dtype=pl.Datetime("us"))
    return (
        pl.concat_str(
            [pl.col("gameday").cast(pl.String), pl.col("gametime").cast(pl.String)],
            separator=" ",
        )
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone("America/New_York", ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
    )


# --------------------------------------------------------------------------- #
# Table registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One raw nflverse table: how to load it and how far back it goes."""

    name: str
    loader: Callable[[list[int]], pl.DataFrame]
    source: str
    min_season: int = SEASON_START
    #: Some vendor assets are a single all-seasons file; those are fetched once
    #: and split locally rather than refetched per season.
    whole_history: bool = False
    note: str = ""


TABLES: dict[str, TableSpec] = {
    spec.name: spec
    for spec in (
        TableSpec("pbp", _load_pbp, "nflverse.pbp"),
        TableSpec("schedules", _load_schedules, "nflverse.sched", whole_history=True),
        TableSpec(
            "market",
            _load_market,
            "market",
            whole_history=True,
            note="ADR 0001: evaluation only, never a feature.",
        ),
        TableSpec("weekly_rosters", _load_weekly_rosters, "nflverse.rosters"),
        TableSpec("injuries", _load_injuries, "nflverse.inj"),
        TableSpec("depth_charts", _load_depth_charts, "nflverse.depth"),
        TableSpec("snap_counts", _load_snap_counts, "nflverse.snaps", min_season=2012),
        TableSpec(
            "ngs",
            _load_ngs,
            "nflverse.ngs",
            min_season=2016,
            whole_history=True,
            note="Single all-seasons asset per stat type.",
        ),
        TableSpec("pfr_advstats", _load_pfr_advstats, "nflverse.pfr", min_season=2018),
        TableSpec(
            "participation",
            _load_participation,
            "nflverse.part",
            min_season=2016,
            note="NFL stopped supplying personnel data after 2023.",
        ),
        TableSpec(
            "qbr",
            _load_qbr,
            "espn.qbr",
            whole_history=True,
            note="No nflreadpy loader; fetched from the release asset.",
        ),
    )
}

TABLE_NAMES: tuple[str, ...] = tuple(TABLES)

#: Default set for ``nfl ingest nflverse`` with no ``--tables``.
DEFAULT_TABLES: tuple[str, ...] = TABLE_NAMES


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def raw_table_dir(table: str, root: Path | None = None) -> Path:
    """``<root>/nflverse/<table>``."""
    return (root or RAW_DIR) / SOURCE / table


def season_partition_path(table: str, season: int, root: Path | None = None) -> Path:
    """``<root>/nflverse/<table>/season=YYYY/part-0.parquet``."""
    return raw_table_dir(table, root) / f"season={season}" / "part-0.parquet"


def parse_seasons(spec: str) -> list[int]:
    """Parse ``"2010-2025"``, ``"2024"`` or ``"2016,2018-2020"`` into a season list.

    Raises ``ValueError`` on a malformed or inverted range, or on a season
    before :data:`nfl.config.SEASON_START`.
    """
    seasons: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, _, end_s = chunk.partition("-")
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"inverted season range: {chunk!r}")
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(chunk))
    if not seasons:
        raise ValueError(f"no seasons parsed from {spec!r}")
    too_early = sorted(s for s in seasons if s < SEASON_START)
    if too_early:
        raise ValueError(f"seasons before {SEASON_START} are out of scope: {too_early}")
    return sorted(seasons)


# --------------------------------------------------------------------------- #
# Fetch / ingest
# --------------------------------------------------------------------------- #


def ingested_at_now() -> datetime:
    """The provenance clock: UTC, stored *naive*.

    Every timestamp this project writes to parquet is UTC-without-a-tzinfo.
    Timezone-aware parquet columns come back as ``TIMESTAMP WITH TIME ZONE`` in
    DuckDB, whose Python conversion path needs ``pytz`` — a dependency we do not
    carry — so a tz-aware column turns an ordinary ``SELECT`` into an
    ``InvalidInputException``. Naive UTC costs nothing and reads everywhere.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _stamp(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Attach the universal provenance columns (``docs/DATA_MODEL.md``)."""
    return df.with_columns(
        pl.lit(source, dtype=pl.String).alias("source"),
        pl.lit(ingested_at_now()).alias("ingested_at"),
        pl.lit(SCHEMA_VERSION, dtype=pl.String).alias("schema_version"),
    )


def fetch_nflverse(table: str, seasons: Iterable[int]) -> pl.DataFrame:
    """Fetch one table for ``seasons`` and return it stamped with provenance.

    Seasons before the table's first available season are dropped silently; if
    that leaves nothing, an empty frame comes back.
    """
    spec = _spec(table)
    wanted = [s for s in sorted(set(seasons)) if s >= spec.min_season]
    if not wanted:
        return pl.DataFrame()
    df = spec.loader(wanted)
    if df.height == 0:
        return df
    if "season" in df.columns:
        df = df.filter(pl.col("season").is_in(wanted))
    return _stamp(df, spec.source)


def _spec(table: str) -> TableSpec:
    try:
        return TABLES[table]
    except KeyError:
        raise KeyError(f"unknown nflverse table {table!r}; known: {list(TABLES)}") from None


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one ``(table, season)`` write did."""

    table: str
    season: int
    path: Path
    rows: int
    written: bool
    reason: str = ""

    @property
    def skipped(self) -> bool:
        return not self.written


def ingest_nflverse(
    seasons: Iterable[int],
    tables: Iterable[str] | None = None,
    *,
    force: bool = False,
    root: Path | None = None,
    results: list[IngestResult] | None = None,
    on_result: Callable[[IngestResult], None] | None = None,
) -> Path:
    """Ingest nflverse tables to hive-partitioned parquet; return the source root.

    Idempotent: an existing season partition is left alone unless ``force``.
    Pass ``results`` to collect per-partition detail, or ``on_result`` for a
    progress callback.
    """
    configure_cache()
    root = root or RAW_DIR
    season_list = sorted(set(seasons))
    table_list = list(tables) if tables is not None else list(DEFAULT_TABLES)
    for table in table_list:
        for result in _ingest_table(table, season_list, force=force, root=root):
            if results is not None:
                results.append(result)
            if on_result is not None:
                on_result(result)
    return root / SOURCE


def _ingest_table(
    table: str,
    seasons: list[int],
    *,
    force: bool,
    root: Path,
) -> list[IngestResult]:
    spec = _spec(table)
    in_scope = [s for s in seasons if s >= spec.min_season]
    out: list[IngestResult] = []

    skipped = [s for s in seasons if s < spec.min_season]
    for season in skipped:
        out.append(
            IngestResult(
                table,
                season,
                season_partition_path(table, season, root),
                0,
                False,
                f"not available before {spec.min_season}",
            )
        )

    todo = (
        in_scope
        if force
        else [s for s in in_scope if not season_partition_path(table, s, root).exists()]
    )
    for season in sorted(set(in_scope) - set(todo)):
        out.append(
            IngestResult(
                table,
                season,
                season_partition_path(table, season, root),
                0,
                False,
                "exists",
            )
        )
    if not todo:
        return out

    # A whole-history asset is one download regardless of how many seasons are
    # missing, so fetch it once and slice; per-season assets are fetched per
    # season to keep peak memory near one season of play-by-play.
    batches = [todo] if spec.whole_history else [[s] for s in todo]
    for batch in batches:
        frame = fetch_nflverse(table, batch)
        for season in batch:
            part = (
                frame.filter(pl.col("season") == season)
                if frame.height and "season" in frame.columns
                else frame
            )
            path = season_partition_path(table, season, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            if part.height == 0:
                out.append(IngestResult(table, season, path, 0, False, "no rows upstream"))
                continue
            part.write_parquet(path, compression="zstd")
            out.append(IngestResult(table, season, path, part.height, True))
    return out
