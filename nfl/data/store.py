"""DuckDB catalog over the Parquet lake.

Per ``docs/ADR/0002-storage-parquet-duckdb.md`` the database file holds no data:
every table is a *view* over a parquet glob, so a fresh ingest is visible
immediately and the catalog can be deleted and rebuilt at any time without loss.

Naming:

* raw tables become ``<source>_<table>`` — ``nflverse_pbp``, ``nflverse_market``
* processed tables keep their bare name — ``coach_records``

Views are created with ``hive_partitioning`` on, so ``WHERE season = 2024``
prunes at the directory level and never opens the other seasons' files.

    >>> from nfl.data.store import open_db, refresh_views
    >>> con = open_db()
    >>> refresh_views(con)
    ['nflverse_pbp', 'nflverse_schedules', ...]
    >>> con.execute("select count(*) from nflverse_pbp where season = 2024").fetchone()
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb

from nfl.config import DUCKDB_PATH, PROCESSED_DIR, RAW_DIR

__all__ = [
    "ViewSpec",
    "connect",
    "discover_tables",
    "open_db",
    "refresh_views",
    "table_names",
]


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """One view: its name, the directory it covers, and the glob behind it."""

    view: str
    directory: Path
    layer: str  # "raw" | "processed"

    @property
    def glob(self) -> str:
        return str(self.directory / "**" / "*.parquet")


def open_db(path: Path | None = None, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the project DuckDB catalog.

    ``path=":memory:"`` is accepted for tests. The parent directory is created
    for a file path, since ``data/`` is gitignored and may not exist yet.
    """
    target = DUCKDB_PATH if path is None else Path(path)
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(target), read_only=read_only)


@contextmanager
def connect(path: Path | None = None, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """:func:`open_db` as a context manager, so the file lock is always released."""
    con = open_db(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def _has_parquet(directory: Path) -> bool:
    return any(directory.rglob("*.parquet"))


def discover_tables(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[ViewSpec]:
    """Find every parquet table on disk and name the view it should get.

    Raw tables live at ``<raw>/<source>/<table>/``; processed tables at
    ``<processed>/<table>/``. Directories with no parquet under them are ignored,
    so a half-finished ingest does not create a broken view.
    """
    raw_root = RAW_DIR if raw_dir is None else Path(raw_dir)
    processed_root = PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    specs: list[ViewSpec] = []

    if raw_root.is_dir():
        for source_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            for table_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                if _has_parquet(table_dir):
                    specs.append(
                        ViewSpec(f"{source_dir.name}_{table_dir.name}", table_dir, "raw")
                    )

    if processed_root.is_dir():
        for table_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
            if _has_parquet(table_dir):
                specs.append(ViewSpec(table_dir.name, table_dir, "processed"))

    return specs


def refresh_views(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[str]:
    """(Re)create one view per parquet table found on disk; return the view names.

    Idempotent, and safe to call after every ingest: views are replaced, and
    views whose directory has disappeared are dropped so the catalog never
    points at a path that is gone.
    """
    specs = discover_tables(raw_dir, processed_dir)
    wanted = {spec.view for spec in specs}

    for spec in specs:
        # DuckDB refuses prepared parameters inside CREATE VIEW, so the glob is
        # inlined; it is a path we constructed, and single quotes are escaped.
        glob = spec.glob.replace("'", "''")
        con.execute(
            f'CREATE OR REPLACE VIEW "{spec.view}" AS SELECT * FROM '
            f"read_parquet('{glob}', hive_partitioning => true, union_by_name => true)"
        )

    for stale in set(_existing_views(con)) - wanted:
        con.execute(f'DROP VIEW IF EXISTS "{stale}"')

    return sorted(wanted)


def _existing_views(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_type = 'VIEW'"
    ).fetchall()
    return [row[0] for row in rows]


def table_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Every view currently registered in the catalog."""
    return sorted(_existing_views(con))
