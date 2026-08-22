# ADR 0002 — Storage is Parquet files queried through DuckDB

- **Status:** Accepted
- **Date:** 2026-08-22
- **Deciders:** Fable (architect), Allan Wessels

## Context

The dataset is play-by-play NFL data from 2010 to present: roughly 50k plays per
season across ~285 games, so on the order of 800k–900k play rows plus rosters,
injuries, weather and schedules. Wide feature frames may reach a few hundred
columns. This is comfortably "large-ish local data", not big data — tens of GB
at the very worst, typically single-digit GB.

Access patterns are:
- bulk columnar scans for training-set construction,
- point lookups for a single game/play at inference time,
- append-only writes to a prediction ledger,
- ad-hoc analytical SQL during evaluation.

The system runs locally on one workstation, with optional later sync to GCS.

## Decision

Store everything as **Parquet files on disk**, organised by entity and
partitioned by season, and query them through **DuckDB**.

```
data/
  raw/         <source>/season=<yyyy>/*.parquet      # as fetched, unmodified
  processed/   <entity>/season=<yyyy>/*.parquet      # validated, normalised
  processed/nfl.duckdb                               # views over the Parquet
  ledger/      predictions/date=<yyyy-mm-dd>/*.parquet
```

DuckDB **attaches** the Parquet files as views rather than ingesting copies, so
Parquet remains the single source of truth and the `.duckdb` file is a
disposable query layer that can be rebuilt at any time.

Polars is the in-process transform engine (it reads/writes Parquet natively and
shares Arrow memory with DuckDB and pyarrow with zero copies).

## Consequences

**Positive**

- No server, no daemon, no migrations, no connection strings. `git clone` +
  `uv sync` + `nfl ingest` reproduces the whole store.
- Columnar scans are fast for training frames; DuckDB gives real SQL for
  evaluation without an ETL step.
- Parquet is the native output format of nflreadpy, so ingest is nearly a
  pass-through.
- GCS sync is a path-prefix change (`data/` → `gs://…/data/`); DuckDB and Polars
  both read remote Parquet.
- Files are immutable and hashable, which makes the feature-hash column in the
  prediction ledger meaningful.

**Negative**

- No enforced referential integrity or constraints; schema discipline is our
  responsibility. Mitigated by pydantic v2 models in `nfl/data/schemas.py`
  validating at ingest boundaries.
- Row-level updates are awkward. Accepted: everything here is append-only or
  full-partition rewrite.
- Concurrent writers are unsafe. Accepted: this is a single-process,
  event-driven system.

## Alternatives considered

- **Postgres.** Rejected: operational overhead for a single-user local system,
  and row-store scans are the wrong shape for training-frame construction.
- **DuckDB as the store (native `.duckdb` tables).** Rejected: it makes the
  database file the source of truth, which is a single binary blob that is
  harder to sync incrementally to GCS, harder to diff, and couples the data to a
  DuckDB storage-format version.
- **Plain CSV.** Rejected: no types, no compression, no predicate pushdown.
- **SQLite.** Rejected: row-store, poor analytical performance, no Parquet.
