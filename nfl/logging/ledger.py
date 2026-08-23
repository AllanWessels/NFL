"""The prediction ledger — append-only Parquet under ``data/ledger/predictions``.

One row per prediction (``docs/DATA_MODEL.md`` §13). The ledger is the
reproducibility anchor for the whole project: given a row you can say which
model version produced which probability, from which snapshot of which game,
against which exact feature vector (``feature_hash``), on which commit
(``git_sha``).

Three properties are structural rather than conventional:

**Append-only.** :func:`append` never rewrites an existing part file; every call
lands a new ``season=YYYY/part-<ts>-<id>.parquet``. The single exception is
:func:`resolve_ledger`, which fills the ``resolved_*`` columns that were written
null at predict time — it fills holes, it never edits a prediction.

**Market columns are eval-only.** ``market_implied_home``,
``kalshi_yes_price_home``, ``closing_spread_home`` and everything derived from
them live in :data:`EVAL_ONLY_COLUMNS`. They are stored (that is the point of
measuring edge) but ADR 0001 forbids them reaching a model, so
:func:`drop_eval_only` exists and :func:`feature_safe` asserts the split. A
ledger frame must never be handed to a feature builder without going through
one of those.

**The snapshot is recorded, not recomputed.** ``snapshot_period`` /
``snapshot_clock`` / ``snapshot_game_seconds_remaining`` plus the key game state
(score, possession, down and distance) are copied onto the row so a prediction
stays interpretable even if the upstream play-by-play is re-ingested.

    >>> from nfl.logging import ledger
    >>> row = ledger.prediction_row(
    ...     game_id="2024_18_KC_DEN", p_home=0.71, confidence=0.6,
    ...     model_name="gbdt", model_version="20240101T000000",
    ... )
    >>> ledger.append([row], root=tmp)          # doctest: +SKIP
    >>> ledger.resolve_ledger(schedules, root=tmp)   # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from nfl import config

log = logging.getLogger(__name__)

#: ``pl.Int32`` is a class, ``pl.Datetime("us", "UTC")`` an instance; the schema
#: below uses both, so it is typed as either (same alias as ``features.assemble``).
PolarsDataType = DataType | DataTypeClass

__all__ = [
    "EVAL_ONLY_COLUMNS",
    "LEDGER_SCHEMA",
    "LEDGER_TABLE",
    "RESOLUTION_COLUMNS",
    "ResolveReport",
    "append",
    "drop_eval_only",
    "empty_ledger",
    "feature_hash",
    "feature_safe",
    "git_sha",
    "ledger_dir",
    "new_prediction_id",
    "prediction_row",
    "read_ledger",
    "register_ledger_views",
    "resolve_ledger",
    "season_from_game_id",
]

#: DuckDB view name for the ledger.
LEDGER_TABLE = "prediction_ledger"

_PREDICTIONS = "predictions"
SCHEMA_VERSION = "1"

#: Every column, in write order, with its Parquet dtype. The schema is fixed:
#: :func:`append` reindexes whatever it is handed onto exactly these columns so
#: every part file in the lake is union-compatible.
LEDGER_SCHEMA: dict[str, PolarsDataType] = {
    # -- identity ---------------------------------------------------------- #
    "prediction_id": pl.String,
    "predicted_at_utc": pl.Datetime("us", "UTC"),
    "game_id": pl.String,
    "season": pl.Int32,
    "week": pl.Int32,
    "home_team": pl.String,
    "away_team": pl.String,
    # -- snapshot ---------------------------------------------------------- #
    "source": pl.String,  # espn | replay | backtest
    "is_live": pl.Boolean,
    "snapshot_play_index": pl.Int64,
    "snapshot_period": pl.Int32,
    "snapshot_clock": pl.String,  # "MM:SS" within the period
    "snapshot_game_seconds_remaining": pl.Int32,
    # -- key game state (interpretability, not features) ------------------- #
    "home_score": pl.Int32,
    "away_score": pl.Int32,
    "posteam": pl.String,
    "down": pl.Int32,
    "ydstogo": pl.Int32,
    "yardline_100": pl.Int32,
    # -- model ------------------------------------------------------------- #
    "model_name": pl.String,
    "model_version": pl.String,
    "model_kind": pl.String,  # baseline | gbdt | mlp | seq | ensemble
    "ensemble_components": pl.String,  # json: [{"name":..,"version":..,"weight":..}]
    "ensemble_member_probs": pl.List(pl.Float64),
    # -- output ------------------------------------------------------------ #
    "p_home": pl.Float64,
    "p_away": pl.Float64,
    "confidence": pl.Float64,
    "confidence_method": pl.String,
    # -- reproducibility --------------------------------------------------- #
    "feature_hash": pl.String,
    "feature_set_version": pl.String,
    "features_nonnull_count": pl.Int32,
    "features_total_count": pl.Int32,
    "git_sha": pl.String,
    "latency_ms": pl.Float64,
    # -- provenance (CLAUDE.md conventions) -------------------------------- #
    "schema_version": pl.String,
    "ingested_at": pl.Datetime("us", "UTC"),
    # -- resolution (filled by resolve_ledger) ----------------------------- #
    "resolved_home_win": pl.Boolean,
    "resolved_home_score": pl.Int32,
    "resolved_away_score": pl.Int32,
    "resolved_at_utc": pl.Datetime("us", "UTC"),
    "brier": pl.Float64,
    "log_loss": pl.Float64,
    # -- market: EVALUATION ONLY (ADR 0001) -------------------------------- #
    "kalshi_yes_price_home": pl.Float64,
    "closing_spread_home": pl.Float64,
    "market_implied_home": pl.Float64,
    "closing_market_implied_home": pl.Float64,
    "market_book": pl.String,
    "edge_vs_market": pl.Float64,
}

#: Columns filled after the fact by :func:`resolve_ledger`.
RESOLUTION_COLUMNS: tuple[str, ...] = (
    "resolved_home_win",
    "resolved_home_score",
    "resolved_away_score",
    "resolved_at_utc",
    "brier",
    "log_loss",
)

#: Market-sourced (or market-derived) columns. Stored for edge measurement,
#: never a model input — ADR 0001 / CLAUDE.md hard rule 1.
EVAL_ONLY_COLUMNS: frozenset[str] = frozenset(
    {
        "kalshi_yes_price_home",
        "closing_spread_home",
        "market_implied_home",
        "closing_market_implied_home",
        "market_book",
        "edge_vs_market",
    }
)

#: Defaults applied by :func:`prediction_row` when a caller omits a column.
_DEFAULTS: dict[str, Any] = {
    "source": "replay",
    "model_kind": "baseline",
    "confidence_method": "ensemble_spread",
    "feature_set_version": "1",
    "schema_version": SCHEMA_VERSION,
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def ledger_dir(root: Path | None = None) -> Path:
    """``data/ledger/predictions`` unless ``root`` overrides ``data/ledger``."""
    base = config.LEDGER_DIR if root is None else Path(root)
    return base / _PREDICTIONS


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def new_prediction_id() -> str:
    """A fresh UUID4 for one prediction."""
    return str(uuid.uuid4())


def season_from_game_id(game_id: str | None) -> int | None:
    """``'2024_18_KC_DEN'`` -> ``2024``. ``None`` when it does not parse."""
    if not game_id:
        return None
    head = str(game_id).split("_", 1)[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


def week_from_game_id(game_id: str | None) -> int | None:
    """``'2024_18_KC_DEN'`` -> ``18``."""
    if not game_id:
        return None
    parts = str(game_id).split("_")
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None


@lru_cache(maxsize=1)
def git_sha(short: bool = True) -> str:
    """The commit the process is running from, or ``"unknown"`` outside a repo.

    Cached: it cannot change inside one process, and shelling out per prediction
    would dominate inference latency.
    """
    env = os.environ.get("NFL_GIT_SHA")
    if env:
        return env
    cmd = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(  # noqa: S603
            cmd,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"
    sha = out.stdout.strip()
    return sha or "unknown"


def feature_hash(values: Iterable[Any], names: Sequence[str] | None = None) -> str:
    """SHA-256 over the *ordered* feature vector — the reproducibility anchor.

    ``names`` is folded in first when given, so two vectors that happen to carry
    the same numbers under different feature sets hash differently. ``None`` and
    NaN both serialise as ``"null"``, so an imputed-to-null value is stable.
    """
    digest = hashlib.sha256()
    if names is not None:
        digest.update("\x1f".join(names).encode())
        digest.update(b"\x1e")
    for value in values:
        digest.update(_hashable(value).encode())
        digest.update(b"\x1f")
    return digest.hexdigest()


def _hashable(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if value != value:  # NaN
            return "null"
        return repr(float(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def prediction_row(
    *,
    game_id: str,
    p_home: float,
    confidence: float,
    model_name: str,
    model_version: str,
    prediction_id: str | None = None,
    predicted_at_utc: datetime | None = None,
    ensemble_components: Sequence[Mapping[str, Any]] | str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build one ledger row dict, defaulted and self-consistent.

    ``p_away`` is derived, ``season``/``week`` come off the ``game_id`` when not
    supplied, ``is_live`` follows ``source == "espn"``, and
    ``ensemble_components`` is JSON-encoded. Unknown keys raise, so a typo in a
    column name is a failure at the call site rather than a silently dropped
    field.
    """
    unknown = sorted(set(fields) - set(LEDGER_SCHEMA))
    if unknown:
        raise KeyError(f"not ledger columns: {unknown}")

    row: dict[str, Any] = dict(_DEFAULTS)
    row.update(fields)
    row["prediction_id"] = prediction_id or new_prediction_id()
    row["predicted_at_utc"] = predicted_at_utc or datetime.now(UTC)
    row["game_id"] = game_id
    row["p_home"] = float(p_home)
    row["p_away"] = 1.0 - float(p_home)
    row["confidence"] = float(confidence)
    row["model_name"] = model_name
    row["model_version"] = model_version
    row.setdefault("git_sha", git_sha())
    row.setdefault("ingested_at", row["predicted_at_utc"])
    row.setdefault("season", season_from_game_id(game_id))
    row.setdefault("week", week_from_game_id(game_id))
    if row.get("is_live") is None:
        row["is_live"] = row.get("source") == "espn"
    if ensemble_components is not None:
        row["ensemble_components"] = (
            ensemble_components
            if isinstance(ensemble_components, str)
            else json.dumps(list(ensemble_components))
        )
    if row.get("market_implied_home") is not None and row.get("edge_vs_market") is None:
        row["edge_vs_market"] = row["p_home"] - float(row["market_implied_home"])
    return row


# --------------------------------------------------------------------------- #
# Frame shaping
# --------------------------------------------------------------------------- #


def empty_ledger() -> pl.DataFrame:
    """A zero-row frame carrying the full ledger schema."""
    return pl.DataFrame(schema=LEDGER_SCHEMA)


def conform(frame: pl.DataFrame | Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """Reindex rows onto :data:`LEDGER_SCHEMA` exactly: missing null, extra dropped.

    Extra columns are dropped rather than kept so a caller cannot smuggle a
    market column into the lake under a name the eval-only guard does not know.
    """
    if not isinstance(frame, pl.DataFrame):
        rows = list(frame)
        if not rows:
            return empty_ledger()
        frame = pl.DataFrame(rows, infer_schema_length=None, strict=False)
    if frame.is_empty() and not frame.columns:
        return empty_ledger()

    extra = [c for c in frame.columns if c not in LEDGER_SCHEMA]
    if extra:
        log.debug("ledger.conform: dropping non-schema columns %s", sorted(extra))
    exprs = []
    for name, dtype in LEDGER_SCHEMA.items():
        if name in frame.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
    return frame.select(exprs)


def drop_eval_only(frame: pl.DataFrame) -> pl.DataFrame:
    """Strip every market-sourced column — the safe handoff out of the ledger."""
    return frame.drop([c for c in EVAL_ONLY_COLUMNS if c in frame.columns])


def feature_safe(frame: pl.DataFrame) -> pl.DataFrame:
    """:func:`drop_eval_only` plus a hard assertion that nothing market survived.

    Belt and braces on top of :func:`nfl.data.market_blocklist.assert_no_market_columns`:
    that guard knows vendor column names, this one knows ledger column names.
    """
    from nfl.data.market_blocklist import assert_no_market_columns

    out = drop_eval_only(frame)
    leaked = sorted(set(out.columns) & EVAL_ONLY_COLUMNS)
    if leaked:  # pragma: no cover - drop_eval_only makes this unreachable
        raise AssertionError(f"eval-only ledger columns survived: {leaked}")
    assert_no_market_columns(out, context="ledger.feature_safe")
    return out


# --------------------------------------------------------------------------- #
# Write / read
# --------------------------------------------------------------------------- #


def _part_name() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"


def append(
    rows: pl.DataFrame | Sequence[Mapping[str, Any]],
    *,
    root: Path | None = None,
) -> list[Path]:
    """Append predictions to the ledger; return the part files written.

    One file per (call, season). Nothing existing is touched, so concurrent
    writers cannot corrupt each other and a crashed write leaves at most one
    orphan file that the next read simply unions in.
    """
    frame = conform(rows)
    if frame.is_empty():
        return []
    _validate(frame)

    target = ledger_dir(root)
    written: list[Path] = []
    for (season,), part in frame.partition_by("season", as_dict=True).items():
        name = "season=unknown" if season is None else f"season={int(season)}"
        path = target / name / _part_name()
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(path)
        written.append(path)
    log.info("ledger: appended %d row(s) in %d file(s)", frame.height, len(written))
    return sorted(written)


def _validate(frame: pl.DataFrame) -> None:
    """Cheap invariants that make a malformed row a write-time failure."""
    if frame.get_column("prediction_id").null_count():
        raise ValueError("ledger rows must carry a prediction_id")
    if frame.get_column("game_id").null_count():
        raise ValueError("ledger rows must carry a game_id")
    p = frame.get_column("p_home")
    if p.null_count():
        raise ValueError("ledger rows must carry p_home")
    bad = frame.filter(~pl.col("p_home").is_between(0.0, 1.0))
    if bad.height:
        raise ValueError(f"p_home outside [0, 1] for {bad.height} row(s)")


def read_ledger(
    root: Path | None = None,
    *,
    seasons: Sequence[int] | int | None = None,
    game_ids: Sequence[str] | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    resolved_only: bool = False,
) -> pl.DataFrame:
    """Read the ledger back as one frame, optionally filtered.

    Returns an empty, fully-typed frame when nothing has been written yet —
    reporting code should not have to special-case a cold start.
    """
    target = ledger_dir(root)
    if not target.is_dir():
        return empty_ledger()
    if seasons is None:
        files = sorted(target.rglob("*.parquet"))
    else:
        wanted = [seasons] if isinstance(seasons, int) else list(seasons)
        files = [p for s in wanted for p in sorted((target / f"season={s}").glob("*.parquet"))]
    if not files:
        return empty_ledger()

    frame = conform(pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed"))
    if game_ids is not None:
        frame = frame.filter(pl.col("game_id").is_in(list(game_ids)))
    if model_name is not None:
        frame = frame.filter(pl.col("model_name") == model_name)
    if model_version is not None:
        frame = frame.filter(pl.col("model_version") == model_version)
    if resolved_only:
        frame = frame.filter(pl.col("resolved_home_win").is_not_null())
    return frame.sort("predicted_at_utc", "prediction_id")


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


class ResolveReport:
    """What one :func:`resolve_ledger` pass did."""

    __slots__ = ("files", "newly_resolved", "rows", "still_open")

    def __init__(self) -> None:
        self.rows = 0
        self.newly_resolved = 0
        self.still_open = 0
        self.files: list[Path] = []

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ResolveReport(rows={self.rows}, newly_resolved={self.newly_resolved}, "
            f"still_open={self.still_open}, files={len(self.files)})"
        )


_SCORE_COLUMNS = (("home_score", "away_score"), ("home_points", "away_points"))


def _outcomes(schedules: pl.DataFrame) -> pl.DataFrame:
    """``(game_id, final_home_score, final_away_score, resolved_home_win)`` for finished games.

    A game is finished when both final scores are present. Ties resolve to
    ``resolved_home_win = null`` rather than a coin flip — a tie is not a home
    win, and scoring it as a loss would quietly bias every metric.
    """
    cols = set(schedules.columns)
    if "game_id" not in cols:
        raise KeyError(f"schedules frame has no game_id; columns={sorted(cols)}")
    pair = next((p for p in _SCORE_COLUMNS if p[0] in cols and p[1] in cols), None)
    if pair is None:
        raise KeyError(
            "schedules frame carries no final score columns "
            f"(looked for {[list(p) for p in _SCORE_COLUMNS]}); columns={sorted(cols)}"
        )
    home, away = pair
    out = schedules.select(
        pl.col("game_id").cast(pl.String),
        pl.col(home).cast(pl.Int32, strict=False).alias("_home_final"),
        pl.col(away).cast(pl.Int32, strict=False).alias("_away_final"),
    ).unique(subset=["game_id"], keep="first")
    out = out.filter(pl.col("_home_final").is_not_null() & pl.col("_away_final").is_not_null())
    return out.with_columns(
        pl.when(pl.col("_home_final") == pl.col("_away_final"))
        .then(None)
        .otherwise(pl.col("_home_final") > pl.col("_away_final"))
        .cast(pl.Boolean)
        .alias("_home_win")
    )


def _resolve_frame(frame: pl.DataFrame, outcomes: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """Fill the null resolution columns of ``frame`` from ``outcomes``."""
    joined = frame.join(outcomes, on="game_id", how="left")
    eps = 1e-15
    p = pl.col("p_home").clip(eps, 1.0 - eps)
    y = pl.col("_home_win").cast(pl.Float64)
    fillable = pl.col("resolved_home_win").is_null() & pl.col("_home_win").is_not_null()
    return joined.with_columns(
        pl.when(fillable)
        .then(pl.col("_home_win"))
        .otherwise(pl.col("resolved_home_win"))
        .alias("resolved_home_win"),
        pl.when(fillable)
        .then(pl.col("_home_final"))
        .otherwise(pl.col("resolved_home_score"))
        .alias("resolved_home_score"),
        pl.when(fillable)
        .then(pl.col("_away_final"))
        .otherwise(pl.col("resolved_away_score"))
        .alias("resolved_away_score"),
        pl.when(fillable)
        .then(pl.lit(now).cast(pl.Datetime("us", "UTC")))
        .otherwise(pl.col("resolved_at_utc"))
        .alias("resolved_at_utc"),
        pl.when(fillable).then((p - y) ** 2).otherwise(pl.col("brier")).alias("brier"),
        pl.when(fillable)
        .then(-(y * p.log() + (1.0 - y) * (1.0 - p).log()))
        .otherwise(pl.col("log_loss"))
        .alias("log_loss"),
    ).drop(["_home_final", "_away_final", "_home_win"])


def resolve_ledger(
    schedules: pl.DataFrame,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> ResolveReport:
    """Fill outcome columns for finished games, in place, part file by part file.

    ``schedules`` is the nflverse schedule frame (``game_id`` plus final scores).
    Only rows whose ``resolved_home_win`` is still null are touched, so this is
    idempotent and cheap to run on a cron.
    """
    report = ResolveReport()
    target = ledger_dir(root)
    if not target.is_dir():
        return report
    outcomes = _outcomes(schedules)
    stamp = now or datetime.now(UTC)

    for path in sorted(target.rglob("*.parquet")):
        frame = conform(pl.read_parquet(path))
        before = int(frame.get_column("resolved_home_win").is_not_null().sum())
        resolved = _resolve_frame(frame, outcomes, stamp)
        after = int(resolved.get_column("resolved_home_win").is_not_null().sum())
        report.rows += resolved.height
        report.still_open += resolved.height - after
        if after > before:
            resolved.write_parquet(path)
            report.newly_resolved += after - before
            report.files.append(path)
    log.info(
        "ledger.resolve: %d row(s) scanned, %d newly resolved, %d still open",
        report.rows,
        report.newly_resolved,
        report.still_open,
    )
    return report


# --------------------------------------------------------------------------- #
# DuckDB
# --------------------------------------------------------------------------- #


def register_ledger_views(con: Any, root: Path | None = None) -> list[str]:
    """Register ``prediction_ledger`` as a hive-partitioned view over the lake.

    ``data/ledger`` sits outside the raw/processed roots that
    :func:`nfl.data.store.refresh_views` walks, so the ledger gets its own
    registration. No-op (returns ``[]``) when nothing has been written yet.
    """
    target = ledger_dir(root)
    if not any(target.rglob("*.parquet")):
        con.execute(f'DROP VIEW IF EXISTS "{LEDGER_TABLE}"')
        return []
    glob = str(target / "**" / "*.parquet").replace("'", "''")
    con.execute(
        f'CREATE OR REPLACE VIEW "{LEDGER_TABLE}" AS SELECT * FROM '
        f"read_parquet('{glob}', hive_partitioning => true, union_by_name => true)"
    )
    return [LEDGER_TABLE]
