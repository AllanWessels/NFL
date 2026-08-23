"""Assembly: every builder's output joined into one modelling table.

The output is one row per ``(game_id, play_id)`` — the snapshot the model sees
at inference time — with three kinds of column:

``meta__*``
    Identity and slicing keys: season, week, game, play, kickoff, clock,
    period, teams. **Never features.** They exist so the splits, the Q4 slices
    and the ledger can find a row again.
``label__home_win``
    The target, and the *only* place in the codebase it is computed: 1.0 home
    win, 0.0 home loss, 0.5 tie, null for a game that has not been played.
``everything else``
    Features. ``bi__``/``ig__`` from play-level builders join on
    ``(game_id, play_id)``; team-level builders return one row per
    ``(game_id, team)`` and are widened into ``home_tm__*`` / ``away_tm__*``;
    game-level builders join on ``game_id``; ``ctx__*`` is the small
    non-market game context read straight off the schedule.

Nothing from the raw frames survives into the output unless a builder emitted
it or it is listed in :data:`META_COLUMNS`. That is deliberate: play-by-play
carries ``spread_line`` and ``vegas_wp``, and the cheapest way never to ship
them is never to copy them. :func:`assemble_dataset` still ends with
:func:`~nfl.data.market_blocklist.assert_no_market_columns` over the finished
frame, as the last gate before anything is written or trained on.

    >>> df = assemble_dataset([2024], pbp=pbp, schedules=schedules)   # doctest: +SKIP
    >>> feature_columns(df)[:3]                                       # doctest: +SKIP
    ['bi__score_diff_home', 'bi__abs_score_diff', 'bi__score_diff_posteam']
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from nfl.config import PROCESSED_DIR, RAW_DIR
from nfl.data.market_blocklist import assert_no_market_columns
from nfl.features.registry import BuilderSpec, Discovery, discover_builders

__all__ = [
    "CONTEXT_PREFIX",
    "LABEL_COLUMN",
    "LABEL_PREFIX",
    "META_COLUMNS",
    "META_PREFIX",
    "AssemblyReport",
    "assemble_dataset",
    "dataset_dir",
    "feature_columns",
    "game_context",
    "home_win_label",
    "kickoff_utc",
    "meta_columns",
    "read_dataset",
    "write_dataset",
]

log = logging.getLogger(__name__)

#: ``pl.Int32`` is a class, ``pl.Datetime("us")`` an instance; both are dtypes.
PolarsDataType = DataType | DataTypeClass

META_PREFIX = "meta__"
LABEL_PREFIX = "label__"
CONTEXT_PREFIX = "ctx__"
LABEL_COLUMN = "label__home_win"

#: The identity/slicing columns every assembled row carries, in output order.
META_COLUMNS: tuple[str, ...] = (
    "meta__season",
    "meta__week",
    "meta__game_id",
    "meta__play_id",
    "meta__kickoff",
    "meta__game_seconds_remaining",
    "meta__period",
    "meta__home_team",
    "meta__away_team",
)

#: Written alongside the data so a stale dataset can be spotted (CLAUDE.md
#: conventions ask for ``source``/``ingested_at``; they are ``meta__`` prefixed
#: so :func:`feature_columns` cannot mistake them for features).
_PROVENANCE = ("meta__source", "meta__ingested_at")

_DATASET = "dataset"


# --------------------------------------------------------------------------- #
# Column helpers
# --------------------------------------------------------------------------- #


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Every column that is a model input: not ``meta__``, not ``label__``."""
    return [c for c in df.columns if not c.startswith((META_PREFIX, LABEL_PREFIX))]


def meta_columns(df: pl.DataFrame) -> list[str]:
    """Every ``meta__`` column present, in frame order."""
    return [c for c in df.columns if c.startswith(META_PREFIX)]


def kickoff_utc(schedules: pl.DataFrame) -> pl.Expr:
    """Kickoff as a naive-UTC timestamp, from ``kickoff`` or ``gameday``+``gametime``.

    nflverse stores the local Eastern date and time; the conversion is
    DST-aware and matches ``nfl.data.nflverse`` so the two never disagree.
    """
    if "kickoff" in schedules.columns:
        return pl.col("kickoff").cast(pl.Datetime("us"), strict=False)
    if not {"gameday", "gametime"} <= set(schedules.columns):
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
# Label and game context
# --------------------------------------------------------------------------- #


def home_win_label(schedules: pl.DataFrame) -> pl.DataFrame:
    """``(game_id, label__home_win)``: 1.0 win, 0.0 loss, 0.5 tie, null if unplayed.

    The one place the target is defined (Phase 4 conventions). It reads final
    scores only — never ``result`` sign conventions, never a market number.
    """
    if not {"home_score", "away_score"} <= set(schedules.columns):
        raise KeyError("schedules needs home_score and away_score to compute the label")
    home = pl.col("home_score").cast(pl.Float64, strict=False)
    away = pl.col("away_score").cast(pl.Float64, strict=False)
    label = (
        pl.when(home.is_null() | away.is_null())
        .then(None)
        .when(home > away)
        .then(1.0)
        .when(home < away)
        .then(0.0)
        .otherwise(0.5)
    )
    return schedules.select(
        pl.col("game_id").cast(pl.String),
        label.cast(pl.Float64).alias(LABEL_COLUMN),
    ).unique(subset=["game_id"], keep="first")


def game_context(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per game of numeric, non-market pre-game context (``ctx__*``).

    Small on purpose: venue and rest facts that are known at kickoff and that
    no other builder owns. Anything richer belongs in ``nfl/features/team.py``.
    """
    cols = set(schedules.columns)
    roof = pl.col("roof").cast(pl.String) if "roof" in cols else pl.lit(None, dtype=pl.String)
    surface = (
        pl.col("surface").cast(pl.String) if "surface" in cols else pl.lit(None, dtype=pl.String)
    )

    def num(name: str) -> pl.Expr:
        return (
            pl.col(name).cast(pl.Float64, strict=False)
            if name in cols
            else pl.lit(None, dtype=pl.Float64)
        )

    game_type = (
        pl.col("game_type").cast(pl.String)
        if "game_type" in cols
        else pl.lit(None, dtype=pl.String)
    )
    home_rest, away_rest = num("home_rest"), num("away_rest")
    return schedules.select(
        pl.col("game_id").cast(pl.String),
        num("div_game").cast(pl.Int8, strict=False).alias("ctx__div_game"),
        (game_type != "REG").cast(pl.Int8).alias("ctx__is_postseason"),
        roof.is_in(["dome", "closed"]).cast(pl.Int8).alias("ctx__is_indoors"),
        roof.is_in(["outdoors", "open"]).cast(pl.Int8).alias("ctx__is_outdoors"),
        (surface != "grass").cast(pl.Int8).alias("ctx__is_artificial_turf"),
        num("temp").alias("ctx__temp_f"),
        num("wind").alias("ctx__wind_mph"),
        home_rest.alias("ctx__home_rest_days"),
        away_rest.alias("ctx__away_rest_days"),
        (home_rest - away_rest).alias("ctx__rest_diff_home"),
    ).unique(subset=["game_id"], keep="first")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def _normalise_seasons(seasons: str | int | Sequence[int] | None) -> list[int] | None:
    if seasons is None:
        return None
    if isinstance(seasons, int):
        return [seasons]
    if isinstance(seasons, str):
        from nfl.data.nflverse import parse_seasons

        return parse_seasons(seasons)
    return sorted({int(s) for s in seasons})


def load_raw_table(
    table: str, seasons: Sequence[int] | None = None, *, root: Path | None = None
) -> pl.DataFrame:
    """Read ``data/raw/nflverse/<table>/season=YYYY/`` for the seasons requested."""
    base = (root or RAW_DIR) / "nflverse" / table
    if not base.is_dir():
        raise FileNotFoundError(
            f"no ingested {table} at {base}; run `nfl ingest nflverse --seasons ...` first"
        )
    if seasons is None:
        files = sorted(base.rglob("*.parquet"))
    else:
        files = [p for s in seasons for p in sorted((base / f"season={s}").glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"no parquet under {base} for seasons {seasons}")
    frames = [pl.read_parquet(f) for f in files]
    common = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    return pl.concat([f.select(common) for f in frames], how="vertical_relaxed")


def _filter_seasons(df: pl.DataFrame, seasons: Sequence[int] | None) -> pl.DataFrame:
    if seasons is None or "season" not in df.columns:
        return df
    return df.filter(pl.col("season").cast(pl.Int64, strict=False).is_in(list(seasons)))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _base_frame(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """The ``meta__`` skeleton: one row per play, keys and slicing columns only."""
    cols = set(pbp.columns)
    missing = sorted({"game_id", "play_id"} - cols)
    if missing:
        raise KeyError(f"pbp is missing required column(s) {missing}")

    def col(name: str, dtype: PolarsDataType) -> pl.Expr:
        if name in cols:
            return pl.col(name).cast(dtype, strict=False)
        return pl.lit(None, dtype=dtype)

    base = pbp.select(
        col("season", pl.Int32).alias("meta__season"),
        col("week", pl.Int32).alias("meta__week"),
        pl.col("game_id").cast(pl.String).alias("meta__game_id"),
        pl.col("play_id").cast(pl.Int64, strict=False).alias("meta__play_id"),
        col("game_seconds_remaining", pl.Int32).alias("meta__game_seconds_remaining"),
        col("qtr", pl.Int32).alias("meta__period"),
        col("home_team", pl.String).alias("meta__home_team"),
        col("away_team", pl.String).alias("meta__away_team"),
    ).unique(subset=["meta__game_id", "meta__play_id"], keep="first", maintain_order=True)

    sched_cols = set(schedules.columns)
    lookup = schedules.select(
        pl.col("game_id").cast(pl.String).alias("meta__game_id"),
        kickoff_utc(schedules).alias("meta__kickoff"),
        *(
            [pl.col("season").cast(pl.Int32, strict=False).alias("_sched_season")]
            if "season" in sched_cols
            else []
        ),
        *(
            [pl.col("week").cast(pl.Int32, strict=False).alias("_sched_week")]
            if "week" in sched_cols
            else []
        ),
        *(
            [pl.col("home_team").cast(pl.String).alias("_sched_home")]
            if "home_team" in sched_cols
            else []
        ),
        *(
            [pl.col("away_team").cast(pl.String).alias("_sched_away")]
            if "away_team" in sched_cols
            else []
        ),
    ).unique(subset=["meta__game_id"], keep="first")

    base = base.join(lookup, on="meta__game_id", how="left")
    # The schedule is authoritative for anything play-by-play may have dropped.
    for meta, fallback in (
        ("meta__season", "_sched_season"),
        ("meta__week", "_sched_week"),
        ("meta__home_team", "_sched_home"),
        ("meta__away_team", "_sched_away"),
    ):
        if fallback in base.columns:
            base = base.with_columns(pl.col(meta).fill_null(pl.col(fallback)).alias(meta))
    return base.drop([c for c in base.columns if c.startswith("_sched_")])


def _widen_team_frame(spec: BuilderSpec, out: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    """``(game_id, team, tm__x)`` -> ``(game_id, home_tm__x, away_tm__x)``."""
    features = spec.feature_columns(out)
    if not features:
        return pl.DataFrame({"meta__game_id": []}, schema={"meta__game_id": pl.String})
    slim = out.select(
        pl.col("game_id").cast(pl.String).alias("meta__game_id"),
        pl.col("team").cast(pl.String).alias("_team"),
        *[pl.col(c) for c in features],
    )
    games = base.select("meta__game_id", "meta__home_team", "meta__away_team").unique(
        subset=["meta__game_id"], keep="first"
    )
    home = slim.rename({c: f"home_{c}" for c in features}).rename({"_team": "meta__home_team"})
    away = slim.rename({c: f"away_{c}" for c in features}).rename({"_team": "meta__away_team"})
    wide = games.join(home, on=["meta__game_id", "meta__home_team"], how="left").join(
        away, on=["meta__game_id", "meta__away_team"], how="left"
    )
    return wide.drop("meta__home_team", "meta__away_team")


class AssemblyReport:
    """What ran, what was skipped, and how wide the result is."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.skipped: list[tuple[str, str]] = []
        self.rows: int = 0
        self.features: int = 0

    def skip(self, name: str, reason: str) -> None:
        self.skipped.append((name, reason))
        log.info("skipping builder %s: %s", name, reason)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AssemblyReport(ran={self.ran}, skipped={self.skipped}, "
            f"rows={self.rows}, features={self.features})"
        )


def assemble_dataset(
    seasons: str | int | Sequence[int] | None = None,
    *,
    pbp: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
    inputs: Mapping[str, pl.DataFrame] | None = None,
    builders: Discovery | Sequence[BuilderSpec] | None = None,
    write: bool = False,
    root: Path | None = None,
    raw_root: Path | None = None,
    report: AssemblyReport | None = None,
) -> pl.DataFrame:
    """Join every available builder into the modelling table for ``seasons``.

    ``pbp``/``schedules`` default to the ingested parquet under
    ``data/raw/nflverse``; pass frames directly (as the tests do) to stay
    offline. ``inputs`` adds further named frames — ``weather``, ``rosters`` —
    which are handed to whichever builder names them in its signature.

    Builders whose module has not landed, or whose required inputs are not
    available, are **skipped and recorded**, never fatal. Set ``write=True`` to
    also persist ``data/processed/dataset/season=YYYY/part-0.parquet``.
    """
    season_list = _normalise_seasons(seasons)
    rep = report if report is not None else AssemblyReport()

    if pbp is None:
        pbp = load_raw_table("pbp", season_list, root=raw_root)
    if schedules is None:
        schedules = load_raw_table("schedules", season_list, root=raw_root)
    pbp = _filter_seasons(pbp, season_list)
    schedules = _filter_seasons(schedules, season_list)
    if pbp.height == 0:
        raise ValueError(f"no play-by-play rows for seasons {season_list}")

    pool: dict[str, pl.DataFrame] = {"pbp": pbp, "schedules": schedules}
    pool.update(dict(inputs or {}))

    found = builders if builders is not None else discover_builders()
    if isinstance(found, Discovery):
        for gap in found.missing:
            rep.skip(gap.name, gap.reason)
        specs: Sequence[BuilderSpec] = found.builders
    else:
        specs = list(found)

    out = _base_frame(pbp, schedules)
    out = out.join(
        home_win_label(schedules), left_on="meta__game_id", right_on="game_id", how="left"
    )
    context = game_context(schedules)
    out = out.join(context, left_on="meta__game_id", right_on="game_id", how="left")

    for spec in specs:
        gaps = spec.missing_inputs(pool)
        if gaps:
            rep.skip(spec.name, f"missing input(s) {list(gaps)}")
            continue
        built = spec.build(**pool)
        features = spec.feature_columns(built)
        if not features:
            rep.skip(spec.name, "returned no prefixed feature columns")
            continue
        clash = sorted(set(features) & set(out.columns))
        if clash:
            raise ValueError(f"builder {spec.name!r} would overwrite existing column(s) {clash}")

        if spec.level == "play":
            slim = built.select(
                pl.col("game_id").cast(pl.String).alias("meta__game_id"),
                pl.col("play_id").cast(pl.Int64, strict=False).alias("meta__play_id"),
                *[pl.col(c) for c in features],
            )
            out = out.join(slim, on=["meta__game_id", "meta__play_id"], how="left")
        elif spec.level == "team":
            out = out.join(_widen_team_frame(spec, built, out), on="meta__game_id", how="left")
        else:  # "game"
            slim = built.select(
                pl.col("game_id").cast(pl.String).alias("meta__game_id"),
                *[pl.col(c) for c in features],
            )
            out = out.join(slim, on="meta__game_id", how="left")
        rep.ran.append(spec.name)

    out = out.with_columns(
        pl.lit("nfl.features.assemble", dtype=pl.String).alias("meta__source"),
        pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("meta__ingested_at"),
    )
    ordered = [
        *[c for c in META_COLUMNS if c in out.columns],
        *[c for c in _PROVENANCE if c in out.columns],
        LABEL_COLUMN,
        *[c for c in out.columns if c not in set(META_COLUMNS) | set(_PROVENANCE) | {LABEL_COLUMN}],
    ]
    out = out.select(ordered).sort(["meta__game_id", "meta__play_id"])

    # The last gate: nothing market-derived leaves this function (ADR 0001).
    assert_no_market_columns(out, context="assemble_dataset")

    rep.rows = out.height
    rep.features = len(feature_columns(out))
    if write:
        write_dataset(out, root=root)
    return out


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def dataset_dir(root: Path | None = None) -> Path:
    """``data/processed/dataset`` unless overridden."""
    return (root or PROCESSED_DIR) / _DATASET


def write_dataset(df: pl.DataFrame, *, root: Path | None = None) -> list[Path]:
    """Write one ``season=YYYY/part-0.parquet`` partition per season; return paths."""
    assert_no_market_columns(df, context="write_dataset")
    target = dataset_dir(root)
    written: list[Path] = []
    if "meta__season" not in df.columns:
        raise KeyError("dataset frame has no meta__season to partition on")
    for (season,), part in df.partition_by("meta__season", as_dict=True).items():
        name = "season=unknown" if season is None else f"season={int(season)}"
        path = target / name / "part-0.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(path)
        written.append(path)
    return sorted(written)


def read_dataset(
    seasons: str | int | Sequence[int] | None = None, *, root: Path | None = None
) -> pl.DataFrame:
    """Read back the assembled dataset, optionally restricted to seasons."""
    season_list = _normalise_seasons(seasons)
    target = dataset_dir(root)
    if not target.is_dir():
        raise FileNotFoundError(f"no dataset at {target}; run `nfl features build` first")
    if season_list is None:
        files = sorted(target.rglob("*.parquet"))
    else:
        files = [p for s in season_list for p in sorted((target / f"season={s}").glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"no dataset parquet under {target} for seasons {season_list}")
    frames = [pl.read_parquet(f) for f in files]
    common = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    out = pl.concat([f.select(common) for f in frames], how="vertical_relaxed")
    assert_no_market_columns(out, context="read_dataset")
    return out
