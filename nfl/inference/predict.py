"""``/predict`` end to end: game id -> snapshot -> features -> model -> ledger.

One function, :func:`predict_game`, ties together everything the earlier phases
built. The path is deliberately the *same* whether the game is happening now or
happened in 2016, because both feeds end at a
:class:`~nfl.data.live.base.LiveSnapshot`:

1. **Resolve the game.** ``2024_01_BAL_KC`` -> the schedules row and (for the
   live wire) the ESPN event id, via the crosswalk in
   :mod:`nfl.data.live.base`. ``--source auto`` picks ``replay`` when the game
   is final and its play-by-play is on disk, ``espn`` otherwise.
2. **Snapshot.** ``LiveFeed.snapshot(game_id, at)``. ``--at`` rewinds either
   feed to a game time; ``None`` means "whatever is happening now".
3. **Features.** :func:`build_feature_rows` produces one row per play the
   snapshot has seen, plus the snap itself. The in-play block comes from the
   snapshot — :func:`nfl.features.ingame.snapshot_history_rows` for ``ig__`` and
   the same :func:`~nfl.features.baseline_inputs.build_baseline_inputs`
   expressions for ``bi__``, both over one synthetic play-by-play frame — and
   the pre-game block (``ctx__``, ``tm__``, ``pl__``) is built as of kickoff
   from ``data/raw`` and cached per game under
   ``data/cache/pregame/<game_id>.parquet``. Player availability is refreshed
   from the snapshot's injury list by
   :func:`~nfl.features.player.roster_from_snapshot`.
4. **Model.** Whatever ``PRODUCTION.json`` points at, or ``name[:version]``.
   The frame is projected onto the manifest's feature list exactly: a column the
   serving path cannot produce is *null and logged*, never silently imputed and
   never quietly dropped. A model that windows over play history — the GRU, or
   an ensemble containing one — is handed the whole history and its answer read
   off the last row; everything else is handed that row alone. Serving a
   sequence model a single row would window it over one play, which is a
   different model from the one the promotion gate measured.
5. **Ledger.** One :func:`nfl.logging.ledger.append` call carrying the snapshot
   clock, the key game state, the feature hash, the git sha and the latency.
   Market fields stay null unless a Kalshi snapshot is attached best-effort —
   evaluation only, ADR 0001.

The trade gate is reported, never enforced: ``collapsed`` is
``not (0.15 <= p <= 0.85)``, the band CLAUDE.md names as the region where a
position is worth taking. Acting on it is a decision for a human (or a later
phase), so this module only says which side of the line the number fell.

    >>> result = predict_game("2024_01_BAL_KC", at="Q4 05:00", source="replay")
    ... # doctest: +SKIP
    >>> result.p_home, result.collapsed                                  # doctest: +SKIP
    (0.9312, True)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from nfl import config
from nfl.data.live.base import (
    AtTime,
    LiveFeed,
    LiveSnapshot,
    load_schedule_frame,
    parse_at,
    seconds_to_clock,
)
from nfl.logging import ledger as ledger_mod
from nfl.models import registry

log = logging.getLogger(__name__)

__all__ = [
    "TRADE_GATE_HIGH",
    "TRADE_GATE_LOW",
    "ModelRef",
    "ParityReport",
    "PredictionError",
    "PredictionResult",
    "SnapshotSummary",
    "Source",
    "build_feature_row",
    "build_feature_rows",
    "clock_sensitive_columns",
    "build_pregame_table",
    "check_parity",
    "load_model",
    "needs_play_window",
    "pregame_cache_path",
    "predict_game",
    "render_prediction",
    "resolve_source",
    "serving_meta",
    "snapshot_to_baseline_row",
    "snapshot_to_baseline_rows",
]

Source = Literal["espn", "replay", "auto"]

#: The band inside which the market has *not* collapsed and a position is worth
#: considering (CLAUDE.md, "Purpose"). Outside it, ``collapsed`` is True.
TRADE_GATE_LOW = 0.15
TRADE_GATE_HIGH = 0.85

#: Model name -> ``"module:QualName"``, for artifacts whose manifest predates
#: :func:`nfl.models.registry.save_model` and therefore carries no
#: ``model_class``. Every one of these ``load()`` classmethods accepts the
#: version *directory*, which is what we hand them.
MODEL_CLASSES: dict[str, str] = {
    "naive_score_time": "nfl.models.naive:ScoreTimeLogistic",
    "xgboost_winprob": "nfl.models.gbdt:XGBoostWinProb",
    "tf_mlp": "nfl.models.tf_mlp:MLPWinProb",
    "tf_seq": "nfl.models.tf_seq:SeqWinProb",
    "stacked_ensemble": "nfl.models.ensemble:StackedEnsemble",
}

#: ``model_kind`` for the ledger, by model name.
MODEL_KINDS: dict[str, str] = {
    "naive_score_time": "baseline",
    "xgboost_winprob": "gbdt",
    "tf_mlp": "mlp",
    "tf_seq": "seq",
    "stacked_ensemble": "ensemble",
}

#: Preference order when ``PRODUCTION.json`` is missing. The naive floor is the
#: safest thing to serve unasked: it is the model every other one has to beat.
FALLBACK_NAMES: tuple[str, ...] = ("naive_score_time", "naive")

#: ``bi__`` columns that move with the game clock, for the same reason
#: :data:`nfl.features.ingame.SNAPSHOT_CLOCK_SENSITIVE_COLUMNS` exists: a
#: snapshot rewound to ``Q4 05:00`` reports 300 seconds left, the play-by-play
#: row for the snap that follows reports whatever the clock actually read.
_BASELINE_CLOCK_SENSITIVE: frozenset[str] = frozenset(
    {
        "bi__game_seconds_remaining",
        "bi__half_seconds_remaining",
        "bi__quarter_seconds_remaining",
        "bi__elapsed_game_seconds",
        "bi__is_final_two_minutes",
    }
)

_PREGAME_CACHE = "pregame"

#: Column prefixes that require the pre-game table to be built.
_PREGAME_PREFIXES = ("home_tm__", "away_tm__", "home_pl__", "away_pl__")


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced (no game, no feed, no model)."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """The handful of snapshot facts a human needs to read a prediction."""

    source: str
    state: str
    period: int
    clock: str
    game_seconds_remaining: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    possession: str | None = None
    down: int | None = None
    distance: int | None = None
    yards_to_endzone: int | None = None
    home_timeouts: int | None = None
    away_timeouts: int | None = None
    timeouts_exact: bool = True
    play_count: int = 0
    at: str | None = None

    @classmethod
    def from_snapshot(cls, snapshot: LiveSnapshot) -> SnapshotSummary:
        return cls(
            source=snapshot.source,
            state=snapshot.state,
            period=snapshot.period,
            clock=seconds_to_clock(snapshot.clock_seconds),
            game_seconds_remaining=snapshot.game_seconds_remaining,
            home_team=snapshot.home_team,
            away_team=snapshot.away_team,
            home_score=snapshot.home_score,
            away_score=snapshot.away_score,
            possession=snapshot.possession,
            down=snapshot.down,
            distance=snapshot.distance,
            yards_to_endzone=snapshot.yards_to_endzone,
            home_timeouts=snapshot.home_timeouts,
            away_timeouts=snapshot.away_timeouts,
            timeouts_exact=snapshot.timeouts_exact,
            play_count=snapshot.play_count,
            at=None if snapshot.at is None else str(snapshot.at),
        )

    @property
    def label(self) -> str:
        period = "OT" if self.period > 4 else f"Q{self.period}"
        return f"{period} {self.clock}"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Which artifact produced a prediction, and what it expected to be fed."""

    name: str
    version: str
    kind: str
    feature_names: tuple[str, ...] = ()
    stage: str = "none"
    git_sha: str = ""

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Everything one call to :func:`predict_game` produced."""

    game_id: str
    p_home: float
    p_away: float
    confidence: float
    confidence_method: str
    model: ModelRef
    snapshot: SnapshotSummary
    feature_hash: str
    latency_ms: float
    components: dict[str, float] = field(default_factory=dict)
    missing_features: tuple[str, ...] = ()
    features_nonnull: int = 0
    features_total: int = 0
    source: str = "replay"
    prediction_id: str | None = None
    ledger_paths: tuple[Path, ...] = ()
    market: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    @property
    def collapsed(self) -> bool:
        """True when ``p_home`` sits outside the 15/85 trade band."""
        return not (TRADE_GATE_LOW <= self.p_home <= TRADE_GATE_HIGH)

    @property
    def favourite(self) -> str:
        return self.snapshot.home_team if self.p_home >= 0.5 else self.snapshot.away_team

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view, for scripts and the slash command."""
        return {
            "game_id": self.game_id,
            "p_home": self.p_home,
            "p_away": self.p_away,
            "confidence": self.confidence,
            "confidence_method": self.confidence_method,
            "collapsed": self.collapsed,
            "model": {
                "name": self.model.name,
                "version": self.model.version,
                "kind": self.model.kind,
            },
            "snapshot": {
                "source": self.snapshot.source,
                "state": self.snapshot.state,
                "clock": self.snapshot.label,
                "score": f"{self.snapshot.away_team} {self.snapshot.away_score} @ "
                f"{self.snapshot.home_team} {self.snapshot.home_score}",
                "possession": self.snapshot.possession,
            },
            "feature_hash": self.feature_hash,
            "features": {
                "total": self.features_total,
                "nonnull": self.features_nonnull,
                "missing": list(self.missing_features),
            },
            "latency_ms": self.latency_ms,
            "prediction_id": self.prediction_id,
            "market": self.market,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class ParityReport:
    """What :func:`check_parity` found comparing serving to training features."""

    game_id: str
    at: str
    play_id: int | None
    compared: tuple[str, ...] = ()
    mismatches: tuple[tuple[str, Any, Any], ...] = ()
    clock_sensitive: tuple[tuple[str, Any, Any], ...] = ()
    skipped_unknowable: tuple[str, ...] = ()
    #: Columns the serving path produced that the offline dataset does not have.
    missing_offline: tuple[str, ...] = ()
    #: Columns the offline dataset has that the serving path did not produce.
    #: A non-null one of these is also counted as a mismatch.
    missing_serving: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def __str__(self) -> str:
        head = (
            f"parity {self.game_id} @ {self.at} (play {self.play_id}): "
            f"{len(self.compared)} column(s) compared, {len(self.mismatches)} mismatch(es)"
        )
        if not self.mismatches:
            return head
        detail = "\n".join(f"  {c}: serve={s!r} offline={o!r}" for c, s, o in self.mismatches[:20])
        return f"{head}\n{detail}"


# --------------------------------------------------------------------------- #
# Game resolution and feeds
# --------------------------------------------------------------------------- #


def _schedule_row(game_id: str, schedules: pl.DataFrame) -> dict[str, Any]:
    row = schedules.filter(pl.col("game_id") == game_id)
    if row.height == 0:
        raise PredictionError(
            f"game_id {game_id!r} is not in the schedules table; "
            "run `nfl ingest nflverse --tables schedules` or check the id"
        )
    return row.row(0, named=True)


def _is_final(row: Mapping[str, Any]) -> bool:
    return row.get("home_score") is not None and row.get("away_score") is not None


def _pbp_available(season: int | None, pbp: pl.DataFrame | str | Path | None) -> bool:
    """Is there play-by-play on disk (or in hand) for ``season``?"""
    if isinstance(pbp, pl.DataFrame):
        return True
    if pbp is not None:
        return Path(pbp).exists()
    if season is None:
        return False
    base = config.RAW_DIR / "nflverse" / "pbp"
    return (base / f"season={season}").is_dir() or base.is_dir()


def resolve_source(
    game_id: str,
    source: Source = "auto",
    *,
    schedules: pl.DataFrame | str | Path | None = None,
    pbp: pl.DataFrame | str | Path | None = None,
) -> str:
    """``auto`` -> ``replay`` for a finished game with local pbp, else ``espn``."""
    if source in ("espn", "replay"):
        return source
    if source != "auto":
        raise PredictionError(f"unknown source {source!r}; expected espn | replay | auto")
    frame = load_schedule_frame(schedules)
    row = _schedule_row(game_id, frame)
    season = row.get("season")
    if _is_final(row) and _pbp_available(int(season) if season is not None else None, pbp):
        return "replay"
    return "espn"


def _open_feed(
    source: str,
    *,
    schedules: pl.DataFrame | str | Path | None,
    pbp: pl.DataFrame | str | Path | None,
) -> LiveFeed:
    if source == "replay":
        from nfl.data.live.replay import HistoricalReplayFeed

        return HistoricalReplayFeed(pbp=pbp, schedules=schedules)
    from nfl.data.live.espn import EspnLiveFeed

    return EspnLiveFeed(schedules=schedules)


def _snapshot(
    game_id: str,
    at: AtTime | None,
    source: str,
    *,
    schedules: pl.DataFrame | str | Path | None,
    pbp: pl.DataFrame | str | Path | None,
) -> LiveSnapshot:
    feed = _open_feed(source, schedules=schedules, pbp=pbp)
    try:
        return feed.snapshot(game_id, at=at)
    finally:
        close = getattr(feed, "close", None)
        if callable(close):
            close()


# --------------------------------------------------------------------------- #
# Features: the in-play block
# --------------------------------------------------------------------------- #


def snapshot_to_baseline_rows(snapshot: LiveSnapshot) -> list[dict[str, Any]]:
    """The ``bi__`` block for every play the snapshot has seen, plus the snap.

    Same expressions as the offline builder, not a re-implementation, and over
    the *same* synthetic play-by-play frame the ``ig__`` block is built from
    (:func:`nfl.features.ingame.snapshot_play_frame`), so the two halves of a
    serving row describe the same plays by construction.

    Building the whole history rather than one row is also what makes the
    builder's own lag correct: it recovers "score before this play" by lagging
    nflverse's post-play totals one row (DATA_MODEL §2.2), and with real
    predecessors present that lag is the real previous play instead of a
    synthesised carrier row. A snapshot with **no** plays — a feed that
    publishes a scoreboard and no play list — still gets that carrier row, or
    the lag would report 0-0 for a game in progress.
    """
    from nfl.features.baseline_inputs import build_baseline_inputs
    from nfl.features.ingame import snapshot_play_frame

    frame = snapshot_play_frame(snapshot)
    carried = frame.height == 1
    if carried:
        carrier = frame.head(1).with_columns(
            pl.lit(0.0).alias("play_id"),
            pl.lit(0.0).alias("order_sequence"),
            pl.lit(float(snapshot.home_score)).alias("total_home_score"),
            pl.lit(float(snapshot.away_score)).alias("total_away_score"),
            *[
                pl.lit(None, dtype=pl.Float64).alias(c)
                for c in ("qtr", "quarter_seconds_remaining", "down", "ydstogo", "yardline_100")
            ],
            pl.lit(None, dtype=pl.String).alias("posteam"),
        )
        frame = pl.concat([carrier, frame])
    built = build_baseline_inputs(frame)
    rows = built.to_dicts()[1:] if carried else built.to_dicts()
    for row in rows:
        row.pop("game_id", None)
        row.pop("play_id", None)
    return rows


def snapshot_to_baseline_row(snapshot: LiveSnapshot) -> dict[str, Any]:
    """The ``bi__`` feature row for the snap a snapshot is poised at."""
    return snapshot_to_baseline_rows(snapshot)[-1]


# --------------------------------------------------------------------------- #
# Features: the pre-game block
# --------------------------------------------------------------------------- #


def pregame_cache_path(game_id: str, root: Path | None = None) -> Path:
    """``data/cache/pregame/<game_id>.parquet``."""
    base = Path(root) if root is not None else config.DATA_DIR / "cache"
    return base / _PREGAME_CACHE / f"{game_id}.parquet"


def _pregame_pbp(
    season: int | None,
    week: int | None,
    pbp: pl.DataFrame | str | Path | None,
    *,
    seasons_back: int = 1,
) -> pl.DataFrame | None:
    """Play-by-play the pre-game builders may read: prior seasons in full, the
    current season only through the **previous** week.

    The week cut is belt-and-braces. Both builders are as-of-kickoff by
    construction, but the cheapest way not to leak this week's game into this
    week's prior is not to hand it over (CLAUDE.md rule 3).
    """
    from nfl.features.ingame import load_pbp_frame

    frame: pl.DataFrame | None
    if isinstance(pbp, pl.DataFrame):
        frame = pbp
    elif pbp is not None:
        path = Path(pbp)
        frame = pl.read_parquet(path) if path.is_file() else None
        if frame is None:
            from nfl.data.live.base import read_parquet_dir

            frame = read_parquet_dir(path, ("game_id", "play_id"))
    elif season is None:
        frame = None
    else:
        parts: list[pl.DataFrame] = []
        for s in range(season - seasons_back, season + 1):
            try:
                parts.append(load_pbp_frame([s]))
            except FileNotFoundError:
                continue
        if not parts:
            return None
        common = [c for c in parts[0].columns if all(c in p.columns for p in parts)]
        frame = pl.concat([p.select(common) for p in parts], how="vertical_relaxed")

    if frame is None or frame.height == 0:
        return None
    if season is not None and week is not None and {"season", "week"} <= set(frame.columns):
        frame = frame.filter(
            (pl.col("season").cast(pl.Int64, strict=False) < season)
            | (pl.col("week").cast(pl.Int64, strict=False) < week)
        )
    return frame if frame.height else None


def _season_stub(season: int | None) -> pl.DataFrame | None:
    """A play-by-play frame with a season and no plays.

    :func:`~nfl.features.player.build_player` reads its season list off the
    play-by-play, and returns nothing at all for a season it cannot see. This
    is the minimum shape it accepts: the schedule-derived block comes back, the
    stat blocks come back null, and nothing is invented.
    """
    if season is None:
        return None
    return pl.DataFrame(
        {
            "season": pl.Series([season], dtype=pl.Int64),
            "week": pl.Series([0], dtype=pl.Int64),
            "game_id": pl.Series([""], dtype=pl.String),
            "posteam": pl.Series([None], dtype=pl.String),
            "defteam": pl.Series([None], dtype=pl.String),
        }
    )


def build_pregame_table(
    game_id: str,
    *,
    schedules: pl.DataFrame | str | Path | None = None,
    pbp: pl.DataFrame | str | Path | None = None,
    cache_root: Path | None = None,
    refresh: bool = False,
    use_cache: bool = True,
) -> pl.DataFrame | None:
    """``(game_id, team, tm__*, pl__*)`` for one game, as of kickoff.

    Cached to :func:`pregame_cache_path` because the team builder walks every
    season of schedules (Elo is cumulative) and the player builder walks a
    season of play-by-play: seconds the first time, milliseconds after. Returns
    ``None`` — with a logged reason — when the raw tables a builder needs are
    not on disk; the features then arrive at the model as nulls, which is the
    honest answer rather than a fabricated one.
    """
    cache = pregame_cache_path(game_id, cache_root)
    if use_cache and cache.is_file() and not refresh:
        log.debug("pregame: cache hit %s", cache)
        return pl.read_parquet(cache)

    try:
        frame = _build_pregame_table(game_id, schedules=schedules, pbp=pbp)
    except (FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        log.warning("pregame features unavailable for %s: %s", game_id, exc)
        return None
    if frame is None:
        return None
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache)
        log.debug("pregame: wrote %s", cache)
    return frame


def _build_pregame_table(
    game_id: str,
    *,
    schedules: pl.DataFrame | str | Path | None,
    pbp: pl.DataFrame | str | Path | None,
) -> pl.DataFrame | None:
    from nfl.features.player import build_player, load_player_inputs
    from nfl.features.team import build_team_features

    schedule_frame = load_schedule_frame(schedules)
    row = _schedule_row(game_id, schedule_frame)
    season = row.get("season")
    week = row.get("week")
    season = int(season) if season is not None else None
    week = int(week) if week is not None else None

    history = _pregame_pbp(season, week, pbp)
    parts: list[pl.DataFrame] = []

    try:
        team = build_team_features(schedule_frame, pbp=history)
        parts.append(team.filter(pl.col("game_id") == game_id))
    except (KeyError, ValueError) as exc:
        log.warning("team features unavailable for %s: %s", game_id, exc)

    # Week 1 (or a season whose earlier weeks are not on disk) leaves no prior
    # play-by-play at all. The player builder still owns two columns that are
    # pure schedule facts — the week and the count of games already played — so
    # it is handed a season stub rather than skipped, and those columns come
    # back filled instead of absent.
    player_pbp = history if history is not None else _season_stub(season)
    if player_pbp is not None:
        extra: dict[str, Any] = {}
        if season is not None and pbp is None:
            loaded = load_player_inputs([season])
            extra = {k: v for k, v in loaded.items() if k not in ("pbp", "schedules")}
        try:
            player = build_player(player_pbp, schedule_frame, game_ids=[game_id], **extra)
            parts.append(player.filter(pl.col("game_id") == game_id))
        except (KeyError, ValueError) as exc:
            log.warning("player features unavailable for %s: %s", game_id, exc)

    parts = [p for p in parts if p.height]
    if not parts:
        return None
    out = parts[0]
    for part in parts[1:]:
        overlap = [c for c in part.columns if c in out.columns and c not in ("game_id", "team")]
        out = out.join(part.drop(overlap), on=["game_id", "team"], how="full", coalesce=True)
    return out


def _widen_pregame(
    pregame: pl.DataFrame | None,
    home_team: str,
    away_team: str,
    prefix: str | tuple[str, ...] = ("tm__", "pl__"),
) -> dict[str, Any]:
    """``(game_id, team, tm__x)`` -> ``{"home_tm__x": ..., "away_tm__x": ...}``.

    Same naming as :func:`nfl.features.assemble.assemble_dataset`, which is what
    makes the serving row line up with the training columns.
    """
    if pregame is None or pregame.height == 0:
        return {}
    features = [c for c in pregame.columns if c.startswith(prefix)]
    out: dict[str, Any] = {}
    for side, team in (("home", home_team), ("away", away_team)):
        match = pregame.filter(pl.col("team") == team)
        if match.height == 0:
            continue
        values = match.row(0, named=True)
        for name in features:
            out[f"{side}_{name}"] = values.get(name)
    return out


def _context_row(game_id: str, schedules: pl.DataFrame) -> dict[str, Any]:
    """The ``ctx__`` block for one game, straight off the schedule."""
    from nfl.features.assemble import game_context

    try:
        built = game_context(schedules.filter(pl.col("game_id") == game_id))
    except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
        log.warning("game context unavailable for %s: %s", game_id, exc)
        return {}
    if built.height == 0:
        return {}
    row = built.row(0, named=True)
    row.pop("game_id", None)
    return row


# --------------------------------------------------------------------------- #
# Features: the whole row
# --------------------------------------------------------------------------- #


def build_feature_rows(
    snapshot: LiveSnapshot,
    *,
    game_id: str | None = None,
    schedules: pl.DataFrame | str | Path | None = None,
    pbp: pl.DataFrame | str | Path | None = None,
    cache_root: Path | None = None,
    refresh_pregame: bool = False,
    use_cache: bool = True,
    pregame: pl.DataFrame | None = None,
    wanted: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Every feature the serving path can produce, **one row per play so far**.

    The last element is the snap being predicted — that is
    :func:`build_feature_row`. The rows before it are the same feature vector
    computed for each play the snapshot has already seen, which is what a
    sequence model needs: a GRU trained on a 32-play window and served a
    one-row frame is windowing over a single play and is not the model that was
    measured.

    Producing the history costs nothing extra. ``snapshot_to_row`` already ran
    :func:`~nfl.features.ingame.build_ingame` over the whole play list and threw
    every row but the last away; both blocks now come off that one pass.

    The ``bi__`` and ``ig__`` blocks vary per row and are exact for each play.
    The ``ctx__``, ``tm__`` and ``pl__`` blocks are **game constants** — they
    are computed as of kickoff and are the same for every play of the game — so
    they are the same on every row by construction rather than by broadcast.
    (``pl__`` carries one snapshot-time refresh from the injury list; that state
    is attributed to the whole game, which is the one place a history row says
    something the offline row for that play would not.)
    """
    from nfl.features.ingame import snapshot_history_rows
    from nfl.features.player import roster_from_snapshot

    resolved = game_id or snapshot.game_id or ""

    ingame = snapshot_history_rows(snapshot)
    baseline = snapshot_to_baseline_rows(snapshot)
    if len(ingame) != len(baseline):  # pragma: no cover - both come off one frame
        raise PredictionError(
            f"serving history is inconsistent for {resolved}: {len(ingame)} in-game rows "
            f"against {len(baseline)} baseline rows"
        )
    rows: list[dict[str, Any]] = []
    for ig, bi in zip(ingame, baseline, strict=True):
        values = dict(ig)
        values.pop("game_id", None)
        values.pop("play_id", None)
        values.update(bi)
        rows.append(values)

    schedule_frame = load_schedule_frame(schedules)
    context = _context_row(resolved, schedule_frame)

    def finish(shared: Mapping[str, Any]) -> list[dict[str, Any]]:
        for row in rows:
            row.update(shared)
        return rows

    if wanted is not None and not any(str(c).startswith(_PREGAME_PREFIXES) for c in wanted):
        log.debug("no pre-game column requested for %s; skipping the pre-game build", resolved)
        return finish(context)

    if pregame is None:
        pregame = build_pregame_table(
            resolved,
            schedules=schedule_frame,
            pbp=pbp,
            cache_root=cache_root,
            refresh=refresh_pregame,
            use_cache=use_cache,
        )

    # The player block always goes through ``roster_from_snapshot``, whether or
    # not a pre-game table was built: the snapshot's injury list is newer than
    # anything the pre-game table can know, and with no table at all the call
    # still returns the full ``pl__`` column set as nulls. Emitting the columns
    # as nulls rather than not at all is what keeps the serving row the same
    # *shape* as the training row.
    player: pl.DataFrame | None = None
    pregame_player: pl.DataFrame | None = None
    if pregame is not None and any(c.startswith("pl__") for c in pregame.columns):
        pregame_player = pregame.select(
            ["game_id", "team", *[c for c in pregame.columns if c.startswith("pl__")]]
        )
    try:
        player = roster_from_snapshot(snapshot, pregame_player, game_id=resolved)
    except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
        log.warning("player block unavailable for %s: %s", resolved, exc)

    return finish(
        {
            **context,
            **_widen_pregame(pregame, snapshot.home_team, snapshot.away_team, prefix="tm__"),
            **_widen_pregame(player, snapshot.home_team, snapshot.away_team, prefix="pl__"),
        }
    )


def build_feature_row(
    snapshot: LiveSnapshot,
    *,
    game_id: str | None = None,
    schedules: pl.DataFrame | str | Path | None = None,
    pbp: pl.DataFrame | str | Path | None = None,
    cache_root: Path | None = None,
    refresh_pregame: bool = False,
    use_cache: bool = True,
    pregame: pl.DataFrame | None = None,
    wanted: Collection[str] | None = None,
) -> dict[str, Any]:
    """Every feature the serving path can produce for one snapshot.

    Keys are the assembled dataset's column names — ``bi__``/``ig__`` from the
    snapshot, ``ctx__`` from the schedule, ``home_tm__``/``away_tm__`` and
    ``home_pl__``/``away_pl__`` from the pre-game table — so the result can be
    projected onto any model's manifest without a translation layer.

    ``wanted`` is the model's feature list, when there is one. It is used only
    to skip work: a model that asks for no pre-game column (the naive floor asks
    for nine ``bi__`` columns and nothing else) does not pay for the team and
    player builders. Pass ``None`` to build everything, which is what the parity
    check does.

    The last row of :func:`build_feature_rows` — same numbers, same cost.
    """
    return build_feature_rows(
        snapshot,
        game_id=game_id,
        schedules=schedules,
        pbp=pbp,
        cache_root=cache_root,
        refresh_pregame=refresh_pregame,
        use_cache=use_cache,
        pregame=pregame,
        wanted=wanted,
    )[-1]


def _model_frame(
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> tuple[pl.DataFrame, list[str]]:
    """A frame carrying exactly ``feature_names``, in order, one row per input.

    A name the serving path did not produce becomes a typed null column and is
    returned in the missing list, so the caller can log it. Silently dropping it
    would change the model's input width; imputing it would invent data.
    """
    batch: Sequence[Mapping[str, Any]] = [rows] if isinstance(rows, Mapping) else list(rows)
    missing: list[str] = []
    data: dict[str, pl.Series] = {}
    for name in feature_names:
        if not any(name in row for row in batch):
            missing.append(name)
            data[name] = pl.Series(name, [None] * len(batch), dtype=pl.Float64)
            continue
        values = [row.get(name) for row in batch]
        if any(isinstance(v, str) for v in values):
            data[name] = pl.Series(
                name, [None if v is None else str(v) for v in values], dtype=pl.String
            )
        else:
            data[name] = pl.Series(
                name, [None if v is None else float(v) for v in values], dtype=pl.Float64
            )
    return pl.DataFrame(data), missing


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #


def _import_class(path: str) -> Any:
    import importlib

    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_spec(spec: str, root: Path | None) -> tuple[str, str, str, list[str]]:
    """``"production"`` / ``"name"`` / ``"name:version"`` -> ``(name, version, stage, warnings)``."""
    warnings: list[str] = []
    if spec in ("production", "prod", "PRODUCTION"):
        pointer = registry.read_pointer(registry.PRODUCTION_STAGE, root)
        if pointer is not None:
            return (
                str(pointer["name"]),
                str(pointer["version"]),
                registry.PRODUCTION_STAGE,
                warnings,
            )
        name = _fallback_name(root)
        warnings.append(
            f"no PRODUCTION pointer in {registry.registry_root(root)}; "
            f"falling back to the latest '{name}' artifact"
        )
        return name, registry.latest(name, root), "none", warnings

    name, _, version = spec.partition(":")
    if not name:
        raise PredictionError(f"cannot parse --model {spec!r}; expected name[:version]")
    if not version:
        try:
            version = registry.latest(name, root)
        except FileNotFoundError as exc:
            raise PredictionError(str(exc)) from exc
    return name, version, "none", warnings


def _fallback_name(root: Path | None) -> str:
    available = registry.list_models(root)
    if not available:
        raise PredictionError(
            f"no trained models under {registry.registry_root(root)}; run `nfl train` first"
        )
    for candidate in FALLBACK_NAMES:
        if candidate in available:
            return candidate
    return available[0]


def load_model(
    spec: str = "production", *, root: Path | None = None
) -> tuple[Any, ModelRef, list[str]]:
    """Load ``spec`` from the registry; return ``(model, ModelRef, warnings)``.

    ``spec`` is ``"production"`` (the stage pointer, with a warned fallback to
    the latest naive artifact), ``"name"`` (its latest version) or
    ``"name:version"``. Artifacts written before the registry recorded a
    ``model_class`` are resolved through :data:`MODEL_CLASSES` instead, so a
    model trained by an earlier phase still serves.
    """
    name, version, stage, warnings = _resolve_spec(spec, root)
    directory = registry.model_dir(name, version, root)
    try:
        manifest = registry.read_manifest(directory)
    except FileNotFoundError as exc:
        raise PredictionError(f"no manifest for {name}/{version}: {exc}") from exc

    if manifest.model_class:
        model = _import_class(manifest.model_class).load(
            registry.resolve_artifact(directory, manifest.artifact)
        )
    else:
        class_path = MODEL_CLASSES.get(manifest.name) or MODEL_CLASSES.get(name)
        if class_path is None:
            raise PredictionError(
                f"manifest for {name}/{version} has no model_class and {name!r} is not in "
                f"MODEL_CLASSES ({sorted(MODEL_CLASSES)}); cannot reconstruct the model"
            )
        # Every ``load()`` in this project accepts the version directory.
        model = _import_class(class_path).load(directory)

    ref = ModelRef(
        name=manifest.name or name,
        version=manifest.version or version,
        kind=MODEL_KINDS.get(manifest.name or name, "baseline"),
        feature_names=tuple(manifest.feature_names),
        stage=manifest.stage or stage,
        git_sha=manifest.git_sha,
    )
    return model, ref, warnings


def _feature_names_for(model: Any, ref: ModelRef, values: Mapping[str, Any]) -> list[str]:
    """The manifest's feature list, or the model's own, or everything we built."""
    if ref.feature_names:
        return list(ref.feature_names)
    own = list(getattr(model, "feature_names", []) or [])
    if own:
        log.warning("manifest for %s has no feature_names; using the model's own list", ref)
        return own
    log.warning("manifest for %s has no feature_names; feeding every built feature", ref)
    return sorted(values)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def _accepts_meta(model: Any, method: str = "predict_proba") -> bool:
    """Does ``model.method`` take a ``meta`` frame? — the sequence-model tell."""
    import inspect

    try:
        return "meta" in inspect.signature(getattr(model, method)).parameters
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - C callables
        return False


def needs_play_window(model: Any) -> bool:
    """Whether anything inside ``model`` windows over play history.

    A sequence model asks for ``meta`` so it can group and order plays by game.
    Ask the model rather than the model's *name*: an ensemble is windowed when
    any one of its members is, and which members it has is a fact about the
    artifact.
    """
    members = getattr(model, "members", None)
    if members:
        return any(needs_play_window(getattr(m, "loaded", m)) for m in members)
    return _accepts_meta(model)


def serving_meta(game_id: str, n_rows: int) -> pl.DataFrame:
    """The ``meta__`` frame that keys a serving history as one game, in order.

    Every row is the same game and ``meta__play_id`` counts the plays, so a
    window can neither reach into another game (there is no other game) nor read
    a play out of order.
    """
    return pl.DataFrame(
        {
            "meta__game_id": pl.Series([game_id] * n_rows, dtype=pl.String),
            "meta__play_id": pl.Series([float(i) for i in range(1, n_rows + 1)], dtype=pl.Float64),
        }
    )


@contextmanager
def _windowed(model: Any, meta: pl.DataFrame | None) -> Iterator[Any]:
    """Bind ``meta`` to an ensemble's registry members for the duration."""
    if meta is None or not getattr(model, "members", None):
        yield model
        return
    from nfl.models.train_ensemble import member_context

    with member_context(model, meta) as bound:
        yield bound


def _predict_with_confidence(
    model: Any, frame: pl.DataFrame, meta: pl.DataFrame | None = None
) -> tuple[float, float, str, dict[str, float]]:
    """``(p, confidence, method, components)`` from whatever surface the model has.

    ``frame`` may be a whole game's play history; the answer is always the
    **last** row, the snap being predicted. The earlier rows are there so a
    sequence member sees the window it was trained with — see
    :func:`build_feature_rows`.
    """

    def last(value: Any) -> float:
        return float(np.asarray(value, dtype=float).ravel()[-1])

    with _windowed(model, meta) as bound:
        if hasattr(bound, "predict_with_confidence"):
            probs, conf, parts = bound.predict_with_confidence(frame)
            components = {str(k): last(v) for k, v in parts.items()}
            # Report the definition the ensemble actually chose, not a fixed
            # label: the ledger's ``confidence_method`` is how a past
            # prediction's number is interpreted, and "ensemble_spread" would be
            # a lie for any other mode.
            mode = str(getattr(bound, "confidence_mode", "") or "ensemble_spread")
            return last(probs), last(conf), mode, components
        raw = (
            bound.predict_proba(frame, meta)
            if meta is not None and _accepts_meta(bound)
            else bound.predict_proba(frame)
        )
    p = last(raw)
    # A single model cannot disagree with itself: distance from the coin flip is
    # the honest fallback, and it is exactly what the backtester reports.
    return p, min(1.0, abs(p - 0.5) * 2.0), "abs_margin", {"p_home": p}


def _market_fields(
    game_id: str, schedules: pl.DataFrame, timeout: float
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """A best-effort Kalshi mid for the ledger. **Evaluation only** (ADR 0001).

    Every failure mode — no module, no market, no network, a slow book — is a
    debug log and a null column. A prediction must never depend on the market.
    """
    try:
        from nfl.data.market import kalshi
    except ImportError as exc:  # pragma: no cover - module is in-tree
        log.debug("kalshi module unavailable: %s", exc)
        return {}, None
    snapshot_game = getattr(kalshi, "snapshot_game", None)
    if snapshot_game is None:  # pragma: no cover - defensive
        return {}, None
    try:
        client = kalshi.KalshiClient(timeout=timeout, max_retries=1, backoff=0.0)
        with client:
            row = snapshot_game(game_id, schedule=schedules, client=client)
    except Exception as exc:  # noqa: BLE001 - never let the market break a prediction
        log.debug("kalshi snapshot failed for %s: %s", game_id, exc)
        return {}, None
    if row is None:
        return {}, None
    fields = {
        "kalshi_yes_price_home": row.kalshi_yes_price,
        "market_implied_home": row.implied_prob_home,
        "market_book": row.kalshi_ticker or "kalshi",
    }
    summary = {
        "ticker": row.kalshi_ticker,
        "implied_prob_home": row.implied_prob_home,
        "collapsed": row.market_collapsed,
    }
    return {k: v for k, v in fields.items() if v is not None}, summary


def predict_game(
    game_id: str,
    at: str | AtTime | None = None,
    source: Source = "auto",
    model: str = "production",
    *,
    schedules: pl.DataFrame | str | Path | None = None,
    pbp: pl.DataFrame | str | Path | None = None,
    models_root: Path | None = None,
    ledger_root: Path | None = None,
    cache_root: Path | None = None,
    write_ledger: bool = True,
    use_cache: bool = True,
    refresh_pregame: bool = False,
    attach_market: bool | None = None,
    market_timeout: float = 2.0,
) -> PredictionResult:
    """Predict ``P(home win)`` for ``game_id`` at ``at``.

    Args:
        game_id: nflverse id, e.g. ``2024_01_BAL_KC``.
        at: ``"Q4 05:00"``, an :class:`~nfl.data.live.base.AtTime`, or ``None``
            for the feed's current state.
        source: ``espn`` (live wire), ``replay`` (nflverse play-by-play) or
            ``auto`` — replay when the game is final and its pbp is on disk.
        model: ``production``, ``name`` or ``name:version``.
        write_ledger: append the prediction to ``data/ledger/predictions``.
        attach_market: fetch a Kalshi mid for the ledger. ``None`` (default)
            means "only for a live ESPN snapshot", where a market exists at all.

    Returns:
        A :class:`PredictionResult`. ``collapsed`` reports the 15/85 trade gate;
        nothing here acts on it.
    """
    started = time.perf_counter()
    at_time = parse_at(at) if not isinstance(at, AtTime) else at
    warnings: list[str] = []

    # The model is resolved first, before any I/O against a feed: an empty
    # registry is the cheapest failure available and there is no point fetching
    # a game we cannot predict.
    model_obj, ref, model_warnings = load_model(model, root=models_root)
    warnings.extend(model_warnings)
    for warning in model_warnings:
        log.warning("%s", warning)

    schedule_frame = load_schedule_frame(schedules)
    resolved_source = resolve_source(game_id, source, schedules=schedule_frame, pbp=pbp)

    try:
        snapshot = _snapshot(game_id, at_time, resolved_source, schedules=schedule_frame, pbp=pbp)
    except (KeyError, FileNotFoundError) as exc:
        if source != "auto" or resolved_source == "espn":
            raise PredictionError(
                f"cannot snapshot {game_id} from {resolved_source}: {exc}"
            ) from exc
        warnings.append(f"replay unavailable ({exc}); falling back to the ESPN feed")
        log.warning("replay unavailable for %s (%s); using espn", game_id, exc)
        resolved_source = "espn"
        snapshot = _snapshot(game_id, at_time, "espn", schedules=schedule_frame, pbp=pbp)

    history = build_feature_rows(
        snapshot,
        game_id=game_id,
        schedules=schedule_frame,
        pbp=pbp,
        cache_root=cache_root,
        refresh_pregame=refresh_pregame,
        use_cache=use_cache,
        wanted=ref.feature_names or None,
    )
    values = history[-1]
    names = _feature_names_for(model_obj, ref, values)

    # A sequence member is fed the whole game so far and its answer is read off
    # the last row; everything else is fed that row alone. Both produce the same
    # prediction for a non-sequence model — this is a window, not a rescoring.
    windowed = needs_play_window(model_obj)
    batch = history if windowed else [values]
    frame, missing = _model_frame(batch, names)
    meta = serving_meta(game_id, len(batch)) if windowed else None
    if windowed:
        log.debug("windowing %s over %d plays of %s", ref, len(batch), game_id)
    if missing:
        log.warning(
            "%d of %d model input(s) not produced by the serving path; sent as null: %s",
            len(missing),
            len(names),
            ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
        )

    p_home, confidence, method, components = _predict_with_confidence(model_obj, frame, meta)
    if not math.isfinite(p_home):  # pragma: no cover - defensive
        raise PredictionError(f"{ref} returned a non-finite probability for {game_id}")
    p_home = float(min(max(p_home, 0.0), 1.0))

    ordered = [values.get(name) for name in names]
    hash_ = ledger_mod.feature_hash(ordered, names)
    nonnull = sum(1 for v in ordered if v is not None and not _is_nan(v))
    latency_ms = (time.perf_counter() - started) * 1000.0

    if attach_market is None:
        attach_market = resolved_source == "espn" and snapshot.state in ("pre", "in")
    market_fields: dict[str, Any] = {}
    market_summary: dict[str, Any] | None = None
    if attach_market:
        market_fields, market_summary = _market_fields(game_id, schedule_frame, market_timeout)

    result = PredictionResult(
        game_id=game_id,
        p_home=p_home,
        p_away=1.0 - p_home,
        confidence=float(min(max(confidence, 0.0), 1.0)),
        confidence_method=method,
        model=ref,
        snapshot=SnapshotSummary.from_snapshot(snapshot),
        feature_hash=hash_,
        latency_ms=latency_ms,
        components=components,
        missing_features=tuple(missing),
        features_nonnull=nonnull,
        features_total=len(names),
        source=resolved_source,
        market=market_summary,
        warnings=tuple(warnings),
    )
    if not write_ledger:
        return result

    row = _ledger_row(result, snapshot, components, market_fields)
    paths = ledger_mod.append([row], root=ledger_root)
    return _replace(result, prediction_id=str(row["prediction_id"]), ledger_paths=tuple(paths))


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def _replace(result: PredictionResult, **changes: Any) -> PredictionResult:
    return replace(result, **changes)


def _ledger_row(
    result: PredictionResult,
    snapshot: LiveSnapshot,
    components: Mapping[str, float],
    market_fields: Mapping[str, Any],
) -> dict[str, Any]:
    members = {k: v for k, v in components.items() if k.startswith("member__")}
    ensemble_components = (
        json.dumps([{"name": k.removeprefix("member__"), "p": v} for k, v in members.items()])
        if members
        else None
    )
    return ledger_mod.prediction_row(
        game_id=result.game_id,
        p_home=result.p_home,
        confidence=result.confidence,
        model_name=result.model.name,
        model_version=result.model.version,
        model_kind=result.model.kind,
        confidence_method=result.confidence_method,
        source=result.source,
        is_live=result.source == "espn" and snapshot.state == "in",
        season=snapshot.season,
        week=snapshot.week,
        home_team=snapshot.home_team,
        away_team=snapshot.away_team,
        snapshot_play_index=snapshot.play_count,
        snapshot_period=snapshot.period,
        snapshot_clock=seconds_to_clock(snapshot.clock_seconds),
        snapshot_game_seconds_remaining=snapshot.game_seconds_remaining,
        home_score=snapshot.home_score,
        away_score=snapshot.away_score,
        posteam=snapshot.possession,
        down=snapshot.down,
        ydstogo=snapshot.distance,
        yardline_100=snapshot.yards_to_endzone,
        feature_hash=result.feature_hash,
        features_nonnull_count=result.features_nonnull,
        features_total_count=result.features_total,
        latency_ms=result.latency_ms,
        ensemble_components=ensemble_components,
        ensemble_member_probs=list(members.values()) or None,
        **dict(market_fields),
    )


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #


def clock_sensitive_columns() -> frozenset[str]:
    """Columns the two paths may differ on because the clock does.

    :data:`nfl.features.ingame.SNAPSHOT_CLOCK_SENSITIVE_COLUMNS` plus the
    ``bi__`` clock block. Computed lazily so importing this module does not drag
    in the feature builders.
    """
    from nfl.features.ingame import SNAPSHOT_CLOCK_SENSITIVE_COLUMNS

    return SNAPSHOT_CLOCK_SENSITIVE_COLUMNS | _BASELINE_CLOCK_SENSITIVE


def _target_play_id(pbp: pl.DataFrame, game_id: str, at: AtTime) -> int | None:
    """The play the snapshot is poised at: the first one not yet run at ``at``."""
    frame = pbp.filter(pl.col("game_id") == game_id)
    if frame.height == 0:
        return None
    sort_key = "order_sequence" if "order_sequence" in frame.columns else "play_id"
    frame = frame.sort(sort_key, nulls_last=False)
    quarter = pl.col("qtr").cast(pl.Int64, strict=False)
    clock = pl.col("quarter_seconds_remaining").cast(pl.Int64, strict=False)
    after = frame.filter(
        (quarter > at.quarter) | ((quarter == at.quarter) & (clock <= at.clock_seconds))
    )
    if after.height == 0:
        return None
    return int(after.get_column("play_id").cast(pl.Int64, strict=False).item(0))


def _equal(serve: Any, offline: Any, *, tolerance: float = 1e-6) -> bool:
    if serve is None or _is_nan(serve):
        return offline is None or _is_nan(offline)
    if offline is None or _is_nan(offline):
        return False
    if isinstance(serve, str) or isinstance(offline, str):
        return str(serve) == str(offline)
    try:
        return abs(float(serve) - float(offline)) <= tolerance
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return bool(serve == offline)


def check_parity(
    game_id: str,
    at: str | AtTime,
    *,
    pbp: pl.DataFrame | str | Path | None = None,
    schedules: pl.DataFrame | str | Path | None = None,
    dataset: pl.DataFrame | None = None,
    cache_root: Path | None = None,
    use_cache: bool = False,
    tolerance: float = 1e-6,
) -> ParityReport:
    """Assert the replay serving row equals the offline dataset row for one play.

    This is the check that makes ``/predict`` on history meaningful: if the
    feature vector the model is served at ``Q4 05:00`` is not the vector it was
    trained on for that same snap, every backtest number is about a different
    model than the one in production.

    Columns excluded, each for a stated reason:

    * :data:`nfl.features.ingame.SNAPSHOT_UNKNOWABLE_COLUMNS` — per-play EPA and
      statistical detail a snapshot's play list does not carry. Null on the
      serving path by design.
    * clock-sensitive columns — a snapshot rewound to ``Q4 05:00`` reports 300
      seconds; the play-by-play row for the snap reports the clock at the snap
      (4:58 in the BAL @ KC fixture). Reported separately, never a failure.
    * columns the offline dataset does not carry (a builder that was skipped).
    """
    from nfl.features.assemble import assemble_dataset, feature_columns
    from nfl.features.ingame import SNAPSHOT_UNKNOWABLE_COLUMNS

    at_time = parse_at(at) if not isinstance(at, AtTime) else at
    if at_time is None:
        raise PredictionError("check_parity needs an --at; 'latest' has no play-by-play row")

    schedule_frame = load_schedule_frame(schedules)
    row = _schedule_row(game_id, schedule_frame)
    season = row.get("season")

    if isinstance(pbp, pl.DataFrame):
        pbp_frame = pbp
    else:
        from nfl.features.ingame import load_pbp_frame

        pbp_frame = (
            pl.read_parquet(pbp)
            if pbp is not None and Path(pbp).is_file()
            else load_pbp_frame(None if season is None else [int(season)])
        )
    pbp_frame = pbp_frame.filter(pl.col("game_id") == game_id)

    snapshot = _snapshot(game_id, at_time, "replay", schedules=schedule_frame, pbp=pbp_frame)
    served = build_feature_row(
        snapshot,
        game_id=game_id,
        schedules=schedule_frame,
        pbp=pbp_frame,
        cache_root=cache_root,
        use_cache=use_cache,
    )

    if dataset is None:
        dataset = assemble_dataset(
            None if season is None else [int(season)], pbp=pbp_frame, schedules=schedule_frame
        )
    play_id = _target_play_id(pbp_frame, game_id, at_time)
    offline = dataset.filter(
        (pl.col("meta__game_id") == game_id) & (pl.col("meta__play_id") == play_id)
    )
    if offline.height == 0:
        raise PredictionError(
            f"no dataset row for {game_id} play {play_id}; run `nfl features build` for {season}"
        )
    offline_row = offline.row(0, named=True)
    offline_features = set(feature_columns(dataset))

    clock_columns = clock_sensitive_columns()
    compared: list[str] = []
    mismatches: list[tuple[str, Any, Any]] = []
    clock_diffs: list[tuple[str, Any, Any]] = []
    skipped: list[str] = []
    missing_offline: list[str] = []

    for name in sorted(served):
        if name not in offline_features:
            missing_offline.append(name)
            continue
        if name in SNAPSHOT_UNKNOWABLE_COLUMNS:
            skipped.append(name)
            continue
        serve_value, offline_value = served[name], offline_row[name]
        if name in clock_columns:
            if not _equal(serve_value, offline_value, tolerance=tolerance):
                clock_diffs.append((name, serve_value, offline_value))
            continue
        compared.append(name)
        if not _equal(serve_value, offline_value, tolerance=tolerance):
            mismatches.append((name, serve_value, offline_value))

    # The other direction: a column the training row carries and the serving
    # path never produced is a parity break too — the model would be fed a null
    # where it learned a value — and comparing only the served keys would miss
    # a whole builder going silently absent at serve time.
    missing_serving: list[str] = []
    for name in sorted(offline_features - set(served)):
        if name in SNAPSHOT_UNKNOWABLE_COLUMNS or name in clock_columns:
            skipped.append(name)
            continue
        missing_serving.append(name)
        if offline_row[name] is not None and not _is_nan(offline_row[name]):
            mismatches.append((name, "<not produced>", offline_row[name]))

    return ParityReport(
        game_id=game_id,
        at=str(at_time),
        play_id=play_id,
        compared=tuple(compared),
        mismatches=tuple(mismatches),
        clock_sensitive=tuple(clock_diffs),
        skipped_unknowable=tuple(sorted(skipped)),
        missing_offline=tuple(missing_offline),
        missing_serving=tuple(missing_serving),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_prediction(result: PredictionResult, console: Any | None = None) -> None:
    """Print a prediction as one rich panel: state, probabilities, gate."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = console or Console()
    snapshot = result.snapshot

    state = Table(box=None, pad_edge=False, show_header=False)
    state.add_column("field", style="dim")
    state.add_column("value")
    state.add_row(
        "score",
        f"{snapshot.away_team} {snapshot.away_score} — {snapshot.home_team} {snapshot.home_score}",
    )
    state.add_row(
        "clock", f"{snapshot.label}  [dim]({snapshot.game_seconds_remaining}s left)[/dim]"
    )
    if snapshot.possession:
        detail = snapshot.possession
        if snapshot.down is not None:
            detail += f", {_ordinal(snapshot.down)} & {snapshot.distance if snapshot.distance is not None else '?'}"
        if snapshot.yards_to_endzone is not None:
            detail += f" at the {snapshot.yards_to_endzone}"
        state.add_row("possession", detail)
    else:
        state.add_row("possession", "[dim]none[/dim]")

    probabilities = Table(box=None, pad_edge=False, show_header=False)
    probabilities.add_column("field", style="dim")
    probabilities.add_column("value")
    probabilities.add_row(
        f"P({snapshot.home_team} win)", Text(f"{result.p_home:.4f}", style="bold cyan")
    )
    probabilities.add_row(f"P({snapshot.away_team} win)", f"{result.p_away:.4f}")
    probabilities.add_row(
        "confidence", f"{result.confidence:.3f}  [dim]({result.confidence_method})[/dim]"
    )
    probabilities.add_row("model", f"{result.model.name} [dim]{result.model.version}[/dim]")
    gate = (
        Text("COLLAPSED — outside the 15/85 band, no position", style="bold yellow")
        if result.collapsed
        else Text("open — inside the 15/85 band", style="bold green")
    )
    probabilities.add_row("trade gate", gate)
    probabilities.add_row(
        "features",
        f"{result.features_nonnull}/{result.features_total} non-null"
        + (
            f"  [yellow]{len(result.missing_features)} missing[/yellow]"
            if result.missing_features
            else ""
        ),
    )
    probabilities.add_row("latency", f"{result.latency_ms:,.0f} ms")
    if result.market is not None and result.market.get("implied_prob_home") is not None:
        probabilities.add_row(
            "market [dim](eval only)[/dim]",
            f"{result.market['implied_prob_home']:.4f} [dim]{result.market.get('ticker') or ''}[/dim]",
        )

    parts: list[Any] = [state, Text(""), probabilities]

    # An ensemble's answer is three answers and a calibrator. Printing only the
    # blend hides which member moved it — and a member that has quietly gone
    # flat (a GRU windowing over one play, a GBDT re-binning itself) looks
    # exactly like a member that agrees with the others.
    members = {
        k.removeprefix("member__"): v
        for k, v in result.components.items()
        if k.startswith("member__")
    }
    if members:
        breakdown = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
        breakdown.add_column("member", style="dim")
        breakdown.add_column(f"P({snapshot.home_team})", justify="right")
        breakdown.add_column("vs blend", justify="right")
        for name, value in members.items():
            breakdown.add_row(name, f"{value:.4f}", f"{value - result.p_home:+.4f}")
        for label in ("raw", "calibrated"):
            if label in result.components:
                breakdown.add_row(f"[dim]{label}[/dim]", f"{result.components[label]:.4f}", "")
        parts += [Text(""), breakdown]

    for warning in result.warnings:
        parts.append(Text(f"! {warning}", style="yellow"))
    if result.prediction_id:
        parts.append(Text(f"ledger {result.prediction_id}", style="dim"))

    console.print(
        Panel(
            Group(*parts),
            title=f"{snapshot.away_team} @ {snapshot.home_team}  [dim]{result.game_id}[/dim]",
            subtitle=f"{snapshot.state} · via {result.source}",
            border_style="yellow" if result.collapsed else "green",
        )
    )


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")
