"""Fit the v1 ensemble out of already-trained registry artifacts, and gate it.

ADR 0003 decided the shape: **naive logistic + XGBoost + TF GRU sequence model**,
blended by ``logit_mean`` (or a logistic stacker if it wins on val Q4), then a
**phase-aware calibrator** fit on the validation split, with **confidence chosen
empirically** rather than asserted. This module is that decision as code, plus
the §5 promotion gate that decides whether the result is allowed to be
PRODUCTION.

**Members come from the registry, not from a training run.** Each member is
loaded by name — the newest version, or a pin — so retraining a member is a
`nfl train --model xgb` away and the ensemble picks it up on its next fit
without a line changing here. The ensemble artifact records the exact
``(name, version)`` of every member it was fitted over, so "which XGBoost was
in that ensemble?" is answerable six months later.

**Members disagree about their feature set, and that is fine.**
:class:`RegistryMember` wraps each artifact and hands it the columns *its own*
manifest asks for: a ``feature_set`` name resolved through
``nfl.models.feature_sets.select_features`` when the sibling module is present,
otherwise the manifest's ``feature_names``. When the recorded list is not a
subset of the frame — the sequence model records *channel* names, which are
derived columns rather than dataset columns — the member is handed the whole
design matrix and left to select for itself, which every model in this package
already does by name.

**Row context.** The sequence model needs to know which game a row belongs to
and in what order, and that information lives in ``meta__`` columns which by
construction never reach a design matrix. :func:`member_context` binds the
matching ``meta__`` rows for the duration of a scoring call; outside one, a
member that needs context says so rather than quietly windowing across games.

**Confidence is measured, not declared.** Candidate definitions — member
disagreement, an MC-dropout head if any member exposes one, and the blend of
the two — are ranked by
:func:`~nfl.models.ensemble.compare_confidence_definitions` on the *validation
Q4 rows*, because that is the population the project trades on. The winner is
recorded in the manifest next to the ranking that chose it.

**The gate.** :func:`promotion_gate` compares a candidate against the naive
floor on val Q4@600/300/120 log-loss *and* Brier, and requires it to be no worse
on the not-collapsed subset. Both sides come from the manifests written at
training time, so the gate is a comparison of recorded numbers rather than a
re-run that might quietly differ. ``nfl promote`` is its command-line face.
"""

# `X` is the feature-matrix name fixed by the WinProbModel contract, as an
# argument and as the local that holds one.
# ruff: noqa: N803, N806

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from nfl.features.splits import SeasonSplit, q4_at
from nfl.models import registry
from nfl.models.base import FitResult
from nfl.models.calibration import Calibrator, make_calibrator
from nfl.models.data import ModelFrames, design_matrix, labels, load_model_frames, seasons_in
from nfl.models.ensemble import (
    CLOCK_CANDIDATES,
    ConfidenceMode,
    MetaKind,
    StackedEnsemble,
    compare_confidence_definitions,
)
from nfl.models.metrics import (
    NOT_COLLAPSED_OWN_SLICE,
    NOT_COLLAPSED_SLICE,
    Q4_MARKS,
    slice_metrics,
)

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MEMBERS",
    "GATE_METRICS",
    "GATE_Q4_SLICES",
    "MEMBER_ALIASES",
    "NAIVE_FLOOR",
    "NOT_COLLAPSED_OWN_SLICE",
    "NOT_COLLAPSED_SLICE",
    "GateCheck",
    "GateResult",
    "RegistryMember",
    "UncertainRegistryMember",
    "gate_table",
    "load_member",
    "member_context",
    "promotion_gate",
    "reference_probabilities",
    "split_member_spec",
    "train_ensemble",
]

#: The v1 membership from ADR 0003 §3.
DEFAULT_MEMBERS: tuple[str, ...] = (
    "naive_score_time",
    "xgboost_winprob",
    "tf_seq_winprob",
)

#: Names a caller may use for a member, mapped to the registry directory that
#: actually holds it. The TF models register under their module name.
MEMBER_ALIASES: dict[str, str] = {
    "naive": "naive_score_time",
    "xgb": "xgboost_winprob",
    "xgboost": "xgboost_winprob",
    "seq": "tf_seq",
    "tf_seq_winprob": "tf_seq",
    "mlp": "tf_mlp",
    "tf_mlp_winprob": "tf_mlp",
}

#: ``manifest["kind"]`` or model name -> ``"module:QualName"``. Only consulted
#: for artifacts written before the registry recorded ``model_class``; a
#: manifest that carries one always wins.
_CLASS_HINTS: dict[str, str] = {
    "logistic_regression": "nfl.models.naive:ScoreTimeLogistic",
    "naive_score_time": "nfl.models.naive:ScoreTimeLogistic",
    "xgboost": "nfl.models.gbdt:XGBoostWinProb",
    "xgboost_winprob": "nfl.models.gbdt:XGBoostWinProb",
    "tf_seq": "nfl.models.tf_seq:SeqWinProb",
    "tf_mlp": "nfl.models.tf_mlp:MLPWinProb",
    "stacked_ensemble": "nfl.models.ensemble:StackedEnsemble",
    "constant": "nfl.models.dummy:ConstantModel",
    "noisy_oracle": "nfl.models.dummy:NoisyOracle",
}

#: Confidence mode -> the label it is reported under.
_CONFIDENCE_LABELS: dict[ConfidenceMode, str] = {
    "disagreement": "disagreement",
    "uncertainty": "mc_dropout",
    "combined": "blend",
    "margin": "margin",
    "margin_uncertainty": "margin_mc",
}
_CONFIDENCE_MODES: dict[str, ConfidenceMode] = {v: k for k, v in _CONFIDENCE_LABELS.items()}

#: The model the gate measures every candidate against (ADR 0003 §5).
NAIVE_FLOOR = "naive_score_time"

#: Slice names as :func:`nfl.models.metrics.slice_metrics` writes them.
GATE_Q4_SLICES: tuple[str, ...] = tuple(f"q4_{m}s" for m in Q4_MARKS)

#: Metrics the gate compares. Both must improve; a candidate that trades log
#: loss for Brier is a candidate that moved probability mass around.
GATE_METRICS: tuple[str, ...] = ("log_loss", "brier")

#: Float slack when asking "is this *not worse*". Guards against a comparison
#: failing on the last bit of a float, not against a real regression.
GATE_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Members
# --------------------------------------------------------------------------- #


def _import_class(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _registry_name(name: str, root: Path | None) -> str:
    """The registry directory holding ``name``, following :data:`MEMBER_ALIASES`."""
    for candidate in (name, MEMBER_ALIASES.get(name, "")):
        if candidate and registry.list_versions(candidate, root):
            return candidate
    known = registry.list_models(root)
    raise FileNotFoundError(
        f"no registry artifact for member {name!r} under {registry.registry_root(root)}; "
        f"available: {known or '(none — run `nfl train` first)'}"
    )


def _member_class(manifest: registry.Manifest, name: str) -> Any:
    """The class that can load this artifact, from the manifest or the hints."""
    if manifest.model_class:
        return _import_class(manifest.model_class)
    for key in (str(manifest.extra.get("kind", "")), name, manifest.name):
        hint = _CLASS_HINTS.get(key)
        if hint:
            return _import_class(hint)
    raise ValueError(
        f"manifest for {name}/{manifest.version} has no model_class and no known kind "
        f"(kind={manifest.extra.get('kind')!r}); retrain it, or add a hint to _CLASS_HINTS"
    )


def _instantiate(directory: Path, manifest: registry.Manifest, name: str) -> Any:
    """Load the artifact, whether its ``load()`` wants the directory or the file.

    The models in this package take the artifact *directory* (they write several
    files into it); the registry records the *file* its ``save()`` produced. Try
    the directory first, because that is what every current member expects, and
    fall back to the recorded artifact for anything registry-written.
    """
    klass = _member_class(manifest, name)
    try:
        return klass.load(directory)
    except (OSError, KeyError, ValueError, TypeError) as first:
        try:
            return klass.load(registry.resolve_artifact(directory, manifest.artifact or "model"))
        except (OSError, KeyError, ValueError, TypeError) as second:
            # Both attempts failed, and the *first* one is almost always the
            # informative half — the fallback is trying a path the model never
            # meant to be handed. A `TypeError` here usually means the artifact
            # was written by a newer version of the model class than the one in
            # this checkout, so say that rather than leaking a path error.
            raise RuntimeError(
                f"cannot load {name}/{manifest.version} with "
                f"{klass.__module__}:{klass.__qualname__} — "
                f"loading the artifact directory raised {first!r}, and the recorded "
                f"artifact {manifest.artifact!r} raised {second!r}. If the first is a "
                "TypeError about an unexpected argument, the artifact was written by a "
                "newer version of this model class: update the checkout, or retrain."
            ) from first


def _manifest_columns(manifest: registry.Manifest) -> list[str]:
    """The feature list an artifact recorded, under whichever key it used."""
    if manifest.feature_names:
        return list(manifest.feature_names)
    recorded = manifest.extra.get("features")
    return [str(c) for c in recorded] if isinstance(recorded, list) else []


@dataclass
class RegistryMember:
    """One registry artifact, pinned by version, wearing the model contract.

    Holds the pin rather than a copy of the weights: :meth:`save` writes the
    ``(name, version)`` pair and :meth:`load` resolves it against the registry
    again. An ensemble artifact is therefore a few kilobytes of pins rather than
    a second copy of a Keras model, and "which members were in it" is a fact
    about the artifact rather than an inference from file sizes.

    ``context`` is the ``meta__`` half of the frame currently being scored; see
    :func:`member_context`. It is ``None`` outside a scoring call, and members
    that do not need it never look.
    """

    member: str
    version: str
    root: str | None = None
    #: Feature columns this member's manifest recorded, if any.
    columns: list[str] = field(default_factory=list)
    #: ``feature_sets`` name recorded by the manifest, if any.
    feature_set: str = ""
    name: str = ""
    model: Any = field(default=None, repr=False)
    context: pl.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or self.member

    # -- loading ---------------------------------------------------------- #

    @property
    def root_path(self) -> Path | None:
        """The registry root to resolve against, or ``None`` for the default.

        A recorded root that no longer exists (a tmp registry from a past run,
        an artifact rsynced to another machine) falls back to the default rather
        than failing: the pin is ``(name, version)``, the root is a hint.
        """
        if self.root and Path(self.root).is_dir():
            return Path(self.root)
        return None

    @property
    def loaded(self) -> Any:
        """The underlying model, loaded from the registry on first use."""
        if self.model is None:
            directory = registry.model_dir(self.member, self.version, self.root_path)
            manifest = registry.read_manifest(directory)
            self.model = _instantiate(directory, manifest, self.member)
        return self.model

    @property
    def pin(self) -> dict[str, Any]:
        """What the ensemble manifest records about this member."""
        return {
            "name": self.name,
            "registry_name": self.member,
            "version": self.version,
            "feature_set": self.feature_set or None,
            "n_features": len(self.columns),
        }

    # -- frame plumbing --------------------------------------------------- #

    def _columns_for(self, X: pl.DataFrame) -> list[str] | None:
        """Which columns this member should see, or ``None`` for "all of them".

        A recorded list is honoured only when the frame actually carries every
        column in it. A partial match means the record describes something other
        than dataset columns (the sequence model records channel names), and
        slicing to the overlap would hand the model a silently truncated frame —
        far worse than handing it everything and letting it select.
        """
        if self.feature_set:
            picked = _select_feature_set(X, self.feature_set)
            if picked is not None:
                return picked
        if self.columns and set(self.columns) <= set(X.columns):
            return list(self.columns)
        return None

    def _frame_for(self, X: pl.DataFrame) -> pl.DataFrame:
        cols = self._columns_for(X)
        return X.select(cols) if cols else X

    def _context_for(self, X: pl.DataFrame) -> pl.DataFrame | None:
        """The bound context, checked against ``X``.

        The row-count check runs whether or not this member consumes the
        context: a mismatch means the caller bound a different frame than the
        one being scored, and the next member down the list may well be the one
        that silently windows across the wrong games.
        """
        if self.context is None:
            return None
        if self.context.height != X.height:
            raise ValueError(
                f"member {self.name!r}: bound context has {self.context.height} rows "
                f"but the frame has {X.height}; bind the meta of the frame being scored"
            )
        return self.context

    def _wants_context(self, method: str = "predict_proba") -> bool:
        import inspect

        try:
            params = inspect.signature(getattr(self.loaded, method)).parameters
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - C callables
            return False
        return "meta" in params

    # -- WinProbModel ----------------------------------------------------- #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        """Refuse. A pinned artifact is a fixed thing; retrain it by name."""
        raise NotImplementedError(
            f"{self.name} is a pinned registry artifact ({self.member}/{self.version}); "
            "retrain it with `nfl train` and re-fit the ensemble"
        )

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        frame = self._frame_for(X)
        context = self._context_for(X)
        model = self.loaded
        out = (
            model.predict_proba(frame, context)
            if self._wants_context()
            else model.predict_proba(frame)
        )
        return np.asarray(out, dtype=float).ravel()

    # -- persistence ------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        target = Path(path)
        if target.suffix != ".json":
            target = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "kind": "registry_member",
                    "member": self.member,
                    "version": self.version,
                    "root": self.root,
                    "columns": self.columns,
                    "feature_set": self.feature_set,
                    "name": self.name,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return target

    @classmethod
    def load(cls, path: Path) -> RegistryMember:
        target = Path(path)
        if not target.is_file() and target.with_suffix(".json").is_file():
            target = target.with_suffix(".json")
        state = json.loads(target.read_text())
        return cls(
            member=str(state["member"]),
            version=str(state["version"]),
            root=state.get("root"),
            columns=list(state.get("columns", [])),
            feature_set=str(state.get("feature_set", "")),
            name=str(state.get("name", "")),
        )


class UncertainRegistryMember(RegistryMember):
    """A member whose model exposes an MC-dropout head, so the ensemble sees it.

    Separate class rather than a flag: :class:`~nfl.models.ensemble.StackedEnsemble`
    detects an uncertainty head with ``hasattr``, so the method has to be absent
    — not present-and-raising — on members that do not have one.
    """

    def predict_with_uncertainty(self, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        frame = self._frame_for(X)
        # The GRU's MC head windows by game exactly as its point estimate does,
        # so it needs the same bound context; a member that takes no ``meta``
        # never sees one.
        context = self._context_for(X)
        model = self.loaded
        mean, std = (
            model.predict_with_uncertainty(frame, context)
            if self._wants_context("predict_with_uncertainty")
            else model.predict_with_uncertainty(frame)
        )
        return (
            np.asarray(mean, dtype=float).ravel(),
            np.asarray(std, dtype=float).ravel(),
        )


def _select_feature_set(X: pl.DataFrame, name: str) -> list[str] | None:
    """Column names for a named feature set, or ``None`` if it cannot be resolved.

    ``nfl.models.feature_sets`` is written on a sibling branch. Until it lands —
    or when it does not know this set — the caller falls back to the manifest's
    explicit column list, so an ensemble is never blocked on a module that may
    not be in the checkout.
    """
    try:
        module = importlib.import_module("nfl.models.feature_sets")
        picked = module.select_features(X, name)
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        log.debug("feature set %r unavailable (%s); falling back to manifest columns", name, exc)
        return None
    if isinstance(picked, pl.DataFrame):
        return list(picked.columns)
    return [str(c) for c in picked]


@contextmanager
def member_context(ensemble: StackedEnsemble, meta: pl.DataFrame) -> Iterator[StackedEnsemble]:
    """Bind ``meta`` to every registry member for the duration of the block.

    The sequence member windows plays by ``meta__game_id`` / ``meta__play_id``,
    which a design matrix cannot carry (they are the answer, near enough — see
    :func:`nfl.models.artifacts.assert_model_input_frame`). Pass the *same*
    frame the design matrix was selected from; row counts are checked, so a
    mismatched context is an error rather than a silently wrong window.
    """
    from nfl.features.assemble import meta_columns

    # Only the ``meta__`` half travels: a member is handed the game key and the
    # play order, never the label or a second copy of the features it already
    # has. Passing the whole frame would put ``label__home_win`` one column away
    # from a model input, which is exactly the adjacency ADR 0001 and the
    # leakage tests exist to prevent.
    keys = (
        meta.select(meta_columns(meta))
        if any(c.startswith("meta__") for c in meta.columns)
        else meta
    )
    bound = [m for m in ensemble.members if isinstance(m, RegistryMember)]
    for m in bound:
        m.context = keys
    try:
        yield ensemble
    finally:
        for m in bound:
            m.context = None


def split_member_spec(spec: str) -> tuple[str, str | None]:
    """``"xgboost_winprob:20260823T052439"`` -> ``("xgboost_winprob", "2026...")``.

    Pinning by version is the difference between "the ensemble that was
    evaluated" and "whatever the registry happened to hold that afternoon".
    Registry names never contain a colon and versions are timestamps, so the
    split is unambiguous.
    """
    name, sep, version = str(spec).partition(":")
    return name.strip(), (version.strip() or None) if sep else None


def load_member(name: str, version: str | None = None, root: Path | None = None) -> RegistryMember:
    """Load one member by registry name, newest version unless pinned."""
    registry_name = _registry_name(name, root)
    resolved = version or registry.latest(registry_name, root)
    directory = registry.model_dir(registry_name, resolved, root)
    manifest = registry.read_manifest(directory)
    model = _instantiate(directory, manifest, registry_name)
    klass = (
        UncertainRegistryMember if hasattr(model, "predict_with_uncertainty") else RegistryMember
    )
    return klass(
        member=registry_name,
        version=resolved,
        root=str(registry.registry_root(root)) if root is not None else None,
        columns=_manifest_columns(manifest),
        feature_set=str(manifest.extra.get("feature_set", "") or ""),
        name=name,
        model=model,
    )


def reference_probabilities(
    frame: pl.DataFrame,
    features: Sequence[str],
    *,
    floor: str = NAIVE_FLOOR,
    version: str | None = None,
    root: Path | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """The floor model's probability on ``frame``, and the pin it came from.

    This is what keys the ``not_collapsed`` slice (ADR 0003 §5): scoring the
    candidate on *the floor's* live rows is what makes the two sides of the gate
    the same population. Returns ``(None, None)`` — and logs why — when the
    floor is not in the registry, which is the state of a fresh checkout and of
    every test that has not trained one; callers then fall back to the model's
    own band, and the gate says so.
    """
    try:
        member = load_member(floor, version, root)
    except (FileNotFoundError, ValueError) as exc:
        log.warning(
            "no floor artifact for the not-collapsed reference (%s); "
            "the slice falls back to each model's own probability",
            exc,
        )
        return None, None
    X = design_matrix(frame, features)
    with member_context(StackedEnsemble(members=[member]), frame):
        probs = np.asarray(member.predict_proba(X), dtype=float).ravel()
    return probs, f"{member.member}/{member.version}"


# --------------------------------------------------------------------------- #
# Clock / slicing helpers
# --------------------------------------------------------------------------- #


def _clock(frame: pl.DataFrame) -> np.ndarray | None:
    """Seconds remaining off whichever column carries it, or ``None``."""
    for candidate in (*CLOCK_CANDIDATES, "meta__game_seconds_remaining"):
        if candidate in frame.columns:
            return frame.get_column(candidate).cast(pl.Float64, strict=False).to_numpy()
    return None


def _apply_calibrator(
    calibrator: Calibrator | None, raw: np.ndarray, clock: np.ndarray | None
) -> np.ndarray:
    if calibrator is None:
        return raw
    if getattr(calibrator, "kind", "") == "phase_aware":
        if clock is None:
            raise ValueError("phase-aware calibration needs a game-clock column on the frame")
        return calibrator.transform(raw, seconds_remaining=clock)
    return calibrator.transform(raw)


_ROW_INDEX = "_ens__row"


def q4_row_indices(frame: pl.DataFrame, marks: Sequence[int] = Q4_MARKS) -> np.ndarray:
    """Positions of the Q4-mark rows within ``frame`` — one per game per mark.

    Positions rather than a sub-frame on purpose. The sequence member's window
    is built from the rows *around* each snapshot, so scoring a Q4-only frame
    would hand it a game with one play in it. Score the whole split once, then
    index into the result.
    """
    indexed = frame.with_row_index(_ROW_INDEX)
    picked: list[np.ndarray] = []
    for mark in marks:
        rows = q4_at(indexed, int(mark))
        if rows.height:
            picked.append(rows.get_column(_ROW_INDEX).to_numpy())
    if not picked:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(picked)).astype(int)


# --------------------------------------------------------------------------- #
# Confidence selection
# --------------------------------------------------------------------------- #


def _confidence_candidates(
    ensemble: StackedEnsemble,
    member_probs: np.ndarray,
    uncertainty: np.ndarray | None,
    p: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """One confidence vector per definition the members can actually support.

    Member disagreement and calibrated margin always exist. The MC-dropout σ,
    and the two compounds that use it, exist only if some member exposes
    ``predict_with_uncertainty`` — since the GRU grew an MC head that is now the
    ADR 0003 §3 membership's normal state, but an ensemble assembled from older
    artifacts will still rank the two definitions it has and say so.
    """
    modes: list[ConfidenceMode] = ["disagreement"]
    if p is not None:
        modes.append("margin")
    if uncertainty is not None and np.asarray(uncertainty).size:
        modes += ["uncertainty", "combined"]
        if p is not None:
            modes.append("margin_uncertainty")
    saved = ensemble.confidence_mode
    out: dict[str, np.ndarray] = {}
    try:
        for mode in modes:
            ensemble.confidence_mode = mode
            conf, _ = ensemble.confidence_from(member_probs, uncertainty, p)
            out[_CONFIDENCE_LABELS[mode]] = conf
    finally:
        ensemble.confidence_mode = saved
    return out


def choose_confidence(
    ensemble: StackedEnsemble,
    p: np.ndarray,
    y: np.ndarray,
    member_probs: np.ndarray,
    uncertainty: np.ndarray | None,
    rows: np.ndarray,
    *,
    requested: str | None = None,
) -> tuple[ConfidenceMode, pl.DataFrame]:
    """Rank candidate confidence definitions on ``rows`` and return the winner.

    ``rows`` is the Q4 subset (see :func:`q4_row_indices`): a definition that
    separates error across a whole season but not inside the last ten minutes is
    the wrong definition for this project, whatever its overall numbers say.

    The ranking is by ``spearman`` — the rank correlation between a bin's
    confidence and its Brier — ascending, so the winner is the most *negatively*
    correlated: more confidence, less error. A definition whose correlation is
    positive is worse than useless (it is confident exactly where it is wrong),
    and :func:`train_ensemble` refuses to ship one.
    """
    candidates = _confidence_candidates(ensemble, member_probs, uncertainty, p)
    forced: ConfidenceMode | None = None
    if requested is not None:
        named = _CONFIDENCE_MODES.get(requested, requested)
        if named not in _CONFIDENCE_LABELS:
            raise ValueError(
                f"unknown confidence {requested!r}; expected one of {sorted(_CONFIDENCE_MODES)}"
            )
        forced = cast("ConfidenceMode", named)

    if rows.size < 2:
        # Nothing to rank against: say so rather than crowning a winner on two
        # rows. A single *candidate* is still ranked — the reliability numbers
        # are the evidence that the one definition on offer works at all.
        table = pl.DataFrame(
            {
                "confidence": list(candidates),
                "note": [f"not ranked: {rows.size} val Q4 rows"] * len(candidates),
            }
        )
        return (forced or "disagreement"), table

    table = compare_confidence_definitions(
        p[rows], y[rows], {k: v[rows] for k, v in candidates.items()}
    )
    if forced is not None:
        return forced, table
    return _CONFIDENCE_MODES[str(table.get_column("confidence")[0])], table


def _winning_spearman(ranking: pl.DataFrame, mode: ConfidenceMode) -> float | None:
    """The chosen definition's rank correlation with Brier, if it was ranked."""
    if "spearman" not in ranking.columns:
        return None
    row = ranking.filter(pl.col("confidence") == _CONFIDENCE_LABELS[mode])
    return float(row.get_column("spearman")[0]) if row.height else None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Precomputed:
    """A model that has already predicted. Lets a backtest reuse one pass."""

    probs: np.ndarray
    name: str

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if X.height != self.probs.size:
            raise ValueError(f"{self.probs.size} precomputed probabilities for {X.height} rows")
        return self.probs


def _score(
    ensemble: StackedEnsemble, frame: pl.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """One member pass over ``frame``: ``(p, confidence, member_probs, uncertainty)``."""
    X = design_matrix(frame, features)
    with member_context(ensemble, frame):
        member_probs = ensemble.member_probabilities(X)
        uncertainty = ensemble.member_uncertainty(X)
    raw = ensemble.combine(member_probs)
    p = _apply_calibrator(ensemble.calibrator, raw, _clock(frame))
    conf, _ = ensemble.confidence_from(member_probs, uncertainty, p)
    return p, conf, member_probs, uncertainty


def train_ensemble(
    members: Sequence[str] = DEFAULT_MEMBERS,
    meta: MetaKind = "logit_mean",
    *,
    versions: Mapping[str, str] | None = None,
    seasons: str | int | Sequence[int] | None = None,
    train_through: int = 2022,
    val_season: int = 2023,
    frames: ModelFrames | None = None,
    root: Path | None = None,
    models_root: Path | None = None,
    reports_root: Path | None = None,
    stride: int = 1,
    name: str = "ensemble_v1",
    calibrator: str = "phase_aware",
    calibrator_base: str = "isotonic",
    phase_min_samples: int = 500,
    confidence: str | None = None,
    floor: str = NAIVE_FLOOR,
    floor_version: str | None = None,
    save: bool = True,
    report: bool = True,
) -> tuple[StackedEnsemble, FitResult]:
    """Fit the ADR 0003 ensemble over the latest artifacts of ``members``.

    The meta-learner and the calibrator are fit on **val**, never on train:
    members are in-sample on the training seasons, and a stacker fitted there
    ranks them by how well they memorised rather than by how well they predict.

    Args:
        members: registry names (or :data:`MEMBER_ALIASES`), each optionally
            pinned as ``"name:version"``.
        meta: ``"logit_mean"`` (the ADR default) or ``"logistic_stacker"``.
        versions: pin specific member versions, ``{member_name: version}``.
        seasons / train_through / val_season / stride / root: dataset selection,
            as :func:`nfl.models.data.load_model_frames` takes them.
        frames: pre-loaded splits, skipping the dataset read (tests).
        models_root: registry root; ``None`` is ``data/models``.
        reports_root: data root the markdown report is written under.
        calibrator: calibrator kind; ``"phase_aware"`` per ADR 0003 §3.
        confidence: force a definition instead of choosing one empirically.
        floor / floor_version: the model whose probability band keys the
            ``not_collapsed`` slice, so the gate scores both sides on the same
            rows. Defaults to the latest :data:`NAIVE_FLOOR`.

    Returns:
        The fitted ensemble and its :class:`~nfl.models.base.FitResult`, whose
        ``extra`` carries the member pins, the confidence ranking and the val /
        test slice tables — the same numbers the promotion gate reads back.
    """
    split = SeasonSplit(
        train_through=train_through, val_season=val_season, test_from=val_season + 1
    )
    data = frames or load_model_frames(seasons, split=split, root=root, stride=stride)
    if data.val.is_empty():
        raise ValueError(
            f"no validation rows for season {val_season}; the meta-learner and the "
            "calibrator are fit on val, so there is nothing to fit"
        )

    pinned = dict(versions or {})
    loaded: list[RegistryMember] = []
    for spec in members:
        member_name, spec_version = split_member_spec(spec)
        loaded.append(
            load_member(member_name, spec_version or pinned.get(member_name), models_root)
        )
    log.info("ensemble members: %s", [f"{m.member}/{m.version}" for m in loaded])

    calib: Calibrator | None = None
    if calibrator and calibrator != "none":
        kwargs: dict[str, Any] = {}
        if calibrator == "phase_aware":
            kwargs = {"base": calibrator_base, "min_samples": phase_min_samples}
        calib = make_calibrator(calibrator, **kwargs)

    ensemble = StackedEnsemble(members=list(loaded), meta=meta, calibrator=calib, name=name)

    x_train, y_train = data.as_xy("train")
    x_val, y_val = data.as_xy("val")
    with member_context(ensemble, data.val):
        result = ensemble.fit(x_train, y_train, x_val, y_val)

    # -- confidence, chosen on the val Q4 rows ---------------------------- #
    p_val, _, member_probs, uncertainty = _score(ensemble, data.val, data.features)
    q4_rows = q4_row_indices(data.val)
    mode, ranking = choose_confidence(
        ensemble, p_val, y_val, member_probs, uncertainty, q4_rows, requested=confidence
    )
    ensemble.confidence_mode = mode
    conf_val, _ = ensemble.confidence_from(member_probs, uncertainty, p_val)
    log.info("confidence definition: %s (%d val Q4 rows)", mode, q4_rows.size)
    spearman = _winning_spearman(ranking, mode)
    if spearman is not None and spearman >= 0.0:
        log.warning(
            "confidence %r has spearman %+.4f on val Q4 rows: it is not negatively "
            "correlated with Brier, so it does not predict error. Shipping it anyway "
            "would be publishing a number that means nothing (ADR 0003 §4).",
            mode,
            spearman,
        )
    result.extra["confidence_spearman"] = spearman

    # -- slices ------------------------------------------------------------ #
    # ``not_collapsed`` is keyed on the floor's band, so the gate compares the
    # candidate and the floor over one row set (ADR 0003 §5).
    ref_val, ref_pin = reference_probabilities(
        data.val, data.features, floor=floor, version=floor_version, root=models_root
    )
    val_table = slice_metrics(data.val, p_val, y_val, reference_probs=ref_val)
    result.extra["val_slices"] = val_table.to_dicts()
    p_test = conf_test = None
    test_table = None
    if data.test.height:
        y_test = labels(data.test)
        p_test, conf_test, _, _ = _score(ensemble, data.test, data.features)
        ref_test, _ = reference_probabilities(
            data.test, data.features, floor=floor, version=floor_version, root=models_root
        )
        test_table = slice_metrics(data.test, p_test, y_test, reference_probs=ref_test)
        result.extra["test_slices"] = test_table.to_dicts()

    result.extra.update(
        {
            "members": [m.pin for m in loaded],
            "not_collapsed_reference": ref_pin,
            "confidence_mode": mode,
            "confidence_ranking": ranking.to_dicts(),
            "confidence_n_val_q4_rows": int(q4_rows.size),
            "calibrator": getattr(calib, "kind", None),
            "meta": meta,
        }
    )

    manifest: registry.Manifest | None = None
    if save:
        manifest = registry.Manifest.from_fit_result(
            result,
            ensemble,
            train_seasons=seasons_in(data.train),
            fingerprint=registry.dataset_fingerprint(
                feature_names=list(data.features),
                rows_by_season=_rows_by_season(data),
            ),
        )
        manifest.extra.update(result.extra)
        registry.save_model(ensemble, manifest=manifest, root=models_root)
        log.info("wrote %s", registry.model_dir(manifest.name, manifest.version, models_root))

    if report:
        path = write_report(
            ensemble,
            data,
            val=(p_val, conf_val),
            test=(p_test, conf_test),
            val_table=val_table,
            test_table=test_table,
            ranking=ranking,
            members=loaded,
            reference=ref_pin,
            reports_root=reports_root,
            models_root=models_root,
        )
        result.extra["report"] = str(path)
        log.info("wrote %s", path)

    return ensemble, result


def _rows_by_season(data: ModelFrames) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in (data.train, data.val, data.test):
        if "meta__season" not in frame.columns or frame.is_empty():
            continue
        for season, n in frame.group_by("meta__season").len().iter_rows():
            if season is not None:
                counts[str(season)] = counts.get(str(season), 0) + int(n)
    return counts


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _md_table(frame: pl.DataFrame, digits: int = 4) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return "—" if value != value else f"{value:.{digits}f}"
        return "—" if value is None else str(value)

    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.iter_rows()]
    return "\n".join(lines)


def write_report(
    ensemble: StackedEnsemble,
    data: ModelFrames,
    *,
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray | None, np.ndarray | None],
    val_table: pl.DataFrame,
    test_table: pl.DataFrame | None,
    ranking: pl.DataFrame,
    members: Sequence[RegistryMember],
    reference: str | None = None,
    reports_root: Path | None = None,
    models_root: Path | None = None,
) -> Path:
    """Write ``data/reports/ensemble-<version>.md`` and return the path.

    The bulk is :mod:`nfl.logging.evaluate`'s standard report, built from
    ledger-schema rows so an ensemble reads exactly like a logged season — the
    same slices, the same reliability table, the same definitions. The sections
    appended after it are the ones the ADR asks for specifically and evaluate
    cannot produce: the two not-collapsed slices (evaluate's version needs a
    market price, ADR 0001), the confidence ranking, and the promotion gate.
    """
    from nfl.logging import evaluate as evaluate_mod
    from nfl.logging.backtest import backtest

    version = ensemble.version or registry.utc_version()
    parts = []
    p_val, conf_val = val
    parts.append(
        backtest(
            _Precomputed(p_val, ensemble.name),
            data.val,
            model_name=ensemble.name,
            model_version=f"{version}:val",
            model_kind="ensemble",
            confidence=conf_val,
            confidence_method=ensemble.confidence_mode,
            source="ensemble-backtest",
        )
    )
    p_test, conf_test = test
    if p_test is not None:
        parts.append(
            backtest(
                _Precomputed(p_test, ensemble.name),
                data.test,
                model_name=ensemble.name,
                model_version=f"{version}:test",
                model_kind="ensemble",
                confidence=conf_test,
                confidence_method=ensemble.confidence_mode,
                source="ensemble-backtest",
            )
        )
    frame = pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]
    report = evaluate_mod.build_report(frame, source="ensemble-backtest", with_market=False)

    lines = [evaluate_mod.to_markdown(report).rstrip(), "", "## Ensemble", ""]
    lines += [
        f"- name: `{ensemble.name}` version `{version}`",
        f"- meta-learner: `{ensemble.meta}`"
        + (f" weights `{[round(w, 4) for w in ensemble.weights]}`" if ensemble.weights else ""),
        f"- calibrator: `{getattr(ensemble.calibrator, 'kind', 'none')}` (fit on val)",
        f"- confidence: `{ensemble.confidence_mode}`",
        f"- not-collapsed reference: `{reference or 'own probability'}`",
        "",
        "### Members",
        "",
        _md_table(pl.DataFrame([m.pin for m in members])),
        "",
        f"### Slices (`{NOT_COLLAPSED_SLICE}` on the floor's band, "
        f"`{NOT_COLLAPSED_OWN_SLICE}` on this model's; ADR 0001 market join pending)",
        "",
        "**val**",
        "",
        _md_table(val_table),
        "",
    ]
    if test_table is not None:
        lines += ["**test**", "", _md_table(test_table), ""]
    lines += ["### Confidence definitions ranked on val Q4 rows", "", _md_table(ranking), ""]

    try:
        gate = promotion_gate(ensemble.name, version, root=models_root)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        lines += ["### Promotion gate", "", f"> not evaluated: {exc}", ""]
    else:
        verdict = "**PASS**" if gate.passed else "**FAIL**"
        lines += [
            "### Promotion gate (ADR 0003 §5)",
            "",
            f"{verdict} — candidate `{gate.candidate}/{gate.candidate_version}` "
            f"vs floor `{gate.floor}/{gate.floor_version}`",
            "",
        ]
        lines += [f"> {note}" for note in gate.notes] + ([""] if gate.notes else [])
        lines += [_md_table(gate_table(gate)), ""]

    target = evaluate_mod.report_path(f"ensemble-{version}", root=reports_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# Promotion gate (ADR 0003 §5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateCheck:
    """One ``(slice, metric)`` comparison and its verdict."""

    slice: str
    metric: str
    candidate: float
    floor: float
    #: ``True`` when the candidate must strictly beat the floor, ``False`` when
    #: it only has to be no worse.
    strict: bool
    passed: bool

    @property
    def delta(self) -> float:
        """Candidate minus floor. Negative is an improvement (lower is better)."""
        return self.candidate - self.floor


@dataclass(frozen=True)
class GateResult:
    """The full comparison, whichever way it went."""

    candidate: str
    candidate_version: str
    floor: str
    floor_version: str
    checks: list[GateCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def _val_slices(name: str, version: str, root: Path | None) -> dict[str, dict[str, float]]:
    """``{slice: metrics}`` from an artifact's manifest, keyed by slice name."""
    manifest = registry.read_manifest(registry.model_dir(name, version, root))
    recorded = manifest.extra.get("val_slices")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(
            f"{name}/{version} has no val_slices in its manifest; it was trained before "
            "the slice tables were recorded — retrain it so the gate has numbers to read"
        )
    return {str(row["slice"]): row for row in recorded if isinstance(row, dict) and "slice" in row}


def _reference_pin(name: str, version: str, root: Path | None) -> str | None:
    """The floor pin a manifest's ``not_collapsed`` slice was keyed on, if recorded."""
    manifest = registry.read_manifest(registry.model_dir(name, version, root))
    pin = manifest.extra.get("not_collapsed_reference")
    return str(pin) if pin else None


def promotion_gate(
    candidate: str,
    version: str | None = None,
    *,
    floor: str = NAIVE_FLOOR,
    floor_version: str | None = None,
    root: Path | None = None,
    tolerance: float = GATE_TOLERANCE,
) -> GateResult:
    """Compare a candidate against the naive floor on the val slices.

    ADR 0003 §5: promotion requires beating the floor on val Q4@600/300/120 for
    **both** log-loss and Brier, and being no worse on the not-collapsed subset.
    Both sides are read from the manifests written at training time, so the gate
    compares the numbers that were actually recorded rather than re-deriving
    them from a dataset that may have moved underneath.

    The not-collapsed slice is keyed on the **floor's** probability band on both
    sides, so the comparison is over one row set. The gate proves that rather
    than assuming it: it checks the row counts match and that the candidate
    recorded the same floor pin as the one being compared, and notes it loudly
    when either says otherwise (a manifest written before this rule, or one
    referenced against a floor that has since been retrained).
    """
    candidate_name = _registry_name(candidate, root)
    candidate_version = version or registry.latest(candidate_name, root)
    floor_name = _registry_name(floor, root)
    resolved_floor = floor_version or registry.latest(floor_name, root)

    cand = _val_slices(candidate_name, candidate_version, root)
    base = _val_slices(floor_name, resolved_floor, root)

    checks: list[GateCheck] = []
    notes: list[str] = []
    if candidate_name == floor_name and candidate_version == resolved_floor:
        notes.append("candidate is the floor; every comparison is a tie and therefore a failure")

    # Until the ledger carries Kalshi snapshots (ADR 0003 §6), "not collapsed"
    # stands in for the market filter with the *floor's* probability band. Both
    # sides must therefore be the same rows; row counts are the cheap proof.
    cand_n = cand.get(NOT_COLLAPSED_SLICE, {}).get("n")
    floor_n = base.get(NOT_COLLAPSED_SLICE, {}).get("n")
    if cand_n is not None and floor_n is not None and int(cand_n) != int(floor_n):
        notes.append(
            f"{NOT_COLLAPSED_SLICE} should be the floor's rows on both sides, but the "
            f"candidate recorded {int(cand_n):,} and the floor {int(floor_n):,}; one of "
            "these manifests predates the shared row set — retrain it before trusting "
            "this row of the table (ADR 0003 §5)"
        )
    cand_ref = _reference_pin(candidate_name, candidate_version, root)
    expected_ref = f"{floor_name}/{resolved_floor}"
    if cand_ref is None:
        notes.append(
            f"{candidate_name}/{candidate_version} records no not_collapsed_reference; "
            "it was scored on its own probability band, not the floor's"
        )
    elif cand_ref != expected_ref and candidate_name != floor_name:
        notes.append(
            f"{candidate_name}/{candidate_version} was scored against floor {cand_ref} but "
            f"is being compared to {expected_ref}; re-fit it, or pass --floor-version "
            f"{cand_ref.rpartition('/')[2]}"
        )

    wanted = [(s, True) for s in GATE_Q4_SLICES] + [(NOT_COLLAPSED_SLICE, False)]
    for slice_name, strict in wanted:
        for metric in GATE_METRICS:
            c = cand.get(slice_name, {}).get(metric)
            f = base.get(slice_name, {}).get(metric)
            if c is None or f is None:
                notes.append(
                    f"{slice_name}/{metric}: missing from "
                    + ("candidate" if c is None else "floor")
                    + " manifest — treated as a failure"
                )
                checks.append(
                    GateCheck(
                        slice=slice_name,
                        metric=metric,
                        candidate=float("nan") if c is None else float(c),
                        floor=float("nan") if f is None else float(f),
                        strict=strict,
                        passed=False,
                    )
                )
                continue
            c, f = float(c), float(f)
            passed = c < f if strict else c <= f + tolerance
            checks.append(
                GateCheck(
                    slice=slice_name,
                    metric=metric,
                    candidate=c,
                    floor=f,
                    strict=strict,
                    passed=bool(passed),
                )
            )

    return GateResult(
        candidate=candidate_name,
        candidate_version=candidate_version,
        floor=floor_name,
        floor_version=resolved_floor,
        checks=checks,
        notes=notes,
    )


def gate_table(result: GateResult) -> pl.DataFrame:
    """The comparison as a printable table — one row per check, verdict included."""
    return pl.DataFrame(
        [
            {
                "slice": c.slice,
                "metric": c.metric,
                "candidate": c.candidate,
                "floor": c.floor,
                "delta": c.delta,
                "rule": "must beat" if c.strict else "not worse",
                "verdict": "pass" if c.passed else "FAIL",
            }
            for c in result.checks
        ],
        schema={
            "slice": pl.String,
            "metric": pl.String,
            "candidate": pl.Float64,
            "floor": pl.Float64,
            "delta": pl.Float64,
            "rule": pl.String,
            "verdict": pl.String,
        },
    )


def gate_summary(result: GateResult) -> str:
    """One line saying what happened, for a log or a commit message."""
    failed = [f"{c.slice}/{c.metric}" for c in result.checks if not c.passed]
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    if result.passed:
        return (
            f"{stamp} gate PASS: {result.candidate}/{result.candidate_version} beats "
            f"{result.floor}/{result.floor_version} on all {len(result.checks)} checks"
        )
    return (
        f"{stamp} gate FAIL: {result.candidate}/{result.candidate_version} vs "
        f"{result.floor}/{result.floor_version}; failed {len(failed)}/{len(result.checks)}: "
        + ", ".join(failed)
    )
