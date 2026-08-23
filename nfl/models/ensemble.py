"""Ensembling and confidence for win-probability models.

Two jobs, both model-agnostic — the ensemble talks only to the
:class:`~nfl.models.base.WinProbModel` contract, so it wraps the logistic
baseline, the GBDT, the MLP and the sequence model without knowing which is
which.

**Combining.** :class:`StackedEnsemble` blends member probabilities three ways:
``mean`` (arithmetic, the conservative choice), ``logit_mean`` (geometric in
odds space — the default, because averaging probabilities near 0 and 1 drags
confident-and-correct members toward the middle) and ``logistic_stacker`` (a
logistic regression *on member logits*, fit on the validation split, which lets
a member that is systematically better carry more weight).

**Confidence.** The project needs more than ``P(home)``: it needs to know when
to keep its money in its pocket. The signal used here is *member disagreement* —
the standard deviation of member predictions in logit space. When four models
with different inductive biases all say 0.72, that is a different object than
when they say 0.45, 0.60, 0.85, 0.95 and average to 0.72. A member exposing
``predict_with_uncertainty`` (the MLP's MC-dropout head) contributes a second
signal: its own predictive standard deviation.

Neither definition is *a priori* correct, so confidence is not asserted — it is
measured. :func:`confidence_reliability` bins predictions by confidence and
reports Brier per bin; a useful confidence score is one where Brier falls as
confidence rises. :func:`compare_confidence_definitions` runs that check over
several candidate definitions at once and ranks them, which is how the
definition gets chosen: empirically, on validation data.
"""

# `X` is the feature-matrix argument name fixed by the WinProbModel contract.
# ruff: noqa: N803, N806

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from nfl.models.base import FitResult
from nfl.models.calibration import (
    Calibrator,
    expected_calibration_error,
    fit_logistic,
    load_calibrator,
    logit,
    sigmoid,
)
from nfl.models.registry import resolve_artifact, utc_version

__all__ = [
    "CLOCK_CANDIDATES",
    "CONFIDENCE_MODES",
    "ConfidenceMode",
    "MetaKind",
    "StackedEnsemble",
    "compare_confidence_definitions",
    "confidence_reliability",
]

MetaKind = Literal["mean", "logit_mean", "logistic_stacker"]

#: How a row's confidence is derived. ``disagreement`` is member spread;
#: ``uncertainty`` is the members' MC-dropout σ; ``margin`` is the calibrated
#: probability's own distance from the coin flip; the two compound modes average
#: a spread term with σ (``combined``) or the margin with σ
#: (``margin_uncertainty``). ADR 0003 §4 says which one ships is decided by
#: :func:`compare_confidence_definitions` on val Q4 rows, not by argument.
ConfidenceMode = Literal["disagreement", "uncertainty", "combined", "margin", "margin_uncertainty"]

#: Every value :data:`ConfidenceMode` allows, for validation and iteration.
CONFIDENCE_MODES: tuple[str, ...] = (
    "disagreement",
    "uncertainty",
    "combined",
    "margin",
    "margin_uncertainty",
)

#: Modes that need the MC-dropout σ of at least one member.
_NEEDS_UNCERTAINTY: frozenset[str] = frozenset({"uncertainty", "combined", "margin_uncertainty"})

#: Modes that need the calibrated probability of the row being scored.
_NEEDS_PROBABILITY: frozenset[str] = frozenset({"margin", "margin_uncertainty"})

#: Where to look for the game clock when a phase-aware calibrator needs it.
#: A feature frame carries the prefixed builder output; a raw play-by-play frame
#: carries the bare name; an assembled frame carries the ``meta__`` copy.
CLOCK_CANDIDATES: tuple[str, ...] = (
    "ig__game_seconds_remaining",
    "bi__game_seconds_remaining",
    "game_seconds_remaining",
    "meta__game_seconds_remaining",
)

MANIFEST_NAME = "manifest.json"
CALIBRATOR_NAME = "calibrator.json"


# --------------------------------------------------------------------------- #
# Small metric helpers
#
# Deliberately local: the ensemble reports a handful of numbers about its own
# fit, and doing so must not couple it to the evaluation module.
# --------------------------------------------------------------------------- #


def _clip(p: Any) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)


def _log_loss(p: Any, y: np.ndarray) -> float:
    q = _clip(p)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def _brier(p: Any, y: np.ndarray) -> float:
    return float(np.mean((_clip(p) - y) ** 2))


def _accuracy(p: Any, y: np.ndarray) -> float:
    return float(np.mean((_clip(p) >= 0.5) == (y >= 0.5)))


def _auc(p: Any, y: np.ndarray) -> float:
    """Rank-based AUC. Tied probabilities share a mid-rank; tied labels are dropped."""
    q = _clip(p)
    hard = np.asarray(y, dtype=float).ravel()
    pos, neg = hard > 0.5, hard < 0.5
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(q, kind="mergesort")
    ranks = np.empty(q.size, dtype=float)
    ranks[order] = np.arange(1, q.size + 1, dtype=float)
    uniq, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inverse.ravel(), ranks)
    ranks = (sums / counts)[inverse.ravel()]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=float)
    out[order] = np.arange(1, values.size + 1, dtype=float)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, without pulling in scipy."""
    if a.size < 2:
        return float("nan")
    ra, rb = _ranks(np.asarray(a, dtype=float)), _ranks(np.asarray(b, dtype=float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(float((ra**2).sum()) * float((rb**2).sum())))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Confidence diagnostics
# --------------------------------------------------------------------------- #


def confidence_reliability(
    p: Any,
    conf: Any,
    y: Any,
    *,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pl.DataFrame:
    """Bin by confidence, report error per bin. The test a confidence score must pass.

    A confidence score is only worth publishing if it *predicts error*: the
    low-confidence bins should carry visibly worse Brier and log-loss than the
    high-confidence bins. Anything else is a number that looks reassuring and
    means nothing.

    ``strategy="quantile"`` (default) uses equal-count bins, which keeps the
    comparison honest when confidence piles up at one end — the usual outcome
    when members mostly agree. ``"uniform"`` uses equal-width bins over the
    observed range.

    Returns one row per non-empty bin — bounds, count, mean confidence, and
    ``brier`` / ``log_loss`` / ``accuracy`` / ``ece`` within the bin — ordered
    from least to most confident.
    """
    prob = _clip(p)
    confidence = np.asarray(conf, dtype=float).ravel()
    lab = np.clip(np.asarray(y, dtype=float).ravel(), 0.0, 1.0)
    if not (prob.size == confidence.size == lab.size):
        raise ValueError(f"length mismatch: p={prob.size}, conf={confidence.size}, y={lab.size}")
    if prob.size == 0:
        raise ValueError("cannot bin zero rows")

    if strategy == "quantile":
        edges = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        lo, hi = float(confidence.min()), float(confidence.max())
        edges = np.unique(np.linspace(lo, hi if hi > lo else lo + 1e-9, n_bins + 1))
    if edges.size < 2:  # every row carries the same confidence
        edges = np.array([float(confidence.min()), float(confidence.min()) + 1e-9])
    idx = np.clip(np.searchsorted(edges[1:-1], confidence, side="right"), 0, edges.size - 2)

    rows: list[dict[str, Any]] = []
    for b in range(edges.size - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": b,
                "conf_lo": float(edges[b]),
                "conf_hi": float(edges[b + 1]),
                "n": n,
                "conf_mean": float(confidence[mask].mean()),
                "brier": _brier(prob[mask], lab[mask]),
                "log_loss": _log_loss(prob[mask], lab[mask]),
                "accuracy": _accuracy(prob[mask], lab[mask]),
                "ece": expected_calibration_error(prob[mask], lab[mask]),
                "mean_pred": float(prob[mask].mean()),
                "mean_actual": float(lab[mask].mean()),
            }
        )
    return pl.DataFrame(rows).sort("conf_mean")


def compare_confidence_definitions(
    p: Any,
    y: Any,
    candidates: dict[str, Any],
    *,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pl.DataFrame:
    """Rank candidate confidence definitions by how well they predict error.

    ``spearman`` is the rank correlation between bin confidence and bin Brier;
    it should be **negative** — more confidence, less error. ``brier_spread`` is
    the least-confident bin's Brier minus the most-confident bin's, so bigger is
    better: it is the separation you would actually trade on. Sorted best-first.
    """
    rows: list[dict[str, Any]] = []
    for label, conf in candidates.items():
        table = confidence_reliability(p, conf, y, n_bins=n_bins, strategy=strategy)
        briers = table.get_column("brier").to_numpy()
        confs = table.get_column("conf_mean").to_numpy()
        rows.append(
            {
                "confidence": label,
                "n_bins": int(table.height),
                "spearman": _spearman(confs, briers),
                "brier_spread": float(briers[0] - briers[-1]) if table.height > 1 else 0.0,
                "brier_low_conf": float(briers[0]),
                "brier_high_conf": float(briers[-1]),
                "monotone": bool(table.height > 1 and bool(np.all(np.diff(briers) <= 0))),
            }
        )
    return pl.DataFrame(rows).sort(["spearman", "brier_spread"], descending=[False, True])


# --------------------------------------------------------------------------- #
# The ensemble
# --------------------------------------------------------------------------- #


@dataclass
class StackedEnsemble:
    """Blend several :class:`~nfl.models.base.WinProbModel` members into one.

    Members are normally *already trained*: the ensemble is fit on the
    validation split, where members are out-of-sample and their relative
    strengths are visible. Set ``fit_members=True`` only when you want the
    ensemble to train them itself, and only on data they have not seen.

    ``calibrator`` is applied last, to the combined probability, and is fit on
    the same validation predictions. A
    :class:`~nfl.models.calibration.PhaseAwareCalibrator` also needs the game
    clock; it comes from ``clock_column`` (auto-detected from
    :data:`CLOCK_CANDIDATES` when unset) or is passed explicitly to
    :meth:`fit` and :meth:`predict_proba`.
    """

    members: list[Any]
    meta: MetaKind = "logit_mean"
    calibrator: Calibrator | None = None
    name: str = "stacked_ensemble"
    confidence_mode: ConfidenceMode = "disagreement"
    #: Member spread, in logits, that drives confidence to zero. 2.0 is roughly
    #: "members straddle 0.27 against 0.73" — disagreement worth respecting.
    disagreement_scale: float = 2.0
    #: Probability-space MC standard deviation that drives confidence to zero.
    uncertainty_scale: float = 0.15
    weights: list[float] | None = None
    fit_members: bool = False
    clock_column: str | None = None
    version: str = ""
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("StackedEnsemble needs at least one member")
        if self.meta not in ("mean", "logit_mean", "logistic_stacker"):
            raise ValueError(f"unknown meta {self.meta!r}")
        if self.confidence_mode not in CONFIDENCE_MODES:
            raise ValueError(f"unknown confidence_mode {self.confidence_mode!r}")

    # -- introspection ---------------------------------------------------- #

    @property
    def member_names(self) -> list[str]:
        """Each member's ``name``, falling back to its class name."""
        return [str(getattr(m, "name", type(m).__name__)) for m in self.members]

    @property
    def n_members(self) -> int:
        return len(self.members)

    def _uncertainty_members(self) -> list[Any]:
        return [m for m in self.members if hasattr(m, "predict_with_uncertainty")]

    def _resolve_clock(self, X: pl.DataFrame, seconds_remaining: Any | None) -> np.ndarray | None:
        if seconds_remaining is not None:
            return np.asarray(seconds_remaining, dtype=float).ravel()
        if self.clock_column is not None:
            if self.clock_column not in X.columns:
                raise KeyError(f"clock_column {self.clock_column!r} not in frame")
            return X.get_column(self.clock_column).to_numpy().astype(float)
        for candidate in CLOCK_CANDIDATES:
            if candidate in X.columns:
                return X.get_column(candidate).to_numpy().astype(float)
        return None

    def _needs_clock(self) -> bool:
        return getattr(self.calibrator, "kind", "") == "phase_aware"

    # -- prediction ------------------------------------------------------- #

    def member_probabilities(self, X: pl.DataFrame) -> np.ndarray:
        """``(n_rows, n_members)`` matrix of member probabilities."""
        cols = [np.asarray(m.predict_proba(X), dtype=float).ravel() for m in self.members]
        heights = {c.size for c in cols}
        if len(heights) != 1:
            raise ValueError(f"members disagree on row count: {sorted(heights)}")
        return np.column_stack(cols)

    def member_uncertainty(self, X: pl.DataFrame) -> np.ndarray | None:
        """Per-member predictive std, or ``None`` when no member reports one.

        Only members exposing ``predict_with_uncertainty`` contribute a column.
        The point estimate always comes from ``predict_proba``, so
        :meth:`predict_proba` and :meth:`predict_with_confidence` can never
        disagree about the probability itself.
        """
        cols = []
        for member in self._uncertainty_members():
            _, std = member.predict_with_uncertainty(X)
            cols.append(np.asarray(std, dtype=float).ravel())
        return np.column_stack(cols) if cols else None

    def combine(self, member_probs: np.ndarray) -> np.ndarray:
        """Blend the member matrix into one probability per row (uncalibrated)."""
        probs = np.atleast_2d(np.asarray(member_probs, dtype=float))
        z = logit(probs.ravel()).reshape(probs.shape)
        if self.meta == "logistic_stacker":
            if self.weights is None:
                raise RuntimeError("logistic_stacker has no weights; call fit() first")
            beta = np.asarray(self.weights, dtype=float)
            if beta.size != probs.shape[1] + 1:
                raise ValueError(
                    f"stacker holds {beta.size - 1} weights but got {probs.shape[1]} members"
                )
            return sigmoid(beta[0] + z @ beta[1:])
        weights = None
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.size != probs.shape[1]:
                raise ValueError(f"{weights.size} weights for {probs.shape[1]} members")
            total = float(weights.sum())
            if total <= 0:
                raise ValueError("member weights must sum to a positive number")
            weights = weights / total
        if self.meta == "mean":
            return np.average(probs, axis=1, weights=weights)
        return sigmoid(np.average(z, axis=1, weights=weights))

    def _calibrate(self, raw: np.ndarray, clock: np.ndarray | None) -> np.ndarray:
        if self.calibrator is None:
            return raw
        if self._needs_clock():
            if clock is None:
                raise ValueError(
                    "phase-aware calibration needs the game clock; set clock_column "
                    "or pass seconds_remaining"
                )
            return self.calibrator.transform(raw, seconds_remaining=clock)
        return self.calibrator.transform(raw)

    def predict_proba(self, X: pl.DataFrame, *, seconds_remaining: Any | None = None) -> np.ndarray:
        """``P(home win)`` per row: members combined, then calibrated."""
        raw = self.combine(self.member_probabilities(X))
        return self._calibrate(raw, self._resolve_clock(X, seconds_remaining))

    def needs_uncertainty(self, mode: str | None = None) -> bool:
        """Whether ``mode`` (default: the configured one) needs member σ."""
        return (mode or self.confidence_mode) in _NEEDS_UNCERTAINTY

    def confidence_from(
        self,
        member_probs: np.ndarray,
        uncertainty: np.ndarray | None = None,
        p: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Confidence in ``[0, 1]`` plus the raw signals behind it.

        Three signals, each normalised to a "penalty" that confidence is one
        minus:

        ``logit_std``
            Population standard deviation of the member logits — how much the
            members disagree. Divided by :attr:`disagreement_scale`.
        ``mc_std``
            Mean per-member MC-dropout standard deviation in probability space —
            how much a member disagrees with itself. Divided by
            :attr:`uncertainty_scale`.
        ``margin``
            ``2 * |p - 0.5|`` on the **calibrated** probability — how far from a
            coin flip the ensemble's own answer is. Already in ``[0, 1]``, so
            its penalty is ``1 - margin``.

        ``p`` is the calibrated probability and is required by the ``margin``
        modes; the other modes ignore it. A single member cannot disagree with
        itself, so disagreement confidence over one member is identically 1 —
        honest about the formula, useless as a signal. Use at least two members,
        or a mode that does not rest on spread.
        """
        probs = np.asarray(member_probs, dtype=float)
        z = logit(probs.ravel()).reshape(probs.shape)
        spread = z.std(axis=1) if probs.shape[1] > 1 else np.zeros(probs.shape[0])
        signals: dict[str, np.ndarray] = {"logit_std": spread}

        mc = None
        if uncertainty is not None and np.asarray(uncertainty).size:
            mc = np.asarray(uncertainty, dtype=float).mean(axis=1)
            signals["mc_std"] = mc

        margin = None
        if p is not None:
            margin = np.clip(2.0 * np.abs(np.asarray(p, dtype=float).ravel() - 0.5), 0.0, 1.0)
            signals["margin"] = margin

        mode = self.confidence_mode
        if mode in _NEEDS_UNCERTAINTY and mc is None:
            if mode == "uncertainty":
                raise ValueError(
                    "confidence_mode='uncertainty' but no member exposes predict_with_uncertainty"
                )
            # The compound modes degrade to their other half rather than fail:
            # an ensemble whose GRU was swapped for one without an MC head still
            # returns a confidence, just a less informed one.
            mode = "margin" if mode == "margin_uncertainty" else "disagreement"
        if mode in _NEEDS_PROBABILITY and margin is None:
            raise ValueError(
                f"confidence_mode={self.confidence_mode!r} needs the calibrated probability; "
                "pass p= (predict_with_confidence does this for you)"
            )

        dis_term = spread / max(self.disagreement_scale, 1e-9)
        mc_term = None if mc is None else np.asarray(mc) / max(self.uncertainty_scale, 1e-9)
        if mode == "disagreement":
            penalty = dis_term
        elif mode == "uncertainty":
            penalty = cast("np.ndarray", mc_term)
        elif mode == "combined":
            penalty = 0.5 * (dis_term + cast("np.ndarray", mc_term))
        elif mode == "margin":
            penalty = 1.0 - cast("np.ndarray", margin)
        else:  # margin_uncertainty
            penalty = 0.5 * ((1.0 - cast("np.ndarray", margin)) + cast("np.ndarray", mc_term))
        confidence = 1.0 - np.clip(penalty, 0.0, 1.0)
        signals["confidence"] = confidence
        return confidence, signals

    def predict_with_confidence(
        self, X: pl.DataFrame, *, seconds_remaining: Any | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """``(p, confidence, components)`` — the ensemble's full output.

        ``components`` carries every intermediate a downstream trader (or a
        debugging session) needs: each member's probability under its own name,
        the uncalibrated blend, the spread signals, and the confidence.
        """
        probs = self.member_probabilities(X)
        uncertainty = self.member_uncertainty(X) if self.needs_uncertainty() else None
        raw = self.combine(probs)
        calibrated = self._calibrate(raw, self._resolve_clock(X, seconds_remaining))
        confidence, signals = self.confidence_from(probs, uncertainty, calibrated)
        components: dict[str, np.ndarray] = {
            f"member__{n}": probs[:, i] for i, n in enumerate(self.member_names)
        }
        components["raw"] = raw
        components["calibrated"] = calibrated
        components.update(signals)
        return calibrated, confidence, components

    # -- fitting ---------------------------------------------------------- #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        *,
        seconds_remaining: Any | None = None,
        val_seconds_remaining: Any | None = None,
    ) -> FitResult:
        """Fit the meta-learner and calibrator on the validation split.

        Falls back to the training split when no validation frame is given, and
        records that in ``extra["meta_fit_on"]``: stacker weights fit on member
        training predictions are optimistic and must not be read as a ranking of
        the members.
        """
        y_train = np.clip(np.asarray(y, dtype=float).ravel(), 0.0, 1.0)
        if self.fit_members:
            for member in self.members:
                member.fit(X, y_train, X_val, y_val)

        on_val = X_val is not None and y_val is not None
        X_meta = X_val if on_val else X
        assert X_meta is not None
        y_meta = np.clip(np.asarray(y_val if on_val else y, dtype=float).ravel(), 0.0, 1.0)
        clock = self._resolve_clock(X_meta, val_seconds_remaining if on_val else seconds_remaining)

        probs = self.member_probabilities(X_meta)
        if self.meta == "logistic_stacker":
            beta = fit_logistic(logit(probs.ravel()).reshape(probs.shape), y_meta)
            self.weights = [float(w) for w in beta]
        raw = self.combine(probs)

        if self.calibrator is not None:
            if self._needs_clock():
                if clock is None:
                    raise ValueError(
                        "phase-aware calibration needs the game clock; set clock_column "
                        "or pass seconds_remaining"
                    )
                self.calibrator.fit(raw, y_meta, seconds_remaining=clock)
            else:
                self.calibrator.fit(raw, y_meta)
        combined = self._calibrate(raw, clock)

        self.feature_names = list(X.columns)
        self.version = self.version or utc_version()
        metrics = {
            "log_loss": _log_loss(combined, y_meta),
            "brier": _brier(combined, y_meta),
            "accuracy": _accuracy(combined, y_meta),
            "ece": expected_calibration_error(combined, y_meta),
            "auc": _auc(combined, y_meta),
        }
        for i, member_name in enumerate(self.member_names):
            metrics[f"member_log_loss__{member_name}"] = _log_loss(probs[:, i], y_meta)
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(y_train.size),
            n_val=int(y_meta.size) if on_val else 0,
            metrics=metrics,
            feature_names=self.feature_names,
            extra={
                "meta": self.meta,
                "members": self.member_names,
                "weights": self.weights,
                "calibrator": getattr(self.calibrator, "kind", None),
                "confidence_mode": self.confidence_mode,
                "meta_fit_on": "val" if on_val else "train",
            },
        )

    # -- persistence ------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        """Write a directory holding the manifest, the calibrator and every member."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        members: list[dict[str, str]] = []
        for i, member in enumerate(self.members):
            member_dir = target / "members" / str(i)
            member_dir.mkdir(parents=True, exist_ok=True)
            member.save(member_dir / "model")
            klass = type(member)
            members.append(
                {
                    "name": self.member_names[i],
                    "class": f"{klass.__module__}:{klass.__qualname__}",
                    "artifact": resolve_artifact(member_dir, "model").name,
                }
            )
        calibrator_file = None
        if self.calibrator is not None:
            self.calibrator.save(target / CALIBRATOR_NAME)
            calibrator_file = CALIBRATOR_NAME
        manifest = {
            "name": self.name,
            "version": self.version,
            "meta": self.meta,
            "confidence_mode": self.confidence_mode,
            "disagreement_scale": self.disagreement_scale,
            "uncertainty_scale": self.uncertainty_scale,
            "weights": self.weights,
            "clock_column": self.clock_column,
            "feature_names": self.feature_names,
            "calibrator": calibrator_file,
            "members": members,
        }
        (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return target

    @classmethod
    def load(cls, path: Path) -> StackedEnsemble:
        """Rebuild an ensemble, importing each member's class on demand."""
        import importlib

        target = Path(path)
        manifest = json.loads((target / MANIFEST_NAME).read_text())
        members = []
        for i, spec in enumerate(manifest["members"]):
            module_name, _, qualname = str(spec["class"]).partition(":")
            obj: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                obj = getattr(obj, part)
            members.append(obj.load(target / "members" / str(i) / spec["artifact"]))
        calibrator = None
        if manifest.get("calibrator"):
            calibrator = load_calibrator(target / str(manifest["calibrator"]))
        return cls(
            members=members,
            meta=manifest.get("meta", "logit_mean"),
            calibrator=calibrator,
            name=manifest.get("name", "stacked_ensemble"),
            confidence_mode=manifest.get("confidence_mode", "disagreement"),
            disagreement_scale=float(manifest.get("disagreement_scale", 2.0)),
            uncertainty_scale=float(manifest.get("uncertainty_scale", 0.15)),
            weights=manifest.get("weights"),
            clock_column=manifest.get("clock_column"),
            version=manifest.get("version", ""),
            feature_names=list(manifest.get("feature_names", [])),
        )
