"""Test doubles implementing the :class:`~nfl.models.base.WinProbModel` contract.

The ensemble, the calibrators and the registry are model-agnostic by design, so
their tests should not need XGBoost, TensorFlow, or a 770k-row dataset — and
should not break when a sibling model's internals change. These two stand in:

:class:`ConstantModel`
    Always the same probability (optionally the training base rate). The
    degenerate member: contributes no discrimination, so an ensemble that
    improves on it is doing real work.
:class:`NoisyOracle`
    Sees the answer through noise. It reads a column holding the *true*
    probability and perturbs it in logit space, which makes it the only easy way
    to build a synthetic where error is a known function of a knob — exactly
    what the confidence-reliability check needs. It also implements
    ``predict_with_uncertainty``, mirroring the MC-dropout head on the MLP, so
    the ensemble's uncertainty path is exercised without importing TensorFlow.

Both are real implementations of the contract, not mocks: they fit, predict,
save and load, so a registry round-trip through them exercises the same code
path a GBDT artifact will.
"""

# `X` is the feature-matrix argument name fixed by the WinProbModel contract.
# ruff: noqa: N803

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.models.base import FitResult
from nfl.models.calibration import logit, sigmoid
from nfl.models.registry import utc_version

__all__ = ["ConstantModel", "NoisyOracle"]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Persist to ``path``, adding ``.json`` when the caller passed a bare stem."""
    target = Path(path)
    if target.suffix != ".json":
        target = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def _read_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() and target.with_suffix(".json").is_file():
        target = target.with_suffix(".json")
    return dict(json.loads(target.read_text()))


@dataclass
class ConstantModel:
    """Predicts one fixed probability for every row.

    ``p=None`` fits the training base rate, which is the strongest thing a
    model with no features can honestly say — and the log-loss every real model
    has to beat before anyone should care about it.
    """

    p: float | None = 0.5
    name: str = "constant"
    version: str = ""
    feature_names: list[str] = field(default_factory=list)

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        labels = np.asarray(y, dtype=float).ravel()
        if self.p is None:
            self.p = float(labels.mean()) if labels.size else 0.5
        self.feature_names = list(X.columns)
        self.version = self.version or utc_version()
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(labels.size),
            n_val=0 if y_val is None else int(np.asarray(y_val).size),
            metrics={"constant_p": float(self.p)},
            feature_names=self.feature_names,
        )

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return np.full(X.height, float(self.p if self.p is not None else 0.5))

    def save(self, path: Path) -> Path:
        return _write_json(
            Path(path),
            {
                "kind": "constant",
                "p": self.p,
                "name": self.name,
                "version": self.version,
                "feature_names": self.feature_names,
            },
        )

    @classmethod
    def load(cls, path: Path) -> ConstantModel:
        state = _read_json(Path(path))
        return cls(
            p=None if state.get("p") is None else float(state["p"]),
            name=str(state.get("name", "constant")),
            version=str(state.get("version", "")),
            feature_names=list(state.get("feature_names", [])),
        )


@dataclass
class NoisyOracle:
    """Reads the true probability from a column and blurs it in logit space.

    ``sigma`` sets the blur; ``scale_col``, when given, multiplies it per row,
    which is how a test builds data where some rows are genuinely harder than
    others. Members seeded differently disagree exactly where sigma is large,
    so member disagreement and realised error move together by construction —
    the property :func:`nfl.models.ensemble.confidence_reliability` measures.

    Noise is drawn from a fresh generator seeded with ``seed`` on every call, so
    predictions are reproducible for a given frame but are *not* stable under
    row subsetting. Fine for tests; this is not a model.
    """

    sigma: float = 1.0
    signal_col: str = "oracle_signal"
    scale_col: str | None = None
    seed: int = 0
    mc_samples: int = 32
    name: str = "noisy_oracle"
    version: str = ""
    feature_names: list[str] = field(default_factory=list)

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        if self.signal_col not in X.columns:
            raise KeyError(f"NoisyOracle needs a {self.signal_col!r} column; got {X.columns}")
        self.feature_names = list(X.columns)
        self.version = self.version or utc_version()
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(np.asarray(y).size),
            n_val=0 if y_val is None else int(np.asarray(y_val).size),
            metrics={"sigma": float(self.sigma)},
            feature_names=self.feature_names,
        )

    def _signal_and_scale(self, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        signal = X.get_column(self.signal_col).to_numpy().astype(float)
        if self.scale_col is None:
            scale = np.ones_like(signal)
        else:
            scale = X.get_column(self.scale_col).to_numpy().astype(float)
        return logit(signal), np.abs(scale)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        z, scale = self._signal_and_scale(X)
        rng = np.random.default_rng(self.seed)
        return sigmoid(z + self.sigma * scale * rng.standard_normal(z.size))

    def predict_with_uncertainty(self, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """MC-style ``(mean, std)`` over ``mc_samples`` perturbations.

        Same shape of return value as the MLP's dropout head, so the ensemble
        can consume either without knowing which it has.
        """
        z, scale = self._signal_and_scale(X)
        rng = np.random.default_rng(self.seed + 1)
        draws = sigmoid(
            z[None, :]
            + self.sigma * scale[None, :] * rng.standard_normal((self.mc_samples, z.size))
        )
        return draws.mean(axis=0), draws.std(axis=0)

    def save(self, path: Path) -> Path:
        return _write_json(
            Path(path),
            {
                "kind": "noisy_oracle",
                "sigma": self.sigma,
                "signal_col": self.signal_col,
                "scale_col": self.scale_col,
                "seed": self.seed,
                "mc_samples": self.mc_samples,
                "name": self.name,
                "version": self.version,
                "feature_names": self.feature_names,
            },
        )

    @classmethod
    def load(cls, path: Path) -> NoisyOracle:
        state = _read_json(Path(path))
        return cls(
            sigma=float(state.get("sigma", 1.0)),
            signal_col=str(state.get("signal_col", "oracle_signal")),
            scale_col=state.get("scale_col"),
            seed=int(state.get("seed", 0)),
            mc_samples=int(state.get("mc_samples", 32)),
            name=str(state.get("name", "noisy_oracle")),
            version=str(state.get("version", "")),
            feature_names=list(state.get("feature_names", [])),
        )
