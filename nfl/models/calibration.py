"""Probability calibration for win-probability models.

A model that ranks well is not automatically *honest*: an XGBoost or MLP head
can be sharply discriminative and still report 0.90 on buckets that only win
82% of the time. Calibration is the post-hoc fix — a monotone map applied to
validation-set predictions and then frozen — and it is what makes the numbers
downstream (Kalshi position sizing) mean what they say.

Four maps live here, all with the same tiny interface
(:meth:`Calibrator.fit` / :meth:`Calibrator.transform` / ``save`` / ``load``):

``identity``
    Does nothing. The honest baseline to beat.
``platt``
    ``sigmoid(a * logit(p) + b)`` — two parameters, fits both over-confidence
    and a systematic home/away bias. Robust on small validation sets.
``isotonic``
    Free-form monotone step fit (pool-adjacent-violators). Strictly more
    expressive than Platt, needs more data, can overfit thin tails.
``temperature``
    ``sigmoid(logit(p) / T)`` — one parameter, pure sharpness correction, no
    bias term. The standard choice for neural nets.

And one wrapper that matters for this project specifically:

``phase_aware``
    Late-game calibration is *not* the same problem as first-quarter
    calibration. With four minutes left a two-score lead is nearly decided and
    models tend to be under-confident; in the first quarter every prediction
    lives near 0.5. :class:`PhaseAwareCalibrator` bins rows by
    ``game_seconds_remaining`` and fits an independent sub-calibrator per
    bucket, falling back to a global fit for buckets too thin to trust.

Calibrators are model-agnostic: they see probabilities and labels, never
features, so they compose with any :class:`~nfl.models.base.WinProbModel`.
Persistence is ``.json`` (human-readable) or ``.npz`` (compact, for isotonic
knot arrays); both round-trip through :func:`load_calibrator`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "CALIBRATORS",
    "DEFAULT_PHASE_EDGES",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PhaseAwareCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "expected_calibration_error",
    "fit_logistic",
    "load_calibrator",
    "logit",
    "make_calibrator",
    "sigmoid",
]

#: Probabilities are clipped this far from {0, 1} before any logit.
EPS = 1e-6

#: Bucket boundaries (seconds remaining) for :class:`PhaseAwareCalibrator`.
#: Five buckets: final two minutes, rest of the two-minute drill, mid Q4,
#: early Q4, and everything before the fourth quarter.
DEFAULT_PHASE_EDGES: tuple[float, ...] = (120.0, 300.0, 600.0, 900.0)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #


def _as_prob(p: Any) -> np.ndarray:
    """Flatten to a float array clipped into ``(0, 1)`` so logits stay finite."""
    arr = np.asarray(p, dtype=float).ravel()
    return np.clip(arr, EPS, 1.0 - EPS)


def _as_label(y: Any) -> np.ndarray:
    """Labels as floats in ``[0, 1]``. Ties are 0.5, per the model contract."""
    return np.clip(np.asarray(y, dtype=float).ravel(), 0.0, 1.0)


def logit(p: Any) -> np.ndarray:
    """``log(p / (1 - p))``, clipped so ``p in {0, 1}`` does not blow up."""
    q = _as_prob(p)
    return np.log(q) - np.log1p(-q)


def sigmoid(z: Any) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(z, dtype=float)
    out = np.empty(x.shape, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def expected_calibration_error(p: Any, y: Any, n_bins: int = 10) -> float:
    """Expected calibration error: bin-count-weighted ``|mean(p) - mean(y)|``.

    Equal-width bins over ``[0, 1]`` (10 by default, matching the project's
    metric convention). Empty bins contribute nothing.
    """
    prob = _as_prob(p)
    lab = _as_label(y)
    if prob.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        total += n * abs(float(prob[mask].mean()) - float(lab[mask].mean()))
    return total / prob.size


def fit_logistic(
    features: Any,
    y: Any,
    *,
    intercept: bool = True,
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fit logistic regression by IRLS; return coefficients.

    Shared by :class:`PlattCalibrator`, :class:`TemperatureCalibrator` and the
    ensemble's logistic stacker, so all three agree on conventions: soft labels
    (0.5 for ties) are supported, a small ridge keeps separable data from
    running off to infinity, and the intercept — when requested — is the *first*
    returned coefficient.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    lab = _as_label(y)
    if x.shape[0] != lab.size:
        raise ValueError(f"design matrix has {x.shape[0]} rows, labels have {lab.size}")
    if intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])
    k = x.shape[1]
    beta = np.zeros(k)
    eye = np.eye(k)
    for _ in range(max_iter):
        mu = sigmoid(x @ beta)
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        grad = x.T @ (lab - mu) - ridge * beta
        hess = x.T @ (w[:, None] * x) + ridge * eye
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    return beta


def _isotonic_knots(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit, thinned to interpolation knots.

    Duplicate probabilities are collapsed to a weighted mean first (otherwise
    the step function is ambiguous at ties), and runs of equal fitted values are
    reduced to their endpoints — dropping collinear interior points, which
    leaves ``np.interp`` exact while keeping the artifact small.
    """
    from sklearn.isotonic import isotonic_regression

    order = np.argsort(p, kind="mergesort")
    xs, ys = p[order], y[order]
    uniq, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inverse.ravel(), ys)
    means = sums / counts
    fitted = np.asarray(
        isotonic_regression(means, sample_weight=counts.astype(float), increasing=True),
        dtype=float,
    )
    if uniq.size <= 2:
        return uniq, fitted
    keep = np.ones(uniq.size, dtype=bool)
    interior = np.arange(1, uniq.size - 1)
    flat = (fitted[interior] == fitted[interior - 1]) & (fitted[interior] == fitted[interior + 1])
    keep[interior] = ~flat
    return uniq[keep], fitted[keep]


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_to_json(state: dict[str, Any]) -> str:
    return json.dumps({k: _jsonable(v) for k, v in state.items()}, indent=2, sort_keys=True)


def _read_state(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            return {k: handle[k] for k in handle.files}
    return dict(json.loads(path.read_text()))


def _as_str(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item() if value.ndim == 0 else value.ravel()[0])
    return str(value)


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _as_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).ravel()


def _prefixed(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in state.items()}


def _unprefixed(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #

#: ``kind`` -> class, populated by ``__init_subclass__``.
CALIBRATORS: dict[str, type[Calibrator]] = {}


class Calibrator(ABC):
    """A frozen monotone map from raw probability to calibrated probability.

    Fit on *validation* predictions, never on training predictions: a model's
    training-set probabilities are already over-fit, so a calibrator fit there
    learns the wrong correction.
    """

    kind: ClassVar[str] = "base"

    def __init__(self) -> None:
        self.fitted: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "kind", "base") != "base":
            CALIBRATORS[cls.kind] = cls

    # -- fitting ---------------------------------------------------------- #

    @abstractmethod
    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None: ...

    @abstractmethod
    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray: ...

    def fit(self, p: Any, y: Any, *, seconds_remaining: Any | None = None) -> Calibrator:
        """Fit on validation predictions ``p`` against outcomes ``y``; return self."""
        prob = _as_prob(p)
        lab = _as_label(y)
        if prob.size != lab.size:
            raise ValueError(f"p has {prob.size} rows, y has {lab.size}")
        if prob.size == 0:
            raise ValueError("cannot fit a calibrator on zero rows")
        secs = None if seconds_remaining is None else np.asarray(seconds_remaining, float).ravel()
        if secs is not None and secs.size != prob.size:
            raise ValueError(f"p has {prob.size} rows, seconds_remaining has {secs.size}")
        self._fit(prob, lab, secs)
        self.fitted = True
        return self

    def transform(self, p: Any, *, seconds_remaining: Any | None = None) -> np.ndarray:
        """Map raw probabilities through the fitted calibration."""
        if not self.fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")
        prob = _as_prob(p)
        secs = None if seconds_remaining is None else np.asarray(seconds_remaining, float).ravel()
        if secs is not None and secs.size != prob.size:
            raise ValueError(f"p has {prob.size} rows, seconds_remaining has {secs.size}")
        return np.clip(self._transform(prob, secs), EPS, 1.0 - EPS)

    def fit_transform(self, p: Any, y: Any, *, seconds_remaining: Any | None = None) -> np.ndarray:
        """Convenience for in-sample inspection. Never report metrics off this."""
        self.fit(p, y, seconds_remaining=seconds_remaining)
        return self.transform(p, seconds_remaining=seconds_remaining)

    def __call__(self, p: Any, *, seconds_remaining: Any | None = None) -> np.ndarray:
        return self.transform(p, seconds_remaining=seconds_remaining)

    # -- persistence ------------------------------------------------------ #

    @abstractmethod
    def _state(self) -> dict[str, Any]:
        """Flat, array/scalar-valued parameter dict. Must include ``kind``."""

    @classmethod
    @abstractmethod
    def _from_state(cls, state: dict[str, Any]) -> Calibrator: ...

    def save(self, path: Path | str) -> Path:
        """Write parameters to ``.json`` (default) or ``.npz`` (by suffix)."""
        target = Path(path)
        if not self.fitted:
            raise RuntimeError("refusing to save an unfitted calibrator")
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self._state()
        if target.suffix == ".npz":
            arrays = {k: np.asarray(v) for k, v in state.items()}
            # numpy types the second positional as `allow_pickle`; every key here
            # is a keyword array name, which is what savez actually accepts.
            np.savez(target, **arrays)  # type: ignore[arg-type]
        else:
            target.write_text(_state_to_json(state))
        return target

    @classmethod
    def load(cls, path: Path | str) -> Calibrator:
        """Read back a saved calibrator, dispatching on the stored ``kind``."""
        state = _read_state(Path(path))
        kind = _as_str(state.get("kind", cls.kind))
        klass = CALIBRATORS.get(kind)
        if klass is None:
            raise KeyError(f"unknown calibrator kind {kind!r}; known: {sorted(CALIBRATORS)}")
        obj = klass._from_state(state)
        obj.fitted = True
        return obj


def load_calibrator(path: Path | str) -> Calibrator:
    """Module-level alias for :meth:`Calibrator.load`."""
    return Calibrator.load(path)


def make_calibrator(kind: str, **kwargs: Any) -> Calibrator:
    """Build an unfitted calibrator by name (``"platt"``, ``"isotonic"``, ...)."""
    klass = CALIBRATORS.get(kind)
    if klass is None:
        raise KeyError(f"unknown calibrator kind {kind!r}; known: {sorted(CALIBRATORS)}")
    return klass(**kwargs)


# --------------------------------------------------------------------------- #
# Concrete calibrators
# --------------------------------------------------------------------------- #


class IdentityCalibrator(Calibrator):
    """Pass-through. The control arm when comparing calibration methods."""

    kind: ClassVar[str] = "identity"

    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None:
        return None

    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray:
        return p

    def _state(self) -> dict[str, Any]:
        return {"kind": self.kind}

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> IdentityCalibrator:
        return cls()


class PlattCalibrator(Calibrator):
    """``sigmoid(a * logit(p) + b)``. Two parameters: sharpness and bias."""

    kind: ClassVar[str] = "platt"

    def __init__(self, *, ridge: float = 1e-6) -> None:
        super().__init__()
        self.ridge = float(ridge)
        self.a: float = 1.0
        self.b: float = 0.0

    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None:
        beta = fit_logistic(logit(p), y, intercept=True, ridge=self.ridge)
        self.b, self.a = float(beta[0]), float(beta[1])

    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray:
        return sigmoid(self.a * logit(p) + self.b)

    def _state(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b, "ridge": self.ridge}

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> PlattCalibrator:
        obj = cls(ridge=_as_float(state.get("ridge", 1e-6)))
        obj.a = _as_float(state["a"])
        obj.b = _as_float(state["b"])
        return obj


class TemperatureCalibrator(Calibrator):
    """``sigmoid(logit(p) / T)``. One parameter, no bias term.

    ``T > 1`` softens an over-confident model, ``T < 1`` sharpens an
    under-confident one. Because there is no intercept it cannot repair a
    systematic home-team bias — that is what :class:`PlattCalibrator` is for.
    """

    kind: ClassVar[str] = "temperature"

    def __init__(self, *, ridge: float = 1e-6) -> None:
        super().__init__()
        self.ridge = float(ridge)
        self.temperature: float = 1.0

    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None:
        beta = fit_logistic(logit(p), y, intercept=False, ridge=self.ridge)
        inv_t = float(beta[0])
        # The NLL is convex in 1/T; a non-positive slope means the raw scores are
        # anti-correlated with the label, where no temperature helps. Fall back
        # to "no change" rather than emitting an inverted map.
        self.temperature = 1.0 / inv_t if inv_t > 1e-6 else 1.0

    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray:
        return sigmoid(logit(p) / self.temperature)

    def _state(self) -> dict[str, Any]:
        return {"kind": self.kind, "temperature": self.temperature, "ridge": self.ridge}

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> TemperatureCalibrator:
        obj = cls(ridge=_as_float(state.get("ridge", 1e-6)))
        obj.temperature = _as_float(state["temperature"])
        return obj


class IsotonicCalibrator(Calibrator):
    """Free-form monotone fit (PAVA), stored as interpolation knots.

    Non-parametric, so it can undo an S-shaped miscalibration Platt cannot —
    at the cost of needing a few thousand validation rows and of clamping
    (never extrapolating) outside the observed probability range.
    """

    kind: ClassVar[str] = "isotonic"

    def __init__(self, *, clip: tuple[float, float] = (EPS, 1.0 - EPS)) -> None:
        super().__init__()
        self.clip = (float(clip[0]), float(clip[1]))
        self.knots_x: np.ndarray = np.array([0.0, 1.0])
        self.knots_y: np.ndarray = np.array([0.0, 1.0])

    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None:
        self.knots_x, self.knots_y = _isotonic_knots(p, y)

    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray:
        if self.knots_x.size == 1:
            return np.full(p.shape, float(self.knots_y[0]))
        out = np.interp(p, self.knots_x, self.knots_y)
        return np.clip(out, self.clip[0], self.clip[1])

    def _state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "knots_x": np.asarray(self.knots_x, dtype=float),
            "knots_y": np.asarray(self.knots_y, dtype=float),
            "clip": np.asarray(self.clip, dtype=float),
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> IsotonicCalibrator:
        clip = _as_array(state.get("clip", (EPS, 1.0 - EPS)))
        obj = cls(clip=(float(clip[0]), float(clip[1])))
        obj.knots_x = _as_array(state["knots_x"])
        obj.knots_y = _as_array(state["knots_y"])
        return obj


class PhaseAwareCalibrator(Calibrator):
    """One sub-calibrator per game-phase bucket, keyed on seconds remaining.

    Miscalibration is not stationary across a game. Early on, everything sits
    near 0.5 and models are usually *under*-confident; inside two minutes the
    state space collapses and models tend to be *over*-confident about comebacks
    that cannot physically happen. A single global map splits the difference and
    is wrong at both ends.

    Buckets are half-open ``[lo, hi)`` intervals defined by ``edges``, ascending
    in seconds remaining (default :data:`DEFAULT_PHASE_EDGES`). Rows whose
    bucket has fewer than ``min_samples`` observations — and rows with a missing
    or null clock — use a global fallback calibrator fit on everything.
    """

    kind: ClassVar[str] = "phase_aware"

    def __init__(
        self,
        base: str | type[Calibrator] = "isotonic",
        *,
        edges: Sequence[float] = DEFAULT_PHASE_EDGES,
        min_samples: int = 500,
        base_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.base_kind = base if isinstance(base, str) else base.kind
        if self.base_kind not in CALIBRATORS:
            raise KeyError(f"unknown base calibrator {self.base_kind!r}")
        if self.base_kind == self.kind:
            raise ValueError("phase_aware cannot nest inside itself")
        self.edges = np.asarray(sorted(float(e) for e in edges), dtype=float)
        self.min_samples = int(min_samples)
        self.base_kwargs = dict(base_kwargs or {})
        self.buckets: list[Calibrator | None] = [None] * self.n_buckets
        self.fallback: Calibrator = make_calibrator(self.base_kind, **self.base_kwargs)

    @property
    def n_buckets(self) -> int:
        """Number of clock buckets: one more than the number of edges."""
        return int(self.edges.size) + 1

    def bucket_of(self, seconds_remaining: Any) -> np.ndarray:
        """Bucket index per row; ``-1`` where the clock is missing."""
        secs = np.asarray(seconds_remaining, dtype=float).ravel()
        idx = np.searchsorted(self.edges, secs, side="right").astype(int)
        idx[~np.isfinite(secs)] = -1
        return idx

    def bucket_label(self, index: int) -> str:
        """Human-readable ``[lo, hi)`` seconds-remaining range for a bucket."""
        lo = "0" if index == 0 else f"{self.edges[index - 1]:g}"
        hi = "inf" if index >= self.edges.size else f"{self.edges[index]:g}"
        return f"[{lo}, {hi})"

    def _fit(self, p: np.ndarray, y: np.ndarray, seconds_remaining: np.ndarray | None) -> None:
        if seconds_remaining is None:
            raise ValueError(
                "PhaseAwareCalibrator.fit requires seconds_remaining "
                "(meta__game_seconds_remaining on an assembled dataset)"
            )
        self.fallback = make_calibrator(self.base_kind, **self.base_kwargs).fit(p, y)
        idx = self.bucket_of(seconds_remaining)
        self.buckets = [None] * self.n_buckets
        for b in range(self.n_buckets):
            mask = idx == b
            if int(mask.sum()) < self.min_samples:
                continue
            self.buckets[b] = make_calibrator(self.base_kind, **self.base_kwargs).fit(
                p[mask], y[mask]
            )

    def _transform(self, p: np.ndarray, seconds_remaining: np.ndarray | None) -> np.ndarray:
        if seconds_remaining is None:
            raise ValueError("PhaseAwareCalibrator.transform requires seconds_remaining")
        idx = self.bucket_of(seconds_remaining)
        out = self.fallback.transform(p)
        for b in range(self.n_buckets):
            cal = self.buckets[b]
            if cal is None:
                continue
            mask = idx == b
            if mask.any():
                out[mask] = cal.transform(p[mask])
        return out

    def _state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "kind": self.kind,
            "base_kind": self.base_kind,
            "edges": np.asarray(self.edges, dtype=float),
            "min_samples": self.min_samples,
            "n_buckets": self.n_buckets,
            "fitted_buckets": np.asarray(
                [1 if c is not None else 0 for c in self.buckets], dtype=float
            ),
        }
        state.update(_prefixed(self.fallback._state(), "fallback__"))
        for b, cal in enumerate(self.buckets):
            if cal is not None:
                state.update(_prefixed(cal._state(), f"bucket{b}__"))
        return state

    @classmethod
    def _from_state(cls, state: dict[str, Any]) -> PhaseAwareCalibrator:
        obj = cls(
            base=_as_str(state["base_kind"]),
            edges=_as_array(state["edges"]).tolist(),
            min_samples=_as_int(state["min_samples"]),
        )
        obj.fallback = _child_from_state(_unprefixed(state, "fallback__"))
        flags = _as_array(state["fitted_buckets"])
        obj.buckets = [
            _child_from_state(_unprefixed(state, f"bucket{b}__")) if flags[b] else None
            for b in range(obj.n_buckets)
        ]
        return obj


def _child_from_state(state: dict[str, Any]) -> Calibrator:
    klass = CALIBRATORS[_as_str(state["kind"])]
    child = klass._from_state(state)
    child.fitted = True
    return child
