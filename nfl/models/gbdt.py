"""``XGBoostWinProb`` — the gradient-boosted workhorse, and the ablation that sizes it.

The dataset is 775 columns of mostly-numeric, frequently-null features. That is
exactly what a histogram GBDT is good at, and exactly what a neural net needs
babysitting for, so this is the model the architecture decision gets measured
against.

**Nulls are the point, not a nuisance.** ``bi__down`` is null on a kickoff;
``home_pl__qb1_*`` is null before a roster is published. XGBoost's default split
sends missing values down a learned default branch, which is strictly more
informative than imputing a median and pretending. Nothing here imputes.

**Strings become categories.** Sixteen feature columns are strings (``roof``,
``surface``, opponent, quarterback id). They are encoded as integer codes with
``feature_types`` marking them categorical, and the code table is persisted so a
later prediction maps ``"GRASS"`` to the same integer the fit saw. Columns whose
cardinality exceeds :data:`MAX_CATEGORY_CARDINALITY` are dropped rather than
one-hot-exploded — a player-id column with 3 000 levels is a memorisation
hazard, not a feature.

**Soft labels are native.** ``binary:logistic`` optimises cross-entropy against
a target in ``[0, 1]``, so a tie at 0.5 needs no special handling at all.

**GPU.** ``device="cuda"`` with ``tree_method="hist"``; if the CUDA build is not
usable the fit retries on CPU and records which it used in ``FitResult.extra``
and in the manifest. Unlike TensorFlow, XGBoost's CUDA support here needs no
library preloading and has no PTX warm-up.

:func:`feature_group_ablation` is the architecture question in one function:
train the full model, then retrain with each feature family dropped, and report
what the val log-loss does. A group whose removal costs nothing is a group not
worth building a wing of the pipeline for.

    >>> model = XGBoostWinProb()                            # doctest: +SKIP
    >>> model.fit(X_train, y_train, X_val, y_val).metrics   # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.features.splits import SeasonSplit
from nfl.models.artifacts import assert_model_input_frame, model_dir, new_version, write_manifest
from nfl.models.base import FitResult
from nfl.models.data import ModelFrames, design_matrix, labels, load_model_frames
from nfl.models.feature_sets import feature_set_columns
from nfl.models.game_aware import Q4Selection, game_weights, q4_selection
from nfl.models.metrics import metrics_dict, slice_metrics

__all__ = [
    "DEFAULT_ABLATION_GROUPS",
    "DEFAULT_PARAMS",
    "MAX_CATEGORY_CARDINALITY",
    "MODEL_FILE",
    "NAME",
    "XGBoostWinProb",
    "cuda_available",
    "feature_group_ablation",
    "train_xgboost",
]

log = logging.getLogger(__name__)

NAME = "xgboost_winprob"
MODEL_FILE = "model.ubj"
SCHEMA_FILE = "schema.json"

#: A string feature with more levels than this is dropped, not encoded.
MAX_CATEGORY_CARDINALITY = 64

#: Booster defaults. Depth 6 and eta 0.05 with early stopping is the standard
#: "sane, not tuned" starting point; the row/column subsampling and the L2 term
#: are what keep 775 correlated features from memorising the training seasons.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
    "seed": 17,
}

#: The feature families the ablation drops one at a time. A ``/`` separates
#: prefixes that must be dropped together — home and away sides of the same
#: builder are one architectural decision, not two.
DEFAULT_ABLATION_GROUPS: tuple[str, ...] = (
    "bi__",
    "ig__",
    "home_tm__/away_tm__",
    "home_pl__/away_pl__",
    "ctx__",
)


def cuda_available() -> bool:
    """Whether this XGBoost build can actually train on a GPU."""
    try:
        import xgboost as xgb

        info = xgb.build_info()
    except Exception:  # pragma: no cover - xgboost always importable here
        return False
    if not info.get("USE_CUDA", False):
        return False
    try:
        probe = xgb.DMatrix(np.zeros((8, 2), dtype=np.float32), label=np.zeros(8))
        xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 1}, probe, 1)
    except Exception as exc:  # pragma: no cover - depends on the host
        log.info("CUDA probe failed, falling back to CPU: %s", exc)
        return False
    return True


def group_prefixes(group: str) -> tuple[str, ...]:
    """``"home_tm__/away_tm__"`` -> ``("home_tm__", "away_tm__")``."""
    return tuple(p.strip() for p in group.split("/") if p.strip())


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


@dataclass
class _Encoding:
    """How a frame becomes the float matrix XGBoost consumes.

    ``columns`` fixes the order, ``categories`` maps each string column's levels
    to the integer codes the booster was trained on, and ``feature_types`` tells
    XGBoost which columns to treat as unordered categories.
    """

    columns: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def feature_types(self) -> list[str]:
        return ["c" if c in self.categories else "q" for c in self.columns]

    @classmethod
    def from_frame(
        cls, X: pl.DataFrame, *, max_cardinality: int = MAX_CATEGORY_CARDINALITY
    ) -> _Encoding:
        """Learn the encoding from a training frame."""
        dtypes = dict(zip(X.columns, X.dtypes, strict=True))
        columns: list[str] = []
        categories: dict[str, list[str]] = {}
        for name in X.columns:
            if dtypes[name] != pl.String:
                columns.append(name)
                continue
            levels = X.get_column(name).drop_nulls().unique().sort().to_list()
            if len(levels) > max_cardinality:
                log.info(
                    "dropping string feature %s: %d levels exceeds max_cardinality %d",
                    name,
                    len(levels),
                    max_cardinality,
                )
                continue
            columns.append(name)
            categories[name] = [str(v) for v in levels]
        return cls(columns=columns, categories=categories)

    def matrix(self, X: pl.DataFrame) -> np.ndarray:
        """``X`` as one contiguous ``float32`` array; nulls become ``NaN``."""
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"frame is missing {len(missing)} encoded column(s): {missing[:5]}")
        exprs = []
        for name in self.columns:
            if name in self.categories:
                mapping = {level: float(i) for i, level in enumerate(self.categories[name])}
                exprs.append(
                    pl.col(name)
                    .cast(pl.String)
                    .replace_strict(mapping, default=None, return_dtype=pl.Float32)
                    .alias(name)
                )
            else:
                exprs.append(pl.col(name).cast(pl.Float32, strict=False).alias(name))
        return X.select(exprs).to_numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class XGBoostWinProb:
    """Histogram GBDT over the full feature table. Null-native, GPU-first.

    Attributes:
        params: booster overrides merged onto :data:`DEFAULT_PARAMS`.
        num_boost_round: ceiling on trees; early stopping usually bites first.
        early_stopping_rounds: val log-loss patience.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (cuda when usable).
        selection_metric: ``"val"`` early-stops on overall validation log-loss
            (Phase 5 behaviour); ``"q4"`` early-stops on the mean log-loss over
            the Q4@600/300/120 marks instead, which is what ADR 0003 asks for
            and requires a ``val_selection`` (or ``meta_val``) at ``fit``.
    """

    name: str = NAME
    params: dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 3000
    early_stopping_rounds: int = 50
    device: str = "auto"
    selection_metric: str = "val"
    max_cardinality: int = MAX_CATEGORY_CARDINALITY
    version: str = ""
    booster: Any = field(default=None, repr=False)
    encoding: _Encoding = field(default_factory=_Encoding, repr=False)
    best_iteration: int = 0
    device_used: str = ""

    def _booster_params(self, device: str) -> dict[str, Any]:
        merged = dict(DEFAULT_PARAMS)
        merged.update(self.params)
        merged["device"] = device
        return merged

    def _dmatrix(
        self,
        X: pl.DataFrame,
        y: np.ndarray | None = None,
        *,
        ref: Any = None,
        weight: np.ndarray | None = None,
        quantised: bool = True,
    ) -> Any:
        """A DMatrix for ``X``. ``quantised=False`` for anything being *scored*.

        ``QuantileDMatrix`` derives its ``max_bin`` cut points from the matrix it
        is handed unless it is given a ``ref``. That is what you want when
        building the training matrix, and catastrophic on the prediction path:
        the bins — and therefore the answer — come from *whatever batch the row
        was scored in*. Measured on this model, one row at a time (which is what
        ``nfl predict`` does) collapsed every feature into a single bin and
        produced a near-constant 0.47 with correlation **0.014** to the same
        model's batch-scored output, and a val log-loss of 0.699 — worse than a
        coin flip. The same snap scored in its game came out 0.65 and in the
        season batch 0.82. A plain ``DMatrix`` bins nothing and is batch-size
        independent, which is the only acceptable behaviour for a served model.
        """
        import xgboost as xgb

        matrix = self.encoding.matrix(X)
        kwargs: dict[str, Any] = {
            "feature_names": list(self.encoding.columns),
            "feature_types": self.encoding.feature_types,
            "enable_categorical": True,
            "missing": np.nan,
        }
        # Binning runs on the CPU whatever ``device`` says, and it honours its
        # own thread count rather than the booster's, so pass it through.
        nthread = self.params.get("nthread")
        if nthread is not None:
            kwargs["nthread"] = int(nthread)
        if y is not None:
            kwargs["label"] = np.asarray(y, dtype=np.float32)
        if weight is not None:
            kwargs["weight"] = np.asarray(weight, dtype=np.float32)
        if not quantised:
            return xgb.DMatrix(matrix, **kwargs)
        if ref is not None:
            return xgb.QuantileDMatrix(matrix, ref=ref, **kwargs)
        return xgb.QuantileDMatrix(matrix, **kwargs)

    # -- WinProbModel ------------------------------------------------------ #

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
        val_selection: Q4Selection | None = None,
    ) -> FitResult:
        """Boost until the selection metric stops improving (or ``num_boost_round``).

        ``sample_weight`` is the per-row weight vector — pass
        :func:`nfl.models.game_aware.game_weights` for game-aware training.
        ``val_selection`` supplies the Q4 rows that ``selection_metric="q4"``
        early-stops on; it is built from the *validation frame's* ``meta__``
        columns, so it has to be handed in rather than derived from ``X_val``.
        """
        import xgboost as xgb

        assert_model_input_frame(X, context=f"{self.name}.fit")
        y = np.asarray(y, dtype=np.float64)
        self.encoding = _Encoding.from_frame(X, max_cardinality=self.max_cardinality)

        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights is not None and weights.shape[0] != X.height:
            raise ValueError(f"sample_weight has {weights.shape[0]} rows, X has {X.height}")

        dtrain = self._dmatrix(X, y, weight=weights)
        evals = [(dtrain, "train")]
        dval = None
        if X_val is not None and y_val is not None:
            assert_model_input_frame(X_val, context=f"{self.name}.fit(val)")
            dval = self._dmatrix(X_val, np.asarray(y_val, dtype=np.float64), ref=dtrain)
            evals.append((dval, "val"))

        # XGBoost early-stops on the *last* entry of ``evals``, so appending the
        # Q4 slice is all it takes to switch the selection criterion. The
        # per-mark 1/n weights make the booster's weighted logloss identical to
        # ``Q4Selection.mean_log_loss``.
        selection = str(self.selection_metric).lower()
        if selection not in {"val", "q4"}:
            raise ValueError(f"selection_metric must be 'val' or 'q4'; got {self.selection_metric}")
        if selection == "q4":
            if dval is None or val_selection is None or len(val_selection) == 0:
                raise ValueError(
                    "selection_metric='q4' needs a validation set and a non-empty val_selection"
                )
            rows = val_selection.rows
            evals.append(
                (
                    self._dmatrix(
                        X_val[rows],  # type: ignore[index]
                        np.asarray(y_val, dtype=np.float64)[rows],
                        ref=dtrain,
                        weight=val_selection.weights,
                    ),
                    "val_q4",
                )
            )

        wanted = "cuda" if self.device == "auto" else self.device
        if wanted == "cuda" and not cuda_available():
            log.warning("CUDA unavailable for xgboost; training on CPU")
            wanted = "cpu"

        history: dict[str, dict[str, list[float]]] = {}
        started = time.perf_counter()
        try:
            booster = xgb.train(
                self._booster_params(wanted),
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds if dval is not None else None,
                evals_result=history,
                verbose_eval=False,
            )
        except Exception as exc:
            if wanted != "cuda":
                raise
            log.warning("cuda training failed (%s); retrying on cpu", exc)
            wanted = "cpu"
            history = {}
            booster = xgb.train(
                self._booster_params(wanted),
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds if dval is not None else None,
                evals_result=history,
                verbose_eval=False,
            )
        elapsed = time.perf_counter() - started

        self.booster = booster
        self.device_used = wanted
        self.best_iteration = int(
            getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1)
        )
        self.version = self.version or new_version()

        metrics = {f"train_{k}": v for k, v in metrics_dict(self.predict_proba(X), y).items()}
        n_val = 0
        if X_val is not None and y_val is not None:
            yv = np.asarray(y_val, dtype=np.float64)
            n_val = int(np.isfinite(yv).sum())
            metrics.update(
                {f"val_{k}": v for k, v in metrics_dict(self.predict_proba(X_val), yv).items()}
            )
        return FitResult(
            model_name=self.name,
            version=self.version,
            n_train=int(X.height),
            n_val=n_val,
            metrics=metrics,
            feature_names=list(self.encoding.columns),
            extra={
                "device": wanted,
                "selection_metric": selection,
                "sample_weighted": weights is not None,
                "effective_train_weight": float(np.sum(weights)) if weights is not None else None,
                "best_iteration": self.best_iteration,
                "n_trees": int(booster.num_boosted_rounds()),
                "fit_seconds": round(elapsed, 2),
                "params": self._booster_params(wanted),
                "dropped_string_features": sorted(set(X.columns) - set(self.encoding.columns)),
                "eval_history": {k: {m: list(v) for m, v in d.items()} for k, d in history.items()},
            },
        )

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """P(home win) for every row of ``X``, using the best early-stopped tree."""
        if self.booster is None:
            raise RuntimeError(f"{self.name} is not fitted")
        assert_model_input_frame(X, context=f"{self.name}.predict_proba")
        dmat = self._dmatrix(X, quantised=False)
        out: np.ndarray = self.booster.predict(dmat, iteration_range=(0, self.best_iteration + 1))
        return out.astype(np.float64)

    def contributions(self, X: pl.DataFrame) -> np.ndarray:
        """Per-feature SHAP contributions, ``(n_rows, n_features + 1)``.

        The last column is the bias. Same unquantised matrix
        :meth:`predict_proba` uses, so an explanation describes the prediction
        that was actually served.
        """
        if self.booster is None:
            raise RuntimeError(f"{self.name} is not fitted")
        assert_model_input_frame(X, context=f"{self.name}.contributions")
        dmat = self._dmatrix(X, quantised=False)
        out = self.booster.predict(
            dmat, iteration_range=(0, self.best_iteration + 1), pred_contribs=True
        )
        return np.asarray(out, dtype=np.float64)

    # -- interpretation ---------------------------------------------------- #

    def feature_importance(self, importance_type: str = "gain") -> pl.DataFrame:
        """``(feature, importance, rank)`` sorted descending; unused features omitted."""
        if self.booster is None:
            raise RuntimeError(f"{self.name} is not fitted")
        scores = self.booster.get_score(importance_type=importance_type)
        if not scores:
            return pl.DataFrame(
                schema={"feature": pl.String, "importance": pl.Float64, "rank": pl.Int32}
            )
        frame = pl.DataFrame(
            {"feature": list(scores.keys()), "importance": [float(v) for v in scores.values()]}
        ).sort("importance", descending=True)
        return frame.with_columns((pl.int_range(1, frame.height + 1, dtype=pl.Int32)).alias("rank"))

    def group_importance(
        self, groups: Sequence[str] = DEFAULT_ABLATION_GROUPS, importance_type: str = "gain"
    ) -> pl.DataFrame:
        """Total and share of importance per feature family."""
        imp = self.feature_importance(importance_type)
        total = float(imp.get_column("importance").sum()) or 1.0
        rows = []
        for group in groups:
            prefixes = group_prefixes(group)
            part = imp.filter(pl.col("feature").str.starts_with(prefixes[0]))
            for prefix in prefixes[1:]:
                part = pl.concat([part, imp.filter(pl.col("feature").str.starts_with(prefix))])
            got = float(part.get_column("importance").sum())
            rows.append(
                {
                    "group": group,
                    "n_features": part.height,
                    "importance": got,
                    "share": got / total,
                }
            )
        return pl.DataFrame(rows).sort("importance", descending=True)

    # -- persistence ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        """Write ``model.ubj`` and ``schema.json`` into the directory ``path``."""
        if self.booster is None:
            raise RuntimeError(f"{self.name} is not fitted; nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / MODEL_FILE))
        (path / SCHEMA_FILE).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "version": self.version,
                    "columns": self.encoding.columns,
                    "categories": self.encoding.categories,
                    "best_iteration": self.best_iteration,
                    "device_used": self.device_used,
                    "params": self.params,
                    "max_cardinality": self.max_cardinality,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> XGBoostWinProb:
        """Load from an artifact directory written by :meth:`save`."""
        import xgboost as xgb

        path = Path(path)
        directory = path.parent if path.is_file() else path
        schema = json.loads((directory / SCHEMA_FILE).read_text(encoding="utf-8"))
        model = cls(
            name=schema["name"],
            version=schema.get("version", ""),
            params=schema.get("params", {}),
            max_cardinality=schema.get("max_cardinality", MAX_CATEGORY_CARDINALITY),
        )
        model.encoding = _Encoding(
            columns=list(schema["columns"]),
            categories={k: list(v) for k, v in schema["categories"].items()},
        )
        model.best_iteration = int(schema.get("best_iteration", 0))
        model.device_used = schema.get("device_used", "")
        booster = xgb.Booster()
        booster.load_model(str(directory / MODEL_FILE))
        model.booster = booster
        return model


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


def train_xgboost(
    seasons: str | int | Sequence[int] | None = None,
    *,
    frames: ModelFrames | None = None,
    split: SeasonSplit | None = None,
    root: Path | None = None,
    models_root: Path | None = None,
    stride: int = 1,
    params: Mapping[str, Any] | None = None,
    device: str = "auto",
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 50,
    save: bool = True,
    top_k_importance: int = 100,
    feature_set: str = "full",
    selection_metric: str = "val",
    weight_by_game: bool = False,
) -> tuple[XGBoostWinProb, FitResult]:
    """Train :class:`XGBoostWinProb` on the assembled dataset and save it.

    The manifest carries the top ``top_k_importance`` features by gain, the
    per-group importance shares, the feature-set name, and the val/test slice
    tables.

    ``feature_set`` narrows the design matrix (see
    :mod:`nfl.models.feature_sets`); ``weight_by_game`` gives every game equal
    weight rather than every play; ``selection_metric="q4"`` early-stops on the
    Q4 marks. The ADR 0003 v1 configuration is
    ``feature_set="v1", weight_by_game=True, selection_metric="q4"``.
    """
    data = (
        frames
        if frames is not None
        else load_model_frames(seasons, split=split, root=root, stride=stride)
    )
    features = feature_set_columns(data.train, feature_set)
    x_train, y_train = design_matrix(data.train, features), labels(data.train)
    x_val, y_val = design_matrix(data.val, features), labels(data.val)

    weights = game_weights(data.train.select("meta__game_id")) if weight_by_game else None
    selection = q4_selection(data.val) if selection_metric == "q4" else None

    model = XGBoostWinProb(
        params=dict(params or {}),
        device=device,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        selection_metric=selection_metric,
    )
    result = model.fit(
        x_train, y_train, x_val, y_val, sample_weight=weights, val_selection=selection
    )
    result.extra["feature_set"] = feature_set
    if selection is not None:
        q4_p = model.predict_proba(x_val)[selection.rows]
        result.extra["val_q4_selection_log_loss"] = selection.mean_log_loss(
            q4_p, y_val[selection.rows]
        )

    importance = model.feature_importance("gain")
    result.extra["feature_importance_gain"] = importance.head(top_k_importance).to_dicts()
    result.extra["group_importance_gain"] = model.group_importance().to_dicts()

    # Key ``not_collapsed`` on the naive floor's band so this manifest is
    # comparable to the floor's row for row (ADR 0003 §5). Imported here because
    # the ensemble trainer is a layer above this one.
    from nfl.models.train_ensemble import reference_probabilities

    ref_val, ref_pin = reference_probabilities(data.val, features, root=models_root)
    val_table = slice_metrics(data.val, model.predict_proba(x_val), y_val, reference_probs=ref_val)
    result.extra["val_slices"] = val_table.to_dicts()
    result.extra["split"] = data.split.to_dict()
    result.extra["not_collapsed_reference"] = ref_pin
    if data.test.height:
        x_test, y_test = design_matrix(data.test, features), labels(data.test)
        ref_test, _ = reference_probabilities(data.test, features, root=models_root)
        test_table = slice_metrics(
            data.test, model.predict_proba(x_test), y_test, reference_probs=ref_test
        )
        result.extra["test_slices"] = test_table.to_dicts()

    if save:
        from nfl.models.data import seasons_in

        target = model_dir(model.name, result.version, root=models_root)
        model.save(target)
        write_manifest(
            target,
            name=model.name,
            version=result.version,
            feature_names=model.encoding.columns,
            metrics=result.metrics,
            train_seasons=seasons_in(data.train),
            extra={
                "kind": "xgboost",
                "feature_set": feature_set,
                "selection_metric": selection_metric,
                "weight_by_game": weight_by_game,
                "val_q4_selection_log_loss": result.extra.get("val_q4_selection_log_loss"),
                "device": result.extra["device"],
                "params": result.extra["params"],
                "best_iteration": model.best_iteration,
                "n_trees": result.extra["n_trees"],
                "fit_seconds": result.extra["fit_seconds"],
                "stride": stride,
                "split": result.extra["split"],
                "dropped_string_features": result.extra["dropped_string_features"],
                "feature_importance_gain": result.extra["feature_importance_gain"],
                "group_importance_gain": result.extra["group_importance_gain"],
                "val_slices": result.extra["val_slices"],
                "test_slices": result.extra.get("test_slices"),
                "not_collapsed_reference": result.extra["not_collapsed_reference"],
            },
        )
        log.info("wrote %s", target)
    return model, result


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #


def feature_group_ablation(
    groups: Sequence[str] = DEFAULT_ABLATION_GROUPS,
    *,
    seasons: str | int | Sequence[int] | None = None,
    frames: ModelFrames | None = None,
    root: Path | None = None,
    stride: int = 3,
    params: Mapping[str, Any] | None = None,
    device: str = "auto",
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 40,
) -> pl.DataFrame:
    """Drop each feature family in turn and report what val log-loss does.

    Trains one model on everything, then one per group with that group's columns
    removed, and returns a frame of

    ``group`` (``"full"`` for the reference), ``n_features``, ``val_log_loss``,
    ``delta_log_loss`` (positive = the model got *worse* without the group, i.e.
    the group earns its keep), ``val_brier``, ``val_auc``, ``best_iteration``
    and ``seconds``.

    ``stride`` defaults to 3 — every third play — because six full fits over
    620k rows is not worth the wall time and consecutive plays are near
    duplicates. Set ``stride=1`` for the slow, exact version.
    """
    data = frames if frames is not None else load_model_frames(seasons, root=root, stride=stride)
    x_train, y_train = data.as_xy("train")
    x_val, y_val = data.as_xy("val")

    def run(label: str, drop: Sequence[str]) -> dict[str, Any]:
        keep = (
            [c for c in x_train.columns if not c.startswith(tuple(drop))]
            if drop
            else list(x_train.columns)
        )
        model = XGBoostWinProb(
            params=dict(params or {}),
            device=device,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        started = time.perf_counter()
        result = model.fit(x_train.select(keep), y_train, x_val.select(keep), y_val)
        seconds = time.perf_counter() - started
        log.info(
            "ablation %-24s features=%4d val_log_loss=%.5f (%.0fs)",
            label,
            len(keep),
            result.metrics["val_log_loss"],
            seconds,
        )
        return {
            "group": label,
            "n_features": len(keep),
            "n_dropped": x_train.width - len(keep),
            "val_log_loss": result.metrics["val_log_loss"],
            "val_brier": result.metrics["val_brier"],
            "val_auc": result.metrics["val_auc"],
            "best_iteration": model.best_iteration,
            "device": model.device_used,
            "seconds": round(seconds, 1),
        }

    rows = [run("full", ())]
    baseline = rows[0]["val_log_loss"]
    for group in groups:
        rows.append(run(f"drop {group}", group_prefixes(group)))
    return pl.DataFrame(rows).with_columns(
        (pl.col("val_log_loss") - baseline).alias("delta_log_loss")
    )
