"""Small, time-boxed hyper-parameter sweeps, scored on the val Q4 slices.

ADR 0003 item 3. Two sweeps, one per learner, both selecting on the same number
the promotion gate uses: the mean validation log-loss over Q4@600/300/120
(:meth:`nfl.models.game_aware.Q4Selection.mean_log_loss`). Overall log-loss is
recorded too, because the two disagree and it is worth seeing by how much.

Both sweeps are **budgeted, not exhaustive**: they draw configurations from a
small discrete grid without replacement and stop at the first trial boundary
past ``budget_seconds``. That is deliberate — 620k rows and a GPU make a fit
cheap enough to run a dozen of, and nowhere near cheap enough to run a proper
Bayesian search of, and a dozen well-spread draws is worth more here than a
fine sweep of one axis.

**``min_child_weight`` under game weights.** Row weights of ``1/plays`` divide
every Hessian sum by ~150, so the Phase 5 default of 10 is effectively a
minimum of ~1500 plays per leaf. The grid below is scaled to match; do not
carry a ``min_child_weight`` from an unweighted run into a weighted one.

Every trial is appended to ``data/models/sweeps/<kind>_<version>.json`` as it
finishes, so an interrupted sweep still leaves a readable record.

    $ uv run python -m nfl.models.sweeps xgb --budget 900
    $ uv run python -m nfl.models.sweeps seq --budget 900
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.config import MODELS_DIR
from nfl.models.artifacts import new_version
from nfl.models.data import ModelFrames, design_matrix, labels, load_model_frames
from nfl.models.feature_sets import feature_set_columns
from nfl.models.game_aware import Q4Selection, game_weights, q4_selection
from nfl.models.metrics import brier, log_loss

__all__ = [
    "SEQ_GRID",
    "SWEEPS_DIR",
    "XGB_GRID",
    "sweep_tf_seq",
    "sweep_xgboost",
]

log = logging.getLogger(__name__)

SWEEPS_DIR = MODELS_DIR / "sweeps"

#: XGBoost search space. Depth 3-6 per the ADR; ``min_child_weight`` scaled for
#: game weights (see the module docstring).
XGB_GRID: dict[str, Sequence[Any]] = {
    "max_depth": (3, 4, 5, 6),
    "eta": (0.03, 0.05, 0.10),
    "min_child_weight": (0.02, 0.1, 0.5, 2.0),
    "colsample_bytree": (0.5, 0.7, 0.9),
    "subsample": (0.7, 0.9),
    "reg_lambda": (1.0, 5.0),
}

#: ``tf_seq`` search space: window length, capacity, and the three knobs that
#: decide whether the tabular branch memorises the game.
SEQ_GRID: dict[str, Sequence[Any]] = {
    "k": (16, 32, 64),
    "gru_units": (48, 64, 96),
    "dropout": (0.3, 0.5),
    "learning_rate": (3e-4, 1e-3),
    "pregame_noise": (0.5, 1.0),
}


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


def _draws(grid: dict[str, Sequence[Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` distinct configurations from ``grid``, first one at the grid centre."""
    keys = list(grid)
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]
    rng = random.Random(seed)
    rng.shuffle(combos)
    # Put a middle-of-the-grid configuration first so an aborted sweep still has
    # a sane reference trial in it.
    centre = {k: v[len(v) // 2] for k, v in grid.items()}
    combos = [centre] + [c for c in combos if c != centre]
    return combos[:n]


def _score(
    probs: np.ndarray, y: np.ndarray, selection: Q4Selection
) -> dict[str, float | dict[str, float]]:
    """Val metrics for one trial: the Q4 selection metric plus the overall pair."""
    q4_p, q4_y = probs[selection.rows], y[selection.rows]
    per_mark = selection.per_mark_log_loss(q4_p, q4_y)
    return {
        "val_q4_log_loss": selection.mean_log_loss(q4_p, q4_y),
        "val_q4_per_mark": {f"q4_{k}s": v for k, v in per_mark.items()},
        "val_log_loss": log_loss(probs, y),
        "val_brier": brier(probs, y),
    }


class _Recorder:
    """Writes the sweep file after every trial, so a kill -9 loses one trial."""

    def __init__(self, kind: str, out_dir: Path | None, meta: dict[str, Any]) -> None:
        self.path = (out_dir or SWEEPS_DIR) / f"{kind}_{new_version()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload: dict[str, Any] = {"kind": kind, **meta, "trials": []}
        self.flush()

    def add(self, trial: dict[str, Any]) -> None:
        self.payload["trials"].append(trial)
        best = min(
            (t for t in self.payload["trials"] if np.isfinite(t.get("val_q4_log_loss", np.nan))),
            key=lambda t: t["val_q4_log_loss"],
            default=None,
        )
        self.payload["best"] = best
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.payload, indent=2, default=str) + "\n", "utf-8")


def _table(recorder: _Recorder) -> pl.DataFrame:
    rows = [
        {
            "trial": t["trial"],
            **{k: v for k, v in t["params"].items()},
            "val_q4_log_loss": t.get("val_q4_log_loss"),
            "val_log_loss": t.get("val_log_loss"),
            "seconds": t.get("seconds"),
        }
        for t in recorder.payload["trials"]
    ]
    return pl.DataFrame(rows).sort("val_q4_log_loss")


def _budgeted(n: int, budget_seconds: float, started: float) -> Iterator[int]:
    """Yield trial indices until the budget is spent (checked between trials)."""
    for i in range(n):
        elapsed = time.perf_counter() - started
        if i and elapsed > budget_seconds:
            log.info("sweep budget spent after %d trials (%.0fs)", i, elapsed)
            return
        yield i


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #


def sweep_xgboost(
    *,
    frames: ModelFrames | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
    feature_set: str = "v1",
    budget_seconds: float = 900.0,
    max_trials: int = 24,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 60,
    device: str = "auto",
    stride: int = 1,
    seed: int = 17,
    weight_by_game: bool = True,
    grid: dict[str, Sequence[Any]] | None = None,
    tag: str = "xgb",
) -> pl.DataFrame:
    """Sweep the booster on val Q4, one JSON record per trial. Returns the table.

    ``grid`` overrides :data:`XGB_GRID` (a one-element grid turns this into a
    single controlled comparison, which is how the feature-set / weighting
    ablation is run); ``tag`` names the sweep file.
    """
    from nfl.models.gbdt import XGBoostWinProb

    space = XGB_GRID if grid is None else grid
    data = frames if frames is not None else load_model_frames(root=root, stride=stride)
    features = feature_set_columns(data.train, feature_set)
    x_train, y_train = design_matrix(data.train, features), labels(data.train)
    x_val, y_val = design_matrix(data.val, features), labels(data.val)
    weights = game_weights(data.train.select("meta__game_id")) if weight_by_game else None
    selection = q4_selection(data.val)

    rec = _Recorder(
        tag,
        out_dir,
        {
            "feature_set": feature_set,
            "n_features": len(features),
            "n_train": x_train.height,
            "n_val": x_val.height,
            "n_q4_rows": len(selection),
            "weight_by_game": weight_by_game,
            "selection_metric": "q4",
            "budget_seconds": budget_seconds,
            "grid": {k: list(v) for k, v in space.items()},
        },
    )
    combos = _draws(space, max_trials, seed)
    started = time.perf_counter()
    for i in _budgeted(len(combos), budget_seconds, started):
        params = combos[i]
        t0 = time.perf_counter()
        model = XGBoostWinProb(
            params=dict(params),
            device=device,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            selection_metric="q4",
        )
        result = model.fit(
            x_train, y_train, x_val, y_val, sample_weight=weights, val_selection=selection
        )
        scores = _score(model.predict_proba(x_val), y_val, selection)
        rec.add(
            {
                "trial": i,
                "params": params,
                **scores,
                "best_iteration": model.best_iteration,
                "device": result.extra["device"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log.info(
            "xgb trial %2d %s val_q4=%.5f (%.0fs)",
            i,
            params,
            scores["val_q4_log_loss"],
            time.perf_counter() - t0,
        )
    log.info("wrote %s", rec.path)
    return _table(rec)


# --------------------------------------------------------------------------- #
# tf_seq
# --------------------------------------------------------------------------- #


def sweep_tf_seq(
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    feature_set: str = "v1",
    budget_seconds: float = 900.0,
    max_trials: int = 12,
    epochs: int = 40,
    patience: int = 6,
    seed: int = 17,
    **fixed: Any,
) -> pl.DataFrame:
    """Sweep the GRU on val Q4. Windows are rebuilt per trial (``k`` varies)."""
    from nfl.features.assemble import feature_columns
    from nfl.models.tf_common import labels_of, read_modelling_frames
    from nfl.models.tf_seq import SEQ_META_COLUMNS, SeqWinProb

    train, val, _ = read_modelling_frames(None, root=root)
    features = feature_columns(train)
    meta_cols = [c for c in SEQ_META_COLUMNS if c in train.columns]
    x_train, y_train = train.select(features), labels_of(train)
    x_val, y_val = val.select(features), labels_of(val)
    weights = game_weights(train.select("meta__game_id"))
    selection = q4_selection(val)

    rec = _Recorder(
        "tf_seq",
        out_dir,
        {
            "feature_set": feature_set,
            "n_train": x_train.height,
            "n_val": x_val.height,
            "n_q4_rows": len(selection),
            "weight_by_game": True,
            "selection_metric": "q4",
            "budget_seconds": budget_seconds,
            "grid": {k: list(v) for k, v in SEQ_GRID.items()},
        },
    )
    combos = _draws(SEQ_GRID, max_trials, seed)
    started = time.perf_counter()
    for i in _budgeted(len(combos), budget_seconds, started):
        params = combos[i]
        t0 = time.perf_counter()
        model = SeqWinProb(
            feature_set=feature_set,
            selection_metric="q4",
            epochs=epochs,
            patience=patience,
            **params,
            **fixed,
        )
        result = model.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            meta=train.select(meta_cols),
            meta_val=val.select(meta_cols),
            sample_weight=weights,
            val_selection=selection,
        )
        scores = _score(model.predict_proba(x_val, val.select(meta_cols)), y_val, selection)
        rec.add(
            {
                "trial": i,
                "params": params,
                **scores,
                "epochs_run": result.extra["epochs_run"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log.info(
            "tf_seq trial %2d %s val_q4=%.5f (%.0fs)",
            i,
            params,
            scores["val_q4_log_loss"],
            time.perf_counter() - t0,
        )
    log.info("wrote %s", rec.path)
    return _table(rec)


# --------------------------------------------------------------------------- #
# CLI (module entry point; `nfl.cli` is out of scope for this branch)
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("xgb", "seq"))
    parser.add_argument("--budget", type=float, default=900.0, help="seconds")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--feature-set", default="v1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    kwargs: dict[str, Any] = {
        "budget_seconds": args.budget,
        "feature_set": args.feature_set,
        "seed": args.seed,
    }
    if args.max_trials is not None:
        kwargs["max_trials"] = args.max_trials
    table = sweep_xgboost(**kwargs) if args.kind == "xgb" else sweep_tf_seq(**kwargs)
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(table)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
