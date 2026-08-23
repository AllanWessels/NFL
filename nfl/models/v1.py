"""Train the ADR 0003 v1 line-up and report it against the naive floor.

One entry point, :func:`train_and_report`, so "retrain v1" is a single command
and the promotion decision is computed rather than eyeballed:

1. the naive logistic floor, exactly as Phase 5 defined it (nine columns, no
   weighting) — changing the floor would make the comparison meaningless;
2. XGBoost on the ``v1`` feature set, game-weighted, early-stopped on val Q4;
3. ``tf_seq`` (GRU) on ``v1``, same weighting and same selection metric;
4. ``tf_mlp`` on ``v1``, likewise — cheap enough to keep in the line-up.

Every model is then scored with :func:`nfl.models.metrics.slice_metrics` on
*both* the validation season and the held-out test seasons, over the five
slices the project reports: overall, Q4@600/300/120, and not-collapsed.

**The gate** (ADR 0003 item 5) is checked by :func:`promotion_gate`: a
candidate is promotable only if it beats the naive floor on *all three* val Q4
marks, on both log-loss and Brier, and is not worse on the not-collapsed
subset. It returns the verdict per model rather than a single boolean, because
"the GRU passes and the MLP does not" is the answer the ADR wants.

    $ uv run python -m nfl.models.v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl.models.artifacts import MANIFEST_NAME, model_dir
from nfl.models.data import ModelFrames, design_matrix, labels, load_model_frames
from nfl.models.metrics import Q4_MARKS, slice_metrics

__all__ = [
    "PHASE5_TEST_LOG_LOSS",
    "SLICE_ORDER",
    "promotion_gate",
    "report_table",
    "train_and_report",
]

log = logging.getLogger(__name__)

#: The five reported slices, in table order.
SLICE_ORDER: tuple[str, ...] = (
    *[f"q4_{m}s" for m in Q4_MARKS],
    "not_collapsed",
    "overall",
)

#: The Phase 5 test log-losses copied out of ADR 0003, so the report can print
#: the "before" column without re-running Phase 5. Keyed by the ADR's row label.
PHASE5_TEST_LOG_LOSS: dict[str, dict[str, float]] = {
    "naive logistic (775-era floor)": {
        "q4_600s": 0.3399,
        "q4_300s": 0.2928,
        "q4_120s": 0.2485,
        "overall": 0.4879,
    },
    "xgboost (775 features)": {
        "q4_600s": 0.3643,
        "q4_300s": 0.3245,
        "q4_120s": 0.2998,
        "overall": 0.4773,
    },
    "tf_mlp (775 features)": {
        "q4_600s": 0.3481,
        "q4_300s": 0.3119,
        "q4_120s": 0.2803,
        "overall": 0.4699,
    },
    "tf_seq (775 features)": {
        "q4_600s": 0.3443,
        "q4_300s": 0.3022,
        "q4_120s": 0.2634,
        "overall": 0.4735,
    },
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _slices(frame: pl.DataFrame, probs: np.ndarray, y: np.ndarray, label: str, split: str):
    table = slice_metrics(frame, probs, y)
    return table.with_columns(pl.lit(label).alias("model"), pl.lit(split).alias("split")).select(
        "model", "split", "slice", "n", "log_loss", "brier", "accuracy", "auc", "ece"
    )


def report_table(rows: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Stack per-model slice tables into the report, in slice order."""
    order = {name: i for i, name in enumerate(SLICE_ORDER)}
    return (
        pl.concat(rows)
        .with_columns(
            pl.col("slice").replace_strict(order, default=99, return_dtype=pl.Int32).alias("_o")
        )
        .sort(["split", "_o"])
        .drop("_o")
    )


def promotion_gate(table: pl.DataFrame, *, floor: str, split: str = "val") -> pl.DataFrame:
    """ADR 0003 item 5, per candidate: does it clear the naive floor?

    Returns one row per non-floor model with a boolean per condition and the
    ``promotable`` conjunction of them.
    """
    part = table.filter(pl.col("split") == split)
    base = {
        row["slice"]: row for row in part.filter(pl.col("model") == floor).iter_rows(named=True)
    }
    rows = []
    for name in part.get_column("model").unique(maintain_order=True):
        if name == floor:
            continue
        mine = {r["slice"]: r for r in part.filter(pl.col("model") == name).iter_rows(named=True)}
        marks = [f"q4_{m}s" for m in Q4_MARKS]
        beats_ll = {m: mine[m]["log_loss"] < base[m]["log_loss"] for m in marks}
        beats_br = {m: mine[m]["brier"] < base[m]["brier"] for m in marks}
        not_worse = mine["not_collapsed"]["log_loss"] <= base["not_collapsed"]["log_loss"]
        rows.append(
            {
                "model": name,
                **{f"beats_{m}_log_loss": beats_ll[m] for m in marks},
                **{f"beats_{m}_brier": beats_br[m] for m in marks},
                "not_worse_not_collapsed": not_worse,
                "promotable": all(beats_ll.values()) and all(beats_br.values()) and not_worse,
            }
        )
    return pl.DataFrame(rows)


def _amend_manifest(directory: Path, extra: dict[str, Any]) -> None:
    """Merge ``extra`` into an artifact's manifest (feature set, slice tables)."""
    path = directory / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_and_report(
    *,
    root: Path | None = None,
    models_root: Path | None = None,
    xgb_params: dict[str, Any] | None = None,
    xgb_configs: Sequence[tuple[str, str, dict[str, Any]]] | None = None,
    seq_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    include_mlp: bool = True,
    device: str = "auto",
    save: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train the line-up, score it on val and test, return ``(report, gate)``."""
    from nfl.models.gbdt import train_xgboost
    from nfl.models.naive import train_naive
    from nfl.models.tf_mlp import train_tf_mlp
    from nfl.models.tf_seq import SEQ_META_COLUMNS, train_tf_seq

    data: ModelFrames = load_model_frames(root=root)
    log.info(
        "frames: train=%d val=%d test=%d", data.train.height, data.val.height, data.test.height
    )
    parts: list[pl.DataFrame] = []

    def score(label: str, predict) -> None:
        for split in ("val", "test"):
            frame = getattr(data, split)
            if not frame.height:
                continue
            parts.append(_slices(frame, predict(frame), labels(frame), label, split))

    # -- 1. naive floor, unchanged from Phase 5 ---------------------------- #
    t0 = time.perf_counter()
    naive, naive_res = train_naive(frames=data, models_root=models_root, save=save)
    log.info("naive fitted in %.0fs", time.perf_counter() - t0)
    naive_features = list(naive_res.feature_names)
    score("naive", lambda f: naive.predict_proba(design_matrix(f, naive_features)))

    # -- 2. XGBoost, one fit per configuration ----------------------------- #
    # More than one is allowed because the feature-set question is not settled
    # by the ADR alone: ``inplay`` and ``v1`` differ by ten columns and the
    # report is what decides between them.
    configs = list(xgb_configs or [("xgb-v1", "v1", dict(xgb_params or {}))])
    xgb_fits: list[tuple[str, Any, Any]] = []
    for label, feature_set, params in configs:
        t0 = time.perf_counter()
        xgb_model, xgb_res = train_xgboost(
            frames=data,
            models_root=models_root,
            device=device,
            params=dict(params),
            feature_set=feature_set,
            weight_by_game=True,
            selection_metric="q4",
            save=save,
        )
        log.info(
            "%s fitted in %.0fs (best_iteration=%d)",
            label,
            time.perf_counter() - t0,
            xgb_model.best_iteration,
        )
        cols = list(xgb_model.encoding.columns)
        score(label, lambda f, m=xgb_model, c=cols: m.predict_proba(design_matrix(f, c)))
        xgb_fits.append((label, xgb_model, xgb_res))

    # -- 3. tf_seq v1 ------------------------------------------------------- #
    t0 = time.perf_counter()
    seq_model, seq_res = train_tf_seq(
        root=root,
        models_root=models_root,
        feature_set="v1",
        selection_metric="q4",
        weight_by_game=True,
        save=save,
        **(seq_params or {}),
    )
    log.info("seq-v1 fitted in %.0fs", time.perf_counter() - t0)
    seq_meta = [c for c in SEQ_META_COLUMNS if c in data.val.columns]

    def seq_predict(frame: pl.DataFrame) -> np.ndarray:
        return seq_model.predict_proba(design_matrix(frame, data.features), frame.select(seq_meta))

    score("seq-v1", seq_predict)
    if save:
        _amend_manifest(
            model_dir(seq_model.name, seq_res.version, root=models_root),
            {"feature_set": "v1", "weight_by_game": True, "selection_metric": "q4"},
        )

    # -- 4. tf_mlp v1 (optional) -------------------------------------------- #
    mlp_res = None
    if include_mlp:
        t0 = time.perf_counter()
        mlp_model, mlp_res = train_tf_mlp(
            root=root,
            models_root=models_root,
            feature_set="v1",
            selection_metric="q4",
            weight_by_game=True,
            save=save,
            **(mlp_params or {}),
        )
        log.info("mlp-v1 fitted in %.0fs", time.perf_counter() - t0)
        mlp_cols = list(mlp_model.preprocessor.feature_names)
        score("mlp-v1", lambda f: mlp_model.predict_proba(design_matrix(f, mlp_cols)))
        if save:
            _amend_manifest(
                model_dir(mlp_model.name, mlp_res.version, root=models_root),
                {"feature_set": "v1", "weight_by_game": True, "selection_metric": "q4"},
            )

    report = report_table(parts)
    gate = promotion_gate(report, floor="naive")
    if save:
        # The slice tables belong on the artifacts too, not only in the report.
        stamped = [("naive", naive.name, naive_res), ("seq-v1", seq_model.name, seq_res)]
        stamped += [(label, m.name, r) for label, m, r in xgb_fits]
        for label, name, res in stamped:
            _amend_manifest(
                model_dir(name, res.version, root=models_root),
                {"v1_report": report.filter(pl.col("model") == label).to_dicts()},
            )
    return report, gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and report the ADR 0003 v1 line-up.")
    parser.add_argument("--xgb-params", default="{}", help="JSON booster overrides")
    parser.add_argument(
        "--xgb-configs",
        default=None,
        help='JSON list of [label, feature_set, params], e.g. [["xgb-v1","v1",{}]]',
    )
    parser.add_argument("--seq-params", default="{}", help="JSON SeqWinProb kwargs")
    parser.add_argument("--mlp-params", default="{}", help="JSON MLPWinProb kwargs")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--out", default=None, help="write the report table to this JSON path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report, gate = train_and_report(
        xgb_params=json.loads(args.xgb_params),
        xgb_configs=(
            [tuple(c) for c in json.loads(args.xgb_configs)] if args.xgb_configs else None
        ),
        seq_params=json.loads(args.seq_params),
        mlp_params=json.loads(args.mlp_params),
        include_mlp=not args.no_mlp,
    )
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220):
        print(report)
        print(gate)
    if args.out:
        Path(args.out).write_text(
            json.dumps({"report": report.to_dicts(), "gate": gate.to_dicts()}, indent=2), "utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
