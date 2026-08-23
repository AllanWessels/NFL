"""Tests for the v1 report assembly and the ADR 0003 promotion gate.

Pure table logic — no fitting. The gate is the thing that decides whether a
model ships, so it is checked against hand-built tables where the right answer
is obvious, including the near-miss cases (wins two marks out of three, wins on
log-loss but not Brier, wins Q4 but regresses on not-collapsed).
"""

from __future__ import annotations

import polars as pl

from nfl.models.v1 import SLICE_ORDER, promotion_gate, report_table

_SLICES = ("q4_600s", "q4_300s", "q4_120s", "not_collapsed", "overall")


def table(**models: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """``model -> {slice: (log_loss, brier)}`` as the report frame shape."""
    rows = []
    for name, slices in models.items():
        for s in _SLICES:
            ll, br = slices[s]
            rows.append(
                {
                    "model": name,
                    "split": "val",
                    "slice": s,
                    "n": 100,
                    "log_loss": ll,
                    "brier": br,
                    "accuracy": 0.7,
                    "auc": 0.8,
                    "ece": 0.02,
                }
            )
    return pl.DataFrame(rows)


def flat(value: float) -> dict[str, tuple[float, float]]:
    return dict.fromkeys(_SLICES, (value, value / 2))


def test_report_table_orders_q4_marks_before_the_summary_slices():
    frame = table(naive=flat(0.3))
    out = report_table([frame])
    assert out.get_column("slice").to_list() == list(SLICE_ORDER)


def test_a_model_better_everywhere_is_promotable():
    gate = promotion_gate(table(naive=flat(0.30), cand=flat(0.28)), floor="naive")
    row = gate.row(0, named=True)
    assert row["model"] == "cand"
    assert row["promotable"] is True


def test_a_model_worse_everywhere_is_not_promotable():
    gate = promotion_gate(table(naive=flat(0.30), cand=flat(0.32)), floor="naive")
    assert gate.row(0, named=True)["promotable"] is False


def test_winning_two_of_three_q4_marks_is_not_enough():
    cand = flat(0.28)
    cand["q4_120s"] = (0.31, 0.14)
    gate = promotion_gate(table(naive=flat(0.30), cand=cand), floor="naive")
    row = gate.row(0, named=True)
    assert row["beats_q4_600s_log_loss"] is True
    assert row["beats_q4_120s_log_loss"] is False
    assert row["promotable"] is False


def test_beating_log_loss_but_not_brier_is_not_enough():
    cand = flat(0.28)
    cand["q4_300s"] = (0.28, 0.20)  # better log-loss, worse Brier than 0.15
    gate = promotion_gate(table(naive=flat(0.30), cand=cand), floor="naive")
    row = gate.row(0, named=True)
    assert row["beats_q4_300s_log_loss"] is True
    assert row["beats_q4_300s_brier"] is False
    assert row["promotable"] is False


def test_regressing_on_not_collapsed_blocks_promotion():
    cand = flat(0.28)
    cand["not_collapsed"] = (0.33, 0.16)
    gate = promotion_gate(table(naive=flat(0.30), cand=cand), floor="naive")
    row = gate.row(0, named=True)
    assert all(row[f"beats_q4_{m}s_log_loss"] for m in (600, 300, 120))
    assert row["not_worse_not_collapsed"] is False
    assert row["promotable"] is False


def test_the_floor_itself_is_not_reported_as_a_candidate():
    gate = promotion_gate(table(naive=flat(0.30), a=flat(0.29), b=flat(0.31)), floor="naive")
    assert gate.get_column("model").to_list() == ["a", "b"]


def test_the_gate_reads_the_requested_split_only():
    val = table(naive=flat(0.30), cand=flat(0.28))
    test = table(naive=flat(0.30), cand=flat(0.40)).with_columns(pl.lit("test").alias("split"))
    both = pl.concat([val, test])
    assert promotion_gate(both, floor="naive", split="val").row(0, named=True)["promotable"]
    assert not promotion_gate(both, floor="naive", split="test").row(0, named=True)["promotable"]
