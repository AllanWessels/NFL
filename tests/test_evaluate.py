"""Evaluation metrics, calibration, edge-vs-market and retrain triggers.

The edge numbers in :func:`test_edge_report_matches_the_hand_computed_bet_ledger`
are worked out by hand in the docstring there, so a regression in the betting
maths shows up as a wrong number rather than a changed-but-plausible one.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.logging import evaluate as ev
from nfl.logging import ledger as ledger_mod


def frame(**columns: object) -> pl.DataFrame:
    """A ledger-schema frame from column-wise lists."""
    n = len(next(iter(columns.values())))  # type: ignore[arg-type]
    columns.setdefault("game_id", [f"2024_{i:02d}_AA_BB" for i in range(1, n + 1)])
    columns.setdefault("prediction_id", [f"id-{i}" for i in range(n)])
    columns.setdefault("model_name", ["m"] * n)
    columns.setdefault("model_version", ["v1"] * n)
    return ledger_mod.conform(pl.DataFrame(dict(columns), strict=False))


# --------------------------------------------------------------------------- #
# Core metrics — known answers
# --------------------------------------------------------------------------- #


def test_log_loss_of_a_coin_flip_is_ln_two() -> None:
    assert ev.metrics([1, 0], [0.5, 0.5]).log_loss == pytest.approx(math.log(2))


def test_a_perfect_forecast_scores_zero() -> None:
    m = ev.metrics([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0])
    assert m.brier == pytest.approx(0.0)
    assert m.accuracy == pytest.approx(1.0)
    assert m.ece == pytest.approx(0.0)
    assert m.auc == pytest.approx(1.0)


def test_brier_is_the_mean_squared_error() -> None:
    # (0.8-1)^2 + (0.3-0)^2 = 0.04 + 0.09 -> 0.065
    assert ev.metrics([1, 0], [0.8, 0.3]).brier == pytest.approx(0.065)


def test_auc_counts_correctly_ordered_pairs() -> None:
    # positives 0.35, 0.80 vs negatives 0.10, 0.40 -> 3 of 4 pairs ordered.
    assert ev.metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).auc == pytest.approx(0.75)


def test_auc_gives_tied_scores_half_credit() -> None:
    assert ev.metrics([0, 1], [0.5, 0.5]).auc == pytest.approx(0.5)


def test_auc_is_nan_when_one_class_is_missing() -> None:
    assert math.isnan(ev.metrics([1, 1], [0.6, 0.7]).auc)


def test_ece_is_the_weighted_calibration_gap() -> None:
    # Every prediction is 0.9 but only half the games are won: gap 0.4 on all
    # the mass, so ECE is 0.4.
    assert ev.metrics([1, 0], [0.9, 0.9]).ece == pytest.approx(0.4)


def test_metrics_on_nothing_is_empty_rather_than_an_error() -> None:
    assert ev.metrics([], []).n == 0


def test_reliability_table_bins_and_reports_the_gap() -> None:
    table = ev.reliability_table([1, 1, 0, 0], [0.95, 0.95, 0.05, 0.05], bins=10)
    assert table.height == 10
    top = table.filter(pl.col("n") > 0).sort("bin")
    assert top.get_column("n").to_list() == [2, 2]
    assert top.get_column("gap").to_list() == pytest.approx([0.05, -0.05])


# --------------------------------------------------------------------------- #
# Kalshi fees
# --------------------------------------------------------------------------- #


def test_kalshi_fee_follows_the_published_formula() -> None:
    # 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly, no rounding needed.
    assert ev.kalshi_fee(0.50, 100, round_up=False) == pytest.approx(1.75)
    assert ev.kalshi_fee(0.50, 100) == pytest.approx(1.75)
    # 0.07 * 1 * 0.6 * 0.4 = 0.0168 -> rounds up to a whole cent.
    assert ev.kalshi_fee(0.60) == pytest.approx(0.02)
    assert ev.kalshi_fee(0.60, round_up=False) == pytest.approx(0.0168)


def test_kalshi_fee_vanishes_at_the_extremes() -> None:
    assert ev.kalshi_fee(1.0, 100) == 0.0
    assert ev.kalshi_fee(0.0, 100) == 0.0


def test_kalshi_fee_peaks_at_a_coin_flip() -> None:
    mid = ev.kalshi_fee(0.5, 1000, round_up=False)
    assert mid > ev.kalshi_fee(0.3, 1000, round_up=False)
    assert mid > ev.kalshi_fee(0.7, 1000, round_up=False)


# --------------------------------------------------------------------------- #
# Edge vs market
# --------------------------------------------------------------------------- #

#: p_home, market price, outcome, closing price. See the test below for the
#: full hand calculation.
EDGE_P = [0.70, 0.40, 0.52, 0.95, 0.30]
EDGE_M = [0.60, 0.55, 0.50, 0.90, 0.45]
EDGE_Y = [True, False, True, True, True]
EDGE_CLOSE = [0.68, 0.50, 0.50, 0.90, 0.45]


def edge_frame() -> pl.DataFrame:
    return frame(
        p_home=EDGE_P,
        market_implied_home=EDGE_M,
        resolved_home_win=EDGE_Y,
        closing_market_implied_home=EDGE_CLOSE,
    )


def test_edge_report_matches_the_hand_computed_bet_ledger() -> None:
    """Five rows, threshold 0.05, bounds [0.15, 0.85], one contract per bet.

    ===  ====  ====  ======  ===============================================
    row     p     m       y  decision
    ===  ====  ====  ======  ===============================================
    1    0.70  0.60  home    edge +0.10 -> BUY HOME at 0.60, wins: +0.40
    2    0.40  0.55  away    edge -0.15 -> BUY AWAY at 0.45, wins: +0.55
    3    0.52  0.50  home    edge +0.02, below threshold -> no bet
    4    0.95  0.90  home    market collapsed (0.90 > 0.85) -> no bet
    5    0.30  0.45  home    edge -0.15 -> BUY AWAY at 0.55, loses: -0.55
    ===  ====  ====  ======  ===============================================

    staked 0.60 + 0.45 + 0.55 = 1.60; gross 0.40; ROI 0.40/1.60 = 0.25.
    """
    report = ev.edge_report(edge_frame(), threshold=0.05, two_sided_fees=False)

    assert report.n_with_market == 5
    assert report.n_bets == 3
    assert report.n_home_bets == 1
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.staked == pytest.approx(1.60)
    assert report.profit_gross == pytest.approx(0.40)
    assert report.roi_gross == pytest.approx(0.25)


def test_brier_skill_score_vs_market_is_hand_computable() -> None:
    """brier_model = 0.9729/5, brier_market = 1.025/5, skill = 1 - ratio."""
    report = ev.edge_report(edge_frame())
    assert report.brier_model == pytest.approx(0.19458)
    assert report.brier_market == pytest.approx(0.205)
    assert report.brier_skill_score == pytest.approx(1.0 - 0.19458 / 0.205)


def test_fees_come_off_the_roi_and_double_when_two_sided() -> None:
    """Fees are 2c per contract at each of 0.60, 0.45 and 0.55 (rounded up)."""
    one = ev.edge_report(edge_frame(), threshold=0.05, two_sided_fees=False)
    two = ev.edge_report(edge_frame(), threshold=0.05, two_sided_fees=True)

    assert one.fees == pytest.approx(0.06)
    assert two.fees == pytest.approx(0.12)
    assert one.profit_net == pytest.approx(0.34)
    assert two.profit_net == pytest.approx(0.28)
    assert two.roi_net == pytest.approx(0.28 / 1.60)
    assert two.roi_gross == one.roi_gross  # fees never touch the gross number


def test_clv_only_counts_bets_whose_price_actually_moved() -> None:
    """Row 1 home +0.08, row 2 away +0.05; row 5's close equals its entry."""
    report = ev.edge_report(edge_frame(), threshold=0.05)
    assert report.n_clv == 2
    assert report.clv == pytest.approx(0.065)


def test_a_higher_threshold_takes_fewer_bets() -> None:
    assert ev.edge_report(edge_frame(), threshold=0.12).n_bets == 2
    assert ev.edge_report(edge_frame(), threshold=0.50).n_bets == 0


def test_the_collapse_gate_is_what_excludes_the_lopsided_row() -> None:
    """Widen the bounds and row 4 becomes a (losing-value) bet."""
    wide = ev.edge_report(edge_frame(), threshold=0.04, bounds=(0.0, 1.0))
    assert wide.n_bets == 4


def test_edge_report_without_bets_still_reports_skill() -> None:
    report = ev.edge_report(edge_frame(), threshold=0.9)
    assert report.n_bets == 0
    assert report.brier_skill_score == pytest.approx(1.0 - 0.19458 / 0.205)
    assert math.isnan(report.roi_gross)


def test_edge_report_on_a_frame_with_no_market_is_empty() -> None:
    report = ev.edge_report(frame(p_home=[0.6], resolved_home_win=[True]))
    assert report.n_with_market == 0


# --------------------------------------------------------------------------- #
# Market join
# --------------------------------------------------------------------------- #


def test_attach_closing_lines_prefers_a_kalshi_snapshot_over_the_close() -> None:
    base = frame(
        p_home=[0.7, 0.7],
        resolved_home_win=[True, True],
        kalshi_yes_price_home=[0.62, None],
        game_id=["g1", "g2"],
    )
    market = pl.DataFrame({"game_id": ["g1", "g2"], "implied_prob_home": [0.55, 0.58]})
    out = ev.attach_closing_lines(base, market)

    # g1 has a live price; g2 falls back to the close.
    assert out.get_column("market_implied_home").to_list() == pytest.approx([0.62, 0.58])
    assert out.get_column("closing_market_implied_home").to_list() == pytest.approx([0.55, 0.58])
    assert out.get_column("edge_vs_market").to_list() == pytest.approx([0.08, 0.12])


def test_attach_closing_lines_leaves_unmatched_games_null() -> None:
    base = frame(p_home=[0.7], resolved_home_win=[True], game_id=["missing"])
    market = pl.DataFrame({"game_id": ["other"], "implied_prob_home": [0.5]})
    out = ev.attach_closing_lines(base, market)
    assert out.get_column("market_implied_home").null_count() == 1


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #


def test_evaluate_frame_reports_phase_q4_and_not_collapsed_slices() -> None:
    n = 40
    rng = np.random.default_rng(0)
    clocks = [3500, 2000, 1200, 800, 200, 30] * 7
    clocks = clocks[:n]
    probs = rng.uniform(0.2, 0.8, n)
    out = frame(
        p_home=list(probs),
        resolved_home_win=[bool(v) for v in rng.integers(0, 2, n)],
        snapshot_game_seconds_remaining=clocks,
        snapshot_period=[4 if c <= 900 else 1 for c in clocks],
        market_implied_home=[0.5] * n,
        game_id=[f"g{i}" for i in range(n)],
    )
    names = [s.name for s in ev.evaluate_frame(out)]

    assert names[0] == "overall"
    assert "Q4 (900-301)" in names
    assert "Q4 last 1min (60-0)" in names
    assert "Q4@600" in names
    assert "not collapsed" in names


def test_q4_marks_take_one_row_per_game_at_or_before_the_mark() -> None:
    out = frame(
        p_home=[0.5, 0.5, 0.5],
        resolved_home_win=[True] * 3,
        snapshot_game_seconds_remaining=[700, 620, 400],
        snapshot_period=[4, 4, 4],
        game_id=["g1", "g1", "g1"],
        prediction_id=["a", "b", "c"],
    )
    at_600 = {s.name: s.metrics.n for s in ev.evaluate_frame(out)}
    # One game, so exactly one row survives the Q4@600 mark (the 620s snapshot).
    assert at_600["Q4@600"] == 1
    assert at_600["Q4@300"] == 1


def test_unresolved_rows_are_excluded_from_every_slice() -> None:
    out = frame(p_home=[0.6, 0.6], resolved_home_win=[True, None])
    assert ev.evaluate_frame(out)[0].metrics.n == 1


# --------------------------------------------------------------------------- #
# Retrain trigger
# --------------------------------------------------------------------------- #


def test_retrain_trigger_fires_on_drift() -> None:
    out = frame(p_home=[0.5] * 10, resolved_home_win=[True, False] * 5)
    signal = ev.retrain_trigger(out, baseline_log_loss=0.20, threshold=0.02)

    assert signal.recent_log_loss == pytest.approx(math.log(2))
    assert signal.drift == pytest.approx(math.log(2) - 0.20)
    assert signal.triggered is True
    assert "drifted" in signal.reason


def test_retrain_trigger_stays_quiet_when_the_model_still_works() -> None:
    out = frame(p_home=[0.99] * 10, resolved_home_win=[True] * 10)
    signal = ev.retrain_trigger(out, baseline_log_loss=0.20, threshold=0.02)
    assert signal.triggered is False
    assert "within tolerance" in signal.reason


def test_retrain_trigger_only_looks_at_the_recent_window() -> None:
    """Old good predictions must not mask a run of recent bad ones."""
    out = frame(
        p_home=[0.99] * 20 + [0.01] * 5,
        resolved_home_win=[True] * 25,
        predicted_at_utc=[f"2024-01-{d:02d}T00:00:00+00:00" for d in range(1, 26)],
    )
    assert ev.retrain_trigger(out, baseline_log_loss=0.02, window=5).triggered is True
    assert ev.retrain_trigger(out, baseline_log_loss=0.02, window=20).triggered is True
    assert ev.retrain_trigger(out.head(20), baseline_log_loss=0.02, window=20).triggered is False


def test_retrain_trigger_says_so_when_there_is_nothing_to_score() -> None:
    signal = ev.retrain_trigger(
        frame(p_home=[0.5], resolved_home_win=[None]), baseline_log_loss=0.2
    )
    assert signal.triggered is False
    assert signal.n == 0


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def test_build_report_scores_each_model_version_separately() -> None:
    out = frame(
        p_home=[0.9, 0.1, 0.6, 0.4],
        resolved_home_win=[True, False, True, False],
        model_name=["gbdt", "gbdt", "mlp", "mlp"],
        model_version=["v1", "v1", "v2", "v2"],
        market_implied_home=[0.5] * 4,
    )
    report = ev.build_report(out, with_market=True)

    assert [(m.model_name, m.model_version) for m in report.models] == [
        ("gbdt", "v1"),
        ("mlp", "v2"),
    ]
    gbdt, mlp = report.models
    # The confident-and-right model must score better than the timid one.
    assert gbdt.slices[0].metrics.log_loss < mlp.slices[0].metrics.log_loss
    assert gbdt.edge is not None and gbdt.edge.brier_skill_score > 0


def test_build_report_says_so_when_nothing_is_resolved() -> None:
    report = ev.build_report(frame(p_home=[0.5], resolved_home_win=[None]), with_market=False)
    assert report.models == []
    assert any("resolve" in note for note in report.notes)


def test_build_report_on_an_empty_ledger_does_not_explode() -> None:
    report = ev.build_report(ledger_mod.empty_ledger(), with_market=False)
    assert report.n_rows == 0
    assert report.notes


def test_build_report_attaches_the_retrain_signal_by_version() -> None:
    out = frame(p_home=[0.5] * 4, resolved_home_win=[True, False, True, False])
    report = ev.build_report(out, with_market=False, baselines={"v1": 0.2})
    assert report.models[0].retrain is not None
    assert report.models[0].retrain.triggered is True


def test_markdown_report_round_trips_to_disk(tmp_path: Path) -> None:
    out = frame(
        p_home=[0.9, 0.1],
        resolved_home_win=[True, False],
        market_implied_home=[0.5, 0.5],
        closing_market_implied_home=[0.5, 0.5],
    )
    report = ev.build_report(out, baselines={"v1": 0.2})
    path = ev.write_markdown(report, tmp_path / "report.md")

    text = path.read_text(encoding="utf-8")
    assert "# Evaluation report" in text
    assert "Edge vs market" in text
    assert "ADR 0001" in text
    assert "Retrain trigger" in text


def test_report_path_lands_under_data_reports(tmp_path: Path) -> None:
    assert ev.report_path("x", tmp_path) == tmp_path / "reports" / "x.md"


def test_json_report_is_machine_readable() -> None:
    import json

    out = frame(p_home=[0.9, 0.1], resolved_home_win=[True, False])
    payload = json.loads(ev.to_json(ev.build_report(out, with_market=False)))
    assert payload["models"][0]["slices"]["overall"]["n"] == 2.0


def test_render_report_prints_every_section() -> None:
    from rich.console import Console

    out = frame(
        p_home=[0.9, 0.1],
        resolved_home_win=[True, False],
        market_implied_home=[0.5, 0.5],
        closing_market_implied_home=[0.6, 0.5],
    )
    console = Console(record=True, width=200)
    ev.render_report(ev.build_report(out, baselines={"v1": 0.2}), console)

    text = console.export_text()
    assert "metrics by slice" in text
    assert "reliability" in text
    assert "edge vs market" in text
    assert "retrain trigger" in text
