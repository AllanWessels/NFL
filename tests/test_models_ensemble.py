"""Ensemble and confidence tests.

Two things are worth testing here and they are quite different in character.

The **blending math** is exact: with constant members the arithmetic mean, the
logit mean and the stacker all have closed forms, so those tests assert numbers,
not tendencies.

The **confidence claim** is statistical, and it is the one that matters: low
confidence must mean high realised error. The synthetic below is built so that
error is a known function of a per-row noise knob — members see the true
probability through noise scaled by that knob — which makes member disagreement
a genuine proxy for difficulty. If :func:`confidence_reliability` cannot recover
that ordering on data where it exists by construction, the metric is broken.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl.models.base import WinProbModel
from nfl.models.calibration import PhaseAwareCalibrator, PlattCalibrator, logit, sigmoid
from nfl.models.dummy import ConstantModel, NoisyOracle
from nfl.models.ensemble import (
    StackedEnsemble,
    compare_confidence_definitions,
    confidence_reliability,
)


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame({"oracle_signal": [0.3, 0.5, 0.7], "x": [1.0, 2.0, 3.0]})


def _graded_synthetic(n: int = 8000, seed: int = 0):
    """Rows whose difficulty is set by a known per-row noise multiplier."""
    rng = np.random.default_rng(seed)
    true = rng.beta(2.0, 2.0, n)
    y = (rng.random(n) < true).astype(float)
    scale = rng.choice([0.15, 0.6, 1.4, 3.0], size=n)
    frame = pl.DataFrame({"oracle_signal": true, "noise_scale": scale})
    members = [
        NoisyOracle(sigma=0.8, scale_col="noise_scale", seed=s, name=f"m{s}") for s in range(5)
    ]
    return frame, y, members


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model", [ConstantModel(), NoisyOracle(), StackedEnsemble([ConstantModel()])]
)
def test_implements_the_model_contract(model):
    assert isinstance(model, WinProbModel)


def test_empty_membership_is_rejected():
    with pytest.raises(ValueError, match="at least one member"):
        StackedEnsemble(members=[])


def test_unknown_meta_is_rejected():
    with pytest.raises(ValueError, match="unknown meta"):
        StackedEnsemble(members=[ConstantModel()], meta="median")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Blending math
# --------------------------------------------------------------------------- #


def test_mean_is_the_arithmetic_average(frame):
    ensemble = StackedEnsemble([ConstantModel(0.2), ConstantModel(0.6)], meta="mean")
    assert ensemble.predict_proba(frame) == pytest.approx(np.full(3, 0.4))


def test_logit_mean_is_the_geometric_average_in_odds(frame):
    ensemble = StackedEnsemble([ConstantModel(0.2), ConstantModel(0.6)], meta="logit_mean")
    expected = sigmoid((logit(0.2) + logit(0.6)) / 2.0)
    assert ensemble.predict_proba(frame) == pytest.approx(np.full(3, float(expected[0])))


def test_logit_mean_and_mean_agree_on_symmetric_members(frame):
    members = [ConstantModel(0.2), ConstantModel(0.8)]
    assert StackedEnsemble(members, meta="mean").predict_proba(frame) == pytest.approx(
        StackedEnsemble(members, meta="logit_mean").predict_proba(frame)
    )


def test_weights_are_normalised(frame):
    ensemble = StackedEnsemble(
        [ConstantModel(0.2), ConstantModel(0.6)], meta="mean", weights=[3.0, 1.0]
    )
    assert ensemble.predict_proba(frame) == pytest.approx(np.full(3, 0.3))


def test_wrong_weight_count_is_rejected(frame):
    ensemble = StackedEnsemble([ConstantModel(0.2)], meta="mean", weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="weights for"):
        ensemble.predict_proba(frame)


def test_stacker_without_fit_refuses_to_predict(frame):
    with pytest.raises(RuntimeError, match="call fit"):
        StackedEnsemble([ConstantModel(0.4)], meta="logistic_stacker").predict_proba(frame)


def test_stacker_downweights_the_noisy_member():
    """A stacker fit on validation should prefer the member that is actually right."""
    frame, y, _ = _graded_synthetic(n=6000, seed=2)
    good = NoisyOracle(sigma=0.2, seed=1, name="good")
    noisy = NoisyOracle(sigma=2.5, seed=2, name="noisy")

    stacked = StackedEnsemble([good, noisy], meta="logistic_stacker")
    stacked_ll = stacked.fit(frame, y, frame, y).metrics["log_loss"]
    averaged_ll = (
        StackedEnsemble([good, noisy], meta="logit_mean")
        .fit(frame, y, frame, y)
        .metrics["log_loss"]
    )

    assert stacked.weights is not None
    intercept, w_good, w_noisy = stacked.weights
    assert w_good > w_noisy
    assert stacked_ll < averaged_ll


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def test_unanimous_members_yield_full_confidence(frame):
    ensemble = StackedEnsemble([ConstantModel(0.7), ConstantModel(0.7)])
    _, confidence, components = ensemble.predict_with_confidence(frame)
    assert confidence == pytest.approx(np.ones(3))
    assert components["logit_std"] == pytest.approx(np.zeros(3))


def test_opposed_members_yield_zero_confidence(frame):
    ensemble = StackedEnsemble([ConstantModel(0.05), ConstantModel(0.95)])
    _, confidence, _ = ensemble.predict_with_confidence(frame)
    assert confidence == pytest.approx(np.zeros(3))


def test_a_lone_member_cannot_disagree_with_itself(frame):
    """Documented degenerate case: one member means confidence is always 1."""
    _, confidence, _ = StackedEnsemble([ConstantModel(0.7)]).predict_with_confidence(frame)
    assert confidence == pytest.approx(np.ones(3))


def test_confidence_ranks_rows_by_their_known_difficulty():
    frame, y, members = _graded_synthetic()
    ensemble = StackedEnsemble(members)
    _, confidence, _ = ensemble.predict_with_confidence(frame)
    scale = frame.get_column("noise_scale").to_numpy()

    means = [float(confidence[scale == s].mean()) for s in sorted(np.unique(scale))]
    assert means == sorted(means, reverse=True), means


def test_confidence_reliability_brier_falls_as_confidence_rises():
    frame, y, members = _graded_synthetic()
    p, confidence, _ = StackedEnsemble(members).predict_with_confidence(frame)

    coarse = confidence_reliability(p, confidence, y, n_bins=3)
    brier = coarse.get_column("brier").to_numpy()
    assert np.all(np.diff(brier) < 0), brier
    assert brier[0] - brier[-1] > 0.02

    # Quantile edges collapse where confidence ties (many rows sit on the
    # clipped floor of 0), so ten requested bins can come back as fewer — but
    # every row must land in exactly one of them.
    fine = confidence_reliability(p, confidence, y, n_bins=10)
    assert 5 <= fine.height <= 10
    assert fine.get_column("n").sum() == len(y)
    assert fine.get_column("conf_mean").is_sorted()


def test_compare_confidence_definitions_prefers_the_real_signal():
    frame, y, members = _graded_synthetic()
    p, confidence, components = StackedEnsemble(members).predict_with_confidence(frame)
    rng = np.random.default_rng(11)

    ranked = compare_confidence_definitions(
        p,
        y,
        {
            "disagreement": confidence,
            "coin_flip": rng.random(len(y)),
            "inverted": -confidence,
        },
    )
    assert ranked.get_column("confidence")[0] == "disagreement"
    assert ranked.get_column("spearman")[0] < -0.6
    assert ranked.get_column("brier_spread")[0] > 0.02


def test_uncertainty_mode_needs_a_member_that_reports_uncertainty(frame):
    ensemble = StackedEnsemble(
        [ConstantModel(0.4), ConstantModel(0.6)], confidence_mode="uncertainty"
    )
    with pytest.raises(ValueError, match="predict_with_uncertainty"):
        ensemble.predict_with_confidence(frame)


def test_combined_mode_falls_back_to_disagreement_without_uncertainty(frame):
    members = [ConstantModel(0.4), ConstantModel(0.6)]
    combined, _, _ = StackedEnsemble(members, confidence_mode="combined").predict_with_confidence(
        frame
    )
    plain, _, _ = StackedEnsemble(members).predict_with_confidence(frame)
    assert combined == pytest.approx(plain)


def test_uncertainty_mode_uses_the_mc_head():
    frame, y, members = _graded_synthetic(n=2000, seed=3)
    ensemble = StackedEnsemble(members, confidence_mode="uncertainty")
    _, confidence, components = ensemble.predict_with_confidence(frame)

    assert "mc_std" in components
    scale = frame.get_column("noise_scale").to_numpy()
    assert float(components["mc_std"][scale == 3.0].mean()) > float(
        components["mc_std"][scale == 0.15].mean()
    )
    assert float(confidence[scale == 3.0].mean()) < float(confidence[scale == 0.15].mean())


def test_components_expose_every_member(frame):
    ensemble = StackedEnsemble([ConstantModel(0.3, name="a"), ConstantModel(0.9, name="b")])
    p, _, components = ensemble.predict_with_confidence(frame)
    assert components["member__a"] == pytest.approx(np.full(3, 0.3))
    assert components["member__b"] == pytest.approx(np.full(3, 0.9))
    assert components["calibrated"] == pytest.approx(p)
    assert components["raw"] == pytest.approx(p)  # no calibrator attached


# --------------------------------------------------------------------------- #
# Calibration inside the ensemble
# --------------------------------------------------------------------------- #


def test_calibrator_is_applied_to_the_combined_probability():
    frame, y, members = _graded_synthetic(n=4000, seed=4)
    ensemble = StackedEnsemble(members, calibrator=PlattCalibrator())
    ensemble.fit(frame, y, frame, y)

    p, _, components = ensemble.predict_with_confidence(frame)
    assert not np.allclose(components["raw"], p)
    assert p == pytest.approx(ensemble.calibrator.transform(components["raw"]))


def test_phase_aware_calibrator_finds_the_clock_column():
    frame, y, members = _graded_synthetic(n=6000, seed=5)
    rng = np.random.default_rng(6)
    frame = frame.with_columns(
        pl.Series("ig__game_seconds_remaining", rng.uniform(0, 3600, frame.height))
    )
    ensemble = StackedEnsemble(
        members, calibrator=PhaseAwareCalibrator("platt", edges=(300.0,), min_samples=100)
    )
    result = ensemble.fit(frame, y, frame, y)

    assert result.extra["calibrator"] == "phase_aware"
    assert ensemble.predict_proba(frame).shape == (frame.height,)


def test_phase_aware_calibrator_without_a_clock_is_an_error():
    frame, y, members = _graded_synthetic(n=2000, seed=7)
    ensemble = StackedEnsemble(members, calibrator=PhaseAwareCalibrator("platt", min_samples=50))
    with pytest.raises(ValueError, match="game clock"):
        ensemble.fit(frame, y, frame, y)


# --------------------------------------------------------------------------- #
# Fit bookkeeping and persistence
# --------------------------------------------------------------------------- #


def test_fit_records_members_and_where_the_meta_learner_was_fit():
    frame, y, members = _graded_synthetic(n=3000, seed=8)
    ensemble = StackedEnsemble(members[:2], meta="logistic_stacker")
    result = ensemble.fit(frame, y, frame, y)

    assert result.model_name == "stacked_ensemble"
    assert result.version
    assert result.n_val == len(y)
    assert result.extra["meta_fit_on"] == "val"
    assert result.extra["members"] == ["m0", "m1"]
    assert set(result.metrics) >= {"log_loss", "brier", "accuracy", "ece", "auc"}
    assert "member_log_loss__m0" in result.metrics

    train_only = StackedEnsemble(members[:2]).fit(frame, y)
    assert train_only.extra["meta_fit_on"] == "train"
    assert train_only.n_val == 0


def test_save_load_round_trip_preserves_predictions(tmp_path):
    frame, y, members = _graded_synthetic(n=3000, seed=9)
    ensemble = StackedEnsemble(
        members[:3], meta="logistic_stacker", calibrator=PlattCalibrator(), name="ens"
    )
    ensemble.fit(frame, y, frame, y)
    before, conf_before, _ = ensemble.predict_with_confidence(frame)

    restored = StackedEnsemble.load(ensemble.save(tmp_path / "ens"))
    after, conf_after, _ = restored.predict_with_confidence(frame)

    assert restored.member_names == ensemble.member_names
    assert restored.meta == "logistic_stacker"
    assert restored.weights == pytest.approx(ensemble.weights)
    assert isinstance(restored.calibrator, PlattCalibrator)
    assert after == pytest.approx(before)
    assert conf_after == pytest.approx(conf_before)
