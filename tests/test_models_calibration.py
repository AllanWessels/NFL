"""Calibration tests.

The load-bearing claim is the one in the module docstring of
``nfl.models.calibration``: fitting a calibrator on validation predictions makes
the probabilities *honest*. So the central test builds a deliberately
miscalibrated synthetic — a known logit-space distortion of a known true
probability — fits on one half and measures ECE on the other. Held-out, always:
a calibrator trivially "improves" the data it was fit on.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl.models.calibration import (
    CALIBRATORS,
    IdentityCalibrator,
    IsotonicCalibrator,
    PhaseAwareCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    fit_logistic,
    load_calibrator,
    logit,
    make_calibrator,
    sigmoid,
)

SINGLE_MAPS = ["platt", "isotonic", "temperature"]


def _synthetic(
    n: int = 6000, *, slope: float = 1.9, bias: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A model whose ranking is perfect but whose probabilities are distorted.

    ``slope > 1`` is over-confidence (pushes probabilities toward 0 and 1),
    ``bias`` is a systematic lean toward the home team.
    """
    rng = np.random.default_rng(seed)
    true = rng.beta(2.0, 2.0, n)
    y = (rng.random(n) < true).astype(float)
    return sigmoid(slope * logit(true) + bias), y


@pytest.fixture(scope="module")
def overconfident() -> tuple[np.ndarray, np.ndarray]:
    return _synthetic()


def _halves(p: np.ndarray, y: np.ndarray) -> tuple[slice, slice]:
    mid = p.size // 2
    return slice(0, mid), slice(mid, p.size)


# --------------------------------------------------------------------------- #
# The point of the module
# --------------------------------------------------------------------------- #


def test_synthetic_is_actually_miscalibrated(overconfident):
    p, y = overconfident
    assert expected_calibration_error(p, y) > 0.05


@pytest.mark.parametrize("kind", SINGLE_MAPS)
def test_calibrator_reduces_ece_out_of_sample(kind, overconfident):
    p, y = overconfident
    fit, held = _halves(p, y)
    before = expected_calibration_error(p[held], y[held])

    calibrator = make_calibrator(kind).fit(p[fit], y[fit])
    after = expected_calibration_error(calibrator.transform(p[held]), y[held])

    assert after < before / 2.0, f"{kind}: ECE {before:.4f} -> {after:.4f}"
    assert after < 0.05


@pytest.mark.parametrize("kind", SINGLE_MAPS)
def test_calibrator_reduces_ece_when_bias_is_present(kind):
    """Every map helps; only the two-parameter ones remove the bias properly."""
    p, y = _synthetic(slope=1.6, bias=0.7, seed=1)
    fit, held = _halves(p, y)
    before = expected_calibration_error(p[held], y[held])
    after = expected_calibration_error(
        make_calibrator(kind).fit(p[fit], y[fit]).transform(p[held]), y[held]
    )
    assert after < before


def test_platt_beats_temperature_on_a_biased_model():
    """Temperature has no intercept, so a home-team lean survives it."""
    p, y = _synthetic(slope=1.6, bias=0.7, seed=1)
    fit, held = _halves(p, y)
    platt = expected_calibration_error(
        PlattCalibrator().fit(p[fit], y[fit]).transform(p[held]), y[held]
    )
    temperature = expected_calibration_error(
        TemperatureCalibrator().fit(p[fit], y[fit]).transform(p[held]), y[held]
    )
    assert platt < temperature


def test_identity_calibrator_changes_nothing(overconfident):
    p, y = overconfident
    out = IdentityCalibrator().fit(p, y).transform(p)
    assert np.allclose(out, p)


# --------------------------------------------------------------------------- #
# Parameter recovery
# --------------------------------------------------------------------------- #


def test_temperature_recovers_the_known_distortion():
    p, y = _synthetic(n=40000, slope=2.0, seed=3)
    calibrator = TemperatureCalibrator().fit(p, y)
    assert calibrator.temperature == pytest.approx(2.0, abs=0.15)


def test_platt_recovers_the_known_bias():
    p, y = _synthetic(n=40000, slope=1.0, bias=0.8, seed=4)
    calibrator = PlattCalibrator().fit(p, y)
    assert calibrator.a == pytest.approx(1.0, abs=0.12)
    assert calibrator.b == pytest.approx(-0.8, abs=0.15)


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20000, 2))
    truth = np.array([-0.4, 1.5, -0.8])
    y = (rng.random(20000) < sigmoid(truth[0] + x @ truth[1:])).astype(float)
    assert fit_logistic(x, y) == pytest.approx(truth, abs=0.08)


def test_isotonic_transform_is_monotone(overconfident):
    p, y = overconfident
    calibrator = IsotonicCalibrator().fit(p, y)
    grid = np.linspace(0.01, 0.99, 200)
    assert np.all(np.diff(calibrator.transform(grid)) >= 0)


# --------------------------------------------------------------------------- #
# Phase-aware calibration
# --------------------------------------------------------------------------- #


def _phase_synthetic(n: int = 8000, seed: int = 5):
    """Late rows over-confident, early rows under-confident — opposite errors."""
    rng = np.random.default_rng(seed)
    true = rng.beta(2.0, 2.0, n)
    y = (rng.random(n) < true).astype(float)
    seconds = rng.uniform(0.0, 3600.0, n)
    late = seconds < 300.0
    p = np.where(late, sigmoid(2.2 * logit(true)), sigmoid(0.55 * logit(true)))
    return p, y, seconds


def test_phase_aware_beats_a_single_global_map():
    p, y, seconds = _phase_synthetic()
    fit, held = _halves(p, y)

    global_map = PlattCalibrator().fit(p[fit], y[fit])
    phase_map = PhaseAwareCalibrator("platt", edges=(300.0,), min_samples=100).fit(
        p[fit], y[fit], seconds_remaining=seconds[fit]
    )

    raw_ece = expected_calibration_error(p[held], y[held])
    global_ece = expected_calibration_error(global_map.transform(p[held]), y[held])
    phase_ece = expected_calibration_error(
        phase_map.transform(p[held], seconds_remaining=seconds[held]), y[held]
    )
    assert phase_ece < global_ece < raw_ece


def test_thin_buckets_fall_back_to_the_global_fit():
    p, y, seconds = _phase_synthetic(n=1000)
    calibrator = PhaseAwareCalibrator("platt", min_samples=10_000).fit(
        p, y, seconds_remaining=seconds
    )
    assert all(bucket is None for bucket in calibrator.buckets)
    assert np.allclose(
        calibrator.transform(p, seconds_remaining=seconds), calibrator.fallback.transform(p)
    )


def test_rows_with_no_clock_use_the_fallback():
    p, y, seconds = _phase_synthetic(n=2000)
    calibrator = PhaseAwareCalibrator("platt", edges=(300.0,), min_samples=50).fit(
        p, y, seconds_remaining=seconds
    )
    missing = np.full(p.size, np.nan)
    assert np.allclose(
        calibrator.transform(p, seconds_remaining=missing), calibrator.fallback.transform(p)
    )


def test_bucket_boundaries_are_half_open():
    calibrator = PhaseAwareCalibrator("platt", edges=(120.0, 300.0))
    assert calibrator.n_buckets == 3
    assert calibrator.bucket_of([0.0, 119.0, 120.0, 299.0, 300.0, 1e9]).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert calibrator.bucket_label(0) == "[0, 120)"
    assert calibrator.bucket_label(2) == "[300, inf)"


def test_phase_aware_requires_a_clock():
    p, y, seconds = _phase_synthetic(n=800)
    with pytest.raises(ValueError, match="seconds_remaining"):
        PhaseAwareCalibrator("platt").fit(p, y)
    fitted = PhaseAwareCalibrator("platt", min_samples=50).fit(p, y, seconds_remaining=seconds)
    with pytest.raises(ValueError, match="seconds_remaining"):
        fitted.transform(p)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".json", ".npz"])
@pytest.mark.parametrize("kind", ["identity", *SINGLE_MAPS])
def test_save_load_round_trip(kind, suffix, tmp_path, overconfident):
    p, y = overconfident
    calibrator = make_calibrator(kind).fit(p, y)
    path = calibrator.save(tmp_path / f"cal{suffix}")

    restored = load_calibrator(path)
    assert type(restored) is type(calibrator)
    assert np.allclose(restored.transform(p), calibrator.transform(p))


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_phase_aware_round_trip(suffix, tmp_path):
    p, y, seconds = _phase_synthetic(n=4000)
    calibrator = PhaseAwareCalibrator("isotonic", edges=(120.0, 300.0), min_samples=100).fit(
        p, y, seconds_remaining=seconds
    )
    restored = load_calibrator(calibrator.save(tmp_path / f"phase{suffix}"))

    assert isinstance(restored, PhaseAwareCalibrator)
    assert restored.base_kind == "isotonic"
    assert [b is None for b in restored.buckets] == [b is None for b in calibrator.buckets]
    assert np.allclose(
        restored.transform(p, seconds_remaining=seconds),
        calibrator.transform(p, seconds_remaining=seconds),
    )


def test_every_registered_kind_round_trips_by_name():
    assert set(CALIBRATORS) == {"identity", "platt", "isotonic", "temperature", "phase_aware"}


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        PlattCalibrator().transform([0.5])


def test_saving_an_unfitted_calibrator_raises(tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        PlattCalibrator().save(tmp_path / "nope.json")


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        PlattCalibrator().fit([0.1, 0.2], [1.0])


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        make_calibrator("magic")


def test_expected_calibration_error_known_values():
    # Perfectly calibrated: predicted 0.5, half the outcomes land.
    assert expected_calibration_error(np.full(100, 0.5), np.tile([0.0, 1.0], 50)) == 0.0
    # Says 0.9, wins half the time: the gap is exactly 0.4.
    assert expected_calibration_error(np.full(100, 0.9), np.tile([0.0, 1.0], 50)) == pytest.approx(
        0.4
    )
