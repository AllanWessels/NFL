"""Ensemble assembly, confidence selection and the ADR 0003 §5 promotion gate.

Everything here runs against a **tmp registry** stocked with the dummy models
from :mod:`nfl.models.dummy`, never ``data/models``. That is not just hygiene:
the point of :mod:`nfl.models.train_ensemble` is that it talks to members
through the registry and the model contract, so a test that needs XGBoost or
TensorFlow to prove it would be testing the wrong thing. A ``ConstantModel``
and three seeded ``NoisyOracle``s exercise every path the real three members
take — differing feature sets, an MC-dropout head on some members and not
others, save/load round trips — in under a second.

The synthetic dataset is a real assembled-dataset *shape*: ``meta__`` columns,
a ``label__home_win``, plays running down the Q4 clock, several games per
season split. That is what the Q4 slicing and the phase-aware calibrator key
on, so a hand-made numeric frame would not reach them.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nfl.features.splits import SeasonSplit
from nfl.models import registry
from nfl.models.base import FitResult
from nfl.models.data import ModelFrames
from nfl.models.dummy import ConstantModel, NoisyOracle
from nfl.models.ensemble import CONFIDENCE_MODES, StackedEnsemble
from nfl.models.train_ensemble import (
    GATE_METRICS,
    GATE_Q4_SLICES,
    NOT_COLLAPSED_OWN_SLICE,
    NOT_COLLAPSED_SLICE,
    RegistryMember,
    UncertainRegistryMember,
    gate_table,
    load_member,
    member_context,
    promotion_gate,
    q4_row_indices,
    reference_probabilities,
    split_member_spec,
    train_ensemble,
)

FEATURES = ["oracle_signal", "noise_scale", "bi__game_seconds_remaining"]


# --------------------------------------------------------------------------- #
# A model that reports what it was handed
# --------------------------------------------------------------------------- #


class ColumnSpy:
    """Records the columns it was asked to predict from. Importable by name.

    Defined at module scope so the registry can round-trip it: the manifest
    stores ``"tests.test_models_train_ensemble:ColumnSpy"`` and resolves it on
    load, exactly as it would a real model class.
    """

    name = "column_spy"

    def __init__(self, version: str = "") -> None:
        self.version = version
        self.seen: list[str] = []
        self.feature_names: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):  # pragma: no cover - not exercised
        self.feature_names = list(X.columns)
        return FitResult(model_name=self.name, version=self.version, n_train=0, n_val=0)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        self.seen = list(X.columns)
        return np.full(X.height, 0.5)

    def save(self, path):
        from pathlib import Path

        target = Path(path).with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": self.version}))
        return target

    @classmethod
    def load(cls, path):
        from pathlib import Path

        target = Path(path)
        if not target.is_file():
            target = target.with_suffix(".json")
        return cls(version=str(json.loads(target.read_text()).get("version", "")))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _season_frame(season: int, n_games: int, seed: int) -> pl.DataFrame:
    """One season of Q4 snapshots: ``n_games`` games, 30 plays each."""
    rng = np.random.default_rng(seed)
    clock = np.arange(900, 0, -30, dtype=float)  # 30 plays, 900s -> 30s
    rows: list[dict[str, object]] = []
    for g in range(n_games):
        true_p = float(rng.beta(2.0, 2.0))
        won = float(rng.random() < true_p)
        # Difficulty varies by game, which is what makes confidence separable.
        scale = float(rng.choice([0.2, 0.6, 1.5, 3.0]))
        for i, secs in enumerate(clock):
            rows.append(
                {
                    "meta__season": season,
                    "meta__game_id": f"{season}_{g:03d}",
                    "meta__play_id": i,
                    "meta__period": 4,
                    "meta__game_seconds_remaining": float(secs),
                    "label__home_win": won,
                    "oracle_signal": true_p,
                    "noise_scale": scale,
                    "bi__game_seconds_remaining": float(secs),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture
def frames() -> ModelFrames:
    """train 2021 / val 2022 / test 2023, in assembled-dataset shape."""
    return ModelFrames(
        train=_season_frame(2021, 40, seed=1),
        val=_season_frame(2022, 40, seed=2),
        test=_season_frame(2023, 30, seed=3),
        features=list(FEATURES),
    )


def _save(model, root, name: str, version: str, **manifest_extra) -> None:
    manifest = registry.Manifest(
        name=name,
        version=version,
        feature_names=list(manifest_extra.pop("feature_names", [])),
        extra=dict(manifest_extra),
    )
    registry.save_model(model, manifest=manifest, root=root)


@pytest.fixture
def stocked(tmp_path):
    """A tmp registry holding one flat member and two noisy ones."""
    root = tmp_path / "models"
    _save(ConstantModel(p=0.55, name="flat"), root, "flat", "20240101T000000")
    for i, sigma in enumerate((0.7, 1.2)):
        _save(
            NoisyOracle(sigma=sigma, scale_col="noise_scale", seed=i, name=f"oracle{i}"),
            root,
            f"oracle{i}",
            "20240101T000000",
        )
    return root


# --------------------------------------------------------------------------- #
# Member loading
# --------------------------------------------------------------------------- #


def test_load_member_takes_the_newest_version_by_default(stocked):
    _save(ConstantModel(p=0.9, name="flat"), stocked, "flat", "20250101T000000")
    member = load_member("flat", root=stocked)
    assert member.version == "20250101T000000"
    assert member.pin["registry_name"] == "flat"


def test_load_member_honours_a_pin(stocked):
    _save(ConstantModel(p=0.9, name="flat"), stocked, "flat", "20250101T000000")
    member = load_member("flat", "20240101T000000", root=stocked)
    assert member.version == "20240101T000000"
    assert member.predict_proba(pl.DataFrame({"x": [1.0]}))[0] == pytest.approx(0.55)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("flat", ("flat", None)),
        ("flat:20240101T000000", ("flat", "20240101T000000")),
        (" flat : 20240101T000000 ", ("flat", "20240101T000000")),
        ("flat:", ("flat", None)),
    ],
)
def test_member_specs_split_into_name_and_pin(spec, expected):
    assert split_member_spec(spec) == expected


def test_a_pinned_member_spec_reaches_the_registry(stocked, frames, tmp_path):
    """``--members name:version`` is how a reported run stays reproducible."""
    _save(ConstantModel(p=0.9, name="flat"), stocked, "flat", "20250101T000000")
    ensemble, result = _train(
        frames, stocked, tmp_path, members=["flat:20240101T000000", "oracle0", "oracle1"]
    )
    pins = {p["name"]: p["version"] for p in result.extra["members"]}
    assert pins["flat"] == "20240101T000000"


def test_an_unknown_member_says_what_is_available(stocked):
    with pytest.raises(FileNotFoundError, match="oracle0"):
        load_member("does_not_exist", root=stocked)


def test_an_mc_dropout_head_is_advertised_to_the_ensemble(stocked):
    """The ensemble detects uncertainty with ``hasattr``, so the wrapper must too."""
    assert isinstance(load_member("oracle0", root=stocked), UncertainRegistryMember)
    assert isinstance(load_member("flat", root=stocked), RegistryMember)
    assert not hasattr(load_member("flat", root=stocked), "predict_with_uncertainty")


def test_a_member_is_sliced_to_the_columns_its_manifest_records(tmp_path):
    root = tmp_path / "models"
    _save(ColumnSpy(), root, "spy", "20240101T000000", feature_names=["a", "b"])
    member = load_member("spy", root=root)
    member.predict_proba(pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}))
    assert member.loaded.seen == ["a", "b"]


def test_a_manifest_list_the_frame_cannot_satisfy_hands_over_everything(tmp_path):
    """The sequence model records channel names, not dataset columns.

    Slicing to the overlap would silently truncate the frame, which is worse
    than handing over the whole design matrix — every model in this package
    selects the columns it wants by name anyway.
    """
    root = tmp_path / "models"
    _save(ColumnSpy(), root, "spy", "20240101T000000", feature_names=["a", "derived_channel"])
    member = load_member("spy", root=root)
    member.predict_proba(pl.DataFrame({"a": [1.0], "b": [2.0]}))
    assert member.loaded.seen == ["a", "b"]


def test_a_mismatched_row_context_is_an_error(stocked):
    ensemble = StackedEnsemble([load_member("oracle0", root=stocked)])
    frame = pl.DataFrame({"oracle_signal": [0.4, 0.6], "noise_scale": [1.0, 1.0]})
    with member_context(ensemble, frame.head(1)), pytest.raises(ValueError, match="bound context"):
        ensemble.predict_proba(frame)


class SeqLikeMCHead:
    """A member whose MC head windows by game, exactly as ``tf_seq``'s does."""

    name = "seq_like"

    def __init__(self) -> None:
        self.meta_seen_by_uncertainty: pl.DataFrame | None = None

    def predict_proba(self, X: pl.DataFrame, meta: pl.DataFrame | None = None) -> np.ndarray:
        return np.full(X.height, 0.5)

    def predict_with_uncertainty(
        self, X: pl.DataFrame, meta: pl.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        self.meta_seen_by_uncertainty = meta
        return np.full(X.height, 0.5), np.full(X.height, 0.1)


def test_the_mc_head_is_handed_the_same_context_the_point_estimate_gets(frames):
    """Without it the GRU's σ would window across game boundaries."""
    model = SeqLikeMCHead()
    member = UncertainRegistryMember(member="seq_like", version="v0", model=model)
    ensemble = StackedEnsemble([member])
    X = frames.val.select(FEATURES)

    with member_context(ensemble, frames.val):
        ensemble.member_uncertainty(X)
    seen = model.meta_seen_by_uncertainty
    assert seen is not None
    assert seen.height == frames.val.height
    assert "meta__game_id" in seen.columns


def test_a_member_whose_mc_head_takes_no_meta_is_not_handed_one(frames, stocked):
    """``hasattr`` detection must not turn into a signature assumption."""
    member = load_member("oracle0", root=stocked)
    assert isinstance(member, UncertainRegistryMember)
    ensemble = StackedEnsemble([member])
    with member_context(ensemble, frames.val):
        std = ensemble.member_uncertainty(frames.val.select(FEATURES))
    assert std is not None and std.shape[0] == frames.val.height


def test_the_context_is_released_after_the_block(stocked):
    member = load_member("oracle0", root=stocked)
    ensemble = StackedEnsemble([member])
    with member_context(ensemble, pl.DataFrame({"x": [1.0]})):
        assert member.context is not None
    assert member.context is None


# --------------------------------------------------------------------------- #
# Q4 row selection
# --------------------------------------------------------------------------- #


def test_q4_row_indices_pick_one_row_per_game_per_mark(frames):
    rows = q4_row_indices(frames.val)
    picked = frames.val[rows.tolist()]
    games = frames.val.get_column("meta__game_id").n_unique()
    # Three marks, one row per game per mark, no duplicates.
    assert rows.size == 3 * games
    assert set(picked.get_column("meta__game_seconds_remaining").unique()) == {600.0, 300.0, 120.0}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _train(frames, stocked, tmp_path, **kwargs):
    return train_ensemble(
        members=kwargs.pop("members", ["flat", "oracle0", "oracle1"]),
        frames=frames,
        models_root=stocked,
        reports_root=tmp_path / "data",
        split=SeasonSplit(train_through=2021, val_season=2022, test_from=2023),
        name=kwargs.pop("name", "ensemble_test"),
        phase_min_samples=20,
        **kwargs,
    )


def test_train_ensemble_writes_a_loadable_artifact(frames, stocked, tmp_path):
    ensemble, result = _train(frames, stocked, tmp_path)

    assert result.model_name == "ensemble_test"
    versions = registry.list_versions("ensemble_test", stocked)
    assert versions == [result.version]

    reloaded = registry.load("ensemble_test", result.version, stocked)
    assert isinstance(reloaded, StackedEnsemble)
    assert reloaded.member_names == ensemble.member_names
    assert reloaded.confidence_mode == ensemble.confidence_mode

    X = frames.val.select(FEATURES)
    with member_context(ensemble, frames.val), member_context(reloaded, frames.val):
        assert reloaded.predict_proba(X) == pytest.approx(ensemble.predict_proba(X))


def test_members_are_pinned_by_name_and_version(frames, stocked, tmp_path):
    _, result = _train(frames, stocked, tmp_path)
    manifest = registry.read_manifest(registry.model_dir("ensemble_test", result.version, stocked))
    pins = {p["name"]: p["version"] for p in manifest.extra["members"]}
    assert pins == {
        "flat": "20240101T000000",
        "oracle0": "20240101T000000",
        "oracle1": "20240101T000000",
    }


def test_the_calibrator_is_fitted_on_val_and_applied(frames, stocked, tmp_path):
    ensemble, _ = _train(frames, stocked, tmp_path)
    assert ensemble.calibrator is not None
    assert ensemble.calibrator.kind == "phase_aware"
    assert ensemble.calibrator.fitted

    X = frames.val.select(FEATURES)
    with member_context(ensemble, frames.val):
        raw = ensemble.combine(ensemble.member_probabilities(X))
        calibrated = ensemble.predict_proba(X)
    assert not np.allclose(raw, calibrated), "calibrator made no difference to the output"


def test_calibration_improves_log_loss_on_val(frames, stocked, tmp_path):
    """The calibrator is fit on these rows, so it had better help on them."""
    ensemble, _ = _train(frames, stocked, tmp_path)
    X = frames.val.select(FEATURES)
    y = frames.val.get_column("label__home_win").to_numpy()
    with member_context(ensemble, frames.val):
        raw = ensemble.combine(ensemble.member_probabilities(X))
        calibrated = ensemble.predict_proba(X)

    def ll(p):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    assert ll(calibrated) < ll(raw)


def test_calibration_can_be_turned_off(frames, stocked, tmp_path):
    ensemble, _ = _train(frames, stocked, tmp_path, calibrator="none", name="ensemble_raw")
    assert ensemble.calibrator is None


def test_slices_cover_overall_q4_marks_and_both_not_collapsed_variants(frames, stocked, tmp_path):
    _, result = _train(frames, stocked, tmp_path)
    names = [row["slice"] for row in result.extra["val_slices"]]
    assert names == [
        "overall",
        *GATE_Q4_SLICES,
        NOT_COLLAPSED_SLICE,
        NOT_COLLAPSED_OWN_SLICE,
    ]
    assert [row["slice"] for row in result.extra["test_slices"]] == names


def test_the_confidence_definition_is_chosen_and_recorded(frames, stocked, tmp_path):
    _, result = _train(frames, stocked, tmp_path)
    mode = result.extra["confidence_mode"]
    assert mode in set(CONFIDENCE_MODES)
    ranked = result.extra["confidence_ranking"]
    # Two of three members expose an MC head, so every definition competes:
    # spread, MC σ, calibrated margin, and the two compounds.
    assert {row["confidence"] for row in ranked} == {
        "disagreement",
        "mc_dropout",
        "blend",
        "margin",
        "margin_mc",
    }
    assert result.extra["confidence_n_val_q4_rows"] > 0


def test_the_chosen_confidence_is_the_best_ranked_one(frames, stocked, tmp_path):
    """The winner is the ranking's first row — most negative spearman wins."""
    _, result = _train(frames, stocked, tmp_path)
    labels = {
        "disagreement": "disagreement",
        "uncertainty": "mc_dropout",
        "combined": "blend",
        "margin": "margin",
        "margin_uncertainty": "margin_mc",
    }
    ranked = result.extra["confidence_ranking"]
    assert labels[result.extra["confidence_mode"]] == ranked[0]["confidence"]
    spearmans = [row["spearman"] for row in ranked]
    assert spearmans == sorted(spearmans)
    assert result.extra["confidence_spearman"] == pytest.approx(ranked[0]["spearman"])


def test_margin_confidence_needs_the_calibrated_probability(frames, stocked, tmp_path):
    """A margin mode cannot invent ``p``; it says so rather than guessing."""
    ensemble, _ = _train(frames, stocked, tmp_path)
    ensemble.confidence_mode = "margin"
    with pytest.raises(ValueError, match="calibrated probability"):
        ensemble.confidence_from(np.array([[0.4, 0.6], [0.5, 0.5]]))


def test_margin_confidence_is_the_distance_from_a_coin_flip(frames, stocked, tmp_path):
    ensemble, _ = _train(frames, stocked, tmp_path)
    ensemble.confidence_mode = "margin"
    member_probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    conf, signals = ensemble.confidence_from(member_probs, None, np.array([0.5, 0.75, 1.0]))
    assert conf == pytest.approx([0.0, 0.5, 1.0])
    assert signals["margin"] == pytest.approx([0.0, 0.5, 1.0])


def test_a_forced_confidence_definition_wins(frames, stocked, tmp_path):
    ensemble, result = _train(frames, stocked, tmp_path, confidence="mc_dropout")
    assert ensemble.confidence_mode == "uncertainty"
    assert result.extra["confidence_mode"] == "uncertainty"


def test_an_unknown_confidence_definition_is_rejected(frames, stocked, tmp_path):
    with pytest.raises(ValueError, match="unknown confidence"):
        _train(frames, stocked, tmp_path, confidence="vibes")


def test_the_stacker_meta_learner_records_its_weights(frames, stocked, tmp_path):
    ensemble, result = _train(
        frames, stocked, tmp_path, meta="logistic_stacker", name="ensemble_stack"
    )
    assert ensemble.weights is not None
    assert len(ensemble.weights) == 4  # intercept + one per member
    assert result.extra["meta"] == "logistic_stacker"


def test_an_empty_validation_split_is_a_clear_error(frames, stocked, tmp_path):
    empty = ModelFrames(
        train=frames.train, val=frames.val.head(0), test=frames.test, features=list(FEATURES)
    )
    with pytest.raises(ValueError, match="fit on val"):
        _train(empty, stocked, tmp_path)


def test_a_markdown_report_is_written(frames, stocked, tmp_path):
    _, result = _train(frames, stocked, tmp_path)
    path = tmp_path / "data" / "reports" / f"ensemble-{result.version}.md"
    assert path.is_file()
    text = path.read_text()
    assert "## Ensemble" in text
    assert "Promotion gate" in text
    for member in ("flat", "oracle0", "oracle1"):
        assert member in text


def test_the_report_is_skipped_when_asked(frames, stocked, tmp_path):
    _train(frames, stocked, tmp_path, report=False)
    assert not (tmp_path / "data" / "reports").exists()


# --------------------------------------------------------------------------- #
# Promotion gate
# --------------------------------------------------------------------------- #


def _slices(log_loss: float, brier: float, not_collapsed: float = 0.6) -> list[dict[str, float]]:
    rows = [{"slice": "overall", "log_loss": log_loss, "brier": brier}]
    rows += [{"slice": s, "log_loss": log_loss, "brier": brier} for s in GATE_Q4_SLICES]
    rows.append(
        {"slice": NOT_COLLAPSED_SLICE, "log_loss": not_collapsed, "brier": not_collapsed / 3.0}
    )
    return rows


def _floor_and_candidate(root, *, candidate_slices, floor_slices=None, flat_manifest=False):
    """Stock a registry with a floor and a candidate carrying val slices."""
    floor = floor_slices if floor_slices is not None else _slices(0.30, 0.10)
    if flat_manifest:
        # The shape `nfl.models.artifacts.write_manifest` writes: val_slices at
        # the top level rather than under `extra`. Both must gate identically.
        directory = registry.model_dir("naive_score_time", "20240101T000000", root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {"name": "naive_score_time", "version": "20240101T000000", "val_slices": floor}
            )
        )
    else:
        _save(
            ConstantModel(p=0.5, name="naive_score_time"),
            root,
            "naive_score_time",
            "20240101T000000",
            val_slices=floor,
        )
    _save(
        ConstantModel(p=0.5, name="cand"),
        root,
        "cand",
        "20240102T000000",
        val_slices=candidate_slices,
    )


def _floor_and_candidate_with_splits(root, *, floor_split, candidate_split):
    """A clearly-winning candidate, so only the split condition can fail it."""
    _save(
        ConstantModel(p=0.5, name="naive_score_time"),
        root,
        "naive_score_time",
        "20240101T000000",
        val_slices=_slices(0.30, 0.10),
        split=floor_split.to_dict(),
    )
    _save(
        ConstantModel(p=0.5, name="cand"),
        root,
        "cand",
        "20240102T000000",
        val_slices=_slices(0.25, 0.08, not_collapsed=0.55),
        split=candidate_split.to_dict(),
        not_collapsed_reference="naive_score_time/20240101T000000",
    )


def test_the_gate_passes_the_split_check_when_both_sides_agree(tmp_path):
    root = tmp_path / "models"
    split = SeasonSplit.rolling(2025)
    _floor_and_candidate_with_splits(root, floor_split=split, candidate_split=split)
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert result.passed
    check = next(c for c in result.checks if c.slice == "split")
    assert check.passed and check.rule_label == "same split"


def test_the_gate_refuses_a_candidate_validated_on_a_different_season(tmp_path):
    """Two models measured on different football cannot be subtracted."""
    root = tmp_path / "models"
    _floor_and_candidate_with_splits(
        root,
        floor_split=SeasonSplit.rolling(2024),
        candidate_split=SeasonSplit.rolling(2025),
    )
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert not result.passed
    failed = {c.slice for c in result.checks if not c.passed}
    assert failed == {"split"}
    assert any("different rows" in note for note in result.notes)


def test_an_unrecorded_split_is_noted_rather_than_failed(tmp_path):
    """Unprovable is not the same as mismatched — old artifacts stay gateable."""
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert result.passed
    assert not any(c.slice == "split" for c in result.checks)
    assert any("no season split recorded" in note for note in result.notes)


def test_gate_passes_when_the_candidate_beats_the_floor_everywhere(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert result.passed
    assert result.floor == "naive_score_time"
    assert len(result.checks) == len(GATE_Q4_SLICES) * len(GATE_METRICS) + len(GATE_METRICS)


def test_gate_reads_a_flat_manifest_the_same_way(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(
        root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55), flat_manifest=True
    )
    assert promotion_gate("cand", root=root, parity_samples=0).passed


def test_gate_fails_when_brier_regresses_even_though_log_loss_improves(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.12, not_collapsed=0.55))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert not result.passed
    failed = {(c.slice, c.metric) for c in result.checks if not c.passed}
    assert failed == {(s, "brier") for s in GATE_Q4_SLICES}


def test_gate_fails_when_not_collapsed_regresses(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.65))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert not result.passed
    assert {c.metric for c in result.checks if not c.passed} == set(GATE_METRICS)
    assert {c.slice for c in result.checks if not c.passed} == {NOT_COLLAPSED_SLICE}


def test_not_collapsed_only_has_to_be_no_worse(tmp_path):
    """Q4 must *improve*; not-collapsed only must not degrade."""
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.60))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert result.passed
    tie = [c for c in result.checks if c.slice == NOT_COLLAPSED_SLICE]
    assert all(c.delta == pytest.approx(0.0) for c in tie)
    assert all(not c.strict for c in tie)


def test_a_tie_on_a_q4_metric_is_not_good_enough(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.30, 0.10, not_collapsed=0.55))
    assert not promotion_gate("cand", root=root, parity_samples=0).passed


def test_a_missing_slice_fails_rather_than_being_skipped(tmp_path):
    root = tmp_path / "models"
    partial = [row for row in _slices(0.25, 0.08) if row["slice"] != "q4_120s"]
    _floor_and_candidate(root, candidate_slices=partial)
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert not result.passed
    assert any("q4_120s" in note for note in result.notes)


def test_a_manifest_without_val_slices_explains_itself(tmp_path):
    root = tmp_path / "models"
    _save(
        ConstantModel(p=0.5, name="naive_score_time"), root, "naive_score_time", "20240101T000000"
    )
    _save(ConstantModel(p=0.5, name="cand"), root, "cand", "20240102T000000")
    with pytest.raises(ValueError, match="no val_slices"):
        promotion_gate("cand", root=root, parity_samples=0)


def test_the_gate_table_shows_both_sides_and_the_verdict(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55))
    table = gate_table(promotion_gate("cand", root=root, parity_samples=0))
    assert table.columns == [
        "slice",
        "metric",
        "candidate",
        "floor",
        "delta",
        "rule",
        "verdict",
    ]
    assert set(table.get_column("verdict").unique()) == {"pass"}
    assert (table.get_column("delta") < 0).all()


def test_the_trainer_keys_not_collapsed_on_the_floors_band(frames, stocked, tmp_path):
    """The gated slice counts the *floor's* live rows, not the ensemble's."""
    _save(
        NoisyOracle(sigma=0.9, scale_col="noise_scale", seed=9, name="naive_score_time"),
        stocked,
        "naive_score_time",
        "20240101T000000",
    )
    _, result = _train(frames, stocked, tmp_path)

    floor_p, pin = reference_probabilities(frames.val, frames.features, root=stocked)
    expected = int(((floor_p >= 0.15) & (floor_p <= 0.85)).sum())
    rows = {row["slice"]: row for row in result.extra["val_slices"]}
    assert rows[NOT_COLLAPSED_SLICE]["n"] == expected
    assert result.extra["not_collapsed_reference"] == pin == "naive_score_time/20240101T000000"
    # The own-probability variant is still reported, and is a different sample.
    assert rows[NOT_COLLAPSED_OWN_SLICE]["n"] != rows[NOT_COLLAPSED_SLICE]["n"]


def test_candidate_and_floor_are_gated_on_the_same_row_count(frames, stocked, tmp_path):
    """The floor's own band *is* the reference band, so both sides count alike."""
    from nfl.models.metrics import slice_metrics

    floor_model = NoisyOracle(sigma=0.9, scale_col="noise_scale", seed=9, name="naive_score_time")
    floor_p = floor_model.predict_proba(frames.val.select(frames.features))
    floor_table = slice_metrics(frames.val, floor_p, frames.val.get_column("label__home_win"))
    _save(floor_model, stocked, "naive_score_time", "20240101T000000")

    _, result = _train(frames, stocked, tmp_path)
    candidate = {row["slice"]: row for row in result.extra["val_slices"]}
    floor = {row["slice"]: row for row in floor_table.to_dicts()}
    assert candidate[NOT_COLLAPSED_SLICE]["n"] == floor[NOT_COLLAPSED_SLICE]["n"]


def test_reference_probabilities_falls_back_when_the_floor_is_missing(frames, stocked):
    """A registry with no floor is a warning and a ``None``, never a crash."""
    probs, pin = reference_probabilities(frames.val, frames.features, root=stocked)
    assert probs is None
    assert pin is None


def test_the_gate_notes_a_candidate_scored_against_another_floor(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55))
    directory = registry.model_dir("cand", "20240102T000000", root)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest.setdefault("extra", {})["not_collapsed_reference"] = "naive_score_time/19990101T000000"
    (directory / "manifest.json").write_text(json.dumps(manifest))

    result = promotion_gate("cand", root=root, parity_samples=0)
    assert any("19990101T000000" in note for note in result.notes)


def test_the_gate_notes_a_candidate_that_recorded_no_reference(tmp_path):
    root = tmp_path / "models"
    _floor_and_candidate(root, candidate_slices=_slices(0.25, 0.08, not_collapsed=0.55))
    result = promotion_gate("cand", root=root, parity_samples=0)
    assert any("no not_collapsed_reference" in note for note in result.notes)


def test_a_trained_ensemble_can_be_gated(frames, stocked, tmp_path):
    """End to end: the manifest the trainer wrote is the manifest the gate reads."""
    _save(
        ConstantModel(p=0.5, name="naive_score_time"),
        stocked,
        "naive_score_time",
        "20240101T000000",
        val_slices=_slices(0.9, 0.3, not_collapsed=0.9),
    )
    _, result = _train(frames, stocked, tmp_path)
    gate = promotion_gate("ensemble_test", result.version, root=stocked, parity_samples=0)
    assert gate.candidate_version == result.version
    assert gate.passed, gate_table(gate)
