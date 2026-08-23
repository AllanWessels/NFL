"""Registry tests — all against ``tmp_path``, never the real ``data/models``.

The registry's job is that an artifact trained today is still loadable, and
still *explicable*, months later. So the tests check the round trip end to end
(save a model, list it, load it, predict) and check the provenance fields that
make an artifact explicable: the dataset fingerprint must move when the dataset
changes and stay put when it does not.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from nfl.models.base import FitResult
from nfl.models.calibration import PlattCalibrator
from nfl.models.dummy import ConstantModel, NoisyOracle
from nfl.models.ensemble import StackedEnsemble
from nfl.models.registry import (
    Manifest,
    dataset_fingerprint,
    git_sha,
    latest,
    list_models,
    list_versions,
    load,
    load_stage,
    model_dir,
    pointer_path,
    promote,
    read_manifest,
    read_pointer,
    resolve_artifact,
    save_model,
    utc_version,
    write_manifest,
)


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame({"oracle_signal": [0.2, 0.5, 0.8], "x": [1.0, 2.0, 3.0]})


@pytest.fixture
def dataset() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta__season": [2022, 2022, 2023],
            "label__home_win": [1.0, 0.0, 1.0],
            "bi__score_diff": [3.0, -7.0, 0.0],
            "bi__game_seconds_remaining": [900.0, 300.0, 120.0],
        }
    )


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_save_then_load_returns_an_equivalent_model(tmp_path, frame):
    model = ConstantModel(p=0.62, name="constant")
    manifest = save_model(model, root=tmp_path, version="20240101T000000", train_seasons=[2021])

    assert (model_dir("constant", "20240101T000000", tmp_path) / "manifest.json").is_file()
    assert manifest.model_class == "nfl.models.dummy:ConstantModel"
    assert manifest.git_sha
    assert manifest.created_at

    restored = load("constant", "20240101T000000", tmp_path)
    assert restored.predict_proba(frame) == pytest.approx(model.predict_proba(frame))


def test_listing_and_latest(tmp_path):
    for version in ("20240101T000000", "20240301T120000", "20240201T000000"):
        save_model(ConstantModel(p=0.5), name="gbdt", version=version, root=tmp_path)
    save_model(ConstantModel(p=0.5), name="mlp", version="20240101T000000", root=tmp_path)

    assert list_models(tmp_path) == ["gbdt", "mlp"]
    assert list_versions("gbdt", tmp_path) == [
        "20240101T000000",
        "20240201T000000",
        "20240301T120000",
    ]
    assert latest("gbdt", tmp_path) == "20240301T120000"


def test_load_defaults_to_the_latest_version(tmp_path, frame):
    save_model(ConstantModel(p=0.1), name="m", version="20240101T000000", root=tmp_path)
    save_model(ConstantModel(p=0.9), name="m", version="20240202T000000", root=tmp_path)
    assert load("m", root=tmp_path).predict_proba(frame) == pytest.approx(np.full(3, 0.9))


def test_missing_model_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no versions"):
        latest("ghost", tmp_path)
    assert list_models(tmp_path) == []
    assert list_versions("ghost", tmp_path) == []


def test_directory_without_a_manifest_is_not_a_version(tmp_path):
    (tmp_path / "half_written" / "20240101T000000").mkdir(parents=True)
    assert list_models(tmp_path) == []


def test_artifact_suffix_added_by_the_model_is_recorded(tmp_path):
    """``ConstantModel.save`` appends ``.json``; the manifest must follow it."""
    manifest = save_model(ConstantModel(), name="c", version="20240101T000000", root=tmp_path)
    assert manifest.artifact == "model.json"
    assert (
        resolve_artifact(model_dir("c", "20240101T000000", tmp_path), "model").name == "model.json"
    )


def test_load_without_model_class_explains_itself(tmp_path):
    directory = model_dir("orphan", "20240101T000000", tmp_path)
    write_manifest(Manifest(name="orphan", version="20240101T000000"), directory)
    with pytest.raises(ValueError, match="no model_class"):
        load("orphan", "20240101T000000", tmp_path)


def test_an_ensemble_round_trips_through_the_registry(tmp_path, frame):
    """The ensemble's artifact is a *directory*; the registry must not care."""
    members = [NoisyOracle(sigma=0.3, seed=0, name="a"), NoisyOracle(sigma=0.9, seed=1, name="b")]
    ensemble = StackedEnsemble(members, calibrator=PlattCalibrator(), name="ens")
    y = np.array([1.0, 0.0, 1.0])
    result = ensemble.fit(frame, y, frame, y)

    save_model(ensemble, root=tmp_path, fit_result=result, train_seasons=[2021, 2022])
    restored = load("ens", result.version, tmp_path)

    assert isinstance(restored, StackedEnsemble)
    assert restored.predict_proba(frame) == pytest.approx(ensemble.predict_proba(frame))


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #


def test_promote_writes_the_production_pointer(tmp_path, frame):
    save_model(
        ConstantModel(p=0.33),
        name="gbdt",
        version="20240101T000000",
        root=tmp_path,
        metrics={"log_loss": 0.61},
    )
    path = promote("gbdt", "20240101T000000", root=tmp_path)

    assert path == pointer_path("production", tmp_path)
    assert path.name == "PRODUCTION.json"
    pointer = json.loads(path.read_text())
    assert pointer["name"] == "gbdt"
    assert pointer["version"] == "20240101T000000"
    assert pointer["path"] == "gbdt/20240101T000000"
    assert pointer["metrics"] == {"log_loss": 0.61}

    assert read_manifest(model_dir("gbdt", "20240101T000000", tmp_path)).stage == "production"
    assert load_stage("production", tmp_path).predict_proba(frame) == pytest.approx(
        np.full(3, 0.33)
    )


def test_promote_defaults_to_the_latest_version(tmp_path):
    save_model(ConstantModel(), name="m", version="20240101T000000", root=tmp_path)
    save_model(ConstantModel(), name="m", version="20240202T000000", root=tmp_path)
    promote("m", root=tmp_path)
    assert read_pointer("production", tmp_path)["version"] == "20240202T000000"


def test_stages_other_than_production_get_their_own_pointer(tmp_path):
    save_model(ConstantModel(), name="m", version="20240101T000000", root=tmp_path)
    path = promote("m", stage="staging", root=tmp_path)
    assert path.name == "STAGING.json"
    assert read_pointer("production", tmp_path) is None
    assert read_pointer("staging", tmp_path)["stage"] == "staging"


def test_promotion_is_a_pointer_move_not_a_copy(tmp_path):
    save_model(ConstantModel(p=0.1), name="m", version="20240101T000000", root=tmp_path)
    save_model(ConstantModel(p=0.9), name="m", version="20240202T000000", root=tmp_path)
    promote("m", "20240202T000000", root=tmp_path)
    promote("m", "20240101T000000", root=tmp_path)  # roll back
    assert read_pointer("production", tmp_path)["version"] == "20240101T000000"
    assert list_versions("m", tmp_path) == ["20240101T000000", "20240202T000000"]


def test_loading_an_unpromoted_stage_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing promoted"):
        load_stage("production", tmp_path)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_fingerprint_is_stable_and_ignores_column_order(dataset):
    reordered = dataset.select(sorted(dataset.columns, reverse=True))
    assert dataset_fingerprint(dataset)["hash"] == dataset_fingerprint(reordered)["hash"]


def test_fingerprint_excludes_meta_and_label_columns(dataset):
    fingerprint = dataset_fingerprint(dataset)
    assert fingerprint["n_features"] == 2
    assert fingerprint["rows_by_season"] == {"2022": 2, "2023": 1}
    assert fingerprint["n_rows"] == 3


def test_fingerprint_moves_when_a_feature_is_added(dataset):
    widened = dataset.with_columns(pl.lit(1.0).alias("bi__new_feature"))
    assert dataset_fingerprint(widened)["hash"] != dataset_fingerprint(dataset)["hash"]


def test_fingerprint_moves_when_a_season_is_backfilled(dataset):
    extra = pl.concat([dataset, dataset.tail(1)])
    assert dataset_fingerprint(extra)["hash"] != dataset_fingerprint(dataset)["hash"]
    assert dataset_fingerprint(extra)["rows_by_season"] == {"2022": 2, "2023": 2}


def test_fingerprint_can_be_built_without_a_frame():
    from_frame = dataset_fingerprint(
        feature_names=["b", "a"], rows_by_season={"2023": 10, "2022": 5}
    )
    assert (
        from_frame["hash"]
        == dataset_fingerprint(feature_names=["a", "b"], rows_by_season={"2022": 5, "2023": 10})[
            "hash"
        ]
    )
    assert from_frame["n_rows"] == 15


def test_git_sha_never_raises():
    assert isinstance(git_sha(), str)
    assert git_sha() != ""


def test_utc_version_is_sortable_and_second_resolution():
    version = utc_version()
    assert len(version) == 15 and version[8] == "T"
    assert sorted(["20240101T000000", "20231231T235959"])[0] == "20231231T235959"


def test_manifest_from_fit_result_captures_everything(dataset):
    result = FitResult(
        model_name="gbdt",
        version="20240101T000000",
        n_train=100,
        n_val=20,
        metrics={"log_loss": 0.55},
        feature_names=["bi__score_diff"],
    )
    manifest = Manifest.from_fit_result(
        result,
        ConstantModel(),
        train_seasons=[2021, 2022],
        fingerprint=dataset_fingerprint(dataset),
    )
    assert manifest.name == "gbdt"
    assert manifest.train_seasons == [2021, 2022]
    assert manifest.metrics == {"log_loss": 0.55}
    assert manifest.dataset_fingerprint["n_features"] == 2
    assert manifest.extra == {"n_train": 100, "n_val": 20}


def test_manifest_round_trips_unknown_keys(tmp_path):
    directory = tmp_path / "m" / "20240101T000000"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"name": "m", "version": "20240101T000000", "invented_field": 7})
    )
    manifest = read_manifest(directory)
    assert manifest.extra["invented_field"] == 7
