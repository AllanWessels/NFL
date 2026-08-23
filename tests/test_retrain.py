"""``nfl retrain``: the plan, the ordering, and what the summary reports.

Nothing here fits a model. What is worth testing about a chain is that it is
*the* chain — the right steps in the right order, ingest actually dropped when
asked, and the run log written whichever way it went. The fits themselves are
covered by each trainer's own tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nfl import retrain as retrain_mod
from nfl.cli import app
from nfl.features.splits import SeasonSplit
from nfl.retrain import (
    INGEST_STEPS,
    RETRAIN_STEPS,
    RetrainResult,
    StepResult,
    current_season,
    recorded_seq_config,
    recorded_xgb_config,
    run_retrain,
    steps_for,
    summary_table,
    to_markdown,
)

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_the_chain_is_in_dependency_order() -> None:
    """Each step consumes what the one before it wrote; the order is the contract."""
    assert RETRAIN_STEPS == (
        "ingest nflverse",
        "ingest weather",
        "features build",
        "train naive",
        "train xgb",
        "train seq",
        "train ensemble",
        "promote",
        "ledger resolve",
        "evaluate",
    )


def test_the_floor_is_trained_before_everything_it_is_the_floor_for() -> None:
    """The naive model keys the not-collapsed slice, so it cannot be trained last."""
    order = list(RETRAIN_STEPS)
    assert order.index("train naive") < order.index("train xgb")
    assert order.index("train naive") < order.index("train seq")
    assert order.index("train ensemble") < order.index("promote")
    assert order.index("features build") < order.index("train naive")


def test_skip_ingest_drops_exactly_the_two_ingest_steps() -> None:
    kept = steps_for(skip_ingest=True)
    assert list(kept) == [s for s in RETRAIN_STEPS if s not in INGEST_STEPS]
    assert set(INGEST_STEPS).isdisjoint(kept)
    assert list(steps_for()) == list(RETRAIN_STEPS)


def test_dry_run_plans_every_step_and_runs_none(tmp_path: Path) -> None:
    result = run_retrain(
        season=2025,
        split=SeasonSplit.rolling(2025),
        dry_run=True,
        models_root=tmp_path / "models",
        data_root=tmp_path,
    )
    assert [s.name for s in result.steps] == list(RETRAIN_STEPS)
    assert {s.status for s in result.steps} == {"planned"}
    assert all(s.seconds == 0.0 for s in result.steps)
    assert result.gate is None and result.promoted is False
    assert result.split == SeasonSplit(2023, 2024, 2025)


def test_dry_run_with_skip_ingest_marks_the_ingest_steps_skipped(tmp_path: Path) -> None:
    result = run_retrain(
        season=2025,
        skip_ingest=True,
        dry_run=True,
        models_root=tmp_path / "models",
        data_root=tmp_path,
    )
    statuses = {s.name: s.status for s in result.steps}
    assert [s.name for s in result.steps] == list(RETRAIN_STEPS)
    assert statuses["ingest nflverse"] == "skipped"
    assert statuses["ingest weather"] == "skipped"
    assert statuses["features build"] == "planned"


def test_dry_run_writes_its_run_log(tmp_path: Path) -> None:
    result = run_retrain(
        season=2025, dry_run=True, models_root=tmp_path / "models", data_root=tmp_path
    )
    assert result.report_path is not None
    assert result.report_path.parent == tmp_path / "reports"
    body = result.report_path.read_text(encoding="utf-8")
    for step in RETRAIN_STEPS:
        assert step in body
    assert "dry run: `True`" in body


# --------------------------------------------------------------------------- #
# Failure stops the chain
# --------------------------------------------------------------------------- #


def test_a_raising_step_stops_the_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("no dataset on disk")

    monkeypatch.setattr("nfl.features.assemble.assemble_dataset", boom)
    result = run_retrain(
        season=2025,
        skip_ingest=True,
        models_root=tmp_path / "models",
        data_root=tmp_path,
    )
    names = [s.name for s in result.steps]
    assert names == ["ingest nflverse", "ingest weather", "features build"]
    failed = result.failed_step
    assert failed is not None and failed.name == "features build"
    assert "no dataset on disk" in failed.detail
    assert not result.succeeded
    # Even a stopped run leaves its log behind — that is the run you want it for.
    assert result.report_path is not None and result.report_path.is_file()
    assert "features build" in result.report_path.read_text(encoding="utf-8")


def test_the_cli_exits_non_zero_when_a_step_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_DATA_DIR", str(tmp_path))

    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("nfl.features.assemble.assemble_dataset", boom)
    result = runner.invoke(app, ["retrain", "--skip-ingest", "--season", "2025"])
    assert result.exit_code == 1
    assert "retrain stopped at 'features build'" in result.output


# --------------------------------------------------------------------------- #
# Recorded hyperparameters
# --------------------------------------------------------------------------- #


def _write_manifest(root: Path, name: str, version: str, payload: dict) -> None:
    directory = root / name / version
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"name": name, "version": version, **payload}), encoding="utf-8"
    )


def test_xgb_params_come_from_the_latest_manifest(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "xgboost_winprob",
        "20200101T000000",
        {"params": {"max_depth": 3}, "feature_set": "full"},
    )
    _write_manifest(
        tmp_path,
        "xgboost_winprob",
        "20250101T000000",
        {
            "params": {"max_depth": 6, "eta": 0.05},
            "feature_set": "servable",
            "selection_metric": "q4",
            "weight_by_game": True,
        },
    )
    config = recorded_xgb_config(tmp_path)
    assert config["params"] == {"max_depth": 6, "eta": 0.05}
    assert config["feature_set"] == "servable"
    assert config["selection_metric"] == "q4"
    assert config["source_version"] == "20250101T000000"


def test_seq_params_come_from_the_latest_manifest(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "tf_seq",
        "20250101T000000",
        {"hyperparams": {"k": 32, "gru_units": 64}, "feature_set": "servable"},
    )
    config = recorded_seq_config(tmp_path)
    assert config["kwargs"]["k"] == 32
    assert config["kwargs"]["feature_set"] == "servable"
    assert config["source_version"] == "20250101T000000"


def test_an_empty_registry_falls_back_to_module_defaults(tmp_path: Path) -> None:
    """A first-ever run has no manifest to read; it must still be runnable."""
    assert recorded_xgb_config(tmp_path)["params"] == {}
    assert recorded_xgb_config(tmp_path)["feature_set"] == "servable"
    assert recorded_seq_config(tmp_path)["kwargs"]["feature_set"] == "servable"


# --------------------------------------------------------------------------- #
# Season, summary, markdown
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2025, 9, 8, tzinfo=UTC), 2025),
        (datetime(2025, 12, 30, tzinfo=UTC), 2025),
        (datetime(2026, 1, 12, tzinfo=UTC), 2025),
        (datetime(2026, 2, 8, tzinfo=UTC), 2025),
        (datetime(2026, 3, 1, tzinfo=UTC), 2026),
    ],
)
def test_current_season_follows_the_football_not_the_calendar(
    when: datetime, expected: int
) -> None:
    assert current_season(when) == expected


def _result(**kwargs: object) -> RetrainResult:
    base = RetrainResult(
        started_at=datetime(2026, 8, 23, 6, 0, tzinfo=UTC),
        season=2026,
        split=SeasonSplit.rolling(2025),
        candidate="ensemble_v1",
        candidate_version="20260823T060000",
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def test_the_summary_answers_gate_pointer_and_drift() -> None:
    rows = dict(summary_table(_result()))
    assert set(rows) >= {"gate", "PRODUCTION", "drift", "split"}
    assert rows["split"] == "<=2023/2024/2025+"
    assert rows["PRODUCTION"] == "nothing promoted"
    assert "no signal" in rows["drift"]


def test_the_summary_names_the_step_that_stopped_the_run() -> None:
    result = _result(steps=[StepResult("train xgb", "failed", 1.5, "RuntimeError: cuda gone")])
    rows = dict(summary_table(result))
    assert "train xgb" in rows["stopped at"]
    assert "cuda gone" in rows["stopped at"]


def test_markdown_carries_the_steps_and_the_summary() -> None:
    result = _result(
        steps=[StepResult("features build", "ok", 12.0, "1,000 rows")],
        pointer={"name": "ensemble_v1", "version": "20260823T060000"},
        promoted=True,
    )
    body = to_markdown(result)
    assert "# Retrain run 2026-08-23T06:00:00Z" in body
    assert "| features build | ok | 12.0 | 1,000 rows |" in body
    assert "moved this run" in body


def test_module_docstring_names_the_two_failure_modes() -> None:
    """The refuse-vs-raise distinction is behaviour, so it has to be written down."""
    assert "raises" in (retrain_mod.__doc__ or "")
    assert "refuses" in (retrain_mod.__doc__ or "")


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_retrain_help_lists_its_flags() -> None:
    result = runner.invoke(app, ["retrain", "--help"])
    assert result.exit_code == 0
    for flag in ("--season", "--skip-ingest", "--dry-run"):
        assert flag in result.output


def test_retrain_appears_in_the_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "retrain" in result.output


def test_train_help_offers_the_split_override() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--split" in result.output
    assert "rolling" in result.output


def test_weather_ingest_defaults_to_a_recursive_glob() -> None:
    """The schedules are partitioned by season, so a single-star glob matches nothing."""
    from nfl.data.weather import DEFAULT_GAMES_PARQUET

    assert "**" in str(DEFAULT_GAMES_PARQUET)
    assert str(DEFAULT_GAMES_PARQUET).endswith("*.parquet")


def test_dry_run_through_the_cli_prints_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["retrain", "--dry-run", "--skip-ingest", "--season", "2025"])
    assert result.exit_code == 0, result.output
    assert "retrain summary" in result.output
    assert "not run (dry run)" in result.output
