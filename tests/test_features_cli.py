"""``nfl features build`` / ``nfl features inspect``. Fixtures only, no network."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from nfl.cli import app

runner = CliRunner()


@pytest.fixture
def offline_raw(
    monkeypatch: pytest.MonkeyPatch, pbp_fixture: pl.DataFrame, schedules_fixture: pl.DataFrame
) -> None:
    """Serve the committed fixtures instead of ``data/raw/nflverse``."""
    frames = {"pbp": pbp_fixture, "schedules": schedules_fixture}

    def fake_load(table: str, seasons: object = None, **kwargs: object) -> pl.DataFrame:
        return frames[table]

    monkeypatch.setattr("nfl.features.assemble.load_raw_table", fake_load)


def test_features_is_a_subcommand_group() -> None:
    result = runner.invoke(app, ["features", "--help"])
    assert result.exit_code == 0
    for command in ("build", "inspect"):
        assert command in result.stdout


def test_build_help_lists_the_options() -> None:
    result = runner.invoke(app, ["features", "build", "--help"])
    assert result.exit_code == 0
    assert "--seasons" in result.stdout


@pytest.mark.parametrize("seasons", ["2009-2024", "2024-2020", "nonsense"])
def test_bad_season_spec_exits_nonzero(seasons: str, offline_raw: None) -> None:
    result = runner.invoke(app, ["features", "build", "--seasons", seasons])
    assert result.exit_code != 0


def test_build_writes_season_partitions(offline_raw: None, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["features", "build", "--seasons", "2023-2024", "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "dataset" / "season=2024" / "part-0.parquet").exists()
    assert "built baseline_inputs" in result.output
    assert "rows" in result.output


def test_dry_run_writes_nothing(offline_raw: None, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["features", "build", "--seasons", "2024", "--out", str(tmp_path), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "dataset").exists()
    assert "dry run" in result.output


def test_inspect_reports_counts_nulls_and_variance(offline_raw: None, tmp_path: Path) -> None:
    build = runner.invoke(
        app, ["features", "build", "--seasons", "2023-2024", "--out", str(tmp_path)]
    )
    assert build.exit_code == 0, build.output

    result = runner.invoke(app, ["features", "inspect", "--path", str(tmp_path), "--top", "5"])
    assert result.exit_code == 0, result.output
    assert "features" in result.output
    assert "home win rate" in result.output
    assert "variance" in result.output


def test_inspect_without_a_dataset_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["features", "inspect", "--path", str(tmp_path)])
    assert result.exit_code == 1
