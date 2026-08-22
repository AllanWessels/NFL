"""Smoke tests: the package imports and the CLI surface exists."""

from __future__ import annotations

import importlib

import pytest
from typer.testing import CliRunner

import nfl
from nfl.cli import app

runner = CliRunner()

SUBMODULES = [
    "nfl.cli",
    "nfl.data",
    "nfl.data.schemas",
    "nfl.features",
    "nfl.inference",
    "nfl.logging",
    "nfl.models",
]


def test_package_has_version() -> None:
    assert nfl.__version__


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_nfl_logging_does_not_shadow_stdlib() -> None:
    """``nfl.logging`` must not break ``import logging`` inside the package."""
    import logging

    assert logging.getLogger("nfl") is not None


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "train", "predict", "evaluate"):
        assert command in result.output


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert nfl.__version__ in result.output


@pytest.mark.parametrize("command", ["ingest", "train", "predict", "evaluate"])
def test_each_command_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_predict_requires_game_id() -> None:
    result = runner.invoke(app, ["predict"])
    assert result.exit_code != 0


def test_predict_accepts_game_id_and_at() -> None:
    result = runner.invoke(app, ["predict", "2024_18_KC_DEN", "--at", "Q4 05:00"])
    assert result.exit_code == 0
    assert "2024_18_KC_DEN" in result.output
