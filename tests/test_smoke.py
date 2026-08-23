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


def test_predict_passes_its_arguments_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI is a thin shell over :func:`nfl.inference.predict.predict_game`.

    The entrypoint is stubbed rather than run: this file is the surface smoke
    test, and the real pipeline is exercised offline in
    ``tests/test_inference_predict.py``.
    """
    from nfl.inference.predict import PredictionError
    from nfl.inference.resolve import GameRef

    seen: dict[str, object] = {}

    def fake_predict(game_id: str, **kwargs: object) -> None:
        seen["game_id"] = game_id
        seen.update(kwargs)
        raise PredictionError("stopped before touching any data")

    def fake_resolve(spec: str, **_: object) -> GameRef:
        """The spec resolver has its own suite; keep this test off the schedules."""
        return GameRef(
            game_id=spec,
            season=2024,
            week=18,
            game_type="REG",
            away_team="KC",
            home_team="DEN",
            kickoff=None,
            espn=None,
            spec=spec,
            how="game_id",
        )

    monkeypatch.setattr("nfl.inference.resolve.resolve_game", fake_resolve)
    monkeypatch.setattr("nfl.inference.predict.predict_game", fake_predict)
    result = runner.invoke(
        app, ["predict", "2024_18_KC_DEN", "--at", "Q4 05:00", "--source", "replay", "--no-ledger"]
    )

    assert result.exit_code == 1
    assert "stopped before touching any data" in result.output
    assert seen == {
        "game_id": "2024_18_KC_DEN",
        "at": "Q4 05:00",
        "source": "replay",
        "model": "production",
        "write_ledger": False,
    }
