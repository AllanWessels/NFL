"""``nfl ingest nflverse`` argument handling. No network, no disk writes."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nfl.cli import app

runner = CliRunner()


def test_ingest_is_a_subcommand_group() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "nflverse" in result.stdout


def test_nflverse_help_lists_the_options() -> None:
    result = runner.invoke(app, ["ingest", "nflverse", "--help"])
    assert result.exit_code == 0
    for option in ("--seasons", "--tables", "--force"):
        assert option in result.stdout


@pytest.mark.parametrize("seasons", ["2009-2024", "2024-2020", "nonsense"])
def test_bad_season_spec_exits_nonzero(seasons: str) -> None:
    result = runner.invoke(app, ["ingest", "nflverse", "--seasons", seasons])
    assert result.exit_code != 0


def test_unknown_table_exits_nonzero() -> None:
    result = runner.invoke(app, ["ingest", "nflverse", "--tables", "pbp,nope"])
    assert result.exit_code != 0
