"""``nfl live snapshot`` end-to-end, through the real typer app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import respx
from typer.testing import CliRunner

from nfl.cli import app
from nfl.data.live.espn import CORE_BASE, SITE_BASE

runner = CliRunner()
GAME_ID = "2024_01_BAL_KC"


def _replay_args(pbp_path: Path, schedules_path: Path, *extra: str) -> list[str]:
    return [
        "live",
        "snapshot",
        GAME_ID,
        "--source",
        "replay",
        "--pbp",
        str(pbp_path),
        "--schedules",
        str(schedules_path),
        *extra,
    ]


def test_replay_snapshot_renders_a_table(pbp_path: Path, schedules_path: Path) -> None:
    result = runner.invoke(app, _replay_args(pbp_path, schedules_path, "--at", "Q4 05:00"))
    assert result.exit_code == 0, result.output
    output = result.output
    assert "BAL @ KC" in output
    assert "Q4 5:00" in output
    assert "4th & 4" in output
    assert "27" in output and "17" in output
    assert "recent drives" in output
    assert "GEHA Field at Arrowhead Stadium" in output


def test_replay_snapshot_without_at_shows_the_final(pbp_path: Path, schedules_path: Path) -> None:
    result = runner.invoke(app, _replay_args(pbp_path, schedules_path))
    assert result.exit_code == 0, result.output
    assert "Final" in result.output


def test_espn_snapshot_renders_a_table(
    schedules_path: Path, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=espn_summary)
        mock.get(url__startswith=f"{CORE_BASE}/events/401671789").respond(json=espn_situation)
        result = runner.invoke(
            app,
            [
                "live",
                "snapshot",
                GAME_ID,
                "--source",
                "espn",
                "--schedules",
                str(schedules_path),
                "--at",
                "Q4 05:00",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "BAL @ KC" in result.output
    assert "via espn" in result.output
    # The rewound-timeouts caveat must be visible, not buried in a field.
    assert "timeouts" in result.output


def test_unknown_source_exits_with_a_message() -> None:
    result = runner.invoke(app, ["live", "snapshot", GAME_ID, "--source", "yahoo"])
    assert result.exit_code == 2
    assert "unknown --source" in result.output


def test_bad_at_is_reported(pbp_path: Path, schedules_path: Path) -> None:
    result = runner.invoke(app, _replay_args(pbp_path, schedules_path, "--at", "sometime"))
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


def test_live_help_lists_the_snapshot_command() -> None:
    result = runner.invoke(app, ["live", "--help"])
    assert result.exit_code == 0
    assert "snapshot" in result.output


def test_the_other_cli_commands_still_work() -> None:
    """The ``live`` sub-app must not have displaced the existing surface."""
    for command in ("ingest", "train", "predict", "evaluate"):
        result = runner.invoke(app, ["--help"])
        assert command in result.output


def test_render_is_importable_without_a_terminal(pbp_path: Path, schedules_path: Path) -> None:
    from nfl.data.live.render import snapshot_tables
    from nfl.data.live.replay import HistoricalReplayFeed

    snapshot = HistoricalReplayFeed(pbp=pbp_path, schedules=schedules_path).snapshot(GAME_ID)
    assert snapshot_tables(snapshot) is not None
    # The snapshot itself round-trips through JSON for the ledger.
    assert json.loads(snapshot.model_dump_json())["game_id"] == GAME_ID
