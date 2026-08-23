"""CLI wiring for ``nfl market`` and the new ``nfl ingest`` subcommands."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from typer.testing import CliRunner

from nfl import cli
from nfl.data.market import closing_lines, kalshi
from nfl.data.schemas import Market

runner = CliRunner()

SCHEDULE = pl.DataFrame(
    {
        "game_id": ["2026_01_CHI_TEN"],
        "season": [2026],
        "week": [1],
        "gameday": ["2026-08-29"],
        "away_team": ["CHI"],
        "home_team": ["TEN"],
    }
)


@pytest.fixture(autouse=True)
def stub_schedule(monkeypatch):
    monkeypatch.setattr(closing_lines, "load_schedule", lambda *a, **k: SCHEDULE)


def _row(**kwargs) -> Market:
    defaults = dict(
        source="kalshi.game",
        game_id="2026_01_CHI_TEN",
        observed_at_utc=datetime(2026, 8, 29, 23, 0, tzinfo=UTC),
        book="kalshi",
        implied_prob_home=0.394,
        implied_prob_away=0.606,
        kalshi_ticker="KXNFLGAME-26AUG29CHITEN-TEN",
        kalshi_yes_price=0.40,
        kalshi_no_price=0.615,
        kalshi_open_interest=71,
        bid_ask_spread=0.08,
    )
    defaults.update(kwargs)
    return Market(**defaults)


def test_market_is_registered():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "market" in result.stdout


def test_market_snapshot(monkeypatch):
    monkeypatch.setattr(kalshi, "snapshot_game", lambda *a, **k: _row())
    result = runner.invoke(cli.app, ["market", "snapshot", "2026_01_CHI_TEN"])

    assert result.exit_code == 0
    assert "2026_01_CHI_TEN" in result.stdout
    assert "p_home=0.3940" in result.stdout


def test_market_snapshot_exits_nonzero_when_no_market(monkeypatch):
    monkeypatch.setattr(kalshi, "snapshot_game", lambda *a, **k: None)
    result = runner.invoke(cli.app, ["market", "snapshot", "2026_01_CHI_TEN"])
    assert result.exit_code == 1


def test_market_history(monkeypatch):
    rows = [_row(observed_at_utc=datetime(2026, 8, 29, hour, tzinfo=UTC)) for hour in (20, 21, 22)]
    monkeypatch.setattr(kalshi, "history_game", lambda *a, **k: rows)
    result = runner.invoke(cli.app, ["market", "history", "2026_01_CHI_TEN", "--limit", "2"])

    assert result.exit_code == 0
    assert "3 bars" in result.stdout


def test_market_history_exits_nonzero_when_empty(monkeypatch):
    monkeypatch.setattr(kalshi, "history_game", lambda *a, **k: [])
    result = runner.invoke(cli.app, ["market", "history", "2026_01_CHI_TEN"])
    assert result.exit_code == 1


def test_ingest_kalshi_command(monkeypatch, tmp_path):
    target = tmp_path / "kalshi.parquet"
    monkeypatch.setattr(kalshi, "ingest_kalshi", lambda *a, **k: target)
    result = runner.invoke(cli.app, ["ingest", "kalshi", "2026_01_CHI_TEN"])

    assert result.exit_code == 0
    assert "wrote" in result.stdout


def test_ingest_closing_lines_command(monkeypatch, tmp_path):
    target = tmp_path / "closing_lines.parquet"
    monkeypatch.setattr(closing_lines, "ingest_closing_lines", lambda *a, **k: [target])
    result = runner.invoke(cli.app, ["ingest", "closing-lines", "--seasons", "2024"])

    assert result.exit_code == 0
    # Rich hard-wraps long paths, so match on the stable prefix.
    assert "wrote" in result.stdout


def test_ingest_closing_lines_reports_missing_schedule(monkeypatch):
    def boom(*a, **k):
        raise closing_lines.ClosingLinesNotFound("no schedule")

    monkeypatch.setattr(closing_lines, "ingest_closing_lines", boom)
    result = runner.invoke(cli.app, ["ingest", "closing-lines"])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2024", [2024]), ("2010-2012", [2010, 2011, 2012]), ("2020,2022", [2020, 2022])],
)
def test_parse_seasons(text, expected):
    assert cli._parse_seasons(text) == expected
