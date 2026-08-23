"""Prediction ledger: append, read back, resolve, and the ADR 0001 guard.

Everything writes under ``tmp_path``; no test touches ``data/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl.data.market_blocklist import MarketColumnLeak
from nfl.logging import ledger as ledger_mod

GAME = "2024_18_KC_DEN"


def make_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "game_id": GAME,
        "p_home": 0.71,
        "confidence": 0.55,
        "model_name": "gbdt",
        "model_version": "20240101T000000",
    }
    base.update(overrides)
    return ledger_mod.prediction_row(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def test_prediction_row_derives_the_fields_it_can() -> None:
    row = make_row(source="espn", snapshot_game_seconds_remaining=300)
    assert row["p_away"] == pytest.approx(0.29)
    assert row["season"] == 2024
    assert row["week"] == 18
    assert row["is_live"] is True
    assert row["prediction_id"]
    assert row["git_sha"]


def test_prediction_row_marks_replay_as_not_live() -> None:
    assert make_row(source="replay")["is_live"] is False


def test_prediction_row_rejects_a_column_that_does_not_exist() -> None:
    with pytest.raises(KeyError, match="not ledger columns"):
        make_row(p_hoem=0.5)


def test_prediction_row_derives_edge_from_a_recorded_market_price() -> None:
    row = make_row(p_home=0.70, market_implied_home=0.60)
    assert row["edge_vs_market"] == pytest.approx(0.10)


def test_feature_hash_is_order_sensitive_and_null_stable() -> None:
    names = ["a", "b"]
    assert ledger_mod.feature_hash([1.0, 2.0], names) == ledger_mod.feature_hash([1.0, 2.0], names)
    assert ledger_mod.feature_hash([1.0, 2.0], names) != ledger_mod.feature_hash([2.0, 1.0], names)
    # The same numbers under a different feature set are a different vector.
    assert ledger_mod.feature_hash([1.0, 2.0], names) != ledger_mod.feature_hash(
        [1.0, 2.0], ["a", "c"]
    )
    # NaN and None are the same "we do not know", so they must hash the same.
    assert ledger_mod.feature_hash([float("nan")]) == ledger_mod.feature_hash([None])


# --------------------------------------------------------------------------- #
# Append / read round-trip
# --------------------------------------------------------------------------- #


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    rows = [make_row(p_home=0.6), make_row(game_id="2023_05_BAL_KC", p_home=0.4)]
    written = ledger_mod.append(rows, root=tmp_path)

    assert len(written) == 2  # one part file per season
    assert {p.parent.name for p in written} == {"season=2024", "season=2023"}

    back = ledger_mod.read_ledger(tmp_path)
    assert back.height == 2
    assert list(back.columns) == list(ledger_mod.LEDGER_SCHEMA)
    assert sorted(back.get_column("p_home").to_list()) == [0.4, 0.6]
    assert back.get_column("resolved_home_win").null_count() == 2


def test_append_never_rewrites_an_existing_part(tmp_path: Path) -> None:
    first = ledger_mod.append([make_row()], root=tmp_path)
    before = first[0].read_bytes()
    ledger_mod.append([make_row(p_home=0.2)], root=tmp_path)

    assert first[0].read_bytes() == before
    assert ledger_mod.read_ledger(tmp_path).height == 2


def test_read_ledger_is_empty_but_typed_before_anything_is_written(tmp_path: Path) -> None:
    frame = ledger_mod.read_ledger(tmp_path)
    assert frame.height == 0
    assert list(frame.columns) == list(ledger_mod.LEDGER_SCHEMA)


def test_read_ledger_filters(tmp_path: Path) -> None:
    ledger_mod.append(
        [
            make_row(model_name="gbdt", model_version="v1"),
            make_row(model_name="mlp", model_version="v2"),
            make_row(game_id="2023_05_BAL_KC", model_name="gbdt", model_version="v1"),
        ],
        root=tmp_path,
    )
    assert ledger_mod.read_ledger(tmp_path, model_name="gbdt").height == 2
    assert ledger_mod.read_ledger(tmp_path, model_version="v2").height == 1
    assert ledger_mod.read_ledger(tmp_path, seasons=[2023]).height == 1
    assert ledger_mod.read_ledger(tmp_path, game_ids=[GAME]).height == 2


def test_conform_drops_unknown_columns_rather_than_storing_them(tmp_path: Path) -> None:
    frame = pl.DataFrame([{**make_row(), "spread_line": -3.5}], infer_schema_length=None)
    ledger_mod.append(frame, root=tmp_path)
    assert "spread_line" not in ledger_mod.read_ledger(tmp_path).columns


def test_append_rejects_a_probability_outside_the_unit_interval(tmp_path: Path) -> None:
    bad = dict(make_row())
    bad["p_home"] = 1.4
    with pytest.raises(ValueError, match="outside"):
        ledger_mod.append([bad], root=tmp_path)


# --------------------------------------------------------------------------- #
# ADR 0001 guard
# --------------------------------------------------------------------------- #


def test_market_columns_are_stored_but_flagged_eval_only(tmp_path: Path) -> None:
    ledger_mod.append(
        [make_row(kalshi_yes_price_home=0.63, market_implied_home=0.61, closing_spread_home=-3.5)],
        root=tmp_path,
    )
    frame = ledger_mod.read_ledger(tmp_path)
    assert frame.get_column("kalshi_yes_price_home").to_list() == [0.63]

    safe = ledger_mod.feature_safe(frame)
    assert not set(safe.columns) & ledger_mod.EVAL_ONLY_COLUMNS
    assert "kalshi_yes_price_home" not in safe.columns


def test_feature_safe_still_runs_the_vendor_blocklist() -> None:
    frame = ledger_mod.empty_ledger().with_columns(pl.lit(None).alias("vegas_wp"))
    with pytest.raises(MarketColumnLeak):
        ledger_mod.feature_safe(frame)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [GAME, "2024_18_SF_SEA", "2024_18_NYJ_BUF"],
            "home_score": [27, 17, None],
            "away_score": [24, 17, None],
        }
    )


def test_resolve_fills_outcomes_and_scores_them(tmp_path: Path, schedules: pl.DataFrame) -> None:
    ledger_mod.append([make_row(p_home=0.75)], root=tmp_path)
    report = ledger_mod.resolve_ledger(schedules, root=tmp_path)

    assert report.newly_resolved == 1
    assert report.still_open == 0

    row = ledger_mod.read_ledger(tmp_path).row(0, named=True)
    assert row["resolved_home_win"] is True
    assert (row["resolved_home_score"], row["resolved_away_score"]) == (27, 24)
    assert row["resolved_at_utc"] is not None
    assert row["brier"] == pytest.approx(0.0625)  # (0.75 - 1)^2
    assert row["log_loss"] == pytest.approx(0.2876820724517809)  # -ln(0.75)


def test_resolve_leaves_an_unfinished_game_open(tmp_path: Path, schedules: pl.DataFrame) -> None:
    ledger_mod.append([make_row(game_id="2024_18_NYJ_BUF")], root=tmp_path)
    report = ledger_mod.resolve_ledger(schedules, root=tmp_path)

    assert (report.newly_resolved, report.still_open) == (0, 1)
    assert ledger_mod.read_ledger(tmp_path).get_column("resolved_home_win").null_count() == 1


def test_resolve_leaves_a_tie_unresolved_rather_than_calling_it_a_loss(
    tmp_path: Path, schedules: pl.DataFrame
) -> None:
    ledger_mod.append([make_row(game_id="2024_18_SF_SEA")], root=tmp_path)
    ledger_mod.resolve_ledger(schedules, root=tmp_path)
    assert ledger_mod.read_ledger(tmp_path).row(0, named=True)["resolved_home_win"] is None


def test_resolve_is_idempotent(tmp_path: Path, schedules: pl.DataFrame) -> None:
    ledger_mod.append([make_row()], root=tmp_path)
    first = ledger_mod.resolve_ledger(schedules, root=tmp_path)
    stamp = ledger_mod.read_ledger(tmp_path).row(0, named=True)["resolved_at_utc"]

    second = ledger_mod.resolve_ledger(
        schedules, root=tmp_path, now=datetime(2030, 1, 1, tzinfo=UTC)
    )
    assert first.newly_resolved == 1
    assert second.newly_resolved == 0
    # A second pass must not restamp a row it already resolved.
    assert ledger_mod.read_ledger(tmp_path).row(0, named=True)["resolved_at_utc"] == stamp


def test_resolve_only_touches_files_it_changes(tmp_path: Path, schedules: pl.DataFrame) -> None:
    open_part = ledger_mod.append([make_row(game_id="2024_18_NYJ_BUF")], root=tmp_path)[0]
    ledger_mod.append([make_row()], root=tmp_path)
    before = open_part.read_bytes()

    report = ledger_mod.resolve_ledger(schedules, root=tmp_path)
    assert open_part not in report.files
    assert open_part.read_bytes() == before


def test_resolve_says_which_score_columns_it_wanted(tmp_path: Path) -> None:
    ledger_mod.append([make_row()], root=tmp_path)
    with pytest.raises(KeyError, match="final score columns"):
        ledger_mod.resolve_ledger(pl.DataFrame({"game_id": [GAME]}), root=tmp_path)


def test_resolve_on_an_empty_ledger_is_a_no_op(tmp_path: Path, schedules: pl.DataFrame) -> None:
    report = ledger_mod.resolve_ledger(schedules, root=tmp_path / "nothing")
    assert (report.rows, report.newly_resolved) == (0, 0)


# --------------------------------------------------------------------------- #
# DuckDB
# --------------------------------------------------------------------------- #


def test_duckdb_view_sees_the_ledger_and_its_hive_partition(tmp_path: Path) -> None:
    import duckdb

    ledger_mod.append([make_row(), make_row(game_id="2023_05_BAL_KC")], root=tmp_path)
    con = duckdb.connect(":memory:")
    try:
        assert ledger_mod.register_ledger_views(con, tmp_path) == [ledger_mod.LEDGER_TABLE]
        total = con.execute(f"select count(*) from {ledger_mod.LEDGER_TABLE}").fetchone()
        assert total is not None and total[0] == 2
        pruned = con.execute(
            f"select count(*) from {ledger_mod.LEDGER_TABLE} where season = 2024"
        ).fetchone()
        assert pruned is not None and pruned[0] == 1
    finally:
        con.close()


def test_registering_an_empty_ledger_creates_no_view(tmp_path: Path) -> None:
    import duckdb

    con = duckdb.connect(":memory:")
    try:
        assert ledger_mod.register_ledger_views(con, tmp_path) == []
    finally:
        con.close()
