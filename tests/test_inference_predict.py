"""The ``/predict`` path, end to end and offline.

Every test here runs the *real* pipeline — feed, feature builders, registry,
ledger — against the committed fixtures for BAL @ KC, 2024 week 1. Nothing
touches the network: the replay feed reads the play-by-play parquet, the ESPN
feed reads recorded JSON through ``respx``, and the model is a real
:class:`~nfl.models.dummy.ConstantModel` saved into a ``tmp_path`` registry.

The load-bearing test is :func:`test_replay_features_match_the_offline_dataset`.
If the vector served at ``Q4 05:00`` is not the vector the model trained on for
that snap, every backtest number describes a different model than the one in
production, and nothing else in this file matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import respx
from typer.testing import CliRunner

from nfl.cli import app
from nfl.data.live.espn import CORE_BASE, SITE_BASE
from nfl.inference.predict import (
    PredictionError,
    build_feature_row,
    check_parity,
    load_model,
    predict_game,
    render_prediction,
    resolve_source,
    snapshot_to_baseline_row,
)
from nfl.logging import ledger as ledger_mod
from nfl.models import registry
from nfl.models.dummy import ConstantModel

GAME_ID = "2024_01_BAL_KC"
EVENT_ID = "401671789"
AT = "Q4 05:00"

#: A handful of real dataset columns plus one the serving path cannot produce,
#: so the "missing model input" path is exercised on every prediction.
MODEL_FEATURES = [
    "bi__score_diff_home",
    "bi__game_seconds_remaining",
    "bi__posteam_is_home",
    "ig__score_diff_home",
    "home_tm__elo_pre",
    "home_pl__week",
    "never__produced",
]


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    """A registry holding one naive-named artifact, and no PRODUCTION pointer."""
    root = tmp_path / "models"
    model = ConstantModel(
        p=0.91,
        name="naive_score_time",
        version="20240101T000000",
        feature_names=list(MODEL_FEATURES),
    )
    registry.save_model(model, root=root, version="20240101T000000")
    return root


@pytest.fixture
def roots(tmp_path: Path, models_root: Path) -> dict[str, Path]:
    return {
        "models_root": models_root,
        "ledger_root": tmp_path / "ledger",
        "cache_root": tmp_path / "cache",
    }


@pytest.fixture
def espn_mock(espn_summary: dict[str, Any], espn_situation: dict[str, Any]) -> Any:
    """ESPN's two endpoints, served from the recorded fixtures."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=espn_summary)
        mock.get(url__startswith=f"{CORE_BASE}/events/{EVENT_ID}").respond(json=espn_situation)
        yield mock


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #


def test_auto_picks_replay_for_a_finished_game_with_local_pbp(
    schedules: pl.DataFrame, pbp_path: Path
) -> None:
    assert resolve_source(GAME_ID, "auto", schedules=schedules, pbp=pbp_path) == "replay"


def test_auto_picks_espn_when_the_game_has_not_been_played(schedules: pl.DataFrame) -> None:
    unplayed = schedules.with_columns(
        pl.lit(None, dtype=schedules.schema["home_score"]).alias("home_score"),
        pl.lit(None, dtype=schedules.schema["away_score"]).alias("away_score"),
    )
    assert resolve_source(GAME_ID, "auto", schedules=unplayed, pbp=None) == "espn"


def test_an_unknown_game_id_is_a_prediction_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(PredictionError, match="not in the schedules"):
        resolve_source("1999_01_XXX_YYY", "auto", schedules=schedules)


# --------------------------------------------------------------------------- #
# The replay path
# --------------------------------------------------------------------------- #


def test_replay_prediction_reads_the_state_we_expect(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    """BAL trail 27-17 at Q4 5:00, 4th & 4 from the KC 14."""
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    snapshot = result.snapshot
    assert (snapshot.home_team, snapshot.away_team) == ("KC", "BAL")
    assert (snapshot.home_score, snapshot.away_score) == (27, 17)
    assert snapshot.label == "Q4 5:00"
    assert snapshot.possession == "BAL"
    assert (snapshot.down, snapshot.distance, snapshot.yards_to_endzone) == (4, 4, 14)
    assert result.source == "replay"


def test_the_model_output_reaches_the_result(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert result.p_home == pytest.approx(0.91)
    assert result.p_away == pytest.approx(0.09)
    assert result.confidence == pytest.approx(0.82)
    assert result.confidence_method == "abs_margin"
    assert result.model.name == "naive_score_time"
    assert result.model.version == "20240101T000000"
    assert result.feature_hash and len(result.feature_hash) == 64
    assert result.latency_ms > 0


def test_the_trade_gate_flags_a_collapsed_probability(
    pbp: pl.DataFrame, schedules: pl.DataFrame, tmp_path: Path, roots: dict[str, Path]
) -> None:
    """0.91 is outside the 15/85 band; 0.60 is inside it."""
    collapsed = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert collapsed.collapsed is True

    registry.save_model(
        ConstantModel(p=0.60, name="naive_score_time", feature_names=list(MODEL_FEATURES)),
        root=roots["models_root"],
        version="20240102T000000",
    )
    open_gate = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert open_gate.model.version == "20240102T000000"
    assert open_gate.collapsed is False


def test_a_model_input_the_serving_path_cannot_produce_is_null_and_reported(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert result.missing_features == ("never__produced",)
    assert result.features_total == len(MODEL_FEATURES)
    assert result.features_nonnull == len(MODEL_FEATURES) - 1


def test_the_pregame_table_is_cached_per_game(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    cached = roots["cache_root"] / "pregame" / f"{GAME_ID}.parquet"
    assert cached.is_file()
    frame = pl.read_parquet(cached)
    assert set(frame.get_column("team")) == {"KC", "BAL"}
    assert any(c.startswith("tm__") for c in frame.columns)

    # The second call must not need the play-by-play at all.
    again = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert again.feature_hash


def test_a_model_that_asks_for_no_pregame_column_does_not_build_one(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    """The naive floor reads nine ``bi__`` columns; it must not pay for Elo.

    Skipping the team and player builders is the difference between a ~2 s and
    a ~0.3 s prediction, and it is safe precisely because the model's manifest
    says which columns it will look at.
    """
    registry.save_model(
        ConstantModel(
            p=0.5,
            name="naive_score_time",
            feature_names=["bi__score_diff_home", "bi__game_seconds_remaining"],
        ),
        root=roots["models_root"],
        version="20240103T000000",
    )
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert result.model.version == "20240103T000000"
    assert result.features_nonnull == 2
    assert not (roots["cache_root"] / "pregame" / f"{GAME_ID}.parquet").exists()


# --------------------------------------------------------------------------- #
# The ESPN path
# --------------------------------------------------------------------------- #


def test_espn_prediction_matches_the_replay_one(
    espn_mock: Any,
    schedules_path: Path,
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    roots: dict[str, Path],
) -> None:
    """Same game, same moment, two feeds: the probability must not move.

    The model here is constant, so what this really pins is that both feeds
    produce a feature vector the model accepts and that the snapshot summary
    agrees on the game state — the pair of failures that would make a live
    prediction differ from its backtest.
    """
    live = predict_game(GAME_ID, at=AT, source="espn", schedules=schedules_path, pbp=pbp, **roots)
    replayed = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert live.source == "espn"
    assert live.p_home == replayed.p_home
    assert live.snapshot.home_score == replayed.snapshot.home_score == 27
    assert live.snapshot.away_score == replayed.snapshot.away_score == 17
    assert live.snapshot.possession == replayed.snapshot.possession == "BAL"
    assert live.snapshot.label == replayed.snapshot.label == "Q4 5:00"


def test_the_espn_path_does_not_call_kalshi_for_a_rewound_game(
    espn_mock: Any, schedules_path: Path, pbp: pl.DataFrame, roots: dict[str, Path]
) -> None:
    """``attach_market`` defaults to off unless the snapshot is live.

    ``respx`` is asserting that nothing outside the two mocked ESPN routes is
    requested, so a stray market call would fail the test rather than hang.
    """
    result = predict_game(GAME_ID, at=AT, source="espn", schedules=schedules_path, pbp=pbp, **roots)
    assert result.market is None


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_a_ledger_row_is_written_with_the_snapshot_and_provenance(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    frame = ledger_mod.read_ledger(root=roots["ledger_root"])
    assert frame.height == 1
    row = frame.row(0, named=True)

    assert row["prediction_id"] == result.prediction_id
    assert row["game_id"] == GAME_ID
    assert row["season"] == 2024
    assert row["source"] == "replay"
    assert row["is_live"] is False
    assert row["snapshot_period"] == 4
    assert row["snapshot_clock"] == "5:00"
    assert row["snapshot_game_seconds_remaining"] == 300
    assert (row["home_score"], row["away_score"]) == (27, 17)
    assert (row["posteam"], row["down"], row["ydstogo"], row["yardline_100"]) == ("BAL", 4, 4, 14)
    assert row["p_home"] == pytest.approx(0.91)
    assert row["p_away"] == pytest.approx(0.09)
    assert row["model_name"] == "naive_score_time"
    assert row["model_kind"] == "baseline"
    assert row["feature_hash"] == result.feature_hash
    assert row["features_total_count"] == len(MODEL_FEATURES)
    assert row["git_sha"]
    assert row["latency_ms"] > 0

    # Market columns are null until a snapshot is attached (ADR 0001).
    for column in ("kalshi_yes_price_home", "market_implied_home", "closing_spread_home"):
        assert row[column] is None


def test_no_ledger_row_when_writing_is_disabled(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    result = predict_game(
        GAME_ID,
        at=AT,
        source="replay",
        schedules=schedules,
        pbp=pbp,
        write_ledger=False,
        **roots,
    )
    assert result.prediction_id is None
    assert ledger_mod.read_ledger(root=roots["ledger_root"]).is_empty()


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #


def test_production_falls_back_to_the_naive_artifact_with_a_warning(
    models_root: Path,
) -> None:
    model, ref, warnings = load_model("production", root=models_root)
    assert ref.name == "naive_score_time"
    assert ref.stage == "none"
    assert warnings and "no PRODUCTION pointer" in warnings[0]
    assert model.predict_proba(pl.DataFrame({"x": [1.0]})).size == 1


def test_a_promoted_model_is_served_without_a_warning(models_root: Path) -> None:
    registry.promote("naive_score_time", "20240101T000000", root=models_root)
    _, ref, warnings = load_model("production", root=models_root)
    assert (ref.name, ref.version) == ("naive_score_time", "20240101T000000")
    assert warnings == []


def test_a_model_can_be_pinned_by_name_and_version(models_root: Path) -> None:
    registry.save_model(
        ConstantModel(p=0.2, name="naive_score_time"), root=models_root, version="20250101T000000"
    )
    _, latest, _ = load_model("naive_score_time", root=models_root)
    _, pinned, _ = load_model("naive_score_time:20240101T000000", root=models_root)
    assert latest.version == "20250101T000000"
    assert pinned.version == "20240101T000000"


def test_an_empty_registry_says_so(tmp_path: Path) -> None:
    with pytest.raises(PredictionError, match="no trained models"):
        load_model("production", root=tmp_path / "empty")


def test_the_fallback_warning_reaches_the_result(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    assert any("no PRODUCTION pointer" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Parity: serving features vs the offline dataset
# --------------------------------------------------------------------------- #


def test_replay_features_match_the_offline_dataset(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    """Train/serve parity for one snap of 2024_01_BAL_KC at Q4 05:00.

    The comparison covers every column both paths carry — ``bi__``, ``ig__``,
    ``ctx__``, ``home_tm__``/``away_tm__``, ``home_pl__``/``away_pl__`` — except
    the two documented exclusions: features a snapshot's play list cannot know
    (per-play EPA and statistical detail), and features that move with the game
    clock, which differ because the snapshot reports the time that was asked for
    and the play-by-play reports the clock at the snap.
    """
    report = check_parity(GAME_ID, AT, pbp=pbp, schedules=schedules)
    assert report.ok, str(report)
    assert report.play_id is not None
    assert len(report.compared) > 500
    assert report.missing_offline == ()
    assert report.missing_serving == ()

    # Every builder's block took part, not just the cheap in-play one.
    for prefix in ("bi__", "ig__", "ctx__", "home_tm__", "away_tm__", "home_pl__", "away_pl__"):
        assert any(c.startswith(prefix) for c in report.compared), prefix


def test_the_clock_is_the_only_thing_the_two_paths_disagree_on(
    pbp: pl.DataFrame, schedules: pl.DataFrame
) -> None:
    """The snapshot says 300 seconds left; the snap happened at 4:58.

    Asserted rather than hidden: the gap is two seconds, and it moves only the
    seconds-remaining family.
    """
    report = check_parity(GAME_ID, AT, pbp=pbp, schedules=schedules)
    differing = {name for name, _, _ in report.clock_sensitive}
    assert "bi__game_seconds_remaining" in differing
    for name, serve, offline in report.clock_sensitive:
        if name.endswith(("seconds_remaining", "elapsed_game_seconds")):
            assert abs(float(serve) - float(offline)) <= 5.0, name


def test_parity_needs_a_snapshot_time(pbp: pl.DataFrame, schedules: pl.DataFrame) -> None:
    with pytest.raises(PredictionError, match="needs an --at"):
        check_parity(GAME_ID, None, pbp=pbp, schedules=schedules)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Feature-row details
# --------------------------------------------------------------------------- #


def test_the_baseline_block_lags_the_score_like_the_offline_builder(
    pbp: pl.DataFrame, schedules: pl.DataFrame, pbp_path: Path
) -> None:
    """``bi__`` scores are pre-snap, which for a snapshot means "as reported"."""
    from nfl.data.live.replay import HistoricalReplayFeed

    snapshot = HistoricalReplayFeed(pbp=pbp, schedules=schedules).snapshot(GAME_ID, at=AT)
    row = snapshot_to_baseline_row(snapshot)
    assert row["bi__home_score"] == 27.0
    assert row["bi__away_score"] == 17.0
    assert row["bi__score_diff_home"] == 10.0
    assert row["bi__score_diff_posteam"] == -10.0  # BAL has the ball, trailing
    assert row["bi__posteam_is_home"] == 0
    assert row["bi__down"] == 4.0
    assert row["bi__yardline_100"] == 14.0
    assert row["bi__game_seconds_remaining"] == 300.0


def test_the_feature_row_carries_every_builder_block(
    pbp: pl.DataFrame, schedules: pl.DataFrame, tmp_path: Path
) -> None:
    from nfl.data.live.replay import HistoricalReplayFeed

    snapshot = HistoricalReplayFeed(pbp=pbp, schedules=schedules).snapshot(GAME_ID, at=AT)
    values = build_feature_row(
        snapshot,
        game_id=GAME_ID,
        schedules=schedules,
        pbp=pbp,
        cache_root=tmp_path / "cache",
    )
    for prefix in ("bi__", "ig__", "ctx__", "home_tm__", "away_tm__", "home_pl__", "away_pl__"):
        assert any(k.startswith(prefix) for k in values), prefix
    # Market data never reaches a feature vector (ADR 0001).
    banned = ("spread", "vegas", "moneyline", "odds", "wp", "kalshi")
    assert not [k for k in values if any(k.startswith(f"{b}") for b in banned)]


# --------------------------------------------------------------------------- #
# Rendering and the CLI
# --------------------------------------------------------------------------- #


def test_the_panel_reports_state_probabilities_and_gate(
    pbp: pl.DataFrame, schedules: pl.DataFrame, roots: dict[str, Path]
) -> None:
    from rich.console import Console

    result = predict_game(GAME_ID, at=AT, source="replay", schedules=schedules, pbp=pbp, **roots)
    console = Console(record=True, width=120)
    render_prediction(result, console)
    text = console.export_text()
    assert "BAL 17" in text and "KC 27" in text
    assert "Q4 5:00" in text
    assert "0.9100" in text
    assert "COLLAPSED" in text
    assert "naive_score_time" in text


def test_the_cli_rejects_an_unknown_source() -> None:
    result = CliRunner().invoke(app, ["predict", GAME_ID, "--source", "carrier-pigeon"])
    assert result.exit_code != 0
    assert "unknown source" in result.output


def test_the_cli_reports_a_prediction_error_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise PredictionError("no trained models under data/models")

    monkeypatch.setattr("nfl.inference.predict.predict_game", boom)
    result = CliRunner().invoke(app, ["predict", GAME_ID])
    assert result.exit_code == 1
    assert "no trained models" in result.output


def test_the_cli_help_documents_every_option() -> None:
    result = CliRunner().invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    for option in ("--at", "--source", "--model", "--no-ledger"):
        assert option in result.output
