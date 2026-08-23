"""CLI surface for ``nfl train`` / ``nfl evaluate`` / ``nfl ledger``.

Nothing here trains or reads ``data/``: the point is that the commands exist,
their help is accurate, and the failure modes (a model module that has not
landed, an empty ledger) produce a sentence rather than a traceback.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from nfl.cli import app
from nfl.logging import ledger as ledger_mod

runner = CliRunner()


def invoke(*args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, list(args))


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #


def test_top_level_help_lists_the_new_commands() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("train", "evaluate", "ledger"):
        assert command in result.output


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("train", "--help"), "--model"),
        (("train", "--help"), "--seasons"),
        (("evaluate", "--help"), "--version"),
        (("evaluate", "--help"), "--ledger"),
        (("evaluate", "--help"), "--season"),
        (("ledger", "--help"), "resolve"),
        (("ledger", "--help"), "show"),
        (("ledger", "show", "--help"), "--last"),
    ],
)
def test_help_advertises_the_documented_options(args: tuple[str, ...], expected: str) -> None:
    result = invoke(*args)
    assert result.exit_code == 0
    assert expected in result.output


def test_train_help_names_every_model_key() -> None:
    output = invoke("train", "--help").output
    for key in ("naive", "xgb", "mlp", "seq", "ensemble"):
        assert key in output


# --------------------------------------------------------------------------- #
# train dispatch
# --------------------------------------------------------------------------- #


def test_train_rejects_an_unknown_model() -> None:
    result = invoke("train", "--model", "transformer")
    assert result.exit_code != 0
    assert "unknown model" in result.output


def test_train_explains_a_model_module_that_has_not_landed(monkeypatch: pytest.MonkeyPatch) -> None:
    from nfl import cli as cli_mod

    monkeypatch.setitem(cli_mod.TRAINERS, "seq", ("nfl.models.does_not_exist", "train_seq"))
    result = invoke("train", "--model", "seq")
    assert result.exit_code == 1
    assert "not available" in result.output


def test_train_dispatches_to_the_sibling_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Result:
        model_name = "naive"
        version = "20240101T000000"
        metrics = {"log_loss": 0.4213, "brier": 0.1401}

    module = types.ModuleType("nfl.models.naive")

    def train_naive(seasons=None, train_through=2022, val_season=2023):  # type: ignore[no-untyped-def]
        calls.append({"seasons": seasons, "train_through": train_through, "val_season": val_season})
        return Result()

    module.train_naive = train_naive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nfl.models.naive", module)

    result = invoke("train", "--model", "naive", "--seasons", "2018-2020")
    assert result.exit_code == 0, result.output
    assert calls == [{"seasons": [2018, 2019, 2020], "train_through": 2022, "val_season": 2023}]
    assert "trained" in result.output
    assert "20240101T000000" in result.output
    assert "log_loss" in result.output


def test_train_accepts_the_legacy_model_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("nfl.models.naive")
    module.train_naive = lambda **kwargs: "done"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nfl.models.naive", module)

    result = invoke("train", "--model", "baseline")
    assert result.exit_code == 0, result.output
    assert "model=naive" in result.output


def test_train_only_passes_arguments_the_trainer_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trainer with a narrow signature must not get a TypeError."""
    seen: list[object] = []
    module = types.ModuleType("nfl.models.gbdt")

    def train_gbdt(seasons=None):  # type: ignore[no-untyped-def]
        seen.append(seasons)
        return "ok"

    module.train_gbdt = train_gbdt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nfl.models.gbdt", module)

    result = invoke("train", "--model", "xgb", "--seasons", "2024", "--val-season", "2023")
    assert result.exit_code == 0, result.output
    assert seen == [[2024]]


def test_train_reports_a_module_missing_its_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "nfl.models.train_ensemble", types.ModuleType("nfl.models.train_ensemble")
    )
    result = invoke("train", "--model", "ensemble")
    assert result.exit_code == 1
    assert "train_ensemble" in result.output


def test_train_passes_ensemble_only_options_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--members``/``--meta`` reach the ensemble trainer and nobody else."""
    seen: list[dict[str, object]] = []
    module = types.ModuleType("nfl.models.train_ensemble")

    def train_ensemble(members=None, meta=None, seasons=None, train_through=2022, val_season=2023):  # type: ignore[no-untyped-def]
        seen.append({"members": members, "meta": meta})
        return "ok"

    module.train_ensemble = train_ensemble  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nfl.models.train_ensemble", module)

    result = invoke(
        "train", "--model", "ensemble", "--members", "naive, xgb ,seq", "--meta", "logistic_stacker"
    )
    assert result.exit_code == 0, result.output
    assert seen == [{"members": ["naive", "xgb", "seq"], "meta": "logistic_stacker"}]


def test_train_renders_a_trainer_that_returns_model_and_result() -> None:
    """Most trainers return ``(model, FitResult)``; the metrics table still prints."""
    from nfl.cli import _render_fit_result

    class Result:
        model_name = "ensemble_v1"
        version = "20240101T000000"
        metrics = {"log_loss": 0.31}

    _render_fit_result((object(), Result()), "ensemble")


# --------------------------------------------------------------------------- #
# promote
# --------------------------------------------------------------------------- #


def _stock_gate_registry(root: Path, *, candidate_log_loss: float) -> None:
    """A floor and a candidate, both carrying the val slices the gate reads."""
    import json

    from nfl.models.train_ensemble import GATE_Q4_SLICES, NOT_COLLAPSED_SLICE

    def slices(log_loss: float, brier: float, nc: float) -> list[dict[str, float]]:
        rows = [{"slice": s, "log_loss": log_loss, "brier": brier} for s in GATE_Q4_SLICES]
        rows.append({"slice": NOT_COLLAPSED_SLICE, "log_loss": nc, "brier": nc / 3.0})
        return rows

    for name, version, payload in (
        ("naive_score_time", "20240101T000000", slices(0.30, 0.10, 0.60)),
        (
            "cand",
            "20240102T000000",
            slices(candidate_log_loss, candidate_log_loss / 3.0, 0.55),
        ),
    ):
        directory = root / name / version
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps({"name": name, "version": version, "val_slices": payload})
        )


@pytest.fixture
def gate_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "models"
    monkeypatch.setattr("nfl.models.registry.MODELS_DIR", root)
    return root


def test_promote_moves_the_pointer_when_the_gate_passes(gate_registry: Path) -> None:
    import json

    _stock_gate_registry(gate_registry, candidate_log_loss=0.20)
    result = invoke("promote", "--candidate", "cand", "--parity-samples", "0")
    assert result.exit_code == 0, result.output
    assert "promoted" in result.output
    pointer = json.loads((gate_registry / "PRODUCTION.json").read_text())
    assert (pointer["name"], pointer["version"]) == ("cand", "20240102T000000")


def test_promote_refuses_and_shows_the_failing_numbers(gate_registry: Path) -> None:
    _stock_gate_registry(gate_registry, candidate_log_loss=0.40)
    result = invoke("promote", "--candidate", "cand", "--parity-samples", "0")
    assert result.exit_code == 1
    assert "NOT promoted" in result.output
    assert "FAIL" in result.output
    assert not (gate_registry / "PRODUCTION.json").exists()


def test_promote_force_promotes_but_says_so_loudly(gate_registry: Path) -> None:
    _stock_gate_registry(gate_registry, candidate_log_loss=0.40)
    result = invoke("promote", "--candidate", "cand", "--parity-samples", "0", "--force")
    assert result.exit_code == 0, result.output
    assert "FORCED PROMOTION" in result.output
    assert "worse than the naive floor" in result.output
    assert (gate_registry / "PRODUCTION.json").is_file()


def test_promote_explains_a_candidate_that_is_not_in_the_registry(gate_registry: Path) -> None:
    _stock_gate_registry(gate_registry, candidate_log_loss=0.20)
    result = invoke("promote", "--candidate", "nope", "--parity-samples", "0")
    assert result.exit_code == 1
    assert "cannot evaluate the promotion gate" in result.output


def test_promote_help_advertises_the_gate_options() -> None:
    output = invoke("promote", "--help").output
    for option in ("--candidate", "--version", "--force"):
        assert option in output


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_on_an_empty_ledger_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    result = invoke("evaluate")
    assert result.exit_code == 0, result.output
    assert "no predictions to evaluate" in result.output


def test_evaluate_scores_the_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    ledger_mod.append(
        [
            ledger_mod.prediction_row(
                game_id=f"2024_0{i}_AA_BB",
                p_home=p,
                confidence=0.5,
                model_name="naive",
                model_version="v1",
                resolved_home_win=won,
            )
            for i, (p, won) in enumerate([(0.9, True), (0.1, False), (0.8, True)], start=1)
        ],
        root=tmp_path,
    )
    result = invoke("evaluate", "--model", "naive", "--no-vs-market")
    assert result.exit_code == 0, result.output
    assert "naive" in result.output
    assert "log_loss" in result.output


def test_evaluate_backtest_needs_a_model_name() -> None:
    result = invoke("evaluate", "--backtest")
    assert result.exit_code != 0
    assert "--model" in result.output


def test_evaluate_backtest_explains_a_missing_registry() -> None:
    result = invoke("evaluate", "--backtest", "--model", "naive")
    assert result.exit_code == 1
    assert "registry" in result.output


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def test_ledger_show_on_an_empty_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    result = invoke("ledger", "show")
    assert result.exit_code == 0
    assert "ledger is empty" in result.output


def test_ledger_show_prints_recent_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # rich truncates table cells to the terminal width; 80 columns is not enough
    # to see a game id, and this test is about the content, not the layout.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    ledger_mod.append(
        [
            ledger_mod.prediction_row(
                game_id="2024_18_KC_DEN",
                p_home=0.7123,
                confidence=0.5,
                model_name="naive",
                model_version="v1",
                snapshot_game_seconds_remaining=300,
            )
        ],
        root=tmp_path,
    )
    result = invoke("ledger", "show", "--last", "5")
    assert result.exit_code == 0, result.output
    assert "2024_18_KC_DEN" in result.output
    assert "0.7123" in result.output
    assert "0 of 1 resolved" in result.output


def test_ledger_resolve_fills_outcomes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    ledger_mod.append(
        [
            ledger_mod.prediction_row(
                game_id="2024_18_KC_DEN",
                p_home=0.7,
                confidence=0.5,
                model_name="naive",
                model_version="v1",
            )
        ],
        root=tmp_path,
    )
    schedules = pl.DataFrame(
        {"game_id": ["2024_18_KC_DEN"], "home_score": [27], "away_score": [24]}
    )
    monkeypatch.setattr("nfl.cli._schedules_with_scores", lambda: schedules)

    result = invoke("ledger", "resolve")
    assert result.exit_code == 0, result.output
    assert "1 newly resolved" in result.output
    assert ledger_mod.read_ledger(tmp_path).row(0, named=True)["resolved_home_win"] is True


def test_ledger_resolve_reports_a_schedule_without_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ledger_mod.config, "LEDGER_DIR", tmp_path)
    ledger_mod.append(
        [
            ledger_mod.prediction_row(
                game_id="2024_18_KC_DEN",
                p_home=0.7,
                confidence=0.5,
                model_name="naive",
                model_version="v1",
            )
        ],
        root=tmp_path,
    )
    monkeypatch.setattr(
        "nfl.cli._schedules_with_scores", lambda: pl.DataFrame({"game_id": ["2024_18_KC_DEN"]})
    )
    result = invoke("ledger", "resolve")
    assert result.exit_code == 1
    assert "final score columns" in result.output
