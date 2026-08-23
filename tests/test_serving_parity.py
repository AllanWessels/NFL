"""What the model is served must be what the model was trained on.

Three claims, all checked against the committed BAL @ KC fixtures and the
*real* builders — no stubs, no hand-made frames:

1. **Every ``servable`` column is actually served.** The 89 in-play columns a
   :class:`~nfl.data.live.base.LiveSnapshot` cannot fill are excluded from the
   feature set by name; what is left has to arrive non-null from a real
   snapshot, or the exclusion list is wrong.
2. **The serving play history is the offline play history.** ``nfl predict``
   now builds one feature row per play so far rather than one row for the snap,
   because a GRU trained on a 32-play window and served a one-row frame is
   windowing over a single play. Those history rows have to equal
   :func:`~nfl.features.ingame.build_ingame`'s rows for the same plays.
3. **The GRU's serving probability is its offline probability.** The end of the
   chain: same snapshot, same model, same number to 1e-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nfl.data.live.base import AtTime, LiveSnapshot
from nfl.data.live.replay import HistoricalReplayFeed
from nfl.features.assemble import assemble_dataset
from nfl.features.ingame import (
    SNAPSHOT_UNKNOWABLE_COLUMNS,
    snapshot_history_rows,
    snapshot_to_row,
)
from nfl.inference.predict import (
    build_feature_row,
    build_feature_rows,
    needs_play_window,
    serving_meta,
    snapshot_to_baseline_row,
)
from nfl.models.feature_sets import (
    UNSERVABLE_COLUMNS,
    feature_set_columns,
    inplay_columns,
    servable_columns,
)

GAME_ID = "2024_01_BAL_KC"
AT = "Q4 05:00"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def game_pbp(pbp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(pbp_path)


@pytest.fixture(scope="module")
def game_schedules(schedules_path: Path) -> pl.DataFrame:
    return pl.read_parquet(schedules_path)


@pytest.fixture(scope="module")
def offline(game_pbp: pl.DataFrame, game_schedules: pl.DataFrame) -> pl.DataFrame:
    """The assembled dataset for the fixture game — the training rows."""
    return assemble_dataset([2024], pbp=game_pbp, schedules=game_schedules).sort("ig__play_index")


@pytest.fixture(scope="module")
def feed(game_pbp: pl.DataFrame, game_schedules: pl.DataFrame) -> HistoricalReplayFeed:
    return HistoricalReplayFeed(pbp=game_pbp, schedules=game_schedules)


@pytest.fixture(scope="module")
def snapshot(feed: HistoricalReplayFeed) -> LiveSnapshot:
    return feed.snapshot(GAME_ID, at=AT)


def _snap_clocks(offline: pl.DataFrame, n: int, low: int, high: int) -> list[tuple[int, int]]:
    """``(quarter_seconds, play_index)`` for ``n`` Q4 snaps between ``low``/``high``.

    Sampling at a play's **own** clock is what makes the comparison about the
    feature pipeline: ask for ``Q4 05:00`` and the snapshot reports 300 seconds
    while the play-by-play row for the snap that follows reports whatever the
    clock actually read, and every clock-derived column differs by that gap.
    """
    rows = offline.filter(
        (pl.col("ig__quarter") == 4)
        & pl.col("ig__quarter_seconds_remaining").is_between(low, high)
        & pl.col("ig__down").is_not_null()
    )
    picked: list[tuple[int, int]] = []
    seen: set[int] = set()
    for row in rows.iter_rows(named=True):
        secs = int(row["ig__quarter_seconds_remaining"])
        if secs in seen:
            continue
        seen.add(secs)
        picked.append((secs, int(row["ig__play_index"])))
        if len(picked) == n:
            break
    return picked


# --------------------------------------------------------------------------- #
# 1. The servable feature set
# --------------------------------------------------------------------------- #


def test_servable_is_inplay_minus_what_the_snapshot_cannot_fill(offline: pl.DataFrame) -> None:
    inplay = set(inplay_columns(offline))
    servable = set(servable_columns(offline))
    assert servable == inplay - UNSERVABLE_COLUMNS
    assert servable < inplay, "servable must actually drop something"
    # The 89 unknowable columns are the whole of the difference.
    assert inplay - servable == inplay & SNAPSHOT_UNKNOWABLE_COLUMNS


def test_servable_is_a_named_feature_set(offline: pl.DataFrame) -> None:
    assert feature_set_columns(offline, "servable") == servable_columns(offline)
    assert set(feature_set_columns(offline, "servable_v1")) >= set(
        feature_set_columns(offline, "servable")
    )


def test_every_servable_column_is_served_non_null(
    snapshot: LiveSnapshot, offline: pl.DataFrame
) -> None:
    """The load-bearing test for ``servable``: no column on the list is null.

    A column that is on the model's input list and arrives null at serve time is
    the exact failure this feature set exists to prevent — the model takes a
    different path down every tree it learned.
    """
    served = {**snapshot_to_row(snapshot), **snapshot_to_baseline_row(snapshot)}
    nulls = [c for c in servable_columns(offline) if served.get(c) is None]
    assert not nulls, f"{len(nulls)} servable column(s) null at serve time: {nulls[:20]}"


def test_the_unknowable_columns_really_are_null(snapshot: LiveSnapshot) -> None:
    """The other direction: the exclusion list is not excluding servable columns."""
    served = snapshot_to_row(snapshot)
    populated = [c for c in SNAPSHOT_UNKNOWABLE_COLUMNS if served.get(c) is not None]
    assert not populated


# --------------------------------------------------------------------------- #
# 2. The serving play history
# --------------------------------------------------------------------------- #


def test_the_history_is_one_row_per_play_plus_the_snap(snapshot: LiveSnapshot) -> None:
    rows = snapshot_history_rows(snapshot)
    assert len(rows) == snapshot.play_count + 1
    assert rows[-1] == snapshot_to_row(snapshot)
    assert [r["ig__play_index"] for r in rows] == list(range(len(rows)))


def test_build_feature_rows_ends_at_build_feature_row(
    snapshot: LiveSnapshot, game_pbp: pl.DataFrame, game_schedules: pl.DataFrame
) -> None:
    kwargs = {
        "game_id": GAME_ID,
        "schedules": game_schedules,
        "pbp": game_pbp,
        "use_cache": False,
    }
    rows = build_feature_rows(snapshot, **kwargs)  # type: ignore[arg-type]
    assert rows[-1] == build_feature_row(snapshot, **kwargs)  # type: ignore[arg-type]
    assert len(rows) == snapshot.play_count + 1


def test_the_serving_history_matches_build_ingame_row_for_row(
    snapshot: LiveSnapshot, offline: pl.DataFrame
) -> None:
    """Every history row equals the offline row for the same play.

    Restricted to ``servable``: the unknowable columns are null on the serving
    side by design, and the snap's own row is compared by the parity test that
    already owns it. What is new here is the *history* — if it drifts, the GRU's
    window is not the window it was trained on.
    """
    served = snapshot_history_rows(snapshot)[:-1]
    truth = offline.head(len(served)).to_dicts()
    columns = [c for c in servable_columns(offline) if c.startswith("ig__")]
    mismatched: list[str] = []
    for serve_row, offline_row in zip(served, truth, strict=True):
        assert serve_row["ig__play_index"] == offline_row["ig__play_index"]
        for name in columns:
            a, b = serve_row[name], offline_row[name]
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-6:
                mismatched.append(f"{name}@{offline_row['ig__play_index']}: {a} != {b}")
    assert not mismatched, mismatched[:20]


def test_serving_meta_keys_one_game_in_play_order() -> None:
    meta = serving_meta(GAME_ID, 5)
    assert meta.height == 5
    assert meta.get_column("meta__game_id").unique().to_list() == [GAME_ID]
    assert meta.get_column("meta__play_id").to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]


# --------------------------------------------------------------------------- #
# 3. The GRU sees the window it was trained on
# --------------------------------------------------------------------------- #


def test_a_sequence_model_is_detected_and_a_gbdt_is_not() -> None:
    from nfl.models.dummy import ConstantModel
    from nfl.models.tf_seq import SeqWinProb

    assert needs_play_window(SeqWinProb())
    assert not needs_play_window(ConstantModel(p=0.5))


@pytest.fixture(scope="module")
def fitted_seq(offline: pl.DataFrame):  # noqa: ANN201 - TF type
    """A tiny GRU fitted on the fixture game and its mirror, on ``servable``."""
    from nfl.models.data import design_matrix, labels
    from nfl.models.tf_seq import SeqWinProb
    from tests.models_fixture import assembled_seasons

    frame = assembled_seasons()
    features = feature_set_columns(frame, "servable")
    model = SeqWinProb(
        k=8,
        play_embed=4,
        gru_units=4,
        pregame_units=4,
        head_units=4,
        dropout=0.0,
        pregame_noise=0.0,
        epochs=2,
        batch_size=256,
        feature_set="servable",
        prefer_gpu=False,
        mixed_precision=False,
    )
    model.fit(
        design_matrix(frame, features),
        labels(frame),
        meta=frame.select("meta__game_id", "meta__play_id"),
    )
    return model, features


@pytest.mark.parametrize("index", [0, 1, 2])
def test_gru_serving_probability_equals_the_offline_probability(
    index: int,
    fitted_seq,  # noqa: ANN001 - TF type
    feed: HistoricalReplayFeed,
    offline: pl.DataFrame,
    game_pbp: pl.DataFrame,
    game_schedules: pl.DataFrame,
) -> None:
    """Same snapshot, same model, same probability — to 1e-4.

    Before the history rows existed this could not pass at all: the serving
    frame was one row, so the GRU's window held one play and the model on the
    serving path was a different model from the one that was measured.
    """
    from nfl.models.data import design_matrix

    model, features = fitted_seq
    clocks = _snap_clocks(offline, 3, 60, 600)
    assert len(clocks) == 3, "fixture should carry three distinct late-Q4 snaps"
    seconds, play_index = clocks[index]

    snapshot = feed.snapshot(GAME_ID, at=AtTime(quarter=4, clock_seconds=seconds))
    rows = build_feature_rows(
        snapshot, game_id=GAME_ID, schedules=game_schedules, pbp=game_pbp, use_cache=False
    )
    assert rows[-1]["ig__play_index"] == play_index, "snapshot landed on a different play"

    serving_frame = pl.DataFrame(
        {name: [None if r.get(name) is None else float(r[name]) for r in rows] for name in features}
    )
    p_serving = float(model.predict_proba(serving_frame, serving_meta(GAME_ID, len(rows)))[-1])

    offline_upto = offline.filter(pl.col("ig__play_index") <= play_index)
    p_offline = float(
        model.predict_proba(
            design_matrix(offline_upto, features),
            offline_upto.select("meta__game_id", "meta__play_id"),
        )[-1]
    )
    assert abs(p_serving - p_offline) < 1e-4, (
        f"play_index={play_index} ({seconds}s): serving {p_serving:.6f} vs offline {p_offline:.6f}"
    )


def test_a_one_row_frame_gives_the_gru_a_different_answer(
    fitted_seq,  # noqa: ANN001 - TF type
    offline: pl.DataFrame,
) -> None:
    """The bug this all exists for, pinned so it cannot come back quietly.

    Scoring the snap alone windows over one play. If that ever *stops* being
    different from scoring it in its game, the window has stopped mattering and
    the sequence model has become a tabular one.
    """
    from nfl.models.data import design_matrix

    model, features = fitted_seq
    upto = offline.head(60)
    windowed = model.predict_proba(
        design_matrix(upto, features), upto.select("meta__game_id", "meta__play_id")
    )[-1]
    alone = model.predict_proba(
        design_matrix(upto.tail(1), features), upto.tail(1).select("meta__game_id", "meta__play_id")
    )[0]
    assert not np.isclose(windowed, alone, atol=1e-4)


# --------------------------------------------------------------------------- #
# 4. The promotion gate's serving-parity condition
# --------------------------------------------------------------------------- #


@pytest.fixture
def parity_registry(tmp_path: Path):  # noqa: ANN201 - xgboost type
    """A registry holding a floor and a real XGBoost candidate on ``servable``.

    The candidate is fitted on the fixture game and its mirror — two real games,
    both outcomes — and both manifests are stocked with ``val_slices`` that make
    the *offline* half of the gate pass, so the serving-parity condition is the
    only thing left under test.
    """
    import json

    from nfl.models.data import design_matrix, labels
    from nfl.models.gbdt import XGBoostWinProb
    from nfl.models.registry import model_dir, save_model
    from nfl.models.train_ensemble import GATE_Q4_SLICES, NOT_COLLAPSED_SLICE
    from tests.models_fixture import assembled_games

    def slices(log_loss: float, brier: float) -> list[dict[str, float]]:
        rows = [{"slice": s, "log_loss": log_loss, "brier": brier, "n": 10} for s in GATE_Q4_SLICES]
        rows.append({"slice": NOT_COLLAPSED_SLICE, "log_loss": log_loss, "brier": brier, "n": 10})
        return rows

    root = tmp_path / "models"
    frame = assembled_games()
    features = feature_set_columns(frame, "servable")
    model = XGBoostWinProb(device="cpu", num_boost_round=24, early_stopping_rounds=8)
    design, y = design_matrix(frame, features), labels(frame)
    model.fit(design, y, design, y)
    manifest = save_model(model, name="cand", version="20240102T000000", root=root)
    manifest.feature_names = list(model.encoding.columns)
    manifest.extra.update(
        {
            "feature_set": "servable",
            "val_slices": slices(0.20, 0.06),
            "not_collapsed_reference": "naive_score_time/20240101T000000",
        }
    )
    (model_dir("cand", "20240102T000000", root) / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )

    floor = model_dir("naive_score_time", "20240101T000000", root)
    floor.mkdir(parents=True, exist_ok=True)
    (floor / "manifest.json").write_text(
        json.dumps(
            {
                "name": "naive_score_time",
                "version": "20240101T000000",
                "val_slices": slices(0.30, 0.10),
            }
        )
    )
    return root, features


def _frames(offline: pl.DataFrame, features: list[str]):  # noqa: ANN202
    from nfl.models.data import ModelFrames

    return ModelFrames(train=offline, val=offline, test=offline.head(0), features=list(features))


def test_serving_parity_measures_both_paths_and_they_agree(
    parity_registry,  # noqa: ANN001
    offline: pl.DataFrame,
    game_pbp: pl.DataFrame,
    game_schedules: pl.DataFrame,
) -> None:
    """A GBDT on ``servable``, served through ``predict_game``, matches offline.

    This is the condition end to end: no stubs on either side, the real replay
    feed, the real feature builders, the real artifact.
    """
    from nfl.models.train_ensemble import serving_parity

    root, features = parity_registry
    parity = serving_parity(
        "cand",
        "20240102T000000",
        n=3,
        root=root,
        frames=_frames(offline, features),
        pbp=game_pbp,
        schedules=game_schedules,
    )
    assert parity.samples, f"no comparable samples; skipped: {parity.skipped}"
    assert parity.passed, parity.table()
    assert parity.worst < 0.02
    assert set(parity.table().columns) >= {"game_id", "p_serving", "p_offline", "delta"}


def test_the_gate_refuses_a_candidate_whose_serving_path_drifts(
    parity_registry,  # noqa: ANN001
    offline: pl.DataFrame,
    game_pbp: pl.DataFrame,
    game_schedules: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline numbers that pass are not enough if the served number differs.

    The candidate's manifest says it beats the floor on every slice. The serving
    path is then made to answer 0.5 for everything — the shape of the bug this
    condition exists for, where the model that reaches production is not the
    model that was measured — and the gate has to refuse it on that alone.
    """
    from nfl.inference import predict as predict_mod
    from nfl.models import train_ensemble
    from nfl.models.train_ensemble import SERVING_PARITY_SLICE, promotion_gate

    root, features = parity_registry
    real = predict_mod.predict_game

    def drifting(*args: object, **kwargs: object):  # noqa: ANN202
        kwargs.setdefault("pbp", game_pbp)
        kwargs.setdefault("schedules", game_schedules)
        return predict_mod._replace(real(*args, **kwargs), p_home=0.5, p_away=0.5)  # noqa: SLF001

    monkeypatch.setattr(predict_mod, "predict_game", drifting)
    monkeypatch.setattr(train_ensemble, "load_model_frames", lambda **_: _frames(offline, features))

    result = promotion_gate("cand", "20240102T000000", root=root, parity_samples=3)
    parity_checks = [c for c in result.checks if c.slice == SERVING_PARITY_SLICE]
    assert len(parity_checks) == 1
    assert not parity_checks[0].passed
    assert all(c.passed for c in result.checks if c.slice != SERVING_PARITY_SLICE)
    assert not result.passed, "the offline slices pass; only serving parity should refuse it"
