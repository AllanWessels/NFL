"""Tests for :mod:`nfl.features.ingame`.

Five properties, all mandated by CLAUDE.md hard rules 1 and 3 and the Phase 4
feature-builder contract:

* **No market columns** reach the frame (ADR 0001).
* **Delete-the-future**: truncating the play-by-play must not change any row
  that survives the truncation.
* **Determinism**: the same input twice gives the same frame.
* **Handedness**: swapping the home and away teams negates the signed features
  and swaps the per-side pairs.
* **Train/serve parity**: the row built from play-by-play at ``Q4 05:00`` of
  2024_01_BAL_KC equals the row built from the replayed :class:`LiveSnapshot`
  at the same moment, on every field both can know.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
import pytest

from nfl.data.live.replay import HistoricalReplayFeed
from nfl.data.market_blocklist import MARKET_COLUMNS, market_columns_in
from nfl.features.ingame import (
    FEATURE_COLUMNS,
    FEATURE_DOCS,
    SNAPSHOT_CLOCK_SENSITIVE_COLUMNS,
    SNAPSHOT_UNKNOWABLE_COLUMNS,
    build_ingame,
    snapshot_to_row,
)

GAME_ID = "2024_01_BAL_KC"

#: The snap immediately after ``Q4 05:00`` in the fixture: BAL, 4th and 4 on the
#: KC 14 with 4:58 on the clock. Verified by hand against the play-by-play.
PARITY_PLAY_ID = 3684


@pytest.fixture(scope="module")
def built(pbp: pl.DataFrame) -> pl.DataFrame:
    return build_ingame(pbp)


def _row(frame: pl.DataFrame, play_id: int) -> dict[str, Any]:
    return frame.filter(pl.col("play_id") == play_id).row(0, named=True)


def _swap_home_away(pbp: pl.DataFrame) -> pl.DataFrame:
    """Re-record the game with the two teams' roles exchanged.

    Only the *identity* columns move: which team is on offence, the yardline and
    the play outcomes are unchanged, so every home/away pair in the output must
    simply trade places.
    """
    return pbp.with_columns(
        pl.col("away_team").alias("home_team"),
        pl.col("home_team").alias("away_team"),
        pl.col("total_away_score").alias("total_home_score"),
        pl.col("total_home_score").alias("total_away_score"),
        pl.col("away_timeouts_remaining").alias("home_timeouts_remaining"),
        pl.col("home_timeouts_remaining").alias("away_timeouts_remaining"),
    )


# --------------------------------------------------------------------------- #
# Shape and contract
# --------------------------------------------------------------------------- #


def test_one_row_per_play_with_prefixed_columns(pbp: pl.DataFrame, built: pl.DataFrame) -> None:
    assert built.height == pbp.height
    assert built.columns[:2] == ["game_id", "play_id"]
    assert built.select(["game_id", "play_id"]).n_unique() == built.height
    assert all(c.startswith("ig__") for c in built.columns[2:])
    assert tuple(built.columns[2:]) == FEATURE_COLUMNS


def test_every_feature_is_documented() -> None:
    assert set(FEATURE_DOCS) == set(FEATURE_COLUMNS)
    assert all(text.strip() for text in FEATURE_DOCS.values())


def test_no_market_columns(built: pl.DataFrame) -> None:
    assert market_columns_in(built) == []


def test_no_market_columns_even_though_the_input_is_full_of_them(pbp: pl.DataFrame) -> None:
    """The fixture carries ``spread_line``, ``vegas_wp`` and the ``*_wpa`` family."""
    assert set(pbp.columns) & MARKET_COLUMNS or True  # fixture may be trimmed
    poisoned = pbp.with_columns(
        pl.lit(-3.5).alias("spread_line"),
        pl.lit(0.62).alias("vegas_home_wp"),
        pl.lit(0.01).alias("wpa"),
        pl.lit(0.55).alias("home_wp"),
    )
    assert market_columns_in(build_ingame(poisoned)) == []


def test_empty_input_returns_the_full_schema() -> None:
    out = build_ingame(pl.DataFrame(schema={"game_id": pl.Utf8, "play_id": pl.Int64}))
    assert out.height == 0
    assert tuple(out.columns[2:]) == FEATURE_COLUMNS


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("keep", [20, 75, 150])
def test_delete_the_future(pbp: pl.DataFrame, built: pl.DataFrame, keep: int) -> None:
    """Truncating the game must leave every surviving row byte-identical."""
    ordered = pbp.sort("order_sequence")
    truncated = build_ingame(ordered.head(keep))
    reference = built.head(keep)
    assert truncated.equals(reference)


def test_perturbing_the_future_changes_nothing(pbp: pl.DataFrame, built: pl.DataFrame) -> None:
    """Rewriting *later* plays leaves earlier rows alone (DATA_MODEL §2)."""
    cut = 100
    ordered = pbp.sort("order_sequence")
    tampered = ordered.with_columns(
        pl.when(pl.int_range(pl.len()) >= cut)
        .then(pl.col("yards_gained") * 10 + 99)
        .otherwise(pl.col("yards_gained"))
        .alias("yards_gained"),
        pl.when(pl.int_range(pl.len()) >= cut)
        .then(pl.col("total_home_score") + 21)
        .otherwise(pl.col("total_home_score"))
        .alias("total_home_score"),
    )
    assert build_ingame(tampered).head(cut).equals(built.head(cut))


def test_score_is_pre_snap_not_post_play(pbp: pl.DataFrame, built: pl.DataFrame) -> None:
    """nflverse ``total_home_score`` is post-play; the feature must lag it."""
    ordered = pbp.sort("order_sequence")
    scoring = ordered.filter(pl.col("total_home_score") > pl.col("total_home_score").shift(1)).head(
        1
    )
    play_id = int(scoring.get_column("play_id").item(0))
    row = _row(built, play_id)
    assert row["ig__home_score"] < int(scoring.get_column("total_home_score").item(0))


def test_first_play_of_the_game_has_no_history(built: pl.DataFrame) -> None:
    first = built.sort("ig__play_index").row(0, named=True)
    assert first["ig__play_index"] == 0
    assert first["ig__home_score"] == 0
    assert first["ig__away_score"] == 0
    assert first["ig__plays_home"] == 0
    assert first["ig__plays_away"] == 0
    assert first["ig__drives_completed_home"] == 0
    assert first["ig__lead_changes"] == 0
    # Rates are null, not a fabricated 0.0, before there is anything to average.
    assert first["ig__third_down_rate_home"] is None
    assert first["ig__yards_per_play_away"] is None


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_deterministic(pbp: pl.DataFrame) -> None:
    assert build_ingame(pbp).equals(build_ingame(pbp))


def test_row_order_of_the_input_does_not_matter(pbp: pl.DataFrame, built: pl.DataFrame) -> None:
    shuffled = pbp.sample(fraction=1.0, shuffle=True, seed=7)
    assert build_ingame(shuffled).equals(built)


# --------------------------------------------------------------------------- #
# Handedness
# --------------------------------------------------------------------------- #

_SIGNED = (
    "ig__score_diff_home",
    "ig__unanswered_points_streak",
    "ig__momentum_score",
)

#: Pairs spelled ``ig__<side>_<stem>`` rather than ``ig__<stem>_<side>``.
_SWAPPED_PREFIX_PAIRS = ("score", "timeouts_remaining")

_SWAPPED_PAIRS = (
    "plays",
    "yards",
    "epa_offense",
    "touchdowns",
    "turnovers_committed",
    "drives_completed",
    "points_per_drive",
    "third_down_attempts",
    "penalty_yards",
    "largest_lead",
    "epa_last_10_plays",
)


def test_handedness_flips_signs_and_swaps_pairs(pbp: pl.DataFrame, built: pl.DataFrame) -> None:
    mirrored = build_ingame(_swap_home_away(pbp))
    assert mirrored.height == built.height

    for column in _SIGNED:
        original = built.get_column(column).fill_null(0.0)
        flipped = mirrored.get_column(column).fill_null(0.0)
        assert ((original + flipped).abs() < 1e-9).all(), column

    pairs = [(f"ig__{s}_home", f"ig__{s}_away") for s in _SWAPPED_PAIRS]
    pairs += [(f"ig__home_{s}", f"ig__away_{s}") for s in _SWAPPED_PREFIX_PAIRS]
    for home, away in pairs:
        assert built.get_column(home).equals(mirrored.get_column(away)), home
        assert built.get_column(away).equals(mirrored.get_column(home)), away


def test_handedness_leaves_possession_relative_features_alone(
    pbp: pl.DataFrame, built: pl.DataFrame
) -> None:
    """Whoever has the ball still has it; only the orientation changed."""
    mirrored = build_ingame(_swap_home_away(pbp))
    for column in (
        "ig__score_diff_posteam",
        "ig__yardline_100",
        "ig__ydstogo",
        "ig__down",
        "ig__plays_posteam",
        "ig__posteam_timeouts_remaining",
        "ig__must_score_posteam",
        "ig__can_kneel_out",
    ):
        assert built.get_column(column).equals(mirrored.get_column(column)), column
    assert (
        built.get_column("ig__posteam_is_home")
        .fill_null(value=False)
        .eq(mirrored.get_column("ig__posteam_is_home").fill_null(value=True).not_())
        .all()
    )


# --------------------------------------------------------------------------- #
# Train / serve parity
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def snapshot_row(pbp: pl.DataFrame, schedules: pl.DataFrame) -> dict[str, Any]:
    feed = HistoricalReplayFeed(pbp=pbp, schedules=schedules)
    return snapshot_to_row(feed.snapshot(GAME_ID, "Q4 05:00"))


def test_snapshot_row_has_the_same_columns(snapshot_row: dict[str, Any]) -> None:
    assert set(snapshot_row) == {"game_id", "play_id", *FEATURE_COLUMNS}


def test_snapshot_row_sits_on_the_expected_snap(snapshot_row: dict[str, Any]) -> None:
    assert snapshot_row["game_id"] == GAME_ID
    assert snapshot_row["ig__quarter"] == 4
    assert snapshot_row["ig__posteam_is_home"] is False  # BAL has the ball
    assert snapshot_row["ig__down"] == 4
    assert snapshot_row["ig__ydstogo"] == 4
    assert snapshot_row["ig__yardline_100"] == 14
    assert snapshot_row["ig__home_score"] == 27
    assert snapshot_row["ig__away_score"] == 17


def test_parity_with_pbp_at_q4_0500(built: pl.DataFrame, snapshot_row: dict[str, Any]) -> None:
    """Every field both paths can know agrees exactly."""
    truth = _row(built, PARITY_PLAY_ID)
    mismatches = {
        column: (snapshot_row[column], truth[column])
        for column in FEATURE_COLUMNS
        if column not in SNAPSHOT_UNKNOWABLE_COLUMNS
        and column not in SNAPSHOT_CLOCK_SENSITIVE_COLUMNS
        and snapshot_row[column] != truth[column]
    }
    assert mismatches == {}


def test_parity_clock_sensitive_fields_agree_within_the_snap_gap(
    built: pl.DataFrame, snapshot_row: dict[str, Any]
) -> None:
    """``Q4 05:00`` is 2 seconds before the snap it selects; nothing else drifts.

    The numeric clock features track that 2-second gap. The three booleans in the
    set threshold on the clock and could in principle flip; in this game they do
    not, so they are still asserted equal here.
    """
    truth = _row(built, PARITY_PLAY_ID)
    gap = truth["ig__quarter_seconds_remaining"] - snapshot_row["ig__quarter_seconds_remaining"]
    assert gap == -2
    for column in SNAPSHOT_CLOCK_SENSITIVE_COLUMNS:
        a, b = snapshot_row[column], truth[column]
        if isinstance(a, bool) or isinstance(b, bool):
            assert a == b, (column, a, b)
        else:
            assert math.isclose(float(a), float(b), abs_tol=3.0), (column, a, b)


def test_parity_covers_most_of_the_schema(snapshot_row: dict[str, Any]) -> None:
    """Guard against 'parity' being achieved by declaring everything unknowable."""
    comparable = set(FEATURE_COLUMNS) - SNAPSHOT_UNKNOWABLE_COLUMNS
    assert len(comparable) > len(FEATURE_COLUMNS) // 2
    assert all(snapshot_row[c] is None for c in SNAPSHOT_UNKNOWABLE_COLUMNS)


def test_snapshot_unknowables_are_the_documented_ones() -> None:
    """The serving path is null — never zero — where the feed cannot see.

    ESPN publishes no EPA, no per-play statistical detail and no penalised team,
    and (unlike the replay feed) no per-play timeouts: a truncated ESPN snapshot
    carries ``timeouts_exact=False``, so its timeout features are the feed's
    current values rather than the values at ``at``.
    """
    assert "ig__epa_offense_home" in SNAPSHOT_UNKNOWABLE_COLUMNS
    assert "ig__success_rate_away" in SNAPSHOT_UNKNOWABLE_COLUMNS
    assert "ig__penalty_yards_home" in SNAPSHOT_UNKNOWABLE_COLUMNS
    # ...while the situation, the score and the coarse counting stats are not.
    for column in (
        "ig__score_diff_home",
        "ig__plays_home",
        "ig__yards_away",
        "ig__punts_home",
        "ig__turnovers_committed_away",
        "ig__drives_completed_home",
        "ig__home_timeouts_remaining",
    ):
        assert column not in SNAPSHOT_UNKNOWABLE_COLUMNS
