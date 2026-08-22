"""``coach_records``: correct counts, and — the point of the module — no leakage.

The fixture schedule is hand-built so every expected number below can be checked
by counting games. See ``tests/fixtures/build_fixtures.py`` for the layout.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nfl.data.coach_records import (
    build_coach_records,
    coach_records_path,
    ingest_coach_records,
)
from nfl.data.nflverse import ingest_nflverse


@pytest.fixture
def records(schedules_fixture: pl.DataFrame) -> pl.DataFrame:
    return build_coach_records(schedules_fixture)


def _row(records: pl.DataFrame, game_id: str) -> dict:
    return records.filter(pl.col("game_id") == game_id).row(0, named=True)


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #


def test_first_ever_game_has_an_empty_record(records: pl.DataFrame) -> None:
    row = _row(records, "2023_01_DEN_KC")
    assert row["home_coach"] == "Andy Reid"
    assert row["home_coach_career_wins"] == 0
    assert row["home_coach_career_losses"] == 0
    assert row["home_coach_career_ties"] == 0
    assert row["home_coach_career_games"] == 0
    assert row["home_coach_career_win_pct"] == 0.0


def test_record_accumulates_across_the_season(records: pl.DataFrame) -> None:
    """Reid: W in week 1, so week 2 sees 1-0-0."""
    row = _row(records, "2023_02_KC_BUF")
    assert row["away_coach"] == "Andy Reid"
    assert row["away_coach_career_wins"] == 1
    assert row["away_coach_career_losses"] == 0
    assert row["away_coach_season_wins"] == 1
    assert row["away_coach_season_games"] == 1


def test_ties_count_as_half_a_win(records: pl.DataFrame) -> None:
    """Reid enters the divisional round 1-0-1 (week 1 W, week 2 T)."""
    row = _row(records, "2023_19_BUF_KC")
    assert row["home_coach"] == "Andy Reid"
    assert (
        row["home_coach_career_wins"],
        row["home_coach_career_losses"],
        row["home_coach_career_ties"],
    ) == (1, 0, 1)
    assert row["home_coach_career_games"] == 2
    assert row["home_coach_career_win_pct"] == pytest.approx(0.75)


def test_career_carries_across_seasons_while_season_resets(records: pl.DataFrame) -> None:
    """2024 week 1: Reid is 2-0-1 for his career and 0-0-0 on the season."""
    row = _row(records, "2024_01_DEN_KC")
    assert row["home_coach"] == "Andy Reid"
    assert row["home_coach_career_wins"] == 2
    assert row["home_coach_career_ties"] == 1
    assert row["home_coach_career_games"] == 3
    assert row["home_coach_season_wins"] == 0
    assert row["home_coach_season_games"] == 0


def test_record_follows_the_coach_not_the_team(records: pl.DataFrame) -> None:
    """McDermott's away record at DEN continues his BUF record — coaches, not teams."""
    row = _row(records, "2024_03_BUF_DEN")
    assert row["away_coach"] == "Sean McDermott"
    # 2023: W, T, L(post); 2024: W  ->  2-1-1 entering week 3.
    assert (
        row["away_coach_career_wins"],
        row["away_coach_career_losses"],
        row["away_coach_career_ties"],
    ) == (2, 1, 1)
    assert row["away_coach_season_wins"] == 1
    assert row["away_coach_season_losses"] == 0


def test_both_sides_are_populated(records: pl.DataFrame) -> None:
    row = _row(records, "2024_02_KC_MIA")
    assert row["away_coach"] == "Andy Reid"
    assert row["home_coach"] == "Mike McDaniel"
    assert row["away_coach_career_wins"] == 3  # 2023 W, W(post); 2024 W
    assert row["home_coach_career_losses"] == 2  # 2023 L; 2024 L


def test_unplayed_games_contribute_nothing(records: pl.DataFrame) -> None:
    """A scheduled-but-unplayed game still gets a row, and adds to nobody's record."""
    row = _row(records, "2024_04_KC_BUF")
    assert row["away_coach"] == "Andy Reid"
    assert row["away_coach_career_games"] == 5  # every completed game so far, no more
    assert row["away_coach_season_games"] == 2


# --------------------------------------------------------------------------- #
# Leakage — mandatory per CLAUDE.md rule 3
# --------------------------------------------------------------------------- #


def test_current_game_never_enters_its_own_record(
    records: pl.DataFrame, schedules_fixture: pl.DataFrame
) -> None:
    """For every row, the record equals the count over strictly earlier games.

    Recomputed here by brute force rather than by the module's own machinery, so
    a bug in the shift/cumsum cannot make the test agree with itself.
    """
    played = schedules_fixture.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    history: list[tuple[str, str, str]] = []  # (gameday, coach, outcome)
    for game in played.sort("gameday", "gametime").iter_rows(named=True):
        margin = game["home_score"] - game["away_score"]
        history.append(
            (game["gameday"], game["home_coach"], "W" if margin > 0 else "L" if margin < 0 else "T")
        )
        history.append(
            (game["gameday"], game["away_coach"], "L" if margin > 0 else "W" if margin < 0 else "T")
        )

    for row in records.iter_rows(named=True):
        for side in ("home", "away"):
            coach = row[f"{side}_coach"]
            prior = [o for day, c, o in history if c == coach and day < row["gameday"]]
            assert row[f"{side}_coach_career_wins"] == prior.count("W"), (row["game_id"], side)
            assert row[f"{side}_coach_career_losses"] == prior.count("L"), (row["game_id"], side)
            assert row[f"{side}_coach_career_ties"] == prior.count("T"), (row["game_id"], side)


def test_deleting_the_future_does_not_change_the_past(
    schedules_fixture: pl.DataFrame, records: pl.DataFrame
) -> None:
    """The leakage taxonomy's as-of test: truncate later games, recompute, compare.

    If any later game were reaching backwards into an earlier row, the truncated
    recomputation would differ.
    """
    cutoff = "2024-09-15"
    truncated = build_coach_records(schedules_fixture.filter(pl.col("gameday") <= cutoff))
    full = records.filter(pl.col("gameday") <= cutoff)

    metric_columns = [c for c in truncated.columns if "_coach_" in c]
    key = ["game_id", *metric_columns]
    assert truncated.sort("game_id").select(key).equals(full.sort("game_id").select(key))


def test_a_coachs_own_win_is_visible_only_on_later_rows(records: pl.DataFrame) -> None:
    """Direct statement of the invariant, on the one game it is easiest to see."""
    before = _row(records, "2023_01_DEN_KC")["home_coach_career_wins"]
    after = _row(records, "2023_02_KC_BUF")["away_coach_career_wins"]
    assert before == 0
    assert after == 1


# --------------------------------------------------------------------------- #
# Contract / ingest
# --------------------------------------------------------------------------- #


def test_missing_input_columns_are_rejected(schedules_fixture: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_coach_records(schedules_fixture.drop("home_coach"))


def test_records_carry_no_market_columns(records: pl.DataFrame) -> None:
    from nfl.data.market_blocklist import assert_no_market_columns

    assert_no_market_columns(records, context="coach_records")


def test_ingest_writes_season_partitions(tmp_path: Path, offline_nflverse: dict) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    ingest_nflverse([2023, 2024], ["schedules"], root=raw)

    out = ingest_coach_records([2023, 2024], raw_root=raw, processed_root=processed)
    assert out == processed / "coach_records"

    for season in (2023, 2024):
        path = coach_records_path(season, processed)
        assert path.exists()
        assert pl.read_parquet(path)["season"].unique().to_list() == [season]


def test_ingest_needs_schedules_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest nflverse"):
        ingest_coach_records([2023], raw_root=tmp_path / "raw", processed_root=tmp_path / "out")
