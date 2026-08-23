"""``nfl.inference.resolve``: every spelling a human might type for one game.

The schedules frame here is a hand-built fixture rather than the committed
nflverse one, because the interesting cases are *shapes* the real table only
occasionally contains: a pair that plays twice in a season, two games for one
team inside a single ±7-day window, and a Thursday game that is already over
while a Sunday game is still upcoming.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from typer.testing import CliRunner

from nfl.cli import app
from nfl.inference.resolve import (
    ACCEPTED_FORMATS,
    GameSpecError,
    resolve_game,
    resolve_team,
    teams,
)

runner = CliRunner()

#: The 32 current franchises, in nflverse's spelling (``LA``, not ``LAR``).
CURRENT_TEAMS = frozenset(
    [
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LA",
        "LAC",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ]
)

# game_id, season, game_type, week, gameday, gametime (ET), away, home, espn
_ROWS = [
    ("2024_10_DAL_PHI", 2024, "REG", 10, "2024-11-10", "13:00", "DAL", "PHI", "401671001"),
    ("2025_01_DAL_PHI", 2025, "REG", 1, "2025-09-04", "20:20", "DAL", "PHI", "401772510"),
    ("2025_01_KC_LAC", 2025, "REG", 1, "2025-09-05", "20:00", "KC", "LAC", "401772714"),
    ("2025_02_PHI_KC", 2025, "REG", 2, "2025-09-14", "16:25", "PHI", "KC", "401772600"),
    ("2025_03_NYG_NYJ", 2025, "REG", 3, "2025-09-21", "13:00", "NYG", "NYJ", "401772601"),
    ("2025_04_PHI_TB", 2025, "REG", 4, "2025-09-28", "13:00", "PHI", "TB", "401772602"),
    # Thursday night, then Sunday afternoon: one team, two games, one window.
    ("2025_05_SF_LA", 2025, "REG", 5, "2025-10-02", "20:15", "SF", "LA", "401772603"),
    ("2025_06_LA_SEA", 2025, "REG", 6, "2025-10-05", "16:05", "LA", "SEA", "401772604"),
    # The divisional rematch: DAL/PHI play twice in 2025.
    ("2025_12_PHI_DAL", 2025, "REG", 12, "2025-11-27", "16:30", "PHI", "DAL", "401772605"),
]

#: A Friday lunchtime: the Thursday game is over, the Sunday game is not.
THAT_FRIDAY = datetime(2025, 10, 3, 12, 0)


@pytest.fixture(scope="module")
def schedules() -> pl.DataFrame:
    return pl.DataFrame(
        _ROWS,
        schema=[
            ("game_id", pl.String),
            ("season", pl.Int32),
            ("game_type", pl.String),
            ("week", pl.Int32),
            ("gameday", pl.String),
            ("gametime", pl.String),
            ("away_team", pl.String),
            ("home_team", pl.String),
            ("espn", pl.String),
        ],
        orient="row",
    )


# --------------------------------------------------------------------------- #
# The alias table
# --------------------------------------------------------------------------- #


def test_alias_table_covers_every_current_team() -> None:
    table = teams()
    assert len(table) == 32
    assert {t.abbr for t in table} == CURRENT_TEAMS


def test_alias_table_has_no_duplicate_rows() -> None:
    abbrs = [t.abbr for t in teams()]
    assert len(abbrs) == len(set(abbrs))


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("PHI", "PHI"),
        ("phi", "PHI"),
        ("Eagles", "PHI"),
        ("Philadelphia", "PHI"),
        ("Philadelphia Eagles", "PHI"),
        ("Niners", "SF"),
        ("49ers", "SF"),
        ("Bucs", "TB"),
        ("Jags", "JAX"),
        ("JAC", "JAX"),
        ("Pats", "NE"),
        ("Skins", "WAS"),
        ("WSH", "WAS"),
        ("LAR", "LA"),
        ("Rams", "LA"),
        ("Oakland Raiders", "LV"),
        ("green bay", "GB"),
    ],
)
def test_resolve_team_accepts_the_spellings_people_use(word: str, expected: str) -> None:
    assert resolve_team(word) == expected


def test_a_word_that_names_two_teams_is_refused_not_guessed() -> None:
    with pytest.raises(GameSpecError) as caught:
        resolve_team("New York")
    assert "NYG" in str(caught.value)
    assert "NYJ" in str(caught.value)


def test_los_angeles_is_also_refused() -> None:
    with pytest.raises(GameSpecError, match="more than one team"):
        resolve_team("Los Angeles")


# --------------------------------------------------------------------------- #
# Every input form
# --------------------------------------------------------------------------- #


def test_nflverse_game_id_passes_through(schedules: pl.DataFrame) -> None:
    ref = resolve_game("2025_01_DAL_PHI", schedules=schedules)
    assert (ref.game_id, ref.how, ref.note) == ("2025_01_DAL_PHI", "game_id", None)
    assert (ref.away_team, ref.home_team) == ("DAL", "PHI")
    assert (ref.season, ref.week) == (2025, 1)
    assert ref.espn == "401772510"


def test_lowercase_game_id_is_normalised(schedules: pl.DataFrame) -> None:
    ref = resolve_game("2025_1_dal_phi", schedules=schedules)
    assert ref.game_id == "2025_01_DAL_PHI"
    assert ref.note is not None and "normalised" in ref.note


def test_a_game_id_not_in_the_schedules_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="not in the schedules table"):
        resolve_game("2025_99_DAL_PHI", schedules=schedules)


def test_at_sign_makes_the_left_side_away(schedules: pl.DataFrame) -> None:
    ref = resolve_game("DAL@PHI", schedules=schedules, season=2025, week=1)
    assert (ref.game_id, ref.how) == ("2025_01_DAL_PHI", "matchup")


def test_the_word_at_makes_the_left_side_away(schedules: pl.DataFrame) -> None:
    ref = resolve_game("DAL at PHI", schedules=schedules, season=2025, week=1)
    assert ref.game_id == "2025_01_DAL_PHI"


def test_vs_makes_the_left_side_home(schedules: pl.DataFrame) -> None:
    ref = resolve_game("PHI vs DAL", schedules=schedules, season=2025, week=1)
    assert ref.game_id == "2025_01_DAL_PHI"
    assert ref.home_team == "PHI"


def test_vs_and_at_disagree_about_the_same_pair(schedules: pl.DataFrame) -> None:
    """``DAL vs PHI`` is the *other* game: Dallas hosting."""
    ref = resolve_game("DAL vs PHI", schedules=schedules, season=2025)
    assert ref.game_id == "2025_12_PHI_DAL"


def test_nicknames_work_on_both_sides(schedules: pl.DataFrame) -> None:
    ref = resolve_game("Cowboys at Eagles", schedules=schedules, season=2025, week=1)
    assert ref.game_id == "2025_01_DAL_PHI"


def test_a_wildcard_side_means_any_opponent(schedules: pl.DataFrame) -> None:
    ref = resolve_game("AWAY@PHI", schedules=schedules, season=2025, week=1)
    assert ref.game_id == "2025_01_DAL_PHI"


def test_both_sides_wildcard_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="names no team"):
        resolve_game("any@any", schedules=schedules)


def test_a_backwards_matchup_resolves_and_says_so(schedules: pl.DataFrame) -> None:
    ref = resolve_game("PHI@DAL", schedules=schedules, season=2025, week=1)
    assert ref.game_id == "2025_01_DAL_PHI"
    assert "swap" in ref.how
    assert ref.note is not None and "swap" in ref.note


def test_a_single_team_resolves_to_its_game_this_week(schedules: pl.DataFrame) -> None:
    ref = resolve_game("PHI", schedules=schedules, today=datetime(2025, 9, 3))
    assert (ref.game_id, ref.how) == ("2025_01_DAL_PHI", "team")


def test_a_single_team_by_nickname(schedules: pl.DataFrame) -> None:
    ref = resolve_game("Eagles", schedules=schedules, today=datetime(2025, 9, 3))
    assert ref.game_id == "2025_01_DAL_PHI"


def test_a_single_team_by_city(schedules: pl.DataFrame) -> None:
    ref = resolve_game("philadelphia", schedules=schedules, today=datetime(2025, 9, 3))
    assert ref.game_id == "2025_01_DAL_PHI"


def test_a_kalshi_event_ticker(schedules: pl.DataFrame) -> None:
    ref = resolve_game("KXNFLGAME-25SEP04DALPHI", schedules=schedules)
    assert (ref.game_id, ref.how) == ("2025_01_DAL_PHI", "kalshi")


def test_a_kalshi_market_ticker_drops_its_team_suffix(schedules: pl.DataFrame) -> None:
    ref = resolve_game("KXNFLGAME-25SEP04DALPHI-PHI", schedules=schedules)
    assert ref.game_id == "2025_01_DAL_PHI"


def test_a_kalshi_ticker_carries_its_own_clock(schedules: pl.DataFrame) -> None:
    """The ticker's date is the window centre, so `today` cannot change the answer."""
    ref = resolve_game("KXNFLGAME-25SEP04DALPHI", schedules=schedules, today=datetime(2031, 1, 1))
    assert ref.game_id == "2025_01_DAL_PHI"


def test_an_unreadable_kalshi_ticker_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="cannot read a team pair"):
        resolve_game("KXNFLGAME-25SEP04ZZZQQQ", schedules=schedules)


def test_an_espn_event_id(schedules: pl.DataFrame) -> None:
    ref = resolve_game("401772510", schedules=schedules)
    assert (ref.game_id, ref.how) == ("2025_01_DAL_PHI", "espn")


def test_an_unknown_espn_event_id_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="ESPN event id"):
        resolve_game("999999999", schedules=schedules)


# --------------------------------------------------------------------------- #
# "This week", and what happens when it is ambiguous
# --------------------------------------------------------------------------- #


def test_two_games_in_the_window_pick_the_upcoming_one_and_report_it(
    schedules: pl.DataFrame,
) -> None:
    ref = resolve_game("Rams", schedules=schedules, today=THAT_FRIDAY)
    assert ref.game_id == "2025_06_LA_SEA"
    assert ref.note is not None
    assert "2025_05_SF_LA" in ref.note
    assert "2025_06_LA_SEA" in ref.note


def test_asking_for_a_past_moment_takes_the_nearest_kickoff_instead(
    schedules: pl.DataFrame,
) -> None:
    """``--at`` is a request to rewind, so a just-finished game is fair game."""
    ref = resolve_game("Rams", schedules=schedules, today=THAT_FRIDAY, prefer_upcoming=False)
    assert ref.game_id == "2025_05_SF_LA"
    assert ref.note is not None


def test_a_pair_that_plays_twice_needs_a_week_or_a_season(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError) as caught:
        resolve_game("DAL@PHI", schedules=schedules, today=datetime(2026, 8, 23))
    message = str(caught.value)
    assert "--week" in message and "--season" in message
    assert "2024_10_DAL_PHI" in message


def test_a_week_disambiguates_the_rematch(schedules: pl.DataFrame) -> None:
    ref = resolve_game("PHI@DAL", schedules=schedules, season=2025, week=12)
    assert ref.game_id == "2025_12_PHI_DAL"
    assert ref.how == "matchup"


def test_a_season_disambiguates_across_years(schedules: pl.DataFrame) -> None:
    ref = resolve_game("DAL@PHI", schedules=schedules, season=2024)
    assert ref.game_id == "2024_10_DAL_PHI"


def test_a_season_week_pair_with_no_game_lists_what_does_match(
    schedules: pl.DataFrame,
) -> None:
    with pytest.raises(GameSpecError) as caught:
        resolve_game("KC@LAC", schedules=schedules, season=2025, week=9)
    assert "2025_01_KC_LAC" in str(caught.value)


def test_alternatives_records_the_games_that_were_not_chosen(schedules: pl.DataFrame) -> None:
    ref = resolve_game("Rams", schedules=schedules, today=THAT_FRIDAY)
    assert "2025_05_SF_LA" in ref.alternatives


# --------------------------------------------------------------------------- #
# Bad input
# --------------------------------------------------------------------------- #


def test_an_unknown_word_lists_the_accepted_formats(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError) as caught:
        resolve_game("Zebras", schedules=schedules)
    message = str(caught.value)
    assert "unknown team 'Zebras'" in message
    assert ACCEPTED_FORMATS in message
    for form in ("DAL@PHI", "PHI vs DAL", "KXNFLGAME", "ESPN event id"):
        assert form in message


def test_an_unknown_team_on_one_side_is_named(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="unknown team 'Zebras'"):
        resolve_game("DAL@Zebras", schedules=schedules)


def test_an_empty_spec_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="empty game spec"):
        resolve_game("   ", schedules=schedules)


def test_a_team_against_itself_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="against itself"):
        resolve_game("PHI vs Eagles", schedules=schedules)


def test_a_team_with_no_scheduled_game_is_an_error(schedules: pl.DataFrame) -> None:
    with pytest.raises(GameSpecError, match="no scheduled game matches"):
        resolve_game("Bengals", schedules=schedules)


# --------------------------------------------------------------------------- #
# The CLI wiring
# --------------------------------------------------------------------------- #


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, schedules: pl.DataFrame) -> dict[str, object]:
    """Point the resolver at the fixture and stop before any model is loaded."""
    from nfl.inference.predict import PredictionError

    monkeypatch.setattr("nfl.inference.resolve.load_schedule_frame", lambda *_a, **_k: schedules)
    seen: dict[str, object] = {}

    def fake_predict(game_id: str, **kwargs: object) -> None:
        seen["game_id"] = game_id
        seen.update(kwargs)
        raise PredictionError("stopped before touching any data")

    monkeypatch.setattr("nfl.inference.predict.predict_game", fake_predict)
    return seen


def test_cli_resolves_a_matchup_and_prints_the_resolution(wired: dict[str, object]) -> None:
    result = runner.invoke(
        app,
        ["predict", "DAL@PHI", "--season", "2025", "--week", "1", "--no-ledger"],
    )
    assert wired["game_id"] == "2025_01_DAL_PHI"
    assert "resolved 'DAL@PHI' -> 2025_01_DAL_PHI" in result.output
    assert "via matchup" in result.output
    assert "DAL at PHI" in result.output


def test_cli_resolves_a_single_team(wired: dict[str, object]) -> None:
    result = runner.invoke(app, ["predict", "Eagles", "--season", "2025", "--week", "1"])
    assert wired["game_id"] == "2025_01_DAL_PHI"
    assert "via team" in result.output


def test_cli_reports_a_bad_spec_with_the_accepted_formats(wired: dict[str, object]) -> None:
    result = runner.invoke(app, ["predict", "Zebras"])
    assert result.exit_code == 1
    assert "cannot resolve 'Zebras'" in result.output
    assert "accepted game specs" in result.output
    assert "game_id" not in wired


def test_cli_at_turns_off_the_preference_for_an_upcoming_game(
    monkeypatch: pytest.MonkeyPatch, schedules: pl.DataFrame
) -> None:
    seen: dict[str, object] = {}

    def fake_resolve(spec: str, **kwargs: object) -> object:
        seen.update(kwargs)
        raise GameSpecError("stop here", formats=False)

    monkeypatch.setattr("nfl.inference.resolve.resolve_game", fake_resolve)
    runner.invoke(app, ["predict", "Rams", "--at", "Q4 05:00"])
    assert seen["prefer_upcoming"] is False

    seen.clear()
    runner.invoke(app, ["predict", "Rams"])
    assert seen["prefer_upcoming"] is True
