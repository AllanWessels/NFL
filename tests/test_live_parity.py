"""Train/serve parity: ESPN and the nflverse replay must describe the same game
the same way.

Both adapters read the *same* game -- BAL @ KC, 2024 week 1 -- from independent
recordings: ``tests/fixtures/espn/summary_401671789.json`` (what ESPN served on
the wire) and ``tests/fixtures/nflverse/pbp_2024_01_BAL_KC.parquet`` (what the
model trains on). If a snapshot field means one thing during training and
another during a live prediction, the model is being fed a different world at
serve time than it learned. That is the failure this file exists to catch.

Where the two genuinely differ -- ESPN publishes no per-play timeouts, nflverse
publishes no in-game injury feed -- the difference is asserted explicitly, so it
stays a known asymmetry rather than a surprise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import respx

from nfl.data.live.base import LiveSnapshot
from nfl.data.live.espn import CORE_BASE, SITE_BASE, EspnLiveFeed
from nfl.data.live.replay import HistoricalReplayFeed

GAME_ID = "2024_01_BAL_KC"
EVENT_ID = "401671789"

#: Late Q4, one score apart -- the situation this whole project targets.
AT = "Q4 05:00"

#: Fields that must agree exactly between the two feeds at the same game time.
#: This is the contract; everything a feature builder is allowed to read from a
#: snapshot's game state should be on this list.
PARITY_FIELDS = (
    "game_id",
    "espn_event_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "state",
    "completed",
    "period",
    "clock_seconds",
    "clock_display",
    "home_score",
    "away_score",
    "possession",
    "posteam_is_home",
    "down",
    "distance",
    "yards_to_endzone",
    "is_red_zone",
)


@pytest.fixture
def espn_snapshot(
    schedules_path: Path, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> LiveSnapshot:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=espn_summary)
        mock.get(url__startswith=f"{CORE_BASE}/events/{EVENT_ID}").respond(json=espn_situation)
        with EspnLiveFeed(schedules=schedules_path, backoff=0.0) as feed:
            return feed.snapshot(GAME_ID, at=AT)


@pytest.fixture
def espn_snapshot_full(
    schedules_path: Path, espn_summary: dict[str, Any], espn_situation: dict[str, Any]
) -> LiveSnapshot:
    """The same game, untruncated."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__startswith=f"{SITE_BASE}/summary").respond(json=espn_summary)
        mock.get(url__startswith=f"{CORE_BASE}/events/{EVENT_ID}").respond(json=espn_situation)
        with EspnLiveFeed(schedules=schedules_path, backoff=0.0) as feed:
            return feed.snapshot(GAME_ID)


@pytest.fixture
def replay_snapshot(pbp_path: Path, schedules_path: Path) -> LiveSnapshot:
    feed = HistoricalReplayFeed(pbp=pbp_path, schedules=schedules_path)
    return feed.snapshot(GAME_ID, at=AT)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_both_feeds_return_the_same_model(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    assert type(espn_snapshot) is type(replay_snapshot) is LiveSnapshot
    assert set(espn_snapshot.model_dump()) == set(replay_snapshot.model_dump())


def test_only_the_source_tag_distinguishes_them(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    assert espn_snapshot.source == "espn"
    assert replay_snapshot.source == "replay"


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", PARITY_FIELDS)
def test_game_state_fields_agree(
    field: str, espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    live = getattr(espn_snapshot, field)
    replayed = getattr(replay_snapshot, field)
    assert live == replayed, f"{field}: espn={live!r} replay={replayed!r}"


def test_the_agreed_state_is_the_one_we_expect(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """Guard against both feeds agreeing on something wrong.

    At Q4 5:00 Baltimore trailed 27-17 and faced 4th & 4 from the Kansas City
    14 -- the snap before Tucker's field goal.
    """
    for snapshot in (espn_snapshot, replay_snapshot):
        assert (snapshot.home_team, snapshot.away_team) == ("KC", "BAL")
        assert (snapshot.home_score, snapshot.away_score) == (27, 17)
        assert (snapshot.period, snapshot.clock_seconds) == (4, 300)
        assert snapshot.possession == "BAL"
        assert (snapshot.down, snapshot.distance, snapshot.yards_to_endzone) == (4, 4, 14)


def test_derived_clock_quantities_agree(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    for attribute in (
        "quarter_seconds_remaining",
        "half_seconds_remaining",
        "game_seconds_remaining",
        "is_overtime",
    ):
        assert getattr(espn_snapshot, attribute) == getattr(replay_snapshot, attribute)
    assert espn_snapshot.score_differential_home == replay_snapshot.score_differential_home == 10


def test_the_last_completed_play_is_the_same_play(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    live, replayed = espn_snapshot.last_play, replay_snapshot.last_play
    assert live is not None and replayed is not None
    assert live.period == replayed.period == 4
    assert live.clock_seconds == replayed.clock_seconds == 306
    assert live.team == replayed.team == "BAL"
    assert (live.home_score, live.away_score) == (replayed.home_score, replayed.away_score)
    # Jackson's scramble. Both feeds describe it, with different wording.
    assert "Jackson" in (live.text or "")
    assert "Jackson" in (replayed.text or "")


def test_wallclock_agrees_to_the_second(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """Wallclock drives the staleness check, so it has to mean the same thing.

    ESPN truncates to whole seconds; nflverse ``time_of_day`` keeps milliseconds.
    """
    live = espn_snapshot.last_play_wallclock
    replayed = replay_snapshot.last_play_wallclock
    assert live is not None and replayed is not None
    assert abs((live - replayed).total_seconds()) < 1.0


def _score_progression(snapshot: LiveSnapshot) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    for play in snapshot.plays:
        if play.home_score is None or play.away_score is None:
            continue
        entry = (play.home_score, play.away_score)
        if not seen or seen[-1] != entry:
            seen.append(entry)
    return seen


def test_both_feeds_agree_on_who_scored_when(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """The score progression -- not just the final tally -- has to line up.

    Not as equality: nflverse shows the transient 6 points between a touchdown
    and its extra point, ESPN jumps straight to 7. So ESPN's progression must be
    a **subsequence** of nflverse's, and both must end on the same score.
    """
    live = _score_progression(espn_snapshot)
    replayed = _score_progression(replay_snapshot)

    remaining = iter(replayed)
    assert all(entry in remaining for entry in live), f"espn={live} replay={replayed}"
    assert live[0] == replayed[0] == (0, 0)
    assert live[-1] == replayed[-1] == (27, 17)


def test_the_only_score_states_espn_skips_are_pre_conversion_ones(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """Pin the asymmetry down: every score state nflverse has and ESPN does not
    is one point short of a state they share."""
    live = set(_score_progression(espn_snapshot))
    extra = [e for e in _score_progression(replay_snapshot) if e not in live]
    for home, away in extra:
        assert (home + 1, away) in live or (home, away + 1) in live


def test_drive_possession_alternates_identically(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """Drive *boundaries* differ (ESPN counts a few administrative rows nflverse
    drops), but the sequence of teams that had the ball must not."""

    def teams(snapshot: LiveSnapshot) -> list[str]:
        out: list[str] = []
        for drive in snapshot.drives:
            if drive.team and (not out or out[-1] != drive.team):
                out.append(drive.team)
        return out

    assert teams(espn_snapshot) == teams(replay_snapshot)


# --------------------------------------------------------------------------- #
# Known asymmetries -- asserted so they cannot drift silently
# --------------------------------------------------------------------------- #


def test_timeouts_are_exact_only_where_the_feed_publishes_them(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """ESPN publishes timeouts for *now* and nowhere else, so a rewound ESPN
    snapshot flags them inexact. nflverse has them per play."""
    assert replay_snapshot.timeouts_exact is True
    assert (replay_snapshot.home_timeouts, replay_snapshot.away_timeouts) == (3, 1)

    assert espn_snapshot.timeouts_exact is False
    assert espn_snapshot.home_timeouts is not None
    assert espn_snapshot.away_timeouts is not None


def test_in_game_injuries_come_only_from_espn(
    espn_snapshot_full: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    assert replay_snapshot.injuries == []
    assert espn_snapshot_full.injuries  # the summary carries a block per team


def test_rewinding_drops_injuries_that_had_not_been_reported_yet(
    espn_snapshot: LiveSnapshot, espn_snapshot_full: LiveSnapshot
) -> None:
    """ESPN serves *today's* injury report on an old game's summary. Rewound to
    2024, none of it was knowable -- and a feature builder must not see it."""
    assert espn_snapshot_full.injuries
    assert all(i.reported_at is not None for i in espn_snapshot_full.injuries)
    assert espn_snapshot.injuries == []


def test_both_feeds_place_the_game_in_the_same_venue(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """The names differ by sponsor ('Arrowhead Stadium' vs 'GEHA Field at
    Arrowhead Stadium') and the ids come from different namespaces, so only the
    indoor/outdoor call is comparable -- and that one feeds the weather logic."""
    assert espn_snapshot.venue is not None
    assert replay_snapshot.venue is not None
    assert replay_snapshot.venue.indoor is False
    assert espn_snapshot.venue.indoor in (False, None)


def test_neither_feed_carries_market_data(
    espn_snapshot: LiveSnapshot, replay_snapshot: LiveSnapshot
) -> None:
    """ADR-0001: odds, spreads and win-probability blocks are dropped on ingest."""
    banned = ("odds", "spread", "moneyline", "over_under", "win_probability", "line")
    for snapshot in (espn_snapshot, replay_snapshot):
        fields = set(snapshot.model_dump())
        assert not [f for f in fields if any(b in f for b in banned)]
