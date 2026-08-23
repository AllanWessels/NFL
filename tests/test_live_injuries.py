"""League-wide ESPN injuries: trimming, name normalisation, name+team joins."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from nfl.data.live.base import LiveInjury
from nfl.data.live.espn_injuries import (
    INJURIES_URL,
    EspnInjuryIndex,
    fetch_espn_injuries,
    injuries_for_teams,
    normalize_name,
    parse_espn_injuries,
)


def test_parse_keeps_only_the_essentials(espn_injuries_payload: dict[str, Any]) -> None:
    injuries = parse_espn_injuries(espn_injuries_payload)
    assert injuries
    assert all(isinstance(i, LiveInjury) for i in injuries)
    assert all(i.player_name for i in injuries)
    assert {i.team for i in injuries} <= {"KC", "BAL", "ARI"}
    # No prose: ``longComment`` is dropped, ``shortComment`` is not.
    assert all(len(i.comment or "") < 2000 for i in injuries)


def test_parse_maps_the_team_id_to_an_nflverse_abbreviation() -> None:
    injuries = parse_espn_injuries(
        {
            "injuries": [
                {
                    "id": "28",  # WSH
                    "injuries": [
                        {
                            "status": "Out",
                            "athlete": {
                                "displayName": "Sam Cosmi",
                                "position": {"abbreviation": "OG"},
                            },
                        }
                    ],
                }
            ]
        }
    )
    assert [i.team for i in injuries] == ["WAS"]


def test_parse_prefers_an_explicit_abbreviation() -> None:
    injuries = parse_espn_injuries(
        {
            "injuries": [
                {
                    "id": "14",
                    "abbreviation": "LAR",
                    "injuries": [{"status": "Out", "athlete": {"displayName": "Puka Nacua"}}],
                }
            ]
        }
    )
    assert [i.team for i in injuries] == ["LA"]


def test_parse_skips_entries_without_a_name() -> None:
    payload = {"injuries": [{"id": "12", "injuries": [{"status": "Out", "athlete": {}}]}]}
    assert parse_espn_injuries(payload) == []


def test_parse_reads_the_timestamp(espn_injuries_payload: dict[str, Any]) -> None:
    injuries = parse_espn_injuries(espn_injuries_payload)
    timestamped = [i for i in injuries if i.reported_at is not None]
    assert timestamped
    assert all(i.reported_at.tzinfo is UTC for i in timestamped)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def test_fetch_hits_the_single_league_wide_endpoint(
    espn_injuries_payload: dict[str, Any],
) -> None:
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(INJURIES_URL).respond(json=espn_injuries_payload)
        injuries = fetch_espn_injuries()
    assert route.call_count == 1  # all 32 teams in one call
    assert injuries


def test_fetch_reuses_a_supplied_client(espn_injuries_payload: dict[str, Any]) -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get(INJURIES_URL).respond(json=espn_injuries_payload)
        with httpx.Client() as client:
            assert fetch_espn_injuries(client=client)
            assert not client.is_closed


def test_fetch_raises_on_error() -> None:
    with respx.mock(assert_all_called=False) as mock:
        mock.get(INJURIES_URL).respond(status_code=500)
        with pytest.raises(httpx.HTTPStatusError):
            fetch_espn_injuries()


# --------------------------------------------------------------------------- #
# Name + team join
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ja'Marr Chase", "jamarr chase"),
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Odell Beckham Jr", "odell beckham"),
        ("Patrick Mahomes II", "patrick mahomes"),
        ("José Ramírez", "jose ramirez"),
        ("  Travis   Kelce ", "travis kelce"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def _index() -> EspnInjuryIndex:
    return EspnInjuryIndex(
        [
            LiveInjury(
                team="KC",
                player_name="Matt Waletzko",
                position="OT",
                status="Injured Reserve",
                reported_at=datetime(2026, 8, 22, tzinfo=UTC),
            ),
            LiveInjury(team="BAL", player_name="Ja'Marr Chase Jr.", position="WR", status="Out"),
            LiveInjury(team="WAS", player_name="Sam Cosmi", position="OG", status="Questionable"),
        ]
    )


def test_lookup_is_exact_on_the_normalised_name() -> None:
    hit = _index().lookup("Matt Waletzko", "KC")
    assert hit is not None and hit.status == "Injured Reserve"


def test_lookup_falls_back_to_initial_and_surname() -> None:
    """ESPN mixes ``"Matt Waletzko"`` and ``"M. Waletzko"`` across surfaces."""
    hit = _index().lookup("M. Waletzko", "KC")
    assert hit is not None and hit.position == "OT"


def test_lookup_ignores_suffixes_and_punctuation() -> None:
    assert _index().lookup("JaMarr Chase", "BAL") is not None


def test_lookup_is_team_scoped() -> None:
    assert _index().lookup("Matt Waletzko", "BAL") is None


def test_lookup_accepts_an_espn_abbreviation() -> None:
    assert _index().lookup("Sam Cosmi", "WSH") is not None


def test_lookup_misses_return_none() -> None:
    assert _index().lookup("Nobody At All", "KC") is None


def test_for_teams_narrows_the_league_wide_list() -> None:
    index = _index()
    assert len(index) == 3
    assert {i.team for i in index.for_teams("KC", "BAL")} == {"KC", "BAL"}
    assert injuries_for_teams(index.injuries, None) == []
