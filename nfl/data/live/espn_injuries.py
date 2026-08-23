"""League-wide ESPN injuries.

``site.api.espn.com/.../nfl/injuries`` returns **all 32 teams in one call**, with
no ``$ref`` chasing and no auth. Two things to know:

* it is ~8.8 MB, almost entirely ``longComment`` prose, so we trim on ingest
  (``longComment``, ``links``, ``headshot``, logos) and keep ~5% of the bytes;
* ``athlete.id`` is ``null`` in this payload, so joins are by **name + team**.
  :class:`EspnInjuryIndex` does that match, falling back to
  "last name + first initial" because ESPN mixes ``"Matt Waletzko"`` and
  ``"M. Waletzko"`` across surfaces.

The per-team core endpoint has stable athlete ids but costs 32 calls plus an
N+1 per player; use this one unless you specifically need the id.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import httpx

from nfl.data.live.base import LiveInjury, espn_to_nflverse_team
from nfl.data.live.espn import DEFAULT_HEADERS, SITE_BASE, _as_datetime

__all__ = [
    "INJURIES_URL",
    "EspnInjuryIndex",
    "fetch_espn_injuries",
    "injuries_for_teams",
    "normalize_name",
    "parse_espn_injuries",
]

INJURIES_URL = f"{SITE_BASE}/injuries"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT = re.compile(r"[^a-z\s]")


def normalize_name(name: str) -> str:
    """``"Ja'Marr Chase Jr."`` -> ``"jamarr chase"``."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = _PUNCT.sub("", text.lower())
    parts = [p for p in text.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def _short_key(name: str) -> str:
    """``"matt waletzko"`` -> ``"m waletzko"`` -- survives ESPN's abbreviated forms."""
    parts = normalize_name(name).split()
    if len(parts) < 2:
        return " ".join(parts)
    return f"{parts[0][0]} {parts[-1]}"


def parse_espn_injuries(payload: dict[str, Any]) -> list[LiveInjury]:
    """Trim the league-wide payload down to what a snapshot needs."""
    out: list[LiveInjury] = []
    for block in payload.get("injuries") or []:
        team = espn_to_nflverse_team(block.get("abbreviation"))
        if team is None:
            from nfl.data.live.base import espn_team_abbr_from_id

            team = espn_team_abbr_from_id(block.get("id"))
        if team is None:
            continue
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = athlete.get("displayName") or athlete.get("fullName")
            if not name:
                continue
            details = item.get("details") or {}
            out.append(
                LiveInjury(
                    team=team,
                    player_name=name,
                    player_id=None if athlete.get("id") is None else str(athlete["id"]),
                    position=(athlete.get("position") or {}).get("abbreviation"),
                    status=item.get("status") or (item.get("type") or {}).get("description"),
                    detail=details.get("detail") or details.get("type"),
                    comment=item.get("shortComment"),
                    reported_at=_as_datetime(item.get("date")),
                )
            )
    return out


def fetch_espn_injuries(
    client: httpx.Client | None = None, timeout: float = 30.0
) -> list[LiveInjury]:
    """One call, every team. Cache the result -- this payload is measured in megabytes."""
    owns = client is None
    client = client or httpx.Client(timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True)
    try:
        response = client.get(INJURIES_URL, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        return parse_espn_injuries(response.json())
    finally:
        if owns:
            client.close()


def injuries_for_teams(injuries: list[LiveInjury], *teams: str | None) -> list[LiveInjury]:
    """Narrow a league-wide list to the two teams in a game."""
    wanted = {t for t in teams if t}
    return [i for i in injuries if i.team in wanted]


class EspnInjuryIndex:
    """Name + team lookup over a league-wide injury list."""

    def __init__(self, injuries: list[LiveInjury]) -> None:
        self.injuries = injuries
        self._exact: dict[tuple[str, str], LiveInjury] = {}
        self._short: dict[tuple[str, str], LiveInjury] = {}
        for injury in injuries:
            self._exact.setdefault((normalize_name(injury.player_name), injury.team), injury)
            self._short.setdefault((_short_key(injury.player_name), injury.team), injury)

    def __len__(self) -> int:
        return len(self.injuries)

    def lookup(self, name: str, team: str) -> LiveInjury | None:
        """Exact normalised name first, then last-name + first-initial."""
        team = espn_to_nflverse_team(team) or team
        hit = self._exact.get((normalize_name(name), team))
        if hit is not None:
            return hit
        return self._short.get((_short_key(name), team))

    def for_teams(self, *teams: str | None) -> list[LiveInjury]:
        return injuries_for_teams(self.injuries, *teams)
