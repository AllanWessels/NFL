"""Live and replayed in-game feeds.

Both adapters return the same :class:`~nfl.data.live.base.LiveSnapshot`, which is
what makes ``/predict`` behave identically on a Sunday game and on a 2016 one.
"""

from __future__ import annotations

from nfl.data.live.base import (
    AtTime,
    GameRef,
    LiveDrive,
    LiveFeed,
    LiveInjury,
    LivePlay,
    LiveSnapshot,
    LiveVenue,
    LiveWeather,
    TeamRecord,
    parse_at,
    resolve_espn_event_id,
    truncate_snapshot,
)
from nfl.data.live.espn import EspnLiveFeed
from nfl.data.live.espn_injuries import EspnInjuryIndex, fetch_espn_injuries
from nfl.data.live.replay import HistoricalReplayFeed

__all__ = [
    "AtTime",
    "EspnInjuryIndex",
    "EspnLiveFeed",
    "GameRef",
    "HistoricalReplayFeed",
    "LiveDrive",
    "LiveFeed",
    "LiveInjury",
    "LivePlay",
    "LiveSnapshot",
    "LiveVenue",
    "LiveWeather",
    "TeamRecord",
    "fetch_espn_injuries",
    "parse_at",
    "resolve_espn_event_id",
    "truncate_snapshot",
]
