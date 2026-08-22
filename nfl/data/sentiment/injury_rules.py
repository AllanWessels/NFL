"""Rule-based injury scorer v1.

Two jobs:

1. **Availability.** Official designation (Out / Doubtful / Questionable / IR /
   PUP / Suspended) plus the Wed/Thu/Fri practice participation trio
   (DNP / LP / FP) -> ``P(player is available)``. The base rate comes from the
   designation; practice participation moves it, which matters almost entirely
   for *Questionable*, the only designation that is genuinely uncertain.

2. **Cost.** Availability is weighted by positional impact from
   ``nfl/data/reference/position_impact.csv`` and (optionally) the player's own
   value weight, giving ``weighted_absence_cost`` and a team-level
   ``injury_sentiment_score`` in ``[-1, 1]``.

Status *transitions* are scored too (``status_delta``): a Wednesday DNP that
becomes a Friday FP is good news, and that direction is what a sentiment signal
is actually for.

Calibration note: these numbers are priors, not fitted values. ``InjuryReport``
carries ``did_play`` so they can be refit offline once a season of reports is
ingested; ``did_play`` is postgame truth and must never become a feature.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from nfl.data.reference import POSITION_IMPACT_CSV
from nfl.data.schemas import InjuryReport
from nfl.data.sentiment.base import clamp

log = logging.getLogger(__name__)

SCORER_VERSION = "0.1.0"

#: P(plays) by official designation, before practice adjustment.
STATUS_AVAILABILITY: dict[str, float] = {
    "": 1.0,
    "ACTIVE": 1.0,
    "PROBABLE": 0.95,
    "QUESTIONABLE": 0.55,
    "DOUBTFUL": 0.08,
    "OUT": 0.0,
    "IR": 0.0,
    "INJURED RESERVE": 0.0,
    "PUP": 0.0,
    "NFI": 0.0,
    "SUSP": 0.0,
    "SUSPENDED": 0.0,
    "DNP": 0.35,
    "LP": 0.75,
    "FP": 0.95,
}

#: How each practice day is read. Friday (the last report) dominates.
_PARTICIPATION_SCORE: dict[str, float] = {"DNP": 0.0, "LP": 0.5, "FP": 1.0}
_DAY_WEIGHTS: tuple[float, float, float] = (0.2, 0.3, 0.5)  # wed, thu, fri

#: Only designations with real uncertainty are moved by practice reports.
_PRACTICE_SENSITIVITY: dict[str, float] = {
    "QUESTIONABLE": 0.45,
    "PROBABLE": 0.10,
    "DOUBTFUL": 0.06,
    "": 0.20,
}

#: Ordering used for upgrade/downgrade deltas (higher = more likely to play).
_STATUS_RANK: dict[str, int] = {
    "IR": 0,
    "PUP": 0,
    "NFI": 0,
    "SUSP": 0,
    "OUT": 1,
    "DOUBTFUL": 2,
    "QUESTIONABLE": 3,
    "PROBABLE": 4,
    "": 5,
    "ACTIVE": 5,
}

#: Maps the schema's ``InjuryReportStatus`` literals and common feed spellings
#: onto the canonical keys above.
STATUS_ALIASES: dict[str, str] = {
    "O": "OUT",
    "D": "DOUBTFUL",
    "Q": "QUESTIONABLE",
    "P": "PROBABLE",
    "INJURY_STATUS_OUT": "OUT",
    "INJURY_STATUS_DOUBTFUL": "DOUBTFUL",
    "INJURY_STATUS_QUESTIONABLE": "QUESTIONABLE",
    "INJURY_STATUS_PROBABLE": "PROBABLE",
    "INJURY_STATUS_ACTIVE": "",
    "INJURY_STATUS_DAY_TO_DAY": "QUESTIONABLE",
    "INJURED RESERVE": "IR",
    "RESERVE/INJURED": "IR",
    "DAY-TO-DAY": "QUESTIONABLE",
    "DID NOT PARTICIPATE IN PRACTICE": "DNP",
    "LIMITED PARTICIPATION IN PRACTICE": "LP",
    "FULL PARTICIPATION IN PRACTICE": "FP",
}

DEFAULT_POSITION_WEIGHT = 0.10
DEFAULT_REPLACEMENT_DROPOFF = 0.45


# --------------------------------------------------------------------------- #
# Position impact reference table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PositionImpact:
    position: str
    position_group: str
    impact_weight: float
    replacement_dropoff: float
    notes: str = ""


@lru_cache(maxsize=4)
def load_position_impact(path: Path | None = None) -> dict[str, PositionImpact]:
    """Load ``position_impact.csv`` into ``{POSITION: PositionImpact}``."""
    target = Path(path) if path is not None else POSITION_IMPACT_CSV
    table: dict[str, PositionImpact] = {}
    with target.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            position = (row.get("position") or "").strip().upper()
            if not position:
                continue
            table[position] = PositionImpact(
                position=position,
                position_group=(row.get("position_group") or "").strip().upper(),
                impact_weight=float(row.get("impact_weight") or DEFAULT_POSITION_WEIGHT),
                replacement_dropoff=float(
                    row.get("replacement_dropoff") or DEFAULT_REPLACEMENT_DROPOFF
                ),
                notes=(row.get("notes") or "").strip(),
            )
    if not table:  # pragma: no cover - defensive
        raise ValueError(f"{target} contained no position rows")
    return table


def position_impact(position: str | None, *, path: Path | None = None) -> PositionImpact:
    """Look up a position, falling back to a conservative default."""
    table = load_position_impact(path)
    key = (position or "").strip().upper()
    if key in table:
        return table[key]
    if key:
        log.debug("injury_rules: unknown position %r, using default weight", position)
    return PositionImpact(
        position=key or "UNK",
        position_group="",
        impact_weight=DEFAULT_POSITION_WEIGHT,
        replacement_dropoff=DEFAULT_REPLACEMENT_DROPOFF,
    )


def position_weight(position: str | None, *, path: Path | None = None) -> float:
    return position_impact(position, path=path).impact_weight


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def canonical_status(status: str | None) -> str:
    """Normalise a designation from any feed onto the canonical key set."""
    if not status:
        return ""
    key = str(status).strip().upper()
    key = STATUS_ALIASES.get(key, key)
    return "" if key in {"NONE", "NULL", "N/A"} else key


def practice_signal(
    wed: str | None = None, thu: str | None = None, fri: str | None = None
) -> float | None:
    """Weighted practice participation in ``[0, 1]``; ``None`` if nothing reported."""
    days = ((wed, _DAY_WEIGHTS[0]), (thu, _DAY_WEIGHTS[1]), (fri, _DAY_WEIGHTS[2]))
    total_weight = 0.0
    total = 0.0
    for raw, weight in days:
        key = canonical_status(raw)
        if key not in _PARTICIPATION_SCORE:
            continue
        total += _PARTICIPATION_SCORE[key] * weight
        total_weight += weight
    if total_weight == 0.0:
        return None
    return total / total_weight


def availability_probability(
    status: str | None,
    *,
    practice_wed: str | None = None,
    practice_thu: str | None = None,
    practice_fri: str | None = None,
) -> float:
    """``P(player is available)`` from designation + practice participation.

    Out / IR / PUP / Suspended are hard zeros -- no practice report rescues them.
    """
    key = canonical_status(status)
    base = STATUS_AVAILABILITY.get(key, 1.0)
    if base == 0.0:
        return 0.0
    signal = practice_signal(practice_wed, practice_thu, practice_fri)
    if signal is None:
        return round(clamp(base, 0.0, 1.0), 4)
    sensitivity = _PRACTICE_SENSITIVITY.get(key, 0.15)
    adjusted = base + sensitivity * (signal - 0.5) * 2.0
    return round(clamp(adjusted, 0.0, 1.0), 4)


def status_delta(previous: str | None, current: str | None) -> float:
    """Signed designation move in ``[-1, 1]``: positive = upgraded (better news)."""
    prev_key, cur_key = canonical_status(previous), canonical_status(current)
    prev_rank = _STATUS_RANK.get(prev_key)
    cur_rank = _STATUS_RANK.get(cur_key)
    if prev_rank is None or cur_rank is None:
        return 0.0
    return clamp((cur_rank - prev_rank) / 5.0)


def weighted_absence_cost(
    availability: float,
    position: str | None,
    *,
    player_value_weight: float = 0.0,
    path: Path | None = None,
) -> float:
    """Expected value lost: ``P(absent) x positional weight x replacement dropoff``.

    ``player_value_weight`` (0 when unknown) scales the positional prior toward
    the specific player, so a WR3 does not cost what a WR1 costs.
    """
    impact = position_impact(position, path=path)
    scale = 1.0 + max(0.0, player_value_weight)
    return round(
        clamp(1.0 - availability, 0.0, 1.0)
        * impact.impact_weight
        * impact.replacement_dropoff
        * scale,
        6,
    )


# --------------------------------------------------------------------------- #
# Report-level scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InjuryItem:
    """One player's injury-report state, feed-agnostic."""

    player_id: str
    team: str
    position: str | None = None
    status: str | None = None
    previous_status: str | None = None
    practice_wed: str | None = None
    practice_thu: str | None = None
    practice_fri: str | None = None
    primary_injury: str | None = None
    secondary_injury: str | None = None
    is_new_injury: bool = False
    weeks_on_report: int = 0
    player_value_weight: float = 0.0
    report_date: date | None = None
    comment: str | None = None


@dataclass(frozen=True, slots=True)
class TeamInjurySummary:
    """Team-level rollup consumed by :mod:`nfl.data.sentiment.sources`."""

    team: str
    n_items: int
    injury_sentiment_score: float
    total_absence_cost: float
    qb_absence_cost: float
    n_out: int
    n_doubtful: int
    n_questionable: int
    n_upgraded: int
    n_downgraded: int
    confidence: float


#: Absence cost at which the team score saturates at -1. One starting QB out
#: (1.00 weight x 0.85 dropoff = 0.85) alone lands near the floor, by design.
COST_SATURATION = 1.0

#: Depth discount for the 2nd, 3rd, ... most expensive player at one position.
#: The injury feed has no depth chart, so the *n*th listed WR is treated as WR
#: (n+1) on the depth chart. This is a prior, not knowledge: it cannot tell a
#: starting QB from a backup QB when only one of them is on the report. Supply
#: ``player_value_weight`` from a snap-share source to fix that properly.
DEPTH_DISCOUNT = 0.35


def score_injury_items(
    items: Sequence[InjuryItem], *, team: str | None = None, path: Path | None = None
) -> TeamInjurySummary:
    """Roll a team's injury-report rows into a ``[-1, 1]`` sentiment score.

    The score blends the *level* (how much value is unavailable) with the
    *direction* (upgrades vs downgrades since the last report).
    """
    subject = team or (items[0].team if items else "")
    scoped = [i for i in items if not subject or i.team == subject]

    total_cost = 0.0
    qb_cost = 0.0
    delta_sum = 0.0
    n_out = n_doubtful = n_questionable = n_up = n_down = 0

    # Raw cost per item, before the depth discount is applied.
    raw_costs: list[tuple[InjuryItem, float]] = []
    for item in scoped:
        availability = availability_probability(
            item.status,
            practice_wed=item.practice_wed,
            practice_thu=item.practice_thu,
            practice_fri=item.practice_fri,
        )
        raw_costs.append(
            (
                item,
                weighted_absence_cost(
                    availability,
                    item.position,
                    player_value_weight=item.player_value_weight,
                    path=path,
                ),
            )
        )

    # Most expensive absence at each position counts in full; the next one at
    # that same position is the backup, and so on.
    depth_rank: dict[int, int] = {}
    seen: dict[str, int] = {}
    for index in sorted(range(len(raw_costs)), key=lambda i: -raw_costs[i][1]):
        position = (raw_costs[index][0].position or "UNK").strip().upper()
        depth_rank[index] = seen.get(position, 0)
        seen[position] = depth_rank[index] + 1

    for index, (item, raw_cost) in enumerate(raw_costs):
        cost = raw_cost * (DEPTH_DISCOUNT ** depth_rank.get(index, 0))
        total_cost += cost
        if (item.position or "").strip().upper() == "QB":
            qb_cost += cost

        key = canonical_status(item.status)
        if key in {"OUT", "IR", "PUP", "NFI", "SUSP", "SUSPENDED"}:
            n_out += 1
        elif key == "DOUBTFUL":
            n_doubtful += 1
        elif key == "QUESTIONABLE":
            n_questionable += 1

        delta = status_delta(item.previous_status, item.status)
        impact = position_impact(item.position, path=path)
        delta_sum += delta * impact.impact_weight
        if delta > 0:
            n_up += 1
        elif delta < 0:
            n_down += 1

    level = -clamp(total_cost / COST_SATURATION, 0.0, 1.0)
    direction = clamp(delta_sum)
    score = clamp(0.75 * level + 0.25 * direction)
    confidence = clamp(len(scoped) / 12.0, 0.0, 1.0) if scoped else 0.0

    return TeamInjurySummary(
        team=subject,
        n_items=len(scoped),
        injury_sentiment_score=round(score, 4),
        total_absence_cost=round(total_cost, 6),
        qb_absence_cost=round(qb_cost, 6),
        n_out=n_out,
        n_doubtful=n_doubtful,
        n_questionable=n_questionable,
        n_upgraded=n_up,
        n_downgraded=n_down,
        confidence=round(confidence, 4),
    )


def to_injury_reports(
    items: Iterable[InjuryItem],
    *,
    game_id: str,
    season: int,
    week: int,
    source: str,
    report_date: date | None = None,
    ingested_at: datetime | None = None,
    path: Path | None = None,
) -> list[InjuryReport]:
    """Enrich raw items into :class:`InjuryReport` rows with derived weights."""
    rows: list[InjuryReport] = []
    for item in items:
        availability = availability_probability(
            item.status,
            practice_wed=item.practice_wed,
            practice_thu=item.practice_thu,
            practice_fri=item.practice_fri,
        )
        impact = position_impact(item.position, path=path)
        rows.append(
            InjuryReport(
                source=source,
                ingested_at=ingested_at,
                game_id=game_id,
                season=season,
                week=week,
                team=item.team,
                player_id=item.player_id,
                report_date=item.report_date or report_date or date.today(),
                report_status=_schema_status(item.status),
                practice_status=item.practice_fri or item.practice_thu or item.practice_wed,
                practice_participation_wed=item.practice_wed,
                practice_participation_thu=item.practice_thu,
                practice_participation_fri=item.practice_fri,
                primary_injury=item.primary_injury,
                secondary_injury=item.secondary_injury,
                is_new_injury=item.is_new_injury,
                weeks_on_report=item.weeks_on_report,
                expected_availability=availability,
                position_importance_weight=impact.impact_weight,
                player_value_weight=item.player_value_weight,
                weighted_absence_cost=weighted_absence_cost(
                    availability,
                    item.position,
                    player_value_weight=item.player_value_weight,
                    path=path,
                ),
            )
        )
    return rows


#: ``InjuryReportStatus`` in the schema is a closed literal set.
_SCHEMA_STATUSES = {"Out", "Doubtful", "Questionable", "IR", "DNP", "LP", "FP", ""}


def _schema_status(status: str | None) -> str:
    key = canonical_status(status)
    titled = key.title() if key not in {"IR", "DNP", "LP", "FP"} else key
    return titled if titled in _SCHEMA_STATUSES else ""


__all__ = [
    "COST_SATURATION",
    "DEFAULT_POSITION_WEIGHT",
    "SCORER_VERSION",
    "STATUS_ALIASES",
    "STATUS_AVAILABILITY",
    "InjuryItem",
    "PositionImpact",
    "TeamInjurySummary",
    "availability_probability",
    "canonical_status",
    "load_position_impact",
    "position_impact",
    "position_weight",
    "practice_signal",
    "score_injury_items",
    "status_delta",
    "to_injury_reports",
    "weighted_absence_cost",
]
