"""Venue reference table and venue resolution for a games frame.

nflverse ships no ``stadiums.csv`` with coordinates, so ``nfl/data/reference/venues.csv``
is a hand-built, committed table keyed on the nflverse ``stadium_id``. Six neutral-site
venues that nflverse never assigns an id to (Croke Park, Olympiastadion Berlin, Santiago
Bernabeu, Melbourne Cricket Ground, Maracana, Stade de France) get synthetic ids
(``DUB00``, ``BER00``, ``MAD00``, ``MEL00``, ``RIO00``, ``PAR00``).

**The neutral-site ``stadium_id`` bug.** In nflverse ``games.csv`` / ``schedules``,
neutral-site games are frequently tagged with the *designated home team's* ``stadium_id``
rather than the venue actually played in. Two distinct flavours occur:

1. *Name right, id stale* -- e.g. ``2026_04_IND_WAS`` is
   ``stadium_id=WAS00, stadium="Tottenham Hotspur Stadium"``. Resolving by the
   ``stadium`` name recovers the real venue.
2. *Both name and id stale* -- the whole 2025 international slate, e.g.
   ``2025_04_MIN_PIT`` (played at Croke Park, Dublin) is
   ``stadium_id=PIT00, stadium="Acrisure Stadium"``. Nothing in the row identifies the
   venue, so these need an explicit per-``game_id`` override
   (:data:`GAME_VENUE_OVERRIDES`).

Flavour 1 also occurs on rows whose ``location`` is ``"Home"`` (a team's designated home
game played abroad, e.g. ``2026_05_PHI_JAX`` at Tottenham), so resolution must *not* be
gated on ``location == "Neutral"``.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import polars as pl

REFERENCE_DIR = Path(__file__).parent / "reference"
VENUES_CSV = REFERENCE_DIR / "venues.csv"

SOURCE = "static.venues"
SCHEMA_VERSION = "1"

#: Roof classifications. ``open`` means the playing surface is exposed to the sky -- a
#: roof over the seating bowl only (Olympiastadion, Maracana, Stade de France, Allianz
#: Arena) still counts as ``open``. ``dome`` is a permanently enclosed field.
#: ``retractable`` is either on any given day; the per-game nflverse ``roof`` column
#: decides that, not this table.
ROOF_TYPES = ("dome", "retractable", "open")

#: Per-``game_id`` venue overrides for rows where *both* ``stadium_id`` and the
#: ``stadium`` name were copied from the designated home team instead of the venue
#: actually played in. Verified against the NFL international-series announcements.
GAME_VENUE_OVERRIDES: dict[str, str] = {
    # 2025 international series. nflverse tags every one of these with the designated
    # home team's stadium_id *and* stadium name, so nothing in the row is usable.
    "2025_01_KC_LAC": "SAO00",  # Corinthians Arena, Sao Paulo (tagged SoFi Stadium)
    "2025_04_MIN_PIT": "DUB00",  # Croke Park, Dublin (tagged Acrisure Stadium)
    "2025_05_MIN_CLE": "LON02",  # Tottenham Hotspur (tagged FirstEnergy Stadium)
    "2025_06_DEN_NYJ": "LON02",  # Tottenham Hotspur (tagged MetLife Stadium)
    "2025_07_LA_JAX": "LON00",  # Wembley (tagged TIAA Bank Stadium)
    "2025_10_ATL_IND": "BER00",  # Olympiastadion Berlin (tagged Lucas Oil Stadium)
    "2025_11_WAS_MIA": "MAD00",  # Santiago Bernabeu (tagged Hard Rock Stadium)
}

_NAME_STOPWORDS = frozenset({"the", "of", "field", "stadium", "arena", "park"})


@dataclass(frozen=True, slots=True)
class VenueRecord:
    """One row of ``reference/venues.csv``."""

    venue_id: str
    name: str
    city: str
    state: str | None
    country: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    roof_type: str
    surface: str
    capacity: int | None
    first_season: int
    last_season: int | None
    home_teams: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def altitude_feet(self) -> float:
        """Elevation in feet -- the unit :class:`nfl.data.schemas.Venue` stores."""
        return round(self.elevation_m * 3.280839895, 1)

    @property
    def is_fixed_indoor(self) -> bool:
        """True only for permanently enclosed venues -- never for retractables."""
        return self.roof_type == "dome"

    def to_venue_kwargs(self) -> dict[str, object]:
        """Keyword arguments for :class:`nfl.data.schemas.Venue`."""
        return {
            "source": SOURCE,
            "schema_version": SCHEMA_VERSION,
            "venue_id": self.venue_id,
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_feet": self.altitude_feet,
            "timezone": self.timezone,
            "roof_type": self.roof_type,
            "surface_type": self.surface,
            "capacity": self.capacity,
            "home_teams": list(self.home_teams),
        }


@dataclass(frozen=True, slots=True)
class VenueResolution:
    """Outcome of :func:`resolve_venue`.

    ``how`` records which rule fired, which is what makes the neutral-site bug
    auditable rather than silent.
    """

    venue: VenueRecord | None
    how: str
    stadium_id: str | None = None
    overridden: bool = False
    notes: tuple[str, ...] = field(default=())

    def __bool__(self) -> bool:
        return self.venue is not None


def normalize_name(name: str | None) -> str:
    """Fold a stadium name to a comparison key.

    Strips accents, punctuation and generic venue nouns so that ``"Tottenham Stadium"``
    and ``"Tottenham Hotspur Stadium"``, or ``"Bernabeu"`` and
    ``"Estadio Santiago Bernabeu"``, land on entries of the same alias family.
    """
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower().replace("&", " and ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in folded)
    tokens = [t for t in cleaned.split() if t and t not in _NAME_STOPWORDS]
    return " ".join(tokens)


def _split_list(raw: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (raw or "").split("|") if part.strip())


def _parse_int(raw: str | None) -> int | None:
    raw = (raw or "").strip()
    return int(raw) if raw else None


@lru_cache(maxsize=4)
def load_venues(path: str | None = None) -> tuple[VenueRecord, ...]:
    """Read, validate and cache ``reference/venues.csv``."""
    csv_path = Path(path) if path else VENUES_CSV
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = tuple(
        VenueRecord(
            venue_id=row["venue_id"].strip(),
            name=row["name"].strip(),
            city=row["city"].strip(),
            state=row["state"].strip() or None,
            country=row["country"].strip(),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation_m=float(row["elevation_m"]),
            timezone=row["timezone"].strip(),
            roof_type=row["roof_type"].strip(),
            surface=row["surface"].strip(),
            capacity=_parse_int(row.get("capacity")),
            first_season=int(row["first_season"]),
            last_season=_parse_int(row.get("last_season")),
            home_teams=_split_list(row.get("home_teams")),
            aliases=_split_list(row.get("aliases")),
        )
        for row in rows
    )
    _validate(records, csv_path)
    return records


def _validate(records: tuple[VenueRecord, ...], csv_path: Path) -> None:
    seen_ids: set[str] = set()
    seen_names: dict[str, str] = {}
    for rec in records:
        if rec.venue_id in seen_ids:
            raise ValueError(f"duplicate venue_id {rec.venue_id!r} in {csv_path}")
        seen_ids.add(rec.venue_id)
        if rec.roof_type not in ROOF_TYPES:
            raise ValueError(f"{rec.venue_id}: roof_type {rec.roof_type!r} not in {ROOF_TYPES}")
        if not -90.0 <= rec.latitude <= 90.0:
            raise ValueError(f"{rec.venue_id}: latitude {rec.latitude} out of range")
        if not -180.0 <= rec.longitude <= 180.0:
            raise ValueError(f"{rec.venue_id}: longitude {rec.longitude} out of range")
        if rec.last_season is not None and rec.last_season < rec.first_season:
            raise ValueError(f"{rec.venue_id}: last_season precedes first_season")
        for name in (rec.name, *rec.aliases):
            key = normalize_name(name)
            if not key:
                continue
            other = seen_names.get(key)
            if other is not None and other != rec.venue_id:
                raise ValueError(
                    f"alias {name!r} maps to both {other} and {rec.venue_id} in {csv_path}"
                )
            seen_names[key] = rec.venue_id


@lru_cache(maxsize=1)
def venues_by_id() -> dict[str, VenueRecord]:
    """``stadium_id -> VenueRecord``."""
    return {rec.venue_id: rec for rec in load_venues()}


@lru_cache(maxsize=1)
def venues_by_name() -> dict[str, VenueRecord]:
    """Normalised venue name (including aliases) ``-> VenueRecord``."""
    index: dict[str, VenueRecord] = {}
    for rec in load_venues():
        for name in (rec.name, *rec.aliases):
            key = normalize_name(name)
            if key:
                index[key] = rec
    return index


def venues_frame() -> pl.DataFrame:
    """The reference table as a polars frame."""
    records = load_venues()
    return pl.DataFrame(
        {
            "venue_id": [r.venue_id for r in records],
            "name": [r.name for r in records],
            "city": [r.city for r in records],
            "state": [r.state for r in records],
            "country": [r.country for r in records],
            "latitude": [r.latitude for r in records],
            "longitude": [r.longitude for r in records],
            "elevation_m": [r.elevation_m for r in records],
            "altitude_feet": [r.altitude_feet for r in records],
            "timezone": [r.timezone for r in records],
            "roof_type": [r.roof_type for r in records],
            "surface": [r.surface for r in records],
            "capacity": [r.capacity for r in records],
            "first_season": [r.first_season for r in records],
            "last_season": [r.last_season for r in records],
            "home_teams": [list(r.home_teams) for r in records],
            "aliases": [list(r.aliases) for r in records],
        }
    )


def resolve_venue(
    stadium_id: str | None,
    stadium_name: str | None = None,
    location: str | None = None,
    game_id: str | None = None,
) -> VenueResolution:
    """Resolve one games-frame row to the venue the game was actually played in.

    Resolution order, most trustworthy first:

    1. **Per-game override** (:data:`GAME_VENUE_OVERRIDES`) -- the only recourse when
       both ``stadium_id`` and ``stadium`` were copied from the designated home team.
    2. **Name beats id.** When ``stadium_name`` resolves to a *different* venue than
       ``stadium_id`` does, the name wins: nflverse backfills the venue name for
       neutral-site games while leaving ``stadium_id`` stale, never the reverse.
    3. ``stadium_id`` lookup.
    4. ``stadium_name`` lookup (covers a missing or unknown id).

    ``location`` is advisory: it is recorded in ``notes`` but never gates resolution,
    because designated-home games played abroad carry ``location == "Home"``.

    Returns a :class:`VenueResolution`; ``bool(result)`` is False when nothing matched.
    """
    by_id = venues_by_id()
    by_name = venues_by_name()
    sid = (stadium_id or "").strip() or None
    notes: list[str] = []
    if location:
        notes.append(f"location={location}")

    if game_id and game_id in GAME_VENUE_OVERRIDES:
        target = GAME_VENUE_OVERRIDES[game_id]
        venue = by_id.get(target)
        if venue is None:  # pragma: no cover - guarded by test_overrides_all_resolve
            raise KeyError(f"override for {game_id} names unknown venue_id {target!r}")
        if sid and sid != target:
            notes.append(f"stadium_id={sid} superseded")
        return VenueResolution(venue, "game_id_override", sid, overridden=True, notes=tuple(notes))

    by_id_hit = by_id.get(sid) if sid else None
    by_name_hit = by_name.get(normalize_name(stadium_name))

    if by_name_hit is not None and by_id_hit is not None and by_name_hit is not by_id_hit:
        notes.append(f"stadium_id={sid} disagrees with stadium={stadium_name!r}")
        return VenueResolution(by_name_hit, "name_over_id", sid, overridden=True, notes=tuple(notes))

    if by_id_hit is not None:
        return VenueResolution(by_id_hit, "stadium_id", sid, notes=tuple(notes))

    if by_name_hit is not None:
        notes.append(f"unknown stadium_id={sid!r}")
        return VenueResolution(by_name_hit, "stadium_name", sid, notes=tuple(notes))

    notes.append(f"unresolved stadium_id={sid!r} stadium={stadium_name!r}")
    return VenueResolution(None, "unresolved", sid, notes=tuple(notes))


VENUE_COLUMNS = (
    "venue_id",
    "venue_name",
    "latitude",
    "longitude",
    "elevation_m",
    "venue_timezone",
    "venue_roof_type",
    "venue_resolved_by",
    "venue_overridden",
)


def resolve_games(games: pl.DataFrame) -> pl.DataFrame:
    """Attach :data:`VENUE_COLUMNS` to a games frame.

    Reads ``game_id`` / ``stadium_id`` / ``stadium`` / ``location`` when present;
    a missing column is treated as all-null.
    """
    height = games.height

    def column(name: str) -> list[object]:
        if name in games.columns:
            return games.get_column(name).to_list()
        return [None] * height

    game_ids = column("game_id")
    stadium_ids = column("stadium_id")
    stadium_names = column("stadium")
    locations = column("location")

    resolutions = [
        resolve_venue(
            stadium_id=stadium_ids[i],  # type: ignore[arg-type]
            stadium_name=stadium_names[i],  # type: ignore[arg-type]
            location=locations[i],  # type: ignore[arg-type]
            game_id=game_ids[i],  # type: ignore[arg-type]
        )
        for i in range(height)
    ]
    venues = [r.venue for r in resolutions]
    return games.with_columns(
        pl.Series("venue_id", [v.venue_id if v else None for v in venues], dtype=pl.String),
        pl.Series("venue_name", [v.name if v else None for v in venues], dtype=pl.String),
        pl.Series("latitude", [v.latitude if v else None for v in venues], dtype=pl.Float64),
        pl.Series("longitude", [v.longitude if v else None for v in venues], dtype=pl.Float64),
        pl.Series("elevation_m", [v.elevation_m if v else None for v in venues], dtype=pl.Float64),
        pl.Series("venue_timezone", [v.timezone if v else None for v in venues], dtype=pl.String),
        pl.Series("venue_roof_type", [v.roof_type if v else None for v in venues], dtype=pl.String),
        pl.Series("venue_resolved_by", [r.how for r in resolutions], dtype=pl.String),
        pl.Series("venue_overridden", [r.overridden for r in resolutions], dtype=pl.Boolean),
    )
