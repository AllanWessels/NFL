"""Venue reference table integrity and the neutral-site ``stadium_id`` fix."""

from __future__ import annotations

import polars as pl
import pytest

from nfl.data.schemas import Venue
from nfl.data.venues import (
    GAME_VENUE_OVERRIDES,
    ROOF_TYPES,
    VENUES_CSV,
    load_venues,
    normalize_name,
    resolve_games,
    resolve_venue,
    venues_by_id,
    venues_by_name,
    venues_frame,
)

# Every (stadium_id, stadium) pair nflverse emits for 2010+, taken from games.csv.
# If the reference table stops covering one of these, weather ingest silently loses games.
NFLVERSE_PAIRS = [
    ("ATL00", "Georgia Dome"),
    ("ATL97", "Mercedes-Benz Stadium"),
    ("BAL00", "M&T Bank Stadium"),
    ("BOS00", "Gillette Stadium"),
    ("BUF00", "New Era Field"),
    ("BUF00", "Ralph Wilson Stadium"),
    ("BUF00", "Highmark Stadium"),
    ("BUF01", "Rogers Centre"),
    ("CAR00", "Bank of America Stadium"),
    ("CHI98", "Soldier Field"),
    ("CIN00", "Paul Brown Stadium"),
    ("CIN00", "Paycor Stadium"),
    ("CLE00", "FirstEnergy Stadium"),
    ("CLE00", "Cleveland Browns Stadium"),
    ("CLE00", "Huntington Bank Field"),
    ("DAL00", "AT&T Stadium"),
    ("DAL00", "Cowboys Stadium"),
    ("DEN00", "Sports Authority Field at Mile High"),
    ("DEN00", "Empower Field at Mile High"),
    ("DEN00", "Invesco Field at Mile High"),
    ("DET00", "Ford Field"),
    ("FRA00", "Deutsche Bank Park"),
    ("GER00", "Allianz Arena"),
    ("GNB00", "Lambeau Field"),
    ("HOU00", "NRG Stadium"),
    ("HOU00", "Reliant Stadium"),
    ("IND00", "Lucas Oil Stadium"),
    ("JAX00", "TIAA Bank Stadium"),
    ("JAX00", "EverBank Field"),
    ("JAX00", "EverBank Stadium"),
    ("KAN00", "Arrowhead Stadium"),
    ("KAN00", "GEHA Field at Arrowhead Stadium"),
    ("LAX01", "SoFi Stadium"),
    ("LAX97", "StubHub Center"),
    ("LAX99", "Los Angeles Memorial Coliseum"),
    ("LON00", "Wembley Stadium"),
    ("LON01", "Twickenham Stadium"),
    ("LON02", "Tottenham Stadium"),
    ("MEX00", "Azteca Stadium"),
    ("MIA00", "Hard Rock Stadium"),
    ("MIA00", "Sun Life Stadium"),
    ("MIN00", "Mall of America Field"),
    ("MIN01", "U.S. Bank Stadium"),
    ("MIN98", "TCF Bank Stadium"),
    ("NAS00", "Nissan Stadium"),
    ("NAS00", "LP Field"),
    ("NOR00", "Mercedes-Benz Superdome"),
    ("NOR00", "Louisiana Superdome"),
    ("NOR00", "Caesars Superdome"),
    ("NYC01", "MetLife Stadium"),
    ("NYC01", "New Meadowlands Stadium"),
    ("OAK00", "O.co Coliseum"),
    ("OAK00", "Oakland-Alameda County Coliseum"),
    ("OAK00", "Ring Central Coliseum"),
    ("PHI00", "Lincoln Financial Field"),
    ("PHO00", "State Farm Stadium"),
    ("PHO00", "University of Phoenix Stadium"),
    ("PIT00", "Heinz Field"),
    ("PIT00", "Acrisure Stadium"),
    ("SAO00", "Arena Corinthians"),
    ("SDG00", "Qualcomm Stadium"),
    ("SEA00", "CenturyLink Field"),
    ("SEA00", "Lumen Field"),
    ("SEA00", "Qwest Field"),
    ("SFO00", "Candlestick Park"),
    ("SFO01", "Levi's Stadium"),
    ("STL00", "Edward Jones Dome"),
    ("TAM00", "Raymond James Stadium"),
    ("VEG00", "Allegiant Stadium"),
    ("WAS00", "FedExField"),
    ("WAS00", "Northwest Stadium"),
]

# The two flavours of the neutral-site bug, taken verbatim from nflverse games.csv.
# (game_id, stadium_id, stadium, location, expected venue_id, expected rule)
NEUTRAL_SITE_ROWS = [
    # Flavour 1: name is right, stadium_id is the designated home team's.
    ("2026_01_SF_LA", "LAX01", "Melbourne Cricket Ground", "Neutral", "MEL00", "name_over_id"),
    ("2026_03_BAL_DAL", "DAL00", "Maracana Stadium", "Neutral", "RIO00", "name_over_id"),
    ("2026_04_IND_WAS", "WAS00", "Tottenham Hotspur Stadium", "Neutral", "LON02", "name_over_id"),
    ("2026_06_HOU_JAX", "JAX00", "Wembley Stadium", "Neutral", "LON00", "name_over_id"),
    ("2026_07_PIT_NO", "NOR00", "Stade de France", "Neutral", "PAR00", "name_over_id"),
    ("2026_09_CIN_ATL", "ATL97", "Bernabeu", "Neutral", "MAD00", "name_over_id"),
    ("2026_10_NE_DET", "DET00", "FC Bayern Munich Stadium", "Neutral", "GER00", "name_over_id"),
    ("2026_11_MIN_SF", "SFO01", "Estadio Banorte", "Neutral", "MEX00", "name_over_id"),
    # ...and the same flavour on a row nflverse marks location="Home": Jacksonville's
    # designated home game in London. Gating on location == "Neutral" would miss it.
    ("2026_05_PHI_JAX", "JAX00", "Tottenham Hotspur Stadium", "Home", "LON02", "name_over_id"),
    # Flavour 2: both stadium_id and stadium name are the designated home team's.
    ("2025_01_KC_LAC", "LAX01", "SoFi Stadium", "Neutral", "SAO00", "game_id_override"),
    ("2025_04_MIN_PIT", "PIT00", "Acrisure Stadium", "Neutral", "DUB00", "game_id_override"),
    ("2025_05_MIN_CLE", "CLE00", "FirstEnergy Stadium", "Neutral", "LON02", "game_id_override"),
    ("2025_06_DEN_NYJ", "NYC01", "MetLife Stadium", "Neutral", "LON02", "game_id_override"),
    ("2025_07_LA_JAX", "JAX00", "TIAA Bank Stadium", "Neutral", "LON00", "game_id_override"),
    ("2025_10_ATL_IND", "IND00", "Lucas Oil Stadium", "Neutral", "BER00", "game_id_override"),
    ("2025_11_WAS_MIA", "MIA00", "Hard Rock Stadium", "Neutral", "MAD00", "game_id_override"),
]

# Genuinely-neutral games at a venue the stadium_id already names correctly (Super Bowls,
# historic international games). These must NOT be rewritten.
CORRECTLY_TAGGED_NEUTRAL_ROWS = [
    ("2024_22_KC_PHI", "NOR00", "Mercedes-Benz Superdome", "NOR00"),
    ("2025_22_SEA_NE", "SFO01", "Levi's Stadium", "SFO01"),
    ("2024_19_MIN_LA", "PHO00", "State Farm Stadium", "PHO00"),
    ("2024_07_NE_JAX", "LON00", "Wembley Stadium", "LON00"),
    ("2024_05_NYJ_MIN", "LON02", "Tottenham Stadium", "LON02"),
    ("2024_01_GB_PHI", "SAO00", "Arena Corinthians", "SAO00"),
    ("2024_10_NYG_CAR", "GER00", "Allianz Arena", "GER00"),
]


def test_csv_is_committed() -> None:
    assert VENUES_CSV.exists(), "reference/venues.csv must be committed, not generated"


def test_loads_and_validates() -> None:
    """``load_venues`` raises on duplicate ids, bad roofs, out-of-range coords, alias clashes."""
    records = load_venues()
    assert len(records) >= 50
    assert len({r.venue_id for r in records}) == len(records)


@pytest.mark.parametrize("record", load_venues(), ids=lambda r: r.venue_id)
def test_record_fields_are_sane(record) -> None:
    assert record.roof_type in ROOF_TYPES
    assert -90.0 <= record.latitude <= 90.0
    assert -180.0 <= record.longitude <= 180.0
    assert -100.0 <= record.elevation_m <= 3000.0
    assert record.timezone.count("/") >= 1
    assert record.name and record.city and record.country
    assert 2010 <= record.first_season <= 2030
    if record.capacity is not None:
        assert 20_000 <= record.capacity <= 120_000


def test_records_validate_against_venue_schema() -> None:
    """The CSV must be loadable straight into the pydantic ``Venue`` model."""
    for record in load_venues():
        venue = Venue(**record.to_venue_kwargs())  # type: ignore[arg-type]
        assert venue.venue_id == record.venue_id
        assert venue.roof_type == record.roof_type


def test_elevation_converts_to_feet() -> None:
    denver = venues_by_id()["DEN00"]
    assert denver.elevation_m == pytest.approx(1610, abs=30)
    assert denver.altitude_feet == pytest.approx(5280, abs=100)


def test_high_altitude_venues() -> None:
    """Mexico City and Denver are the two altitude outliers; both must be right."""
    assert venues_by_id()["MEX00"].elevation_m > 2000
    assert venues_by_id()["DEN00"].elevation_m > 1500


@pytest.mark.parametrize(("stadium_id", "stadium"), NFLVERSE_PAIRS)
def test_every_nflverse_pair_resolves(stadium_id: str, stadium: str) -> None:
    result = resolve_venue(stadium_id, stadium, "Home", f"2020_01_AA_{stadium_id[:2]}")
    assert result, f"{stadium_id} / {stadium} did not resolve"
    assert result.venue is not None
    assert result.venue.venue_id == stadium_id


@pytest.mark.parametrize(
    ("game_id", "stadium_id", "stadium", "location", "expected", "how"), NEUTRAL_SITE_ROWS
)
def test_neutral_site_bug_is_corrected(
    game_id: str, stadium_id: str, stadium: str, location: str, expected: str, how: str
) -> None:
    result = resolve_venue(stadium_id, stadium, location, game_id)
    assert result.venue is not None
    assert result.venue.venue_id == expected
    assert result.how == how
    assert result.overridden is True
    assert result.stadium_id == stadium_id  # the stale id is preserved for auditing


@pytest.mark.parametrize(
    ("game_id", "stadium_id", "stadium", "expected"), CORRECTLY_TAGGED_NEUTRAL_ROWS
)
def test_correctly_tagged_neutral_games_are_left_alone(
    game_id: str, stadium_id: str, stadium: str, expected: str
) -> None:
    result = resolve_venue(stadium_id, stadium, "Neutral", game_id)
    assert result.venue is not None
    assert result.venue.venue_id == expected
    assert result.overridden is False


def test_overrides_all_name_known_venues() -> None:
    known = venues_by_id()
    for game_id, venue_id in GAME_VENUE_OVERRIDES.items():
        assert venue_id in known, f"{game_id} override names unknown venue {venue_id}"


def test_unknown_venue_returns_falsy_resolution() -> None:
    result = resolve_venue("ZZZ99", "Some Field Nobody Built", "Home", "2030_01_AA_BB")
    assert not result
    assert result.venue is None
    assert result.how == "unresolved"


def test_missing_stadium_id_falls_back_to_name() -> None:
    result = resolve_venue(None, "Lambeau Field", "Home", "2020_01_AA_GB")
    assert result.venue is not None
    assert result.venue.venue_id == "GNB00"
    assert result.how == "stadium_name"


@pytest.mark.parametrize(
    ("raw", "expected_id"),
    [
        ("Tottenham Stadium", "LON02"),
        ("Tottenham Hotspur Stadium", "LON02"),
        ("  tottenham hotspur STADIUM ", "LON02"),
        ("Estadio Santiago Bernabéu", "MAD00"),
        ("Bernabeu", "MAD00"),
        ("Levi's Stadium", "SFO01"),
        ("Levis Stadium", "SFO01"),
        ("M&T Bank Stadium", "BAL00"),
        ("U.S. Bank Stadium", "MIN01"),
        ("US Bank Stadium", "MIN01"),
    ],
)
def test_name_normalisation_handles_accents_case_and_punctuation(
    raw: str, expected_id: str
) -> None:
    assert venues_by_name()[normalize_name(raw)].venue_id == expected_id


def test_similar_bank_names_do_not_collide() -> None:
    """A too-aggressive normaliser would merge these; the loader's validator catches it."""
    index = venues_by_name()
    assert index[normalize_name("Huntington Bank Field")].venue_id == "CLE00"
    assert index[normalize_name("TCF Bank Stadium")].venue_id == "MIN98"
    assert index[normalize_name("Deutsche Bank Park")].venue_id == "FRA00"
    assert index[normalize_name("Bank of America Stadium")].venue_id == "CAR00"


def test_mercedes_benz_stadium_is_not_the_superdome() -> None:
    index = venues_by_name()
    assert index[normalize_name("Mercedes-Benz Stadium")].venue_id == "ATL97"
    assert index[normalize_name("Mercedes-Benz Superdome")].venue_id == "NOR00"


def test_venues_frame_shape() -> None:
    frame = venues_frame()
    assert frame.height == len(load_venues())
    assert frame.get_column("venue_id").is_duplicated().sum() == 0


def test_resolve_games_adds_columns() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["2025_04_MIN_PIT", "2024_01_ARI_BUF"],
            "stadium_id": ["PIT00", "BUF00"],
            "stadium": ["Acrisure Stadium", "Highmark Stadium"],
            "location": ["Neutral", "Home"],
        }
    )
    out = resolve_games(games)
    assert out.get_column("venue_id").to_list() == ["DUB00", "BUF00"]
    assert out.get_column("venue_overridden").to_list() == [True, False]
    assert out.get_column("venue_resolved_by").to_list() == ["game_id_override", "stadium_id"]
    # Dublin is east of Greenwich but only just; Orchard Park is firmly west.
    assert out.get_column("longitude").to_list() == pytest.approx([-6.2512, -78.7870], abs=1e-4)


def test_resolve_games_tolerates_missing_columns() -> None:
    out = resolve_games(pl.DataFrame({"game_id": ["2024_01_AA_BB"]}))
    assert out.get_column("venue_id").to_list() == [None]
    assert out.get_column("venue_resolved_by").to_list() == ["unresolved"]
