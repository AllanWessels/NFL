"""Regenerate the tiny offline fixtures used by the data tests.

Run with ``uv run python tests/fixtures/build_fixtures.py``. The parquet files it
writes are committed (see the ``!tests/fixtures`` rules in ``.gitignore``) so the
test suite never touches the network. Keep them *small* and hand-checkable: the
point is that a human can verify the expected coach records by counting on their
fingers.

Column sets mirror the real nflverse assets — all 46 schedules columns, and a
representative slice of the 372 play-by-play columns chosen to include every
market and third-party win-probability column the blocklist must catch.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

FIXTURE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# schedules — 9 games over 2 seasons, hand-built so the records are checkable.
#
#   Andy Reid (KC)       2023: W, T, W(post)   2024: W, W
#   Sean McDermott (BUF) 2023: W, T, L(post)   2024: W, L
#   Sean Payton (DEN)    2023: L               2024: L, W
#   Mike McDaniel (MIA)  2023: L               2024: L, L
#
# Game 9 is scheduled but unplayed (null scores) and must contribute nothing.
# --------------------------------------------------------------------------- #

_GAMES: list[dict[str, object]] = [
    {
        "game_id": "2023_01_DEN_KC",
        "season": 2023,
        "game_type": "REG",
        "week": 1,
        "gameday": "2023-09-10",
        "weekday": "Sunday",
        "gametime": "16:25",
        "away_team": "DEN",
        "away_score": 10,
        "home_team": "KC",
        "home_score": 20,
        "away_coach": "Sean Payton",
        "home_coach": "Andy Reid",
        "stadium_id": "KAN00",
        "stadium": "GEHA Field at Arrowhead Stadium",
        "roof": "outdoors",
        "espn": "401547353",
    },
    {
        "game_id": "2023_01_MIA_BUF",
        "season": 2023,
        "game_type": "REG",
        "week": 1,
        "gameday": "2023-09-10",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "MIA",
        "away_score": 21,
        "home_team": "BUF",
        "home_score": 24,
        "away_coach": "Mike McDaniel",
        "home_coach": "Sean McDermott",
        "stadium_id": "BUF00",
        "stadium": "Highmark Stadium",
        "roof": "outdoors",
        "espn": "401547354",
    },
    {
        "game_id": "2023_02_KC_BUF",
        "season": 2023,
        "game_type": "REG",
        "week": 2,
        "gameday": "2023-09-17",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "KC",
        "away_score": 27,
        "home_team": "BUF",
        "home_score": 27,
        "away_coach": "Andy Reid",
        "home_coach": "Sean McDermott",
        "stadium_id": "BUF00",
        "stadium": "Highmark Stadium",
        "roof": "outdoors",
        "espn": "401547355",
    },
    {
        "game_id": "2023_19_BUF_KC",
        "season": 2023,
        "game_type": "DIV",
        "week": 19,
        "gameday": "2024-01-21",
        "weekday": "Sunday",
        "gametime": "18:30",
        "away_team": "BUF",
        "away_score": 24,
        "home_team": "KC",
        "home_score": 27,
        "away_coach": "Sean McDermott",
        "home_coach": "Andy Reid",
        "stadium_id": "KAN00",
        "stadium": "GEHA Field at Arrowhead Stadium",
        "roof": "outdoors",
        "espn": "401547600",
    },
    {
        "game_id": "2024_01_DEN_KC",
        "season": 2024,
        "game_type": "REG",
        "week": 1,
        "gameday": "2024-09-08",
        "weekday": "Sunday",
        "gametime": "16:25",
        "away_team": "DEN",
        "away_score": 14,
        "home_team": "KC",
        "home_score": 21,
        "away_coach": "Sean Payton",
        "home_coach": "Andy Reid",
        "stadium_id": "KAN00",
        "stadium": "GEHA Field at Arrowhead Stadium",
        "roof": "outdoors",
        "espn": "401671700",
    },
    {
        "game_id": "2024_01_MIA_BUF",
        "season": 2024,
        "game_type": "REG",
        "week": 1,
        "gameday": "2024-09-08",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "MIA",
        "away_score": 17,
        "home_team": "BUF",
        "home_score": 31,
        "away_coach": "Mike McDaniel",
        "home_coach": "Sean McDermott",
        "stadium_id": "BUF00",
        "stadium": "Highmark Stadium",
        "roof": "outdoors",
        "espn": "401671701",
    },
    {
        "game_id": "2024_02_KC_MIA",
        "season": 2024,
        "game_type": "REG",
        "week": 2,
        "gameday": "2024-09-15",
        "weekday": "Sunday",
        "gametime": "13:00",
        "away_team": "KC",
        "away_score": 30,
        "home_team": "MIA",
        "home_score": 20,
        "away_coach": "Andy Reid",
        "home_coach": "Mike McDaniel",
        "stadium_id": "MIA00",
        "stadium": "Hard Rock Stadium",
        "roof": "outdoors",
        "espn": "401671702",
    },
    {
        "game_id": "2024_03_BUF_DEN",
        "season": 2024,
        "game_type": "REG",
        "week": 3,
        "gameday": "2024-09-22",
        "weekday": "Sunday",
        "gametime": "16:05",
        "away_team": "BUF",
        "away_score": 20,
        "home_team": "DEN",
        "home_score": 23,
        "away_coach": "Sean McDermott",
        "home_coach": "Sean Payton",
        "stadium_id": "DEN00",
        "stadium": "Empower Field at Mile High",
        "roof": "outdoors",
        "espn": "401671703",
    },
    {
        # Scheduled, never played: must contribute nothing to any record.
        "game_id": "2024_04_KC_BUF",
        "season": 2024,
        "game_type": "REG",
        "week": 4,
        "gameday": "2024-09-29",
        "weekday": "Sunday",
        "gametime": "20:20",
        "away_team": "KC",
        "away_score": None,
        "home_team": "BUF",
        "home_score": None,
        "away_coach": "Andy Reid",
        "home_coach": "Sean McDermott",
        "stadium_id": "BUF00",
        "stadium": "Highmark Stadium",
        "roof": "outdoors",
        "espn": "401671704",
    },
]

#: Market quotes, indexed by game. Deliberately distinctive values so a test can
#: assert an exact number survived the split into the ``market`` table.
_LINES: dict[str, tuple[object, ...]] = {
    "2023_01_DEN_KC": (-7.0, 53.5, -340, 270, -110, -110, -110, -110),
    "2023_01_MIA_BUF": (-3.5, 49.5, -190, 160, -105, -115, -110, -110),
    "2023_02_KC_BUF": (2.5, 45.5, 115, -135, -110, -110, -108, -112),
    "2023_19_BUF_KC": (-2.5, 45.5, -145, 125, -110, -110, -110, -110),
    "2024_01_DEN_KC": (-8.5, 41.5, -400, 310, -110, -110, -112, -108),
    "2024_01_MIA_BUF": (-2.5, 48.5, -145, 122, -110, -110, -110, -110),
    "2024_02_KC_MIA": (-3.0, 42.5, -160, 135, -108, -112, -110, -110),
    "2024_03_BUF_DEN": (6.5, 38.5, 240, -300, -110, -110, -110, -110),
    "2024_04_KC_BUF": (-1.5, 47.5, -125, 105, -110, -110, -110, -110),
}

_LINE_FIELDS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
)

#: Every schedules column in nflverse order, so the fixture is schema-faithful.
_SCHEDULES_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "location",
    "result",
    "total",
    "overtime",
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_qb_id",
    "home_qb_id",
    "away_qb_name",
    "home_qb_name",
    "away_coach",
    "home_coach",
    "referee",
    "stadium_id",
    "stadium",
]

_INT_COLUMNS = {
    "season",
    "week",
    "away_score",
    "home_score",
    "result",
    "total",
    "overtime",
    "gsis",
    "pff",
    "ftn",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "under_odds",
    "over_odds",
    "div_game",
    "temp",
    "wind",
}
_FLOAT_COLUMNS = {"spread_line", "total_line"}


def build_schedules() -> pl.DataFrame:
    rows = []
    for game in _GAMES:
        row: dict[str, object] = dict.fromkeys(_SCHEDULES_COLUMNS)
        row.update(game)
        for field, value in zip(_LINE_FIELDS, _LINES[str(game["game_id"])], strict=True):
            row[field] = value
        home, away = game["home_score"], game["away_score"]
        if home is not None and away is not None:
            row["result"] = int(home) - int(away)  # type: ignore[arg-type]
            row["total"] = int(home) + int(away)  # type: ignore[arg-type]
            row["overtime"] = 0
        row["location"] = "Home"
        row["surface"] = "grass"
        row["div_game"] = 1 if game["game_type"] == "REG" and game["week"] in (2, 3) else 0
        row["away_rest"] = 7
        row["home_rest"] = 7
        row["referee"] = "Carl Cheffers"
        rows.append(row)

    schema = {
        col: (
            pl.Int64 if col in _INT_COLUMNS else pl.Float64 if col in _FLOAT_COLUMNS else pl.String
        )
        for col in _SCHEDULES_COLUMNS
    }
    return pl.DataFrame(rows, schema=schema)


# --------------------------------------------------------------------------- #
# play-by-play — two games, four plays each, carrying every blocked column.
# --------------------------------------------------------------------------- #

_PBP_GAMES = (
    ("2023_01_DEN_KC", 2023, 1, "KC", "DEN", "Andy Reid", "Sean Payton", "KAN00", -7.0, 53.5),
    ("2024_01_DEN_KC", 2024, 1, "KC", "DEN", "Andy Reid", "Sean Payton", "KAN00", -8.5, 41.5),
)


def build_pbp() -> pl.DataFrame:
    rows = []
    for game_id, season, week, home, away, hc, ac, stadium_id, spread, total in _PBP_GAMES:
        for i in range(4):
            posteam, defteam = (home, away) if i % 2 == 0 else (away, home)
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": 40.0 + i * 25,
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_date": "2023-09-10" if season == 2023 else "2024-09-08",
                    "home_team": home,
                    "away_team": away,
                    "posteam": posteam,
                    "defteam": defteam,
                    "qtr": 1 + i,
                    "down": 1.0 + (i % 4),
                    "ydstogo": 10 - i,
                    "yardline_100": 75 - i * 5,
                    "game_seconds_remaining": 3600 - i * 800,
                    "half_seconds_remaining": 1800 - i * 400,
                    "quarter_seconds_remaining": 900 - i * 100,
                    "total_home_score": i * 3,
                    "total_away_score": i * 2,
                    "score_differential": float(i),
                    "posteam_timeouts_remaining": 3,
                    "defteam_timeouts_remaining": 3,
                    "play_type": "pass" if i % 2 else "run",
                    "ep": 1.5 + i * 0.1,
                    "epa": 0.25 - i * 0.1,
                    "roof": "outdoors",
                    "surface": "astroturf",
                    "stadium_id": stadium_id,
                    "home_coach": hc,
                    "away_coach": ac,
                    # --- everything below must never reach a feature frame ---
                    "spread_line": spread,
                    "total_line": total,
                    "wp": 0.60 + i * 0.02,
                    "def_wp": 0.40 - i * 0.02,
                    "home_wp": 0.62 + i * 0.02,
                    "away_wp": 0.38 - i * 0.02,
                    "home_wp_post": 0.63 + i * 0.02,
                    "away_wp_post": 0.37 - i * 0.02,
                    "wpa": 0.01 * (i + 1),
                    "air_wpa": 0.004 * (i + 1),
                    "yac_wpa": 0.003 * (i + 1),
                    "comp_air_wpa": 0.002 * (i + 1),
                    "comp_yac_wpa": 0.001 * (i + 1),
                    "total_home_pass_wpa": 0.02 * i,
                    "total_away_pass_wpa": -0.02 * i,
                    "total_home_rush_wpa": 0.01 * i,
                    "total_away_rush_wpa": -0.01 * i,
                    "vegas_wp": 0.71 + i * 0.01,
                    "vegas_home_wp": 0.73 + i * 0.01,
                    "vegas_wpa": 0.008 * (i + 1),
                    "vegas_home_wpa": 0.009 * (i + 1),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    schedules = build_schedules()
    schedules.write_parquet(FIXTURE_DIR / "nflverse_schedules.parquet")
    pbp = build_pbp()
    pbp.write_parquet(FIXTURE_DIR / "nflverse_pbp.parquet")
    print(f"schedules: {schedules.height} rows x {schedules.width} cols")
    print(f"pbp:       {pbp.height} rows x {pbp.width} cols")


if __name__ == "__main__":
    main()
