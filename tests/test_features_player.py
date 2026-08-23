"""Tests for :mod:`nfl.features.player`.

Two kinds of input:

* **Synthetic** — a four-team, four-week league built by :func:`synthetic_pbp`
  below, with deliberately lopsided unit strengths so the opponent adjustment
  has something to bite on. Everything about as-of-ness, the §6 formula, starter
  selection and the delete-the-future guarantee is checked here, where the right
  answer is arithmetic rather than trivia.
* **Real** — ``tests/fixtures/nflverse/pbp_players_2024_01_BAL_KC.parquet``, the
  178 plays of BAL @ KC 2024 week 1 sliced down to the player-id columns this
  module reads. Regenerate it with::

      pl.scan_parquet("data/raw/nflverse/pbp/season=2024/part-0.parquet")
        .filter(pl.col("game_id") == "2024_01_BAL_KC")
        .select(sorted(_PBP_COLUMNS_USED | {"home_team", "away_team",
                                            "play_type", "qtr"}))

  The box-score assertions against it are the check that the pbp column
  semantics (a sack is a dropback but not an attempt; a punt returner belongs to
  ``defteam`` but a kick returner to ``posteam``) are read the way nflfastR
  means them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from nfl.data.live.base import LiveInjury, LiveSnapshot
from nfl.data.market_blocklist import MARKET_COLUMNS, assert_no_market_columns
from nfl.features import player as feat

FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"

SEASON = 2023
TEAMS = ("AAA", "BBB", "CCC", "DDD")

#: Round-robin ``(week, away, home)``; every team plays every week.
SCHEDULE: tuple[tuple[int, str, str], ...] = (
    (1, "AAA", "BBB"),
    (1, "CCC", "DDD"),
    (2, "AAA", "CCC"),
    (2, "BBB", "DDD"),
    (3, "AAA", "DDD"),
    (3, "BBB", "CCC"),
    (4, "BBB", "AAA"),
    (4, "DDD", "CCC"),
)

#: Offensive EPA per play each team generates, before the defence is applied.
OFFENSE_SKILL = {"AAA": 0.30, "BBB": 0.10, "CCC": -0.10, "DDD": -0.30}
#: EPA per play each defence removes. AAA is the league's best, DDD the worst.
DEFENSE_SKILL = {"AAA": 0.40, "BBB": 0.15, "CCC": -0.05, "DDD": -0.30}


def _players(team: str) -> dict[str, str]:
    return {
        "QB": f"{team}-QB1",
        "QB2": f"{team}-QB2",
        "RB": f"{team}-RB1",
        "WR1": f"{team}-WR1",
        "WR2": f"{team}-WR2",
        "WR3": f"{team}-WR3",
        "TE": f"{team}-TE1",
        "DL": f"{team}-DL1",
        "LB": f"{team}-LB1",
        "CB": f"{team}-CB1",
        "S": f"{team}-S1",
        "KR": f"{team}-WR3",
    }


def synthetic_pbp(
    *,
    weeks: tuple[int, ...] = (1, 2, 3, 4),
    pass_plays: int = 30,
    rush_plays: int = 30,
    with_market_columns: bool = False,
) -> pl.DataFrame:
    """A tiny but structurally faithful league: 4 teams, round robin, known EPA.

    ``pass_plays``/``rush_plays`` are per team per game and default high enough
    that a unit clears :data:`nfl.features.player.MIN_UNIT_PLAYS` after one game.
    """
    rows: list[dict[str, object]] = []
    for week, away, home in SCHEDULE:
        if week not in weeks:
            continue
        game_id = f"{SEASON}_{week:02d}_{away}_{home}"
        for offense, defense in ((home, away), (away, home)):
            names = _players(offense)
            stoppers = _players(defense)
            epa = OFFENSE_SKILL[offense] - DEFENSE_SKILL[defense]
            for i in range(pass_plays):
                sack = i % 15 == 0
                complete = (not sack) and i % 3 != 0
                rows.append(
                    {
                        "game_id": game_id,
                        "play_id": float(len(rows) + 1),
                        "season": SEASON,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "posteam": offense,
                        "defteam": defense,
                        "play_type": "pass",
                        "pass_attempt": 1.0,
                        "rush_attempt": 0.0,
                        "qb_dropback": 1.0,
                        "sack": 1.0 if sack else 0.0,
                        "complete_pass": 1.0 if complete else 0.0,
                        "interception": 1.0 if i % 29 == 0 and not sack else 0.0,
                        "pass_touchdown": 1.0 if complete and i % 11 == 0 else 0.0,
                        "rush_touchdown": 0.0,
                        "epa": epa,
                        "cpoe": 2.0 if complete else -1.0,
                        "passing_yards": 10.0 if complete else None,
                        "receiving_yards": 10.0 if complete else None,
                        "yards_after_catch": 4.0 if complete else None,
                        "rushing_yards": None,
                        "yards_gained": 10.0 if complete else 0.0,
                        "return_yards": 0.0,
                        "passer_player_id": names["QB"],
                        "receiver_player_id": None if sack else names[f"WR{i % 3 + 1}"],
                        "rusher_player_id": None,
                        "sack_player_id": stoppers["DL"] if sack else None,
                        "interception_player_id": (
                            stoppers["S"] if i % 29 == 0 and not sack else None
                        ),
                        "solo_tackle_1_player_id": stoppers["LB"] if complete else None,
                        "pass_defense_1_player_id": None if complete or sack else stoppers["CB"],
                        "qb_hit_1_player_id": stoppers["DL"] if sack else None,
                        "kickoff_returner_player_id": None,
                        "punt_returner_player_id": None,
                    }
                )
            for i in range(rush_plays):
                rows.append(
                    {
                        "game_id": game_id,
                        "play_id": float(len(rows) + 1),
                        "season": SEASON,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "posteam": offense,
                        "defteam": defense,
                        "play_type": "run",
                        "pass_attempt": 0.0,
                        "rush_attempt": 1.0,
                        "qb_dropback": 0.0,
                        "sack": 0.0,
                        "complete_pass": 0.0,
                        "interception": 0.0,
                        "pass_touchdown": 0.0,
                        "rush_touchdown": 1.0 if i % 17 == 0 else 0.0,
                        "epa": epa,
                        "cpoe": None,
                        "passing_yards": None,
                        "receiving_yards": None,
                        "yards_after_catch": None,
                        "rushing_yards": 4.0,
                        "yards_gained": 4.0,
                        "return_yards": 0.0,
                        "passer_player_id": None,
                        "receiver_player_id": None,
                        "rusher_player_id": names["RB"],
                        "sack_player_id": None,
                        "interception_player_id": None,
                        "solo_tackle_1_player_id": stoppers["LB"],
                        "pass_defense_1_player_id": None,
                        "qb_hit_1_player_id": None,
                        "kickoff_returner_player_id": None,
                        "punt_returner_player_id": None,
                    }
                )
            # One punt per team per game, so the returner block has a sample.
            rows.append(
                {
                    "game_id": game_id,
                    "play_id": float(len(rows) + 1),
                    "season": SEASON,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "posteam": offense,
                    "defteam": defense,
                    "play_type": "punt",
                    "pass_attempt": 0.0,
                    "rush_attempt": 0.0,
                    "qb_dropback": 0.0,
                    "sack": 0.0,
                    "complete_pass": 0.0,
                    "interception": 0.0,
                    "pass_touchdown": 0.0,
                    "rush_touchdown": 0.0,
                    "epa": 0.0,
                    "cpoe": None,
                    "passing_yards": None,
                    "receiving_yards": None,
                    "yards_after_catch": None,
                    "rushing_yards": None,
                    "yards_gained": 8.0,
                    "return_yards": 8.0,
                    "passer_player_id": None,
                    "receiver_player_id": None,
                    "rusher_player_id": None,
                    "sack_player_id": None,
                    "interception_player_id": None,
                    "solo_tackle_1_player_id": None,
                    "pass_defense_1_player_id": None,
                    "qb_hit_1_player_id": None,
                    "kickoff_returner_player_id": None,
                    "punt_returner_player_id": _players(defense)["KR"],
                }
            )
    frame = pl.DataFrame(rows)
    if with_market_columns:
        frame = frame.with_columns(
            pl.lit(-3.5).alias("spread_line"),
            pl.lit(47.5).alias("total_line"),
            pl.lit(0.62).alias("vegas_wp"),
            pl.lit(0.58).alias("wp"),
            pl.lit(0.01).alias("wpa"),
        )
    return frame


def synthetic_schedules(weeks: tuple[int, ...] = (1, 2, 3, 4)) -> pl.DataFrame:
    rows = [
        {
            "game_id": f"{SEASON}_{week:02d}_{away}_{home}",
            "season": SEASON,
            "week": week,
            "gameday": f"{SEASON}-09-{10 + 7 * (week - 1):02d}",
            "gametime": "13:00",
            "home_team": home,
            "away_team": away,
        }
        for week, away, home in SCHEDULE
        if week in weeks
    ]
    return pl.DataFrame(rows)


def synthetic_depth_charts(weeks: tuple[int, ...] = (1, 2, 3, 4)) -> pl.DataFrame:
    rows = []
    for week in weeks:
        for team in TEAMS:
            names = _players(team)
            entries = [
                ("QB", names["QB"], 1, "Offense"),
                ("QB", names["QB2"], 2, "Offense"),
                ("RB", names["RB"], 1, "Offense"),
                ("WR", names["WR1"], 1, "Offense"),
                ("WR", names["WR2"], 2, "Offense"),
                ("WR", names["WR3"], 3, "Offense"),
                ("TE", names["TE"], 1, "Offense"),
                ("DE", names["DL"], 1, "Defense"),
                ("ILB", names["LB"], 1, "Defense"),
                ("CB", names["CB"], 1, "Defense"),
                ("FS", names["S"], 1, "Defense"),
                # A gunner listed at his offensive position must not outrank WR1.
                ("WR", names["WR3"], 1, "Special Teams"),
            ]
            for position, gsis, depth, formation in entries:
                rows.append(
                    {
                        "season": SEASON,
                        "week": week,
                        "club_code": team,
                        "gsis_id": gsis,
                        "position": position,
                        "depth_team": str(depth),
                        "formation": formation,
                    }
                )
    return pl.DataFrame(rows)


def synthetic_rosters(weeks: tuple[int, ...] = (1, 2, 3, 4)) -> pl.DataFrame:
    rows = []
    for week in weeks:
        for index, team in enumerate(TEAMS):
            names = _players(team)
            for position, gsis in (
                ("QB", names["QB"]),
                ("QB", names["QB2"]),
                ("RB", names["RB"]),
                ("WR", names["WR1"]),
                ("WR", names["WR2"]),
                ("WR", names["WR3"]),
                ("TE", names["TE"]),
                ("DE", names["DL"]),
                ("ILB", names["LB"]),
                ("CB", names["CB"]),
                ("FS", names["S"]),
            ):
                rows.append(
                    {
                        "season": SEASON,
                        "week": week,
                        "team": team,
                        "gsis_id": gsis,
                        "position": position,
                        "status": "ACT",
                        "birth_date": f"{1996 + index}-05-04",
                        "years_exp": float(index + 1),
                        "draft_number": float(10 * index + 1),
                        "pfr_id": f"{gsis}-pfr",
                    }
                )
    return pl.DataFrame(rows)


def synthetic_injuries() -> pl.DataFrame:
    """DDD loses its starting QB in week 3; BBB has a questionable WR throughout."""
    rows = []
    for week in (1, 2, 3, 4):
        rows.append(
            {
                "season": SEASON,
                "week": week,
                "team": "BBB",
                "gsis_id": _players("BBB")["WR1"],
                "position": "WR",
                "report_status": "Questionable",
                "practice_status": "Limited Participation In Practice",
            }
        )
    rows.append(
        {
            "season": SEASON,
            "week": 3,
            "team": "DDD",
            "gsis_id": _players("DDD")["QB"],
            "position": "QB",
            "report_status": "Out",
            "practice_status": "Did Not Participate In Practice",
        }
    )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def pbp_players() -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / "pbp_players_2024_01_BAL_KC.parquet")


@pytest.fixture(scope="module")
def league() -> pl.DataFrame:
    return synthetic_pbp()


# --------------------------------------------------------------------------- #
# player_game_stats against the committed real game
# --------------------------------------------------------------------------- #

MAHOMES = "00-0033873"
JACKSON = "00-0034796"
HENRY = "00-0032764"
FLOWERS = "00-0037838"
HARDMAN = "00-0035140"
STEELE = "00-0039325"


def test_player_game_stats_matches_the_box_score(pbp_players: pl.DataFrame) -> None:
    stats = feat.player_game_stats(pbp_players)
    mahomes = stats.filter(pl.col("player_id") == MAHOMES).to_dicts()[0]
    assert mahomes["team"] == "KC"
    assert mahomes["opponent"] == "BAL"
    assert mahomes["pass_attempts"] == 28.0  # official attempts exclude sacks
    assert mahomes["completions"] == 20.0
    assert mahomes["passing_yards"] == 291.0
    assert mahomes["passing_tds"] == 1.0
    assert mahomes["interceptions_thrown"] == 1.0
    assert mahomes["sacks_taken"] == 2.0
    # A sack is a dropback, so dropbacks exceed attempts by exactly the sacks.
    assert mahomes["pass_dropbacks"] == mahomes["pass_attempts"] + mahomes["sacks_taken"]

    henry = stats.filter(pl.col("player_id") == HENRY).to_dicts()[0]
    assert (henry["team"], henry["rush_attempts"], henry["rushing_yards"]) == ("BAL", 13.0, 46.0)
    assert henry["rushing_tds"] == 1.0

    flowers = stats.filter(pl.col("player_id") == FLOWERS).to_dicts()[0]
    assert (flowers["targets"], flowers["receptions"], flowers["receiving_yards"]) == (
        12.0,
        9.0,
        111.0,
    )


def test_player_game_stats_is_one_row_per_player_game(pbp_players: pl.DataFrame) -> None:
    stats = feat.player_game_stats(pbp_players)
    assert stats.select("game_id", "player_id").is_duplicated().sum() == 0
    assert set(stats.get_column("team").unique()) == {"KC", "BAL"}


def test_returners_belong_to_the_returning_team(pbp_players: pl.DataFrame) -> None:
    """nflfastR is asymmetric: punt returner is ``defteam``, kick returner ``posteam``."""
    stats = feat.player_game_stats(pbp_players)
    punt = stats.filter((pl.col("punt_returns") > 0) & (pl.col("player_id") == HARDMAN))
    assert punt.height == 1
    assert punt.get_column("team").item() == "KC"
    kick = stats.filter((pl.col("kick_returns") > 0) & (pl.col("player_id") == STEELE))
    assert kick.height == 1
    assert kick.get_column("team").item() == "KC"


def test_defensive_credit_splits_half_sacks_and_assists(league: pl.DataFrame) -> None:
    stats = feat.player_game_stats(league)
    # Every synthetic tackle is a solo tackle, so it counts 1.0 each.
    lb = stats.filter(pl.col("player_id") == "AAA-LB1")
    assert lb.height > 0
    assert (lb.get_column("tackles") > 0).all()


def test_player_game_stats_requires_the_key_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        feat.player_game_stats(pl.DataFrame({"game_id": ["x"], "season": [2023]}))


# --------------------------------------------------------------------------- #
# §6 formula
# --------------------------------------------------------------------------- #


def test_rank_multiplier_endpoints() -> None:
    assert feat.rank_multiplier(1) == pytest.approx(1 - feat.DEFAULT_K)
    assert feat.rank_multiplier(32) == pytest.approx(1 + feat.DEFAULT_K)
    assert feat.rank_multiplier(16.5) == pytest.approx(1.0)


def test_production_against_a_top_defence_adjusts_upward() -> None:
    """The sign assertion DATA_MODEL §6 asks for by name."""
    assert feat.adjust_stat(100.0, rank=1) > 100.0
    assert feat.adjust_stat(100.0, rank=32) < 100.0
    assert feat.adjust_stat(100.0, rank=1) > feat.adjust_stat(100.0, rank=32)
    # Continuous variant: weakness_z is oriented the same way as rank.
    assert feat.adjust_stat(100.0, weakness_z=-2.0) > 100.0
    assert feat.adjust_stat(100.0, weakness_z=2.0) < 100.0


def test_adjust_stat_wants_exactly_one_strength_argument() -> None:
    with pytest.raises(ValueError):
        feat.adjust_stat(10.0)
    with pytest.raises(ValueError):
        feat.adjust_stat(10.0, rank=1, weakness_z=0.0)


def test_fifty_yards_against_the_best_defence_can_beat_a_hundred_against_the_23rd() -> None:
    """The project thesis, and the ``k`` it actually requires.

    With the §6 linear rank map, ``50 / m(1) > 100 / m(23)`` iff

        50 * (1 + k*(23-16.5)/15.5) > 100 * (1 - k)   <=>   k > 0.4133.

    So at the default ``k = 0.25`` the ordering does *not* flip — the adjustment
    is deliberately conservative — and it takes an aggressive ``k`` to express
    the thesis literally. Both halves are pinned here so that tuning ``k`` on
    the validation season is a visible decision rather than an accident.
    """
    alpha = (23 - 16.5) / 15.5
    crossover = 50 / (100 + 50 * alpha)
    assert crossover == pytest.approx(0.41333, abs=1e-4)
    assert feat.adjust_stat(50.0, rank=1, k=crossover) == pytest.approx(
        feat.adjust_stat(100.0, rank=23, k=crossover)
    )

    aggressive = 0.5
    assert feat.adjust_stat(50.0, rank=1, k=aggressive) > feat.adjust_stat(
        100.0, rank=23, k=aggressive
    )
    assert feat.adjust_stat(50.0, rank=1, k=feat.DEFAULT_K) < feat.adjust_stat(
        100.0, rank=23, k=feat.DEFAULT_K
    )
    # Equal production is always ordered by opponent, at any k.
    for k in (0.1, feat.DEFAULT_K, 0.5):
        assert feat.adjust_stat(100.0, rank=1, k=k) > feat.adjust_stat(100.0, rank=23, k=k)


# --------------------------------------------------------------------------- #
# unit_strength
# --------------------------------------------------------------------------- #


def test_unit_strength_is_empty_in_week_one(league: pl.DataFrame) -> None:
    strength = feat.unit_strength(league)
    week1 = strength.filter(pl.col("week") == 1)
    assert week1.height > 0
    assert week1.get_column("epa_per_play").null_count() == week1.height
    assert (week1.get_column("weakness_z") == 0.0).all()
    assert (week1.get_column("rank") == 16.5).all()


def test_unit_strength_orients_defence_and_offence_the_same_way(league: pl.DataFrame) -> None:
    strength = feat.unit_strength(league)
    week4 = strength.filter(pl.col("week") == 4)

    pass_d = week4.filter(pl.col("unit") == "pass_defense").sort("rank")
    # AAA has the biggest DEFENSE_SKILL, so it allows the least EPA: rank 1.
    assert pass_d.get_column("team").to_list()[0] == "AAA"
    assert pass_d.get_column("team").to_list()[-1] == "DDD"
    assert pass_d.get_column("weakness_z").to_list()[0] < 0  # strong unit -> low z
    assert pass_d.get_column("weakness_z").to_list()[-1] > 0

    pass_o = week4.filter(pl.col("unit") == "pass_offense").sort("rank")
    # AAA has the biggest OFFENSE_SKILL, so it *scores* the most: also rank 1,
    # and its weakness_z is negative -- the sign is flipped for offences.
    assert pass_o.get_column("team").to_list()[0] == "AAA"
    assert pass_o.get_column("weakness_z").to_list()[0] < 0


def test_unit_strength_ignores_the_current_and_later_weeks(league: pl.DataFrame) -> None:
    full = feat.unit_strength(league)
    truncated = feat.unit_strength(league.filter(pl.col("week") <= 2))
    common = ["season", "week", "unit", "team"]
    left = full.filter(pl.col("week") <= 2).sort(common)
    right = truncated.sort(common)
    assert left.equals(right)


# --------------------------------------------------------------------------- #
# opponent_adjust
# --------------------------------------------------------------------------- #


def _strength(rows: list[tuple[int, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [SEASON] * len(rows),
            "week": [r[0] for r in rows],
            "unit": ["pass_defense"] * len(rows),
            "team": [r[1] for r in rows],
            "weakness_z": [r[2] for r in rows],
            "rank": [16.5 + r[2] * 5 for r in rows],
        }
    )


def test_opponent_adjust_per_game_divides_by_the_multiplier() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["p", "p"],
            "season": [SEASON, SEASON],
            "week": [1, 2],
            "opponent": ["AAA", "DDD"],
            "yards": [100.0, 100.0],
        }
    )
    strength = _strength([(1, "AAA", -2.0), (2, "DDD", 2.0)])
    out = feat.opponent_adjust(stats, strength, stat_columns=["yards"], unit="pass_defense")
    values = out.sort("week").get_column("adj_yards").to_list()
    assert values[0] == pytest.approx(100.0 / (1 - 2 * feat.DEFAULT_K))
    assert values[1] == pytest.approx(100.0 / (1 + 2 * feat.DEFAULT_K))
    assert values[0] > 100.0 > values[1]


def test_opponent_adjust_uses_a_neutral_multiplier_for_an_unrated_opponent() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["p"],
            "season": [SEASON],
            "week": [1],
            "opponent": ["ZZZ"],
            "yards": [100.0],
        }
    )
    out = feat.opponent_adjust(
        stats, _strength([(1, "AAA", -2.0)]), stat_columns=["yards"], unit="pass_defense"
    )
    assert out.get_column("opponent_adjustment_factor").item() == 1.0
    assert out.get_column("adj_yards").item() == 100.0


def test_opponent_adjust_aggregates_only_earlier_weeks() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["p"] * 3,
            "season": [SEASON] * 3,
            "week": [1, 2, 3],
            "opponent": ["XXX"] * 3,
            "yards": [10.0, 20.0, 999.0],
        }
    )
    strength = _strength([(w, "XXX", 0.0) for w in (1, 2, 3)])
    targets = pl.DataFrame({"player_id": ["p"], "season": [SEASON], "week": [3]})
    out = feat.opponent_adjust(
        stats,
        strength,
        stat_columns=["yards"],
        targets=targets,
        unit="pass_defense",
        recency_lambda=1.0,
        shrinkage_games=0.0,
    )
    assert out.get_column("games_in_adjustment_sample").item() == 2.0
    assert out.get_column("adj_yards").item() == pytest.approx(15.0)


def test_opponent_adjust_weights_recent_games_more() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["p"] * 2,
            "season": [SEASON] * 2,
            "week": [1, 2],
            "opponent": ["XXX"] * 2,
            "yards": [0.0, 100.0],
        }
    )
    strength = _strength([(w, "XXX", 0.0) for w in (1, 2)])
    targets = pl.DataFrame({"player_id": ["p"], "season": [SEASON], "week": [3]})
    flat = (
        feat.opponent_adjust(
            stats,
            strength,
            stat_columns=["yards"],
            targets=targets,
            unit="pass_defense",
            recency_lambda=1.0,
            shrinkage_games=0.0,
        )
        .get_column("adj_yards")
        .item()
    )
    decayed = (
        feat.opponent_adjust(
            stats,
            strength,
            stat_columns=["yards"],
            targets=targets,
            unit="pass_defense",
            recency_lambda=0.5,
            shrinkage_games=0.0,
        )
        .get_column("adj_yards")
        .item()
    )
    assert flat == pytest.approx(50.0)
    # lambda**1 on week 2 and lambda**2 on week 1: (0.5*100 + 0.25*0) / 0.75.
    assert decayed == pytest.approx(100 * 0.5 / 0.75)
    assert decayed > flat


def test_opponent_adjust_shrinks_a_small_sample_toward_the_group_mean() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["hot", "cold", "cold"],
            "season": [SEASON] * 3,
            "week": [1, 1, 2],
            "opponent": ["XXX"] * 3,
            "yards": [200.0, 0.0, 0.0],
        }
    )
    strength = _strength([(w, "XXX", 0.0) for w in (1, 2)])
    targets = pl.DataFrame(
        {
            "player_id": ["hot", "cold"],
            "season": [SEASON] * 2,
            "week": [3, 3],
            "position_group": ["WR", "WR"],
        }
    )
    out = feat.opponent_adjust(
        stats,
        strength,
        stat_columns=["yards"],
        targets=targets,
        unit="pass_defense",
        group_column="position_group",
        recency_lambda=1.0,
        shrinkage_games=4.0,
    ).sort("player_id")
    hot = out.filter(pl.col("player_id") == "hot").get_column("adj_yards").item()
    # One game of 200 shrunk toward the pooled mean by c=4 games of prior.
    assert 0.0 < hot < 200.0
    unshrunk = (
        feat.opponent_adjust(
            stats,
            strength,
            stat_columns=["yards"],
            targets=targets,
            unit="pass_defense",
            group_column="position_group",
            recency_lambda=1.0,
            shrinkage_games=0.0,
        )
        .filter(pl.col("player_id") == "hot")
        .get_column("adj_yards")
        .item()
    )
    assert unshrunk == pytest.approx(200.0)
    assert hot < unshrunk


def test_opponent_adjust_rejects_a_missing_column() -> None:
    stats = pl.DataFrame({"player_id": ["p"], "season": [SEASON], "week": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        feat.opponent_adjust(stats, _strength([]), stat_columns=["yards"])


# --------------------------------------------------------------------------- #
# as_of_aggregate
# --------------------------------------------------------------------------- #


def test_as_of_aggregate_trailing_window_keeps_the_most_recent_games() -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["p"] * 5,
            "season": [SEASON] * 5,
            "week": [1, 2, 3, 4, 5],
            "yards": [1.0, 2.0, 4.0, 8.0, 16.0],
        }
    )
    targets = pl.DataFrame({"player_id": ["p"], "season": [SEASON], "week": [5]})
    everything = feat.as_of_aggregate(stats, targets, value_columns=["yards"])
    trailing = feat.as_of_aggregate(stats, targets, value_columns=["yards"], trailing_games=2)
    assert everything.get_column("yards").item() == 15.0
    assert everything.get_column("games").item() == 4.0
    assert trailing.get_column("yards").item() == 12.0  # weeks 3 and 4
    assert trailing.get_column("games").item() == 2.0


# --------------------------------------------------------------------------- #
# Starter identification
# --------------------------------------------------------------------------- #


def _team_weeks(weeks: tuple[int, ...] = (1, 2, 3, 4)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [SEASON] * len(TEAMS) * len(weeks),
            "week": [w for w in weeks for _ in TEAMS],
            "team": list(TEAMS) * len(weeks),
        }
    )


def test_identify_starters_prefers_the_depth_chart(league: pl.DataFrame) -> None:
    stats = feat.player_game_rates(feat.player_game_stats(league))
    starters = feat.identify_starters(
        _team_weeks(),
        depth_charts=synthetic_depth_charts(),
        rosters=synthetic_rosters(),
        stats=stats,
    )
    week2 = starters.filter((pl.col("week") == 2) & (pl.col("team") == "AAA"))
    slots = dict(zip(week2.get_column("slot"), week2.get_column("player_id"), strict=True))
    assert slots["QB1"] == "AAA-QB1"
    assert slots["RB1"] == "AAA-RB1"
    assert slots["TE1"] == "AAA-TE1"
    assert (week2.get_column("starter_source") == "depth_chart").all()


def test_a_special_teams_listing_does_not_steal_the_wr1_slot(league: pl.DataFrame) -> None:
    """WR3 is listed at depth 1 on special teams; WR1 must still be WR1."""
    stats = feat.player_game_rates(feat.player_game_stats(league))
    starters = feat.identify_starters(
        _team_weeks(), depth_charts=synthetic_depth_charts(), stats=stats
    )
    week1 = starters.filter((pl.col("week") == 1) & (pl.col("team") == "AAA"))
    slots = dict(zip(week1.get_column("slot"), week1.get_column("player_id"), strict=True))
    assert slots["WR1"] == "AAA-WR1"
    assert slots["WR2"] == "AAA-WR2"
    assert slots["WR3"] == "AAA-WR3"


def test_identify_starters_limits_each_group_to_its_slot_count(league: pl.DataFrame) -> None:
    starters = feat.identify_starters(
        _team_weeks(),
        depth_charts=synthetic_depth_charts(),
        rosters=synthetic_rosters(),
        stats=feat.player_game_rates(feat.player_game_stats(league)),
    )
    counts = (
        starters.group_by(["season", "week", "team", "position_group"])
        .agg(pl.len().alias("n"))
        .join(
            pl.DataFrame(
                {
                    "position_group": list(feat.SLOT_COUNTS),
                    "limit": pl.Series(list(feat.SLOT_COUNTS.values()), dtype=pl.UInt32),
                }
            ),
            on="position_group",
        )
    )
    assert (counts.get_column("n") <= counts.get_column("limit")).all()
    # There is never more than one QB1 for a team in a week.
    qb1 = starters.filter(pl.col("slot") == "QB1")
    assert qb1.select("season", "week", "team").is_duplicated().sum() == 0


def test_identify_starters_falls_back_to_pbp_usage(league: pl.DataFrame) -> None:
    """With no depth chart, no roster and no snaps, prior-week usage still finds the QB."""
    stats = feat.player_game_rates(feat.player_game_stats(league))
    starters = feat.identify_starters(_team_weeks(), stats=stats)
    week2 = starters.filter((pl.col("week") == 2) & (pl.col("team") == "AAA"))
    slots = dict(zip(week2.get_column("slot"), week2.get_column("player_id"), strict=True))
    assert slots["QB1"] == "AAA-QB1"
    assert slots["RB1"] == "AAA-RB1"
    assert (week2.get_column("starter_source") == "pbp_usage").all()
    # Week 1 has no prior usage, so pbp alone cannot name a starter.
    assert starters.filter(pl.col("week") == 1).height == 0


def test_identify_starters_with_no_sources_returns_the_empty_schema() -> None:
    out = feat.identify_starters(_team_weeks())
    assert out.height == 0
    assert "slot" in out.columns


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_team_availability_weights_a_quarterback_far_above_a_receiver() -> None:
    out = feat.team_availability(_team_weeks(), synthetic_injuries())
    qb_out = out.filter((pl.col("week") == 3) & (pl.col("team") == "DDD")).to_dicts()[0]
    wr_questionable = out.filter((pl.col("week") == 3) & (pl.col("team") == "BBB")).to_dicts()[0]
    assert qb_out["injury_n_out"] == 1.0
    assert qb_out["injury_qb_absence_cost"] > 0.5  # QB 1.00 weight x 0.85 dropoff
    assert wr_questionable["injury_n_questionable"] == 1.0
    assert qb_out["injury_expected_value_lost"] > 10 * wr_questionable["injury_expected_value_lost"]
    assert qb_out["injury_score"] < wr_questionable["injury_score"] < 0.0


def test_player_availability_is_zero_for_a_hard_out() -> None:
    out = feat.player_availability(synthetic_injuries())
    qb = out.filter(pl.col("position") == "QB")
    assert qb.get_column("availability_prob").item() == 0.0
    wr = out.filter(pl.col("position") == "WR")
    assert 0.0 < wr.get_column("availability_prob").to_list()[0] < 1.0


def test_team_availability_without_injuries_is_empty() -> None:
    assert feat.team_availability(_team_weeks(), None).height == 0


# --------------------------------------------------------------------------- #
# build_player
# --------------------------------------------------------------------------- #


def _build(weeks: tuple[int, ...] = (1, 2, 3, 4), **kwargs: object) -> pl.DataFrame:
    return feat.build_player(
        synthetic_pbp(weeks=weeks),
        synthetic_schedules(weeks),
        rosters=synthetic_rosters(weeks),
        depth_charts=synthetic_depth_charts(weeks),
        injuries=synthetic_injuries().filter(pl.col("week").is_in(list(weeks))),
        **kwargs,  # type: ignore[arg-type]
    )


def test_build_player_is_one_row_per_game_and_team() -> None:
    out = _build()
    assert out.height == len(SCHEDULE) * 2
    assert out.select("game_id", "team").is_duplicated().sum() == 0
    assert out.columns[:2] == ["game_id", "team"]


def test_every_feature_column_is_prefixed_and_documented() -> None:
    out = _build()
    features = [c for c in out.columns if c not in {"game_id", "team"}]
    assert all(c.startswith(feat.PREFIX) for c in features)
    assert features == list(feat.FEATURE_DOCS)
    assert all(feat.FEATURE_DOCS[c].strip() for c in features)


def test_build_player_has_no_market_columns() -> None:
    """ADR 0001: the blocked names must not survive, even when pbp carries them."""
    out = feat.build_player(
        synthetic_pbp(with_market_columns=True),
        synthetic_schedules(),
        rosters=synthetic_rosters(),
        depth_charts=synthetic_depth_charts(),
    )
    assert_no_market_columns(out)
    assert not (set(out.columns) & MARKET_COLUMNS)


def test_build_player_column_set_survives_missing_optional_inputs() -> None:
    lean = feat.build_player(synthetic_pbp(), synthetic_schedules())
    full = _build()
    assert list(lean.columns) == list(full.columns)
    assert lean.height == full.height
    # With no injury feed the availability block is unknown, not zero.
    assert lean.get_column("pl__injury_expected_value_lost").null_count() == lean.height
    assert full.get_column("pl__injury_expected_value_lost").null_count() == 0
    # ...and with no roster feed there is no bio.
    assert lean.get_column("pl__qb1_age_years").null_count() == lean.height


def test_week_one_rows_have_no_as_of_history() -> None:
    out = _build().filter(pl.col("pl__week") == 1)
    assert out.height == 4
    assert (out.get_column("pl__prior_games_in_season") == 0.0).all()
    assert out.get_column("pl__qb1_adj_epa_per_dropback").null_count() == out.height
    assert out.get_column("pl__ol_sack_rate_allowed").null_count() == out.height


def test_qb1_adjusted_epa_ranks_the_league_the_way_the_league_is_built() -> None:
    out = _build().filter(pl.col("pl__week") == 4)
    ranked = out.sort("pl__qb1_adj_epa_per_dropback", descending=True).get_column("team").to_list()
    assert ranked == ["AAA", "BBB", "CCC", "DDD"]
    assert out.filter(pl.col("team") == "AAA").get_column("pl__qb1_position_rank").item() == 1.0
    assert out.filter(pl.col("team") == "DDD").get_column("pl__qb1_position_rank").item() == 4.0


def test_opponent_adjustment_moves_production_toward_schedule_neutrality() -> None:
    """CCC's raw EPA is dragged down by facing AAA; the adjustment gives it back."""
    out = _build().filter(pl.col("pl__week") == 4)
    for team in TEAMS:
        row = out.filter(pl.col("team") == team).to_dicts()[0]
        assert row["pl__qb1_epa_per_dropback"] is not None
        assert row["pl__qb1_adj_epa_per_dropback"] is not None
    # Teams that played the strongest defence get the largest upward correction.
    corrections = {
        row["team"]: row["pl__qb1_adj_epa_per_dropback"] - row["pl__qb1_epa_per_dropback"]
        for row in out.to_dicts()
    }
    assert corrections["DDD"] > corrections["AAA"]


def test_starting_qb_out_shows_up_in_the_availability_block() -> None:
    out = _build()
    ddd_week3 = out.filter((pl.col("team") == "DDD") & (pl.col("pl__week") == 3)).to_dicts()[0]
    ddd_week2 = out.filter((pl.col("team") == "DDD") & (pl.col("pl__week") == 2)).to_dicts()[0]
    assert ddd_week3["pl__qb1_available"] == 0.0
    assert ddd_week3["pl__qb1_availability_prob"] == 0.0
    assert ddd_week3["pl__injury_qb_absence_cost"] > 0.5
    assert ddd_week3["pl__starters_out_count"] >= 1.0
    assert ddd_week2["pl__qb1_available"] == 1.0


def test_ol_and_dl_proxies_track_the_synthetic_sack_rates() -> None:
    out = _build().filter(pl.col("pl__week") == 4)
    assert out.get_column("pl__ol_sack_rate_allowed").null_count() == 0
    assert (out.get_column("pl__ol_sack_rate_allowed") > 0).all()
    assert (out.get_column("pl__ol_sack_rate_allowed_rank") >= 1).all()
    assert (out.get_column("pl__dl_sack_rate_generated") > 0).all()


def test_special_teams_returner_comes_from_earlier_weeks() -> None:
    out = _build().filter(pl.col("pl__week") == 4)
    assert out.get_column("pl__st_returner_player_id").null_count() == 0
    assert (out.get_column("pl__st_punt_return_yards_per_return") == 8.0).all()
    assert (out.get_column("pl__st_returns_in_sample") > 0).all()


def test_bio_block_reads_age_and_experience_from_the_roster() -> None:
    out = _build().filter(pl.col("pl__week") == 2).sort("team")
    ages = out.get_column("pl__qb1_age_years").to_list()
    assert all(20 < age < 35 for age in ages)
    assert out.get_column("pl__qb1_season_number").to_list() == [2.0, 3.0, 4.0, 5.0]
    assert out.get_column("pl__qb1_draft_pick").to_list() == [1.0, 11.0, 21.0, 31.0]
    # Documented as null until a contract source exists.
    assert out.get_column("pl__qb1_contract_years_remaining").null_count() == out.height
    assert out.get_column("pl__qb1_draft_round").null_count() == out.height


# --------------------------------------------------------------------------- #
# Delete-the-future
# --------------------------------------------------------------------------- #


def test_deleting_later_weeks_does_not_change_earlier_rows() -> None:
    """Leakage taxonomy item 2, applied to every column this builder emits."""
    full = _build(weeks=(1, 2, 3, 4))
    truncated = _build(weeks=(1, 2, 3))
    keys = ["game_id", "team"]
    left = full.join(truncated.select(keys), on=keys, how="semi").sort(keys)
    right = truncated.sort(keys)
    assert left.height == right.height == 12
    assert left.equals(right)


def test_deleting_later_weeks_leaves_the_starters_untouched() -> None:
    stats_full = feat.player_game_rates(feat.player_game_stats(synthetic_pbp()))
    stats_short = feat.player_game_rates(feat.player_game_stats(synthetic_pbp(weeks=(1, 2, 3))))
    keys = ["season", "week", "team", "position_group", "slot_index"]
    full = feat.identify_starters(_team_weeks((1, 2, 3)), stats=stats_full).sort(keys)
    short = feat.identify_starters(_team_weeks((1, 2, 3)), stats=stats_short).sort(keys)
    assert full.equals(short)


def test_perturbing_a_later_week_does_not_change_an_earlier_row() -> None:
    """Not just truncation: rewriting week 4 must leave weeks 1-3 identical."""
    base = synthetic_pbp()
    perturbed = base.with_columns(
        pl.when(pl.col("week") == 4)
        .then(pl.col("epa") * 100.0)
        .otherwise(pl.col("epa"))
        .alias("epa")
    )
    keys = ["game_id", "team"]
    kwargs: dict[str, object] = {
        "rosters": synthetic_rosters(),
        "depth_charts": synthetic_depth_charts(),
        "injuries": synthetic_injuries(),
    }
    left = feat.build_player(base, synthetic_schedules(), **kwargs)  # type: ignore[arg-type]
    right = feat.build_player(perturbed, synthetic_schedules(), **kwargs)  # type: ignore[arg-type]
    early = pl.col("pl__week") <= 3
    assert left.filter(early).sort(keys).equals(right.filter(early).sort(keys))


# --------------------------------------------------------------------------- #
# The committed real fixture, end to end
# --------------------------------------------------------------------------- #


def test_build_player_runs_on_the_committed_fixture(pbp_players: pl.DataFrame) -> None:
    schedules = pl.read_parquet(FIXTURES / "schedules_2024.parquet")
    out = feat.build_player(pbp_players, schedules, game_ids=["2024_01_BAL_KC"])
    assert out.height == 2
    assert out.get_column("team").to_list() == ["BAL", "KC"]
    assert list(out.columns) == list(feat.OUTPUT_COLUMNS)
    assert_no_market_columns(out)
    # Week 1 of a single-game slice: there is no prior week to summarise.
    assert (out.get_column("pl__week") == 1.0).all()
    assert (out.get_column("pl__prior_games_in_season") == 0.0).all()
    assert out.get_column("pl__qb1_adj_epa_per_dropback").null_count() == 2


def test_schedules_fixture_market_columns_never_reach_the_output(
    pbp_players: pl.DataFrame,
) -> None:
    schedules = pl.read_parquet(FIXTURES / "schedules_2024.parquet")
    out = feat.build_player(pbp_players, schedules, game_ids=["2024_01_BAL_KC"])
    assert not (set(out.columns) & MARKET_COLUMNS)


# --------------------------------------------------------------------------- #
# Live path
# --------------------------------------------------------------------------- #


def _snapshot(injuries: list[LiveInjury]) -> LiveSnapshot:
    return LiveSnapshot(
        source="test",
        fetched_at=datetime(2023, 9, 24, 17, 0, tzinfo=UTC),
        game_id=f"{SEASON}_03_BBB_CCC",
        season=SEASON,
        week=3,
        home_team="CCC",
        away_team="BBB",
        state="in",
        injuries=injuries,
    )


def test_roster_from_snapshot_returns_the_pregame_columns() -> None:
    pregame = _build()
    out = feat.roster_from_snapshot(_snapshot([]), pregame)
    assert list(out.columns) == list(feat.OUTPUT_COLUMNS)
    assert sorted(out.get_column("team")) == ["BBB", "CCC"]
    expected = pregame.filter(pl.col("game_id") == f"{SEASON}_03_BBB_CCC").sort("team")
    assert out.sort("team").equals(expected)


def test_roster_from_snapshot_flags_an_in_game_quarterback_loss() -> None:
    pregame = _build()
    snapshot = _snapshot(
        [LiveInjury(team="CCC", player_name="CCC QB", position="QB", status="Out")]
    )
    out = feat.roster_from_snapshot(snapshot, pregame).sort("team")
    rows = {row["team"]: row for row in out.to_dicts()}
    assert rows["CCC"]["pl__qb1_available"] == 0.0
    assert rows["BBB"]["pl__qb1_available"] == 1.0
    assert rows["CCC"]["pl__injury_expected_value_lost"] > 0.5


def test_roster_from_snapshot_never_lowers_the_pregame_cost() -> None:
    """The live list is a superset snapshot, so the two costs combine with max."""
    pregame = _build()
    before = (
        pregame.filter((pl.col("game_id") == f"{SEASON}_03_BBB_CCC") & (pl.col("team") == "BBB"))
        .get_column("pl__injury_expected_value_lost")
        .item()
    )
    snapshot = _snapshot(
        [LiveInjury(team="BBB", player_name="BBB LS", position="LS", status="Questionable")]
    )
    after = (
        feat.roster_from_snapshot(snapshot, pregame)
        .filter(pl.col("team") == "BBB")
        .get_column("pl__injury_expected_value_lost")
        .item()
    )
    assert after >= before


def test_roster_from_snapshot_without_a_pregame_table_returns_the_schema() -> None:
    out = feat.roster_from_snapshot(_snapshot([]))
    assert list(out.columns) == list(feat.OUTPUT_COLUMNS)
    assert out.height == 2
    features = [c for c in out.columns if c not in {"game_id", "team"}]
    assert out.select(features).null_count().sum_horizontal().item() == 2 * len(features)
