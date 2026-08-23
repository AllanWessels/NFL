"""Tests for ``nfl.features.team`` — the ``tm__`` pre-game team feature block.

Four things are load-bearing here and each gets its own section:

1. **Delete-the-future.** ``docs/DATA_MODEL.md`` leakage taxonomy item 2. Rebuild
   with later weeks deleted and every earlier row must be byte-identical.
2. **No market columns.** ADR 0001, enforced even when the caller hands the
   builder a play-by-play frame stuffed with spread and win-probability columns.
3. **Elo sanity.** Symmetry under mirroring, convergence away from the mean,
   a home-field bonus that points the right way, and a mean that never drifts.
4. **Rest days.** A hand-built fixture whose calendar can be checked on fingers:
   a Thursday game is a short week, a fortnight's gap is a bye.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl.data.market_blocklist import MARKET_COLUMNS, market_columns_in
from nfl.features import team as tm
from nfl.features.team import (
    FEATURE_DOCS,
    PREFIX,
    EloConfig,
    build_team_features,
    compute_elo,
    elo_expected,
    elo_update,
)

#: Weeks 1-4 of 2024, projected to :data:`nfl.features.team.PBP_COLUMNS`. It
#: lives outside ``fixtures/nflverse/`` on purpose: the replay feed treats that
#: directory as a play-by-play store and globs every ``pbp*.parquet`` in it, so
#: a second file covering the same game would double its play count.
#:
#: Regenerate with::
#:
#:     uv run python -c "
#:     import polars as pl
#:     from nfl.features.team import PBP_COLUMNS
#:     (pl.scan_parquet('data/raw/nflverse/pbp/season=2024/part-0.parquet')
#:        .filter(pl.col('week') <= 4).select(PBP_COLUMNS).collect()
#:        .write_parquet('tests/fixtures/features/pbp_2024_weeks1_4.parquet',
#:                       compression='zstd'))"
PBP_FIXTURE = Path(__file__).parent / "fixtures" / "features" / "pbp_2024_weeks1_4.parquet"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pbp_weeks() -> pl.DataFrame:
    """2024 weeks 1-4 play-by-play, projected to ``PBP_COLUMNS``."""
    return pl.read_parquet(PBP_FIXTURE)


@pytest.fixture(scope="module")
def features(schedules: pl.DataFrame, pbp_weeks: pl.DataFrame) -> pl.DataFrame:
    return build_team_features(schedules, pbp_weeks)


def make_schedule(games: list[dict[str, object]]) -> pl.DataFrame:
    """A minimal nflverse-shaped schedules frame from hand-written rows."""
    defaults: dict[str, object] = {
        "game_type": "REG",
        "gametime": "13:00",
        "location": "Home",
        "div_game": 0,
        "roof": "outdoors",
        "surface": "grass",
        "stadium_id": None,
        "stadium": None,
        "home_coach": None,
        "away_coach": None,
        "home_score": None,
        "away_score": None,
    }
    rows = [{**defaults, **g} for g in games]
    schema = {
        "game_id": pl.String,
        "season": pl.Int64,
        "game_type": pl.String,
        "week": pl.Int64,
        "gameday": pl.String,
        "gametime": pl.String,
        "home_team": pl.String,
        "away_team": pl.String,
        "home_score": pl.Float64,
        "away_score": pl.Float64,
        "location": pl.String,
        "div_game": pl.Int64,
        "roof": pl.String,
        "surface": pl.String,
        "stadium_id": pl.String,
        "stadium": pl.String,
        "home_coach": pl.String,
        "away_coach": pl.String,
    }
    return pl.DataFrame({name: [r.get(name) for r in rows] for name in schema}, schema=schema)


def round_robin(season: int, teams: list[str], weeks: int, margin: int = 7) -> pl.DataFrame:
    """A toy season where ``teams[0]`` beats everyone and ``teams[-1]`` loses to everyone."""
    start = date(season, 9, 8)
    games: list[dict[str, object]] = []
    for week in range(1, weeks + 1):
        day = start + timedelta(days=7 * (week - 1))
        rotated = teams[1:]
        rotated = rotated[week - 1 :] + rotated[: week - 1]
        pairs = [(teams[0], rotated[0])] + [
            (rotated[i], rotated[len(rotated) - i]) for i in range(1, len(rotated) // 2 + 1)
        ]
        for index, (home, away) in enumerate(pairs):
            if home == away:
                continue
            strong = teams.index(home) < teams.index(away)
            games.append(
                {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "week": week,
                    "gameday": day.isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_score": float(20 + margin) if strong else 20.0,
                    "away_score": 20.0 if strong else float(20 + margin),
                }
            )
            del index
    return make_schedule(games)


# --------------------------------------------------------------------------- #
# Shape and contract
# --------------------------------------------------------------------------- #


def test_one_row_per_game_and_team(schedules: pl.DataFrame, features: pl.DataFrame) -> None:
    assert features.height == 2 * schedules.height
    assert features.select("game_id", "team").n_unique() == features.height
    assert features.columns[:2] == ["game_id", "team"]


def test_every_feature_is_prefixed_and_documented(features: pl.DataFrame) -> None:
    emitted = [c for c in features.columns if c not in ("game_id", "team")]
    assert all(c.startswith(PREFIX) for c in emitted)
    assert set(emitted) == set(FEATURE_DOCS)
    assert all(FEATURE_DOCS[c].strip() for c in emitted)


def test_builder_runs_without_play_by_play(schedules: pl.DataFrame) -> None:
    """The schedule/rating/context blocks must not depend on pbp being available."""
    out = build_team_features(schedules, None)
    assert out.height == 2 * schedules.height
    assert out.get_column(f"{PREFIX}elo_pre").null_count() == 0
    assert out.get_column(f"{PREFIX}off_epa_per_play_season").null_count() == out.height


def test_missing_epa_column_is_a_loud_failure(
    schedules: pl.DataFrame, pbp_weeks: pl.DataFrame
) -> None:
    with pytest.raises(ValueError, match="epa"):
        build_team_features(schedules, pbp_weeks.drop("epa"))


# --------------------------------------------------------------------------- #
# ADR 0001 — no market columns
# --------------------------------------------------------------------------- #


def test_no_market_columns_in_output(features: pl.DataFrame) -> None:
    assert market_columns_in(features) == []
    assert set(features.columns).isdisjoint(MARKET_COLUMNS)


def test_market_columns_on_the_inputs_do_not_survive(
    schedules: pl.DataFrame, pbp_weeks: pl.DataFrame
) -> None:
    """Hand the builder contaminated inputs; the output must still be clean."""
    dirty_pbp = pbp_weeks.with_columns(
        pl.lit(-3.5).alias("spread_line"),
        pl.lit(47.5).alias("total_line"),
        pl.lit(0.62).alias("vegas_wp"),
        pl.lit(0.58).alias("wp"),
    )
    dirty_schedules = schedules.with_columns(
        pl.lit(-3.5).alias("spread_line"),
        pl.lit(-160).alias("home_moneyline"),
    )
    out = build_team_features(dirty_schedules, dirty_pbp)
    assert market_columns_in(out) == []


def test_pbp_projection_lists_no_market_column() -> None:
    assert market_columns_in(list(tm.PBP_COLUMNS)) == []


# --------------------------------------------------------------------------- #
# Leakage — delete the future
# --------------------------------------------------------------------------- #


def test_deleting_later_weeks_does_not_change_earlier_rows(
    schedules: pl.DataFrame, pbp_weeks: pl.DataFrame, features: pl.DataFrame
) -> None:
    """The leakage taxonomy's item 2, applied to the whole feature block."""
    cutoff = 3
    truncated = build_team_features(
        schedules.filter(pl.col("week") <= cutoff),
        pbp_weeks.filter(pl.col("week") <= cutoff),
    )
    early = features.filter(pl.col(f"{PREFIX}week") <= cutoff).sort("game_id", "team")
    assert truncated.sort("game_id", "team").equals(early)


def test_deleting_a_later_season_does_not_change_earlier_rows() -> None:
    """Cross-season version: Elo carry-over and the prior-season block included."""
    teams = ["AAA", "BBB", "CCC", "DDD"]
    both = pl.concat(
        [round_robin(2020, teams, weeks=6), round_robin(2021, teams, weeks=6)], how="vertical"
    )
    full = build_team_features(both)
    first_only = build_team_features(both.filter(pl.col("season") == 2020))
    early = full.filter(pl.col(f"{PREFIX}season") == 2020).sort("game_id", "team")
    assert first_only.sort("game_id", "team").equals(early)


def test_a_teams_own_game_never_enters_its_own_row(
    schedules: pl.DataFrame, pbp_weeks: pl.DataFrame
) -> None:
    """Week-1 rows have no prior games, so every season-to-date rate is null."""
    out = build_team_features(schedules, pbp_weeks)
    week_one = out.filter(pl.col(f"{PREFIX}week") == 1)
    for column in (
        "off_epa_per_play_season",
        "def_epa_per_play_season",
        "points_for_per_game_season",
        "win_pct_season",
    ):
        assert week_one.get_column(PREFIX + column).null_count() == week_one.height
    assert (week_one.get_column(f"{PREFIX}games_played") == 0).all()


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #


def test_elo_expected_is_symmetric_and_centred() -> None:
    assert elo_expected(1500.0, 1500.0) == pytest.approx(0.5)
    assert elo_expected(1600.0, 1500.0) + elo_expected(1500.0, 1600.0) == pytest.approx(1.0)
    assert elo_expected(1600.0, 1500.0) > 0.5


def test_elo_update_is_zero_sum() -> None:
    home, away = elo_update(1500.0, 1500.0, 24.0, 17.0)
    assert home + away == pytest.approx(3000.0)
    assert home > 1500.0 > away


def test_elo_home_advantage_favours_the_home_team() -> None:
    config = EloConfig(home_advantage=55.0)
    assert elo_expected(1500.0 + config.home_advantage, 1500.0) > 0.5
    # An identical pair of teams: the home side gains less by winning than the
    # away side would, because it was already expected to.
    home_win_home, _ = elo_update(1500.0, 1500.0, 20.0, 10.0, config=config)
    _, away_win_away = elo_update(1500.0, 1500.0, 10.0, 20.0, config=config)
    assert (home_win_home - 1500.0) < (away_win_away - 1500.0)


def test_elo_home_advantage_is_off_at_neutral_sites() -> None:
    config = EloConfig(home_advantage=55.0)
    neutral = elo_update(1500.0, 1500.0, 20.0, 10.0, config=config, neutral_site=True)
    hosted = elo_update(1500.0, 1500.0, 20.0, 10.0, config=config, neutral_site=False)
    assert neutral[0] > hosted[0]


def test_elo_win_prob_sums_to_one_across_a_game(features: pl.DataFrame) -> None:
    per_game = features.group_by("game_id").agg(
        pl.col(f"{PREFIX}elo_win_prob").sum().alias("total")
    )
    assert per_game.get_column("total").to_numpy() == pytest.approx(1.0)


def test_elo_win_prob_is_exactly_even_for_equal_ratings_at_a_neutral_site() -> None:
    schedule = make_schedule(
        [
            {
                "game_id": "2020_01_BBB_AAA",
                "season": 2020,
                "week": 1,
                "gameday": "2020-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 20.0,
                "away_score": 17.0,
                "location": "Neutral",
            }
        ]
    )
    out = build_team_features(schedule)
    assert out.get_column(f"{PREFIX}elo_win_prob").to_list() == [0.5, 0.5]
    assert out.get_column(f"{PREFIX}is_neutral_site").to_list() == [True, True]


def test_elo_converges_on_a_dominant_team() -> None:
    """A team that wins every game must climb, and the league mean must not move."""
    teams = ["AAA", "BBB", "CCC", "DDD"]
    schedule = pl.concat(
        [round_robin(season, teams, weeks=6) for season in (2020, 2021, 2022)], how="vertical"
    )
    games = tm._normalise_games(schedule)
    elo = compute_elo(games, EloConfig())
    final = games.join(elo, on="game_id").sort(["kickoff", "game_id"])
    ratings: dict[str, float] = {}
    for row in final.iter_rows(named=True):
        ratings[row["home_franchise"]] = row["home_elo_post"]
        ratings[row["away_franchise"]] = row["away_elo_post"]

    assert ratings["AAA"] == max(ratings.values())
    assert ratings["DDD"] == min(ratings.values())
    assert ratings["AAA"] > 1600.0
    assert ratings["DDD"] < 1400.0
    assert sum(ratings.values()) / len(ratings) == pytest.approx(1500.0)


def test_elo_is_symmetric_under_mirroring_the_schedule() -> None:
    """Swap every home and away team (and their scores) at neutral sites.

    Nothing about the games changes except the label on each side, so every
    rating must come out identical. This is the test that catches a home/away
    mix-up in the fold.
    """
    teams = ["AAA", "BBB", "CCC", "DDD"]
    schedule = round_robin(2020, teams, weeks=6).with_columns(pl.lit("Neutral").alias("location"))
    mirrored = schedule.with_columns(
        pl.col("away_team").alias("home_team"),
        pl.col("home_team").alias("away_team"),
        pl.col("away_score").alias("home_score"),
        pl.col("home_score").alias("away_score"),
    )

    def final_ratings(frame: pl.DataFrame) -> dict[str, float]:
        games = tm._normalise_games(frame)
        joined = games.join(compute_elo(games), on="game_id").sort(["kickoff", "game_id"])
        out: dict[str, float] = {}
        for row in joined.iter_rows(named=True):
            out[row["home_franchise"]] = row["home_elo_post"]
            out[row["away_franchise"]] = row["away_elo_post"]
        return out

    original, flipped = final_ratings(schedule), final_ratings(mirrored)
    assert original.keys() == flipped.keys()
    for name, value in original.items():
        assert flipped[name] == pytest.approx(value)


def test_elo_regresses_toward_the_mean_between_seasons() -> None:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    schedule = pl.concat(
        [round_robin(season, teams, weeks=6) for season in (2020, 2021)], how="vertical"
    )
    config = EloConfig(season_regression=0.5)
    out = build_team_features(schedule, elo=config)
    ordered = out.filter(pl.col(f"{PREFIX}franchise") == "AAA").sort(
        f"{PREFIX}season", f"{PREFIX}week"
    )
    end_of_2020 = ordered.filter(pl.col(f"{PREFIX}season") == 2020).get_column(f"{PREFIX}elo_pre")[
        -1
    ]
    start_of_2021 = ordered.filter(pl.col(f"{PREFIX}season") == 2021).get_column(
        f"{PREFIX}elo_pre"
    )[0]
    assert 1500.0 < start_of_2021 < end_of_2020


def test_franchise_relocation_carries_the_rating() -> None:
    """San Diego's rating must follow the Chargers to Los Angeles."""
    schedule = make_schedule(
        [
            {
                "game_id": "2016_01_AAA_SD",
                "season": 2016,
                "week": 1,
                "gameday": "2016-09-11",
                "home_team": "SD",
                "away_team": "AAA",
                "home_score": 40.0,
                "away_score": 3.0,
            },
            {
                "game_id": "2017_01_AAA_LAC",
                "season": 2017,
                "week": 1,
                "gameday": "2017-09-10",
                "home_team": "LAC",
                "away_team": "AAA",
                "home_score": 20.0,
                "away_score": 17.0,
            },
        ]
    )
    out = build_team_features(schedule)
    lac = out.filter(pl.col("team") == "LAC")
    assert lac.get_column(f"{PREFIX}franchise").to_list() == ["LAC"]
    assert lac.get_column(f"{PREFIX}elo_pre")[0] > 1500.0
    assert lac.get_column(f"{PREFIX}games_played")[0] == 0  # different season, fresh record


# --------------------------------------------------------------------------- #
# Schedule context — rest, travel, timezone
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rest_schedule() -> pl.DataFrame:
    """A calendar built so the rest arithmetic can be checked by hand.

    ``AAA``: Sun 13 Sep, then Thu 17 Sep (4 days, short week), then Sun 4 Oct
    (17 days, a bye), then Sun 11 Oct (7 days, normal).
    """
    return make_schedule(
        [
            {
                "game_id": "2020_01_BBB_AAA",
                "season": 2020,
                "week": 1,
                "gameday": "2020-09-13",
                "home_team": "AAA",
                "away_team": "BBB",
                "home_score": 20.0,
                "away_score": 17.0,
            },
            {
                "game_id": "2020_02_AAA_CCC",
                "season": 2020,
                "week": 2,
                "gameday": "2020-09-17",
                "gametime": "20:20",
                "home_team": "CCC",
                "away_team": "AAA",
                "home_score": 10.0,
                "away_score": 27.0,
            },
            {
                "game_id": "2020_04_DDD_AAA",
                "season": 2020,
                "week": 4,
                "gameday": "2020-10-04",
                "home_team": "AAA",
                "away_team": "DDD",
                "home_score": 14.0,
                "away_score": 21.0,
            },
            {
                "game_id": "2020_05_AAA_BBB",
                "season": 2020,
                "week": 5,
                "gameday": "2020-10-11",
                "home_team": "BBB",
                "away_team": "AAA",
                "home_score": 3.0,
                "away_score": 30.0,
            },
        ]
    )


def test_rest_days_short_week_and_bye(rest_schedule: pl.DataFrame) -> None:
    out = build_team_features(rest_schedule)
    aaa = out.filter(pl.col("team") == "AAA").sort(f"{PREFIX}week")
    assert aaa.get_column(f"{PREFIX}rest_days").to_list() == [None, 4, 17, 7]
    assert aaa.get_column(f"{PREFIX}short_week").to_list() == [None, True, False, False]
    assert aaa.get_column(f"{PREFIX}bye_last_week").to_list() == [None, False, True, False]


def test_rest_advantage_mirrors_across_the_matchup(rest_schedule: pl.DataFrame) -> None:
    out = build_team_features(rest_schedule).filter(pl.col("game_id") == "2020_05_AAA_BBB")
    advantage = out.sort("team").get_column(f"{PREFIX}rest_advantage").to_list()
    assert advantage[0] == pytest.approx(-advantage[1])


def test_rest_days_do_not_span_the_offseason() -> None:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    schedule = pl.concat(
        [round_robin(season, teams, weeks=3) for season in (2020, 2021)], how="vertical"
    )
    out = build_team_features(schedule)
    openers = out.filter(pl.col(f"{PREFIX}week") == 1)
    assert openers.get_column(f"{PREFIX}rest_days").null_count() == openers.height


def test_consecutive_road_games_counts_the_current_trip(rest_schedule: pl.DataFrame) -> None:
    out = build_team_features(rest_schedule)
    aaa = out.filter(pl.col("team") == "AAA").sort(f"{PREFIX}week")
    # home, away, home, away
    assert aaa.get_column(f"{PREFIX}consecutive_road_games").to_list() == [0, 1, 0, 1]


def test_travel_and_timezone_use_venue_coordinates(features: pl.DataFrame) -> None:
    """Sunday-in-London and the West-Coast trips have to show up somewhere."""
    travelled = features.filter(pl.col(f"{PREFIX}travel_miles") > 0)
    assert travelled.height > 0
    assert features.get_column(f"{PREFIX}travel_miles").max() > 2000.0
    shifts = features.get_column(f"{PREFIX}timezone_shift_hours").drop_nulls()
    assert shifts.min() < 0 and shifts.max() > 0


def test_kickoff_hour_and_primetime(features: pl.DataFrame) -> None:
    hours = features.get_column(f"{PREFIX}kickoff_hour_local").drop_nulls()
    assert hours.min() >= 0 and hours.max() <= 23
    primetime = features.filter(pl.col(f"{PREFIX}is_primetime"))
    assert primetime.height > 0
    assert primetime.get_column(f"{PREFIX}kickoff_hour_et").min() >= 18


# --------------------------------------------------------------------------- #
# Rolling form and ranks
# --------------------------------------------------------------------------- #


def test_season_to_date_rates_are_plausible(features: pl.DataFrame) -> None:
    later = features.filter(pl.col(f"{PREFIX}week") >= 3)
    epa = later.get_column(f"{PREFIX}off_epa_per_play_season").drop_nulls()
    assert epa.len() > 0
    assert epa.min() > -1.0 and epa.max() < 1.0
    for column in ("off_third_down_pct_season", "off_red_zone_td_pct_season", "fg_pct_season"):
        values = later.get_column(PREFIX + column).drop_nulls()
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_defensive_metrics_mirror_the_opponents_offence(
    schedules: pl.DataFrame, pbp_weeks: pl.DataFrame
) -> None:
    """``def_*`` is the opponent's ``off_*``; a whole league's worth must net out."""
    out = build_team_features(schedules, pbp_weeks).filter(pl.col(f"{PREFIX}week") == 5)
    off = out.get_column(f"{PREFIX}off_epa_per_play_season").drop_nulls()
    deff = out.get_column(f"{PREFIX}def_epa_per_play_season").drop_nulls()
    assert off.len() and deff.len()
    # League-wide, EPA gained equals EPA allowed to within sampling of who has
    # played whom, so the two means sit on top of each other.
    assert off.mean() == pytest.approx(deff.mean(), abs=0.06)


def test_ranks_are_dense_and_bounded(features: pl.DataFrame) -> None:
    ranked = features.filter(pl.col(f"{PREFIX}rank_overall").is_not_null())
    assert ranked.height > 0
    for column in features.columns:
        if not column.startswith(f"{PREFIX}rank_"):
            continue
        values = features.get_column(column).drop_nulls()
        assert values.min() >= 1
        assert values.max() <= 32


def test_ranks_span_the_whole_league_and_survive_bye_weeks(features: pl.DataFrame) -> None:
    """The rank scale must not shrink on a week where only some teams play."""
    week_column = pl.col(f"{PREFIX}week")
    full = features.filter(week_column == 4)
    assert full.height == 32  # every team plays in week 4 of 2024
    assert sorted(full.get_column(f"{PREFIX}rank_offense").to_list()) == list(range(1, 33))

    sizes = features.group_by(week_column).len().sort("len")
    bye_week = int(
        sizes.filter(pl.col("len") < 32).sort(week_column).get_column(f"{PREFIX}week")[0]
    )
    partial = features.filter(week_column == bye_week)
    assert partial.height < 32
    # Ranks are still drawn from the 32-team scale, not from the week's field.
    assert partial.get_column(f"{PREFIX}rank_offense").max() > partial.height


def test_rank_direction_matches_the_metric(features: pl.DataFrame) -> None:
    week = features.filter(
        (pl.col(f"{PREFIX}week") == 4) & pl.col(f"{PREFIX}off_epa_per_play_season").is_not_null()
    )
    best = week.sort(f"{PREFIX}rank_offense").head(1)
    best_epa = week.get_column(f"{PREFIX}off_epa_per_play_season").max()
    assert best.get_column(f"{PREFIX}off_epa_per_play_season")[0] == pytest.approx(best_epa)

    # Defence is ranked the other way: rank 1 allows the fewest EPA per play.
    defensive = features.filter(
        (pl.col(f"{PREFIX}week") == 4) & pl.col(f"{PREFIX}def_epa_per_play_season").is_not_null()
    )
    top_defence = defensive.sort(f"{PREFIX}rank_defense").head(1)
    assert top_defence.get_column(f"{PREFIX}def_epa_per_play_season")[0] == pytest.approx(
        defensive.get_column(f"{PREFIX}def_epa_per_play_season").min()
    )


def test_power_rating_follows_point_differential() -> None:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    out = build_team_features(round_robin(2020, teams, weeks=6, margin=14))
    late = out.filter(pl.col(f"{PREFIX}week") == 6)
    best = late.filter(pl.col("team") == "AAA").get_column(f"{PREFIX}power_rating")[0]
    worst = late.filter(pl.col("team") == "DDD").get_column(f"{PREFIX}power_rating")[0]
    assert best > 0 > worst
    diffs = late.get_column(f"{PREFIX}power_rating_diff").drop_nulls()
    assert diffs.sum() == pytest.approx(0.0, abs=1e-9)


def test_strength_of_schedule_is_opponent_elo(features: pl.DataFrame) -> None:
    sos = features.get_column(f"{PREFIX}sos_elo_played_season").drop_nulls()
    assert sos.len() > 0
    assert sos.min() > 1200.0 and sos.max() < 1800.0


def test_prior_season_block_is_populated_in_the_second_season() -> None:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    schedule = pl.concat(
        [round_robin(season, teams, weeks=6) for season in (2020, 2021)], how="vertical"
    )
    out = build_team_features(schedule)
    first = out.filter(pl.col(f"{PREFIX}season") == 2020)
    second = out.filter(pl.col(f"{PREFIX}season") == 2021)
    assert first.get_column(f"{PREFIX}win_pct_prior").null_count() == first.height
    assert second.get_column(f"{PREFIX}win_pct_prior").null_count() == 0
    aaa = second.filter(pl.col("team") == "AAA").get_column(f"{PREFIX}win_pct_prior")[0]
    ddd = second.filter(pl.col("team") == "DDD").get_column(f"{PREFIX}win_pct_prior")[0]
    assert aaa == pytest.approx(1.0)
    assert ddd == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Coach and weather joins
# --------------------------------------------------------------------------- #


def test_coach_records_join_and_tenure_counts(schedules_fixture: pl.DataFrame) -> None:
    """The 9-game coach fixture: Andy Reid's tenure and record are hand-checkable."""
    out = build_team_features(schedules_fixture)
    kc = out.filter(pl.col("team") == "KC").sort(f"{PREFIX}season", f"{PREFIX}week")
    assert kc.get_column(f"{PREFIX}coach_tenure_games").to_list() == list(range(kc.height))
    career = kc.get_column(f"{PREFIX}coach_career_win_pct").to_list()
    assert career[0] == pytest.approx(0.0)  # first game in the fixture's history
    assert all(v is not None for v in career)
    assert career[-1] > 0.5


def test_weather_columns_are_null_when_weather_is_absent(features: pl.DataFrame) -> None:
    for column in features.columns:
        if column.startswith(f"{PREFIX}wx_"):
            assert features.get_column(column).null_count() == features.height


def test_weather_join_when_present(schedules: pl.DataFrame) -> None:
    weather = pl.DataFrame(
        {
            "game_id": ["2024_01_BAL_KC"],
            "temperature_f": [71.5],
            "wind_speed_mph": [6.0],
            "is_indoor": [False],
        }
    )
    out = build_team_features(schedules, weather=weather)
    row = out.filter(pl.col("game_id") == "2024_01_BAL_KC")
    assert row.get_column(f"{PREFIX}wx_temperature_f").to_list() == [71.5, 71.5]
    assert row.get_column(f"{PREFIX}wx_is_indoor").to_list() == [False, False]
    other = out.filter(pl.col("game_id") != "2024_01_BAL_KC")
    assert other.get_column(f"{PREFIX}wx_temperature_f").null_count() == other.height
