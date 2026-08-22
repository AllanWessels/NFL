"""Every schema in ``nfl.data.schemas`` instantiates and enforces its invariants."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel, ValidationError

from nfl.data import schemas
from nfl.data.schemas import (
    MARKET_FIELD_NAMES,
    Coach,
    Game,
    GameResult,
    GameState,
    InGameInjury,
    InjuryReport,
    Market,
    MomentumWindow,
    Player,
    PredictionLedgerRow,
    RosterSnapshot,
    Sentiment,
    SituationalFlags,
    TeamGameCumulative,
    TeamStrength,
    Venue,
    Weather,
)

KICKOFF = datetime(2025, 1, 5, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_game(**overrides: object) -> Game:
    kwargs: dict[str, object] = {
        "source": "nflverse.sched",
        "game_id": "2024_18_KC_DEN",
        "season": 2024,
        "week": 18,
        "game_date": date(2025, 1, 5),
        "kickoff_utc": KICKOFF,
        "home_team": "DEN",
        "away_team": "KC",
        "venue_id": "DEN00",
    }
    kwargs.update(overrides)
    return Game(**kwargs)  # type: ignore[arg-type]


def make_cumulative(team: str = "DEN") -> TeamGameCumulative:
    return TeamGameCumulative(team=team)


def make_game_state(**overrides: object) -> GameState:
    kwargs: dict[str, object] = {
        "source": "nflverse.pbp",
        "game_id": "2024_18_KC_DEN",
        "play_id": 3421,
        "play_index": 128,
        "quarter": 4,
        "quarter_seconds_remaining": 300,
        "half_seconds_remaining": 300,
        "game_seconds_remaining": 300,
        "elapsed_game_seconds": 3300,
        "home_score": 20,
        "away_score": 17,
        "posteam": "KC",
        "defteam": "DEN",
        "posteam_is_home": False,
        "down": 2,
        "ydstogo": 7,
        "yardline_100": 35,
        "home": make_cumulative("DEN"),
        "away": make_cumulative("KC"),
    }
    kwargs.update(overrides)
    return GameState(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Instantiation coverage
# --------------------------------------------------------------------------- #


def _concrete_models() -> list[type[BaseModel]]:
    return [
        obj
        for name in schemas.__all__
        if isinstance(obj := getattr(schemas, name), type) and issubclass(obj, BaseModel)
    ]


def test_all_exported_models_are_pydantic() -> None:
    exported = _concrete_models()
    assert len(exported) >= 20
    for model in exported:
        assert model.model_fields is not None


def test_game_instantiates_and_round_trips() -> None:
    game = make_game()
    assert game.home_team == "DEN"
    assert Game.model_validate(game.model_dump()) == game


def test_game_rejects_same_team_both_sides() -> None:
    with pytest.raises(ValidationError):
        make_game(away_team="DEN")


def test_game_rejects_rematch_week_in_the_future() -> None:
    """A 'previous meeting' after the current week would be future information."""
    with pytest.raises(ValidationError):
        make_game(rematch_week=19)


def test_game_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_game(spread_home=-3.5)


def test_game_result_computes_label() -> None:
    result = GameResult(
        source="nflverse.sched",
        game_id="2024_18_KC_DEN",
        home_score_final=27,
        away_score_final=24,
    )
    assert result.home_win is True
    assert result.result_margin == 3
    assert result.total_points == 51


def test_game_result_tie_has_no_label() -> None:
    result = GameResult(
        source="nflverse.sched",
        game_id="2024_18_KC_DEN",
        home_score_final=24,
        away_score_final=24,
    )
    assert result.home_win is None


def test_game_schema_has_no_result_fields() -> None:
    """Postgame truth lives on GameResult so a pregame frame cannot carry it."""
    postgame = {"home_score_final", "away_score_final", "home_win", "result_margin", "overtime"}
    assert postgame.isdisjoint(Game.model_fields)


# --------------------------------------------------------------------------- #
# GameState
# --------------------------------------------------------------------------- #


def test_game_state_instantiates() -> None:
    state = make_game_state()
    assert state.game_seconds_remaining == 300
    assert state.home.team == "DEN"


def test_game_state_derived_score_fields() -> None:
    state = make_game_state()
    assert state.score_differential_home == 3
    assert state.abs_score_differential == 3
    assert state.is_one_score_game is True
    assert state.is_two_score_game is False
    # KC has possession and is the away team.
    assert state.score_differential_posteam == -3


def test_game_state_derived_field_position_flags() -> None:
    assert make_game_state(yardline_100=12).in_red_zone is True
    assert make_game_state(yardline_100=35).in_red_zone is False
    assert make_game_state(yardline_100=35).in_fg_range is True
    assert make_game_state(yardline_100=60).in_fg_range is False


def test_game_state_timeout_perspective() -> None:
    state = make_game_state(home_timeouts_remaining=1, away_timeouts_remaining=3)
    assert state.posteam_timeouts_remaining == 3
    assert state.defteam_timeouts_remaining == 1


def test_game_state_rejects_posteam_equal_to_defteam() -> None:
    with pytest.raises(ValidationError):
        make_game_state(defteam="KC")


def test_game_state_rejects_impossible_clock() -> None:
    with pytest.raises(ValidationError):
        make_game_state(game_seconds_remaining=-1)


def test_game_state_round_trips() -> None:
    """Computed fields are dropped on dump-then-validate; stored fields survive."""
    state = make_game_state()
    payload = state.model_dump(exclude=set(GameState.model_computed_fields))
    assert GameState.model_validate(payload) == state


def test_game_state_nests_momentum_and_flags_by_default() -> None:
    state = make_game_state()
    assert isinstance(state.momentum, MomentumWindow)
    assert isinstance(state.flags, SituationalFlags)
    assert state.momentum.momentum_score == 0.0
    assert state.flags.garbage_time is False


def test_in_game_injury_carries_announcement_gate() -> None:
    injury = InGameInjury(
        player_id="00-0033873",
        team="KC",
        position="QB",
        announced_at_play_index=97,
        status="questionable",
    )
    assert injury.announced_at_play_index < make_game_state().play_index


def test_team_game_cumulative_defaults_are_zeroed() -> None:
    cumulative = make_cumulative()
    assert cumulative.total_yards == 0
    assert cumulative.drive_outcomes == []
    assert cumulative.in_game_injuries == []


def test_team_game_cumulative_accepts_drive_outcomes() -> None:
    cumulative = TeamGameCumulative(team="KC", drive_outcomes=["TD", "PUNT", "FG"])
    assert cumulative.drive_outcomes == ["TD", "PUNT", "FG"]


def test_team_game_cumulative_rejects_unknown_drive_outcome() -> None:
    with pytest.raises(ValidationError):
        TeamGameCumulative(team="KC", drive_outcomes=["TOUCHDOWN"])


# --------------------------------------------------------------------------- #
# Team / Coach / Player
# --------------------------------------------------------------------------- #


def test_team_strength_instantiates() -> None:
    team = TeamStrength(
        source="derived",
        team="DEN",
        season=2024,
        week=18,
        as_of_date=date(2025, 1, 4),
        elo_rating=1552.0,
    )
    assert team.elo_rating == 1552.0
    assert team.rest_days == 7


def test_team_strength_rejects_out_of_range_rank() -> None:
    with pytest.raises(ValidationError):
        TeamStrength(
            source="derived",
            team="DEN",
            season=2024,
            week=18,
            as_of_date=date(2025, 1, 4),
            offense_rank=33,
        )


def test_coach_instantiates() -> None:
    coach = Coach(
        source="static",
        coach_id="payton_sean",
        name="Sean Payton",
        team="DEN",
        season=2024,
        fourth_down_go_rate=0.31,
    )
    assert coach.role == "HC"
    assert coach.fourth_down_go_rate == 0.31


def test_player_instantiates_with_nested_defaults() -> None:
    player = Player(
        source="nflverse.rosters",
        player_id="00-0036355",
        name="Bo Nix",
        team="DEN",
        position="QB",
        position_group="QB",
        season=2024,
        week=18,
        as_of_date=date(2025, 1, 4),
        season_number=1,
        is_rookie=True,
    )
    assert player.contract.contract_years_remaining is None
    assert player.stats.pass_attempts == 0
    assert player.adjusted.opponent_adjustment_factor_mean == 1.0


def test_player_rejects_bad_position_group() -> None:
    with pytest.raises(ValidationError):
        Player(
            source="nflverse.rosters",
            player_id="x",
            name="x",
            team="DEN",
            position="QB",
            position_group="QUARTERBACK",
            season=2024,
            week=1,
            as_of_date=date(2024, 9, 1),
        )


# --------------------------------------------------------------------------- #
# Venue / Weather / Injury / Sentiment / Roster
# --------------------------------------------------------------------------- #


def test_venue_instantiates() -> None:
    venue = Venue(
        source="static",
        venue_id="DEN00",
        name="Empower Field at Mile High",
        city="Denver",
        state="CO",
        latitude=39.7439,
        longitude=-105.0201,
        altitude_feet=5280.0,
        timezone="America/Denver",
    )
    assert venue.roof_type == "open"
    assert venue.altitude_feet == 5280.0


def test_venue_rejects_impossible_coordinates() -> None:
    with pytest.raises(ValidationError):
        Venue(
            source="static",
            venue_id="X",
            name="X",
            city="X",
            latitude=100.0,
            longitude=0.0,
        )


def test_weather_instantiates_and_defaults_to_forecast() -> None:
    weather = Weather(
        source="openmeteo",
        game_id="2024_18_KC_DEN",
        observed_at_utc=KICKOFF,
        temperature_f=31.0,
        wind_speed_mph=14.0,
    )
    assert weather.is_forecast is True
    assert weather.precipitation_type == "none"


def test_injury_report_instantiates() -> None:
    report = InjuryReport(
        source="nflverse.inj",
        game_id="2024_18_KC_DEN",
        season=2024,
        week=18,
        team="KC",
        player_id="00-0033873",
        report_date=date(2025, 1, 3),
        report_status="Questionable",
        expected_availability=0.6,
    )
    assert report.did_play is None
    assert report.report_date < KICKOFF.date()


def test_sentiment_instantiates_and_defaults_to_rules_scorer() -> None:
    sentiment = Sentiment(
        source="news",
        entity_type="team",
        entity_id="KC",
        game_id="2024_18_KC_DEN",
        window_start=datetime(2024, 12, 29, tzinfo=UTC),
        window_end=datetime(2025, 1, 5, 12, 0, tzinfo=UTC),
        n_documents=14,
        news_sentiment_score=-0.2,
    )
    assert sentiment.scorer == "rules"
    assert sentiment.window_end <= KICKOFF


def test_sentiment_rejects_inverted_window() -> None:
    with pytest.raises(ValidationError):
        Sentiment(
            source="news",
            entity_type="team",
            entity_id="KC",
            game_id="2024_18_KC_DEN",
            window_start=datetime(2025, 1, 6, tzinfo=UTC),
            window_end=datetime(2025, 1, 5, tzinfo=UTC),
        )


def test_roster_snapshot_pregame_instantiates() -> None:
    snapshot = RosterSnapshot(
        source="nflverse.depth",
        game_id="2024_18_KC_DEN",
        team="DEN",
        as_of=datetime(2025, 1, 5, 16, 0, tzinfo=UTC),
        starting_qb_id="00-0036355",
        active_player_ids=["00-0036355"],
    )
    assert snapshot.snapshot_kind == "pregame"
    assert snapshot.ol_starters_available == 5


def test_roster_snapshot_in_game_requires_play_index() -> None:
    """The in-game snapshot's leakage gate is mandatory."""
    with pytest.raises(ValidationError):
        RosterSnapshot(
            source="espn.live",
            game_id="2024_18_KC_DEN",
            team="DEN",
            snapshot_kind="in_game",
            as_of=KICKOFF,
        )


def test_roster_snapshot_in_game_with_play_index_is_valid() -> None:
    snapshot = RosterSnapshot(
        source="espn.live",
        game_id="2024_18_KC_DEN",
        team="DEN",
        snapshot_kind="in_game",
        as_of=KICKOFF,
        as_of_play_index=128,
    )
    assert snapshot.as_of_play_index == 128


# --------------------------------------------------------------------------- #
# Market guard (ADR 0001)
# --------------------------------------------------------------------------- #


def make_market(**overrides: object) -> Market:
    kwargs: dict[str, object] = {
        "source": "market",
        "game_id": "2024_18_KC_DEN",
        "observed_at_utc": KICKOFF,
        "book": "kalshi",
        "implied_prob_home": 0.62,
    }
    kwargs.update(overrides)
    return Market(**kwargs)  # type: ignore[arg-type]


def test_market_instantiates_and_is_flagged() -> None:
    market = make_market()
    assert market.market_only is True


def test_market_only_cannot_be_false() -> None:
    """The ADR 0001 guard is structural, not a convention."""
    with pytest.raises(ValidationError):
        make_market(market_only=False)


def test_market_collapsed_gate() -> None:
    assert make_market(implied_prob_home=0.62).market_collapsed is False
    assert make_market(implied_prob_home=0.91).market_collapsed is True
    assert make_market(implied_prob_home=0.08).market_collapsed is True
    assert make_market(implied_prob_home=None).market_collapsed is False


def test_no_feature_schema_shares_a_market_field_name() -> None:
    """No non-market schema may carry a market-sourced column."""
    for name in schemas.__all__:
        model = getattr(schemas, name)
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            continue
        if model is Market:
            continue
        overlap = MARKET_FIELD_NAMES & set(model.model_fields)
        assert not overlap, f"{name} carries market-sourced fields: {sorted(overlap)}"


def test_market_field_names_are_populated() -> None:
    assert "spread_close_home" in MARKET_FIELD_NAMES
    assert "kalshi_yes_price" in MARKET_FIELD_NAMES
    assert "market_collapsed" in MARKET_FIELD_NAMES


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def test_prediction_ledger_row_instantiates() -> None:
    row = PredictionLedgerRow(
        source="nfl.inference",
        prediction_id="0f0f0f",
        created_at_utc=KICKOFF,
        game_id="2024_18_KC_DEN",
        snapshot_play_index=128,
        snapshot_game_seconds_remaining=300,
        feature_hash="deadbeef",
        feature_set_version="fs-0.1.0",
        model_version="baseline-0.1.0",
        model_kind="baseline",
        p_home_win=0.71,
        confidence=0.55,
    )
    assert row.p_away_win == pytest.approx(0.29)
    assert row.resolved_home_win is None


def test_prediction_ledger_row_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValidationError):
        PredictionLedgerRow(
            source="nfl.inference",
            prediction_id="x",
            created_at_utc=KICKOFF,
            game_id="g",
            feature_hash="h",
            feature_set_version="v",
            model_version="v",
            model_kind="gbdt",
            p_home_win=1.2,
            confidence=0.5,
        )


def test_models_are_frozen() -> None:
    game = make_game()
    with pytest.raises(ValidationError):
        game.season = 2025  # type: ignore[misc]
