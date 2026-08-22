"""Pydantic v2 schemas mirroring ``docs/DATA_MODEL.md``.

These are validation boundaries, not the storage format: data lands as Parquet
(see ``docs/ADR/0002-storage-parquet-duckdb.md``) and these models validate rows
crossing an ingest or inference boundary.

Two invariants are enforced structurally rather than by convention:

* :class:`Market` carries ``market_only: Literal[True]`` so market rows can never
  be mistaken for feature rows (``docs/ADR/0001-no-market-inputs.md``).
* Postgame-only fields are collected on :class:`GameResult` rather than being
  mixed into :class:`Game`, so a pregame frame cannot accidentally carry them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# --------------------------------------------------------------------------- #
# Enumerated literals
# --------------------------------------------------------------------------- #

SeasonType = Literal["PRE", "REG", "POST"]
TimeSlot = Literal["early", "late", "SNF", "MNF", "TNF", "intl", "other"]
RoofType = Literal["dome", "outdoors", "closed", "open"]
VenueRoof = Literal["dome", "retractable", "open"]
PositionGroup = Literal["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST"]
CoachRole = Literal["HC", "OC", "DC", "STC"]
RosterStatus = Literal["ACT", "INA", "IR", "PUP", "NFI", "SUSP", "PS"]
InjuryReportStatus = Literal["Out", "Doubtful", "Questionable", "IR", "DNP", "LP", "FP", ""]
InGameInjuryStatus = Literal["questionable", "doubtful", "out", "probable", "returned"]
DriveOutcome = Literal[
    "TD", "FG", "PUNT", "TO", "DOWNS", "END_HALF", "MISSED_FG", "SAFETY", "END_GAME"
]
PrecipitationType = Literal["none", "rain", "snow", "sleet", "mix"]
SentimentEntity = Literal["team", "player", "coach"]
SentimentScorer = Literal["rules", "ollama"]
SnapshotKind = Literal["pregame", "in_game"]
ModelKind = Literal["baseline", "gbdt", "mlp", "seq", "ensemble"]
ConfidenceMethod = Literal["ensemble_spread", "mc_dropout"]

# --------------------------------------------------------------------------- #
# Bases
# --------------------------------------------------------------------------- #


class NFLBase(BaseModel):
    """Base for every schema: strict-ish, immutable, no silent extra fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class Provenance(NFLBase):
    """Universal columns present on every persisted table.

    ``ingested_at`` is *our* wall clock and must never be used as a feature.
    """

    source: str = Field(description="Provenance tag, e.g. 'nflverse.pbp'.")
    ingested_at: datetime | None = Field(default=None, description="When we fetched the row.")
    schema_version: str = Field(default="1", description="Schema version for this table.")


# --------------------------------------------------------------------------- #
# 1. Game
# --------------------------------------------------------------------------- #


class Game(Provenance):
    """One game. Pregame-knowable fields only; results live on :class:`GameResult`."""

    game_id: str
    season: int = Field(ge=1999, le=2100)
    season_type: SeasonType = "REG"
    week: int = Field(ge=1, le=25)
    game_date: date
    kickoff_utc: datetime
    kickoff_local: datetime | None = None
    weekday: str | None = None
    time_slot: TimeSlot = "other"

    home_team: str = Field(min_length=2, max_length=4)
    away_team: str = Field(min_length=2, max_length=4)
    venue_id: str
    neutral_site: bool = False
    international: bool = False
    divisional_game: bool = False
    conference_game: bool = False
    rematch_week: int | None = None
    playoff_seed_home: int | None = None
    playoff_seed_away: int | None = None

    roof: RoofType = "outdoors"
    surface: str | None = None
    referee: str | None = None
    espn_game_id: str | None = None

    @model_validator(mode="after")
    def _teams_differ(self) -> Game:
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must differ")
        if self.rematch_week is not None and self.rematch_week >= self.week:
            raise ValueError("rematch_week must precede week (leakage guard)")
        return self


class GameResult(Provenance):
    """Postgame truth. Label construction and evaluation only -- never a feature."""

    game_id: str
    home_score_final: int = Field(ge=0)
    away_score_final: int = Field(ge=0)
    overtime: bool = False
    attendance: int | None = None
    game_length_minutes: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def home_win(self) -> bool | None:
        """The label. ``None`` on a tie."""
        if self.home_score_final == self.away_score_final:
            return None
        return self.home_score_final > self.away_score_final

    @computed_field  # type: ignore[prop-decorator]
    @property
    def result_margin(self) -> int:
        return self.home_score_final - self.away_score_final

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_points(self) -> int:
        return self.home_score_final + self.away_score_final


# --------------------------------------------------------------------------- #
# 2. GameState (per-play snapshot)
# --------------------------------------------------------------------------- #


class InGameInjury(NFLBase):
    """An injury announced during the game.

    ``announced_at_play_index`` is the leakage gate: only injuries announced
    strictly before play ``t`` may appear in the snapshot at ``t``.
    """

    player_id: str
    team: str
    position: str | None = None
    announced_at_play_index: int = Field(ge=0)
    status: InGameInjuryStatus
    body_part: str | None = None
    is_starter: bool = False
    pregame_snap_share: float = Field(default=0.0, ge=0.0, le=1.0)
    severity_weight: float = Field(default=0.0, ge=0.0)


class TeamGameCumulative(NFLBase):
    """Cumulative production for one team over plays strictly before ``t``.

    Every field here must be lagged: nflverse ships post-play columns, and using
    them unlagged is the single most likely leak in this project.
    """

    team: str

    total_plays: int = 0
    total_yards: int = 0
    pass_attempts: int = 0
    pass_completions: int = 0
    pass_yards: int = 0
    rush_attempts: int = 0
    rush_yards: int = 0
    yards_per_play: float = 0.0
    first_downs: int = 0
    third_down_attempts: int = 0
    third_down_conversions: int = 0
    third_down_rate: float = 0.0
    fourth_down_attempts: int = 0
    fourth_down_conversions: int = 0
    red_zone_trips: int = 0
    red_zone_touchdowns: int = 0
    sacks_taken: int = 0
    sacks_made: int = 0
    qb_hits_taken: int = 0
    tackles_for_loss_made: int = 0
    explosive_plays: int = 0
    three_and_outs: int = 0
    time_of_possession_seconds: int = 0

    epa_total: float = 0.0
    epa_offense: float = 0.0
    epa_defense: float = 0.0
    epa_pass: float = 0.0
    epa_rush: float = 0.0
    success_rate: float = 0.0
    # Third-party win-prob model output: diagnostics only, never a feature.
    wpa_total: float | None = None

    interceptions_thrown: int = 0
    fumbles: int = 0
    fumbles_lost: int = 0
    turnovers_committed: int = 0
    turnovers_forced: int = 0
    turnover_margin: int = 0
    defensive_touchdowns: int = 0
    takeaway_points_scored: int = 0
    turnovers_last_10_plays: int = 0

    penalties: int = 0
    penalty_yards: int = 0
    pre_snap_penalties: int = 0
    defensive_penalties_first_down: int = 0

    punts: int = 0
    punt_net_average: float = 0.0
    field_goals_attempted: int = 0
    field_goals_made: int = 0
    longest_field_goal_made: int = 0
    extra_points_attempted: int = 0
    extra_points_made: int = 0
    two_point_attempts: int = 0
    two_point_conversions: int = 0
    kick_return_yards: int = 0
    punt_return_yards: int = 0
    average_drive_start_yardline_100: float = 75.0
    blocked_kicks_against: int = 0
    special_teams_touchdowns: int = 0

    drives_completed: int = 0
    drive_outcomes: list[DriveOutcome] = Field(default_factory=list)
    scoring_drive_rate: float = 0.0
    points_per_drive: float = 0.0

    adj_pass_yards: float = 0.0
    adj_rush_yards: float = 0.0
    adj_epa_offense: float = 0.0
    production_vs_expectation: float = 0.0

    in_game_injuries: list[InGameInjury] = Field(default_factory=list)
    starting_qb_out: bool = False
    qb_change_occurred: bool = False
    injury_severity_index: float = 0.0
    key_players_out: int = 0
    snap_share_lost: float = Field(default=0.0, ge=0.0, le=1.0)


class MomentumWindow(NFLBase):
    """Trailing-window momentum measures, all ending strictly before play ``t``."""

    epa_last_5_plays_posteam: float = 0.0
    epa_last_10_plays_posteam: float = 0.0
    epa_last_5_plays_defteam: float = 0.0
    epa_last_10_plays_defteam: float = 0.0
    success_rate_last_10_posteam: float = 0.0
    points_last_5_minutes_home: int = 0
    points_last_5_minutes_away: int = 0
    yards_last_10_plays_posteam: int = 0
    score_swing_last_10_minutes: int = 0
    momentum_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    unanswered_points_streak: int = 0
    last_3_drive_outcomes_posteam: list[DriveOutcome] = Field(default_factory=list)
    last_3_drive_outcomes_defteam: list[DriveOutcome] = Field(default_factory=list)
    consecutive_scoring_drives_posteam: int = 0
    consecutive_empty_drives_posteam: int = 0


class SituationalFlags(NFLBase):
    """Cheap pre-snap booleans that carve the late-game state space."""

    garbage_time: bool = False
    must_score_posteam: bool = False
    victory_formation_likely: bool = False
    onside_kick_situation: bool = False
    fourth_down_go_situation: bool = False
    field_goal_to_tie: bool = False
    field_goal_to_win: bool = False
    td_to_tie: bool = False
    td_to_win: bool = False
    two_point_needed_after_td: bool = False
    receiving_second_half_kickoff: str | None = None
    end_of_half_possession: bool = False
    clock_kill_mode: bool = False
    desperation_index: float = 0.0


class GameState(Provenance):
    """The central entity: complete cumulative state immediately *before* play ``t``.

    Structural rule: nothing in this model may be derived from the outcome of
    play ``t`` or any later play. See the leakage taxonomy in
    ``docs/DATA_MODEL.md``.
    """

    game_id: str
    play_id: int
    play_index: int = Field(ge=0)
    snapshot_utc: datetime | None = None

    # Clock
    quarter: int = Field(ge=1, le=10)
    quarter_seconds_remaining: int = Field(ge=0, le=900)
    half_seconds_remaining: int = Field(ge=0, le=1800)
    game_seconds_remaining: int = Field(ge=0, le=3600)
    elapsed_game_seconds: int = Field(ge=0)
    is_two_minute_warning_half: bool = False
    is_final_two_minutes: bool = False
    is_overtime: bool = False
    clock_running: bool | None = None

    # Score (pre-snap)
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    lead_changes_so_far: int = 0
    times_tied_so_far: int = 0
    largest_lead_home: int = 0
    largest_lead_away: int = 0
    seconds_leading_home: int = 0
    possessions_to_tie: float = 0.0

    # Possession / field position
    posteam: str | None = None
    defteam: str | None = None
    posteam_is_home: bool | None = None
    down: int | None = Field(default=None, ge=1, le=4)
    ydstogo: int = Field(default=0, ge=0)
    yardline_100: int = Field(default=50, ge=1, le=99)
    goal_to_go: bool = False
    drive_number: int = Field(default=0, ge=0)
    drive_play_index: int = Field(default=0, ge=0)
    drive_start_yardline_100: int = Field(default=75, ge=1, le=99)
    drive_elapsed_seconds: int = 0
    drive_yards_so_far: int = 0
    series_number: int | None = None
    posteam_consecutive_possessions: int = 0
    play_type_expected: str | None = None
    shotgun: bool | None = None
    no_huddle: bool | None = None
    personnel_offense: str | None = None
    personnel_defense: str | None = None
    defenders_in_box: int | None = None

    # Timeouts
    home_timeouts_remaining: int = Field(default=3, ge=0, le=6)
    away_timeouts_remaining: int = Field(default=3, ge=0, le=6)
    timeout_leverage: float = 0.0
    can_kneel_out: bool = False
    two_minute_drill_active: bool = False

    # Cumulative per team
    home: TeamGameCumulative
    away: TeamGameCumulative

    momentum: MomentumWindow = Field(default_factory=MomentumWindow)
    flags: SituationalFlags = Field(default_factory=SituationalFlags)
    expected_remaining_possessions: float = 0.0

    @model_validator(mode="after")
    def _consistency(self) -> GameState:
        if self.posteam is not None and self.posteam == self.defteam:
            raise ValueError("posteam and defteam must differ")
        if self.game_seconds_remaining > 3600 and not self.is_overtime:
            raise ValueError("game_seconds_remaining exceeds regulation")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score_differential_home(self) -> int:
        return self.home_score - self.away_score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def abs_score_differential(self) -> int:
        return abs(self.home_score - self.away_score)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_one_score_game(self) -> bool:
        return self.abs_score_differential <= 8

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_two_score_game(self) -> bool:
        return 8 < self.abs_score_differential <= 16

    @computed_field  # type: ignore[prop-decorator]
    @property
    def in_red_zone(self) -> bool:
        return self.yardline_100 <= 20

    @computed_field  # type: ignore[prop-decorator]
    @property
    def in_fg_range(self) -> bool:
        return self.yardline_100 <= 38

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score_differential_posteam(self) -> int:
        if self.posteam_is_home is None:
            return 0
        diff = self.score_differential_home
        return diff if self.posteam_is_home else -diff

    @computed_field  # type: ignore[prop-decorator]
    @property
    def posteam_timeouts_remaining(self) -> int:
        if self.posteam_is_home is None:
            return 0
        return self.home_timeouts_remaining if self.posteam_is_home else self.away_timeouts_remaining

    @computed_field  # type: ignore[prop-decorator]
    @property
    def defteam_timeouts_remaining(self) -> int:
        if self.posteam_is_home is None:
            return 0
        return self.away_timeouts_remaining if self.posteam_is_home else self.home_timeouts_remaining


# --------------------------------------------------------------------------- #
# 3. Team
# --------------------------------------------------------------------------- #


class TeamStrength(Provenance):
    """Season-to-date team strength, valid only through ``as_of_date``.

    Every rank and rating must be the *pre-game* value. An off-by-one on
    ``elo_rating`` is the classic leak in this class of model.
    """

    team: str
    season: int
    week: int
    as_of_date: date

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    division_rank: int | None = None
    conference_rank: int | None = None

    elo_rating: float = 1500.0
    elo_rating_change_last_4: float = 0.0
    srs_rating: float = 0.0
    pythagorean_win_pct: float = Field(default=0.5, ge=0.0, le=1.0)

    overall_rank: int | None = Field(default=None, ge=1, le=32)
    offense_rank: int | None = Field(default=None, ge=1, le=32)
    defense_rank: int | None = Field(default=None, ge=1, le=32)
    special_teams_rank: int | None = Field(default=None, ge=1, le=32)
    pass_offense_rank: int | None = Field(default=None, ge=1, le=32)
    rush_offense_rank: int | None = Field(default=None, ge=1, le=32)
    pass_defense_rank: int | None = Field(default=None, ge=1, le=32)
    rush_defense_rank: int | None = Field(default=None, ge=1, le=32)
    red_zone_offense_rank: int | None = Field(default=None, ge=1, le=32)
    red_zone_defense_rank: int | None = Field(default=None, ge=1, le=32)
    third_down_offense_rank: int | None = Field(default=None, ge=1, le=32)
    third_down_defense_rank: int | None = Field(default=None, ge=1, le=32)

    epa_per_play_offense: float = 0.0
    epa_per_play_defense: float = 0.0
    epa_per_pass_offense: float = 0.0
    epa_per_rush_offense: float = 0.0
    epa_per_pass_defense: float = 0.0
    epa_per_rush_defense: float = 0.0
    success_rate_offense: float = 0.0
    success_rate_defense: float = 0.0
    dvoa_like_offense: float = 0.0
    dvoa_like_defense: float = 0.0

    points_per_game_for: float = 0.0
    points_per_game_against: float = 0.0
    yards_per_game_for: float = 0.0
    yards_per_game_against: float = 0.0
    turnover_margin_per_game: float = 0.0
    sack_rate_offense: float = 0.0
    sack_rate_defense: float = 0.0
    penalty_yards_per_game: float = 0.0

    pace_seconds_per_play: float = 0.0
    pace_plays_per_game: float = 0.0
    pass_rate_over_expected: float = 0.0
    neutral_pass_rate: float = 0.0
    fourth_down_aggressiveness: float = 0.0

    strength_of_schedule_played: float = 0.0
    strength_of_schedule_remaining: float = 0.0

    rest_days: int = 7
    opponent_rest_days: int = 7
    rest_advantage: int = 0
    on_bye_last_week: bool = False
    short_week: bool = False
    consecutive_road_games: int = 0
    travel_miles_this_trip: float = 0.0
    travel_miles_last_14_days: float = 0.0
    timezone_shift_hours: int = 0
    crossed_international_date_line: bool = False
    altitude_change_feet: float = 0.0

    home_win_pct_3yr: float = Field(default=0.5, ge=0.0, le=1.0)
    away_win_pct_3yr: float = Field(default=0.5, ge=0.0, le=1.0)
    record_in_one_score_games: float = Field(default=0.5, ge=0.0, le=1.0)

    roster_continuity_index: float = Field(default=0.0, ge=0.0, le=1.0)
    starters_available_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    qb_is_starter: bool = True
    qb_games_started_season: int = 0


# --------------------------------------------------------------------------- #
# 4. Coach
# --------------------------------------------------------------------------- #


class Coach(Provenance):
    """Coach record and tendencies, as-of the game (current game excluded)."""

    coach_id: str
    name: str
    role: CoachRole = "HC"
    team: str
    season: int
    age: int | None = None
    first_season_as_hc: int | None = None
    seasons_with_team: int = 0
    total_seasons_hc: int = 0

    career_wins: int = 0
    career_losses: int = 0
    career_ties: int = 0
    career_win_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    win_pct_last_16_games: float = Field(default=0.0, ge=0.0, le=1.0)
    playoff_wins: int = 0
    playoff_losses: int = 0
    wins_vs_opponent_hc: int = 0
    losses_vs_opponent_hc: int = 0
    record_in_one_score_games: float = Field(default=0.5, ge=0.0, le=1.0)
    record_after_bye: float = Field(default=0.5, ge=0.0, le=1.0)

    fourth_down_go_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    two_point_attempt_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_usage_efficiency: float = 0.0
    challenge_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_seconds_per_offensive_play: float = 0.0
    clock_management_index: float = 0.0

    is_interim: bool = False
    hired_midseason_week: int | None = None
    on_hot_seat: bool = False


# --------------------------------------------------------------------------- #
# 5. Player
# --------------------------------------------------------------------------- #


class PlayerContract(NFLBase):
    """Contract state. Motivation proxy; hand-maintained or OverTheCap import."""

    contract_years_remaining: float | None = None
    contract_apy_usd: float | None = None
    cap_hit_usd: float | None = None
    is_franchise_tagged: bool = False
    in_contract_year: bool = False


class PlayerSeasonStats(NFLBase):
    """Raw as-of-week production. Nulls where the position makes a stat moot."""

    snap_share_offense: float = Field(default=0.0, ge=0.0, le=1.0)
    snap_share_defense: float = Field(default=0.0, ge=0.0, le=1.0)
    snap_share_st: float = Field(default=0.0, ge=0.0, le=1.0)

    targets: int = 0
    receptions: int = 0
    receiving_yards: int = 0
    receiving_tds: int = 0
    target_share: float = Field(default=0.0, ge=0.0, le=1.0)
    air_yards_share: float = Field(default=0.0, ge=0.0, le=1.0)
    yards_per_route_run: float | None = None

    rush_attempts: int = 0
    rushing_yards: int = 0
    rushing_tds: int = 0
    yards_after_contact_per_att: float | None = None

    pass_attempts: int = 0
    completions: int = 0
    passing_yards: int = 0
    passing_tds: int = 0
    interceptions: int = 0
    sacks_taken: int = 0
    time_to_throw: float | None = None
    cpoe: float | None = None

    epa_per_play: float = 0.0
    epa_per_dropback: float | None = None

    pressures: int = 0
    sacks_made: float = 0.0
    qb_hits: int = 0
    tackles: int = 0
    tackles_for_loss: int = 0
    pass_breakups: int = 0
    interceptions_made: int = 0
    forced_fumbles: int = 0
    coverage_targets: int | None = None
    coverage_yards_allowed: float | None = None
    passer_rating_allowed: float | None = None

    fg_attempts: int = 0
    fg_made: int = 0
    fg_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    fg_made_50plus: int = 0
    punt_net_average: float | None = None


class PlayerAdjustedStats(NFLBase):
    """Opponent-adjusted production. Formula: ``docs/DATA_MODEL.md`` section 6.

    Opponent ranks used in the adjustment are as-of the week each game was
    played -- using end-of-season ranks is a leak.
    """

    adj_receiving_yards_per_game: float = 0.0
    adj_rushing_yards_per_game: float = 0.0
    adj_passing_yards_per_game: float = 0.0
    adj_epa_per_play: float = 0.0
    adj_pressure_rate: float = 0.0
    adj_coverage_grade: float = 0.0
    opponent_adjustment_factor_mean: float = 1.0
    games_in_adjustment_sample: int = Field(default=0, ge=0)


class Player(Provenance):
    """One player at one (season, week). Biography is static; stats are as-of."""

    player_id: str
    name: str
    team: str
    position: str
    position_group: PositionGroup
    season: int
    week: int
    as_of_date: date

    depth_chart_rank: int | None = None
    jersey_number: int | None = None
    birth_date: date | None = None
    age_at_game: float | None = None
    height_inches: int | None = None
    weight_lbs: int | None = None
    college: str | None = None
    draft_year: int | None = None
    draft_round: int | None = None
    draft_pick: int | None = None
    undrafted: bool = False
    season_number: int = Field(default=1, ge=1)
    games_played_career: int = 0
    games_started_season: int = 0
    is_rookie: bool = False

    roster_status: RosterStatus = "ACT"
    weeks_on_ir_season: int = 0
    is_suspended: bool = False
    new_to_team_this_season: bool = False
    traded_midseason: bool = False

    contract: PlayerContract = Field(default_factory=PlayerContract)
    stats: PlayerSeasonStats = Field(default_factory=PlayerSeasonStats)
    adjusted: PlayerAdjustedStats = Field(default_factory=PlayerAdjustedStats)

    position_rank_overall: int | None = None
    position_rank_on_team: int | None = None
    value_over_replacement: float = 0.0


# --------------------------------------------------------------------------- #
# 7. Venue
# --------------------------------------------------------------------------- #


class Venue(Provenance):
    """Static venue reference, committed to the repo."""

    venue_id: str
    name: str
    city: str
    state: str | None = None
    country: str = "USA"
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    altitude_feet: float = 0.0
    timezone: str = "America/New_York"
    roof_type: VenueRoof = "open"
    surface_type: str = "grass"
    capacity: int | None = None
    opened_year: int | None = None
    home_teams: list[str] = Field(default_factory=list)
    orientation_degrees: float | None = None
    historical_home_win_pct: float = Field(default=0.5, ge=0.0, le=1.0)
    historical_points_per_game: float = 0.0
    noise_index: float | None = None


# --------------------------------------------------------------------------- #
# 8. Weather
# --------------------------------------------------------------------------- #


class Weather(Provenance):
    """Weather at a venue at an instant.

    ``is_forecast`` matters: training on perfect hindsight observations that the
    live path will not have is a subtle train/serve skew.
    """

    game_id: str
    observed_at_utc: datetime
    is_forecast: bool = True

    temperature_f: float | None = None
    feels_like_f: float | None = None
    humidity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    wind_speed_mph: float | None = Field(default=None, ge=0.0)
    wind_gust_mph: float | None = Field(default=None, ge=0.0)
    wind_direction_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    wind_relative_to_field_deg: float | None = None
    precipitation_mm: float | None = Field(default=None, ge=0.0)
    precipitation_type: PrecipitationType = "none"
    snow_depth_cm: float | None = Field(default=None, ge=0.0)
    cloud_cover_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    visibility_m: float | None = None
    pressure_hpa: float | None = None

    is_indoor: bool = False
    roof_closed: bool | None = None
    severe_weather_flag: bool = False
    kicking_difficulty_index: float = 0.0
    passing_difficulty_index: float = 0.0


# --------------------------------------------------------------------------- #
# 9. Injury
# --------------------------------------------------------------------------- #


class InjuryReport(Provenance):
    """Official weekly injury report row.

    Only rows with ``report_date`` before kickoff are usable as features.
    ``did_play`` / ``snap_share_if_played`` are postgame truth used to calibrate
    ``expected_availability`` offline -- never features.
    """

    game_id: str
    season: int
    week: int
    team: str
    player_id: str
    report_date: date
    report_status: InjuryReportStatus = ""
    practice_status: str | None = None
    practice_participation_wed: str | None = None
    practice_participation_thu: str | None = None
    practice_participation_fri: str | None = None
    primary_injury: str | None = None
    secondary_injury: str | None = None
    is_new_injury: bool = False
    weeks_on_report: int = 0
    expected_availability: float = Field(default=1.0, ge=0.0, le=1.0)

    did_play: bool | None = None
    snap_share_if_played: float | None = None

    position_importance_weight: float = 1.0
    player_value_weight: float = 0.0
    weighted_absence_cost: float = 0.0


# --------------------------------------------------------------------------- #
# 10. Sentiment
# --------------------------------------------------------------------------- #


class Sentiment(Provenance):
    """Rule-based (default) sentiment over injury reports and news headlines.

    No cloud LLMs. ``window_end`` is the leakage gate.
    """

    entity_type: SentimentEntity
    entity_id: str
    game_id: str
    window_start: datetime
    window_end: datetime

    n_documents: int = 0
    n_injury_report_items: int = 0
    injury_sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    news_sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    composite_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_volatility: float = Field(default=0.0, ge=0.0)
    negative_headline_count: int = 0
    positive_headline_count: int = 0
    suspension_mentioned: bool = False
    holdout_mentioned: bool = False
    locker_room_discord_flag: bool = False
    coach_job_security_flag: bool = False

    scorer: SentimentScorer = "rules"
    scorer_version: str = "0.1.0"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _window_ordered(self) -> Sentiment:
        if self.window_end < self.window_start:
            raise ValueError("window_end must not precede window_start")
        return self


# --------------------------------------------------------------------------- #
# 11. RosterSnapshot
# --------------------------------------------------------------------------- #


class RosterSnapshot(Provenance):
    """Who is available to a team at a moment. ``as_of`` gates every field."""

    game_id: str
    team: str
    snapshot_kind: SnapshotKind = "pregame"
    as_of: datetime
    as_of_play_index: int | None = None

    active_player_ids: list[str] = Field(default_factory=list)
    inactive_player_ids: list[str] = Field(default_factory=list)
    starting_qb_id: str | None = None
    backup_qb_id: str | None = None
    starters_by_position: dict[str, str] = Field(default_factory=dict)

    ol_starters_available: int = 5
    ol_continuity_index: float = Field(default=1.0, ge=0.0, le=1.0)
    skill_position_availability: float = Field(default=1.0, ge=0.0, le=1.0)
    defensive_availability: float = Field(default=1.0, ge=0.0, le=1.0)
    total_snap_share_available: float = Field(default=1.0, ge=0.0, le=1.0)
    total_value_available: float = 0.0
    value_lost_to_injury: float = 0.0

    kicker_id: str | None = None
    punter_id: str | None = None
    emergency_qb_designated: bool = False
    practice_squad_elevations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _in_game_needs_play_index(self) -> RosterSnapshot:
        if self.snapshot_kind == "in_game" and self.as_of_play_index is None:
            raise ValueError("in_game roster snapshots require as_of_play_index (leakage gate)")
        return self


# --------------------------------------------------------------------------- #
# 12. Market -- EVALUATION ONLY
# --------------------------------------------------------------------------- #


class Market(Provenance):
    """Betting / Kalshi data. **NEVER a model input** -- see ADR 0001.

    ``market_only`` is a ``Literal[True]``: it cannot be set to ``False``, so a
    market row is structurally distinguishable from a feature row at runtime.
    """

    market_only: Literal[True] = True

    game_id: str
    observed_at_utc: datetime
    book: str = "consensus"

    spread_home: float | None = None
    spread_open_home: float | None = None
    spread_close_home: float | None = None
    total_open: float | None = None
    total_close: float | None = None
    moneyline_home: int | None = None
    moneyline_away: int | None = None
    implied_prob_home: float | None = Field(default=None, ge=0.0, le=1.0)
    implied_prob_away: float | None = Field(default=None, ge=0.0, le=1.0)

    kalshi_ticker: str | None = None
    kalshi_yes_price: float | None = Field(default=None, ge=0.0, le=1.0)
    kalshi_no_price: float | None = Field(default=None, ge=0.0, le=1.0)
    kalshi_volume: int | None = None
    kalshi_open_interest: int | None = None
    bid_ask_spread: float | None = None

    line_movement_since_open: float | None = None
    reverse_line_movement: bool | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_collapsed(self) -> bool:
        """True when the market has moved past the 85/15 trade gate."""
        p = self.implied_prob_home
        if p is None:
            return False
        return p < 0.15 or p > 0.85


#: Join/provenance keys that legitimately appear on non-market tables too.
_GENERIC_FIELD_NAMES: frozenset[str] = frozenset(Provenance.model_fields) | {
    "game_id",
    "observed_at_utc",
}

#: Field names that must never appear in a training feature frame (ADR 0001).
#: Used by the guard test in ``tests/test_schemas.py`` and, from Phase 4, by the
#: feature-frame assembler.
MARKET_FIELD_NAMES: frozenset[str] = (
    frozenset(Market.model_fields) | {"market_collapsed"}
) - _GENERIC_FIELD_NAMES


# --------------------------------------------------------------------------- #
# 13. Prediction ledger
# --------------------------------------------------------------------------- #


class PredictionLedgerRow(Provenance):
    """One prediction, plus its later resolution. Not a model input."""

    prediction_id: str
    created_at_utc: datetime
    game_id: str
    snapshot_play_index: int | None = None
    snapshot_game_seconds_remaining: int | None = None

    feature_hash: str
    feature_set_version: str
    model_version: str
    model_kind: ModelKind

    p_home_win: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_method: ConfidenceMethod = "ensemble_spread"
    ensemble_member_probs: list[float] | None = None
    is_live: bool = False

    resolved_home_win: bool | None = None
    resolved_at_utc: datetime | None = None
    market_implied_prob_home_at_snapshot: float | None = None
    edge_vs_market: float | None = None
    brier: float | None = None
    log_loss: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def p_away_win(self) -> float:
        return 1.0 - self.p_home_win


__all__ = [
    "MARKET_FIELD_NAMES",
    "Coach",
    "Game",
    "GameResult",
    "GameState",
    "InGameInjury",
    "InjuryReport",
    "Market",
    "MomentumWindow",
    "NFLBase",
    "Player",
    "PlayerAdjustedStats",
    "PlayerContract",
    "PlayerSeasonStats",
    "PredictionLedgerRow",
    "Provenance",
    "RosterSnapshot",
    "Sentiment",
    "SituationalFlags",
    "TeamGameCumulative",
    "TeamStrength",
    "Venue",
    "Weather",
]
