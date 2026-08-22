# Data Model

Exhaustive attribute specification for every entity in the NFL win-probability
system. **This is deliberately over-inclusive** — attributes are cheap to
specify and expensive to think of later. Pruning happens in Phase 4 once feature
importance and leakage tests tell us what actually matters.

## Reading this document

Every attribute table has these columns:

| Column | Meaning |
|---|---|
| **Attribute** | Field name as it appears in `nfl/data/schemas.py` and in Parquet. |
| **Type** | Python/pydantic type. `Decimal`-free: floats throughout, ints where exact. |
| **Source** | Where the value comes from. See source keys below. |
| **Avail** | When the value is knowable: `pregame` (fixed before kickoff), `in-game` (evolves play by play), `postgame` (only after the game ends — **label or evaluation use only**), `static` (reference data). |
| **Leakage risk** | How this field could smuggle future information into a play-`t` feature vector, and the mitigation. |

### Source keys

| Key | Description |
|---|---|
| `nflverse.pbp` | nflreadpy `load_pbp()` — play-by-play, 1999+, rich for 2010+. |
| `nflverse.sched` | `load_schedules()` — games, kickoff times, results, venue. |
| `nflverse.rosters` | `load_rosters()` / `load_rosters_weekly()`. |
| `nflverse.players` | `load_players()` — biographical, draft, physical. |
| `nflverse.inj` | `load_injuries()` — weekly official injury reports. |
| `nflverse.snaps` | `load_snap_counts()`. |
| `nflverse.depth` | `load_depth_charts()`. |
| `nflverse.pfr` | `load_pfr_*()` advanced stats passthrough. |
| `nflverse.part` | `load_participation()` — personnel/formation (2016+). |
| `nflverse.ftn` | `load_ftn_charting()` — charting data (2022+). |
| `espn.live` | ESPN unofficial JSON scoreboard/summary, behind the `LiveFeed` protocol. |
| `openmeteo` | Open-Meteo historical/forecast weather API by lat/lon + hour. |
| `derived` | Computed by `nfl/features` from other stored fields. |
| `static` | Hand-maintained reference table committed to the repo. |
| `news` | Headline scraper + official injury report text. |
| `market` | Odds/Kalshi feed. **Evaluation only — see ADR 0001.** |

### Universal columns

Every Parquet table carries these, omitted from the per-entity tables below:

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `source` | `str` | derived | static | None. Provenance tag. |
| `ingested_at` | `datetime` | derived | static | None — this is *our* clock, never a game clock. Must never be used as a feature. |
| `schema_version` | `str` | derived | static | None. |

---

## 1. Game

One row per game. Identity and fixed pregame context. Result fields are
`postgame` and exist only to build labels and evaluate.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `game_id` | `str` | nflverse.sched | pregame | None. Format `YYYY_WW_AWAY_HOME`. |
| `season` | `int` | nflverse.sched | pregame | None. |
| `season_type` | `Literal["REG","POST","PRE"]` | nflverse.sched | pregame | None. |
| `week` | `int` | nflverse.sched | pregame | None. |
| `game_date` | `date` | nflverse.sched | pregame | None. |
| `kickoff_utc` | `datetime` | nflverse.sched | pregame | None. Anchor for weather lookup. |
| `kickoff_local` | `datetime` | derived | pregame | None. |
| `weekday` | `str` | derived | pregame | None. Thu/Sun/Mon/Sat games differ in rest. |
| `time_slot` | `Literal["early","late","SNF","MNF","TNF","intl","other"]` | derived | pregame | None. |
| `home_team` | `str` | nflverse.sched | pregame | None. Canonical 2–3 letter code. |
| `away_team` | `str` | nflverse.sched | pregame | None. |
| `venue_id` | `str` | nflverse.sched | pregame | None. |
| `neutral_site` | `bool` | nflverse.sched | pregame | None. |
| `international` | `bool` | derived | pregame | None. |
| `divisional_game` | `bool` | derived | pregame | None. |
| `conference_game` | `bool` | derived | pregame | None. |
| `rematch_week` | `int \| None` | derived | pregame | Week of the prior meeting *this season*, if any. Must be `< week`. |
| `playoff_seed_home` | `int \| None` | nflverse.sched | pregame | None (postseason only). |
| `playoff_seed_away` | `int \| None` | nflverse.sched | pregame | None. |
| `roof` | `Literal["dome","outdoors","closed","open"]` | nflverse.sched | pregame | None. |
| `surface` | `str` | nflverse.sched | pregame | None. |
| `referee` | `str \| None` | nflverse.sched | pregame | None. Crew penalty tendencies are a legitimate pregame feature. |
| `overtime` | `bool` | nflverse.sched | postgame | **High.** Never a feature; used to segment evaluation. |
| `home_score_final` | `int` | nflverse.sched | postgame | **High.** Label construction only. |
| `away_score_final` | `int` | nflverse.sched | postgame | **High.** Label construction only. |
| `home_win` | `bool \| None` | derived | postgame | **This is the label.** `None` for ties. |
| `result_margin` | `int` | derived | postgame | **High.** Evaluation/segmentation only. |
| `total_points` | `int` | derived | postgame | **High.** Evaluation only. |
| `attendance` | `int \| None` | nflverse.sched | postgame | Low but postgame-recorded; excluded from features. |
| `game_length_minutes` | `int \| None` | derived | postgame | **High.** Excluded. |
| `espn_game_id` | `str \| None` | espn.live | pregame | None. Join key for live feed. |

---

## 2. GameState — per-play snapshot

**The central entity.** One row per play, carrying the complete cumulative
in-game state *as of the moment immediately before that play is run*. This is
what the model sees at inference time.

Every field here is `in-game` and every field carries the same structural
leakage hazard: it must be computed from plays `< t` (or from the pre-snap
situation of play `t`), never from the outcome of play `t` or later. The
per-field notes call out where that is subtle. A mandatory test asserts that
perturbing plays `> t` leaves the row at `t` unchanged.

### 2.1 Identity and clock

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `game_id` | `str` | nflverse.pbp | in-game | None. |
| `play_id` | `int` | nflverse.pbp | in-game | None. |
| `play_index` | `int` | derived | in-game | Monotone counter from 0. None. |
| `snapshot_utc` | `datetime` | derived | in-game | Wall-clock of the snap. Used for live/historical alignment, never as a feature directly. |
| `quarter` | `int` | nflverse.pbp | in-game | None (1–4, 5+ = OT). |
| `quarter_seconds_remaining` | `int` | nflverse.pbp | in-game | None. Pre-snap. |
| `half_seconds_remaining` | `int` | nflverse.pbp | in-game | None. |
| `game_seconds_remaining` | `int` | nflverse.pbp | in-game | None. **Primary time feature.** |
| `elapsed_game_seconds` | `int` | derived | in-game | None. |
| `is_two_minute_warning_half` | `bool` | derived | in-game | None. |
| `is_final_two_minutes` | `bool` | derived | in-game | None. |
| `is_overtime` | `bool` | derived | in-game | None — knowable at the snap. Distinct from `Game.overtime`, which is postgame. |
| `clock_running` | `bool \| None` | espn.live | in-game | None. |

### 2.2 Score

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `home_score` | `int` | nflverse.pbp | in-game | Must be the **pre-snap** score. nflverse `total_home_score` is post-play — must be lagged by one play. Common leakage bug. |
| `away_score` | `int` | nflverse.pbp | in-game | Same lag requirement. |
| `score_differential_home` | `int` | derived | in-game | `home_score - away_score`, pre-snap. |
| `score_differential_posteam` | `int` | derived | in-game | From the possessing team's perspective. |
| `abs_score_differential` | `int` | derived | in-game | None. |
| `possessions_to_tie` | `float` | derived | in-game | Score deficit converted to scoring drives needed (8/7/3-point ladder). None. |
| `is_one_score_game` | `bool` | derived | in-game | `abs diff <= 8`. None. |
| `is_two_score_game` | `bool` | derived | in-game | None. |
| `lead_changes_so_far` | `int` | derived | in-game | Count over plays `< t`. None if strictly lagged. |
| `times_tied_so_far` | `int` | derived | in-game | None if strictly lagged. |
| `largest_lead_home` | `int` | derived | in-game | Max over plays `< t`. None if lagged. |
| `largest_lead_away` | `int` | derived | in-game | None if lagged. |
| `seconds_leading_home` | `int` | derived | in-game | Cumulative. None if lagged. |

### 2.3 Possession and field position

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `posteam` | `str` | nflverse.pbp | in-game | None. Pre-snap. |
| `defteam` | `str` | nflverse.pbp | in-game | None. |
| `posteam_is_home` | `bool` | derived | in-game | None. |
| `down` | `int \| None` | nflverse.pbp | in-game | None. `None` on kickoffs/XPs. |
| `ydstogo` | `int` | nflverse.pbp | in-game | None. |
| `yardline_100` | `int` | nflverse.pbp | in-game | Distance to opponent end zone, 1–99. None. |
| `in_red_zone` | `bool` | derived | in-game | `yardline_100 <= 20`. None. |
| `in_fg_range` | `bool` | derived | in-game | `yardline_100 <= 38` (adjust for kicker/weather later). None. |
| `goal_to_go` | `bool` | nflverse.pbp | in-game | None. |
| `drive_number` | `int` | nflverse.pbp | in-game | None. |
| `drive_play_index` | `int` | derived | in-game | Play number within the current drive. None. |
| `drive_start_yardline_100` | `int` | derived | in-game | None. |
| `drive_elapsed_seconds` | `int` | derived | in-game | None. |
| `drive_yards_so_far` | `int` | derived | in-game | **Within-drive**, plays `< t` only. |
| `series_number` | `int` | nflverse.pbp | in-game | None. |
| `posteam_consecutive_possessions` | `int` | derived | in-game | Rare (onside/turnover chains). None. |
| `play_type_expected` | `str \| None` | derived | in-game | Pre-snap tendency estimate, not the actual play. **Must not read the play's outcome.** |
| `shotgun` | `bool \| None` | nflverse.pbp | in-game | Pre-snap formation. Safe. |
| `no_huddle` | `bool \| None` | nflverse.pbp | in-game | Safe. |
| `personnel_offense` | `str \| None` | nflverse.part | in-game | Pre-snap. Safe (2016+ only). |
| `personnel_defense` | `str \| None` | nflverse.part | in-game | Pre-snap. Safe. |
| `defenders_in_box` | `int \| None` | nflverse.part | in-game | Pre-snap. Safe. |

### 2.4 Timeouts and clock management

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `home_timeouts_remaining` | `int` | nflverse.pbp | in-game | Pre-snap value required; nflverse column is pre-play. Verify. |
| `away_timeouts_remaining` | `int` | nflverse.pbp | in-game | Same. |
| `posteam_timeouts_remaining` | `int` | derived | in-game | None. |
| `defteam_timeouts_remaining` | `int` | derived | in-game | None. |
| `timeout_leverage` | `float` | derived | in-game | Combined timeouts × clock situation. Late-game critical. None. |
| `can_kneel_out` | `bool` | derived | in-game | Whether the leader can run out the clock given downs/timeouts. Pure function of pre-snap state. None. |
| `two_minute_drill_active` | `bool` | derived | in-game | None. |

### 2.5 Cumulative team production (both teams, plays `< t`)

Each of the following exists twice, suffixed `_home` and `_away`. Listed once.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `total_plays` | `int` | derived | in-game | Lag required. |
| `total_yards` | `int` | derived | in-game | Lag required. |
| `pass_attempts` | `int` | derived | in-game | Lag. |
| `pass_completions` | `int` | derived | in-game | Lag. |
| `pass_yards` | `int` | derived | in-game | Lag. |
| `rush_attempts` | `int` | derived | in-game | Lag. |
| `rush_yards` | `int` | derived | in-game | Lag. |
| `yards_per_play` | `float` | derived | in-game | Lag. |
| `first_downs` | `int` | derived | in-game | Lag. |
| `third_down_attempts` | `int` | derived | in-game | Lag. |
| `third_down_conversions` | `int` | derived | in-game | Lag. |
| `third_down_rate` | `float` | derived | in-game | Lag. Smoothed to avoid 0/0. |
| `fourth_down_attempts` | `int` | derived | in-game | Lag. |
| `fourth_down_conversions` | `int` | derived | in-game | Lag. |
| `red_zone_trips` | `int` | derived | in-game | Lag. |
| `red_zone_touchdowns` | `int` | derived | in-game | Lag. |
| `sacks_taken` | `int` | derived | in-game | Lag. |
| `sacks_made` | `int` | derived | in-game | Lag. |
| `qb_hits_taken` | `int` | derived | in-game | Lag. |
| `tackles_for_loss_made` | `int` | derived | in-game | Lag. |
| `explosive_plays` | `int` | derived | in-game | Runs ≥12 or passes ≥20 yds. Lag. |
| `three_and_outs` | `int` | derived | in-game | Lag. |
| `time_of_possession_seconds` | `int` | derived | in-game | Lag. |
| `epa_total` | `float` | nflverse.pbp | in-game | Cumulative EPA over plays `< t`. **nflverse EPA is computed with a model that used the full season — acceptable for in-game aggregation but must not be recomputed with future data.** Lag. |
| `epa_offense` | `float` | nflverse.pbp | in-game | Lag. |
| `epa_defense` | `float` | nflverse.pbp | in-game | Lag. |
| `epa_pass` | `float` | nflverse.pbp | in-game | Lag. |
| `epa_rush` | `float` | nflverse.pbp | in-game | Lag. |
| `success_rate` | `float` | derived | in-game | Fraction of plays with EPA > 0. Lag. |
| `wpa_total` | `float` | nflverse.pbp | in-game | **Medium.** nflverse WPA comes from *their* win-prob model. Store for diagnostics; do **not** use as a feature — it is a third-party model output, which ADR 0001's spirit forbids as a label/target. |

### 2.6 Turnovers, penalties, special teams (both teams, plays `< t`)

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `interceptions_thrown` | `int` | derived | in-game | Lag. |
| `fumbles` | `int` | derived | in-game | Lag. |
| `fumbles_lost` | `int` | derived | in-game | Lag. |
| `turnovers_committed` | `int` | derived | in-game | Lag. |
| `turnovers_forced` | `int` | derived | in-game | Lag. |
| `turnover_margin` | `int` | derived | in-game | Lag. |
| `defensive_touchdowns` | `int` | derived | in-game | Lag. |
| `takeaway_points_scored` | `int` | derived | in-game | Points off turnovers. Lag. |
| `turnovers_last_10_plays` | `int` | derived | in-game | Momentum window. Lag. |
| `penalties` | `int` | derived | in-game | Lag. |
| `penalty_yards` | `int` | derived | in-game | Lag. |
| `pre_snap_penalties` | `int` | derived | in-game | Discipline proxy. Lag. |
| `defensive_penalties_first_down` | `int` | derived | in-game | Lag. |
| `punts` | `int` | derived | in-game | Lag. |
| `punt_net_average` | `float` | derived | in-game | Lag. |
| `field_goals_attempted` | `int` | derived | in-game | Lag. |
| `field_goals_made` | `int` | derived | in-game | Lag. |
| `longest_field_goal_made` | `int` | derived | in-game | Lag. |
| `extra_points_attempted` | `int` | derived | in-game | Lag. |
| `extra_points_made` | `int` | derived | in-game | Lag. |
| `two_point_attempts` | `int` | derived | in-game | Lag. |
| `two_point_conversions` | `int` | derived | in-game | Lag. |
| `kick_return_yards` | `int` | derived | in-game | Lag. |
| `punt_return_yards` | `int` | derived | in-game | Lag. |
| `average_drive_start_yardline_100` | `float` | derived | in-game | Field-position battle. Lag. |
| `blocked_kicks_against` | `int` | derived | in-game | Lag. |
| `special_teams_touchdowns` | `int` | derived | in-game | Lag. |

### 2.7 Drive history

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `drives_completed_home` | `int` | derived | in-game | Lag. |
| `drives_completed_away` | `int` | derived | in-game | Lag. |
| `drive_outcomes_home` | `list[str]` | derived | in-game | Ordered outcomes (`TD`,`FG`,`PUNT`,`TO`,`DOWNS`,`END_HALF`,`MISSED_FG`,`SAFETY`). Lag — the **current** drive's outcome must be absent. |
| `drive_outcomes_away` | `list[str]` | derived | in-game | Same. |
| `scoring_drive_rate_home` | `float` | derived | in-game | Lag. |
| `scoring_drive_rate_away` | `float` | derived | in-game | Lag. |
| `points_per_drive_home` | `float` | derived | in-game | Lag. |
| `points_per_drive_away` | `float` | derived | in-game | Lag. |
| `last_3_drive_outcomes_posteam` | `list[str]` | derived | in-game | Momentum. Lag. |
| `last_3_drive_outcomes_defteam` | `list[str]` | derived | in-game | Lag. |
| `consecutive_scoring_drives_posteam` | `int` | derived | in-game | Lag. |
| `consecutive_empty_drives_posteam` | `int` | derived | in-game | Lag. |
| `expected_remaining_possessions` | `float` | derived | in-game | From clock, pace and timeouts. Pure pre-snap function. None. |

### 2.8 Momentum windows

All computed over a trailing window of plays or game-seconds ending strictly
before play `t`. Window sizes are hyperparameters (default: 5 plays, 10 plays,
1 drive, 5 game-minutes).

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `epa_last_5_plays_posteam` | `float` | derived | in-game | Lag. |
| `epa_last_10_plays_posteam` | `float` | derived | in-game | Lag. |
| `epa_last_5_plays_defteam` | `float` | derived | in-game | Lag. |
| `epa_last_10_plays_defteam` | `float` | derived | in-game | Lag. |
| `success_rate_last_10_posteam` | `float` | derived | in-game | Lag. |
| `points_last_5_minutes_home` | `int` | derived | in-game | Lag. |
| `points_last_5_minutes_away` | `int` | derived | in-game | Lag. |
| `yards_last_10_plays_posteam` | `int` | derived | in-game | Lag. |
| `score_swing_last_10_minutes` | `int` | derived | in-game | Lag. |
| `momentum_score` | `float` | derived | in-game | Composite of the above; single scalar in `[-1, 1]`. Lag. |
| `unanswered_points_streak` | `int` | derived | in-game | Signed toward home. Lag. |

### 2.9 Opponent-adjusted in-game production

Yards/EPA by unit, scaled by the opponent unit's season strength — so 300 yards
against the league's best defense is worth more than 300 against the worst. See
§6 for the adjustment formula; the ranks used are **as-of the week of the game**,
never end-of-season.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `adj_pass_yards_home` | `float` | derived | in-game | **Medium.** Uses opponent pass-defense rank; that rank must be as-of-week, not final. |
| `adj_rush_yards_home` | `float` | derived | in-game | Same. |
| `adj_epa_offense_home` | `float` | derived | in-game | Same. |
| `adj_pass_yards_away` | `float` | derived | in-game | Same. |
| `adj_rush_yards_away` | `float` | derived | in-game | Same. |
| `adj_epa_offense_away` | `float` | derived | in-game | Same. |
| `production_vs_expectation_home` | `float` | derived | in-game | Actual minus opponent-adjusted expected production to this point. Same. |
| `production_vs_expectation_away` | `float` | derived | in-game | Same. |

### 2.10 In-game injuries and availability

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `in_game_injuries_home` | `list[InGameInjury]` | espn.live / nflverse.pbp | in-game | **High.** Only injuries *announced* before play `t`. Post-hoc injury reports must not be back-dated. |
| `in_game_injuries_away` | `list[InGameInjury]` | espn.live / nflverse.pbp | in-game | Same. |
| `starting_qb_out_home` | `bool` | derived | in-game | **High.** Timestamp-gated. |
| `starting_qb_out_away` | `bool` | derived | in-game | Same. |
| `qb_change_occurred_home` | `bool` | derived | in-game | Inferred from passer identity in plays `< t`. Safe if lagged. |
| `qb_change_occurred_away` | `bool` | derived | in-game | Same. |
| `injury_severity_index_home` | `float` | derived | in-game | Weighted by position value of players lost. Timestamp-gated. |
| `injury_severity_index_away` | `float` | derived | in-game | Same. |
| `key_players_out_home` | `int` | derived | in-game | Count of top-N-snap-share players unavailable. Timestamp-gated. |
| `key_players_out_away` | `int` | derived | in-game | Same. |
| `snap_share_lost_home` | `float` | nflverse.snaps | in-game | Fraction of pregame-projected snap share now unavailable. Timestamp-gated. |
| `snap_share_lost_away` | `float` | nflverse.snaps | in-game | Same. |

### 2.11 Situational flags

Cheap booleans that let a model carve the state space without deep interactions.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `garbage_time` | `bool` | derived | in-game | Threshold on score diff × time remaining. Pre-snap. None. |
| `must_score_posteam` | `bool` | derived | in-game | Deficit exceeds remaining possessions × 8. None. |
| `victory_formation_likely` | `bool` | derived | in-game | None. |
| `onside_kick_situation` | `bool` | derived | in-game | None. |
| `fourth_down_go_situation` | `bool` | derived | in-game | None. |
| `field_goal_to_tie` | `bool` | derived | in-game | None. |
| `field_goal_to_win` | `bool` | derived | in-game | None. |
| `td_to_tie` | `bool` | derived | in-game | None. |
| `td_to_win` | `bool` | derived | in-game | None. |
| `two_point_needed_after_td` | `bool` | derived | in-game | None. |
| `receiving_second_half_kickoff` | `str \| None` | derived | in-game | Known at the opening coin toss. None. |
| `end_of_half_possession` | `bool` | derived | in-game | None. |
| `clock_kill_mode` | `bool` | derived | in-game | Leader is draining clock. None. |
| `desperation_index` | `float` | derived | in-game | Continuous version of `must_score_posteam`. None. |

---

## 3. Team

Two granularities: a **season-to-date** row per (team, season, week) and a
**rolling** view. All ranks and ratings are **as-of the week of the game**,
computed from games strictly before it.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `team` | `str` | static | static | None. |
| `season` | `int` | nflverse.sched | pregame | None. |
| `week` | `int` | nflverse.sched | pregame | None. |
| `as_of_date` | `date` | derived | pregame | **Critical.** Every stat below is valid only through this date. |
| `games_played` | `int` | derived | pregame | None if as-of. |
| `wins` / `losses` / `ties` | `int` | derived | pregame | As-of only — never final record. |
| `win_pct` | `float` | derived | pregame | As-of. |
| `division_rank` | `int` | derived | pregame | As-of. |
| `conference_rank` | `int` | derived | pregame | As-of. |
| `elo_rating` | `float` | derived | pregame | Elo-style rating updated after each completed game. **Must be the pre-game value.** Off-by-one here is the classic leak. |
| `elo_rating_change_last_4` | `float` | derived | pregame | As-of. |
| `srs_rating` | `float` | derived | pregame | Simple rating system (margin + SoS). As-of. |
| `pythagorean_win_pct` | `float` | derived | pregame | As-of. |
| `overall_rank` | `int` | derived | pregame | 1–32 composite. As-of. |
| `offense_rank` | `int` | derived | pregame | As-of. |
| `defense_rank` | `int` | derived | pregame | As-of. |
| `special_teams_rank` | `int` | derived | pregame | As-of. |
| `pass_offense_rank` / `rush_offense_rank` | `int` | derived | pregame | As-of. |
| `pass_defense_rank` / `rush_defense_rank` | `int` | derived | pregame | As-of. |
| `red_zone_offense_rank` / `red_zone_defense_rank` | `int` | derived | pregame | As-of. |
| `third_down_offense_rank` / `third_down_defense_rank` | `int` | derived | pregame | As-of. |
| `epa_per_play_offense` | `float` | nflverse.pbp | pregame | As-of. |
| `epa_per_play_defense` | `float` | nflverse.pbp | pregame | As-of. |
| `epa_per_pass_offense` / `epa_per_rush_offense` | `float` | nflverse.pbp | pregame | As-of. |
| `epa_per_pass_defense` / `epa_per_rush_defense` | `float` | nflverse.pbp | pregame | As-of. |
| `success_rate_offense` / `success_rate_defense` | `float` | derived | pregame | As-of. |
| `dvoa_like_offense` / `dvoa_like_defense` | `float` | derived | pregame | Opponent-adjusted efficiency (our own; see §6). As-of. |
| `points_per_game_for` / `points_per_game_against` | `float` | derived | pregame | As-of. |
| `yards_per_game_for` / `yards_per_game_against` | `float` | derived | pregame | As-of. |
| `turnover_margin_per_game` | `float` | derived | pregame | As-of. |
| `sack_rate_offense` / `sack_rate_defense` | `float` | derived | pregame | As-of. |
| `penalty_yards_per_game` | `float` | derived | pregame | As-of. |
| `pace_seconds_per_play` | `float` | derived | pregame | Neutral-situation pace. As-of. |
| `pace_plays_per_game` | `float` | derived | pregame | As-of. |
| `pass_rate_over_expected` | `float` | nflverse.pbp | pregame | As-of. |
| `neutral_pass_rate` | `float` | derived | pregame | As-of. |
| `fourth_down_aggressiveness` | `float` | derived | pregame | Go-rate over expectation. As-of. |
| `strength_of_schedule_played` | `float` | derived | pregame | Mean opponent rating faced so far. As-of. |
| `strength_of_schedule_remaining` | `float` | derived | pregame | **Low but nonzero** — uses future opponents' *current* ratings only, never their future results. |
| `rest_days` | `int` | derived | pregame | Days since previous game. None. |
| `opponent_rest_days` | `int` | derived | pregame | None. |
| `rest_advantage` | `int` | derived | pregame | None. |
| `on_bye_last_week` | `bool` | derived | pregame | None. |
| `short_week` | `bool` | derived | pregame | ≤5 days. None. |
| `consecutive_road_games` | `int` | derived | pregame | None. |
| `travel_miles_this_trip` | `float` | derived | pregame | Great-circle from previous venue. None. |
| `travel_miles_last_14_days` | `float` | derived | pregame | None. |
| `timezone_shift_hours` | `int` | derived | pregame | Signed. None. |
| `crossed_international_date_line` | `bool` | derived | pregame | None. |
| `altitude_change_feet` | `float` | derived | pregame | Denver. None. |
| `home_win_pct_3yr` | `float` | derived | pregame | Rolling, excludes current game. As-of. |
| `away_win_pct_3yr` | `float` | derived | pregame | As-of. |
| `record_in_one_score_games` | `float` | derived | pregame | As-of. Regression-to-mean signal. |
| `record_as_underdog` | `float` | derived | pregame | **Derived from market.** Evaluation only — excluded from features by ADR 0001. |
| `roster_continuity_index` | `float` | nflverse.rosters | pregame | Snap-weighted overlap vs last season. None. |
| `starters_available_pct` | `float` | derived | pregame | From the final injury report. **Must use the report published before kickoff.** |
| `qb_is_starter` | `bool` | nflverse.depth | pregame | Same gating. |
| `qb_games_started_season` | `int` | derived | pregame | As-of. |

---

## 4. Coach

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `coach_id` | `str` | static | static | None. |
| `name` | `str` | static | static | None. |
| `role` | `Literal["HC","OC","DC","STC"]` | static | static | None. |
| `team` | `str` | static | pregame | None. |
| `season` | `int` | static | pregame | None. |
| `age` | `int \| None` | static | pregame | None. |
| `first_season_as_hc` | `int \| None` | static | static | None. |
| `seasons_with_team` | `int` | derived | pregame | None. |
| `total_seasons_hc` | `int` | derived | pregame | None. |
| `career_wins` / `career_losses` | `int` | derived | pregame | **Must exclude the current game.** As-of. |
| `career_win_pct` | `float` | derived | pregame | As-of. |
| `win_pct_last_16_games` | `float` | derived | pregame | As-of. |
| `playoff_record` | `str` | derived | pregame | As-of. |
| `record_vs_opponent_hc` | `str` | derived | pregame | Head-to-head, as-of. |
| `record_in_one_score_games` | `float` | derived | pregame | As-of. |
| `record_after_bye` | `float` | derived | pregame | As-of. |
| `record_as_home_underdog` | `float` | derived | pregame | **Market-derived.** Evaluation only. |
| `fourth_down_go_rate` | `float` | derived | pregame | As-of. Real tactical signal in late Q4. |
| `two_point_attempt_rate` | `float` | derived | pregame | As-of. |
| `timeout_usage_efficiency` | `float` | derived | pregame | Rule-based score of late-game timeout discipline. As-of. |
| `challenge_success_rate` | `float` | derived | pregame | As-of. |
| `avg_seconds_per_offensive_play` | `float` | derived | pregame | As-of. |
| `clock_management_index` | `float` | derived | pregame | Composite. As-of. |
| `is_interim` | `bool` | static | pregame | None. |
| `hired_midseason_week` | `int \| None` | static | pregame | None. |
| `on_hot_seat` | `bool` | news | pregame | Soft signal from news. Low risk; timestamp-gated to before kickoff. |

---

## 5. Player

One row per (player, season, week). Physical/biographical fields are static;
performance fields are as-of-week.

### 5.1 Identity and biography

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `player_id` | `str` | nflverse.players | static | None. GSIS id. |
| `name` | `str` | nflverse.players | static | None. |
| `team` | `str` | nflverse.rosters | pregame | None. |
| `position` | `str` | nflverse.rosters | pregame | None. |
| `position_group` | `Literal["QB","RB","WR","TE","OL","DL","LB","DB","ST"]` | derived | pregame | None. |
| `depth_chart_rank` | `int \| None` | nflverse.depth | pregame | None — from the pre-kickoff depth chart. |
| `jersey_number` | `int \| None` | nflverse.rosters | pregame | None. |
| `birth_date` | `date \| None` | nflverse.players | static | None. |
| `age_at_game` | `float` | derived | pregame | None. |
| `height_inches` / `weight_lbs` | `int \| None` | nflverse.players | pregame | None. |
| `college` | `str \| None` | nflverse.players | static | None. |
| `draft_year` / `draft_round` / `draft_pick` | `int \| None` | nflverse.players | static | None. |
| `undrafted` | `bool` | derived | static | None. |
| `season_number` | `int` | derived | pregame | Years of NFL experience, 1-indexed. None. |
| `games_played_career` | `int` | derived | pregame | As-of. |
| `games_started_season` | `int` | derived | pregame | As-of. |
| `is_rookie` | `bool` | derived | pregame | None. |

### 5.2 Contract and status

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `contract_years_remaining` | `float \| None` | static | pregame | None. Hand-maintained / OverTheCap import. Motivation proxy. |
| `contract_apy_usd` | `float \| None` | static | pregame | None. |
| `cap_hit_usd` | `float \| None` | static | pregame | None. |
| `is_franchise_tagged` | `bool` | static | pregame | None. |
| `in_contract_year` | `bool` | derived | pregame | None. |
| `roster_status` | `Literal["ACT","INA","IR","PUP","NFI","SUSP","PS"]` | nflverse.rosters | pregame | **Medium** — status must be the one in effect before kickoff, not an end-of-season snapshot. |
| `weeks_on_ir_season` | `int` | derived | pregame | As-of. |
| `is_suspended` | `bool` | derived | pregame | Timestamp-gated. |
| `new_to_team_this_season` | `bool` | derived | pregame | None. |
| `traded_midseason` | `bool` | derived | pregame | Timestamp-gated to trade date. |

### 5.3 Performance (as-of-week, raw)

Position-specific; nulls where inapplicable.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `snap_share_offense` / `snap_share_defense` / `snap_share_st` | `float` | nflverse.snaps | pregame | As-of. |
| `targets` / `receptions` / `receiving_yards` / `receiving_tds` | `int` | nflverse.pbp | pregame | As-of. |
| `target_share` / `air_yards_share` | `float` | nflverse.pbp | pregame | As-of. |
| `yards_per_route_run` | `float \| None` | nflverse.pfr | pregame | As-of. |
| `rush_attempts` / `rushing_yards` / `rushing_tds` | `int` | nflverse.pbp | pregame | As-of. |
| `yards_after_contact_per_att` | `float \| None` | nflverse.pfr | pregame | As-of. |
| `pass_attempts` / `completions` / `passing_yards` / `passing_tds` / `interceptions` | `int` | nflverse.pbp | pregame | As-of. |
| `sacks_taken` / `time_to_throw` / `cpoe` | `float \| None` | nflverse.pbp/pfr | pregame | As-of. |
| `epa_per_play` | `float` | nflverse.pbp | pregame | As-of. |
| `epa_per_dropback` | `float \| None` | nflverse.pbp | pregame | As-of. |
| `pressures` / `sacks_made` / `qb_hits` / `tackles` / `tfl` | `int` | nflverse.pfr | pregame | As-of. |
| `pass_breakups` / `interceptions_made` / `forced_fumbles` | `int` | nflverse.pbp | pregame | As-of. |
| `coverage_targets` / `coverage_yards_allowed` / `passer_rating_allowed` | `float \| None` | nflverse.pfr | pregame | As-of. |
| `fg_attempts` / `fg_made` / `fg_pct` / `fg_made_50plus` | `int`/`float` | nflverse.pbp | pregame | As-of. Kicker quality matters enormously in late-Q4 win prob. |
| `punt_net_average` | `float \| None` | nflverse.pbp | pregame | As-of. |
| `position_rank_overall` | `int` | derived | pregame | Rank among same-position players by composite value. As-of. |
| `position_rank_on_team` | `int` | derived | pregame | As-of. |
| `value_over_replacement` | `float` | derived | pregame | Composite. As-of. |

### 5.4 Opponent-adjusted performance

See §6 for the formula. Each raw stat above has an `adj_` counterpart; the
headline ones are enumerated here.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `adj_receiving_yards_per_game` | `float` | derived | pregame | **Medium.** Opponent ranks must be as-of the week each game was played, not end-of-season. |
| `adj_rushing_yards_per_game` | `float` | derived | pregame | Same. |
| `adj_passing_yards_per_game` | `float` | derived | pregame | Same. |
| `adj_epa_per_play` | `float` | derived | pregame | Same. |
| `adj_pressure_rate` | `float` | derived | pregame | Same, vs opponent OL rank. |
| `adj_coverage_grade` | `float` | derived | pregame | Same, vs opponent receiving-corps rank. |
| `opponent_adjustment_factor_mean` | `float` | derived | pregame | Mean multiplier applied; stored for auditability. |
| `games_in_adjustment_sample` | `int` | derived | pregame | Guards against tiny-sample adjustments. |

---

## 6. Opponent adjustment — formula

Raw counting stats are misleading because schedules are unequal. Every `adj_*`
field is produced by the same procedure.

**Definitions.** For player `p` in game `g` against opponent unit `u` (e.g. the
opposing pass defense), let:

- `s(p, g)` = the raw stat in that game,
- `r(u, g)` = the opponent unit's rank **as of the week of game `g`**, integer 1
  (strongest) to 32 (weakest),
- `N = 32`.

**Step 1 — rank to strength multiplier.** Convert rank to a symmetric
multiplier centred on 1.0:

```
m(u, g) = 1 + k * ( (r(u, g) - (N + 1) / 2) / ((N - 1) / 2) )
```

so rank 1 → `1 - k`, rank 32 → `1 + k`, rank 16.5 → `1.0`. The `k` parameter
(default `k = 0.25`) sets adjustment aggressiveness and is tuned on the
validation season, not the test season.

Rationale for the sign: production *against a strong unit* (low rank number)
should be scaled **up**. With the definition above, `m` is small for a strong
opponent, so we divide:

**Step 2 — adjust the single-game stat.**

```
adj_s(p, g) = s(p, g) / m(u, g)
```

**Step 3 — aggregate over the sample.** Over the player's games `g ∈ G`, all
strictly before the game being predicted:

```
adj_stat(p) = ( Σ_{g ∈ G} w(g) * adj_s(p, g) ) / ( Σ_{g ∈ G} w(g) )
```

with recency weight `w(g) = λ^(weeks_ago(g))`, default `λ = 0.95`.

**Step 4 — shrink small samples** toward the position-group mean `μ_pos`:

```
adj_stat_shrunk(p) = ( n * adj_stat(p) + c * μ_pos ) / ( n + c )
```

where `n = |G|` and `c` (default 4) is the shrinkage constant in "games worth of
prior". `games_in_adjustment_sample` stores `n` so the shrinkage is auditable.

**Continuous-strength variant.** Where a continuous rating (EPA/play allowed,
`dvoa_like_defense`) exists, prefer it to the integer rank: replace `m` with

```
m(u, g) = 1 + k * z(u, g)
```

where `z` is the opponent unit's rating z-scored across the league as of that
week, clipped to `[-2.5, 2.5]`, and **oriented so that higher `z` means a weaker
opponent unit** — matching the rank convention, where a higher rank number is a
weaker unit. Concretely: for a defense, z-score `epa_per_play_defense` allowed
(higher = worse defense = weaker); for `dvoa_like_defense`, negate first if the
convention has lower = better. Getting this orientation backwards silently
inverts every adjustment, so the sign is asserted in the adjustment's unit test
(production against a top-5 defense must adjust *upward*).

Team-level `adj_*` fields (§2.9, §3) use this variant.

**Leakage rule.** `r(u, g)` and `z(u, g)` are **as-of the week of game `g`**.
Using end-of-season ranks to retroactively adjust week-2 production is a leak:
the week-2 feature would encode how good that defense turned out to be. A test
asserts that the adjustment for any historical game is unchanged when later
weeks' results are perturbed.

---

## 7. Venue

Static reference table, hand-maintained and committed.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `venue_id` | `str` | static | static | None. |
| `name` | `str` | static | static | None. |
| `city` / `state` / `country` | `str` | static | static | None. |
| `latitude` / `longitude` | `float` | static | static | None. Weather join key. |
| `altitude_feet` | `float` | static | static | None. |
| `timezone` | `str` | static | static | None. IANA name. |
| `roof_type` | `Literal["dome","retractable","open"]` | static | static | None. |
| `surface_type` | `str` | static | static | None. Turf vs grass; injury-rate relevance. |
| `capacity` | `int` | static | static | None. |
| `opened_year` | `int` | static | static | None. |
| `home_teams` | `list[str]` | static | static | None. |
| `is_neutral_for_game` | `bool` | derived | pregame | None. |
| `orientation_degrees` | `float \| None` | static | static | None. Sun/wind interaction. |
| `historical_home_win_pct` | `float` | derived | pregame | As-of; excludes the current game. |
| `historical_points_per_game` | `float` | derived | pregame | As-of. Venue scoring environment. |
| `noise_index` | `float \| None` | static | static | Hand-rated crowd-noise/false-start factor. None. |

---

## 8. Weather

One row per (game_id, observation hour). Forecast vs observed matters: for a
*live* prediction we only have the forecast + observations up to now.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `game_id` | `str` | derived | pregame | None. |
| `observed_at_utc` | `datetime` | openmeteo | in-game | None. |
| `is_forecast` | `bool` | openmeteo | pregame | **Important.** Historical training rows must use forecast-equivalent data or be flagged, so the model isn't trained on perfect hindsight weather it won't have live. |
| `temperature_f` | `float` | openmeteo | in-game | Observations after `t` must not be used. |
| `feels_like_f` | `float` | openmeteo | in-game | Same. |
| `humidity_pct` | `float` | openmeteo | in-game | Same. |
| `wind_speed_mph` | `float` | openmeteo | in-game | Same. Big effect on FG range and deep passing. |
| `wind_gust_mph` | `float` | openmeteo | in-game | Same. |
| `wind_direction_deg` | `float` | openmeteo | in-game | Same. |
| `wind_relative_to_field_deg` | `float \| None` | derived | in-game | Uses `Venue.orientation_degrees`. Same. |
| `precipitation_mm` | `float` | openmeteo | in-game | Same. |
| `precipitation_type` | `Literal["none","rain","snow","sleet","mix"]` | openmeteo | in-game | Same. |
| `snow_depth_cm` | `float` | openmeteo | in-game | Same. |
| `cloud_cover_pct` | `float` | openmeteo | in-game | Same. |
| `visibility_m` | `float` | openmeteo | in-game | Same. |
| `pressure_hpa` | `float` | openmeteo | in-game | Same. |
| `is_indoor` | `bool` | derived | pregame | None. When true, all of the above are neutralised. |
| `roof_closed` | `bool \| None` | espn.live | pregame | None. |
| `severe_weather_flag` | `bool` | derived | in-game | Same as components. |
| `kicking_difficulty_index` | `float` | derived | in-game | Composite of wind, temp, precipitation. Same. |
| `passing_difficulty_index` | `float` | derived | in-game | Same. |

---

## 9. Injury

Official weekly injury-report rows. Distinct from `InGameInjury` (§2.10).

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `game_id` | `str` | derived | pregame | None. |
| `season` / `week` | `int` | nflverse.inj | pregame | None. |
| `team` | `str` | nflverse.inj | pregame | None. |
| `player_id` | `str` | nflverse.inj | pregame | None. |
| `report_date` | `date` | nflverse.inj | pregame | **Critical.** Only reports with `report_date < kickoff` are usable. |
| `report_status` | `Literal["Out","Doubtful","Questionable","IR","DNP","LP","FP",""]` | nflverse.inj | pregame | Use the **final** pre-kickoff report, never a corrected/retroactive one. |
| `practice_status` | `str \| None` | nflverse.inj | pregame | Same. |
| `practice_participation_wed/thu/fri` | `str \| None` | nflverse.inj | pregame | Same. |
| `primary_injury` | `str \| None` | nflverse.inj | pregame | Body part. Same. |
| `secondary_injury` | `str \| None` | nflverse.inj | pregame | Same. |
| `is_new_injury` | `bool` | derived | pregame | Not on the prior week's report. Same. |
| `weeks_on_report` | `int` | derived | pregame | Same. |
| `expected_availability` | `float` | derived | pregame | Rule-based probability of playing, from `report_status` + practice history. Same. |
| `did_play` | `bool \| None` | nflverse.snaps | postgame | **High.** Postgame truth. Used to *calibrate* `expected_availability` offline; never a feature. |
| `snap_share_if_played` | `float \| None` | nflverse.snaps | postgame | **High.** Same treatment. |
| `position_importance_weight` | `float` | static | static | QB ≫ RB3. None. |
| `player_value_weight` | `float` | derived | pregame | From `Player.value_over_replacement`. As-of. |
| `weighted_absence_cost` | `float` | derived | pregame | `(1 - expected_availability) * position_importance_weight * player_value_weight`. Same. |

### 9.1 InGameInjury (nested in GameState)

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `player_id` | `str` | espn.live | in-game | Timestamp-gated. |
| `team` | `str` | espn.live | in-game | Same. |
| `position` | `str` | derived | in-game | Same. |
| `announced_at_play_index` | `int` | derived | in-game | **The gate.** Only injuries with `announced_at_play_index < t` are visible. |
| `status` | `Literal["questionable","doubtful","out","probable","returned"]` | espn.live | in-game | Same. |
| `body_part` | `str \| None` | espn.live | in-game | Same. |
| `is_starter` | `bool` | derived | in-game | Same. |
| `pregame_snap_share` | `float` | nflverse.snaps | pregame | Prior-week share; safe. |
| `severity_weight` | `float` | derived | in-game | Same gate as `status`. |

---

## 10. Sentiment

Rule-based scores over injury-report text and news headlines. **No cloud LLMs.**
A local Ollama scorer may be benchmarked and adopted only if the rule-based
version is measurably worse (see CLAUDE.md hard rule 4).

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `entity_type` | `Literal["team","player","coach"]` | derived | pregame | None. |
| `entity_id` | `str` | derived | pregame | None. |
| `game_id` | `str` | derived | pregame | None. |
| `window_start` / `window_end` | `datetime` | derived | pregame | **Critical.** `window_end` must be `<= kickoff` for pregame features, `<= snapshot_utc` for in-game. |
| `n_documents` | `int` | news | pregame | Same gate. |
| `n_injury_report_items` | `int` | nflverse.inj | pregame | Same gate. |
| `injury_sentiment_score` | `float` | derived | pregame | `[-1, 1]`; rule-based over report status transitions (upgraded/downgraded). Same gate. |
| `news_sentiment_score` | `float` | derived | pregame | `[-1, 1]`; lexicon + negation rules over headlines. Same gate. |
| `composite_sentiment` | `float` | derived | pregame | Weighted blend. Same gate. |
| `sentiment_volatility` | `float` | derived | pregame | Std-dev across documents in the window. Same gate. |
| `negative_headline_count` | `int` | derived | pregame | Same gate. |
| `positive_headline_count` | `int` | derived | pregame | Same gate. |
| `suspension_mentioned` | `bool` | derived | pregame | Same gate. |
| `holdout_mentioned` | `bool` | derived | pregame | Same gate. |
| `locker_room_discord_flag` | `bool` | derived | pregame | Same gate. |
| `coach_job_security_flag` | `bool` | derived | pregame | Same gate. |
| `scorer` | `Literal["rules","ollama"]` | derived | static | None. Which scorer produced this row. |
| `scorer_version` | `str` | derived | static | None. |
| `confidence` | `float` | derived | pregame | Scorer's own confidence; low when `n_documents` is small. |

**Explicitly excluded:** any sentiment derived from betting markets, line
movement commentary, or Kalshi discussion. That is market data wearing a
disguise (ADR 0001).

---

## 11. RosterSnapshot

The set of players available to a team at a moment. Pregame version is built
from the final depth chart + injury report; in-game version is updated as
injuries are announced.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `game_id` | `str` | derived | pregame | None. |
| `team` | `str` | derived | pregame | None. |
| `snapshot_kind` | `Literal["pregame","in_game"]` | derived | pregame | None. |
| `as_of` | `datetime` | derived | pregame | **The gate for everything below.** |
| `as_of_play_index` | `int \| None` | derived | in-game | Gate for in-game snapshots. |
| `active_player_ids` | `list[str]` | nflverse.rosters | pregame | Must reflect the gate, not the final game-day actives list if that is published after kickoff. |
| `inactive_player_ids` | `list[str]` | nflverse.rosters | pregame | Same. |
| `starting_qb_id` | `str \| None` | nflverse.depth | pregame | Same. |
| `backup_qb_id` | `str \| None` | nflverse.depth | pregame | Same. |
| `starters_by_position` | `dict[str, str]` | nflverse.depth | pregame | Same. |
| `ol_starters_available` | `int` | derived | pregame | Same. |
| `ol_continuity_index` | `float` | derived | pregame | Same OL five as last week. Same. |
| `skill_position_availability` | `float` | derived | pregame | Snap-share-weighted. Same. |
| `defensive_availability` | `float` | derived | pregame | Same. |
| `total_snap_share_available` | `float` | derived | pregame | Same. |
| `total_value_available` | `float` | derived | pregame | Sum of `value_over_replacement`. Same. |
| `value_lost_to_injury` | `float` | derived | pregame | Same. |
| `kicker_id` / `punter_id` | `str \| None` | nflverse.depth | pregame | Same. Kicker identity is materially predictive late. |
| `emergency_qb_designated` | `bool` | nflverse.rosters | pregame | Same. |
| `practice_squad_elevations` | `list[str]` | nflverse.rosters | pregame | Same. |

---

## 12. Market — **EVALUATION ONLY, NOT A FEATURE**

> **This entity is never an input to any model.** See
> `docs/ADR/0001-no-market-inputs.md`. Every row carries `market_only: True`,
> a literal type that cannot be set to `False`, so market rows are structurally
> distinguishable from feature rows. It exists to answer one question: *did we
> beat the market, and by how much?*

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `market_only` | `Literal[True]` | derived | static | **The guard.** Always `True`. |
| `game_id` | `str` | market | pregame | Join key. |
| `observed_at_utc` | `datetime` | market | in-game | — |
| `book` | `str` | market | pregame | e.g. `kalshi`, `pinnacle`, `consensus`. |
| `spread_home` | `float \| None` | market | pregame | **Catastrophic if used as a feature.** Excluded by construction. |
| `spread_open_home` | `float \| None` | market | pregame | Same. |
| `spread_close_home` | `float \| None` | market | pregame | Same. |
| `total_open` / `total_close` | `float \| None` | market | pregame | Same. |
| `moneyline_home` / `moneyline_away` | `int \| None` | market | pregame | Same. |
| `implied_prob_home` | `float \| None` | derived | pregame | Vig-removed. Same. |
| `implied_prob_away` | `float \| None` | derived | pregame | Same. |
| `kalshi_ticker` | `str \| None` | market | pregame | Same. |
| `kalshi_yes_price` | `float \| None` | market | in-game | Live in-game price. Same. |
| `kalshi_no_price` | `float \| None` | market | in-game | Same. |
| `kalshi_volume` | `int \| None` | market | in-game | Same. |
| `kalshi_open_interest` | `int \| None` | market | in-game | Same. |
| `bid_ask_spread` | `float \| None` | derived | in-game | Liquidity gate for position sizing. Same. |
| `line_movement_since_open` | `float \| None` | derived | pregame | Same. |
| `reverse_line_movement` | `bool \| None` | derived | pregame | Same. |
| `market_collapsed` | `bool` | derived | in-game | `implied_prob outside [0.15, 0.85]` — the trade gate from the project brief. Evaluation/decision layer only. |

---

## 13. Prediction ledger row

Written by `nfl/logging` on every prediction. Not a model input.

| Attribute | Type | Source | Avail | Leakage risk |
|---|---|---|---|---|
| `prediction_id` | `str` | derived | static | UUID. |
| `created_at_utc` | `datetime` | derived | static | None. |
| `game_id` | `str` | derived | in-game | None. |
| `snapshot_play_index` | `int \| None` | derived | in-game | None. |
| `snapshot_game_seconds_remaining` | `int \| None` | derived | in-game | None. |
| `feature_hash` | `str` | derived | static | SHA-256 of the ordered feature vector. Reproducibility anchor. |
| `feature_set_version` | `str` | derived | static | None. |
| `model_version` | `str` | derived | static | None. |
| `model_kind` | `str` | derived | static | `baseline`/`gbdt`/`mlp`/`seq`/`ensemble`. |
| `p_home_win` | `float` | derived | static | The output. |
| `p_away_win` | `float` | derived | static | `1 - p_home_win`. |
| `confidence` | `float` | derived | static | None. |
| `confidence_method` | `str` | derived | static | `ensemble_spread` / `mc_dropout`. |
| `ensemble_member_probs` | `list[float] \| None` | derived | static | For spread diagnostics. |
| `is_live` | `bool` | derived | static | Live feed vs historical replay. |
| `resolved_home_win` | `bool \| None` | derived | postgame | Filled in after the game. Resolution only. |
| `resolved_at_utc` | `datetime \| None` | derived | postgame | — |
| `market_implied_prob_home_at_snapshot` | `float \| None` | market | postgame | **Evaluation only.** Joined in after the fact, never at predict time. |
| `edge_vs_market` | `float \| None` | derived | postgame | `p_home_win - market_implied_prob_home_at_snapshot`. Evaluation only. |
| `brier` / `log_loss` | `float \| None` | derived | postgame | Evaluation only. |

---

## Leakage taxonomy — the four ways this project can break

1. **Post-play state at play `t`.** nflverse post-play columns
   (`total_home_score`, `posteam_score_post`, drive results) describe the state
   *after* the play. Every cumulative field in §2 must be lagged by one play.
   *Test:* score at play `t` equals score at play `t-1` plus points scored on
   play `t-1`, and never reflects play `t`.
2. **End-of-season ranks used for mid-season adjustment.** §6. All opponent
   ranks and ratings are as-of-week. *Test:* recompute a week-3 feature with
   weeks 4+ deleted; it must be identical.
3. **Retroactive injury/roster truth.** Whether a player actually played is
   postgame information. Features may use only the pre-kickoff report and the
   announcement timeline. *Test:* `did_play` and `snap_share_if_played` never
   appear in a training frame's columns.
4. **Market contamination.** Any column sourced from `market`, or derived from
   one, entering a feature frame. *Test:* the training feature-set column list
   is disjoint from the `Market` schema's field names, and no schema in a
   feature frame has `market_only`.
