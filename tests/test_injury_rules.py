"""Rule-based injury scorer tests."""

from __future__ import annotations

from datetime import date

import pytest

from nfl.data.schemas import InjuryReport
from nfl.data.sentiment import injury_rules
from nfl.data.sentiment.injury_rules import InjuryItem

# --------------------------------------------------------------------------- #
# Reference table
# --------------------------------------------------------------------------- #


def test_position_impact_table_loads_and_qb_is_the_ceiling():
    table = injury_rules.load_position_impact()
    assert table["QB"].impact_weight == 1.0
    assert all(row.impact_weight <= 1.0 for row in table.values())
    assert all(0.0 <= row.replacement_dropoff <= 1.0 for row in table.values())


def test_unknown_position_gets_conservative_default():
    impact = injury_rules.position_impact("SNAPCHAT_MANAGER")
    assert impact.impact_weight == injury_rules.DEFAULT_POSITION_WEIGHT


def test_qb_outweighs_every_other_position():
    assert injury_rules.position_weight("QB") > injury_rules.position_weight("LT")
    assert injury_rules.position_weight("LT") > injury_rules.position_weight("LS")


# --------------------------------------------------------------------------- #
# Status normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Questionable", "QUESTIONABLE"),
        ("INJURY_STATUS_OUT", "OUT"),
        ("Q", "QUESTIONABLE"),
        ("Injured Reserve", "IR"),
        ("Did Not Participate In Practice", "DNP"),
        (None, ""),
        ("", ""),
    ],
)
def test_canonical_status(raw, expected):
    assert injury_rules.canonical_status(raw) == expected


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", ["Out", "IR", "PUP", "Suspended"])
def test_hard_zero_designations_are_not_rescued_by_practice(status):
    assert (
        injury_rules.availability_probability(
            status, practice_wed="FP", practice_thu="FP", practice_fri="FP"
        )
        == 0.0
    )


def test_questionable_moves_with_practice_participation():
    base = injury_rules.availability_probability("Questionable")
    full = injury_rules.availability_probability(
        "Questionable", practice_wed="FP", practice_thu="FP", practice_fri="FP"
    )
    none = injury_rules.availability_probability(
        "Questionable", practice_wed="DNP", practice_thu="DNP", practice_fri="DNP"
    )
    assert none < base < full
    assert full == pytest.approx(1.0)
    assert none == pytest.approx(0.10, abs=0.01)


def test_friday_practice_dominates_wednesday():
    late_upgrade = injury_rules.availability_probability(
        "Questionable", practice_wed="DNP", practice_thu="LP", practice_fri="FP"
    )
    late_downgrade = injury_rules.availability_probability(
        "Questionable", practice_wed="FP", practice_thu="LP", practice_fri="DNP"
    )
    assert late_upgrade > late_downgrade


def test_no_designation_means_available():
    assert injury_rules.availability_probability("") == 1.0
    assert injury_rules.availability_probability(None) == 1.0


def test_practice_signal_is_none_when_nothing_reported():
    assert injury_rules.practice_signal(None, None, None) is None
    assert injury_rules.practice_signal("garbage", None, None) is None


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #


def test_status_delta_signs():
    assert injury_rules.status_delta("Out", "Questionable") > 0  # upgraded
    assert injury_rules.status_delta("Questionable", "Out") < 0  # downgraded
    assert injury_rules.status_delta("Questionable", "Questionable") == 0.0
    assert injury_rules.status_delta(None, None) == 0.0


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_absence_cost_scales_with_position_and_availability():
    qb_out = injury_rules.weighted_absence_cost(0.0, "QB")
    qb_probable = injury_rules.weighted_absence_cost(0.95, "QB")
    ls_out = injury_rules.weighted_absence_cost(0.0, "LS")

    assert qb_out > ls_out
    assert qb_probable < qb_out
    assert injury_rules.weighted_absence_cost(1.0, "QB") == 0.0


def test_player_value_weight_scales_the_positional_prior():
    star = injury_rules.weighted_absence_cost(0.0, "WR", player_value_weight=1.0)
    depth = injury_rules.weighted_absence_cost(0.0, "WR", player_value_weight=0.0)
    assert star > depth


# --------------------------------------------------------------------------- #
# Team rollup
# --------------------------------------------------------------------------- #


def test_qb_out_dominates_the_team_score():
    qb_out = injury_rules.score_injury_items(
        [InjuryItem(player_id="p1", team="KC", position="QB", status="Out")]
    )
    ls_out = injury_rules.score_injury_items(
        [InjuryItem(player_id="p2", team="KC", position="LS", status="Out")]
    )
    assert qb_out.injury_sentiment_score < ls_out.injury_sentiment_score
    assert qb_out.qb_absence_cost > 0.0
    assert -1.0 <= qb_out.injury_sentiment_score <= 1.0


def test_clean_report_scores_zero():
    summary = injury_rules.score_injury_items([])
    assert summary.injury_sentiment_score == 0.0
    assert summary.n_items == 0
    assert summary.confidence == 0.0


def test_upgrades_lift_the_score_relative_to_downgrades():
    upgraded = injury_rules.score_injury_items(
        [
            InjuryItem(
                player_id="p1",
                team="KC",
                position="WR",
                status="Questionable",
                previous_status="Out",
            )
        ]
    )
    downgraded = injury_rules.score_injury_items(
        [
            InjuryItem(
                player_id="p1",
                team="KC",
                position="WR",
                status="Questionable",
                previous_status="",
            )
        ]
    )
    assert upgraded.injury_sentiment_score > downgraded.injury_sentiment_score
    assert upgraded.n_upgraded == 1
    assert downgraded.n_downgraded == 1


def test_rollup_counts_designations_and_scopes_by_team():
    summary = injury_rules.score_injury_items(
        [
            InjuryItem(player_id="a", team="KC", position="QB", status="Out"),
            InjuryItem(player_id="b", team="KC", position="WR", status="Doubtful"),
            InjuryItem(player_id="c", team="KC", position="CB", status="Questionable"),
            InjuryItem(player_id="d", team="DEN", position="QB", status="Out"),
        ],
        team="KC",
    )
    assert (summary.n_items, summary.n_out, summary.n_doubtful, summary.n_questionable) == (
        3,
        1,
        1,
        1,
    )
    assert summary.team == "KC"


def test_score_is_clamped_by_a_catastrophic_report():
    items = [
        InjuryItem(player_id=f"p{i}", team="KC", position="QB", status="Out") for i in range(5)
    ]
    assert injury_rules.score_injury_items(items).injury_sentiment_score >= -1.0


def test_second_player_at_a_position_is_discounted_as_a_backup():
    one = injury_rules.score_injury_items(
        [InjuryItem(player_id="a", team="KC", position="WR", status="Out")]
    )
    two = injury_rules.score_injury_items(
        [
            InjuryItem(player_id="a", team="KC", position="WR", status="Out"),
            InjuryItem(player_id="b", team="KC", position="WR", status="Out"),
        ]
    )
    added = two.total_absence_cost - one.total_absence_cost
    assert 0 < added < one.total_absence_cost  # the WR2 costs less than the WR1


def test_depth_discount_does_not_apply_across_different_positions():
    same = injury_rules.score_injury_items(
        [
            InjuryItem(player_id="a", team="KC", position="DE", status="Out"),
            InjuryItem(player_id="b", team="KC", position="DE", status="Out"),
        ]
    )
    different = injury_rules.score_injury_items(
        [
            InjuryItem(player_id="a", team="KC", position="DE", status="Out"),
            InjuryItem(player_id="b", team="KC", position="CB", status="Out"),
        ]
    )
    assert different.total_absence_cost > same.total_absence_cost


# --------------------------------------------------------------------------- #
# Schema rows
# --------------------------------------------------------------------------- #


def test_to_injury_reports_produces_valid_schema_rows():
    rows = injury_rules.to_injury_reports(
        [
            InjuryItem(
                player_id="p1",
                team="KC",
                position="QB",
                status="Questionable",
                practice_wed="DNP",
                practice_fri="FP",
                primary_injury="Ankle",
                report_date=date(2024, 9, 4),
            )
        ],
        game_id="2024_01_BAL_KC",
        season=2024,
        week=1,
        source="espn.injuries",
    )
    row = rows[0]
    assert isinstance(row, InjuryReport)
    assert row.report_status == "Questionable"
    assert 0.0 <= row.expected_availability <= 1.0
    assert row.position_importance_weight == 1.0
    assert row.weighted_absence_cost > 0.0
    # Postgame truth stays empty at ingest -- it is never a feature.
    assert row.did_play is None


def test_unknown_status_degrades_to_empty_schema_literal():
    rows = injury_rules.to_injury_reports(
        [InjuryItem(player_id="p", team="KC", position="WR", status="Weird Status")],
        game_id="g",
        season=2024,
        week=1,
        source="test",
        report_date=date(2024, 9, 4),
    )
    assert rows[0].report_status == ""
