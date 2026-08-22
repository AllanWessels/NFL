"""Rule-based headline scorer tests."""

from __future__ import annotations

import pytest

from nfl.data.sentiment import news_rules
from nfl.data.sentiment.base import Document

# --------------------------------------------------------------------------- #
# Direction and category
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("headline", "category"),
    [
        ("Star QB out for the season with torn ACL", "injury"),
        ("Receiver suspended six games for violating the personal conduct policy", "suspension"),
        ("Team acquires veteran edge rusher in trade", "trade"),
        ("Head coach fired after 0-6 start", "coaching_change"),
        ("Starter benched, backup quarterback will start Sunday", "qb_change"),
        ("Blizzard expected at kickoff", "weather"),
        ("Locker room turmoil deepens after sideline altercation", "discord"),
        ("Quiet day at the facility", "neutral"),
    ],
)
def test_category_detection(headline, category):
    assert news_rules.score_headline(headline).category == category


def test_bad_news_is_negative_and_good_news_is_positive():
    assert news_rules.score_headline("QB placed on injured reserve").score < 0
    assert news_rules.score_headline("QB activated from injured reserve").score > 0


def test_scores_are_clamped():
    piled_on = news_rules.score_headline(
        "Torn ACL, torn achilles, season-ending surgery, placed on injured reserve"
    )
    assert -1.0 <= piled_on.score <= 1.0
    assert piled_on.score == -1.0


def test_neutral_headline_scores_zero_with_low_confidence():
    score = news_rules.score_headline("Team holds walkthrough on Wednesday")
    assert score.score == 0.0
    assert score.category == "neutral"
    assert score.confidence < 0.2


# --------------------------------------------------------------------------- #
# Negation
# --------------------------------------------------------------------------- #


def test_negation_flips_sign_rather_than_cancelling():
    plain = news_rules.score_headline("Mahomes expected to miss Sunday's game")
    negated = news_rules.score_headline("Mahomes not expected to miss Sunday's game")
    assert plain.score < 0
    assert negated.score > 0
    assert negated.negated is True


def test_negation_window_is_bounded():
    # The cue is far enough away that it should not flip the injury term.
    far = news_rules.score_headline(
        "No decision has been made yet by the front office about the veteran, who will miss Sunday"
    )
    assert far.score < 0


# --------------------------------------------------------------------------- #
# Longest match
# --------------------------------------------------------------------------- #


def test_longest_phrase_wins_over_its_substring():
    score = news_rules.score_headline("Rice cleared to play, activated from injured reserve")
    assert score.score > 0
    assert "activated from injured reserve" in score.matched_terms
    # The bare "injured reserve" term must not double-count inside the phrase.
    assert "injured reserve" not in score.matched_terms


# --------------------------------------------------------------------------- #
# ADR 0001: market chatter must not become sentiment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "headline",
    [
        "Chiefs open as 7.5-point favorite as the betting line moves",
        "Kalshi traders push the moneyline toward Kansas City",
        "Sharp money hits the over/under at every sportsbook",
    ],
)
def test_market_headlines_score_zero(headline):
    score = news_rules.score_headline(headline)
    assert score.score == 0.0
    assert score.category == "market"


def test_market_terms_do_not_suppress_real_news_in_the_same_headline():
    score = news_rules.score_headline("Line moves after QB ruled out with a concussion")
    assert score.category == "injury"
    assert score.score < 0


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #


def test_flags_are_raised():
    summary = news_rules.score_headlines(
        [
            "Star receiver suspended for six games",
            "Holdout continues as contract dispute drags on",
            "Locker room rift widens",
            "Head coach on the hot seat after another loss",
        ]
    )
    assert summary.suspension_mentioned
    assert summary.holdout_mentioned
    assert summary.locker_room_discord_flag
    assert summary.coach_job_security_flag


def test_negated_flag_term_does_not_raise_the_flag():
    summary = news_rules.score_headlines(["Player will not be suspended, league says"])
    assert not summary.suspension_mentioned


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_summary_counts_and_volatility():
    summary = news_rules.score_headlines(
        [
            "QB out for the season with a torn ACL",
            "WR cleared to play",
            "Team holds a walkthrough",
        ]
    )
    assert summary.n_documents == 3
    assert summary.negative_headline_count == 1
    assert summary.positive_headline_count == 1
    assert summary.sentiment_volatility > 0.0
    assert -1.0 <= summary.news_sentiment_score <= 1.0


def test_empty_window_is_neutral_and_unconfident():
    summary = news_rules.score_headlines([])
    assert summary.n_documents == 0
    assert summary.news_sentiment_score == 0.0
    assert summary.sentiment_volatility == 0.0
    assert summary.confidence == 0.0


def test_single_signal_has_zero_volatility():
    summary = news_rules.score_headlines(["QB ruled out"])
    assert summary.sentiment_volatility == 0.0


# --------------------------------------------------------------------------- #
# Scorer interface
# --------------------------------------------------------------------------- #


def test_rule_based_scorer_implements_the_protocol():
    scorer = news_rules.RuleBasedScorer()
    assert scorer.enabled
    assert scorer.name == "rules"

    scores = scorer.score_documents(
        [Document(text="QB ruled out", team="KC"), Document(text="Nothing happened", team="KC")]
    )
    assert len(scores) == 2
    assert scores[0].score < 0


def test_scorer_is_deterministic():
    scorer = news_rules.RuleBasedScorer()
    document = Document(text="Coach fired after a locker room rift", team="NYJ")
    assert scorer.score_document(document) == scorer.score_document(document)
