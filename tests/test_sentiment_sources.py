"""ESPN feed parsing + Sentiment row construction. All HTTP is respx-mocked."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import polars as pl
import pytest
import respx

from nfl.data.schemas import Sentiment
from nfl.data.sentiment import sources
from nfl.data.sentiment.injury_rules import InjuryItem

WINDOW_START = datetime(2026, 8, 20, tzinfo=UTC)
WINDOW_END = datetime(2026, 8, 22, 18, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Injuries
# --------------------------------------------------------------------------- #


def test_parse_injuries_maps_espn_team_ids(espn_injuries):
    items = sources.parse_injuries(espn_injuries)
    teams = {item.team for item in items}
    assert teams == {"KC", "DEN"}  # team id 999 is unknown and dropped


def test_parse_injuries_synthesizes_player_ids_because_espn_id_is_null(espn_injuries):
    items = sources.parse_injuries(espn_injuries)
    rice = next(item for item in items if "rice" in item.player_id)
    assert rice.player_id == "KC:rashee_rice"
    assert rice.position == "WR"
    assert rice.status == "Questionable"


def test_espn_team_id_table_uses_nflverse_abbreviations():
    assert sources.ESPN_TEAM_ID_TO_ABBR["28"] == "WAS"  # ESPN says WSH
    assert sources.ESPN_TEAM_ID_TO_ABBR["14"] == "LA"  # ESPN says LAR


@respx.mock
def test_fetch_espn_injuries(espn_injuries):
    respx.get(sources.ESPN_INJURIES_URL).mock(return_value=httpx.Response(200, json=espn_injuries))
    items = sources.fetch_espn_injuries()
    assert len(items) == 3


@respx.mock
def test_two_byte_body_is_an_error_not_a_silent_empty_list():
    # site.api.espn.com/.../teams/{abbr}/injuries returns exactly this.
    respx.get(sources.ESPN_INJURIES_URL).mock(return_value=httpx.Response(200, content=b"{}"))
    with pytest.raises(sources.SourceError):
        sources.fetch_espn_injuries()


@respx.mock
def test_http_error_raises():
    respx.get(sources.ESPN_INJURIES_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(sources.SourceError):
        sources.fetch_espn_injuries()


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #


def test_parse_news_extracts_team_tags(espn_news):
    documents = sources.parse_news(espn_news)
    assert len(documents) == 4  # the empty headline is skipped
    nix = documents[0]
    assert nix.teams == ("DEN",)
    assert nix.published_at == datetime(2026, 8, 21, 13, 0, tzinfo=UTC)


@respx.mock
def test_fetch_espn_news(espn_news):
    respx.get(sources.ESPN_NEWS_URL).mock(return_value=httpx.Response(200, json=espn_news))
    assert len(sources.fetch_espn_news(limit=10)) == 4


# --------------------------------------------------------------------------- #
# Leakage gate
# --------------------------------------------------------------------------- #


def test_in_window_gates_old_documents(espn_news):
    documents = sources.parse_news(espn_news)
    kept = sources.in_window(documents, WINDOW_START, WINDOW_END)
    assert all(doc.published_at >= WINDOW_START for doc in kept)
    assert "Stale headline" not in " ".join(doc.headline for doc in kept)


def test_undated_documents_are_dropped_not_assumed_in_window():
    undated = sources.NewsDocument(headline="No timestamp", published_at=None)
    assert sources.in_window([undated], WINDOW_START, WINDOW_END) == []


def test_documents_after_window_end_are_gated_out(espn_news):
    documents = sources.parse_news(espn_news)
    early_end = datetime(2026, 8, 21, 14, 0, tzinfo=UTC)
    kept = sources.in_window(documents, WINDOW_START, early_end)
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# Sentiment rows
# --------------------------------------------------------------------------- #


def test_build_team_sentiment_is_a_valid_schema_row(espn_injuries, espn_news):
    row = sources.build_team_sentiment(
        game_id="2026_01_DEN_KC",
        team="DEN",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        injuries=sources.parse_injuries(espn_injuries),
        news=sources.parse_news(espn_news),
    )
    assert isinstance(row, Sentiment)
    assert row.entity_type == "team"
    assert row.entity_id == "DEN"
    assert row.scorer == "rules"
    assert row.n_injury_report_items == 1
    assert row.n_documents == 1
    assert -1.0 <= row.composite_sentiment <= 1.0


def test_qb_out_makes_the_team_negative(espn_injuries, espn_news):
    rows = sources.build_game_sentiment(
        game_id="2026_01_DEN_KC",
        home_team="KC",
        away_team="DEN",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        injuries=sources.parse_injuries(espn_injuries),
        news=sources.parse_news(espn_news),
    )
    kc, den = rows[0], rows[1]
    assert kc.entity_id == "KC" and den.entity_id == "DEN"
    # Both teams have a QB on the report, and the feed carries no depth chart,
    # so the *injury* scores are comparable; the news is what separates them.
    assert den.news_sentiment_score < kc.news_sentiment_score
    assert den.composite_sentiment < kc.composite_sentiment
    assert den.composite_sentiment < 0


def test_market_headline_does_not_move_the_score(espn_injuries, espn_news):
    row = sources.build_team_sentiment(
        game_id="2026_01_DEN_KC",
        team="KC",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        injuries=[],
        news=sources.parse_news(espn_news),
    )
    # KC's two in-window headlines: one positive (activated), one market (line
    # movement). Only the positive one counts.
    assert row.n_documents == 2
    assert row.news_sentiment_score > 0
    assert row.negative_headline_count == 0


def test_inverted_window_is_rejected():
    with pytest.raises(ValueError, match="window_end"):
        sources.build_team_sentiment(
            game_id="g",
            team="KC",
            window_start=WINDOW_END,
            window_end=WINDOW_START,
        )


def test_no_evidence_gives_zero_composite_and_zero_confidence():
    row = sources.build_team_sentiment(
        game_id="g", team="KC", window_start=WINDOW_START, window_end=WINDOW_END
    )
    assert row.composite_sentiment == 0.0
    assert row.confidence == 0.0
    assert row.n_documents == 0


def test_composite_ignores_a_side_with_no_evidence():
    row = sources.build_team_sentiment(
        game_id="g",
        team="KC",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        injuries=[InjuryItem(player_id="p", team="KC", position="QB", status="Out")],
    )
    # No news at all, so composite equals the injury score rather than being
    # diluted toward zero by an empty news side.
    assert row.composite_sentiment == row.injury_sentiment_score


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


@respx.mock
def test_ingest_writes_partitioned_parquet(espn_injuries, espn_news, tmp_path):
    respx.get(sources.ESPN_INJURIES_URL).mock(return_value=httpx.Response(200, json=espn_injuries))
    respx.get(sources.ESPN_NEWS_URL).mock(return_value=httpx.Response(200, json=espn_news))

    path = sources.ingest_sentiment(
        ["2026_01_DEN_KC"],
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        root=tmp_path,
    )
    assert path.parent == tmp_path / "sentiment" / "rules" / "season=2026"

    frame = pl.read_parquet(path)
    assert frame.height == 2  # one row per team
    assert sorted(frame["entity_id"].to_list()) == ["DEN", "KC"]
    assert set(frame["scorer"].to_list()) == {"rules"}


def test_ingest_rejects_empty_game_list():
    with pytest.raises(ValueError, match="at least one game_id"):
        sources.ingest_sentiment([], window_start=WINDOW_START, window_end=WINDOW_END)
