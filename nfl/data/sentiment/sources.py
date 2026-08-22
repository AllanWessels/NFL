"""ESPN news + injury feeds, and the :class:`Sentiment` rows built from them.

Two endpoints, both free and unauthenticated (verified 2026-08-22):

* ``/apis/site/v2/sports/football/nfl/injuries`` -- **all 32 teams in one call**.
  It is ~8.8 MB because of the Rotowire-style ``longComment`` paragraphs, so
  :func:`parse_injuries` drops ``longComment``, ``links`` and ``headshot`` on
  ingest (~95% smaller). ``athlete.id`` is ``null`` in this payload, so player
  identity is a ``{team}:{name}`` slug until a stable id source is joined in.
* ``/apis/site/v2/sports/football/nfl/news`` -- headlines with team/athlete
  category tags.

Team keys are ESPN *numeric* ids; :data:`ESPN_TEAM_ID_TO_ABBR` maps them to
nflverse abbreviations (the only two that differ are WSH->WAS and LAR->LA).

``site.api.espn.com/.../teams/{abbr}/injuries`` returns a 2-byte body and is not
used. ``nfl.com/feeds/rss/news`` is 404 and is not used either.

Leakage: every :class:`Sentiment` row carries ``window_start``/``window_end``.
``window_end`` must be <= kickoff for a pregame feature. Callers pass the window;
this module never invents one from wall-clock time for a historical game.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from nfl import config
from nfl.data.schemas import Sentiment
from nfl.data.sentiment import injury_rules, news_rules
from nfl.data.sentiment.base import Document, DocumentScore, DocumentScorer
from nfl.data.sentiment.injury_rules import InjuryItem
from nfl.data.sentiment.news_rules import RuleBasedScorer

log = logging.getLogger(__name__)

ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_INJURIES_URL = f"{ESPN_SITE_BASE}/injuries"
ESPN_NEWS_URL = f"{ESPN_SITE_BASE}/news"

SOURCE_INJURIES = "espn.injuries"
SOURCE_NEWS = "espn.news"
SCHEMA_VERSION = "1"

#: ESPN numeric team id -> nflverse abbreviation (WSH->WAS, LAR->LA applied).
ESPN_TEAM_ID_TO_ABBR: dict[str, str] = {
    "1": "ATL", "2": "BUF", "3": "CHI", "4": "CIN", "5": "CLE", "6": "DAL",
    "7": "DEN", "8": "DET", "9": "GB", "10": "TEN", "11": "IND", "12": "KC",
    "13": "LV", "14": "LA", "15": "MIA", "16": "MIN", "17": "NE", "18": "NO",
    "19": "NYG", "20": "NYJ", "21": "PHI", "22": "ARI", "23": "PIT", "24": "LAC",
    "25": "SF", "26": "SEA", "27": "TB", "28": "WAS", "29": "CAR", "30": "JAX",
    "33": "BAL", "34": "HOU",
}  # fmt: skip

#: Fields stripped from the injuries payload on ingest (size, not privacy).
DROPPED_INJURY_FIELDS = ("longComment", "links", "headshot")

DEFAULT_TIMEOUT = 20.0
MAX_RETRIES = 3


class SourceError(RuntimeError):
    """A feed returned something unusable."""


@dataclass(frozen=True, slots=True)
class NewsDocument:
    """One ESPN news article, reduced to what the scorer needs."""

    headline: str
    published_at: datetime | None
    teams: tuple[str, ...] = ()
    athletes: tuple[str, ...] = ()
    description: str = ""
    article_id: str = ""

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.description}".strip()

    def to_document(self, team: str | None = None) -> Document:
        return Document(
            text=self.text,
            published_at=self.published_at,
            team=team or (self.teams[0] if self.teams else None),
            source=SOURCE_NEWS,
            doc_id=self.article_id,
        )


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _get_json(
    url: str, *, client: httpx.Client | None = None, params: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    owned = client is None
    client = client or httpx.Client(timeout=DEFAULT_TIMEOUT)
    try:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.get(url, params=dict(params or {}))
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code >= 500 or response.status_code == 429:
                    last = SourceError(f"{response.status_code} from {url}")
                elif response.status_code >= 400:
                    raise SourceError(f"{response.status_code} from {url}")
                else:
                    if len(response.content) < 16:
                        raise SourceError(f"empty body from {url} ({len(response.content)} bytes)")
                    return response.json()
            log.warning("espn: retrying %s (attempt %d)", url, attempt + 1)
        raise SourceError(f"GET {url} failed after {MAX_RETRIES} attempts") from last
    finally:
        if owned:
            client.close()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # ESPN sometimes emits "2026-08-21T01:25Z" without seconds; isoformat
        # handles that, but not other shapes -- fail soft rather than drop the row.
        log.debug("espn: unparseable timestamp %r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Injuries
# --------------------------------------------------------------------------- #


def player_slug(team: str, display_name: str) -> str:
    """Stable-ish player key: ``athlete.id`` is null in the injuries payload."""
    return f"{team}:{display_name.strip().lower().replace(' ', '_')}"


def parse_injuries(payload: Mapping[str, Any]) -> list[InjuryItem]:
    """ESPN injuries payload -> :class:`InjuryItem` rows (heavy fields dropped)."""
    items: list[InjuryItem] = []
    for team_block in payload.get("injuries", []) or []:
        team_id = str(team_block.get("id", ""))
        team = ESPN_TEAM_ID_TO_ABBR.get(team_id)
        if team is None:
            log.info("espn: unknown team id %r in injuries payload", team_id)
            continue
        for entry in team_block.get("injuries", []) or []:
            athlete = entry.get("athlete") or {}
            name = str(athlete.get("displayName", "") or "").strip()
            if not name:
                continue
            position = ((athlete.get("position") or {}).get("abbreviation")) or None
            details = entry.get("details") or {}
            reported_at = _parse_ts(entry.get("date"))
            items.append(
                InjuryItem(
                    player_id=str(athlete.get("id") or player_slug(team, name)),
                    team=team,
                    position=position,
                    status=str(entry.get("status", "") or ""),
                    primary_injury=(details.get("type") or details.get("detail") or None),
                    report_date=reported_at.date() if reported_at else None,
                    comment=str(entry.get("shortComment", "") or ""),
                )
            )
    return items


def fetch_espn_injuries(*, client: httpx.Client | None = None) -> list[InjuryItem]:
    """All 32 teams' injury reports in one call."""
    return parse_injuries(_get_json(ESPN_INJURIES_URL, client=client))


# --------------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------------- #


def parse_news(payload: Mapping[str, Any]) -> list[NewsDocument]:
    """ESPN news payload -> :class:`NewsDocument` rows."""
    documents: list[NewsDocument] = []
    for article in payload.get("articles", []) or []:
        headline = str(article.get("headline", "") or "").strip()
        if not headline:
            continue
        teams: list[str] = []
        athletes: list[str] = []
        for category in article.get("categories", []) or []:
            kind = str(category.get("type", "") or "")
            if kind == "team":
                team_id = str(
                    category.get("teamId") or (category.get("team") or {}).get("id") or ""
                )
                abbr = ESPN_TEAM_ID_TO_ABBR.get(team_id)
                if abbr and abbr not in teams:
                    teams.append(abbr)
            elif kind == "athlete":
                athlete_id = str(
                    category.get("athleteId") or (category.get("athlete") or {}).get("id") or ""
                )
                if athlete_id and athlete_id not in athletes:
                    athletes.append(athlete_id)
        documents.append(
            NewsDocument(
                headline=headline,
                published_at=_parse_ts(article.get("published") or article.get("lastModified")),
                teams=tuple(teams),
                athletes=tuple(athletes),
                description=str(article.get("description", "") or ""),
                article_id=str(article.get("id", "") or ""),
            )
        )
    return documents


def fetch_espn_news(*, limit: int = 50, client: httpx.Client | None = None) -> list[NewsDocument]:
    return parse_news(_get_json(ESPN_NEWS_URL, client=client, params={"limit": limit}))


# --------------------------------------------------------------------------- #
# Windowing + Sentiment rows
# --------------------------------------------------------------------------- #


def in_window(
    documents: Iterable[NewsDocument], window_start: datetime, window_end: datetime
) -> list[NewsDocument]:
    """Leakage gate: keep only documents published inside ``[start, end]``.

    Documents with no timestamp are **dropped**, not assumed in-window -- an
    undated article could postdate kickoff.
    """
    kept: list[NewsDocument] = []
    for document in documents:
        if document.published_at is None:
            continue
        if window_start <= document.published_at <= window_end:
            kept.append(document)
    return kept


def build_team_sentiment(
    *,
    game_id: str,
    team: str,
    window_start: datetime,
    window_end: datetime,
    injuries: Sequence[InjuryItem] = (),
    news: Sequence[NewsDocument] = (),
    scorer: DocumentScorer | None = None,
    ingested_at: datetime | None = None,
) -> Sentiment:
    """One team's :class:`Sentiment` row for one game window."""
    if window_end < window_start:
        raise ValueError("window_end must not precede window_start")
    active_scorer: DocumentScorer = scorer or RuleBasedScorer()

    team_injuries = [item for item in injuries if item.team == team]
    injury_summary = injury_rules.score_injury_items(team_injuries, team=team)

    windowed = in_window(news, window_start, window_end)
    team_news = [doc for doc in windowed if team in doc.teams] or (
        [] if any(doc.teams for doc in windowed) else list(windowed)
    )
    scores: list[DocumentScore] = active_scorer.score_documents(
        [doc.to_document(team) for doc in team_news]
    )
    news_summary = news_rules.summarize(scores)

    composite = _composite(
        injury_summary.injury_sentiment_score,
        news_summary.news_sentiment_score,
        n_injury_items=injury_summary.n_items,
        n_documents=news_summary.n_documents,
    )
    confidence = round(
        min(1.0, 0.5 * injury_summary.confidence + 0.5 * news_summary.confidence), 4
    )

    return Sentiment(
        source=f"{SOURCE_INJURIES}+{SOURCE_NEWS}",
        ingested_at=ingested_at or datetime.now(UTC),
        schema_version=SCHEMA_VERSION,
        entity_type="team",
        entity_id=team,
        game_id=game_id,
        window_start=window_start,
        window_end=window_end,
        n_documents=news_summary.n_documents,
        n_injury_report_items=injury_summary.n_items,
        injury_sentiment_score=injury_summary.injury_sentiment_score,
        news_sentiment_score=news_summary.news_sentiment_score,
        composite_sentiment=composite,
        sentiment_volatility=news_summary.sentiment_volatility,
        negative_headline_count=news_summary.negative_headline_count,
        positive_headline_count=news_summary.positive_headline_count,
        suspension_mentioned=news_summary.suspension_mentioned,
        holdout_mentioned=news_summary.holdout_mentioned,
        locker_room_discord_flag=news_summary.locker_room_discord_flag,
        coach_job_security_flag=news_summary.coach_job_security_flag,
        scorer="ollama" if getattr(active_scorer, "name", "rules") == "ollama" else "rules",
        scorer_version=getattr(active_scorer, "version", news_rules.SCORER_VERSION),
        confidence=confidence,
    )


#: The injury report is the harder signal; headlines are noisier commentary.
INJURY_WEIGHT = 0.65
NEWS_WEIGHT = 0.35


def _composite(
    injury_score: float, news_score: float, *, n_injury_items: int, n_documents: int
) -> float:
    """Evidence-weighted blend; a side with no evidence does not drag the blend."""
    weights = [
        (INJURY_WEIGHT if n_injury_items else 0.0, injury_score),
        (NEWS_WEIGHT if n_documents else 0.0, news_score),
    ]
    total = sum(weight for weight, _ in weights)
    if total == 0.0:
        return 0.0
    return round(max(-1.0, min(1.0, sum(weight * score for weight, score in weights) / total)), 4)


def build_game_sentiment(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    window_start: datetime,
    window_end: datetime,
    injuries: Sequence[InjuryItem] = (),
    news: Sequence[NewsDocument] = (),
    scorer: DocumentScorer | None = None,
) -> list[Sentiment]:
    """Both teams' :class:`Sentiment` rows for one game."""
    return [
        build_team_sentiment(
            game_id=game_id,
            team=team,
            window_start=window_start,
            window_end=window_end,
            injuries=injuries,
            news=news,
            scorer=scorer,
        )
        for team in (home_team, away_team)
    ]


def to_frame(rows: Sequence[Sentiment]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"game_id": pl.Utf8, "entity_id": pl.Utf8})
    return pl.DataFrame([row.model_dump() for row in rows], infer_schema_length=None)


def output_dir(season: int, root: Path | None = None) -> Path:
    base = root if root is not None else config.RAW_DIR
    return base / "sentiment" / "rules" / f"season={season}"


def fetch_sentiment(
    game_ids: Sequence[str],
    *,
    window_start: datetime,
    window_end: datetime,
    client: httpx.Client | None = None,
    scorer: DocumentScorer | None = None,
) -> pl.DataFrame:
    """Fetch both feeds once and score every requested game against them."""
    injuries = fetch_espn_injuries(client=client)
    news = fetch_espn_news(client=client)
    rows: list[Sentiment] = []
    for game_id in game_ids:
        parts = game_id.split("_")
        if len(parts) != 4:
            log.warning("sentiment: skipping malformed game_id %r", game_id)
            continue
        _, _, away, home = parts
        rows.extend(
            build_game_sentiment(
                game_id=game_id,
                home_team=home,
                away_team=away,
                window_start=window_start,
                window_end=window_end,
                injuries=injuries,
                news=news,
                scorer=scorer,
            )
        )
    return to_frame(rows)


def ingest_sentiment(
    game_ids: Sequence[str],
    *,
    window_start: datetime,
    window_end: datetime,
    season: int | None = None,
    root: Path | None = None,
    client: httpx.Client | None = None,
    scorer: DocumentScorer | None = None,
) -> Path:
    """Fetch, score and persist to ``data/raw/sentiment/rules/season=YYYY/``."""
    if not game_ids:
        raise ValueError("ingest_sentiment requires at least one game_id")
    resolved_season = season or int(game_ids[0].split("_")[0])
    frame = fetch_sentiment(
        game_ids,
        window_start=window_start,
        window_end=window_end,
        client=client,
        scorer=scorer,
    )
    directory = output_dir(resolved_season, root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sentiment_{window_end.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    frame.write_parquet(path)
    log.info("sentiment: wrote %d rows to %s", frame.height, path)
    return path


__all__ = [
    "DROPPED_INJURY_FIELDS",
    "ESPN_INJURIES_URL",
    "ESPN_NEWS_URL",
    "ESPN_TEAM_ID_TO_ABBR",
    "INJURY_WEIGHT",
    "NEWS_WEIGHT",
    "NewsDocument",
    "SourceError",
    "build_game_sentiment",
    "build_team_sentiment",
    "fetch_espn_injuries",
    "fetch_espn_news",
    "fetch_sentiment",
    "in_window",
    "ingest_sentiment",
    "output_dir",
    "parse_injuries",
    "parse_news",
    "player_slug",
    "to_frame",
]
