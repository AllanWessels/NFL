"""Rule-based headline scorer v1: lexicon + negation, no model, no network.

Design notes worth knowing before tuning the weights:

* **Longest match wins.** Phrases beat single words, so "expected to play" is not
  scored as "play"; the lexicon is matched longest-first.
* **Negation flips, it does not cancel.** "Not expected to miss Sunday" is *good*
  news, not neutral. A negation cue within :data:`NEGATION_WINDOW` tokens before
  a matched term inverts that term's sign.
* **Category is the strongest match**, by absolute weight, not the first hit.
* **Market chatter scores 0.0** and is categorised ``market``. Headlines about
  spreads, lines, odds or Kalshi are market data wearing a disguise (ADR 0001)
  and must not leak into a feature via the sentiment table.
* Scores are from the *subject team's* perspective: negative = bad for them.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from nfl.data.sentiment.base import CATEGORIES, Category, Document, DocumentScore, clamp

SCORER_NAME = "rules"
SCORER_VERSION = "0.1.0"

#: How many tokens back from a matched term a negation cue still applies.
NEGATION_WINDOW = 4

NEGATION_CUES: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "denies",
        "denied",
        "avoids",
        "avoided",
        "unlikely",
        "isnt",
        "isn't",
        "wasnt",
        "wasn't",
        "wont",
        "won't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "dodges",
        "dodged",
        "escapes",
        "escaped",
    }
)
# Deliberately *not* a cue: "cleared". "Cleared to play" is positive news, and
# treating it as a negation flipped every good headline that mentioned it.

#: ``term -> (category, weight)``. Weight is the per-hit contribution before
#: aggregation; the sign is the news direction for the subject team.
LEXICON: dict[str, tuple[Category, float]] = {
    # -- injury ---------------------------------------------------------- #
    "torn acl": ("injury", -1.0),
    "torn achilles": ("injury", -1.0),
    "season-ending": ("injury", -1.0),
    "season ending": ("injury", -1.0),
    "out for the season": ("injury", -1.0),
    "placed on injured reserve": ("injury", -0.85),
    "injured reserve": ("injury", -0.8),
    "carted off": ("injury", -0.8),
    "will miss": ("injury", -0.6),
    "expected to miss": ("injury", -0.6),
    "ruled out": ("injury", -0.65),
    "did not practice": ("injury", -0.35),
    "limited in practice": ("injury", -0.2),
    "questionable": ("injury", -0.25),
    "doubtful": ("injury", -0.5),
    "concussion protocol": ("injury", -0.55),
    "concussion": ("injury", -0.5),
    "hamstring": ("injury", -0.3),
    "sprain": ("injury", -0.3),
    "fracture": ("injury", -0.7),
    "broken": ("injury", -0.6),
    "surgery": ("injury", -0.7),
    "injury": ("injury", -0.3),
    "injured": ("injury", -0.35),
    "banged up": ("injury", -0.3),
    # -- return / positive ----------------------------------------------- #
    "activated from injured reserve": ("return", 0.7),
    "cleared to play": ("return", 0.6),
    "expected to play": ("return", 0.5),
    "returns to practice": ("return", 0.45),
    "full participant": ("return", 0.4),
    "activated": ("return", 0.4),
    "back at practice": ("return", 0.4),
    "good to go": ("return", 0.5),
    "upgraded": ("return", 0.35),
    "signs extension": ("contract", 0.3),
    "contract extension": ("contract", 0.3),
    # -- suspension ------------------------------------------------------ #
    "indefinitely suspended": ("suspension", -0.9),
    "suspended": ("suspension", -0.7),
    "suspension": ("suspension", -0.65),
    "banned": ("suspension", -0.7),
    "violating the personal conduct policy": ("suspension", -0.7),
    "performance-enhancing": ("suspension", -0.6),
    "arrested": ("suspension", -0.6),
    "holdout": ("contract", -0.5),
    "holding out": ("contract", -0.5),
    "contract dispute": ("contract", -0.4),
    # -- trade ----------------------------------------------------------- #
    "traded": ("trade", -0.3),
    "trade request": ("trade", -0.5),
    "requests trade": ("trade", -0.5),
    "released": ("trade", -0.35),
    "waived": ("trade", -0.25),
    "acquires": ("trade", 0.3),
    "acquired": ("trade", 0.3),
    "signs": ("trade", 0.2),
    # -- coaching -------------------------------------------------------- #
    "fired": ("coaching_change", -0.6),
    "relieved of his duties": ("coaching_change", -0.6),
    "steps down": ("coaching_change", -0.5),
    "interim head coach": ("coaching_change", -0.45),
    "hot seat": ("coaching_change", -0.35),
    "job security": ("coaching_change", -0.3),
    "coordinator change": ("coaching_change", -0.3),
    "play-calling duties": ("coaching_change", -0.25),
    # -- qb change ------------------------------------------------------- #
    "benched": ("qb_change", -0.6),
    "benching": ("qb_change", -0.55),
    "quarterback change": ("qb_change", -0.5),
    "qb change": ("qb_change", -0.5),
    "backup quarterback": ("qb_change", -0.45),
    "will start at quarterback": ("qb_change", -0.2),
    "named starter": ("qb_change", 0.2),
    "starting quarterback": ("qb_change", -0.15),
    # -- weather --------------------------------------------------------- #
    "blizzard": ("weather", -0.4),
    "heavy snow": ("weather", -0.35),
    "high winds": ("weather", -0.3),
    "wind advisory": ("weather", -0.25),
    "heavy rain": ("weather", -0.2),
    "frigid": ("weather", -0.2),
    "weather delay": ("weather", -0.25),
    # -- locker room ----------------------------------------------------- #
    "locker room": ("discord", -0.3),
    "turmoil": ("discord", -0.5),
    "dysfunction": ("discord", -0.5),
    "rift": ("discord", -0.45),
    "calls out teammates": ("discord", -0.45),
    "frustrated": ("discord", -0.25),
    "sideline altercation": ("discord", -0.5),
    "war of words": ("discord", -0.35),
    # -- market: scored to zero on purpose (ADR 0001) --------------------- #
    "point spread": ("market", 0.0),
    "betting line": ("market", 0.0),
    "line movement": ("market", 0.0),
    "moneyline": ("market", 0.0),
    "odds": ("market", 0.0),
    "over/under": ("market", 0.0),
    "kalshi": ("market", 0.0),
    "sportsbook": ("market", 0.0),
    "favorite": ("market", 0.0),
    "underdog": ("market", 0.0),
    "parlay": ("market", 0.0),
}

#: Terms that raise a boolean flag on the team's :class:`Sentiment` row.
FLAG_TERMS: dict[str, tuple[str, ...]] = {
    "suspension_mentioned": ("suspended", "suspension", "banned", "indefinitely suspended"),
    "holdout_mentioned": ("holdout", "holding out", "contract dispute"),
    "locker_room_discord_flag": (
        "locker room",
        "turmoil",
        "dysfunction",
        "rift",
        "calls out teammates",
        "sideline altercation",
        "war of words",
    ),
    "coach_job_security_flag": (
        "fired",
        "hot seat",
        "job security",
        "relieved of his duties",
        "interim head coach",
        "steps down",
    ),
}

_TERMS_LONGEST_FIRST: tuple[str, ...] = tuple(sorted(LEXICON, key=len, reverse=True))
_TOKEN_RE = re.compile(r"[a-z0-9'/-]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Match:
    term: str
    category: Category
    weight: float
    negated: bool

    @property
    def signed_weight(self) -> float:
        return -self.weight if self.negated else self.weight


def _is_negated(text: str, start: int) -> bool:
    """True when a negation cue sits within :data:`NEGATION_WINDOW` tokens before."""
    preceding = _tokenize(text[:start])[-NEGATION_WINDOW:]
    return any(token in NEGATION_CUES for token in preceding)


def find_matches(text: str) -> list[Match]:
    """Longest-match-first lexicon scan with per-match negation detection."""
    lowered = (text or "").lower()
    consumed = [False] * len(lowered)
    matches: list[Match] = []
    for term in _TERMS_LONGEST_FIRST:
        start = lowered.find(term)
        while start != -1:
            end = start + len(term)
            if not any(consumed[start:end]):
                for index in range(start, end):
                    consumed[index] = True
                category, weight = LEXICON[term]
                matches.append(
                    Match(
                        term=term,
                        category=category,
                        weight=weight,
                        negated=_is_negated(lowered, start),
                    )
                )
            start = lowered.find(term, end)
    return matches


def score_headline(text: str) -> DocumentScore:
    """Score one headline -> signed score in ``[-1, 1]`` plus its category.

    Aggregation is "strongest match, softly reinforced": the largest-magnitude
    hit sets the level and the remaining hits nudge it, so a headline naming
    five injuries does not saturate at -1 for that reason alone.
    """
    matches = find_matches(text)
    if not matches:
        return DocumentScore(score=0.0, category="neutral", confidence=0.1)

    non_market = [m for m in matches if m.category != "market"]
    if not non_market:
        return DocumentScore(
            score=0.0,
            category="market",
            confidence=0.9,
            matched_terms=tuple(m.term for m in matches),
        )

    strongest = max(non_market, key=lambda m: abs(m.signed_weight))
    others = [m for m in non_market if m is not strongest]
    score = strongest.signed_weight + 0.25 * sum(m.signed_weight for m in others)

    flags = {
        flag
        for flag, terms in FLAG_TERMS.items()
        if any(m.term in terms and not m.negated for m in matches)
    }
    confidence = clamp(0.35 + 0.15 * len(non_market), 0.0, 1.0)
    return DocumentScore(
        score=round(clamp(score), 4),
        category=strongest.category,
        confidence=round(confidence, 4),
        matched_terms=tuple(m.term for m in matches),
        negated=strongest.negated,
        flags=frozenset(flags),
    )


@dataclass(frozen=True, slots=True)
class NewsSummary:
    """Aggregate of every headline in one window for one team."""

    n_documents: int
    news_sentiment_score: float
    sentiment_volatility: float
    negative_headline_count: int
    positive_headline_count: int
    suspension_mentioned: bool = False
    holdout_mentioned: bool = False
    locker_room_discord_flag: bool = False
    coach_job_security_flag: bool = False
    confidence: float = 0.0
    category_counts: dict[str, int] | None = None


#: Headlines below this magnitude are noise, not signal.
NEUTRAL_BAND = 0.05


def score_headlines(texts: Sequence[str]) -> NewsSummary:
    """Score a window of headlines and roll them up."""
    scores = [score_headline(text) for text in texts]
    return summarize(scores)


def summarize(scores: Sequence[DocumentScore]) -> NewsSummary:
    """Roll :class:`DocumentScore` objects into a :class:`NewsSummary`."""
    counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    for score in scores:
        counts[score.category] = counts.get(score.category, 0) + 1

    signals = [s.score for s in scores if abs(s.score) > NEUTRAL_BAND]
    mean = sum(signals) / len(signals) if signals else 0.0
    volatility = statistics.pstdev(signals) if len(signals) > 1 else 0.0
    flags: set[str] = set()
    for score in scores:
        flags |= set(score.flags)

    return NewsSummary(
        n_documents=len(scores),
        news_sentiment_score=round(clamp(mean), 4),
        sentiment_volatility=round(volatility, 4),
        negative_headline_count=sum(1 for s in signals if s < 0),
        positive_headline_count=sum(1 for s in signals if s > 0),
        suspension_mentioned="suspension_mentioned" in flags,
        holdout_mentioned="holdout_mentioned" in flags,
        locker_room_discord_flag="locker_room_discord_flag" in flags,
        coach_job_security_flag="coach_job_security_flag" in flags,
        confidence=round(clamp(len(signals) / 8.0, 0.0, 1.0), 4),
        category_counts=counts,
    )


class RuleBasedScorer:
    """The default scorer. Implements :class:`~nfl.data.sentiment.base.DocumentScorer`."""

    name = SCORER_NAME
    version = SCORER_VERSION

    @property
    def enabled(self) -> bool:
        return True

    def score_document(self, document: Document) -> DocumentScore:
        return score_headline(document.text)

    def score_documents(self, documents: list[Document]) -> list[DocumentScore]:
        return [self.score_document(document) for document in documents]


__all__ = [
    "FLAG_TERMS",
    "LEXICON",
    "NEGATION_CUES",
    "NEGATION_WINDOW",
    "NEUTRAL_BAND",
    "SCORER_NAME",
    "SCORER_VERSION",
    "Match",
    "NewsSummary",
    "RuleBasedScorer",
    "find_matches",
    "score_headline",
    "score_headlines",
    "summarize",
]
