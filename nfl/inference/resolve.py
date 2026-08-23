"""Turn whatever a human typed into one nflverse ``game_id``.

``nfl predict 2025_01_DAL_PHI`` is the machine spelling of a game. Nobody
remembers it. This module accepts the spellings people actually use and returns
a :class:`GameRef` that says which game was picked *and how*, so the CLI can
print the resolution rather than silently guessing:

===========================  =========================================
``2025_01_DAL_PHI``          nflverse id -- validated, then passed through
``DAL@PHI`` / ``DAL at PHI`` left is **away**, right is **home**
``PHI vs DAL``               left is **home**, right is **away**
``AWAY@PHI``                 wildcard side: PHI's next home game
``PHI`` / ``Eagles`` /       one team -> that team's game this week
``Philadelphia``
``KXNFLGAME-26AUG29CHITEN``  Kalshi event (or market) ticker
``401772510``                ESPN event id, via the schedules ``espn`` column
===========================  =========================================

Everything is case-insensitive. Team words come from
``nfl/data/reference/team_aliases.csv`` -- abbreviation, nflverse and ESPN
abbreviations, city, nickname, full name, plus the common shorthands
("Niners", "Bucs", "Jags", "Pats", "Skins"). A word that names two teams
("New York", "Los Angeles") is a hard error listing both, never a coin flip.

**"This week"** means the game whose kickoff is nearest to now within ±7 days.
Inside that window an upcoming or in-progress game beats one that just
finished, because "predict the Eagles game" on a Sunday night almost never
means the 1pm game that is already over -- unless ``--at`` was given, which is
by definition a request to rewind, and then nearest-in-time wins outright.
Two games in the window (a Thursday/Monday week boundary) resolves to the
nearest and *says so*. A matchup that recurs outside the window -- divisional
opponents play twice -- is refused until ``--week``/``--season`` disambiguates.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import date as date_type
from functools import lru_cache
from pathlib import Path

import polars as pl

from nfl.data.live.base import load_schedule_frame
from nfl.data.market.kalshi import parse_event_ticker
from nfl.data.reference import TEAM_ALIASES_CSV

__all__ = [
    "ACCEPTED_FORMATS",
    "WINDOW_DAYS",
    "GameRef",
    "GameSpecError",
    "Team",
    "resolve_game",
    "resolve_team",
    "teams",
]

#: Half-width of the "this week" window, in days.
WINDOW_DAYS = 7

#: How long after kickoff a game still counts as "not yet finished" when the
#: resolver prefers upcoming games. Four hours covers overtime and a TV window.
_GAME_LENGTH = timedelta(hours=4)

#: Kickoff slot used when ``gametime`` is missing or malformed -- the modal
#: slot, matching :func:`nfl.features.team._kickoff_expr`.
_DEFAULT_KICKOFF_ET = "13:00"
_SCHEDULE_TZ = "America/New_York"

ACCEPTED_FORMATS = """accepted game specs (case-insensitive):
  2025_01_DAL_PHI              nflverse game id
  DAL@PHI, DAL at PHI          away @ home
  PHI vs DAL                   home vs away
  AWAY@PHI                     wildcard side: PHI's game, at home
  PHI / Eagles / Philadelphia  one team: that team's game this week
  KXNFLGAME-25SEP04DALPHI      Kalshi event or market ticker
  401772510                    ESPN event id
narrow an ambiguous spec with --season / --week."""


class GameSpecError(ValueError):
    """A game spec that could not be turned into exactly one game."""

    def __init__(self, message: str, *, formats: bool = True) -> None:
        super().__init__(f"{message}\n{ACCEPTED_FORMATS}" if formats else message)


# --------------------------------------------------------------------------- #
# The team table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Team:
    """One row of ``team_aliases.csv``. ``abbr`` is the nflverse abbreviation."""

    abbr: str
    common: str
    espn: str
    city: str
    nickname: str
    full_name: str
    aliases: tuple[str, ...]

    def words(self) -> tuple[str, ...]:
        """Every string that should resolve to this team."""
        return (
            self.abbr,
            self.common,
            self.espn,
            self.city,
            self.nickname,
            self.full_name,
            *self.aliases,
        )


def _key(text: str) -> str:
    """``"St. Louis Rams"`` -> ``"stlouisrams"``: the lookup normal form."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@lru_cache(maxsize=1)
def teams(path: str | None = None) -> tuple[Team, ...]:
    """The 32 teams, read from ``nfl/data/reference/team_aliases.csv``."""
    target = Path(path) if path is not None else TEAM_ALIASES_CSV
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[Team] = []
    for row in rows:
        raw = (row.get("aliases") or "").strip()
        out.append(
            Team(
                abbr=(row["nflverse"] or "").strip().upper(),
                common=(row["abbr"] or "").strip().upper(),
                espn=(row["espn"] or "").strip().upper(),
                city=(row["city"] or "").strip(),
                nickname=(row["nickname"] or "").strip(),
                full_name=(row["full_name"] or "").strip(),
                aliases=tuple(a.strip() for a in raw.split("|") if a.strip()),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def _index() -> dict[str, frozenset[str]]:
    """Normalised word -> the set of teams it names (>1 means ambiguous)."""
    index: dict[str, set[str]] = {}
    for team in teams():
        for word in team.words():
            if word:
                index.setdefault(_key(word), set()).add(team.abbr)
    return {k: frozenset(v) for k, v in index.items()}


def resolve_team(text: str) -> str:
    """``"niners"`` -> ``"SF"``. Raises :class:`GameSpecError` if unknown or ambiguous.

    Ambiguity is a refusal, not a guess: "New York" names two franchises and
    picking one of them silently is how a prediction ends up logged against the
    wrong game.
    """
    hits = _index().get(_key(text))
    if not hits:
        raise GameSpecError(f"unknown team {text.strip()!r}")
    if len(hits) > 1:
        options = ", ".join(f"{t.abbr} ({t.full_name})" for t in teams() if t.abbr in hits)
        raise GameSpecError(f"{text.strip()!r} names more than one team: {options}", formats=False)
    return next(iter(hits))


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GameRef:
    """One game, plus a record of how the spec got there.

    Distinct from :class:`nfl.data.live.base.GameRef`, which is a feed-level
    pydantic model addressing a game in ESPN *or* nflverse terms. This one is
    the resolver's answer: an nflverse ``game_id`` that is known to exist in
    the schedules table, and the provenance the CLI prints.
    """

    game_id: str
    season: int
    week: int
    game_type: str | None
    away_team: str
    home_team: str
    kickoff: datetime | None
    espn: str | None
    spec: str
    how: str
    note: str | None = None
    alternatives: tuple[str, ...] = ()

    @property
    def matchup(self) -> str:
        return f"{self.away_team}@{self.home_team}"

    @property
    def when(self) -> str:
        if self.kickoff is None:
            return "unknown"
        return f"{self.kickoff:%Y-%m-%d %H:%M} UTC"

    def describe(self) -> str:
        """The one-line resolution the CLI prints above the panel.

        No square brackets anywhere: this string is handed to ``rich``, which
        would read ``[team]`` as a style tag and swallow it.
        """
        kind = "week" if (self.game_type or "REG") == "REG" else f"{self.game_type} week"
        return (
            f"resolved {self.spec!r} -> {self.game_id} "
            f"({self.away_team} at {self.home_team}, {self.season} {kind} {self.week}, "
            f"kickoff {self.when}) via {self.how}"
        )


# --------------------------------------------------------------------------- #
# Schedules helpers
# --------------------------------------------------------------------------- #

_REQUIRED = ("game_id", "season", "week", "away_team", "home_team")


def _kickoff_expr(frame: pl.DataFrame) -> pl.Expr:
    """``gameday`` + ``gametime`` (Eastern, DST-aware) as a naive-UTC timestamp."""
    if "gameday" not in frame.columns:
        return pl.lit(None, dtype=pl.Datetime("us")).alias("_kickoff")
    day = pl.col("gameday").cast(pl.String).str.slice(0, 10)
    if "gametime" in frame.columns:
        raw = pl.col("gametime").cast(pl.String)
        clock = (
            pl.when(raw.is_null() | (raw.str.len_chars() < 4))
            .then(pl.lit(_DEFAULT_KICKOFF_ET))
            .otherwise(raw.str.slice(0, 5))
        )
    else:
        clock = pl.lit(_DEFAULT_KICKOFF_ET)
    return (
        pl.concat_str([day, clock], separator=" ")
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone(_SCHEDULE_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .alias("_kickoff")
    )


def _prepare(source: pl.DataFrame | str | Path | None) -> pl.DataFrame:
    frame = load_schedule_frame(source)
    missing = [c for c in _REQUIRED if c not in frame.columns]
    if missing:
        raise GameSpecError(f"schedules frame is missing columns: {missing}", formats=False)
    return frame.with_columns(_kickoff_expr(frame))


def _as_naive_utc(when: datetime | date_type | None) -> datetime:
    if when is None:
        return datetime.now(UTC).replace(tzinfo=None)
    if isinstance(when, datetime):
        return when.astimezone(UTC).replace(tzinfo=None) if when.tzinfo else when
    return datetime(when.year, when.month, when.day, 12, 0)


def _row_to_ref(row: dict[str, object], *, spec: str, how: str, **extra: object) -> GameRef:
    espn = row.get("espn")
    espn_id = None if espn is None else str(espn).split(".")[0].strip() or None
    return GameRef(
        game_id=str(row["game_id"]),
        season=int(row["season"]),  # type: ignore[arg-type]
        week=int(row["week"]),  # type: ignore[arg-type]
        game_type=None if row.get("game_type") is None else str(row["game_type"]),
        away_team=str(row["away_team"]),
        home_team=str(row["home_team"]),
        kickoff=row.get("_kickoff"),  # type: ignore[arg-type]
        espn=None if espn_id in ("", "None", "nan") else espn_id,
        spec=spec,
        how=how,
        **extra,  # type: ignore[arg-type]
    )


def _summarise(frame: pl.DataFrame, limit: int = 6, *, by: str = "game_id") -> str:
    """A short, deterministic list of candidate games for an error message."""
    rows = frame.sort(by).head(limit).iter_rows(named=True)
    listed = ", ".join(f"{r['game_id']} ({r['season']} wk {r['week']})" for r in rows)
    if frame.height > limit:
        listed += f", ... (+{frame.height - limit} more)"
    return listed


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #

_GAME_ID_RE = re.compile(r"^(\d{4})_(\d{1,2})_([A-Za-z]{2,4})_([A-Za-z]{2,4})$")
_ESPN_RE = re.compile(r"^\d{5,}$")
_KALSHI_RE = re.compile(r"^KX[A-Z0-9]+-", re.IGNORECASE)
_MATCHUP_RE = re.compile(
    r"^(?P<left>.+?)\s*(?:(?P<at>@|\bat\b)|(?P<vs>\bvs?\.?\b|\bversus\b))\s*(?P<right>.+)$",
    re.IGNORECASE,
)

#: Side words that mean "whoever the opponent is".
_WILDCARDS = frozenset({"away", "home", "any", "anyone", "opp", "opponent", "x", "tbd", "*", "?"})


def _side(text: str) -> str | None:
    """A matchup side: a team abbreviation, or ``None`` for a wildcard."""
    stripped = text.strip()
    if not stripped or stripped in ("*", "?") or _key(stripped) in _WILDCARDS:
        return None
    return resolve_team(stripped)


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def resolve_game(
    spec: str,
    *,
    schedules: pl.DataFrame | str | Path | None = None,
    today: datetime | date_type | None = None,
    season: int | None = None,
    week: int | None = None,
    prefer_upcoming: bool = True,
) -> GameRef:
    """Resolve a human game spec to exactly one scheduled game.

    Args:
        spec: any format listed in :data:`ACCEPTED_FORMATS`.
        schedules: a frame, a parquet path, or ``None`` for the usual source.
        today: the clock for "this week"; defaults to now (UTC).
        season, week: narrow the candidates before the window is applied.
        prefer_upcoming: inside the ±7-day window, prefer a game that has not
            finished. Pass ``False`` when the caller asked for a past moment
            (``--at``), which makes nearest-in-time the only rule.

    Returns:
        A :class:`GameRef`. ``note`` is set whenever a choice was made between
        several candidates, and callers are expected to print it.

    Raises:
        GameSpecError: unparseable, unknown, or still ambiguous.
    """
    text = spec.strip()
    if not text:
        raise GameSpecError("empty game spec")
    frame = _prepare(schedules)
    ref_time = _as_naive_utc(today)

    match = _GAME_ID_RE.match(text)
    if match:
        return _resolve_game_id(match, text, frame)
    if _ESPN_RE.match(text):
        return _resolve_espn(text, frame)
    swapped: pl.DataFrame | None = None
    if _KALSHI_RE.match(text):
        candidates, how, ref_time = _kalshi_candidates(text, frame, ref_time)
    else:
        matchup = _MATCHUP_RE.match(text)
        if matchup:
            candidates, swapped = _matchup_candidates(matchup, frame)
            how = "matchup"
        else:
            team = resolve_team(text)
            candidates = frame.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
            how = "team"

    return _choose(
        candidates,
        spec=text,
        how=how,
        ref_time=ref_time,
        season=season,
        week=week,
        prefer_upcoming=prefer_upcoming,
        swapped=swapped,
    )


def _resolve_game_id(match: re.Match[str], text: str, frame: pl.DataFrame) -> GameRef:
    year, wk, away, home = match.groups()
    game_id = f"{year}_{int(wk):02d}_{resolve_team(away)}_{resolve_team(home)}"
    row = frame.filter(pl.col("game_id") == game_id)
    if row.height == 0:
        raise GameSpecError(
            f"{text!r} looks like an nflverse game id but {game_id} is not in the schedules table",
            formats=False,
        )
    note = None if game_id == text.upper() else f"normalised {text!r} to {game_id}"
    return _row_to_ref(row.row(0, named=True), spec=text, how="game_id", note=note)


def _resolve_espn(text: str, frame: pl.DataFrame) -> GameRef:
    if "espn" not in frame.columns:
        raise GameSpecError("the schedules table has no 'espn' column", formats=False)
    row = frame.filter(
        pl.col("espn").cast(pl.String).str.strip_chars().str.replace(r"\.0$", "") == text
    )
    if row.height == 0:
        raise GameSpecError(f"no scheduled game carries ESPN event id {text}")
    return _row_to_ref(row.row(0, named=True), spec=text, how="espn")


def _kalshi_candidates(
    text: str, frame: pl.DataFrame, ref_time: datetime
) -> tuple[pl.DataFrame, str, datetime]:
    """A Kalshi event *or* market ticker -> the games it could name.

    ``KXNFLGAME-26AUG29CHITEN-TEN`` is a market inside event
    ``KXNFLGAME-26AUG29CHITEN``; the trailing team segment is dropped before
    :func:`~nfl.data.market.kalshi.parse_event_ticker` sees it, because that
    function splits the pair out of the last segment and a market suffix would
    defeat it. The ticker's date becomes the clock, so a ticker resolves the
    same way tomorrow as it does today.
    """
    parts = text.upper().split("-")
    event = "-".join(parts[:2])
    kickoff_date, away, home = parse_event_ticker(event)
    if away is None or home is None:
        raise GameSpecError(
            f"cannot read a team pair out of Kalshi ticker {text!r}; "
            "use the matchup or the nflverse game id instead",
            formats=False,
        )
    candidates = frame.filter((pl.col("away_team") == away) & (pl.col("home_team") == home))
    if kickoff_date is not None:
        ref_time = datetime(kickoff_date.year, kickoff_date.month, kickoff_date.day, 12, 0)
    return candidates, "kalshi", ref_time


def _matchup_candidates(
    match: re.Match[str], frame: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """``(games in the stated orientation, games in the other one)``.

    The second frame is the home/away swap. It is a fallback, not a candidate:
    it is used only when the stated orientation yields nothing, because a user
    who writes ``PHI vs DAL`` for a game Dallas hosts has made a recoverable
    mistake, and refusing a matchup that plainly exists helps nobody. When it
    is used the caller says so in the resolution line.
    """
    left, right = _side(match["left"]), _side(match["right"])
    away, home = (left, right) if match["at"] else (right, left)
    if away is None and home is None:
        raise GameSpecError(f"{match.string.strip()!r} names no team on either side")
    if away is not None and away == home:
        raise GameSpecError(f"{match.string.strip()!r} is a team against itself", formats=False)

    exprs = []
    if away is not None:
        exprs.append(pl.col("away_team") == away)
    if home is not None:
        exprs.append(pl.col("home_team") == home)
    oriented = frame.filter(pl.all_horizontal(exprs))
    if away is None or home is None:
        return oriented, None
    swapped = frame.filter((pl.col("away_team") == home) & (pl.col("home_team") == away))
    return oriented, swapped


def _choose(
    candidates: pl.DataFrame,
    *,
    spec: str,
    how: str,
    ref_time: datetime,
    season: int | None,
    week: int | None,
    prefer_upcoming: bool,
    swapped: pl.DataFrame | None = None,
) -> GameRef:
    """Narrow by season/week, then by the ±7-day window, then by nearest kickoff."""

    def narrow(frame: pl.DataFrame) -> pl.DataFrame:
        if season is not None:
            frame = frame.filter(pl.col("season") == season)
        if week is not None:
            frame = frame.filter(pl.col("week") == week)
        return frame

    pool = narrow(candidates)
    note: str | None = None
    if pool.height == 0 and swapped is not None:
        pool = narrow(swapped)
        if pool.height:
            candidates = swapped
            how = f"{how} (home/away swapped)"
            note = f"no game matches {spec!r} as written; using the home/away swap"

    if pool.height == 0:
        scope = "".join(
            [
                f" in season {season}" if season is not None else "",
                f" week {week}" if week is not None else "",
            ]
        )
        if candidates.height and scope:
            raise GameSpecError(
                f"no game matches {spec!r}{scope}; without that filter it matches: "
                f"{_summarise(candidates)}",
                formats=False,
            )
        raise GameSpecError(f"no scheduled game matches {spec!r}")

    if pool.height > 1:
        pool, chosen_note = _nearest(
            pool, spec=spec, ref_time=ref_time, prefer_upcoming=prefer_upcoming
        )
        note = "; ".join(n for n in (note, chosen_note) if n) or None

    row = pool.row(0, named=True)
    others = tuple(
        str(g) for g in candidates.get_column("game_id").to_list() if str(g) != str(row["game_id"])
    )
    return _row_to_ref(row, spec=spec, how=how, note=note, alternatives=others[:8])


def _nearest(
    pool: pl.DataFrame, *, spec: str, ref_time: datetime, prefer_upcoming: bool
) -> tuple[pl.DataFrame, str | None]:
    dated = pool.filter(pl.col("_kickoff").is_not_null())
    if dated.height == 0:
        raise GameSpecError(
            f"{spec!r} matches {pool.height} games and none of them has a kickoff time; "
            f"pass --week / --season. Candidates: {_summarise(pool)}",
            formats=False,
        )
    dated = dated.with_columns(
        (pl.col("_kickoff") - pl.lit(ref_time)).dt.total_seconds().abs().alias("_delta")
    )
    window = dated.filter(pl.col("_delta") <= WINDOW_DAYS * 86400)
    if window.height == 0:
        raise GameSpecError(
            f"{spec!r} matches {dated.height} games, none within "
            f"±{WINDOW_DAYS} days of {ref_time:%Y-%m-%d %H:%M} UTC; "
            f"pass --week / --season. Nearest: {_summarise(dated, by='_delta')}",
            formats=False,
        )

    considered = window
    if prefer_upcoming:
        live = window.filter(pl.col("_kickoff") >= pl.lit(ref_time - _GAME_LENGTH))
        if live.height:
            considered = live
    ordered = considered.sort(["_delta", "game_id"])
    picked = ordered.head(1)

    if window.height == 1:
        return picked, None
    rejected = ", ".join(
        f"{r['game_id']} ({r['_kickoff']:%Y-%m-%d %H:%M} UTC)"
        for r in window.sort(["_delta", "game_id"]).iter_rows(named=True)
        if str(r["game_id"]) != str(picked.get_column("game_id").item(0))
    )
    chosen = picked.row(0, named=True)
    note = (
        f"{window.height} games match {spec!r} within ±{WINDOW_DAYS} days of "
        f"{ref_time:%Y-%m-%d %H:%M} UTC; chose {chosen['game_id']} "
        f"({chosen['_kickoff']:%Y-%m-%d %H:%M} UTC) over {rejected}"
    )
    return picked, note
