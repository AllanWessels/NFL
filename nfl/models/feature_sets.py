"""Named feature sets — which of the 775 dataset columns a model is allowed to see.

ADR 0003 settles the architecture question the Phase 5 ablation asked: the
500+ game-constant ``tm__``/``pl__``/``ctx__`` columns do not earn their keep,
they let a learner memorise *which game* a row belongs to (~3.4k independent
games wearing 622k rows). So v1 trains on the live in-play state plus a
deliberately tiny pre-game prior, and the rest of the table stays in the
dataset for research only.

Four names, resolved against whatever columns a frame actually has:

``inplay``
    Everything prefixed ``bi__`` (the 25 scoreboard/situation columns) or
    ``ig__`` (the 220 running in-game aggregates). 245 columns.
``prior``
    :data:`PRIOR_COLUMNS` — ten pre-game numbers, listed and justified below.
``v1``
    ``inplay + prior``. The ADR 0003 model input.
``full``
    Every feature column on the frame; the Phase 5 baseline, kept so the
    v1-vs-full comparison can still be run.

Selection is *intersection*, never assertion: a name missing from the frame is
skipped rather than raised on, because the dataset gains columns between
builds and a v1 model must stay trainable on an older parquet. What you get
back is always in the order declared here, so two frames produce the same
design-matrix column order.

    >>> import polars as pl
    >>> df = pl.DataFrame({"bi__down": [1.0], "ctx__is_indoors": [1], "meta__season": [2024]})
    >>> feature_set_columns(df, "v1")
    ['bi__down', 'ctx__is_indoors']
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nfl.features.assemble import feature_columns

__all__ = [
    "FEATURE_SET_NAMES",
    "INPLAY_PREFIXES",
    "PRIOR_COLUMNS",
    "feature_set_columns",
    "inplay_columns",
    "prior_columns",
    "select_features",
]

#: The two families that describe the live state of the game in progress.
#: ``bi__`` is the scoreboard a fan reads; ``ig__`` is everything accumulated
#: since kickoff (drive counts, EPA totals, success rates, ...).
INPLAY_PREFIXES: tuple[str, ...] = ("bi__", "ig__")

#: The compressed pre-game prior: ten columns, one per idea, chosen to say
#: "how good are these two teams and what are the conditions" without handing
#: the model anything it could use to identify the game.
#:
#: ===================================  ============================================
#: column                               why it is here
#: ===================================  ============================================
#: ``home_tm__elo_win_prob``            Elo's own P(home win) — the ADR's "Elo
#:                                      win-prob diff" in its already-differenced,
#:                                      already-squashed form (the away column is
#:                                      its complement, so both would be one
#:                                      feature twice).
#: ``home_tm__elo_diff``                Elo rating differential, home - away. The
#:                                      unsquashed version: a GBDT can re-shape it,
#:                                      the logistic prior above cannot.
#: ``home_tm__power_rating_diff``       Power-rating differential, the second and
#:                                      independent team-strength estimate.
#: ``ctx__rest_diff_home``              Rest-day differential (short week, bye).
#: ``home_pl__qb1_adj_epa_per_dropback``  QB1 strength, home side.
#: ``away_pl__qb1_adj_epa_per_dropback``  QB1 strength, away side.
#: ``ctx__is_indoors``                  Roof/indoor flag.
#: ``ctx__is_artificial_turf``          Surface, the other venue constant.
#: ``ctx__div_game``                    Divisional matchups run closer than the
#:                                      ratings say.
#: ``ctx__is_postseason``               Postseason games are a different
#:                                      population (no rest-day noise, no tanking).
#: ===================================  ============================================
#:
#: **On the QB1 slot.** ADR 0003 asks for "QB1 availability both sides".
#: ``home_pl__qb1_available`` and ``home_pl__qb1_availability_prob`` exist in the
#: schema but are **100% null in every season of the current dataset build** —
#: the injury/availability join has not landed — so selecting them would add two
#: constant columns and nothing else. ``qb1_adj_epa_per_dropback`` (9.9% null,
#: std 0.116) is the populated stand-in: it carries "who is under centre and how
#: good are they", which is what the availability flag was a proxy for. Swap the
#: two names here once the availability join is real.
PRIOR_COLUMNS: tuple[str, ...] = (
    "home_tm__elo_win_prob",
    "home_tm__elo_diff",
    "home_tm__power_rating_diff",
    "ctx__rest_diff_home",
    "home_pl__qb1_adj_epa_per_dropback",
    "away_pl__qb1_adj_epa_per_dropback",
    "ctx__is_indoors",
    "ctx__is_artificial_turf",
    "ctx__div_game",
    "ctx__is_postseason",
)

#: Every name :func:`feature_set_columns` accepts.
FEATURE_SET_NAMES: tuple[str, ...] = ("inplay", "prior", "v1", "full")


def _features(df: pl.DataFrame) -> list[str]:
    """Feature columns of ``df`` — no ``meta__``, no ``label__``."""
    return feature_columns(df)


def inplay_columns(df: pl.DataFrame) -> list[str]:
    """The ``bi__`` + ``ig__`` columns present on ``df``, in frame order."""
    return [c for c in _features(df) if c.startswith(INPLAY_PREFIXES)]


def prior_columns(df: pl.DataFrame) -> list[str]:
    """The :data:`PRIOR_COLUMNS` present on ``df``, in the order declared above."""
    have = set(df.columns)
    return [c for c in PRIOR_COLUMNS if c in have]


def feature_set_columns(df: pl.DataFrame, name: str) -> list[str]:
    """Column names of the named feature set, resolved against ``df``.

    Raises ``ValueError`` on an unknown name (never silently returns nothing)
    and on a name that resolves to no columns at all, which always means the
    frame is not an assembled dataset.
    """
    key = str(name).strip().lower()
    if key not in FEATURE_SET_NAMES:
        raise ValueError(f"unknown feature set {name!r}; expected one of {list(FEATURE_SET_NAMES)}")
    if key == "inplay":
        cols = inplay_columns(df)
    elif key == "prior":
        cols = prior_columns(df)
    elif key == "v1":
        cols = inplay_columns(df) + prior_columns(df)
    else:
        cols = _features(df)
    if not cols:
        raise ValueError(
            f"feature set {key!r} matched no columns on a {df.width}-column frame; "
            "is this an assembled dataset?"
        )
    return cols


def select_features(df: pl.DataFrame, name: str = "v1") -> pl.DataFrame:
    """``df`` narrowed to the named feature set — a design matrix, ready for ``fit``.

    ``meta__``/``label__`` columns are dropped by construction, so the result
    passes :func:`nfl.models.artifacts.assert_model_input_frame`. Keep the
    original frame around if you still need the meta columns (the Q4 slicing
    and the sequence windowing both do).
    """
    return df.select(feature_set_columns(df, name))


def describe(df: pl.DataFrame, names: Sequence[str] = FEATURE_SET_NAMES) -> pl.DataFrame:
    """``(feature_set, n_columns)`` for each name — what a manifest records."""
    rows = []
    for name in names:
        try:
            n = len(feature_set_columns(df, name))
        except ValueError:
            n = 0
        rows.append({"feature_set": name, "n_columns": n})
    return pl.DataFrame(rows, schema={"feature_set": pl.String, "n_columns": pl.Int64})
