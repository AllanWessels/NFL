"""Named feature sets — which of the 775 dataset columns a model is allowed to see.

ADR 0003 settles the architecture question the Phase 5 ablation asked: the
500+ game-constant ``tm__``/``pl__``/``ctx__`` columns do not earn their keep,
they let a learner memorise *which game* a row belongs to (~3.4k independent
games wearing 622k rows). So v1 trains on the live in-play state plus a
deliberately tiny pre-game prior, and the rest of the table stays in the
dataset for research only.

Six names, resolved against whatever columns a frame actually has:

``inplay``
    Everything prefixed ``bi__`` (the 25 scoreboard/situation columns) or
    ``ig__`` (the 220 running in-game aggregates). 245 columns.
``servable``
    ``inplay`` minus the columns the serving path cannot produce — see
    :data:`UNSERVABLE_COLUMNS`. **156 columns**, and the v1 default: a model
    trained on ``inplay`` is trained on 89 columns that arrive null at serve
    time, which is a different model than the one that was measured.
``prior``
    :data:`PRIOR_COLUMNS` — ten pre-game numbers, listed and justified below.
``servable_prior``
    ``prior`` minus the two QB1 columns the serving path rebuilds differently
    (see :data:`UNSERVABLE_PRIOR_COLUMNS`). Eight columns.
``servable_v1``
    ``servable + servable_prior``. Everything v1 is allowed to see *and* can
    actually be served, so the in-play block keeps its pre-game context.
``v1``
    ``inplay + prior``. The ADR 0003 model input as first written, kept so
    artifacts that recorded ``feature_set="v1"`` keep resolving to the columns
    they were trained on. New training runs use ``servable`` /
    ``servable_v1`` — :data:`V1_FEATURE_SET`.
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
    >>> feature_set_columns(df, "servable")
    ['bi__down']
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nfl.features.assemble import feature_columns
from nfl.features.ingame import SNAPSHOT_UNKNOWABLE_COLUMNS

__all__ = [
    "FEATURE_SET_NAMES",
    "INPLAY_PREFIXES",
    "PRIOR_COLUMNS",
    "UNSERVABLE_CLOCK_COLUMNS",
    "UNSERVABLE_COLUMNS",
    "UNSERVABLE_PRIOR_COLUMNS",
    "V1_FEATURE_SET",
    "feature_set_columns",
    "inplay_columns",
    "prior_columns",
    "select_features",
    "servable_columns",
    "servable_prior_columns",
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

#: Clock-derived in-play columns excluded from ``servable`` because the serving
#: path cannot reproduce them.
#:
#: **Measured, not assumed.** 134 replay snapshots across 40 val-season games,
#: each taken at its own play's clock so no request-vs-snap gap is in play, were
#: compared column by column against the assembled dataset row for that play.
#: Of the 156 in-play columns that survive :data:`UNSERVABLE_COLUMNS`, exactly
#: one ever disagreed — ``bi__goal_to_go``, and only on administrative rows
#: (a timeout has no yard line) where the serving derivation returned null and
#: nflverse writes ``0``. That was a bug in
#: :func:`nfl.features.ingame.snapshot_play_frame`, not an irreproducible
#: column, and it is fixed; the set is therefore empty.
#:
#: Every column in :data:`nfl.features.ingame.SNAPSHOT_CLOCK_SENSITIVE_COLUMNS`
#: *is* servable: it differs from the training row only when the snapshot is
#: rewound to a time that is not a snap, and then it differs by the clock gap
#: itself. Excluding ``ig__game_seconds_remaining`` from a win-probability model
#: to avoid an 11-second gap would cost far more than the gap does; the gate
#: samples at the snap's own clock instead, where the gap is zero.
UNSERVABLE_CLOCK_COLUMNS: frozenset[str] = frozenset()

#: The ten :data:`PRIOR_COLUMNS` are not all servable. The two QB1 columns are
#: rebuilt at serve time from the snapshot's injury list by
#: :func:`nfl.features.player.roster_from_snapshot`, which is a different
#: computation from the as-of-kickoff offline one, and they disagreed on
#: **all** 134 sampled snapshots. The other eight — Elo, power rating, rest,
#: and the four venue/context flags — matched on every one.
UNSERVABLE_PRIOR_COLUMNS: frozenset[str] = frozenset(
    {
        "home_pl__qb1_adj_epa_per_dropback",
        "away_pl__qb1_adj_epa_per_dropback",
    }
)

#: The feature set new v1 training runs use. ``servable`` per ADR 0003 as
#: amended: in-play state only, and only the part of it that survives the trip
#: through a :class:`~nfl.data.live.base.LiveSnapshot`.
V1_FEATURE_SET = "servable"

#: In-play columns a model must not be trained on, because ``nfl predict``
#: cannot fill them.
#:
#: :data:`nfl.features.ingame.SNAPSHOT_UNKNOWABLE_COLUMNS` — 89 of the 245
#: in-play columns — is the bulk of it: per-play EPA, completions, first downs,
#: sacks, penalties, kick results. A :class:`~nfl.data.live.base.LivePlay`
#: carries a play's type, yardage, scores and coarse turnover/penalty flags and
#: nothing else, so every aggregate built on the detail is null at serve time
#: and populated in training. A GBDT trained on those 89 columns and served
#: 89 nulls takes a different path down every tree it learned; the measured
#: shift on the v1 XGBoost artifact was ~0.06 in probability.
UNSERVABLE_COLUMNS: frozenset[str] = SNAPSHOT_UNKNOWABLE_COLUMNS | UNSERVABLE_CLOCK_COLUMNS

#: Every name :func:`feature_set_columns` accepts.
FEATURE_SET_NAMES: tuple[str, ...] = (
    "inplay",
    "servable",
    "prior",
    "servable_prior",
    "v1",
    "servable_v1",
    "full",
)


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


def servable_columns(df: pl.DataFrame) -> list[str]:
    """The in-play columns the serving path actually produces, in frame order."""
    return [c for c in inplay_columns(df) if c not in UNSERVABLE_COLUMNS]


def servable_prior_columns(df: pl.DataFrame) -> list[str]:
    """:func:`prior_columns` minus :data:`UNSERVABLE_PRIOR_COLUMNS`."""
    return [c for c in prior_columns(df) if c not in UNSERVABLE_PRIOR_COLUMNS]


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
    elif key == "servable":
        cols = servable_columns(df)
    elif key == "prior":
        cols = prior_columns(df)
    elif key == "servable_prior":
        cols = servable_prior_columns(df)
    elif key == "v1":
        cols = inplay_columns(df) + prior_columns(df)
    elif key == "servable_v1":
        cols = servable_columns(df) + servable_prior_columns(df)
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
