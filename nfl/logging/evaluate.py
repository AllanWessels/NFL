"""Evaluation and reporting over the ledger (or any backtest frame).

Everything here consumes one frame in the ledger schema — live predictions read
back with :func:`nfl.logging.ledger.read_ledger`, or an offline backtest from
:func:`nfl.logging.backtest.backtest`. That is the whole point of the shared
schema: an offline sweep and a season of live logging are scored by the same
code, so "the backtest said 0.19" and "production says 0.23" are comparable
numbers rather than two different definitions of log loss.

What gets reported, per ``(model_name, model_version)``:

* **Overall metrics** — ``log_loss``, ``brier``, ``accuracy``, ``ece`` (10 bins),
  ``auc``.
* **By game phase** — bins on ``snapshot_game_seconds_remaining``, because a
  model that is excellent at 0:30 and mediocre at 30:00 is exactly what this
  project wants and a single averaged number hides it.
* **Q4 marks** — the state at 600 / 300 / 120 seconds remaining, one row per
  game (``nfl.features.splits.q4_at`` semantics).
* **Not collapsed** — rows where the market still sits inside
  ``[0.15, 0.85]``. The trade gate from the project brief.
* **Reliability table** — 10 equal-width probability bins, predicted vs actual.

**Edge vs market.** Market data enters here and *only* here (ADR 0001). The
comparison is against the de-vigged closing moneyline from
:mod:`nfl.data.market.closing_lines`, plus a Kalshi snapshot price when one was
recorded on the row. Three numbers come out of it:

1. **Brier skill score vs market** — ``1 - brier_model / brier_market`` on the
   subset where a market price exists. Positive means we beat the closing line.
2. **Flat-stake ROI** — hypothetical, one contract per qualifying bet, taken
   when ``|p_model - p_market| > threshold`` and the market has not collapsed.
   Reported gross and net of Kalshi trading fees.
3. **CLV** — closing line value: how the price moved between the snapshot we
   bet at and the close, signed by the side taken. Positive CLV with negative
   ROI is variance; negative CLV is a broken model.

**Retrain triggers.** :func:`retrain_trigger` compares a rolling window of
recent log loss against the metric the model reported at training time. Drift
beyond a threshold raises a flag; it does not retrain anything.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl import config
from nfl.logging import ledger as ledger_mod

log = logging.getLogger(__name__)

__all__ = [
    "COLLAPSE_BOUNDS",
    "DEFAULT_EDGE_THRESHOLD",
    "KALSHI_FEE_COEFFICIENT",
    "PHASE_BINS",
    "Q4_MARKS",
    "EdgeReport",
    "EvaluationReport",
    "MetricSet",
    "RetrainSignal",
    "SliceMetrics",
    "attach_closing_lines",
    "edge_report",
    "evaluate_frame",
    "kalshi_fee",
    "metrics",
    "reliability_table",
    "render_report",
    "report_path",
    "retrain_trigger",
    "write_markdown",
]

#: The market-collapse trade gate (mirrors ``nfl.features.splits.COLLAPSE_BOUNDS``).
COLLAPSE_BOUNDS: tuple[float, float] = (0.15, 0.85)

#: Minimum ``|p_model - p_market|`` before the hypothetical bettor acts.
DEFAULT_EDGE_THRESHOLD = 0.05

#: Q4 marks the project is judged at, in seconds remaining.
Q4_MARKS: tuple[int, ...] = (600, 300, 120)

#: ``(label, lo, hi)`` on ``game_seconds_remaining``, inclusive both ends.
PHASE_BINS: tuple[tuple[str, int, int], ...] = (
    ("Q1 (3600-2701)", 2701, 3600),
    ("Q2 (2700-1801)", 1801, 2700),
    ("Q3 (1800-901)", 901, 1800),
    ("Q4 (900-301)", 301, 900),
    ("Q4 last 5min (300-61)", 61, 300),
    ("Q4 last 1min (60-0)", 0, 60),
)

# --------------------------------------------------------------------------- #
# Kalshi fees
# --------------------------------------------------------------------------- #

#: Coefficient in Kalshi's published trading-fee formula
#: ``fee = ceil_to_cent(coefficient * contracts * price * (1 - price))``.
#:
#: **Verification status:** the fee schedule could not be re-fetched while this
#: module was written (the lookup agent exhausted its web budget and
#: ``kalshi.com`` rate-limited a direct fetch), so this is the long-standing
#: published general-series coefficient, 0.07, and it is a *constant you can
#: override* rather than a hard-coded literal. Re-check
#: <https://kalshi.com/docs/kalshi-fee-schedule.pdf> before quoting a net ROI
#: number outside this repo; ``--fee-coefficient`` on ``nfl evaluate`` exists
#: for exactly that. Maker orders and settlement are assumed free, which is the
#: general-series behaviour; the equities-index series has its own schedule and
#: does not apply to NFL game markets.
KALSHI_FEE_COEFFICIENT = 0.07


def kalshi_fee(
    price: float,
    contracts: float = 1.0,
    *,
    coefficient: float = KALSHI_FEE_COEFFICIENT,
    round_up: bool = True,
) -> float:
    """Kalshi trading fee in dollars for ``contracts`` at ``price``.

    ``fee = coefficient * contracts * price * (1 - price)``, rounded up to the
    next cent. The ``price * (1 - price)`` shape means the fee peaks at 50c and
    vanishes at the extremes — which is why it bites hardest on exactly the
    not-collapsed markets this project wants to trade.
    """
    raw = coefficient * float(contracts) * float(price) * (1.0 - float(price))
    if raw <= 0:
        return 0.0
    if not round_up:
        return raw
    # ``0.07 * 100 * 0.5 * 0.5`` is 1.7500000000000002 in binary floating point,
    # and a naive ceil would charge a cent that Kalshi does not. Snap to the
    # ninth decimal first so only a genuine fraction of a cent rounds up.
    return math.ceil(round(raw * 100.0, 9)) / 100.0


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

_EPS = 1e-15


def _arrays(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).ravel()
    p_arr = np.asarray(p, dtype=float).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p differ in shape: {y_arr.shape} vs {p_arr.shape}")
    keep = np.isfinite(y_arr) & np.isfinite(p_arr)
    return y_arr[keep], np.clip(p_arr[keep], _EPS, 1.0 - _EPS)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Ties (``y == 0.5``) count as half right, matching the 0.5 label."""
    return float(np.mean(1.0 - np.abs(np.round(p) - y)))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over ``bins`` equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            total += n * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total / max(len(p), 1)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC with tie handling; ``nan`` when one class is absent.

    Ties in the label (``y == 0.5``) are dropped: a drawn game is neither a
    positive nor a negative and there is no honest place to put it.
    """
    mask = (y == 0.0) | (y == 1.0)
    y_b, p_b = y[mask], p[mask]
    n_pos = int((y_b == 1.0).sum())
    n_neg = int((y_b == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p_b, kind="mergesort")
    ranks = np.empty(len(p_b), dtype=float)
    ranks[order] = np.arange(1, len(p_b) + 1, dtype=float)
    # Average ranks within tied probabilities so identical scores cannot be
    # ordered by accident of sort stability.
    sorted_p = p_b[order]
    start = 0
    for i in range(1, len(sorted_p) + 1):
        if i == len(sorted_p) or sorted_p[i] != sorted_p[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y_b == 1.0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True, slots=True)
class MetricSet:
    """The five headline numbers plus the support they were computed on."""

    n: int
    log_loss: float
    brier: float
    accuracy: float
    ece: float
    auc: float
    base_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "log_loss": self.log_loss,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "ece": self.ece,
            "auc": self.auc,
            "base_rate": self.base_rate,
        }


_EMPTY = MetricSet(0, *(float("nan"),) * 6)


def metrics(y: Any, p: Any, *, bins: int = 10) -> MetricSet:
    """log_loss / brier / accuracy / ece / auc for one ``(labels, probs)`` pair."""
    y_arr, p_arr = _arrays(y, p)
    if len(y_arr) == 0:
        return _EMPTY
    return MetricSet(
        n=len(y_arr),
        log_loss=_log_loss(y_arr, p_arr),
        brier=_brier(y_arr, p_arr),
        accuracy=_accuracy(y_arr, p_arr),
        ece=_ece(y_arr, p_arr, bins),
        auc=_auc(y_arr, p_arr),
        base_rate=float(np.mean(y_arr)),
    )


def reliability_table(y: Any, p: Any, *, bins: int = 10) -> pl.DataFrame:
    """Predicted vs realised frequency per probability bin — the calibration curve."""
    y_arr, p_arr = _arrays(y, p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    if len(p_arr):
        idx = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, bins - 1)
    else:
        idx = np.empty(0, dtype=int)
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f}{']' if b == bins - 1 else ')'}",
                "n": n,
                "mean_predicted": float(p_arr[mask].mean()) if n else None,
                "observed": float(y_arr[mask].mean()) if n else None,
                "gap": float(p_arr[mask].mean() - y_arr[mask].mean()) if n else None,
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #

PROB_COLUMN = "p_home"
LABEL_COLUMN = "resolved_home_win"
CLOCK_COLUMN = "snapshot_game_seconds_remaining"
MARKET_COLUMN = "market_implied_home"


def _labelled(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows with both a probability and a resolved outcome."""
    if LABEL_COLUMN not in frame.columns or PROB_COLUMN not in frame.columns:
        raise KeyError(f"frame needs {PROB_COLUMN!r} and {LABEL_COLUMN!r}; columns={frame.columns}")
    return frame.filter(pl.col(PROB_COLUMN).is_not_null() & pl.col(LABEL_COLUMN).is_not_null())


def _yp(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame.get_column(LABEL_COLUMN).cast(pl.Float64, strict=False).to_numpy()
    p = frame.get_column(PROB_COLUMN).cast(pl.Float64).to_numpy()
    return y, p


def _q4_at(frame: pl.DataFrame, seconds: int) -> pl.DataFrame:
    """One row per game: the last snapshot no later than ``seconds`` remaining."""
    if CLOCK_COLUMN not in frame.columns:
        return frame.head(0)
    cand = frame.filter(
        pl.col(CLOCK_COLUMN).is_not_null()
        & (pl.col(CLOCK_COLUMN) <= 900)
        & (pl.col(CLOCK_COLUMN) >= seconds)
    )
    if "snapshot_period" in cand.columns:
        cand = cand.filter(pl.col("snapshot_period").is_null() | (pl.col("snapshot_period") == 4))
    return cand.sort([CLOCK_COLUMN, "prediction_id"]).unique(
        subset=["game_id"], keep="first", maintain_order=True
    )


def _not_collapsed(
    frame: pl.DataFrame, bounds: tuple[float, float] = COLLAPSE_BOUNDS
) -> pl.DataFrame:
    """Rows the market has not already resolved. Rows with no price are dropped.

    Unlike :func:`nfl.features.splits.not_collapsed` this *requires* a price:
    the whole point of the slice is "games the market still thinks are live",
    and a missing price cannot answer that question either way.
    """
    if MARKET_COLUMN not in frame.columns:
        return frame.head(0)
    m = pl.col(MARKET_COLUMN).cast(pl.Float64, strict=False)
    return frame.filter(m.is_not_null() & m.is_between(bounds[0], bounds[1]))


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    """One named subset and its metrics."""

    name: str
    metrics: MetricSet


def evaluate_frame(
    frame: pl.DataFrame,
    *,
    bins: int = 10,
    q4_marks: Sequence[int] = Q4_MARKS,
    bounds: tuple[float, float] = COLLAPSE_BOUNDS,
) -> list[SliceMetrics]:
    """Overall + phase + Q4-mark + not-collapsed metrics for one model version."""
    labelled = _labelled(frame)
    out = [SliceMetrics("overall", metrics(*_yp(labelled), bins=bins))]

    if CLOCK_COLUMN in labelled.columns:
        for label, lo, hi in PHASE_BINS:
            part = labelled.filter(pl.col(CLOCK_COLUMN).is_between(lo, hi))
            if part.height:
                out.append(SliceMetrics(label, metrics(*_yp(part), bins=bins)))

    for mark in q4_marks:
        part = _q4_at(labelled, mark)
        if part.height:
            out.append(SliceMetrics(f"Q4@{mark}", metrics(*_yp(part), bins=bins)))

    nc = _not_collapsed(labelled, bounds)
    if nc.height:
        out.append(SliceMetrics("not collapsed", metrics(*_yp(nc), bins=bins)))
    return out


# --------------------------------------------------------------------------- #
# Market join
# --------------------------------------------------------------------------- #


def attach_closing_lines(
    frame: pl.DataFrame,
    market: pl.DataFrame | None = None,
    *,
    seasons: Sequence[int] | None = None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Join the de-vigged closing moneyline onto a ledger frame.

    ``closing_market_implied_home`` is always the close.
    ``market_implied_home`` is the price *at the snapshot*: a Kalshi price when
    one was recorded on the row (``kalshi_yes_price_home``), otherwise the
    closing line as the only available proxy. ``edge_vs_market`` is refreshed
    from whichever of those is in play.

    This is the one function in the codebase that puts market numbers next to
    model numbers, and it runs after prediction, never before (ADR 0001).
    """
    if frame.is_empty():
        return frame
    if market is None:
        from nfl.data.market import closing_lines

        wanted = seasons
        if wanted is None and "season" in frame.columns:
            wanted = sorted(
                int(s) for s in frame.get_column("season").drop_nulls().unique().to_list()
            )
        try:
            market = closing_lines.fetch_closing_lines(wanted, root=root)
        except (closing_lines.ClosingLinesNotFound, KeyError) as exc:
            log.warning("no closing lines available (%s); edge report will be empty", exc)
            market = None

    out = frame
    if market is not None and not market.is_empty() and "game_id" in market.columns:
        cols = [pl.col("game_id").cast(pl.String)]
        if "implied_prob_home" in market.columns:
            cols.append(pl.col("implied_prob_home").cast(pl.Float64).alias("_close_implied"))
        for src, dst in (("spread_close_home", "_close_spread"), ("book", "_book")):
            if src in market.columns:
                cols.append(pl.col(src).alias(dst))
        lookup = market.select(cols).unique(subset=["game_id"], keep="first")
        out = out.drop(
            [c for c in ("_close_implied", "_close_spread", "_book") if c in out.columns]
        )
        out = out.join(lookup, on="game_id", how="left")

    if "_close_implied" in out.columns:
        out = out.with_columns(
            pl.coalesce(
                pl.col("closing_market_implied_home").cast(pl.Float64), pl.col("_close_implied")
            ).alias("closing_market_implied_home")
        )
    if "_close_spread" in out.columns:
        out = out.with_columns(
            pl.coalesce(
                pl.col("closing_spread_home").cast(pl.Float64),
                pl.col("_close_spread").cast(pl.Float64, strict=False),
            ).alias("closing_spread_home")
        )
    if "_book" in out.columns:
        out = out.with_columns(
            pl.coalesce(pl.col("market_book"), pl.col("_book").cast(pl.String)).alias("market_book")
        )
    out = out.drop([c for c in ("_close_implied", "_close_spread", "_book") if c in out.columns])

    return out.with_columns(
        pl.coalesce(
            pl.col(MARKET_COLUMN).cast(pl.Float64),
            pl.col("kalshi_yes_price_home").cast(pl.Float64),
            pl.col("closing_market_implied_home").cast(pl.Float64),
        ).alias(MARKET_COLUMN)
    ).with_columns((pl.col(PROB_COLUMN) - pl.col(MARKET_COLUMN)).alias("edge_vs_market"))


# --------------------------------------------------------------------------- #
# Edge vs market
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EdgeReport:
    """Model-vs-market comparison on the subset where a price exists."""

    n_with_market: int
    brier_model: float
    brier_market: float
    brier_skill_score: float
    log_loss_model: float
    log_loss_market: float
    threshold: float
    bounds: tuple[float, float]
    n_bets: int
    n_home_bets: int
    hit_rate: float
    staked: float
    profit_gross: float
    profit_net: float
    roi_gross: float
    roi_net: float
    fees: float
    fee_coefficient: float
    two_sided_fees: bool
    clv: float
    n_clv: int

    def as_dict(self) -> dict[str, Any]:
        return {
            f: getattr(self, f)
            for f in (
                "n_with_market",
                "brier_model",
                "brier_market",
                "brier_skill_score",
                "log_loss_model",
                "log_loss_market",
                "threshold",
                "n_bets",
                "n_home_bets",
                "hit_rate",
                "staked",
                "profit_gross",
                "profit_net",
                "roi_gross",
                "roi_net",
                "fees",
                "fee_coefficient",
                "two_sided_fees",
                "clv",
                "n_clv",
            )
        }


_EMPTY_EDGE = EdgeReport(
    n_with_market=0,
    brier_model=float("nan"),
    brier_market=float("nan"),
    brier_skill_score=float("nan"),
    log_loss_model=float("nan"),
    log_loss_market=float("nan"),
    threshold=DEFAULT_EDGE_THRESHOLD,
    bounds=COLLAPSE_BOUNDS,
    n_bets=0,
    n_home_bets=0,
    hit_rate=float("nan"),
    staked=0.0,
    profit_gross=0.0,
    profit_net=0.0,
    roi_gross=float("nan"),
    roi_net=float("nan"),
    fees=0.0,
    fee_coefficient=KALSHI_FEE_COEFFICIENT,
    two_sided_fees=True,
    clv=float("nan"),
    n_clv=0,
)


def edge_report(
    frame: pl.DataFrame,
    *,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
    bounds: tuple[float, float] = COLLAPSE_BOUNDS,
    contracts: float = 1.0,
    fee_coefficient: float = KALSHI_FEE_COEFFICIENT,
    two_sided_fees: bool = True,
    round_fees: bool = True,
) -> EdgeReport:
    """Brier skill vs market, hypothetical flat-stake ROI, and CLV.

    **The bet.** One contract (``contracts``) per qualifying row: home when
    ``p_model - p_market > threshold``, away when the reverse, and only while
    ``p_market`` sits inside ``bounds``. Cost is the price of the side taken;
    settlement pays $1 if that side wins.

    **The fees.** ``two_sided_fees`` charges :func:`kalshi_fee` twice — entry
    and exit — which is the pessimistic case for a position you trade out of.
    A position held to settlement pays it once; pass ``two_sided_fees=False``
    for that. ROI is reported both gross and net so the fee drag is visible
    rather than buried.

    **The caveat.** This is a *hypothetical*: it assumes fills at the recorded
    price, ignores depth, slippage and the fact that betting into a market moves
    it. Treat a positive ROI here as "worth a paper-trading run", never as
    realised edge.
    """
    if frame.is_empty() or MARKET_COLUMN not in frame.columns:
        return _EMPTY_EDGE
    lo, hi = float(bounds[0]), float(bounds[1])
    labelled = _labelled(frame).filter(pl.col(MARKET_COLUMN).is_not_null())
    if labelled.is_empty():
        return _EMPTY_EDGE

    y = labelled.get_column(LABEL_COLUMN).cast(pl.Float64).to_numpy()
    p = np.clip(labelled.get_column(PROB_COLUMN).cast(pl.Float64).to_numpy(), _EPS, 1 - _EPS)
    m = np.clip(labelled.get_column(MARKET_COLUMN).cast(pl.Float64).to_numpy(), _EPS, 1 - _EPS)

    brier_model = _brier(y, p)
    brier_market = _brier(y, m)
    skill = 1.0 - brier_model / brier_market if brier_market > 0 else float("nan")

    edge = p - m
    tradeable = (m >= lo) & (m <= hi)
    bet_home = tradeable & (edge > threshold)
    bet_away = tradeable & (-edge > threshold)
    bet = bet_home | bet_away

    n_bets = int(bet.sum())
    if n_bets == 0:
        return EdgeReport(
            n_with_market=len(y),
            brier_model=brier_model,
            brier_market=brier_market,
            brier_skill_score=skill,
            log_loss_model=_log_loss(y, p),
            log_loss_market=_log_loss(y, m),
            threshold=threshold,
            bounds=(lo, hi),
            n_bets=0,
            n_home_bets=0,
            hit_rate=float("nan"),
            staked=0.0,
            profit_gross=0.0,
            profit_net=0.0,
            roi_gross=float("nan"),
            roi_net=float("nan"),
            fees=0.0,
            fee_coefficient=fee_coefficient,
            two_sided_fees=two_sided_fees,
            clv=float("nan"),
            n_clv=0,
        )

    # Price paid for the side taken, and whether that side won.
    price = np.where(bet_home[bet], m[bet], 1.0 - m[bet])
    won = np.where(bet_home[bet], y[bet], 1.0 - y[bet])
    staked = float(np.sum(price) * contracts)
    gross = float(np.sum(contracts * (won - price)))

    sides = 2.0 if two_sided_fees else 1.0
    fees = float(
        sides
        * sum(
            kalshi_fee(pr, contracts, coefficient=fee_coefficient, round_up=round_fees)
            for pr in price
        )
    )
    net = gross - fees

    clv, n_clv = _clv(labelled, bet, bet_home, m)

    return EdgeReport(
        n_with_market=len(y),
        brier_model=brier_model,
        brier_market=brier_market,
        brier_skill_score=skill,
        log_loss_model=_log_loss(y, p),
        log_loss_market=_log_loss(y, m),
        threshold=threshold,
        bounds=(lo, hi),
        n_bets=n_bets,
        n_home_bets=int(bet_home.sum()),
        hit_rate=float(np.mean(won)),
        staked=staked,
        profit_gross=gross,
        profit_net=net,
        roi_gross=gross / staked if staked else float("nan"),
        roi_net=net / staked if staked else float("nan"),
        fees=fees,
        fee_coefficient=fee_coefficient,
        two_sided_fees=two_sided_fees,
        clv=clv,
        n_clv=n_clv,
    )


def _clv(
    labelled: pl.DataFrame, bet: np.ndarray, bet_home: np.ndarray, m: np.ndarray
) -> tuple[float, int]:
    """Mean closing-line value on the bets taken, in probability points.

    Only rows where the entry price and the close are genuinely different
    observations count: if the entry price *is* the closing line (no Kalshi
    snapshot was recorded), CLV is definitionally zero and including it would
    dilute the number toward zero rather than say anything.
    """
    if "closing_market_implied_home" not in labelled.columns:
        return float("nan"), 0
    close = labelled.get_column("closing_market_implied_home").cast(pl.Float64).to_numpy()
    usable = bet & np.isfinite(close) & (np.abs(close - m) > 1e-12)
    if not usable.any():
        return float("nan"), 0
    # Home bet: value is the close moving up. Away bet: the close moving down.
    move = np.where(bet_home[usable], close[usable] - m[usable], m[usable] - close[usable])
    return float(np.mean(move)), int(usable.sum())


# --------------------------------------------------------------------------- #
# Retrain triggers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RetrainSignal:
    """Rolling-window drift of realised log loss against the training-time metric."""

    model_name: str
    model_version: str
    window: int
    n: int
    baseline_log_loss: float
    recent_log_loss: float
    drift: float
    threshold: float
    triggered: bool
    reason: str


def retrain_trigger(
    frame: pl.DataFrame,
    *,
    baseline_log_loss: float,
    window: int = 500,
    threshold: float = 0.02,
    model_name: str = "",
    model_version: str = "",
) -> RetrainSignal:
    """Flag when recent log loss has drifted above the training-time metric.

    ``window`` is the number of most-recent *resolved* predictions to score.
    ``threshold`` is an absolute log-loss increase, not a ratio: log loss is
    already on a log scale, and a 0.02 nats regression means the same thing
    whether the baseline is 0.15 or 0.45.

    Flags only; nothing here retrains anything.
    """
    labelled = _labelled(frame)
    if "predicted_at_utc" in labelled.columns:
        labelled = labelled.sort("predicted_at_utc")
    recent = labelled.tail(window)
    if recent.is_empty():
        return RetrainSignal(
            model_name,
            model_version,
            window,
            0,
            baseline_log_loss,
            float("nan"),
            float("nan"),
            threshold,
            False,
            "no resolved predictions in window",
        )
    recent_ll = _log_loss(*_arrays(*_yp(recent)))
    drift = recent_ll - baseline_log_loss
    triggered = bool(drift > threshold)
    reason = (
        f"log loss drifted +{drift:.4f} over the last {recent.height} predictions "
        f"(threshold {threshold:.4f})"
        if triggered
        else f"within tolerance ({drift:+.4f} vs threshold {threshold:.4f})"
    )
    return RetrainSignal(
        model_name,
        model_version,
        window,
        recent.height,
        baseline_log_loss,
        recent_ll,
        drift,
        threshold,
        triggered,
        reason,
    )


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ModelReport:
    """Everything computed for one ``(model_name, model_version)``."""

    model_name: str
    model_version: str
    n_rows: int
    n_resolved: int
    slices: list[SliceMetrics] = field(default_factory=list)
    reliability: pl.DataFrame | None = None
    edge: EdgeReport | None = None
    retrain: RetrainSignal | None = None


@dataclass(slots=True)
class EvaluationReport:
    """A full evaluation pass: one :class:`ModelReport` per model version."""

    generated_at: datetime
    source: str
    n_rows: int
    models: list[ModelReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_report(
    frame: pl.DataFrame,
    *,
    source: str = "ledger",
    threshold: float = DEFAULT_EDGE_THRESHOLD,
    bounds: tuple[float, float] = COLLAPSE_BOUNDS,
    fee_coefficient: float = KALSHI_FEE_COEFFICIENT,
    two_sided_fees: bool = True,
    bins: int = 10,
    baselines: Mapping[str, float] | None = None,
    window: int = 500,
    drift_threshold: float = 0.02,
    with_market: bool = True,
    market: pl.DataFrame | None = None,
    root: Path | None = None,
) -> EvaluationReport:
    """Score every model version present in ``frame`` and assemble the report.

    ``baselines`` maps ``model_version`` (or ``model_name``) to the log loss the
    model reported at training time; a version with no baseline simply gets no
    retrain signal rather than a fabricated one.
    """
    report = EvaluationReport(generated_at=datetime.now(UTC), source=source, n_rows=frame.height)
    if frame.is_empty():
        report.notes.append("no predictions to evaluate")
        return report

    if with_market:
        frame = attach_closing_lines(frame, market, root=root)
        if frame.get_column(MARKET_COLUMN).null_count() == frame.height:
            report.notes.append("no market prices matched; edge report omitted")
            with_market = False

    resolved_total = int(frame.get_column(LABEL_COLUMN).is_not_null().sum())
    if resolved_total == 0:
        report.notes.append("no resolved outcomes — run `nfl ledger resolve` first")
        return report

    keys = (
        frame.select("model_name", "model_version")
        .fill_null("?")
        .unique(maintain_order=True)
        .sort("model_name", "model_version")
    )
    for name, version in keys.iter_rows():
        part = frame.filter(
            (pl.col("model_name").fill_null("?") == name)
            & (pl.col("model_version").fill_null("?") == version)
        )
        labelled = _labelled(part)
        model = ModelReport(
            model_name=name,
            model_version=version,
            n_rows=part.height,
            n_resolved=labelled.height,
        )
        if labelled.is_empty():
            report.models.append(model)
            continue
        model.slices = evaluate_frame(part, bins=bins, bounds=bounds)
        model.reliability = reliability_table(*_yp(labelled), bins=bins)
        if with_market:
            model.edge = edge_report(
                part,
                threshold=threshold,
                bounds=bounds,
                fee_coefficient=fee_coefficient,
                two_sided_fees=two_sided_fees,
            )
        baseline = None
        if baselines:
            baseline = baselines.get(version, baselines.get(name))
        if baseline is not None:
            model.retrain = retrain_trigger(
                part,
                baseline_log_loss=float(baseline),
                window=window,
                threshold=drift_threshold,
                model_name=name,
                model_version=version,
            )
        report.models.append(model)
    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{digits}f}"


def render_report(report: EvaluationReport, console: Any) -> None:
    """Print a report to a ``rich`` console."""
    from rich.table import Table

    console.print(
        f"[bold]evaluation[/bold] source={report.source} rows={report.n_rows:,} "
        f"generated={report.generated_at:%Y-%m-%d %H:%M:%S}Z"
    )
    for note in report.notes:
        console.print(f"  [yellow]{note}[/yellow]")

    for model in report.models:
        console.print(
            f"\n[bold cyan]{model.model_name}[/bold cyan] "
            f"[dim]{model.model_version}[/dim] "
            f"({model.n_resolved:,}/{model.n_rows:,} resolved)"
        )
        if not model.slices:
            console.print("  [yellow]nothing resolved for this version[/yellow]")
            continue

        table = Table(title="metrics by slice")
        table.add_column("slice")
        table.add_column("n", justify="right")
        for header in ("log_loss", "brier", "accuracy", "ece", "auc"):
            table.add_column(header, justify="right")
        for item in model.slices:
            m = item.metrics
            table.add_row(
                item.name,
                f"{m.n:,}",
                _fmt(m.log_loss),
                _fmt(m.brier),
                _fmt(m.accuracy),
                _fmt(m.ece),
                _fmt(m.auc),
            )
        console.print(table)

        if model.reliability is not None and model.reliability.height:
            rel = Table(title="reliability (10 bins)")
            rel.add_column("bin")
            rel.add_column("n", justify="right")
            rel.add_column("predicted", justify="right")
            rel.add_column("observed", justify="right")
            rel.add_column("gap", justify="right")
            for row in model.reliability.iter_rows(named=True):
                if not row["n"]:
                    continue
                rel.add_row(
                    row["bin"],
                    f"{row['n']:,}",
                    _fmt(row["mean_predicted"], 3),
                    _fmt(row["observed"], 3),
                    _fmt(row["gap"], 3),
                )
            console.print(rel)

        if model.edge is not None:
            _render_edge(model.edge, console)
        if model.retrain is not None:
            colour = "red" if model.retrain.triggered else "green"
            flag = "RETRAIN" if model.retrain.triggered else "ok"
            console.print(
                f"  retrain trigger: [{colour}]{flag}[/{colour}] — {model.retrain.reason} "
                f"(baseline {_fmt(model.retrain.baseline_log_loss)}, "
                f"recent {_fmt(model.retrain.recent_log_loss)}, n={model.retrain.n:,})"
            )


def _render_edge(edge: EdgeReport, console: Any) -> None:
    from rich.table import Table

    table = Table(title="edge vs market (evaluation only — ADR 0001)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    skill_colour = "green" if edge.brier_skill_score > 0 else "red"
    table.add_row("rows with a market price", f"{edge.n_with_market:,}")
    table.add_row("brier (model)", _fmt(edge.brier_model))
    table.add_row("brier (market)", _fmt(edge.brier_market))
    table.add_row(
        "brier skill vs market",
        f"[{skill_colour}]{_fmt(edge.brier_skill_score)}[/{skill_colour}]",
    )
    table.add_row(
        "log loss (model / market)", f"{_fmt(edge.log_loss_model)} / {_fmt(edge.log_loss_market)}"
    )
    table.add_row(
        f"bets at |edge| > {edge.threshold:.2f}, market in [{edge.bounds[0]:g}, {edge.bounds[1]:g}]",
        f"{edge.n_bets:,} ({edge.n_home_bets:,} home)",
    )
    table.add_row("hit rate", _fmt(edge.hit_rate, 3))
    table.add_row("staked", f"${edge.staked:,.2f}")
    table.add_row("ROI gross", _fmt(edge.roi_gross, 4))
    table.add_row(
        f"ROI net of fees ({edge.fee_coefficient:g}, {'2-sided' if edge.two_sided_fees else '1-sided'})",
        _fmt(edge.roi_net, 4),
    )
    table.add_row("fees paid", f"${edge.fees:,.2f}")
    table.add_row("CLV (prob points)", f"{_fmt(edge.clv, 4)} (n={edge.n_clv:,})")
    console.print(table)


def report_path(name: str, root: Path | None = None) -> Path:
    """``data/reports/<name>.md``."""
    base = (config.DATA_DIR if root is None else Path(root)) / "reports"
    return base / f"{name}.md"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def to_markdown(report: EvaluationReport) -> str:
    """Render the report as markdown."""
    lines = [
        "# Evaluation report",
        "",
        f"- generated: `{report.generated_at.isoformat()}`",
        f"- source: `{report.source}`",
        f"- rows: {report.n_rows:,}",
        "",
    ]
    for note in report.notes:
        lines += [f"> {note}", ""]

    for model in report.models:
        lines += [
            f"## {model.model_name} `{model.model_version}`",
            "",
            f"{model.n_resolved:,} of {model.n_rows:,} predictions resolved.",
            "",
        ]
        if model.slices:
            lines += [
                _md_table(
                    ["slice", "n", "log_loss", "brier", "accuracy", "ece", "auc"],
                    [
                        [
                            s.name,
                            f"{s.metrics.n:,}",
                            _fmt(s.metrics.log_loss),
                            _fmt(s.metrics.brier),
                            _fmt(s.metrics.accuracy),
                            _fmt(s.metrics.ece),
                            _fmt(s.metrics.auc),
                        ]
                        for s in model.slices
                    ],
                ),
                "",
            ]
        if model.reliability is not None and model.reliability.height:
            lines += [
                "### Reliability",
                "",
                _md_table(
                    ["bin", "n", "predicted", "observed", "gap"],
                    [
                        [
                            r["bin"],
                            f"{r['n']:,}",
                            _fmt(r["mean_predicted"], 3),
                            _fmt(r["observed"], 3),
                            _fmt(r["gap"], 3),
                        ]
                        for r in model.reliability.iter_rows(named=True)
                        if r["n"]
                    ],
                ),
                "",
            ]
        if model.edge is not None:
            e = model.edge
            lines += [
                "### Edge vs market",
                "",
                "Evaluation only — market data is never a model input (ADR 0001).",
                "",
                _md_table(
                    ["metric", "value"],
                    [
                        ["rows with market price", f"{e.n_with_market:,}"],
                        ["brier (model)", _fmt(e.brier_model)],
                        ["brier (market)", _fmt(e.brier_market)],
                        ["brier skill score", _fmt(e.brier_skill_score)],
                        ["log loss (model)", _fmt(e.log_loss_model)],
                        ["log loss (market)", _fmt(e.log_loss_market)],
                        ["edge threshold", f"{e.threshold:g}"],
                        ["market bounds", f"[{e.bounds[0]:g}, {e.bounds[1]:g}]"],
                        ["bets", f"{e.n_bets:,} ({e.n_home_bets:,} home)"],
                        ["hit rate", _fmt(e.hit_rate, 3)],
                        ["staked", f"{e.staked:,.2f}"],
                        ["ROI gross", _fmt(e.roi_gross)],
                        ["ROI net", _fmt(e.roi_net)],
                        ["fees", f"{e.fees:,.2f}"],
                        [
                            "fee model",
                            f"ceil_cent({e.fee_coefficient:g}*C*P*(1-P)), "
                            f"{'2-sided' if e.two_sided_fees else '1-sided'}",
                        ],
                        ["CLV", f"{_fmt(e.clv)} (n={e.n_clv:,})"],
                    ],
                ),
                "",
            ]
        if model.retrain is not None:
            r = model.retrain
            lines += [
                "### Retrain trigger",
                "",
                f"- **{'TRIGGERED' if r.triggered else 'ok'}** — {r.reason}",
                f"- baseline log loss: {_fmt(r.baseline_log_loss)}",
                f"- recent log loss (last {r.n:,} of window {r.window:,}): "
                f"{_fmt(r.recent_log_loss)}",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    report: EvaluationReport, path: Path | None = None, *, name: str | None = None
) -> Path:
    """Write :func:`to_markdown` under ``data/reports/`` (or ``path``)."""
    target = (
        Path(path)
        if path is not None
        else report_path(name or f"evaluation-{report.generated_at:%Y%m%dT%H%M%S}")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_markdown(report), encoding="utf-8")
    return target


def to_json(report: EvaluationReport) -> str:
    """Machine-readable form of the report — for CI gates and dashboards."""
    payload = {
        "generated_at": report.generated_at.isoformat(),
        "source": report.source,
        "n_rows": report.n_rows,
        "notes": report.notes,
        "models": [
            {
                "model_name": m.model_name,
                "model_version": m.model_version,
                "n_rows": m.n_rows,
                "n_resolved": m.n_resolved,
                "slices": {s.name: s.metrics.as_dict() for s in m.slices},
                "edge": m.edge.as_dict() if m.edge else None,
                "retrain": (
                    {
                        "triggered": m.retrain.triggered,
                        "drift": m.retrain.drift,
                        "baseline_log_loss": m.retrain.baseline_log_loss,
                        "recent_log_loss": m.retrain.recent_log_loss,
                        "reason": m.retrain.reason,
                    }
                    if m.retrain
                    else None
                ),
            }
            for m in report.models
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def evaluate_ledger(
    *,
    root: Path | None = None,
    seasons: Sequence[int] | int | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    **kwargs: Any,
) -> EvaluationReport:
    """Read the ledger and score it — the ``nfl evaluate --ledger`` path."""
    frame = ledger_mod.read_ledger(
        root, seasons=seasons, model_name=model_name, model_version=model_version
    )
    return build_report(frame, source="ledger", root=root, **kwargs)
