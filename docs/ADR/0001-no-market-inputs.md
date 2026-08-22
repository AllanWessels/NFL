# ADR 0001 — Market data is never a model input

- **Status:** Accepted
- **Date:** 2026-08-22
- **Deciders:** Fable (architect), Allan Wessels

## Context

The system's purpose is to find mispricings in the Kalshi NFL win market —
specifically, to take a position when our probability estimate diverges from the
market and the market has not yet collapsed past ~85/15.

Betting lines, spreads, totals, moneylines and Kalshi mid-prices are cheap,
abundant, and enormously predictive. A model given the closing line as a feature
will lean on it almost exclusively, because the market aggregates injury news,
weather, lineup changes and sharp money far more efficiently than any feature
we can engineer.

## Decision

**Market data is never an input to any model in this project.** Not as a
feature, not as a prior, not as a label, not as a sample weight, not as a
distillation target.

Market data *is* ingested and stored — flagged `market_only = True` in the data
model — and used exclusively at evaluation time:

- as a benchmark to beat (log-loss / Brier of the closing line),
- to compute realised edge (`p_model - p_market`) and its correlation with
  outcomes,
- to size hypothetical positions in backtests.

Labels are the **actual game outcome** and nothing else.

## Consequences

**Positive**

- Any measured edge is real. A model that has seen the line cannot be said to
  beat the line; the comparison would be circular.
- The model degrades gracefully where the market is thin or stale — exactly the
  late-Q4 regime we care about.
- The evaluation story is honest and simple: our number vs their number vs what
  happened.

**Negative**

- We forgo a genuinely strong signal, so raw accuracy will very likely be worse
  than a line-informed model. That is accepted: raw accuracy is not the
  objective, *edge* is.
- We must reconstruct information the market prices in (injuries, weather,
  rest, travel) from primary sources ourselves. This is most of the work in
  Phases 3–4.

## Enforcement

- `nfl/data/schemas.py` marks market entities with an explicit
  `market_only: Literal[True]` field, so they cannot be silently treated as
  feature rows.
- Feature builders live in `nfl/features/` and take only non-market tables as
  arguments; market frames are not importable into that namespace by convention.
- Any PR introducing a market-derived column into a training frame is rejected
  at review. A test asserting the training feature set excludes market columns
  is to be added with the first real feature builder (Phase 4).

## Alternatives considered

- **Use the line as a feature, then subtract it out.** Rejected: residualising
  against the market makes the model's target the market's error, which changes
  the problem and reintroduces market dependence at inference time (we would
  need a live line to predict).
- **Use the opening line only.** Rejected: same circularity, less signal, and it
  still requires a market feed at inference.
