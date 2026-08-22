# Architecture

## Goal

At any moment `t` in an NFL game, produce `P(home wins | state at t)` and a
calibrated confidence. Optimised for late-Q4 accuracy, where the Kalshi market
is most often mispriced relative to the true state of the game.

## Pipeline

```
                    ┌──────────────────────────────────────────┐
                    │  nfl ingest                              │
   nflverse  ─────► │  pbp / schedules / rosters / injuries    │
   Open-Meteo ────► │  historical weather by venue+kickoff     │ ──► data/raw/*.parquet
   news/injury ───► │  headlines, injury reports               │
   ESPN (live) ───► │  LiveFeed adapter (same shape as pbp)    │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼  normalise + validate (pydantic)
                              data/processed/*.parquet
                                    + nfl.duckdb
                                       │
                    ┌──────────────────┴───────────────────────┐
                    │  nfl.features                            │
                    │   pregame:  Elo, unit ranks, EPA, pace,   │
                    │             rest, travel, coach records,  │
                    │             opponent-adjusted player stats│
                    │   in-game:  cumulative state per play —   │
                    │             score, clock, possession,     │
                    │             down/dist, field pos, TOs,    │
                    │             turnovers, drive outcomes,    │
                    │             yards by unit, penalties,     │
                    │             in-game injuries, momentum    │
                    └──────────────────┬───────────────────────┘
                                       │  one row per play
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  nfl.models                              │
                    │   baseline: logistic(score diff, clock)  │
                    │   gbdt:     XGBoost (GPU hist)           │
                    │   mlp:      TF dense net                 │
                    │   seq:      TF sequence over play history│
                    │   ensemble: stack + calibration          │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  nfl.inference.predict(game_id, at)      │
                    │   → p_home, p_away, confidence           │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │  nfl.logging  — prediction ledger        │
                    │   ts, game_id, snapshot_t, feature_hash,  │
                    │   model_version, p_home, confidence       │
                    │   ...later resolved vs outcome AND vs     │
                    │   market (closing line / Kalshi)          │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼   nfl evaluate
                    log-loss · Brier · calibration curve · edge
```

## Layers

**`nfl/data`** — one adapter per source, each returning validated pydantic
models or a Polars frame with a declared schema. The live ESPN feed sits behind
a `LiveFeed` protocol so historical replay and live play use the same code path;
a historical game replayed at `Q4 05:00` must produce byte-identical features to
what the live feed would have produced at that moment.

**`nfl/features`** — pure functions from (game history up to `t`, pregame
tables) to a feature vector. No I/O, no globals, no future information. Every
builder ships with a leakage test that asserts the feature at play `t` is
unchanged when plays `>t` are perturbed.

**`nfl/models`** — each model exposes `fit`, `predict_proba`, `save`, `load`
and a `version` string. Splits are season-based: train ≤2022, validate 2023,
test 2024+. Never random splits — plays within a game are massively correlated.

**`nfl/inference`** — `predict(game_id, at=None)`: resolve the snapshot, build
features, run the selected model version, compute confidence, write the ledger
row, return the result.

**`nfl/logging`** — append-only Parquet ledger queried through DuckDB. Every
prediction is recoverable and attributable to an exact model version and
feature hash.

## Confidence

Two candidates, to be decided by ADR after baseline results:
1. **Ensemble spread** — disagreement across ensemble members, calibrated
   against realised error.
2. **MC-dropout** — predictive variance from stochastic forward passes on the
   TF nets.

Whichever produces better-calibrated confidence intervals on the held-out
seasons wins; the loser is deleted, not kept "just in case".

## Storage

Parquet on local disk, queried via DuckDB (`data/processed/nfl.duckdb` attaches
the Parquet files rather than copying them). Rationale in
`docs/ADR/0002-storage-parquet-duckdb.md`. GCS sync is a later phase and changes
only the path prefix.

## Hardware

RTX 5080 Laptop (16 GB, Blackwell, compute capability 12.0a). XGBoost uses
`device="cuda"` with `hist`. TensorFlow 2.21 works on this GPU but with two
caveats, both verified during scaffolding:

- The pip CUDA wheels are not on the loader path, so `nfl.gpu.preload_cuda_libraries()`
  must run before `import tensorflow` or TF silently drops to CPU.
- TF 2.21 ships no prebuilt kernels for sm_120a, so kernels are JIT-compiled
  from PTX on first use — a long one-time warm-up per kernel shape.

Model sizes are deliberately small — the bottleneck here is feature quality and
calibration, not parameter count — so the JIT warm-up, not raw throughput, is
the thing to watch.

## What is deliberately absent

- **Market data as a feature.** ADR 0001. Lines/Kalshi enter only at evaluation.
- **A daemon.** v1 is event-driven: `/predict {game}` runs the pipeline once.
- **Cloud LLM calls.** Sentiment is rule-based, optionally local Ollama.
