# CLAUDE.md — NFL Win-Probability Project

## Purpose

Build an ML system that ingests a snapshot of an NFL game at any moment (target
use case: late Q4) and emits `P(home wins)`, `P(away wins) = 1 - P(home)`, and a
**confidence** score. Downstream intent: take Kalshi positions when the market
has *not* collapsed past ~85/15. Runs locally on an RTX 5080 (16 GB), optionally
on GCP.

## Roles

| Role | Model | Responsibility |
|---|---|---|
| **Fable** | architect / auditor | Plans, ADRs, reviews **every** PR before merge. Picks model architecture after baseline results. |
| **Opus** | developer | Writes all code. Runs tests. Opens PRs. |
| **Haiku** | retriever | Web lookups, doc fetches, trivial mechanical tasks. **Never writes code.** |

## Hard rules

1. **Market data is NEVER a model input.** Betting lines, spreads, totals,
   Kalshi prices, implied probabilities — none of it reaches a feature vector.
   It is stored, flagged `market_only=True`, and used *only* in evaluation to
   measure edge. Any PR that feeds market data into training is rejected.
   See `docs/ADR/0001-no-market-inputs.md`.
2. **Labels are game outcomes only.** The target is the actual winner of the
   game. Not the closing line, not a market-implied probability, not a
   third-party win-probability model's output.
3. **No leakage.** A feature computed for play *t* may use only information
   available at play *t*. Cross-season and cross-game aggregates must be
   as-of-date. Leakage tests are mandatory for every feature builder.
4. **No cloud LLMs.** Sentiment scoring is rule-based by default. A local
   Ollama model may be benchmarked against it, and adopted only if the
   rule-based scorer is *measurably* worse. Nothing leaves the machine.
5. **TensorFlow + numpy** for neural models; XGBoost for GBDT. Not PyTorch.
6. **Parquet + DuckDB** for storage. Local-first, GCS-syncable. No server DB.
   See `docs/ADR/0002-storage-parquet-duckdb.md`.
7. **Concise output.** Reports state what changed, what passed, what is
   uncertain. No filler, no restating the plan back.

## Commands

```bash
uv sync --extra gpu --dev      # local install/refresh (includes CUDA wheels)
uv sync --dev                  # CPU-only, what CI runs
uv run pytest                  # test suite
uv run ruff check .            # lint
uv run ruff format .           # format
uv run mypy nfl                # type check (advisory for now)
uv run pre-commit install      # enable git hooks once per clone

uv run nfl ingest nflverse --seasons 2010-2025 [--tables pbp,schedules] [--force]
uv run nfl train --model baseline          # naive | xgb | mlp | seq | ensemble
uv run nfl predict 2024_18_KC_DEN --at "Q4 05:00"
uv run nfl predict "DAL@PHI" --season 2025 --week 1   # matchup / team / ticker / espn id
uv run nfl predict Eagles                              # that team's game this week
uv run nfl evaluate --season 2024

uv run nfl promote --candidate ensemble_v1 # gate, then move the PRODUCTION pointer
uv run nfl promote --candidate ensemble_v1 --version 20260823T061500
uv run nfl promote --candidate ensemble_v1 --parity-samples 0   # offline slices only
uv run nfl promote --candidate ensemble_v1 --floor naive_score_time --force
```

Slash command: `/predict <game> [--at "Q4 05:00"]` wraps `uv run nfl predict`. `<game>`
accepts an nflverse id, a matchup (`DAL@PHI`, `PHI vs DAL`), a single team
(`PHI`, `Eagles`, `Philadelphia`), a Kalshi ticker or an ESPN event id — see
`nfl/inference/resolve.py`.

### Promotion and the production pointer

`nfl promote` is the only way a model becomes PRODUCTION. It runs the ADR 0003 §5
gate — the candidate must **beat** the naive floor on val Q4@600/300/120 on both
log-loss and Brier, and be **no worse** on the not-collapsed subset — printing the
full comparison table either way, so a refused promotion says exactly which number
refused it. Both sides are read from the `val_slices` recorded in each artifact's
manifest at training time, so the gate compares numbers that were actually
measured rather than re-deriving them from a dataset that may have moved.

The not-collapsed subset is keyed on the **floor's** probability band `[0.15, 0.85]`
so both sides are scored on the same rows; the per-model variant is reported
alongside as `not_collapsed_own` and is never gated.

The gate's third condition is **serving parity**: 20 `(game, Q4 time)` snapshots
from val are scored through `nfl predict` on the replay feed *and* through the
offline dataset row for the same play, and every one must land within `0.02`.
Every other number in the table is an offline measurement — without this one the
gate can promote a model whose production behaviour was never checked, which is
exactly how a GRU served a one-play window and a GBDT served a constant 0.47 got
past it. See ADR 0003's serving-parity amendment. `--parity-samples 0` skips it.

On a pass it writes **`data/models/PRODUCTION.json`** — a pointer holding the
`name`/`version`, the promotion timestamp, the git sha, the artifact's metrics
and its dataset fingerprint. It is a pointer, not a copy: the artifact never
moves, so a rollback is another one-line pointer write. `nfl predict` resolves
`--model production` (its default) through that pointer, so promoting is what
changes what the CLI serves. Roll back by promoting the previous version.
`--force` promotes past a failed gate and says so in red; it exists for
emergencies and leaves a loud line in the output.

## GPU notes (RTX 5080)

Two things bite here, both verified on this machine:

1. **TensorFlow will not find CUDA on its own.** The `nvidia-*` wheels install
   into `site-packages/nvidia/*/lib`, which TF does not add to the loader path,
   so it logs *"Cannot dlopen some GPU libraries"* and silently falls back to
   CPU. Call `nfl.gpu.preload_cuda_libraries()` **before** `import tensorflow`,
   or export `LD_LIBRARY_PATH` to those directories. Any module that imports
   TensorFlow for training must do the preload first.
2. **TF 2.21 has no prebuilt kernels for compute capability 12.0a** (Blackwell,
   which is what the 5080 is). It JIT-compiles from PTX on first use, which can
   take tens of minutes. It *works*, it is just slow to warm up — budget for it,
   and do not mistake the pause for a hang.

The CUDA wheels live behind the `gpu` extra so CI does not download ~6 GB it
cannot use.

## Branch & PR workflow

- **Never commit directly to `main`.** `main` is protected by convention;
  Fable reviews everything.
- Branch naming: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`.
- One phase (or one coherent unit of work) per branch.
- Before opening a PR: `uv run ruff check . && uv run ruff format . && uv run pytest`
  must all pass locally; CI runs the same three.
- **Do not push without Fable's sign-off.** Opus commits on the feature branch
  and reports; Fable decides when it goes up.
- Commit messages: imperative mood, scoped prefix
  (`feat(data): ...`, `docs(adr): ...`, `test(features): ...`).

## Repo layout

```
NFL/
  CLAUDE.md                    # this file
  README.md
  pyproject.toml               # uv-managed, Python 3.12
  .pre-commit-config.yaml
  .github/workflows/ci.yml     # uv sync -> ruff -> pytest
  .claude/commands/predict.md  # /predict slash command
  docs/
    ARCHITECTURE.md            # pipeline + model design
    DATA_MODEL.md              # exhaustive entity/attribute spec
    ADR/                       # architecture decision records
  nfl/
    __init__.py
    cli.py                     # typer app: ingest / train / predict / evaluate
    data/                      # ingest adapters + pydantic schemas
    features/                  # pre-game & in-game feature builders
    models/                    # baselines, gbdt, tf nets, ensemble, calibration
    inference/                 # predict(game_id, at) entrypoint
    logging/                   # prediction ledger (parquet/duckdb), metrics
  tests/
  data/                        # gitignored: raw/, processed/
  models/                      # gitignored: trained artifacts
```

## Conventions

- Python 3.12, `from __future__ import annotations` in modules with annotations.
- Pydantic v2 for all external-data schemas (`nfl/data/schemas.py`).
- Polars for bulk transforms; pandas only where a library forces it.
- Every dataset written to Parquet carries a `source` and `ingested_at` column.
- Every prediction written to the ledger carries: timestamp, game id, snapshot
  time, feature hash, model version, probability, confidence.
