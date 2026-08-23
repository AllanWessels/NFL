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

## Slash commands (preferred over raw `uv run`)

| command | does |
|---|---|
| `/predict <game> [--at "Q4 05:00"]` | win probability + confidence + trade gate, logs to ledger |
| `/retrain [--skip-ingest]` | weekly pipeline: ingest → features → train → ensemble → gate → promote |
| `/evaluate` | ledger accuracy/calibration/edge vs market + drift flag |
| `/ledger [--last 20]` | recent predictions with resolved outcomes |
| `/status` | production model, last retrain, drift |
| `/ingest [--seasons 2026]` | data refresh + feature rebuild only |
| `/snapshot <game>` | live game state without predicting |
| `/promote [--candidate ensemble_v1]` | run the gate on a candidate |

Weekly routine: Tuesday `/retrain`; game day `/predict TEAM@TEAM` late in Q4; Monday `/evaluate`.

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
uv run nfl ingest weather --seasons 2025   # --games-parquet defaults to the season partitions
uv run nfl train --model baseline          # naive | xgb | mlp | seq | ensemble
uv run nfl train --model xgb --split 2023/2024/2025   # pin the split; default rolls
uv run nfl predict 2024_18_KC_DEN --at "Q4 05:00"
uv run nfl predict "DAL@PHI" --season 2025 --week 1   # matchup / team / ticker / espn id
uv run nfl predict Eagles                              # that team's game this week
uv run nfl evaluate --season 2024

uv run nfl retrain                         # the whole weekly loop, in order
uv run nfl retrain --skip-ingest --dry-run # what it would do, without doing it

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

Because the splits roll, "the same val season" is a gate condition rather than an
assumption: the `split` row FAILS when the candidate's manifest and the floor's
record different boundaries. Retraining one member and not the others across a
season rollover is exactly how a candidate ends up looking better than a floor it
never shared a row with. An artifact old enough not to record its split gets a
loud note instead — unprovable is not the same as mismatched.

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

## Retraining

### Rolling season splits

The split is **not** a pair of years someone typed once. `L` is the latest
season whose **regular season has finished** on disk — every `game_type == "REG"`
row for it carries a result — detected from the nflverse schedules by
`nfl.features.splits.latest_complete_season`. From that:

```
train <= L-2      val = L-1      test = L
```

The newest complete season is the *test* set because it is the one nothing has
been tuned against; the one before it is val, so the promotion gate always
judges a model on the newest football it is allowed to have seen. A season still
being played is never eligible — its unplayed games have null results, so it
cannot be `L`. Pinning 2022/2023/2024 forever means that the moment 2024 ends,
"test" silently becomes two seasons and val stops being the most recent thing
anyone measured.

Resolution order for the default (`nfl.features.splits.default_split`):

1. the `NFL_SEASON_SPLIT` environment variable, a `SeasonSplit.parse` spec
   (`"2023/2024/2025"`, `"rolling:2025"`, or a bare `"2025"`) — the test suite
   pins this so fixture datasets are not re-cut by whatever is in `data/raw`;
2. `SeasonSplit.rolling(latest_complete_season())`;
3. `FALLBACK_SPLIT` (2022/2023/2024) when there are no schedules on disk at all.

`--split` on `nfl train` and `nfl retrain` overrides it per run. **Whichever
split is used is written into every artifact's manifest**, and the promotion
gate has a `split` row that FAILS when the candidate and the floor were
validated on different seasons: two models measured on different football cannot
be subtracted from one another.

### `nfl retrain`

One command, ten steps, in the only order that works — each consumes what the
one before it wrote:

| # | step | notes |
|---|---|---|
| 1 | `ingest nflverse` | current season, idempotent; `--skip-ingest` drops it |
| 2 | `ingest weather` | needs the schedules from step 1 |
| 3 | `features build` | **all** seasons — the rolling boundaries moved |
| 4 | `train naive` | the floor first: its band keys `not_collapsed` for the rest |
| 5 | `train xgb` | `servable`, sweep-best params read back from the latest manifest |
| 6 | `train seq` | `servable`, likewise (`hyperparams` in the manifest) |
| 7 | `train ensemble` | `logit_mean` + phase-aware isotonic |
| 8 | `promote` | the ADR 0003 §5 gate. **Never `--force` from here** |
| 9 | `ledger resolve` | fill outcomes for games that finished |
| 10 | `evaluate --ledger` | score the ledger, take the drift reading |

Flags: `--season`, `--split`, `--skip-ingest`, `--dry-run`, `--parity-samples`.

Two kinds of not-succeeding, deliberately different. A step that **raises** stops
the chain there — everything downstream would be reading stale or absent inputs.
A gate that **refuses** does not: the chain continues into ledger resolve and
evaluate, because those describe the model *already in production*, which is
still the one being served and is exactly what you want to read when a candidate
has just failed to replace it. The command exits non-zero either way.

Every step is timed, and the run — steps, timings, summary, full gate table — is
written to `data/reports/retrain-<timestamp>.md` whether it succeeded or not.
The closing summary answers three questions: **gate result**, **PRODUCTION
pointer**, **drift flag**.

### Drift thresholds

`nfl.logging.evaluate.retrain_trigger` compares recent realised log loss against
what the artifact recorded at training time. Both `nfl evaluate --ledger` and the
retrain summary print it.

| knob | default | meaning |
|---|---|---|
| `--window` | **500** | most-recent *resolved* predictions scored |
| `--drift-threshold` | **0.02** | absolute log-loss increase that flags `RETRAIN` |
| `--baseline-log-loss` | from the manifest | the training-time number to compare against |

The threshold is **absolute, not a ratio**: log loss is already on a log scale,
so 0.02 nats means the same regression whether the baseline is 0.15 or 0.45. The
baseline is read from each model version's own manifest, so the flag shows up
without anyone having to remember a number — a warning nobody sees by default is
not a warning. A version with no baseline on record gets a loud "no baseline"
line rather than a fabricated signal. The trigger **flags only**; it never
retrains anything.

### Weekly schedule

```bash
scripts/weekly_retrain.sh            # runs `uv run nfl retrain` from the repo root
                                     # log -> data/logs/retrain-<UTC stamp>.log
scripts/install_cron.sh              # print the crontab entry it would install
scripts/install_cron.sh --install    # install it (Tuesday 06:00 local)
scripts/install_cron.sh --remove     # take it out again
```

Tuesday 06:00 **local** is the slot: Monday Night Football has finished,
nflverse has published the week, and there are six days left to notice that the
gate refused. `NFL_CRON_SCHEDULE` overrides the cron expression. Installing is
idempotent — the entry carries a marker comment and an existing one is replaced,
not duplicated. The wrapper resolves the repo root from its own path (cron starts
in `$HOME`) and puts `~/.local/bin` on `PATH` (cron's `PATH` has no `uv`).

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
  scripts/
    weekly_retrain.sh          # cron entrypoint for `nfl retrain`
    install_cron.sh            # installs the Tuesday 06:00 crontab entry
  nfl/
    __init__.py
    cli.py                     # typer app: ingest / train / retrain / predict / evaluate
    retrain.py                 # the weekly chain, its timings and its run log
    data/                      # ingest adapters + pydantic schemas
    features/                  # pre-game & in-game feature builders
    models/                    # baselines, gbdt, tf nets, ensemble, calibration
    inference/                 # predict(game_id, at) entrypoint
    logging/                   # prediction ledger (parquet/duckdb), metrics
  tests/
  data/                        # gitignored: raw/, processed/, reports/, logs/
  models/                      # gitignored: trained artifacts
```

## Conventions

- Python 3.12, `from __future__ import annotations` in modules with annotations.
- Pydantic v2 for all external-data schemas (`nfl/data/schemas.py`).
- Polars for bulk transforms; pandas only where a library forces it.
- Every dataset written to Parquet carries a `source` and `ingested_at` column.
- Every prediction written to the ledger carries: timestamp, game id, snapshot
  time, feature hash, model version, probability, confidence.
