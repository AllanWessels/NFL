# NFL Win-Probability Forecasting

An ML system that ingests the state of an NFL game at any moment (target: late Q4) and
emits **P(home wins)**, **P(away wins) = 1 − P(home)**, and a **confidence**. Built to take
Kalshi positions when the market has *not* collapsed past ~85/15. Runs locally on a single
GPU (RTX 5080); optionally on GCP.

> Hard rules: market data (lines, Kalshi prices) is **never** a model input — it is used only
> to measure edge. Labels are actual game outcomes. No cloud LLMs. See `CLAUDE.md` and
> `docs/ADR/`.

---

## Quick start (slash commands)

Open Claude Code in this repo and use:

| command | when | does |
|---|---|---|
| `/predict DAL@PHI [--at "Q4 05:00"]` | game day, late Q4 | win probability, confidence, member breakdown, trade gate; logs to ledger |
| `/retrain [--skip-ingest]` | Tuesdays (or cron) | ingest → features → train 3 members → ensemble → promotion gate → promote → evaluate |
| `/evaluate` | Mondays | ledger accuracy, calibration, edge vs market, drift flag |
| `/ledger [--last 20]` | any time | recent predictions with resolved outcomes |
| `/status` | any time | production model, last retrain, drift |
| `/ingest [--seasons 2026]` | occasional | data refresh + feature rebuild only |
| `/snapshot <game>` | occasional | live game state without predicting |
| `/promote [--candidate ensemble_v1]` | occasional | run the promotion gate on a candidate |

`<game>` accepts any of: `2025_01_DAL_PHI` · `DAL@PHI` / `DAL at PHI` (left = away) ·
`PHI vs DAL` (left = home) · `PHI` / `Eagles` / `Philadelphia` (that team's game this week) ·
a Kalshi ticker `KXNFLGAME-26AUG29CHITEN-TEN` · an ESPN event id. Add `--season`/`--week`
when the matchup is ambiguous. Source auto-resolves: a finished game with local play-by-play
replays from disk; otherwise it polls ESPN live.

Example output:

```
╭──── CIN @ KC  2024_02_CIN_KC ────╮
│ CIN 25 — KC 23   Q4 2:00         │
│ KC, 1st & 10 at the 59           │
│ P(KC win)   0.3584               │
│ confidence  0.556                │
│ trade gate  open (15/85 band)    │
│ member             P(KC)  vs bl. │
│ naive_score_time  0.4089  +0.0505│
│ xgboost_winprob   0.4306  +0.0722│
│ tf_seq_winprob    0.3898  +0.0314│
│ raw 0.4097 → calibrated 0.3584   │
╰──────────────────────────────────╯
```

Equivalent CLI: `uv run nfl predict …`, `uv run nfl retrain`, `uv run nfl evaluate --ledger`,
`uv run nfl ledger show`, `uv run nfl promote`, `uv run nfl ingest nflverse|weather`,
`uv run nfl features build`, `uv run nfl train --model naive|xgb|seq|mlp|ensemble`.

## Setup

```bash
uv sync --extra gpu --dev      # local, with CUDA wheels (CPU-only: uv sync --dev)
uv run pre-commit install
uv run pytest                  # ~1,400 tests, no network
/ingest --seasons 2010-2025    # ~5 min: nflverse + weather + features (770k plays × 775 features)
/retrain --skip-ingest         # ~4 min on GPU: trains, gates, promotes
```

Weekly automation: `bash scripts/install_cron.sh --install` (Tuesday 06:00 local → `/retrain`).

---

## Accuracy (production `ensemble_v1`, held-out test seasons 2024–25, 570 games)

| Point in game | Accuracy | Log-loss | Brier | AUC |
|---|---|---|---|---|
| Q4, 10:00 left | 83.8% | 0.382 | 0.110 | 0.925 |
| Q4, 5:00 left | 86.9% | 0.310 | 0.095 | 0.946 |
| Q4, 2:00 left | 88.7% | 0.244 | 0.079 | 0.963 |
| All plays, whole game | 74.4% | 0.495 | 0.164 | 0.837 |
| **Not-collapsed rows (model p in 15–85%)** | **65.5%** | 0.621 | 0.215 | 0.714 |

Validation season (2024, used for calibration/gating): ensemble log-loss **0.4475** vs the
score/time-only floor's 0.4667; Q4 marks ~0.02 log-loss better than the floor.

Headline Q4 accuracy is high mostly because many games are already decided. The last row
is the one that matters for trading: inside the band where a position is still available
the model is ~65–70% accurate. **Calibration is good** — probabilities mean what they say:

| model p (Q4 rows) | n | accuracy | mean p | actual home-win rate |
|---|---|---|---|---|
| 0.15–0.25 | 1480 | 77% | .185 | .233 |
| 0.25–0.35 | 916 | 77% | .287 | .228 |
| 0.35–0.45 | 2840 | 62% | .399 | .382 |
| 0.45–0.55 | 1469 | 53% | .502 | .578 |
| 0.55–0.65 | 524 | 60% | .595 | .596 |
| 0.65–0.75 | 1382 | 73% | .705 | .728 |
| 0.75–0.85 | 627 | 80% | .768 | .800 |

*mean p* = the average predicted probability in that band; if it matches the actual win
rate the model is honest. The one soft spot is 0.45–0.55 (model 0.50, home wins 0.58 — a
home-field effect under-weighted in coin-flip endgames).

In-band accuracy by time left: 10:00→68% · 5:00→68% · 2:00→65% · 1:00→64% · <1:00→67%.
By score margin: tied 63% · 1–3 pts 62% · 4–7 pts 74% · 8–10 pts 89%.
One row per game: at 10:00 left ~40% of games are still in the band (67% accuracy there,
95% on the collapsed ones); at 2:00 left ~30% are in band (67% / 98%).

Edge *versus the market* is not yet measured — that needs Kalshi/closing-line joins in the
ledger from live games (`/evaluate` reports it once rows exist).

---

## Model architecture

```
live snapshot (ESPN or local replay)
   → 156 "servable" in-play features: score, clock, possession, down/distance, field
     position, timeouts, cumulative yards/plays/drives/turnovers/penalties, momentum
     windows, situational flags — only features a live feed can reproduce exactly
   → three members on the same row:
       1. logistic regression on 9 score/time features       (the floor; CPU)
       2. XGBoost, CUDA hist, ~215 trees, game-weighted, Q4-selected
       3. TensorFlow GRU over the last 32 plays + current row (GPU, mixed precision)
   → logit-mean of the three
   → phase-aware isotonic calibration (separate map per time-remaining bucket, fit on val)
   → P(home), P(away) = 1 − P(home), confidence = calibrated margin
   → trade gate (15–85 band) + ledger row
```

**Why only in-play features?** 775 features were built and evaluated (220 in-game, 25
baseline, 184 team × 2 sides, 77 player × 2 sides, ~10 context). Drop-one ablation showed
the pre-game team/player/context families *hurt* out-of-sample (tree and NN learners used
game-constant columns to memorise which game a row came from: train log-loss 0.25 vs val
0.49). Even a compressed 10-column prior cost +0.02 val Q4 log-loss. They remain in the
dataset for research; getting them in as a well-regularised prior is the main open lever.

**Feature contribution** (XGBoost gain, production member): score-margin variants ≈ 50%
(`bi__score_diff_home` 22%, `ig__score_diff_home` 15%, `ig__abs_score_diff` 7%,
`ig__largest_lead_home` 3%), then clock, possession, plays/yards, timeouts, drive number,
field-goal-to-win, time of possession, last-5-plays yards, turnovers, expected remaining
possessions at 0.6–1.5% each. Top-10 = 56% of gain.

**Promotion gate** (ADR 0003): a candidate replaces PRODUCTION only if it beats the floor on
val Q4@600/300/120 log-loss *and* Brier, is not worse on the shared not-collapsed rows, was
trained on the same season split, and reproduces its offline probabilities from replayed
live snapshots within 0.02 (serving parity). Failing candidates are kept as artifacts but
never served.

**Training**: rolling season split — train ≤ L−2, validate L−1, test L, where L is the
latest completed regular season. Row weight = 1 / plays-in-game so every game counts
equally. Early stopping and hyper-parameter selection on the val Q4 slices, not overall
log-loss. Full retrain ≈ 4 min on the RTX 5080.

**Confidence**: `margin_uncertainty` — the calibrated distance from 0.5 (chosen empirically
over ensemble disagreement and GRU MC-dropout; disagreement was *anti*-predictive).

---

## System architecture

```
nflverse (pbp/schedules/rosters/injuries 2010→)  ─┐
Open-Meteo ERA5 + forecast (weather)              ├─► data/raw/*/season=YYYY/*.parquet ─► DuckDB views
ESPN site+core API (live score/plays/situation)  ─┤
Kalshi + closing lines  (EVALUATION ONLY)         ─┘

nfl/features/   ingame (220), baseline (25), team (184), player (77), assembly → data/processed/dataset
nfl/models/     naive, gbdt (XGBoost), tf_mlp, tf_seq (GRU), calibration, ensemble, registry, gate
nfl/inference/  resolve_game → LiveFeed snapshot → serving feature row → model → ledger
nfl/logging/    append-only prediction ledger (parquet + DuckDB), evaluate, backtest, drift
nfl/retrain.py  the weekly pipeline
```

Key guarantees, all covered by tests: no market column can reach a feature matrix
(`nfl/data/market_blocklist.py`); every feature builder passes a delete-the-future leakage
test; the serving path and the offline dataset agree column-for-column on replayed
snapshots (670 columns, 0 mismatches); ESPN and replay feeds agree on 20 state fields.

## Data

- **nflverse** play-by-play 2010–present (770k plays, 4,363 games), schedules (with ESPN
  crosswalk, coaches, roof), weekly rosters, injuries, depth charts, snap counts, NGS, PFR,
  participation, QBR. Market columns are split into a separate `market_only` table at ingest.
- **Weather**: Open-Meteo ERA5 hourly at kickoff for outdoor games (53-venue reference table
  with neutral-site fixes); forecast for upcoming games.
- **Live**: ESPN scoreboard + summary + core `situation` (the only source of timeouts);
  historical replay feed built from nflverse for train/serve parity.
- **Sentiment/injuries**: rule-based scorers (no cloud LLMs; a local Ollama scorer exists
  behind a flag for benchmarking).
- **Market** (eval only): Kalshi read-only client (2026 dollar-string schema, liquidity
  filter), nflverse closing lines de-vigged.

## Ledger and monitoring

Every `/predict` appends a row (47 columns: game, snapshot clock, score state, p_home,
confidence, model version, feature hash, git sha, latency, market-at-time fields flagged
eval-only). `/ledger` resolves outcomes; `/evaluate` reports log-loss/Brier/accuracy by
game phase, reliability table, Brier skill vs closing line, flat-stake ROI (Kalshi fee
coefficient parameterised — verify before trusting net numbers), CLV, and a **drift flag**:
rolling-window ledger log-loss more than +0.02 above the served model's training-time
baseline ⇒ `/retrain`.

## Repo layout

```
CLAUDE.md            roles, hard rules, commands        docs/DATA_MODEL.md   exhaustive attribute spec
docs/ADR/            0001 no market inputs · 0002 storage · 0003 v1 architecture (+ serving parity)
nfl/data/            ingest, schemas, live feeds, market, sentiment, reference tables
nfl/features/        builders, assembly, splits, leakage checks
nfl/models/          members, calibration, ensemble, registry, feature sets, sweeps, gate
nfl/inference/       /predict path and game resolution
nfl/logging/         ledger, evaluate, backtest
scripts/             weekly_retrain.sh, install_cron.sh
.claude/commands/    slash commands
data/ (gitignored)   raw, processed, models (PRODUCTION.json pointer), ledger, reports
```

## Known gaps / next work

- Pre-game team/player knowledge adds no measurable lift yet; QB-availability features are
  all-null (injury join bug) — fix, then re-ablate as a regularised prior.
- Isotonic calibrator saturates to exact 0/1 on blowouts; Platt or clipping would be safer.
- 0.45–0.55 band under-weights home field.
- Kalshi fee formula unverified; edge vs market unmeasured until live rows accumulate.
- GCP deployment (Dockerfile/GCS sync) not started.

## Development

Feature branches + PRs; `main` requires the CI `test` check. Fable (architect) reviews,
Opus writes code, Haiku does lookups. `uv run ruff check . && uv run ruff format . && uv run pytest`
before every commit.
