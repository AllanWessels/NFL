# ADR 0003 — v1 model architecture

**Status:** accepted 2026-08-23 · **Decider:** Fable (architect)

## Evidence (Phase 5, train ≤2022 / val 2023 / test 2024–25)

Test log-loss at Q4 slices (one row per game):

| model | features | Q4@600s | Q4@300s | Q4@120s | overall |
|---|---|---|---|---|---|
| naive logistic (floor) | 9 | **0.3399** | **0.2928** | **0.2485** | 0.4879 |
| XGBoost | 775 | 0.3643 | 0.3245 | 0.2998 | 0.4773 |
| XGBoost | in-play only (245) | – | – | 0.2518 | 0.4860 |
| TF MLP | 775 | 0.3481 | 0.3119 | 0.2803 | **0.4699** |
| TF GRU seq | 775 | 0.3443 | 0.3022 | 0.2634 | 0.4735 |

Ablation (XGB, val): dropping all game-constant families (`tm__`, `pl__`, `ctx__`) *improves* log-loss by 0.015;
`pl__` alone is worth 0.00003. Train/val gap (0.25 vs 0.49) shows the learners memorise games via
game-constant columns — ~3.4k independent games masquerading as 622k rows.

## Decision

1. **Feature set = in-play state + a compressed pre-game prior.** `bi__` + `ig__` (245 cols) plus ≤10
   prior columns (Elo win-prob diff, power-rating diff, rest-day diff, QB1 availability both sides,
   roof/indoor). The 500+ raw `tm__`/`pl__`/`ctx__` columns stay in the dataset for research but are
   not model inputs in v1.
2. **Game-aware training.** Row weight = 1 / plays-in-game (each game contributes equally); early
   stopping and hyper-parameter selection on the **val Q4 slices**, not overall log-loss.
3. **Ensemble = naive logistic + XGBoost(GPU) + TF GRU sequence model**, combined by `logit_mean`
   (or logistic stacker if it wins on val Q4), then **phase-aware calibration** (seconds-remaining
   buckets) fit on val.
4. **Confidence** is chosen empirically by `confidence_reliability` on val Q4 rows, from five
   candidates: member disagreement, GRU MC-dropout σ, the calibrated margin `2·|p−0.5|`, and the
   two blends (σ with disagreement, σ with margin). The winner must be **negatively** rank-
   correlated with per-bin Brier — more confidence, less error — or it is a reassuring number that
   means nothing. The ranking table and the winner both go in the ensemble manifest.
5. **Promotion gate:** a candidate is promoted to PRODUCTION only if it beats the naive floor on
   val Q4@600/300/120 log-loss *and* Brier, and is not worse on the not-collapsed subset.
   **The not-collapsed subset is one row set, defined by the floor's probability band** `[0.15,
   0.85]`, so candidate and floor are scored on identical rows and a difference in the number is a
   difference in skill. Keying it on each model's own band — the first cut — compared populations
   rather than models: a sharper candidate collapses more rows, keeps the harder ones, and loses
   the comparison for being better. That variant survives as `not_collapsed_own`, reported for
   information and never gated. `nfl promote` is the gate's command-line face, and it writes
   `data/models/PRODUCTION.json` when the gate passes.
6. Market data remains evaluation-only (ADR 0001); "not collapsed" for trading is defined on the
   *market* price once the ledger has Kalshi snapshots, on the floor's p until then.

## Consequences

- Player features (`pl__`) are retained but de-prioritised; no further player-feature work until a
  compressed prior shows lift.
- TF on the RTX 5080 is practical (PTX JIT ≈ 24 s cold); XGBoost uses `grow_gpu_hist`.
- Retraining is cheap (< 2 min end-to-end), so weekly in-season retrains are the default.

## Amendment — serving parity (2026-08-23)

The v1 line-up above was trained and measured entirely offline. Serving it exposed three ways the
row `nfl predict` builds is not the row the model was trained on. All three are now closed, and
item 1's feature set and item 5's gate are amended accordingly.

1. **89 of the 245 in-play columns arrive null at serve time.** A `LiveSnapshot` carries a play's
   type, yardage, scores and coarse turnover/penalty flags; it carries no EPA and no per-play
   statistical detail, so every aggregate built on that detail is null on the serving path and
   populated in training. This was known (`SNAPSHOT_UNKNOWABLE_COLUMNS`) and *not acted on*: the
   models were trained on all 245. **The v1 feature set is now `servable` — in-play minus those
   89, 156 columns** (`nfl.models.feature_sets.V1_FEATURE_SET`). The pre-game prior is dropped from
   the default: two of its ten columns (`{home,away}_pl__qb1_adj_epa_per_dropback`) are rebuilt
   from the snapshot's injury list rather than as of kickoff and disagreed on 134 of 134 sampled
   snapshots. `servable_v1` (156 + the eight servable prior columns) was trained alongside and is
   *worse* on all three val Q4 marks, so `servable` ships.
2. **The GRU was served a one-play window.** `predict` built a single feature row, so the sequence
   member windowed over one play — a different model from the 32-play one that was measured. The
   serving path now builds the whole play history from the snapshot
   (`ingame.snapshot_history_rows`) and reads the answer off the last row.
3. **XGBoost re-quantised itself at inference.** `predict_proba` built a `QuantileDMatrix`, which
   derives its bin edges from the batch it is handed. Scored one row at a time — which is what
   `nfl predict` does — every feature collapsed into a single bin and the model emitted a
   near-constant 0.47 (correlation **0.014** with its own batch-scored output; val log-loss 0.699,
   worse than a coin flip). This, not a feature-distribution shift, was the BAL @ KC pathology:
   the same snap scored 0.45 alone, 0.65 in its game and 0.82 in the season batch. The prediction
   path now uses a plain `DMatrix`. No monotone constraint was added: with correct scoring the
   model is already monotone in `score_diff` to within 0.05 over a ±21-point sweep.

**Item 5 gains a third condition.** Every other number in the gate is measured offline, so on its
own the gate promotes a model whose production behaviour is unmeasured. `nfl promote` now samples
20 `(game, Q4 time)` snapshots from val, scores each through `predict_game` on the replay feed and
through the offline dataset row for the same play, and requires
**`|p_serving − p_offline| < 0.02`** on every one. Snapshots are taken at the sampled play's *own*
clock: asking for a round `Q4 05:00` puts the snapshot ~11 s away from the snap the dataset row
describes, and the measurement would be about that gap rather than about the pipeline.
`--parity-samples 0` skips it, and then the table is a comparison of two artifacts rather than a
promotion decision.
