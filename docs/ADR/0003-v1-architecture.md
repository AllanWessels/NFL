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
4. **Confidence** = member disagreement, optionally blended with GRU MC-dropout σ; the definition is
   chosen empirically by `confidence_reliability` on val Q4 rows.
5. **Promotion gate:** a candidate is promoted to PRODUCTION only if it beats the naive floor on
   val Q4@600/300/120 log-loss *and* Brier, and is not worse on the not-collapsed subset.
6. Market data remains evaluation-only (ADR 0001); "not collapsed" for trading is defined on the
   *market* price once the ledger has Kalshi snapshots, on the model's own p until then.

## Consequences

- Player features (`pl__`) are retained but de-prioritised; no further player-feature work until a
  compressed prior shows lift.
- TF on the RTX 5080 is practical (PTX JIT ≈ 24 s cold); XGBoost uses `grow_gpu_hist`.
- Retraining is cheap (< 2 min end-to-end), so weekly in-season retrains are the default.
