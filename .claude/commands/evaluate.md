---
description: Evaluate live ledger performance — accuracy, calibration, edge vs market, drift flag
argument-hint: [--ledger] [--backtest --model NAME [--version V]] [--season 2025]
allowed-tools: Bash(uv run nfl evaluate:*), Bash(uv run nfl ledger resolve:*)
---

First run `uv run nfl ledger resolve` so finished games have outcomes, then
`uv run nfl evaluate --ledger $ARGUMENTS` (if the user passed --backtest, run that instead).
Summarise: rows evaluated, log-loss/Brier/accuracy overall and at Q4 marks, calibration (ECE),
edge vs market if market columns are present, and the drift flag with its baseline.
Recommend `/retrain` if drift is flagged.
