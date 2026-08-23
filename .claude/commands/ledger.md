---
description: Show recent predictions from the ledger and resolve outcomes
argument-hint: [--last 20]
allowed-tools: Bash(uv run nfl ledger:*)
---

Run `uv run nfl ledger resolve` then `uv run nfl ledger show $ARGUMENTS` (default --last 20).
Present the table compactly: game, snapshot clock, p_home, confidence, model version,
resolved outcome and per-row Brier where available.
