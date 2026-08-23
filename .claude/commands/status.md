---
description: Project status — production model, last retrain, ledger size, drift
allowed-tools: Bash(cat:*), Bash(ls:*), Bash(uv run nfl evaluate:*), Bash(uv run nfl ledger show:*)
---

Report concisely: PRODUCTION pointer (`cat data/models/PRODUCTION.json` — model, version, promoted_at,
val log-loss), the newest `data/reports/retrain-*.md` summary table, ledger row count from
`uv run nfl ledger show --last 1`, and the drift line from `uv run nfl evaluate --ledger`.
