---
description: Weekly retrain — ingest → features → train members → ensemble → promotion gate → ledger → evaluate
argument-hint: [--skip-ingest] [--dry-run] [--season 2026] [--split 2023/2024/2025]
allowed-tools: Bash(uv run nfl retrain:*)
---

Run `uv run nfl retrain $ARGUMENTS` from the repo root (GPU, ~4 min). Then report, concisely:
the step table, the promotion gate result (PASS/FAIL and which checks), whether PRODUCTION
moved and to which version, the drift line, and the path of the run log under data/reports/.
If the gate refused the candidate, say what is still in production and why the candidate failed.
