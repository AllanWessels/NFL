---
description: Run the promotion gate on a candidate model and promote it to PRODUCTION if it passes
argument-hint: [--candidate ensemble_v1] [--version V] [--force]
allowed-tools: Bash(uv run nfl promote:*)
---

Run `uv run nfl promote $ARGUMENTS` (default `--candidate ensemble_v1`). Print the gate table,
say whether PRODUCTION moved and to which version. Never pass --force unless the user typed it.
