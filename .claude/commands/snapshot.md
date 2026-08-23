---
description: Show the live game state snapshot (score, clock, possession, drives) without predicting
argument-hint: <game> [--at "Q4 05:00"] [--source espn|replay]
allowed-tools: Bash(uv run nfl live snapshot:*)
---

Run `uv run nfl live snapshot $ARGUMENTS` and show the panel. Accepts the same game specs as /predict.
