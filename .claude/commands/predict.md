---
description: Predict win probability for an NFL game (optionally at a point in time)
argument-hint: <game_id> [--at "Q4 05:00"] [--model-version <v>]
allowed-tools: Bash(uv run nfl predict:*)
---

Run the win-probability prediction for the requested game and report the result.

!`uv run nfl predict $ARGUMENTS`

Summarise the output above for the user: the predicted P(home win) / P(away win),
the confidence, the model version used, and the ledger row that was written.
If the command failed, report the error and stop — do not fabricate a probability.
