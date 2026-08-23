---
description: Predict win probability for an NFL game (optionally at a point in time)
argument-hint: <game_id> [--at "Q4 05:00"] [--source auto|espn|replay] [--model production|name[:version]] [--no-ledger]
allowed-tools: Bash(uv run nfl predict:*)
---

Run the win-probability prediction for the requested game and report the result.
Arguments are passed straight through to the CLI, so `--at`, `--source`,
`--model` and `--no-ledger` all work as documented by `uv run nfl predict --help`.

!`uv run nfl predict $ARGUMENTS`

Summarise the output above for the user: the predicted P(home win) / P(away win),
the confidence, the model name and version used, the trade gate (`collapsed`
means the probability sits outside the 15/85 band), and the ledger row that was
written. Mention any warning line — a missing PRODUCTION pointer, features the
serving path could not produce — verbatim rather than smoothing it over.
If the command failed, report the error and stop — do not fabricate a probability.
