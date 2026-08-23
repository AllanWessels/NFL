---
description: Predict win probability for an NFL game (optionally at a point in time)
argument-hint: <game> [--season 2025] [--week 1] [--at "Q4 05:00"] [--source auto|espn|replay] [--model production|name[:version]] [--no-ledger]
allowed-tools: Bash(uv run nfl predict:*)
---

Run the win-probability prediction for the requested game and report the result.
Arguments are passed straight through to the CLI, so `--season`, `--week`, `--at`,
`--source`, `--model` and `--no-ledger` all work as documented by
`uv run nfl predict --help`.

`<game>` does not have to be an nflverse id. All of these resolve (case-insensitive):

| spec | meaning |
|---|---|
| `2025_01_DAL_PHI` | nflverse game id, validated against the schedules table |
| `DAL@PHI`, `DAL at PHI` | left team is **away**, right team is **home** |
| `PHI vs DAL` | left team is **home**, right team is **away** |
| `AWAY@PHI` | wildcard side — Philadelphia's game, at home |
| `PHI`, `Eagles`, `Philadelphia` | that team's game this week (nearest kickoff within ±7 days) |
| `KXNFLGAME-25SEP04DALPHI` | Kalshi event or market ticker |
| `401772510` | ESPN event id |

A spec that matches more than one game inside the ±7-day window resolves to the
nearest kickoff and says so. A spec that matches games only outside the window —
divisional opponents play twice a season — is refused until `--season`/`--week`
narrows it. Quote specs containing `@` or spaces.

!`uv run nfl predict $ARGUMENTS`

Summarise the output above for the user: **the resolution line** (which game the
spec turned into, and why, when the spec was not already a game id), the
predicted P(home win) / P(away win), the confidence, the model name and version
used, the trade gate (`collapsed` means the probability sits outside the 15/85
band), and the ledger row that was written. Mention any warning line — a missing
PRODUCTION pointer, features the serving path could not produce, an ambiguous
spec that was resolved to the nearest kickoff — verbatim rather than smoothing it
over. If the command failed, report the error (including the list of accepted
game formats, if it printed one) and stop — do not fabricate a probability.
