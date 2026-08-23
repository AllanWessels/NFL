---
description: Ingest data — nflverse play-by-play/schedules/rosters and weather for a season
argument-hint: [--seasons 2026]
allowed-tools: Bash(uv run nfl ingest:*), Bash(uv run nfl features build:*)
---

Run, in order, from the repo root (default `--seasons` = current season if none given):
1. `uv run nfl ingest nflverse $ARGUMENTS`
2. `uv run nfl ingest weather $ARGUMENTS`
3. `uv run nfl features build --seasons 2010-<latest>`
Report rows written per table, weather games fetched/skipped, and the dataset shape.
