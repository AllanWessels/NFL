"""Small, hand-built reference tables committed to the repo.

These are static lookups (position impact weights, stadium coordinates, team
names and aliases), not ingested data. They live in-repo because they are tiny,
change rarely, and must be reviewable in a diff.
"""

from __future__ import annotations

from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent
POSITION_IMPACT_CSV = REFERENCE_DIR / "position_impact.csv"
TEAM_ALIASES_CSV = REFERENCE_DIR / "team_aliases.csv"

__all__ = ["POSITION_IMPACT_CSV", "REFERENCE_DIR", "TEAM_ALIASES_CSV"]
