"""Project-wide paths and settings."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LEDGER_DIR = DATA_DIR / "ledger"
DUCKDB_PATH = DATA_DIR / "nfl.duckdb"
SEASON_START = 2010
