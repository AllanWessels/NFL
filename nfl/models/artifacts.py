"""Where a trained model lives on disk, and what it carries with it.

Every model in this package writes ``data/models/<name>/<version>/`` holding

``manifest.json``
    Name, version, git sha, train seasons, the exact feature list, the metrics
    the fit reported and a UTC ``created_at``. Enough to answer "what was this,
    and on what?" without loading the model.
``<something>.{joblib,ubj}``
    The estimator itself, in whatever format its library round-trips.

``version`` is a UTC timestamp ``YYYYMMDDTHHMMSS`` (Phase 5 conventions), which
sorts lexicographically and never collides in practice because a fit takes more
than a second.

:func:`assert_model_input_frame` is the training-time gate that ADR 0001 asks
for: no market column, no ``label__``, no ``meta__`` in a design matrix. It is
cheap, so call it on every frame handed to ``fit`` or ``predict_proba``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl.config import MODELS_DIR
from nfl.data.market_blocklist import assert_no_market_columns
from nfl.features.assemble import LABEL_PREFIX, META_PREFIX

__all__ = [
    "MANIFEST_NAME",
    "VERSION_FORMAT",
    "assert_model_input_frame",
    "git_sha",
    "latest_version",
    "model_dir",
    "new_version",
    "read_manifest",
    "write_manifest",
]

MANIFEST_NAME = "manifest.json"
VERSION_FORMAT = "%Y%m%dT%H%M%S"


def new_version(now: datetime | None = None) -> str:
    """A fresh artifact version: the UTC timestamp ``YYYYMMDDTHHMMSS``."""
    return (now or datetime.now(UTC)).strftime(VERSION_FORMAT)


def git_sha(root: Path | None = None) -> str | None:
    """The current commit, or ``None`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or Path.cwd()),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    sha = out.stdout.strip()
    return sha or None


def model_dir(name: str, version: str, *, root: Path | None = None) -> Path:
    """``data/models/<name>/<version>`` (created on demand)."""
    path = (root or MODELS_DIR) / name / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_version(name: str, *, root: Path | None = None) -> Path:
    """The newest artifact directory for ``name``; raises if there is none."""
    base = (root or MODELS_DIR) / name
    versions = sorted(p for p in base.glob("*") if (p / MANIFEST_NAME).is_file())
    if not versions:
        raise FileNotFoundError(f"no trained {name!r} artifact under {base}")
    return versions[-1]


def write_manifest(
    path: Path,
    *,
    name: str,
    version: str,
    feature_names: Iterable[str],
    metrics: Mapping[str, float] | None = None,
    train_seasons: Iterable[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into ``path`` and return the file written."""
    features = [str(c) for c in feature_names]
    payload: dict[str, Any] = {
        "name": name,
        "version": version,
        "git_sha": git_sha(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_seasons": sorted(int(s) for s in (train_seasons or [])),
        "n_features": len(features),
        "feature_names": features,
        "metrics": {k: float(v) for k, v in dict(metrics or {}).items()},
    }
    payload.update(dict(extra or {}))
    target = path / MANIFEST_NAME
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def read_manifest(path: Path) -> dict[str, Any]:
    """Read ``manifest.json`` from an artifact directory (or the file itself)."""
    target = path if path.is_file() else path / MANIFEST_NAME
    data: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return data


def assert_model_input_frame(X: pl.DataFrame, *, context: str = "model input") -> None:
    """Fail loudly if a design matrix carries market, label or meta columns.

    The three ways a win-probability model can cheat, all caught in one place:
    a betting line (ADR 0001), the answer (``label__home_win``), or an identity
    column that stands in for it (``meta__game_id`` memorises the outcome).
    """
    assert_no_market_columns(X, context=context)
    banned = [c for c in X.columns if c.startswith((LABEL_PREFIX, META_PREFIX))]
    if banned:
        raise AssertionError(
            f"{context}: label/meta columns must never be model inputs: {sorted(banned)}"
        )
