"""Model artifact registry over ``data/models/``.

The registry is a filesystem convention, not a service — same reasoning as
ADR 0002 for storage. Everything it knows lives in JSON next to the weights, so
a trained model stays loadable after a laptop rebuild, a `git clone`, or an
`rsync` to GCS::

    data/models/
      PRODUCTION.json                  # pointer: which (name, version) is live
      gbdt/
        20240117T041500/
          model.json                   # whatever the model's save() wrote
          manifest.json                # provenance for that artifact
        20240118T093012/
          ...

``manifest.json`` carries the three things you need to answer "can I trust this
prediction?" six months from now:

* **git sha** — the exact code that produced the weights.
* **dataset fingerprint** — a hash of the feature column names plus the row
  count per season. Rebuild the dataset, add a feature, backfill a season, and
  the fingerprint moves; a model whose fingerprint no longer matches the current
  dataset is a model you should retrain, not one you should quietly serve.
* **metrics** — what it scored at training time, so a promotion is a comparison
  rather than a guess.

The registry never imports a model class eagerly: ``manifest["model_class"]`` is
a ``"module:QualName"`` string resolved on demand, so loading a GBDT artifact
does not drag TensorFlow into the process.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nfl.config import MODELS_DIR
from nfl.models.base import FitResult

__all__ = [
    "MANIFEST_NAME",
    "PRODUCTION_STAGE",
    "VERSION_FORMAT",
    "Manifest",
    "dataset_fingerprint",
    "git_sha",
    "latest",
    "list_models",
    "list_versions",
    "load",
    "load_stage",
    "model_dir",
    "pointer_path",
    "promote",
    "read_manifest",
    "read_pointer",
    "registry_root",
    "resolve_artifact",
    "save_model",
    "utc_version",
    "write_manifest",
]

#: Version strings are UTC timestamps: sortable as text, unique per second.
VERSION_FORMAT = "%Y%m%dT%H%M%S"

MANIFEST_NAME = "manifest.json"

PRODUCTION_STAGE = "production"


# --------------------------------------------------------------------------- #
# Provenance primitives
# --------------------------------------------------------------------------- #


def utc_version(when: datetime | None = None) -> str:
    """A fresh artifact version: UTC ``YYYYMMDDTHHMMSS``."""
    return (when or datetime.now(UTC)).strftime(VERSION_FORMAT)


def git_sha(repo: Path | None = None, *, short: bool = False) -> str:
    """Current commit sha, or ``"unknown"`` outside a git checkout.

    Never raises: a manifest missing its sha is worth more than a training run
    that dies at the last step because someone exported a tarball without
    ``.git``. A dirty tree is flagged with a ``-dirty`` suffix, because a sha
    that does not describe the working code is a lie.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not sha:
        return "unknown"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def dataset_fingerprint(
    df: pl.DataFrame | None = None,
    *,
    feature_names: list[str] | None = None,
    rows_by_season: dict[str, int] | None = None,
    season_column: str = "meta__season",
) -> dict[str, Any]:
    """Identify the training data: feature column names + row count per season.

    Pass an assembled frame and both halves are derived from it (features via
    :func:`nfl.features.assemble.feature_columns`, counts by ``meta__season``);
    or pass the two pieces directly when the frame is no longer in memory.

    Names are hashed **sorted**, so a harmless column reordering does not
    invalidate a model, while an added, removed or renamed feature does. Row
    counts are per season so a backfill of one year is visible.
    """
    if df is not None:
        if feature_names is None:
            from nfl.features.assemble import feature_columns

            feature_names = feature_columns(df)
        if rows_by_season is None:
            if season_column in df.columns:
                counts = df.group_by(season_column).len().sort(season_column)
                rows_by_season = {
                    str(row[0]): int(row[1]) for row in counts.iter_rows() if row[0] is not None
                }
            else:
                rows_by_season = {"all": int(df.height)}
    names = sorted(feature_names or [])
    seasons = {str(k): int(v) for k, v in sorted((rows_by_season or {}).items())}
    canonical = json.dumps({"feature_names": names, "rows_by_season": seasons}, sort_keys=True)
    return {
        "hash": hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "n_features": len(names),
        "n_rows": sum(seasons.values()),
        "rows_by_season": seasons,
    }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class Manifest:
    """Everything known about one artifact directory."""

    name: str
    version: str
    git_sha: str = ""
    created_at: str = ""
    train_seasons: list[int] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    dataset_fingerprint: dict[str, Any] = field(default_factory=dict)
    #: ``"module:QualName"`` — resolved lazily by :func:`load`.
    model_class: str = ""
    #: Artifact file or directory name, relative to the version directory.
    artifact: str = "model"
    stage: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in payload.items() if k not in known}
        kwargs = {k: v for k, v in payload.items() if k in known}
        manifest = cls(**kwargs)
        if extra:
            manifest.extra = {**manifest.extra, **extra}
        return manifest

    @classmethod
    def from_fit_result(
        cls,
        result: FitResult,
        model: Any = None,
        *,
        train_seasons: list[int] | None = None,
        fingerprint: dict[str, Any] | None = None,
        artifact: str = "model",
    ) -> Manifest:
        """Build a manifest from a model's :class:`~nfl.models.base.FitResult`."""
        return cls(
            name=result.model_name,
            version=result.version or utc_version(),
            git_sha=git_sha(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            train_seasons=list(train_seasons or []),
            feature_names=list(result.feature_names),
            metrics={k: float(v) for k, v in result.metrics.items()},
            dataset_fingerprint=dict(fingerprint or {}),
            model_class=_class_path(model) if model is not None else "",
            artifact=artifact,
            extra={"n_train": result.n_train, "n_val": result.n_val},
        )


def _class_path(obj: Any) -> str:
    klass = obj if isinstance(obj, type) else type(obj)
    return f"{klass.__module__}:{klass.__qualname__}"


def _import_class(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"model_class {path!r} must be 'module:QualName'")
    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def registry_root(root: Path | None = None) -> Path:
    """``data/models`` unless overridden (tests pass ``tmp_path``)."""
    return Path(root) if root is not None else MODELS_DIR


def model_dir(name: str, version: str, root: Path | None = None) -> Path:
    """``<root>/<name>/<version>``."""
    return registry_root(root) / name / version


def pointer_path(stage: str = PRODUCTION_STAGE, root: Path | None = None) -> Path:
    """``<root>/PRODUCTION.json`` for the production stage, ``<STAGE>.json`` otherwise."""
    return registry_root(root) / f"{stage.upper()}.json"


def write_manifest(manifest: Manifest, directory: Path | None = None, **_: Any) -> Path:
    """Write ``manifest.json`` into ``directory`` (default: the artifact's own dir)."""
    target = (
        Path(directory) if directory is not None else model_dir(manifest.name, manifest.version)
    )
    target.mkdir(parents=True, exist_ok=True)
    path = target / MANIFEST_NAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return path


def read_manifest(path: Path) -> Manifest:
    """Read a manifest given the file or its containing version directory."""
    target = Path(path)
    if target.is_dir():
        target = target / MANIFEST_NAME
    if not target.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} at {target}")
    return Manifest.from_dict(json.loads(target.read_text()))


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def list_models(root: Path | None = None) -> list[str]:
    """Model names that have at least one manifested version, sorted."""
    base = registry_root(root)
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and list_versions(d.name, root))


def list_versions(name: str, root: Path | None = None) -> list[str]:
    """Versions of ``name`` that carry a manifest, oldest first."""
    base = registry_root(root) / name
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and (d / MANIFEST_NAME).is_file())


def latest(name: str, root: Path | None = None) -> str:
    """Newest version of ``name``. Version strings are timestamps, so text order wins."""
    versions = list_versions(name, root)
    if not versions:
        raise FileNotFoundError(f"no versions of model {name!r} under {registry_root(root)}")
    return versions[-1]


def resolve_artifact(directory: Path, artifact: str) -> Path:
    """Find what the model's ``save()`` actually wrote.

    Models are free to append their own suffix (``model.json``, ``model.keras``,
    a directory) to the path they are handed, so the manifest's recorded name is
    treated as a stem to match if the exact path is gone.
    """
    exact = directory / artifact
    if exact.exists():
        return exact
    matches = sorted(p for p in directory.glob(f"{artifact}*") if p.name != MANIFEST_NAME)
    if not matches:
        raise FileNotFoundError(f"no artifact matching {artifact!r} in {directory}")
    return matches[0]


def load(name: str, version: str | None = None, root: Path | None = None) -> Any:
    """Load a model instance. ``version=None`` means :func:`latest`."""
    resolved = version or latest(name, root)
    directory = model_dir(name, resolved, root)
    manifest = read_manifest(directory)
    if not manifest.model_class:
        raise ValueError(
            f"manifest for {name}/{resolved} has no model_class; "
            "cannot reconstruct the model (was it written by save_model?)"
        )
    klass = _import_class(manifest.model_class)
    return klass.load(resolve_artifact(directory, manifest.artifact))


def save_model(
    model: Any,
    *,
    name: str | None = None,
    version: str | None = None,
    root: Path | None = None,
    manifest: Manifest | None = None,
    fit_result: FitResult | None = None,
    train_seasons: list[int] | None = None,
    fingerprint: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifact: str = "model",
) -> Manifest:
    """Write ``model`` plus its manifest into ``<root>/<name>/<version>/``.

    The one place artifacts are created, so every registry entry is guaranteed
    to carry a git sha, a timestamp and (when supplied) a dataset fingerprint.
    """
    if manifest is None and fit_result is not None:
        manifest = Manifest.from_fit_result(
            fit_result,
            model,
            train_seasons=train_seasons,
            fingerprint=fingerprint,
            artifact=artifact,
        )
    if manifest is None:
        manifest = Manifest(
            name=str(name or getattr(model, "name", None) or type(model).__name__),
            version=version or utc_version(),
            git_sha=git_sha(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            train_seasons=list(train_seasons or []),
            feature_names=list(getattr(model, "feature_names", []) or []),
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            dataset_fingerprint=dict(fingerprint or {}),
            model_class=_class_path(model),
            artifact=artifact,
        )
    if name is not None:
        manifest.name = name
    if version is not None:
        manifest.version = version
    if not manifest.model_class:
        manifest.model_class = _class_path(model)
    if not manifest.created_at:
        manifest.created_at = datetime.now(UTC).isoformat(timespec="seconds")
    if not manifest.git_sha:
        manifest.git_sha = git_sha()

    directory = model_dir(manifest.name, manifest.version, root)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / manifest.artifact)
    manifest.artifact = resolve_artifact(directory, manifest.artifact).name
    write_manifest(manifest, directory)
    return manifest


# --------------------------------------------------------------------------- #
# Stage pointers
# --------------------------------------------------------------------------- #


def promote(
    name: str,
    version: str | None = None,
    stage: str = PRODUCTION_STAGE,
    root: Path | None = None,
) -> Path:
    """Point ``stage`` at ``name``/``version``; return the pointer path.

    Promotion is deliberately a *pointer write*, not a copy: the artifact never
    moves, so rolling back is another one-line pointer write to the previous
    version rather than a restore.
    """
    resolved = version or latest(name, root)
    directory = model_dir(name, resolved, root)
    manifest = read_manifest(directory)
    manifest.stage = stage
    write_manifest(manifest, directory)

    pointer = {
        "stage": stage,
        "name": name,
        "version": resolved,
        "path": f"{name}/{resolved}",
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "metrics": manifest.metrics,
        "dataset_fingerprint": manifest.dataset_fingerprint,
    }
    path = pointer_path(stage, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True))
    return path


def read_pointer(stage: str = PRODUCTION_STAGE, root: Path | None = None) -> dict[str, Any] | None:
    """The stage pointer, or ``None`` if nothing has been promoted."""
    path = pointer_path(stage, root)
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text()))


def load_stage(stage: str = PRODUCTION_STAGE, root: Path | None = None) -> Any:
    """Load whichever model the stage pointer names."""
    pointer = read_pointer(stage, root)
    if pointer is None:
        raise FileNotFoundError(f"nothing promoted to {stage!r} at {pointer_path(stage, root)}")
    return load(str(pointer["name"]), str(pointer["version"]), root)
