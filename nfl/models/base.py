"""Common model interface."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable
import numpy as np
import polars as pl

@dataclass
class FitResult:
    model_name: str
    version: str
    n_train: int
    n_val: int
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

@runtime_checkable
class WinProbModel(Protocol):
    name: str
    def fit(self, X: pl.DataFrame, y: np.ndarray, X_val: pl.DataFrame | None = None, y_val: np.ndarray | None = None) -> FitResult: ...
    def predict_proba(self, X: pl.DataFrame) -> np.ndarray: ...  # P(home win), shape (n,)
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "WinProbModel": ...
