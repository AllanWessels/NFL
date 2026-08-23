"""Inference entrypoint: predict(game_id, at) -> win probability + confidence."""

from __future__ import annotations

from nfl.inference.predict import (
    TRADE_GATE_HIGH,
    TRADE_GATE_LOW,
    ModelRef,
    ParityReport,
    PredictionError,
    PredictionResult,
    SnapshotSummary,
    check_parity,
    predict_game,
    render_prediction,
)

__all__ = [
    "TRADE_GATE_HIGH",
    "TRADE_GATE_LOW",
    "ModelRef",
    "ParityReport",
    "PredictionError",
    "PredictionResult",
    "SnapshotSummary",
    "check_parity",
    "predict_game",
    "render_prediction",
]
