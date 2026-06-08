"""Serialize QWK-optimized thresholds into final_model_meta.json after fit-final."""
from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from src.models.registry import ENSEMBLE_REGRESSION_MODELS


def extract_threshold_meta(pipe: Pipeline, model_name: str) -> dict[str, Any]:
    if model_name not in ENSEMBLE_REGRESSION_MODELS:
        return {}
    model = pipe.named_steps.get("model")
    if model is None:
        return {}
    if model_name == "regression_vote" and hasattr(model, "get_thresholds"):
        return {
            "thresholds": [t.tolist() for t in model.get_thresholds()],
        }
    if hasattr(model, "thresholds_"):
        return {"thresholds": model.thresholds_.tolist()}
    return {}
