"""Competition metrics: QWK (primary), F1 macro, balanced accuracy, per-class tables."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
    }


def classification_metrics_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    include_report: bool = False,
) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    if include_report:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
    return metrics


def metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
