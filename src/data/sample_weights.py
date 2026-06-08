"""Inverse-frequency weights from target quantile bins (ordinal regressors in train-ensemble/tune)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_sii_bin_weights(
    y: pd.Series | np.ndarray,
    *,
    n_bins: int = 4,
) -> np.ndarray:
    """Inverse-frequency weights from quantile bins of the target."""
    y_series = pd.Series(y)
    bins = min(n_bins, y_series.nunique())
    if bins < 2:
        return np.ones(len(y_series), dtype=float)
    try:
        bin_labels = pd.qcut(y_series, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return np.ones(len(y_series), dtype=float)
    counts = bin_labels.value_counts()
    weights_map = {label: len(y_series) / (len(counts) * count) for label, count in counts.items()}
    return bin_labels.map(weights_map).astype(float).to_numpy()
