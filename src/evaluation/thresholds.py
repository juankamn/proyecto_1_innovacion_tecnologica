"""
Map regressor outputs to ordinal classes 0–3.

ThresholdOrdinalClassifier fits a regressor then optimizes cut points for QWK on train.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.validation import check_is_fitted

DEFAULT_START_THRESHOLDS = (0.5, 1.5, 2.5)


def round_with_thresholds(
    raw_preds: np.ndarray,
    thresholds: np.ndarray | list[float],
) -> np.ndarray:
    """Map continuous scores to ordinal classes 0-3 using three cut points."""
    t = np.asarray(thresholds, dtype=float)
    if t.shape != (3,):
        raise ValueError("thresholds must have exactly 3 values")
    scores = np.asarray(raw_preds, dtype=float)
    return np.where(
        scores < t[0],
        0,
        np.where(
            scores < t[1],
            1,
            np.where(scores < t[2], 2, 3),
        ),
    ).astype(int)


def optimize_thresholds(
    y_true: np.ndarray,
    raw_preds: np.ndarray,
    *,
    start_vals: list[float] | None = None,
) -> np.ndarray:
    """Find threshold triple that maximizes quadratic weighted kappa."""
    y_true = np.asarray(y_true, dtype=int)
    raw_preds = np.asarray(raw_preds, dtype=float)
    x0 = np.array(start_vals or list(DEFAULT_START_THRESHOLDS), dtype=float)

    def objective(thresholds: np.ndarray) -> float:
        ordered = np.sort(thresholds)
        rounded = round_with_thresholds(raw_preds, ordered)
        return -float(cohen_kappa_score(y_true, rounded, weights="quadratic"))

    result = minimize(objective, x0=x0, method="Powell")
    if not result.success:
        return np.sort(x0)
    return np.sort(result.x)


class ThresholdOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a regressor; discretize continuous predictions with optimized thresholds."""

    def __init__(
        self,
        regressor: RegressorMixin,
        *,
        thresholds: np.ndarray | list[float] | None = None,
        optimize_on_fit: bool = True,
        start_thresholds: list[float] | None = None,
        min_class: int = 0,
        max_class: int = 3,
    ):
        self.regressor = regressor
        self.thresholds = thresholds
        self.optimize_on_fit = optimize_on_fit
        self.start_thresholds = start_thresholds
        self.min_class = min_class
        self.max_class = max_class

    def fit(self, X, y, sample_weight=None):
        self.regressor_ = self.regressor
        if sample_weight is not None:
            self.regressor_.fit(X, y, sample_weight=sample_weight)
        else:
            self.regressor_.fit(X, y)
        y_arr = np.asarray(y, dtype=float)
        raw_train = self.regressor_.predict(X)
        if self.thresholds is not None:
            self.thresholds_ = np.sort(np.asarray(self.thresholds, dtype=float))
        elif self.optimize_on_fit:
            self.thresholds_ = optimize_thresholds(
                y_arr.astype(int),
                raw_train,
                start_vals=self.start_thresholds,
            )
        else:
            self.thresholds_ = np.array(DEFAULT_START_THRESHOLDS, dtype=float)
        self.classes_ = np.arange(self.min_class, self.max_class + 1)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        raw = self.regressor_.predict(X)
        return round_with_thresholds(raw, self.thresholds_)

    def predict_continuous(self, X) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        return np.asarray(self.regressor_.predict(X), dtype=float)
