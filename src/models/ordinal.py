"""Simple round-and-clip ordinal wrapper (used by Ridge ordinal baseline)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class RegressorToOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a regressor; round and clip predictions to integer class labels."""

    def __init__(self, regressor: RegressorMixin, min_class: int = 0, max_class: int = 3):
        self.regressor = regressor
        self.min_class = min_class
        self.max_class = max_class

    def fit(self, X, y):
        self.regressor_ = self.regressor
        self.regressor_.fit(X, y)
        self.classes_ = np.arange(self.min_class, self.max_class + 1)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        raw = self.regressor_.predict(X)
        rounded = np.rint(raw).astype(int)
        return np.clip(rounded, self.min_class, self.max_class)
