"""Tests for QWK threshold optimization."""
from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.thresholds import (
    optimize_thresholds,
    round_with_thresholds,
)


class TestThresholds(unittest.TestCase):
    def test_round_with_thresholds_default_cuts(self) -> None:
        scores = np.array([0.0, 0.6, 1.2, 2.0, 3.0])
        expected = np.array([0, 1, 1, 2, 3])
        result = round_with_thresholds(scores, [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(result, expected)

    def test_optimize_thresholds_perfect_separation(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        raw_preds = np.array([0.1, 0.2, 1.1, 1.2, 2.1, 2.2, 3.0, 3.1])
        thresholds = optimize_thresholds(y_true, raw_preds)
        rounded = round_with_thresholds(raw_preds, thresholds)
        from sklearn.metrics import cohen_kappa_score

        qwk = cohen_kappa_score(y_true, rounded, weights="quadratic")
        self.assertGreaterEqual(qwk, 0.99)


if __name__ == "__main__":
    unittest.main()
