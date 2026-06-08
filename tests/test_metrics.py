"""Tests for evaluation metrics."""
from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.metrics import quadratic_weighted_kappa


class TestMetrics(unittest.TestCase):
    def test_qwk_perfect_prediction(self) -> None:
        y = np.array([0, 1, 2, 3, 1, 2])
        self.assertAlmostEqual(quadratic_weighted_kappa(y, y), 1.0)


if __name__ == "__main__":
    unittest.main()
