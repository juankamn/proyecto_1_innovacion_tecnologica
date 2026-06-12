"""Tests for OOF report helpers."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.oof_report import _build_limitations_summary, collect_oof_predictions


class TestOofReport(unittest.TestCase):
    def test_collect_oof_predictions_shape_matches_labels(self) -> None:
        rng = np.random.default_rng(42)
        n = 80
        X = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "c": rng.normal(size=n),
            }
        )
        y = pd.Series(rng.integers(0, 4, size=n))

        preds = collect_oof_predictions(
            "decision_tree",
            X,
            y,
            best_params={},
            cv_folds=3,
        )
        self.assertEqual(preds.shape, (n,))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1, 2, 3}))

    def test_build_limitations_summary_flags_minority_class(self) -> None:
        per_class = pd.DataFrame(
            {
                "class": [0, 1, 2, 3],
                "precision": [0.7, 0.4, 0.35, 0.3],
                "recall": [0.8, 0.3, 0.31, 0.09],
                "f1": [0.75, 0.34, 0.33, 0.14],
                "support": [1594, 730, 378, 34],
            }
        )
        summary = _build_limitations_summary(per_class)
        self.assertEqual(summary["minority_class"], 3)
        self.assertEqual(summary["minority_support_oof"], 34)
        self.assertAlmostEqual(summary["minority_recall_oof"], 0.09)


if __name__ == "__main__":
    unittest.main()
