"""Smoke tests for ordinal regression ensemble pipelines."""
from __future__ import annotations

import unittest

from src.data.dataset import load_train_val_split
from src.models.ensemble_regression import (
    fit_pipeline_with_optional_weights,
    get_ensemble_regression_pipelines,
)


class TestEnsembleRegression(unittest.TestCase):
    def test_ensemble_pipelines_fit_small_sample(self) -> None:
        data = load_train_val_split()
        X = data["X_train"].iloc[:60]
        y = data["y_train"].iloc[:60]
        preprocessor = data["preprocessor"]
        pipelines = get_ensemble_regression_pipelines(preprocessor)

        for name, pipe in pipelines.items():
            with self.subTest(model=name):
                if name == "regression_vote":
                    pipe.fit(X, y)
                else:
                    fit_pipeline_with_optional_weights(
                        pipe, X, y, use_sample_weights=False
                    )
                preds = pipe.predict(X.iloc[:5])
                self.assertEqual(len(preds), 5)
                self.assertTrue(set(preds).issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
