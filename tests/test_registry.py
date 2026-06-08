"""Smoke tests: all registered pipelines fit on a small subsample."""
from __future__ import annotations

import unittest

from src.data.dataset import load_train_val_split
from src.models.registry import get_all_model_pipelines


class TestRegistry(unittest.TestCase):
    def test_all_pipelines_fit_small_sample(self) -> None:
        data = load_train_val_split()
        X = data["X_train"].iloc[:80]
        y = data["y_train"].iloc[:80]
        preprocessor = data["preprocessor"]
        pipelines = get_all_model_pipelines(preprocessor)

        for name, pipe in pipelines.items():
            with self.subTest(model=name):
                pipe.fit(X, y)
                preds = pipe.predict(X.iloc[:5])
                self.assertEqual(len(preds), 5)


if __name__ == "__main__":
    unittest.main()
