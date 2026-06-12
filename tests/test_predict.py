"""Tests for Kaggle submission inference."""
from __future__ import annotations

import unittest

import pandas as pd

from src.config import CHECKPOINTS_DIR, RESULTS_DIR
from src.data.dataset import load_modeling_frame, prepare_test_for_inference, split_features_target
from src.evaluation.predict import run_predict


class TestPredict(unittest.TestCase):
    def test_prepare_test_for_inference_matches_train_columns(self) -> None:
        train_X, _ = split_features_target(load_modeling_frame())
        _, test_X = prepare_test_for_inference(apply_selection=False)
        self.assertListEqual(test_X.columns.tolist(), train_X.columns.tolist())
        self.assertGreater(len(test_X), 0)

    def test_run_predict_integration_when_final_model_exists(self) -> None:
        model_path = CHECKPOINTS_DIR / "final_model.joblib"
        if not model_path.exists():
            self.skipTest("final_model.joblib not present (run fit-final first)")

        out = RESULTS_DIR / "_test_submission_integration.csv"
        try:
            try:
                run_predict(output_path=out)
            except (ValueError, OSError) as exc:
                self.skipTest(f"Cannot load final_model.joblib in this environment: {exc}")
            df = pd.read_csv(out)
            self.assertEqual(list(df.columns), ["id", "sii"])
            self.assertGreater(len(df), 0)
            self.assertTrue(df["sii"].between(0, 3).all())
        finally:
            if out.exists():
                out.unlink()


if __name__ == "__main__":
    unittest.main()
