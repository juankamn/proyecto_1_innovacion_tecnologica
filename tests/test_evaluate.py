"""Tests for evaluate command behavior when model fits on full train."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.evaluation.evaluate import model_was_fit_on_full_train, print_evaluation_summary


class TestEvaluateSkipMetrics(unittest.TestCase):
    def test_model_was_fit_on_full_train_reads_meta(self) -> None:
        loaded = {"meta": {"fit_on_full_train": True}}
        self.assertTrue(model_was_fit_on_full_train(loaded))

        loaded_false = {"meta": {"fit_on_full_train": False}}
        self.assertFalse(model_was_fit_on_full_train(loaded_false))

    def test_print_evaluation_summary_skips_qwk_when_requested(self) -> None:
        result = {
            "metrics_skipped": True,
            "cv_reference": {
                "model": "lgbm_regressor",
                "qwk_mean": 0.41,
                "qwk_std": 0.05,
            },
        }
        with patch("builtins.print") as mock_print:
            print_evaluation_summary(result)
        output = " ".join(str(c[0][0]) for c in mock_print.call_args_list)
        self.assertIn("skipped", output.lower())
        self.assertIn("0.41", output)
        self.assertNotIn("F1 macro", output)

    @patch("src.data.dataset.load_train_val_split")
    @patch("src.evaluation.evaluate.joblib.load")
    def test_run_full_evaluation_skips_when_fit_on_full_train(
        self,
        mock_load: MagicMock,
        mock_split: MagicMock,
    ) -> None:
        from src.evaluation.evaluate import run_full_evaluation

        mock_load.return_value = {
            "pipeline": MagicMock(),
            "meta": {"fit_on_full_train": True, "model": "lgbm_regressor"},
        }
        mock_split.return_value = {
            "X_test": MagicMock(),
            "y_test": MagicMock(),
        }

        with patch(
            "src.evaluation.evaluate.load_cv_metrics_for_model",
            return_value={"model": "lgbm_regressor", "qwk_mean": 0.41, "qwk_std": 0.05},
        ):
            with patch("pathlib.Path.exists", return_value=True):
                result = run_full_evaluation()

        self.assertTrue(result["metrics_skipped"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["cv_reference"]["qwk_mean"], 0.41)


if __name__ == "__main__":
    unittest.main()
