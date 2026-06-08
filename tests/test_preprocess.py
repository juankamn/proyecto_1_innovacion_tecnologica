"""Tests for modeling-ready data shape after cleaning."""
from __future__ import annotations

import unittest

import numpy as np

from src.config import EXPECTED_MODELING_SHAPE
from src.data.dataset import load_modeling_frame, load_train_val_split
from src.data.preprocess import build_preprocessor


class TestPreprocess(unittest.TestCase):
    def test_modeling_frame_shape(self) -> None:
        df = load_modeling_frame()
        self.assertEqual(df.shape, EXPECTED_MODELING_SHAPE)
        self.assertIn("sii", df.columns)

    def test_derived_features_present(self) -> None:
        df = load_modeling_frame()
        self.assertIn("BIA_DEE_minus_BMR", df.columns)
        self.assertIn("BMI_age_ratio", df.columns)

    def test_season_columns_dropped(self) -> None:
        df = load_modeling_frame()
        season_cols = [c for c in df.columns if "-Season" in c or "Enroll_Season" in c]
        self.assertEqual(season_cols, [])

    def test_iterative_imputer_fits_without_nan_error(self) -> None:
        data = load_train_val_split()
        X = data["X_train"].iloc[:120]
        preprocessor = build_preprocessor(X)
        transformed = preprocessor.fit_transform(X)
        self.assertFalse(np.isnan(transformed).any())


if __name__ == "__main__":
    unittest.main()
