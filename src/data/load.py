"""Thin wrappers to read Kaggle train.csv / test.csv from data/raw/."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TEST_CSV_PATH, TRAIN_CSV_PATH


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}. "
            "Place train.csv and test.csv under data/raw/ (see docs/instalacion.md)."
        )
    return pd.read_csv(path)


def load_train(path: Path | None = None) -> pd.DataFrame:
    return _read_csv(path or TRAIN_CSV_PATH)


def load_test(path: Path | None = None) -> pd.DataFrame:
    return _read_csv(path or TEST_CSV_PATH)
