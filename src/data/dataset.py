"""Train/val splits and test alignment — bridge between preprocess.py and training scripts."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.load import load_test, load_train
from src.data.preprocess import (
    add_derived_features,
    build_preprocessor,
    clean_for_modeling,
    drop_season_columns,
    fix_invalid_values,
)
from src.config import DROP_SEASON_COLUMNS


def load_modeling_frame(*, apply_selection: bool = False) -> pd.DataFrame:
    """Labeled rows only, PCIAT/leakage removed; optional top-N feature subset."""
    raw = load_train()
    _, df_model = clean_for_modeling(raw)
    if apply_selection:
        from src.data.feature_selection import apply_feature_selection

        df_model = apply_feature_selection(df_model)
    return df_model


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor_from_X(X_train: pd.DataFrame) -> ColumnTransformer:
    return build_preprocessor(X_train)


def load_train_val_split(*, apply_selection: bool = False) -> dict[str, Any]:
    """Standard bundle for train/tune/evaluate: X/y splits + unfitted preprocessor."""
    df_model = load_modeling_frame(apply_selection=apply_selection)
    X, y = split_features_target(df_model)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    preprocessor = build_preprocessor_from_X(X_train)
    return {
        "df_model": df_model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


def merge_classes_23(y: pd.Series) -> pd.Series:
    """Map severe/moderate (2,3) into a single class 2 for design experiments."""
    return y.replace({3: 2})


def train_null_drop_columns() -> list[str]:
    """Column drops derived from train only — keeps test preprocessing consistent."""
    from src.config import ID_COLUMNS, NULL_COLUMN_THRESHOLD
    from src.data.variable_families import pciat_columns

    work = fix_invalid_values(load_train())
    work = work.dropna(subset=[TARGET_COLUMN])
    leakage_cols = [c for c in pciat_columns() + ID_COLUMNS if c in work.columns]
    work = work.drop(columns=leakage_cols, errors="ignore")

    null_pct = work.isnull().mean()
    drop_cols = null_pct[null_pct > NULL_COLUMN_THRESHOLD].index.tolist()
    keep_paq_c = "PAQ_C-PAQ_C_Total"
    return [c for c in drop_cols if c != keep_paq_c]


def _transform_rows_for_modeling(df: pd.DataFrame, *, null_drop_columns: list[str]) -> pd.DataFrame:
    from src.config import ID_COLUMNS
    from src.data.variable_families import pciat_columns

    work = fix_invalid_values(df)
    leakage_cols = [c for c in pciat_columns() + ID_COLUMNS if c in work.columns]
    work = work.drop(columns=leakage_cols, errors="ignore")
    work = work.drop(columns=[c for c in null_drop_columns if c in work.columns], errors="ignore")
    work = add_derived_features(work)
    if DROP_SEASON_COLUMNS:
        work = drop_season_columns(work)
    return work


def prepare_test_for_inference(*, apply_selection: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """
    Return (ids, X) for Kaggle test.csv aligned to training feature columns.
    Null-column drops follow train statistics, not test-only rates.
    """
    raw = load_test()
    ids = raw["id"].copy()
    work = _transform_rows_for_modeling(raw, null_drop_columns=train_null_drop_columns())
    train_X, _ = split_features_target(load_modeling_frame(apply_selection=apply_selection))
    X = work.reindex(columns=train_X.columns.tolist())
    return ids, X


def prepare_test_features() -> pd.DataFrame:
    """Feature matrix for test.csv (same columns/order as training, no ids)."""
    _, X = prepare_test_for_inference(apply_selection=False)
    return X
