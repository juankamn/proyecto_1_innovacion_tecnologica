"""
Dataset cleaning and sklearn ColumnTransformer (notebook section 10).

Flow: fix invalid values → drop leakage/high-null cols → derived features → imputer/scaler/encoder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DROP_SEASON_COLUMNS,
    ID_COLUMNS,
    NULL_COLUMN_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    USE_ITERATIVE_IMPUTER,
)
from src.data.variable_families import ORDINAL_INT_COLUMNS, pciat_columns


def get_variable_families() -> dict[str, list[str]]:
    from src.data.variable_families import VARIABLE_CATEGORIES

    return VARIABLE_CATEGORIES


# --- Row/column cleaning (before modeling split) ---
def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known sentinel values with NaN (BMI, weight, CGAS cap)."""
    out = df.copy()
    if "Physical-BMI" in out.columns:
        out.loc[out["Physical-BMI"] == 0, "Physical-BMI"] = np.nan
    if "Physical-Weight" in out.columns:
        out.loc[out["Physical-Weight"] == 0, "Physical-Weight"] = np.nan
    if "CGAS-CGAS_Score" in out.columns:
        out.loc[out["CGAS-CGAS_Score"] > 100, "CGAS-CGAS_Score"] = np.nan
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BIA-BIA_DEE" in out.columns and "BIA-BIA_BMR" in out.columns:
        out["BIA_DEE_minus_BMR"] = out["BIA-BIA_DEE"] - out["BIA-BIA_BMR"]
    if "Physical-BMI" in out.columns and "Basic_Demos-Age" in out.columns:
        age = out["Basic_Demos-Age"].replace(0, np.nan)
        out["BMI_age_ratio"] = out["Physical-BMI"] / age
    return out


def drop_season_columns(df: pd.DataFrame) -> pd.DataFrame:
    season_cols = [
        c
        for c in df.columns
        if c != TARGET_COLUMN and ("-Season" in c or "Enroll_Season" in c)
    ]
    if not season_cols:
        return df
    return df.drop(columns=season_cols)


def clean_for_modeling(
    df: pd.DataFrame,
    *,
    null_threshold: float = NULL_COLUMN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (df_eda, df_for_modeling).

    Drops rows without sii, removes PCIAT (leakage), and columns above null_threshold.
    """
    work = fix_invalid_values(df)
    work = work.dropna(subset=[TARGET_COLUMN])
    leakage_cols = [c for c in pciat_columns() + ID_COLUMNS if c in work.columns]
    work = work.drop(columns=leakage_cols)

    null_pct = work.isnull().mean()
    drop_cols = null_pct[null_pct > null_threshold].index.tolist()
    keep_paq_c = "PAQ_C-PAQ_C_Total"
    drop_cols = [c for c in drop_cols if c != keep_paq_c]
    work = work.drop(columns=drop_cols)

    work = add_derived_features(work)
    if DROP_SEASON_COLUMNS:
        work = drop_season_columns(work)

    df_for_modeling = work.copy()
    return work, df_for_modeling


def impute_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Median/mode imputation for exploratory analysis only (not for model fit)."""
    out = df.copy()
    num_cols = out.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = out.select_dtypes(include=["object"]).columns
    for col in num_cols:
        if col != TARGET_COLUMN:
            out[col] = out[col].fillna(out[col].median())
    for col in cat_cols:
        mode = out[col].mode()
        if len(mode):
            out[col] = out[col].fillna(mode.iloc[0])
    return out


def _numeric_imputer():
    if USE_ITERATIVE_IMPUTER:
        # BayesianRidge (default) overflows on this dataset; use a NaN-tolerant estimator.
        return IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                max_iter=50,
                random_state=RANDOM_STATE,
            ),
            max_iter=10,
            random_state=RANDOM_STATE,
            initial_strategy="median",
            skip_complete=True,
        )
    return SimpleImputer(strategy="median")


# --- sklearn preprocessing pipeline (fit on train, transform train/test) ---
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric (impute+scale), ordinal int (impute), categorical (impute+one-hot)."""
    ordinal_int_cols = [c for c in X.columns if c in ORDINAL_INT_COLUMNS]
    num_cols_model = [
        c
        for c in X.select_dtypes(include=["float64", "int64"]).columns
        if c not in ordinal_int_cols
    ]
    cat_cols_model = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", _numeric_imputer()),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols_model),
            ("ord_int", ordinal_pipe, ordinal_int_cols),
            ("cat", categorical_pipe, cat_cols_model),
        ]
    )
