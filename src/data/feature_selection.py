"""
Optional top-N features from permutation_importance_oof.csv (used by final pipeline).

If the CSV is missing, training uses all modeling columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FEATURE_SELECTION_TOP_N, RESULTS_DIR, TARGET_COLUMN


def load_top_feature_names(
    top_n: int | None = None,
    *,
    importance_path: Path | None = None,
) -> list[str] | None:
    """Return top-N feature names from CSV, or None if selection is disabled."""
    n = top_n if top_n is not None else FEATURE_SELECTION_TOP_N
    if n <= 0:
        return None
    if importance_path is not None:
        path = importance_path
    else:
        oof_path = RESULTS_DIR / "permutation_importance_oof.csv"
        legacy_path = RESULTS_DIR / "permutation_importance_final.csv"
        path = oof_path if oof_path.exists() else legacy_path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return None
    sort_col = "importance_mean" if "importance_mean" in df.columns else df.columns[1]
    ranked = df.sort_values(sort_col, ascending=False)["feature"].head(n).tolist()
    return ranked


def apply_feature_selection(
    df: pd.DataFrame,
    *,
    top_n: int | None = None,
    importance_path: Path | None = None,
) -> pd.DataFrame:
    """Keep target plus top-N raw features when importance file is available."""
    selected = load_top_feature_names(top_n, importance_path=importance_path)
    if not selected:
        return df
    keep = [TARGET_COLUMN] + [c for c in selected if c in df.columns and c != TARGET_COLUMN]
    if len(keep) <= 1:
        return df
    return df[keep].copy()
