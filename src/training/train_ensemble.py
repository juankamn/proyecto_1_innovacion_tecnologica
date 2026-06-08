"""
Ordinal regression models (LGBM/XGB/ExtraTrees + vote) — appended to train CSVs.

Regressors predict continuous scores; ThresholdOrdinalClassifier maps to sii via QWK thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SKLEARN_N_JOBS,
    configure_plotting,
    ensure_experiment_dirs,
    warn_if_python_joblib_incompatible,
)

from src.data.dataset import load_train_val_split
from src.data.feature_selection import load_top_feature_names
from src.data.sample_weights import calculate_sii_bin_weights
from src.evaluation.evaluate import (
    annotate_holdout_with_levels,
    cross_validate_pipeline,
    evaluate_pipeline,
    save_model_comparison_chart,
)
from src.evaluation.metrics import classification_metrics_extended
from src.evaluation.thresholds import optimize_thresholds, round_with_thresholds
from src.models.ensemble_regression import (
    _extratrees_regressor,
    _lgbm_regressor,
    _xgb_regressor,
    fit_pipeline_with_optional_weights,
    get_extratrees_regressor_pipeline,
    get_lgbm_regressor_pipeline,
    get_regression_vote_pipeline,
    get_xgb_regressor_pipeline,
)
from src.models.registry import ENSEMBLE_REGRESSION_MODELS, MODEL_LEVELS


def _ensemble_apply_selection() -> bool:
    return load_top_feature_names() is not None


def _regressor_factories() -> dict[str, callable]:
    return {
        "lgbm_regressor": _lgbm_regressor,
        "xgb_regressor": _xgb_regressor,
        "extratrees_regressor": _extratrees_regressor,
    }


def _regressor_pipeline_builders() -> dict[str, callable]:
    return {
        "lgbm_regressor": get_lgbm_regressor_pipeline,
        "xgb_regressor": get_xgb_regressor_pipeline,
        "extratrees_regressor": get_extratrees_regressor_pipeline,
        "regression_vote": get_regression_vote_pipeline,
    }


def _mode_vote_rows(preds: np.ndarray) -> np.ndarray:
    from scipy import stats

    result = stats.mode(preds, axis=0, keepdims=False)
    return np.asarray(result.mode, dtype=int).reshape(-1)


def cross_validate_regression_vote_oof(
    X,
    y,
    preprocessor,
    *,
    cv_folds: int = CV_FOLDS,
) -> dict[str, float]:
    """Custom CV for regression_vote: threshold per fold, then majority vote QWK."""
    y_arr = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    factories = _regressor_factories()
    oof_discrete: dict[str, np.ndarray] = {
        name: np.zeros(len(y_arr), dtype=int) for name in factories
    }

    for train_idx, val_idx in cv.split(X, y_arr):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_arr[train_idx]
        sw = calculate_sii_bin_weights(y_train)

        for name, factory in factories.items():
            builder = _regressor_pipeline_builders()[name]
            pipe = builder(preprocessor)
            fit_pipeline_with_optional_weights(pipe, X_train, y_train)
            model = pipe.named_steps["model"]
            preprocess = pipe.named_steps["preprocess"]
            X_val_t = preprocess.transform(X_val)
            raw_val = model.predict_continuous(X_val_t)
            raw_train = model.predict_continuous(preprocess.transform(X_train))
            thresholds = optimize_thresholds(y_train, raw_train)
            oof_discrete[name][val_idx] = round_with_thresholds(raw_val, thresholds)

    vote_preds = _mode_vote_rows(np.vstack([oof_discrete[n] for n in factories]))
    metrics = classification_metrics_extended(y_arr, vote_preds)
    return {
        "qwk_mean": float(metrics["qwk"]),
        "qwk_std": 0.0,
        "f1_macro_mean": float(metrics["f1_macro"]),
        "f1_macro_std": 0.0,
        "balanced_accuracy_mean": float(metrics["balanced_accuracy"]),
        "balanced_accuracy_std": 0.0,
    }


def _append_or_create(df_new: pd.DataFrame, path) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model"], keep="last")
        return combined
    return df_new


def run_train_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Same evaluation pattern as train.py; rows merged into existing CSVs by model name.
    configure_plotting()
    warn_if_python_joblib_incompatible()
    ensure_experiment_dirs()
    if SKLEARN_N_JOBS == 1:
        print("SKLEARN_N_JOBS=1 (sequential CV; set SKLEARN_N_JOBS=-1 on Python 3.14.2+)")
    apply_selection = _ensemble_apply_selection()
    data = load_train_val_split(apply_selection=apply_selection)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    builders = _regressor_pipeline_builders()
    holdout_rows: list[dict] = []
    cv_rows: list[dict] = []

    for name in ENSEMBLE_REGRESSION_MODELS:
        print(f"Training ensemble model {name}...")
        pipe = builders[name](preprocessor)
        if name == "regression_vote":
            pipe.fit(X_train, y_train)
        else:
            fit_pipeline_with_optional_weights(pipe, X_train, y_train)

        holdout = evaluate_pipeline(pipe, X_test, y_test)
        holdout_rows.append(
            {
                "model": name,
                "level": MODEL_LEVELS.get(name, "advanced"),
                **holdout,
            }
        )

        if name == "regression_vote":
            cv = cross_validate_regression_vote_oof(
                X_train, y_train, preprocessor, cv_folds=CV_FOLDS
            )
        else:
            cv = cross_validate_pipeline(pipe, X_train, y_train)
        cv_rows.append({"model": name, "level": MODEL_LEVELS.get(name, "advanced"), **cv})

    holdout_df = annotate_holdout_with_levels(pd.DataFrame(holdout_rows))
    cv_df = pd.DataFrame(cv_rows)

    holdout_path = RESULTS_DIR / "all_models_holdout.csv"
    cv_path = RESULTS_DIR / "all_models_cv.csv"
    holdout_merged = _append_or_create(holdout_df, holdout_path)
    cv_merged = _append_or_create(cv_df, cv_path)
    holdout_merged.to_csv(holdout_path, index=False)
    cv_merged.to_csv(cv_path, index=False)

    best = cv_df.sort_values("qwk_mean", ascending=False).iloc[0]
    print(
        f"Best ensemble CV: {best['model']} "
        f"(QWK={best['qwk_mean']:.4f} ± {best['qwk_std']:.4f})"
    )
    all_chart_path = FIGURES_DIR / "model_comparison_qwk_all.png"
    save_model_comparison_chart(
        holdout_merged,
        all_chart_path,
        metric="qwk",
        title="Model comparison — all models (QWK holdout 80/20)",
    )

    print(f"Saved: {holdout_path}")
    print(f"Saved: {cv_path}")
    print(f"Saved: {all_chart_path}")
    if apply_selection:
        print(f"Feature selection: top {len(X_train.columns)} features")

    return holdout_df, cv_df


def main() -> None:
    run_train_ensemble()


if __name__ == "__main__":
    main()
