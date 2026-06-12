"""
10-fold out-of-fold evaluation for the final pipeline (no train leakage).

Produces confusion matrix, per-class metrics, and oof_report_summary.json for the informe.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

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
from src.data.dataset import (
    build_preprocessor_from_X,
    load_modeling_frame,
    load_train_val_split,
    merge_classes_23,
    split_features_target,
)
from src.data.feature_selection import load_top_feature_names
from src.data.sample_weights import calculate_sii_bin_weights
from src.evaluation.evaluate import save_confusion_matrix, save_per_class_metrics
from src.evaluation.metrics import classification_metrics_extended, metrics_per_class, qwk_scorer
from src.evaluation.thresholds import optimize_thresholds, round_with_thresholds
from src.models.ensemble_regression import fit_pipeline_with_optional_weights
from src.models.registry import ENSEMBLE_REGRESSION_MODELS
from src.training.fit_final import _use_merged_classes
from src.training.train_ensemble import (
    _mode_vote_rows,
    _regressor_factories,
    _regressor_pipeline_builders,
)
from src.training.tune import _build_pipeline, select_best_model_from_cv


# --- Config: prefer final_model_meta.json, else infer from tuning + CV tables ---
def _load_report_config() -> dict[str, Any]:
    meta_path = RESULTS_DIR / "final_model_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "model": str(meta["model"]),
            "best_params": meta.get("best_params", {}),
            "merged_classes_23": bool(meta.get("merged_classes_23", False)),
            "feature_selection": bool(meta.get("feature_selection", False)),
        }

    params_path = RESULTS_DIR / "tuning_best_params.json"
    best_params: dict[str, Any] = {}
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            best_params = json.load(f).get("best_params", {})

    model_name = select_best_model_from_cv()
    use_feature_selection = (
        model_name in ENSEMBLE_REGRESSION_MODELS and load_top_feature_names() is not None
    )
    return {
        "model": model_name,
        "best_params": best_params,
        "merged_classes_23": _use_merged_classes(),
        "feature_selection": use_feature_selection,
    }


def _prepare_xy(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    df_model = load_modeling_frame(apply_selection=config["feature_selection"])
    X, y = split_features_target(df_model)
    if config["merged_classes_23"]:
        y = merge_classes_23(y)
    return X, y


def _fit_fold_pipeline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, Any],
) -> Pipeline:
    preprocessor = build_preprocessor_from_X(X_train)
    pipe = _build_pipeline(model_name, preprocessor)
    if best_params:
        pipe.set_params(**best_params)
    if model_name in ENSEMBLE_REGRESSION_MODELS and model_name != "regression_vote":
        fit_pipeline_with_optional_weights(pipe, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _predict_fold(
    pipe: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
) -> np.ndarray:
    if model_name in {"lgbm_regressor", "xgb_regressor", "extratrees_regressor"}:
        model = pipe.named_steps["model"]
        preprocess = pipe.named_steps["preprocess"]
        raw_train = model.predict_continuous(preprocess.transform(X_train))
        raw_val = model.predict_continuous(preprocess.transform(X_val))
        thresholds = optimize_thresholds(np.asarray(y_train), raw_train)
        return round_with_thresholds(raw_val, thresholds)
    return np.asarray(pipe.predict(X_val), dtype=int)


# --- Core OOF loop: refit per fold, predict validation fold only ---
def collect_oof_predictions(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    best_params: dict[str, Any] | None = None,
    cv_folds: int = CV_FOLDS,
) -> np.ndarray:
    """One honest prediction per row from stratified K-fold OOF."""
    y_arr = np.asarray(y, dtype=int)
    best_params = best_params or {}
    oof = np.zeros(len(y_arr), dtype=int)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    if model_name == "regression_vote":
        factories = _regressor_factories()
        oof_discrete: dict[str, np.ndarray] = {
            name: np.zeros(len(y_arr), dtype=int) for name in factories
        }
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_arr), start=1):
            print(f"  OOF fold {fold}/{cv_folds} (regression_vote)...")
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y_arr[train_idx]
            preprocessor = build_preprocessor_from_X(X_train)
            for name in factories:
                builder = _regressor_pipeline_builders()[name]
                pipe = builder(preprocessor)
                if best_params:
                    prefixed = {k: v for k, v in best_params.items() if k.startswith("model__")}
                    if prefixed:
                        pipe.set_params(**prefixed)
                fit_pipeline_with_optional_weights(pipe, X_train, y_train)
                model = pipe.named_steps["model"]
                preprocess = pipe.named_steps["preprocess"]
                raw_val = model.predict_continuous(preprocess.transform(X_val))
                raw_train = model.predict_continuous(preprocess.transform(X_train))
                thresholds = optimize_thresholds(y_train, raw_train)
                oof_discrete[name][val_idx] = round_with_thresholds(raw_val, thresholds)
        return _mode_vote_rows(np.vstack([oof_discrete[n] for n in factories]))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_arr), start=1):
        print(f"  OOF fold {fold}/{cv_folds}...")
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        pipe = _fit_fold_pipeline(model_name, X_train, y_train, best_params)
        oof[val_idx] = _predict_fold(pipe, model_name, X_train, y_train, X_val)
    return oof


def _save_permutation_holdout(
    model_name: str,
    config: dict[str, Any],
    *,
    figures_dir: Path,
) -> pd.DataFrame:
    """Feature importance on honest 80/20 holdout (same pipeline config as final)."""
    data = load_train_val_split(apply_selection=config["feature_selection"])
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    if config["merged_classes_23"]:
        y_train = merge_classes_23(y_train)
        y_test = merge_classes_23(y_test)

    pipe = _fit_fold_pipeline(model_name, X_train, y_train, config["best_params"])
    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=qwk_scorer,
        n_jobs=SKLEARN_N_JOBS,
    )
    feature_names = list(X_test.columns)
    if len(feature_names) != len(perm.importances_mean):
        feature_names = [f"feature_{i}" for i in range(len(perm.importances_mean))]

    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    perm_png = figures_dir / "permutation_importance_oof.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    top = perm_df.head(15)
    sns.barplot(data=top, x="importance_mean", y="feature", ax=ax)
    ax.set_title(f"Permutation importance (QWK, holdout 80/20) — {model_name}")
    fig.tight_layout()
    fig.savefig(perm_png, dpi=150)
    plt.close(fig)

    perm_csv = RESULTS_DIR / "permutation_importance_oof.csv"
    perm_df.to_csv(perm_csv, index=False)
    print(f"Saved: {perm_png}")
    print(f"Saved: {perm_csv}")
    return perm_df


def _build_limitations_summary(per_class: pd.DataFrame) -> dict[str, Any]:
    """Highlight minority-class OOF weakness for reports (see docs/limitaciones_modelo.md)."""
    if per_class.empty:
        return {"note": "No per-class OOF metrics available."}

    worst = per_class.loc[per_class["recall"].idxmin()]
    minority = per_class.loc[per_class["support"].idxmin()]
    return {
        "minority_class": int(minority["class"]),
        "minority_support_oof": int(minority["support"]),
        "minority_recall_oof": float(minority["recall"]),
        "minority_f1_oof": float(minority["f1"]),
        "lowest_recall_class": int(worst["class"]),
        "lowest_recall_oof": float(worst["recall"]),
        "per_class_oof": per_class.to_dict(orient="records"),
        "note": (
            "Severe class (sii=3) is under-represented in train (~1.2%); "
            "low recall OOF is expected. See docs/limitaciones_modelo.md."
        ),
    }


def run_oof_report() -> dict[str, Any]:
    configure_plotting()
    warn_if_python_joblib_incompatible()
    ensure_experiment_dirs()

    # OOF predictions → metrics/figures → JSON summary (incl. class-3 limitations).
    config = _load_report_config()
    model_name = config["model"]
    X, y = _prepare_xy(config)

    print(f"OOF report for: {model_name}")
    print(f"Samples: {len(y)}, folds: {CV_FOLDS}")
    print("Estimated runtime: ~5-15 min (one tuned model, 10-fold CV).")
    t0 = time.perf_counter()

    y_pred = collect_oof_predictions(
        model_name,
        X,
        y,
        best_params=config["best_params"],
    )
    y_true = np.asarray(y, dtype=int)
    metrics = classification_metrics_extended(y_true, y_pred, include_report=True)
    per_class = metrics_per_class(y_true, y_pred)

    figures_dir = FIGURES_DIR
    cm_path = figures_dir / "confusion_matrix_oof.png"
    per_class_path = RESULTS_DIR / "per_class_metrics_oof.csv"

    save_confusion_matrix(
        y_true,
        y_pred,
        title=f"Confusion matrix (OOF {CV_FOLDS}-fold) — {model_name}",
        output_path=cm_path,
    )
    save_per_class_metrics(per_class, per_class_path)
    print(f"Saved: {cm_path}")
    print(f"Saved: {per_class_path}")

    print("Permutation importance (holdout 80/20, same pipeline config)...")
    perm_df = _save_permutation_holdout(model_name, config, figures_dir=figures_dir)

    limitations = _build_limitations_summary(per_class)
    summary = {
        "model": model_name,
        "cv_folds": CV_FOLDS,
        "merged_classes_23": config["merged_classes_23"],
        "feature_selection": config["feature_selection"],
        "oof_qwk": metrics["qwk"],
        "oof_f1_macro": metrics["f1_macro"],
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "limitations": limitations,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    summary_path = RESULTS_DIR / "oof_report_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOOF QWK: {metrics['qwk']:.4f}")
    print(f"OOF F1 macro: {metrics['f1_macro']:.4f}")
    print(f"OOF balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Saved: {summary_path}")
    print(f"Done in {summary['elapsed_seconds']}s")

    return {
        "metrics": metrics,
        "per_class": per_class,
        "permutation_importance": perm_df,
        "summary": summary,
        "y_pred": y_pred,
    }


def main() -> None:
    run_oof_report()


if __name__ == "__main__":
    main()
