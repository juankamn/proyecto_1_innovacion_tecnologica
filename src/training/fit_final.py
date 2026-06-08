"""Production fit: best CV model + tuned params → final_model.joblib (used by predict)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    CHECKPOINTS_DIR,
    FINAL_FIT_ON_FULL_TRAIN,
    RESULTS_DIR,
    ensure_experiment_dirs,
)
from src.data.dataset import (
    load_modeling_frame,
    load_train_val_split,
    merge_classes_23,
    split_features_target,
)
from src.data.feature_selection import load_top_feature_names
from src.models.ensemble_meta import extract_threshold_meta
from src.models.ensemble_regression import fit_pipeline_with_optional_weights
from src.models.registry import ENSEMBLE_REGRESSION_MODELS
from src.training.tune import _build_pipeline, select_best_model_from_cv


def _use_merged_classes() -> bool:
    """Read design experiment CSV; use 3-class target only if merge beat 4-class QWK."""
    design_path = RESULTS_DIR / "design_merge_classes_23.csv"
    if not design_path.exists():
        return False
    df = pd.read_csv(design_path)
    four = df[df["variant"] == "four_classes"]
    merged = df[df["variant"] == "merge_classes_23"]
    if four.empty or merged.empty:
        return False
    return float(merged.iloc[0]["qwk"]) > float(four.iloc[0]["qwk"])


def run_fit_final() -> Path:
    ensure_experiment_dirs()

    # Resolve model name, class merge flag, and hyperparameters from prior CLI steps.
    model_name = select_best_model_from_cv()
    use_merged = _use_merged_classes()

    params_path = RESULTS_DIR / "tuning_best_params.json"
    best_params: dict = {}
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            best_params = json.load(f).get("best_params", {})

    use_feature_selection = (
        model_name in ENSEMBLE_REGRESSION_MODELS and load_top_feature_names() is not None
    )

    # Full train (default) vs holdout eval split when FINAL_FIT_ON_FULL_TRAIN=False.
    if FINAL_FIT_ON_FULL_TRAIN:
        df_model = load_modeling_frame(apply_selection=use_feature_selection)
        X, y = split_features_target(df_model)
        if use_merged:
            y = merge_classes_23(y)
        from src.data.dataset import build_preprocessor_from_X

        preprocessor = build_preprocessor_from_X(X)
        X_train, y_train = X, y
        X_eval, y_eval = None, None
    else:
        data = load_train_val_split(apply_selection=use_feature_selection)
        X_train = data["X_train"]
        y_train = data["y_train"]
        if use_merged:
            y_train = merge_classes_23(y_train)
        preprocessor = data["preprocessor"]
        X_eval, y_eval = data["X_test"], data["y_test"]
        if use_merged and y_eval is not None:
            y_eval = merge_classes_23(y_eval)

    # Build pipeline, apply tuned params, fit with optional sample weights for regressors.
    pipe = _build_pipeline(model_name, preprocessor)
    if best_params:
        pipe.set_params(**best_params)
    if model_name in ENSEMBLE_REGRESSION_MODELS and model_name != "regression_vote":
        fit_pipeline_with_optional_weights(pipe, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)

    # Persist sklearn pipeline + metadata (thresholds, flags) for predict/report-oof.
    meta = {
        "model": model_name,
        "best_params": best_params,
        "merged_classes_23": use_merged,
        "fit_on_full_train": FINAL_FIT_ON_FULL_TRAIN,
        "feature_selection": use_feature_selection,
        **extract_threshold_meta(pipe, model_name),
    }
    out_path = CHECKPOINTS_DIR / "final_model.joblib"
    joblib.dump({"pipeline": pipe, "meta": meta}, out_path)

    meta_path = RESULTS_DIR / "final_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Final model: {model_name} (merged_23={use_merged})")
    print(f"Saved: {out_path}")

    if X_eval is not None and y_eval is not None:
        from src.evaluation.evaluate import evaluate_pipeline

        metrics = evaluate_pipeline(pipe, X_eval, y_eval)
        print(f"Holdout QWK: {metrics['qwk']:.4f}")

    return out_path


def main() -> None:
    run_fit_final()


if __name__ == "__main__":
    main()
