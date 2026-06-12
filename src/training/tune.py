"""RandomizedSearchCV on the top CV model; saves tuning_best_params.json and tuned_model.joblib."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    CHECKPOINTS_DIR,
    CV_FOLDS,
    N_TUNE_ITER,
    RANDOM_STATE,
    RESULTS_DIR,
    SKLEARN_N_JOBS,
    ensure_experiment_dirs,
)
from src.data.dataset import load_train_val_split
from src.evaluation.metrics import qwk_scorer
from src.models.boosting import get_lgbm_pipeline, get_xgb_pipeline
from src.models.baseline import get_baseline_models
from src.models.classical import get_knn_pipeline, get_svc_pipeline
from src.models.ensemble_regression import (
    fit_pipeline_with_optional_weights,
    get_extratrees_regressor_pipeline,
    get_lgbm_regressor_pipeline,
    get_regression_vote_pipeline,
    get_xgb_regressor_pipeline,
)
from src.models.registry import ENSEMBLE_REGRESSION_MODELS


# --- Per-model search spaces (empty dict = skip tuning, save defaults only) ---
def _get_param_distributions(model_name: str) -> dict[str, Any]:
    if model_name == "lightgbm":
        return {
            "model__n_estimators": randint(100, 400),
            "model__learning_rate": uniform(0.01, 0.15),
            "model__num_leaves": randint(16, 64),
            "model__min_child_samples": randint(5, 50),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.0, 1.0),
        }
    if model_name == "xgboost":
        return {
            "model__n_estimators": randint(100, 400),
            "model__learning_rate": uniform(0.01, 0.15),
            "model__max_depth": randint(3, 10),
            "model__min_child_weight": randint(1, 10),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.0, 1.0),
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", None],
        }
    if model_name == "svm":
        return {
            "model__C": uniform(0.1, 10),
            "model__gamma": ["scale", "auto"],
        }
    if model_name == "knn":
        return {
            "model__n_neighbors": randint(3, 31),
            "model__weights": ["uniform", "distance"],
        }
    if model_name == "lgbm_regressor":
        return {
            "model__regressor__n_estimators": randint(100, 400),
            "model__regressor__learning_rate": uniform(0.01, 0.15),
            "model__regressor__num_leaves": randint(16, 64),
            "model__regressor__min_child_samples": randint(5, 50),
        }
    if model_name == "xgb_regressor":
        return {
            "model__regressor__n_estimators": randint(100, 400),
            "model__regressor__learning_rate": uniform(0.01, 0.15),
            "model__regressor__max_depth": randint(3, 10),
            "model__regressor__min_child_weight": randint(1, 10),
        }
    if model_name == "extratrees_regressor":
        return {
            "model__regressor__n_estimators": randint(100, 400),
            "model__regressor__max_depth": [None, 10, 20, 30],
            "model__regressor__min_samples_leaf": randint(1, 10),
        }
    return {}


def _build_pipeline(model_name: str, preprocessor):
    builders = {
        "lightgbm": get_lgbm_pipeline,
        "xgboost": get_xgb_pipeline,
        "svm": get_svc_pipeline,
        "knn": get_knn_pipeline,
        "lgbm_regressor": get_lgbm_regressor_pipeline,
        "xgb_regressor": get_xgb_regressor_pipeline,
        "extratrees_regressor": get_extratrees_regressor_pipeline,
        "regression_vote": get_regression_vote_pipeline,
    }
    if model_name in builders:
        return builders[model_name](preprocessor)
    baselines = get_baseline_models(preprocessor)
    if model_name in baselines:
        return baselines[model_name]
    raise ValueError(f"No tuning pipeline for model: {model_name}")


def select_best_model_from_cv(cv_path: Path | None = None) -> str:
    path = cv_path or (RESULTS_DIR / "all_models_cv.csv")
    if not path.exists():
        raise FileNotFoundError(f"Run training first. Missing {path}")
    cv_df = pd.read_csv(path)
    return str(cv_df.sort_values("qwk_mean", ascending=False).iloc[0]["model"])


def run_tune(model_name: str | None = None) -> dict[str, Any]:
    ensure_experiment_dirs()
    # Tuning uses 80/20 train split only (not full train — that is fit-final).
    data = load_train_val_split()
    X_train = data["X_train"]
    y_train = data["y_train"]
    preprocessor = data["preprocessor"]

    model_name = model_name or select_best_model_from_cv()
    pipe = _build_pipeline(model_name, preprocessor)
    param_dist = _get_param_distributions(model_name)

    if not param_dist:
        # Models without a search grid still get a checkpoint for inspection.
        print(f"No param grid for {model_name}; saving default pipeline.")
        if model_name in ENSEMBLE_REGRESSION_MODELS and model_name != "regression_vote":
            fit_pipeline_with_optional_weights(pipe, X_train, y_train)
        else:
            pipe.fit(X_train, y_train)
        joblib.dump(pipe, CHECKPOINTS_DIR / "tuned_model.joblib")
        result = {"model": model_name, "best_params": {}, "best_qwk_cv": None}
        with open(RESULTS_DIR / "tuning_best_params.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # RandomizedSearchCV with QWK scorer; best estimator → tuned_model.joblib.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_TUNE_ITER,
        scoring=qwk_scorer,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=SKLEARN_N_JOBS,
        refit=True,
        verbose=1,
    )
    print(f"Tuning {model_name} ({N_TUNE_ITER} iterations, QWK scoring)...")
    fit_kwargs: dict[str, Any] = {}
    if model_name in ENSEMBLE_REGRESSION_MODELS and model_name != "regression_vote":
        from src.data.sample_weights import calculate_sii_bin_weights

        fit_kwargs["model__sample_weight"] = calculate_sii_bin_weights(y_train)
    search.fit(X_train, y_train, **fit_kwargs)

    best_pipe = search.best_estimator_
    joblib.dump(best_pipe, CHECKPOINTS_DIR / "tuned_model.joblib")

    trials = pd.DataFrame(search.cv_results_)
    trials_path = RESULTS_DIR / "tuning_search.csv"
    trials.to_csv(trials_path, index=False)

    result = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_qwk_cv": float(search.best_score_),
    }
    params_path = RESULTS_DIR / "tuning_best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Best QWK (CV): {search.best_score_:.4f}")
    print(f"Saved: {params_path}")
    print(f"Saved: {CHECKPOINTS_DIR / 'tuned_model.joblib'}")
    return result


def main() -> None:
    run_tune()


if __name__ == "__main__":
    main()
