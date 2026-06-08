"""
Model catalog for the `train` command (10 classifiers) and level labels for charts.

Ordinal regressors are added separately in train_ensemble.py.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models.baseline import get_baseline_models
from src.models.boosting import get_lgbm_pipeline, get_xgb_pipeline
from src.models.classical import (
    get_knn_pipeline,
    get_naive_bayes_pipeline,
    get_ridge_ordinal_pipeline,
    get_svc_pipeline,
)
def get_all_model_pipelines(
    preprocessor,
    *,
    include_naive_bayes: bool = True,
) -> dict[str, Pipeline]:
    # Baselines + classical + boosting; each value is preprocess → classifier.
    pipelines = get_baseline_models(preprocessor)
    pipelines.update(
        {
            "knn": get_knn_pipeline(preprocessor),
            "ridge_ordinal": get_ridge_ordinal_pipeline(preprocessor),
            "svm": get_svc_pipeline(preprocessor),
            "lightgbm": get_lgbm_pipeline(preprocessor),
            "xgboost": get_xgb_pipeline(preprocessor),
        }
    )
    if include_naive_bayes:
        pipelines["naive_bayes"] = get_naive_bayes_pipeline(preprocessor)
    return pipelines


MODEL_LEVELS: dict[str, str] = {
    "dummy": "reference",
    "logistic_regression": "basic",
    "knn": "basic",
    "ridge_ordinal": "basic",
    "decision_tree": "basic",
    "naive_bayes": "basic",
    "random_forest": "intermediate",
    "svm": "intermediate",
    "lightgbm": "advanced",
    "xgboost": "advanced",
    "lgbm_regressor": "advanced",
    "xgb_regressor": "advanced",
    "extratrees_regressor": "advanced",
    "regression_vote": "advanced",
}

ENSEMBLE_REGRESSION_MODELS = frozenset(
    {"lgbm_regressor", "xgb_regressor", "extratrees_regressor", "regression_vote"}
)
