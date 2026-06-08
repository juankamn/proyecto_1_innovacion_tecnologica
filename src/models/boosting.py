"""Multiclass gradient boosting classifiers (direct 4-class prediction)."""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def get_lgbm_pipeline(preprocessor, **kwargs) -> Pipeline:
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LGBMClassifier(**params)),
        ]
    )


def get_xgb_pipeline(preprocessor, **kwargs) -> Pipeline:
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )
