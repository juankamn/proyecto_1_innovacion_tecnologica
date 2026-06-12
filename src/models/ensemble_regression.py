"""
Winning approach: regress on sii, discretize with QWK-optimized thresholds.

Includes single regressors (LGBM/XGB/ExtraTrees) and regression_vote (mode over three).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.config import RANDOM_STATE, USE_SAMPLE_WEIGHTS
from src.data.sample_weights import calculate_sii_bin_weights
from src.evaluation.thresholds import ThresholdOrdinalClassifier


def _lgbm_regressor(**kwargs) -> Any:
    from lightgbm import LGBMRegressor

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return LGBMRegressor(**params)


def _xgb_regressor(**kwargs) -> Any:
    from xgboost import XGBRegressor

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    params.update(kwargs)
    return XGBRegressor(**params)


def _extratrees_regressor(**kwargs) -> ExtraTreesRegressor:
    params = {
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return ExtraTreesRegressor(**params)


# --- Standard ordinal regressor pipeline: preprocess → ThresholdOrdinalClassifier ---
def get_regressor_pipeline(
    preprocessor,
    regressor_factory,
    *,
    optimize_thresholds: bool = True,
) -> Pipeline:
    model = ThresholdOrdinalClassifier(
        regressor_factory(),
        optimize_on_fit=optimize_thresholds,
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def get_lgbm_regressor_pipeline(preprocessor, **kwargs) -> Pipeline:
    return get_regressor_pipeline(preprocessor, lambda: _lgbm_regressor(**kwargs))


def get_xgb_regressor_pipeline(preprocessor, **kwargs) -> Pipeline:
    return get_regressor_pipeline(preprocessor, lambda: _xgb_regressor(**kwargs))


def get_extratrees_regressor_pipeline(preprocessor, **kwargs) -> Pipeline:
    return get_regressor_pipeline(preprocessor, lambda: _extratrees_regressor(**kwargs))


def fit_pipeline_with_optional_weights(
    pipe: Pipeline,
    X,
    y,
    *,
    use_sample_weights: bool = USE_SAMPLE_WEIGHTS,
) -> Pipeline:
    fit_params: dict[str, Any] = {}
    if use_sample_weights:
        fit_params["model__sample_weight"] = calculate_sii_bin_weights(y)
    pipe.fit(X, y, **fit_params)
    return pipe


# --- Ensemble: three independent ordinal regressors, majority vote on discrete class ---
class RegressionVoteClassifier(BaseEstimator, ClassifierMixin):
    """Mode vote over LGBM, XGBoost and ExtraTrees regressors with ordinal thresholds."""

    def __init__(
        self,
        preprocessor,
        *,
        use_sample_weights: bool = USE_SAMPLE_WEIGHTS,
    ):
        self.preprocessor = preprocessor
        self.use_sample_weights = use_sample_weights

    def fit(self, X, y, sample_weight=None):
        factories = [
            _lgbm_regressor,
            _xgb_regressor,
            _extratrees_regressor,
        ]
        self.pipelines_: list[Pipeline] = []
        sw = sample_weight
        if sw is None and self.use_sample_weights:
            sw = calculate_sii_bin_weights(y)
        for factory in factories:
            pipe = Pipeline(
                [
                    ("preprocess", clone(self.preprocessor)),
                    (
                        "model",
                        ThresholdOrdinalClassifier(factory(), optimize_on_fit=True),
                    ),
                ]
            )
            if sw is not None:
                pipe.fit(X, y, model__sample_weight=sw)
            else:
                pipe.fit(X, y)
            self.pipelines_.append(pipe)
        self.classes_ = np.array([0, 1, 2, 3])
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "pipelines_")
        preds = np.vstack([pipe.predict(X) for pipe in self.pipelines_])
        result = stats.mode(preds, axis=0, keepdims=False)
        return np.asarray(result.mode, dtype=int).reshape(-1)

    def get_thresholds(self) -> list[np.ndarray]:
        check_is_fitted(self, "pipelines_")
        thresholds: list[np.ndarray] = []
        for pipe in self.pipelines_:
            model = pipe.named_steps["model"]
            thresholds.append(np.asarray(model.thresholds_, dtype=float))
        return thresholds


def get_regression_vote_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", "passthrough"),
            ("model", RegressionVoteClassifier(preprocessor)),
        ]
    )


def get_ensemble_regression_pipelines(preprocessor) -> dict[str, Pipeline]:
    return {
        "lgbm_regressor": get_lgbm_regressor_pipeline(preprocessor),
        "xgb_regressor": get_xgb_regressor_pipeline(preprocessor),
        "extratrees_regressor": get_extratrees_regressor_pipeline(preprocessor),
        "regression_vote": get_regression_vote_pipeline(preprocessor),
    }
