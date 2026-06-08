"""KNN, SVM, Naive Bayes, and Ridge-with-rounding ordinal baseline."""
from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import RANDOM_STATE
from src.models.ordinal import RegressorToOrdinalClassifier


def get_knn_pipeline(preprocessor, n_neighbors: int = 15) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                KNeighborsClassifier(n_neighbors=n_neighbors),
            ),
        ]
    )


def get_ridge_ordinal_pipeline(preprocessor, alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RegressorToOrdinalClassifier(Ridge(alpha=alpha, random_state=RANDOM_STATE)),
            ),
        ]
    )


def get_svc_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_naive_bayes_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", GaussianNB()),
        ]
    )
