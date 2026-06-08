"""Reference and tree/linear baselines (dummy, logistic, DT, RF)."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def get_baseline_models(preprocessor) -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    DummyClassifier(strategy="stratified", random_state=42),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
