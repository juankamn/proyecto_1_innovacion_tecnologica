"""
Holdout/CV metrics, charts, and optional holdout error analysis.

Note: when fit-final uses all labeled train, use report-oof for honest metrics (not evaluate).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    FINAL_FIT_ON_FULL_TRAIN,
    RANDOM_STATE,
    RESULTS_DIR,
    SKLEARN_N_JOBS,
    configure_plotting,
    ensure_experiment_dirs,
)

from src.evaluation.metrics import (
    classification_metrics_extended,
    metrics_per_class,
    qwk_scorer,
)
from src.models.registry import MODEL_LEVELS


def annotate_holdout_with_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["level"] = out["model"].map(MODEL_LEVELS).fillna("other")
    return out


# --- Single-model metrics on a fixed holdout split ---
def evaluate_pipeline(pipe: Pipeline, X, y) -> dict[str, Any]:
    y_pred = pipe.predict(X)
    return classification_metrics_extended(np.asarray(y), np.asarray(y_pred))


def cross_validate_pipeline(
    pipe: Pipeline,
    X,
    y,
    *,
    cv_folds: int = CV_FOLDS,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={
            "qwk": qwk_scorer,
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
        },
        n_jobs=SKLEARN_N_JOBS,
        error_score="raise",
    )
    return {
        "qwk_mean": float(scores["test_qwk"].mean()),
        "qwk_std": float(scores["test_qwk"].std()),
        "f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "f1_macro_std": float(scores["test_f1_macro"].std()),
        "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
    }


# --- Figure/CSV writers shared with oof_report ---
def save_confusion_matrix(
    y_true,
    y_pred,
    *,
    title: str,
    output_path: Path,
) -> None:
    ensure_experiment_dirs()
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_class_metrics(df: pd.DataFrame, output_path: Path) -> None:
    ensure_experiment_dirs()
    df.to_csv(output_path, index=False)


def save_model_comparison_chart(
    holdout_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "qwk",
    title: str | None = None,
) -> None:
    ensure_experiment_dirs()
    plot_df = holdout_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))
    sns.barplot(data=plot_df, x=metric, y="model", hue="level", ax=ax, dodge=False)
    ax.set_title(title or f"Model comparison ({metric} holdout)")
    ax.set_xlabel(metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_all_models_holdout_chart(
    holdout_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Bar chart from merged holdout CSV (baseline + ensemble models)."""
    configure_plotting()
    ensure_experiment_dirs()
    holdout_path = holdout_path or (RESULTS_DIR / "all_models_holdout.csv")
    if not holdout_path.exists():
        raise FileNotFoundError(f"Missing {holdout_path}. Run train and train-ensemble first.")
    holdout_df = annotate_holdout_with_levels(pd.read_csv(holdout_path))
    out = output_path or (FIGURES_DIR / "model_comparison_qwk_all.png")
    save_model_comparison_chart(
        holdout_df,
        out,
        metric="qwk",
        title="Model comparison — all models (QWK holdout 80/20)",
    )
    return out


def load_cv_metrics_for_model(model_name: str) -> dict[str, float | str] | None:
    """Reference QWK from cross-validation (honest comparison metric)."""
    path = RESULTS_DIR / "all_models_cv.csv"
    if not path.exists():
        return None
    cv_df = pd.read_csv(path)
    rows = cv_df[cv_df["model"] == model_name]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "model": model_name,
        "qwk_mean": float(row["qwk_mean"]),
        "qwk_std": float(row.get("qwk_std", 0.0)),
    }


def model_was_fit_on_full_train(loaded: Any) -> bool:
    if isinstance(loaded, dict) and "meta" in loaded:
        return bool(loaded["meta"].get("fit_on_full_train", False))
    return FINAL_FIT_ON_FULL_TRAIN


# --- Holdout diagnostics: confusion matrix, per-class table, permutation importance ---
def run_error_analysis(
    model_path: Path,
    X_test,
    y_test,
    *,
    figures_dir: Path | None = None,
    model_name: str = "final",
    report_metrics: bool = True,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    figures_dir = figures_dir or FIGURES_DIR
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        pipe = loaded["pipeline"]
    else:
        pipe = loaded

    if not report_metrics:
        return {
            "metrics": None,
            "per_class": None,
            "permutation_importance": None,
            "metrics_skipped": True,
        }

    y_pred = pipe.predict(X_test)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    metrics = classification_metrics_extended(y_true, y_pred, include_report=True)
    per_class = metrics_per_class(y_true, y_pred)

    save_confusion_matrix(
        y_true,
        y_pred,
        title=f"Confusion matrix — {model_name}",
        output_path=figures_dir / f"confusion_matrix_{model_name}.png",
    )
    save_per_class_metrics(
        per_class,
        figures_dir.parent / f"per_class_metrics_{model_name}.csv",
    )

    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=qwk_scorer,
        n_jobs=SKLEARN_N_JOBS,
    )
    if hasattr(X_test, "columns"):
        raw_names = list(X_test.columns)
    else:
        raw_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    feature_names = _align_feature_names(raw_names, len(perm.importances_mean))
    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    perm_path = figures_dir / f"permutation_importance_{model_name}.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    top = perm_df.head(15)
    sns.barplot(data=top, x="importance_mean", y="feature", ax=ax)
    ax.set_title(f"Permutation importance (QWK) — {model_name}")
    fig.tight_layout()
    fig.savefig(perm_path, dpi=150)
    plt.close(fig)

    perm_csv_path = figures_dir.parent / f"permutation_importance_{model_name}.csv"
    perm_df.to_csv(perm_csv_path, index=False)

    cm_path = figures_dir / f"confusion_matrix_{model_name}.png"
    print(f"Saved: {cm_path}")
    print(f"Saved: {perm_path}")
    print(f"Saved: {perm_csv_path}")

    return {
        "metrics": metrics,
        "per_class": per_class,
        "permutation_importance": perm_df,
    }


def _align_feature_names(names: list[str], n_importances: int) -> list[str]:
    if len(names) == n_importances:
        return names
    if len(names) > n_importances:
        return names[:n_importances]
    return names + [f"feature_{i}" for i in range(len(names), n_importances)]


def _feature_names_after_preprocess(pipe: Pipeline, X_sample) -> list[str]:
    preprocess = pipe.named_steps.get("preprocess") or pipe.named_steps.get("prep")
    if preprocess is None:
        return list(X_sample.columns)
    try:
        names = preprocess.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        transformed = preprocess.transform(X_sample.iloc[:1])
        return [f"feature_{i}" for i in range(transformed.shape[1])]


def run_full_evaluation() -> dict[str, Any]:
    """CLI evaluate: skipped automatically if model was fit on full train."""
    from src.config import CHECKPOINTS_DIR
    from src.data.dataset import load_train_val_split, merge_classes_23

    data = load_train_val_split()
    model_path = CHECKPOINTS_DIR / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Run fit-final first. Missing {model_path}")

    loaded = joblib.load(model_path)
    meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}
    model_name = str(meta.get("model", ""))

    if model_was_fit_on_full_train(loaded):
        return {
            "metrics": None,
            "per_class": None,
            "permutation_importance": None,
            "metrics_skipped": True,
            "skip_reason": "fit_on_full_train",
            "cv_reference": load_cv_metrics_for_model(model_name) if model_name else None,
            "meta": meta,
        }

    y_test = data["y_test"]
    if meta.get("merged_classes_23"):
        y_test = merge_classes_23(y_test)

    use_feature_selection = bool(meta.get("feature_selection", False))
    if use_feature_selection:
        data = load_train_val_split(apply_selection=True)
        y_test = data["y_test"]
        if meta.get("merged_classes_23"):
            y_test = merge_classes_23(y_test)

    return run_error_analysis(
        model_path,
        data["X_test"],
        y_test,
        model_name="final",
        report_metrics=True,
    )


def print_evaluation_summary(result: dict[str, Any]) -> None:
    if result.get("metrics_skipped"):
        print("Holdout metrics skipped: final model was trained on all labeled train.csv.")
        print("Report QWK from experiments/results/all_models_cv.csv (cross-validation).")
        cv_ref = result.get("cv_reference")
        if cv_ref:
            print(
                f"CV reference - {cv_ref['model']}: "
                f"QWK={cv_ref['qwk_mean']:.4f} +/- {cv_ref['qwk_std']:.4f}"
            )
        else:
            print("Run train and train-ensemble to populate all_models_cv.csv.")
        return

    metrics = result["metrics"]
    if metrics is None:
        return
    print(f"QWK: {metrics['qwk']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")


def main() -> None:
    result = run_full_evaluation()
    print_evaluation_summary(result)


if __name__ == "__main__":
    main()
