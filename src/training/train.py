"""Train and compare all registered models; save holdout and CV metrics."""
from __future__ import annotations

import pandas as pd

from src.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    SKLEARN_N_JOBS,
    configure_plotting,
    ensure_experiment_dirs,
    warn_if_python_joblib_incompatible,
)

from src.data.dataset import load_train_val_split
from src.evaluation.evaluate import (
    annotate_holdout_with_levels,
    cross_validate_pipeline,
    evaluate_pipeline,
    save_model_comparison_chart,
)
from src.models.registry import MODEL_LEVELS, get_all_model_pipelines


def run_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Environment setup (plots, output dirs, joblib warning).
    configure_plotting()
    warn_if_python_joblib_incompatible()
    ensure_experiment_dirs()
    if SKLEARN_N_JOBS == 1:
        print("SKLEARN_N_JOBS=1 (sequential CV/tuning; set SKLEARN_N_JOBS=-1 on Python 3.14.2+)")

    # Stratified 80/20 split; preprocessor schema comes from X_train only.
    data = load_train_val_split()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    pipelines = get_all_model_pipelines(preprocessor)

    holdout_rows: list[dict] = []
    cv_rows: list[dict] = []

    # Baseline + classical + boosting classifiers (10 models); QWK on holdout and CV.
    for name, pipe in pipelines.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        holdout = evaluate_pipeline(pipe, X_test, y_test)
        holdout_rows.append(
            {
                "model": name,
                "level": MODEL_LEVELS.get(name, "other"),
                **holdout,
            }
        )

        cv = cross_validate_pipeline(pipe, X_train, y_train)
        cv_rows.append({"model": name, "level": MODEL_LEVELS.get(name, "other"), **cv})

    holdout_df = annotate_holdout_with_levels(pd.DataFrame(holdout_rows))
    cv_df = pd.DataFrame(cv_rows)

    # Write comparison tables and bar chart (baseline-only figure).
    holdout_path = RESULTS_DIR / "all_models_holdout.csv"
    cv_path = RESULTS_DIR / "all_models_cv.csv"
    holdout_df.to_csv(holdout_path, index=False)
    cv_df.to_csv(cv_path, index=False)

    save_model_comparison_chart(
        holdout_df,
        FIGURES_DIR / "model_comparison_qwk.png",
        metric="qwk",
        title="Model comparison — baseline classifiers (QWK holdout 80/20)",
    )

    best = cv_df.sort_values("qwk_mean", ascending=False).iloc[0]
    print(f"Best CV model: {best['model']} (QWK={best['qwk_mean']:.4f} ± {best['qwk_std']:.4f})")
    figure_path = FIGURES_DIR / "model_comparison_qwk.png"
    print(f"Saved: {holdout_path}")
    print(f"Saved: {cv_path}")
    print(f"Saved: {figure_path}")

    return holdout_df, cv_df


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
