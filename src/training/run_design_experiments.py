"""Design experiment: 4-class target vs merged classes 2+3."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, ensure_experiment_dirs
from src.data.dataset import load_train_val_split, merge_classes_23
from src.evaluation.evaluate import evaluate_pipeline
from src.training.tune import _build_pipeline, select_best_model_from_cv


def run_design_experiments() -> pd.DataFrame:
    ensure_experiment_dirs()
    data = load_train_val_split()
    model_name = select_best_model_from_cv()

    params_path = RESULTS_DIR / "tuning_best_params.json"
    best_params = {}
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            best_params = json.load(f).get("best_params", {})

    rows: list[dict] = []

    # Compare 4-class sii vs merging moderate+severe (2+3) on the same tuned pipeline.
    for variant, y_train, y_test in [
        ("four_classes", data["y_train"], data["y_test"]),
        (
            "merge_classes_23",
            merge_classes_23(data["y_train"]),
            merge_classes_23(data["y_test"]),
        ),
    ]:
        preprocessor = data["preprocessor"]
        pipe = _build_pipeline(model_name, preprocessor)
        if best_params:
            pipe.set_params(**best_params)
        pipe.fit(data["X_train"], y_train)
        metrics = evaluate_pipeline(pipe, data["X_test"], y_test)
        rows.append({"variant": variant, "model": model_name, **metrics})
        print(f"{variant}: QWK={metrics['qwk']:.4f}, F1={metrics['f1_macro']:.4f}")

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "design_merge_classes_23.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


def main() -> None:
    run_design_experiments()


if __name__ == "__main__":
    main()
