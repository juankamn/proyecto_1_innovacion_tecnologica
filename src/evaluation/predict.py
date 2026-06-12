"""Generate Kaggle submission from the fitted final model."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import CHECKPOINTS_DIR, RESULTS_DIR, TARGET_COLUMN, ensure_experiment_dirs
from src.data.dataset import prepare_test_for_inference


def run_predict(output_path: Path | None = None) -> Path:
    ensure_experiment_dirs()

    # Load production checkpoint; align test features to train schema from meta flags.
    model_path = CHECKPOINTS_DIR / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run: python -m src.main fit-final"
        )

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    meta = bundle.get("meta", {})
    apply_selection = bool(meta.get("feature_selection", False))
    merged_classes = bool(meta.get("merged_classes_23", False))

    ids, X = prepare_test_for_inference(apply_selection=apply_selection)
    preds = np.asarray(pipe.predict(X), dtype=int)
    max_class = 2 if merged_classes else 3  # clip to valid class range
    preds = np.clip(preds, 0, max_class)

    out_path = output_path or (RESULTS_DIR / "submission.csv")
    submission = pd.DataFrame({TARGET_COLUMN: preds}, index=ids)
    submission.index.name = "id"
    submission.reset_index().to_csv(out_path, index=False)

    print(f"Model: {meta.get('model', 'unknown')}")
    print(f"Rows: {len(submission)}")
    print(f"Class distribution: {pd.Series(preds).value_counts().sort_index().to_dict()}")
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    run_predict()


if __name__ == "__main__":
    main()
