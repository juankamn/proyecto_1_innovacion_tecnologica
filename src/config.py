"""Project paths, modeling constants, and runtime helpers for the ML pipeline."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# --- Paths: raw Kaggle CSVs and experiment outputs ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV_PATH = DATA_RAW_DIR / "train.csv"
TEST_CSV_PATH = DATA_RAW_DIR / "test.csv"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- Target, data cleaning, and cross-validation defaults ---
TARGET_COLUMN = "sii"
ID_COLUMNS = ["id"]
NULL_COLUMN_THRESHOLD = 0.70
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10
N_TUNE_ITER = 40
FINAL_FIT_ON_FULL_TRAIN = True

# --- Feature engineering / training options (see notebooks 04–05) ---
DROP_SEASON_COLUMNS = True
FEATURE_SELECTION_TOP_N = 35
USE_ITERATIVE_IMPUTER = True
USE_SAMPLE_WEIGHTS = True

EXPECTED_MODELING_SHAPE = (2736, 46)


# --- sklearn/joblib parallelism: auto-fallback on broken Python builds ---
def _python_has_loky_resource_tracker_bug() -> bool:
    """CPython 3.13.10 / 3.14.0–3.14.1 break joblib/loky multiprocessing (fixed in 3.13.11+ and 3.14.2+)."""
    v = sys.version_info
    if v.major != 3:
        return False
    if v.minor == 13 and v.micro == 10:
        return True
    return v.minor == 14 and v.micro < 2


def resolve_sklearn_n_jobs() -> int:
    """Parallel jobs for sklearn/joblib. Override with env SKLEARN_N_JOBS."""
    env = os.environ.get("SKLEARN_N_JOBS")
    if env is not None:
        return int(env)
    if _python_has_loky_resource_tracker_bug():
        return 1
    return -1


SKLEARN_N_JOBS = resolve_sklearn_n_jobs()


def warn_if_python_joblib_incompatible() -> None:
    if not _python_has_loky_resource_tracker_bug():
        return
    warnings.warn(
        f"Python {sys.version.split()[0]} has a known joblib/loky multiprocessing bug. "
        "Using SKLEARN_N_JOBS=1 (sequential). Upgrade to Python 3.14.2+ or 3.13.11+, "
        "or set SKLEARN_N_JOBS=-1 after upgrading.",
        UserWarning,
        stacklevel=2,
    )


def ensure_experiment_dirs() -> None:
    for path in (RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_plotting() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (6, 4)
