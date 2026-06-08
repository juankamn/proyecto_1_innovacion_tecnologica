"""
Shared Jupyter bootstrap. Run from any notebook under notebooks/:

    %run init_notebook.py

Then call one of:
    df = load_raw_train()
    df, df_for_modeling = load_modeling_data()
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    FIGURES_DIR,
    ID_COLUMNS,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    configure_plotting,
)
from src.data.load import load_train
from src.data.preprocess import build_preprocessor, clean_for_modeling, impute_for_eda
from src.data.variable_families import VARIABLE_CATEGORIES, pciat_columns

configure_plotting()
qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")

print(f"Project root: {PROJECT_ROOT}")
print(f"Train data:   {TRAIN_CSV_PATH}")


def load_raw_train():
    """Raw train.csv (3960 x 82)."""
    df = load_train()
    print(f"Dimensiones del dataset: {df.shape}")
    print(df.head())
    return df


def load_modeling_data():
    """Raw df + df_for_modeling after section 10 cleaning (2736 x 46, sin imputar)."""
    df = load_train()
    print(f"Dimensiones del dataset: {df.shape}")
    _, df_for_modeling = clean_for_modeling(df)
    print(f"df_for_modeling (sin imputar): {df_for_modeling.shape}")
    return df, df_for_modeling
