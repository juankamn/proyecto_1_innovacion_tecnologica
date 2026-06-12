"""
CLI entry point for the final delivery workflow.

  python -m src.main check
  python -m src.main train
  python -m src.main train-ensemble
  python -m src.main tune
  python -m src.main design
  python -m src.main fit-final
  python -m src.main report-oof   # honest OOF figures for the notebook
  python -m src.main predict      # submission.csv for Kaggle (after fit-final)

Optional (only if FINAL_FIT_ON_FULL_TRAIN=False in config):
  python -m src.main evaluate
"""
from __future__ import annotations

import argparse
import sys

from src.config import EXPECTED_MODELING_SHAPE, TRAIN_CSV_PATH
from src.data.load import load_train
from src.data.preprocess import clean_for_modeling


# --- CLI command handlers (thin wrappers around src.training / src.evaluation) ---
def cmd_check() -> None:
    print(f"Train file: {TRAIN_CSV_PATH}")
    df = load_train()
    print(f"Raw shape: {df.shape}")
    _, df_model = clean_for_modeling(df)
    print(f"Modeling-ready shape: {df_model.shape}")
    if df_model.shape != EXPECTED_MODELING_SHAPE:
        print(f"WARNING: expected shape {EXPECTED_MODELING_SHAPE}")


def cmd_train() -> None:
    from src.training.train import run_train

    run_train()


def cmd_train_ensemble() -> None:
    from src.training.train_ensemble import run_train_ensemble

    run_train_ensemble()


def cmd_tune() -> None:
    from src.training.tune import run_tune

    run_tune()


def cmd_design() -> None:
    from src.training.run_design_experiments import run_design_experiments

    run_design_experiments()


def cmd_fit_final() -> None:
    from src.training.fit_final import run_fit_final

    run_fit_final()


def cmd_evaluate() -> None:
    from src.evaluation.evaluate import print_evaluation_summary, run_full_evaluation

    result = run_full_evaluation()
    print_evaluation_summary(result)


def cmd_report_oof() -> None:
    from src.evaluation.oof_report import run_oof_report

    run_oof_report()


def cmd_predict() -> None:
    from src.evaluation.predict import run_predict

    run_predict()


def main(argv: list[str] | None = None) -> None:
    # Default to `check` when no subcommand is passed.
    parser = argparse.ArgumentParser(description="Proyecto innovacion tecnologica — ML pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Sanity check data load and cleaning")
    sub.add_parser("train", help="Train and compare all models")
    sub.add_parser("train-ensemble",help="Train ordinal regression ensemble models (append to CSVs)")
    sub.add_parser("tune", help="Tune best CV model with RandomizedSearchCV")
    sub.add_parser("design", help="Run design experiment (merge classes 2+3)")
    sub.add_parser("fit-final", help="Fit final model on full train data")
    sub.add_parser(
        "evaluate",
        help="Optional holdout figures (only if fit-final did NOT use full train)",
    )
    sub.add_parser(
        "report-oof",
        help="Honest OOF confusion matrix and metrics for final pipeline (notebook)",
    )
    sub.add_parser(
        "predict",
        help="Write submission.csv from final_model.joblib (Kaggle test set)",
    )

    args = parser.parse_args(argv)
    commands = {
        "check": cmd_check,
        "train": cmd_train,
        "train-ensemble": cmd_train_ensemble,
        "tune": cmd_tune,
        "design": cmd_design,
        "fit-final": cmd_fit_final,
        "evaluate": cmd_evaluate,
        "report-oof": cmd_report_oof,
        "predict": cmd_predict,
    }

    if args.command is None:
        cmd_check()
        return

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler()


if __name__ == "__main__":
    main()
