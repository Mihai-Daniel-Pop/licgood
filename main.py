"""CLI entry point: train models on one ticker, optionally walk-forward.
The GUI lives in gui_qt.py."""

import argparse

import numpy as np

from src.data_loader import DataLoader
from src.models import CLASSIFIERS, REGRESSORS, SkModel
from src.preprocessor import DataPreprocessor


def main():
    p = argparse.ArgumentParser(description="Train stock prediction models.")
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--mode", default="Classification",
                   choices=["Classification", "Regression"])
    p.add_argument("--models", nargs="+", default=list(CLASSIFIERS),
                   choices=list(CLASSIFIERS))
    p.add_argument("--tune", action="store_true",
                   help="Optuna hyperparameter tuning (RF and XGBoost).")
    p.add_argument("--walk-forward", action="store_true")
    args = p.parse_args()

    print(f"\n=== {args.ticker} ({args.start} -> {args.end}) ===")
    raw = DataLoader(args.ticker, args.start, args.end).get_data()
    if raw is None or raw.empty:
        print("No data downloaded.")
        return

    processor = DataPreprocessor(raw)
    processor.add_technical_indicators()
    registry = REGRESSORS if args.mode == "Regression" else CLASSIFIERS
    if args.mode == "Regression":
        X, y = processor.prepare_data_for_regression()
    else:
        X, y = processor.prepare_data_for_training()
    print(f"Feature matrix: {X.shape}")

    for name in args.models:
        print(f"\n=== {name} ===")
        model = registry[name]()
        if isinstance(model, SkModel):
            model.train(X, y, tune=args.tune)
        else:
            model.train(X, y)
        if args.walk_forward:
            scores = model.walk_forward_score(X, y, n_splits=5, verbose=True)
            print(f"{name} walk-forward: mean {np.mean(scores):.4f}  "
                  f"std {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
