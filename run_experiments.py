

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.models import CLASSIFIERS, REGRESSORS
from src.preprocessor import DataPreprocessor

MODEL_COLORS = {
    "Random Forest": "#1f9d57",
    "XGBoost": "#2f6fed",
    "LSTM": "#7c5cdc",
    "NEAT": "#d9820a",
    "Jordan": "#0ea5b7",
}


def plot_ticker(df_scores, ticker, mode, out_dir):
    # NB: bracket access is mandatory here — df.mode is the DataFrame method
    sub = df_scores[(df_scores["ticker"] == ticker) & (df_scores["mode"] == mode)]
    if sub.empty:
        return
    models = [m for m in MODEL_COLORS if m in sub["model"].unique()]
    n_folds = sub["fold"].max()
    x = np.arange(n_folds)
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(models):
        scores = sub[sub["model"] == name].sort_values("fold")["score"].values
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, scores, width,
               label=f"{name} ({np.mean(scores):.3f})",
               color=MODEL_COLORS[name], edgecolor="white")
    ax.axhline(0.5, color="#d83a4a", linestyle="--", linewidth=1,
               label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_ylabel("Directional accuracy" if mode == "Regression" else "Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{ticker} — walk-forward CV ({mode.lower()})")
    ax.legend(loc="lower right", ncols=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f"wf_{ticker}_{mode.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  chart -> {path}", flush=True)


def latex_tables(summary, out_path):
    """One LaTeX table per mode: rows = tickers, columns = models."""
    chunks = []
    for mode in summary["mode"].unique():
        pivot = summary[summary["mode"] == mode].pivot(
            index="ticker", columns="model", values="mean_std")
        pivot = pivot[[m for m in MODEL_COLORS if m in pivot.columns]]
        metric = "directional accuracy" if mode == "Regression" else "accuracy"
        chunks.append(pivot.to_latex(
            caption=(f"Walk-forward {metric} (mean $\\pm$ std over folds) "
                     f"in {mode.lower()} mode."),
            label=f"tab:wf_{mode.lower()}",
            column_format="l" + "c" * len(pivot.columns),
        ).replace("±", "$\\pm$"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(chunks))
    print(f"LaTeX tables -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Walk-forward experiments for Chapter 7.")
    p.add_argument("--tickers", nargs="+", default=["NVDA", "AAPL", "TSLA", "SPY"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-06-04")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--modes", nargs="+", default=["Classification", "Regression"],
                   choices=["Classification", "Regression"])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    t_start = time.time()

    for ticker in args.tickers:
        print(f"\n=== {ticker} ({args.start} -> {args.end}) ===", flush=True)
        raw = DataLoader(ticker, args.start, args.end).get_data()
        if raw is None or raw.empty:
            print(f"  no data for {ticker}, skipping")
            continue

        for mode in args.modes:
            processor = DataPreprocessor(raw)
            processor.add_technical_indicators()
            registry = REGRESSORS if mode == "Regression" else CLASSIFIERS
            if mode == "Regression":
                X, y = processor.prepare_data_for_regression()
            else:
                X, y = processor.prepare_data_for_training()
            print(f"  {len(X)} usable rows, {X.shape[1]} features", flush=True)

            for name, factory in registry.items():
                t0 = time.time()
                scores = factory().walk_forward_score(X, y, n_splits=args.splits,
                                                      verbose=False)
                print(f"  {ticker} {mode:<14} {name:<14} "
                      f"mean={np.mean(scores):.4f}  std={np.std(scores):.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
                rows.extend({"ticker": ticker, "mode": mode, "model": name,
                             "fold": fold, "score": score}
                            for fold, score in enumerate(scores, start=1))

            # persist after every (ticker, mode) so partial runs survive
            df_scores = pd.DataFrame(rows)
            df_scores.to_csv(os.path.join(args.out, "walk_forward_scores.csv"),
                             index=False)
            plot_ticker(df_scores, ticker, mode, args.out)

    if not rows:
        print("Nothing was run.")
        return

    df_scores = pd.DataFrame(rows)
    summary = df_scores.groupby(["mode", "ticker", "model"])["score"].agg(
        mean="mean", std="std", folds="count").reset_index()
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    latex_tables(summary, os.path.join(args.out, "summary_tables.tex"))

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
