from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SPACE_ORDER = ["S1", "S2", "S3", "S4"]
ADJACENT_PAIRS = [("S1", "S2"), ("S2", "S3"), ("S3", "S4")]


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=2026):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def paired_frame(df, small, large, group_cols):
    a = df[df["space"] == small][group_cols + ["test_auc"]].rename(columns={"test_auc": "small_auc"})
    b = df[df["space"] == large][group_cols + ["test_auc"]].rename(columns={"test_auc": "large_auc"})
    out = a.merge(b, on=group_cols, how="inner")
    out["penalty"] = out["small_auc"] - out["large_auc"]
    out["paradox"] = out["penalty"] > 0
    return out


def summarize_expansion(df):
    group_cols = ["dataset", "optimizer", "budget_mode", "checkpoint", "seed"]
    rows = []
    for small, large in ADJACENT_PAIRS:
        paired = paired_frame(df, small, large, group_cols)
        for keys, g in paired.groupby(["optimizer", "budget_mode", "checkpoint"], dropna=False):
            opt, mode, checkpoint = keys
            penalties = g["penalty"].to_numpy()
            lo, hi = bootstrap_ci(penalties)
            try:
                w = wilcoxon(penalties, alternative="two-sided", zero_method="wilcox")
                p = float(w.pvalue)
            except ValueError:
                p = 1.0
            rows.append({
                "small_space": small,
                "large_space": large,
                "optimizer": opt,
                "budget_mode": mode,
                "checkpoint": checkpoint,
                "n_pairs": len(g),
                "mean_penalty": float(np.mean(penalties)),
                "median_penalty": float(np.median(penalties)),
                "ci95_low": lo,
                "ci95_high": hi,
                "paradox_incidence": float(np.mean(g["paradox"])),
                "wilcoxon_p": p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p"] = holm_adjust(out["wilcoxon_p"].to_numpy())
    return out


def summarize_by_dataset(df):
    group_cols = ["dataset", "optimizer", "budget_mode", "checkpoint", "seed"]
    rows = []
    for small, large in ADJACENT_PAIRS:
        paired = paired_frame(df, small, large, group_cols)
        for keys, g in paired.groupby(["dataset", "optimizer", "budget_mode", "checkpoint"], dropna=False):
            dataset, opt, mode, checkpoint = keys
            rows.append({
                "dataset": dataset,
                "small_space": small,
                "large_space": large,
                "optimizer": opt,
                "budget_mode": mode,
                "checkpoint": checkpoint,
                "mean_penalty": g["penalty"].mean(),
                "median_penalty": g["penalty"].median(),
                "paradox_incidence": g["paradox"].mean(),
                "n_seeds": len(g),
            })
    return pd.DataFrame(rows)


def estimate_recovery(summary):
    rows = []
    keys = ["small_space", "large_space", "optimizer", "budget_mode"]
    for group_keys, g in summary.groupby(keys):
        g = g.sort_values("checkpoint")
        nonharmful = g[g["mean_penalty"] <= 0]
        recovery = float(nonharmful.iloc[0]["checkpoint"]) if not nonharmful.empty else np.nan
        rows.append(dict(zip(keys, group_keys), estimated_recovery_budget=recovery))
    return pd.DataFrame(rows)


def plot_mean_performance(df, out_dir):
    means = (
        df.groupby(["optimizer", "budget_mode", "checkpoint", "space"], as_index=False)["test_auc"]
        .mean()
    )
    for (opt, mode), g in means.groupby(["optimizer", "budget_mode"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for space in SPACE_ORDER:
            sg = g[g["space"] == space].sort_values("checkpoint")
            if len(sg):
                ax.plot(sg["checkpoint"], sg["test_auc"], marker="o", label=space)
        ax.set_xlabel("Trials" if mode == "trials" else "Seconds")
        ax.set_ylabel("Mean test ROC-AUC")
        ax.set_title(f"Realized performance vs. budget — {opt}, {mode}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"performance__{opt}__{mode}.png", dpi=200)
        plt.close(fig)


def plot_penalty(summary, out_dir):
    for (opt, mode), g in summary.groupby(["optimizer", "budget_mode"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for small, large in ADJACENT_PAIRS:
            sg = g[(g.small_space == small) & (g.large_space == large)].sort_values("checkpoint")
            if len(sg):
                ax.plot(sg["checkpoint"], sg["mean_penalty"], marker="o", label=f"{small}→{large}")
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Trials" if mode == "trials" else "Seconds")
        ax.set_ylabel("Mean expansion penalty (ROC-AUC)")
        ax.set_title(f"Search-space expansion penalty — {opt}, {mode}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"expansion_penalty__{opt}__{mode}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/checkpoint_results.csv")
    parser.add_argument("--output", default="analysis")
    args = parser.parse_args()

    in_file = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_file)
    required = {"dataset", "space", "optimizer", "seed", "budget_mode", "checkpoint", "test_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {sorted(missing)}")

    summary = summarize_expansion(df)
    by_dataset = summarize_by_dataset(df)
    recovery = estimate_recovery(summary)

    summary.to_csv(out_dir / "expansion_summary.csv", index=False)
    by_dataset.to_csv(out_dir / "expansion_by_dataset.csv", index=False)
    recovery.to_csv(out_dir / "recovery_budget.csv", index=False)

    plot_mean_performance(df, out_dir)
    plot_penalty(summary, out_dir)

    print("Wrote:")
    print(out_dir / "expansion_summary.csv")
    print(out_dir / "expansion_by_dataset.csv")
    print(out_dir / "recovery_budget.csv")
    print("and PNG figures in the same directory.")


if __name__ == "__main__":
    main()
