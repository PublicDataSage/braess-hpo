from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment import load_dataset, fit_eval_params


def load_trials(root: Path, output_dirs: list[str]) -> pd.DataFrame:
    frames = []
    for out_name in output_dirs:
        trials_dir = root / out_name / "trials"
        if not trials_dir.exists():
            print(f"WARNING: missing trials directory: {trials_dir}")
            continue

        files = sorted(trials_dir.glob("*.csv"))
        print(f"{out_name}: found {len(files)} trial files")

        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            frames.append(df)

    if not frames:
        raise SystemExit("No trial CSV files were found.")

    df = pd.concat(frames, ignore_index=True)

    required = {
        "dataset", "space", "optimizer", "seed", "state",
        "val_auc", "params_json"
    }
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df[df["state"].astype(str).str.upper().eq("COMPLETE")].copy()
    df["val_auc"] = pd.to_numeric(df["val_auc"], errors="coerce")
    df = df.dropna(subset=["val_auc", "params_json"])
    return df


def evaluate_best_row(row: pd.Series, data_cache: dict) -> dict:
    dataset = str(row["dataset"])
    if dataset not in data_cache:
        print(f"Loading dataset: {dataset}")
        data_cache[dataset] = load_dataset(dataset)

    params = json.loads(row["params_json"])
    seed = int(row["seed"])
    test_auc = fit_eval_params(params, data_cache[dataset], seed=seed)

    return {
        "dataset": dataset,
        "space": str(row["space"]),
        "optimizer": str(row["optimizer"]),
        "reference_seed": seed,
        "reference_trial_number": int(row["trial_number"]) if "trial_number" in row.index else None,
        "reference_model": params.get("model"),
        "reference_val_auc": float(row["val_auc"]),
        "reference_test_auc": float(test_auc),
        "params_json": row["params_json"],
        "source_file": row["source_file"],
    }


def build_reference(df: pd.DataFrame, spaces: list[str], pool_optimizers: bool):
    data_cache = {}
    work = df[df["space"].isin(spaces)].copy()

    if pool_optimizers:
        group_cols = ["dataset", "space"]
    else:
        group_cols = ["dataset", "space", "optimizer"]

    rows = []
    for keys, g in work.groupby(group_cols, sort=True):
        best_idx = g["val_auc"].idxmax()
        best = g.loc[best_idx]
        rec = evaluate_best_row(best, data_cache)
        rec["n_completed_trials_in_pool"] = int(len(g))
        if pool_optimizers:
            rec["reference_pool"] = "random+tpe"
        else:
            rec["reference_pool"] = rec["optimizer"]
        rows.append(rec)

    return pd.DataFrame(rows)


def opportunity_table(ref: pd.DataFrame, s_small: str, s_large: str):
    id_cols = ["dataset", "reference_pool"]
    small = ref[ref["space"] == s_small].copy()
    large = ref[ref["space"] == s_large].copy()

    small = small.rename(columns={
        "reference_val_auc": f"{s_small}_ref_val_auc",
        "reference_test_auc": f"{s_small}_ref_test_auc",
        "reference_model": f"{s_small}_ref_model",
        "n_completed_trials_in_pool": f"{s_small}_n_trials",
    })
    large = large.rename(columns={
        "reference_val_auc": f"{s_large}_ref_val_auc",
        "reference_test_auc": f"{s_large}_ref_test_auc",
        "reference_model": f"{s_large}_ref_model",
        "n_completed_trials_in_pool": f"{s_large}_n_trials",
    })

    keep_small = id_cols + [
        f"{s_small}_ref_val_auc", f"{s_small}_ref_test_auc",
        f"{s_small}_ref_model", f"{s_small}_n_trials"
    ]
    keep_large = id_cols + [
        f"{s_large}_ref_val_auc", f"{s_large}_ref_test_auc",
        f"{s_large}_ref_model", f"{s_large}_n_trials"
    ]

    out = small[keep_small].merge(large[keep_large], on=id_cols, how="inner")
    out["opportunity_gain_val"] = (
        out[f"{s_large}_ref_val_auc"] - out[f"{s_small}_ref_val_auc"]
    )
    out["reference_test_difference"] = (
        out[f"{s_large}_ref_test_auc"] - out[f"{s_small}_ref_test_auc"]
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build an aggregate high-budget reference from existing per-trial HPO logs."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["adult_10seed", "credit_10seed", "magic_10seed", "spambase_10seed"],
    )
    parser.add_argument("--spaces", default="S3,S4")
    parser.add_argument("--output", default="reference_analysis")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    if len(spaces) != 2:
        raise SystemExit("--spaces must contain exactly two spaces, e.g. S3,S4")

    trials = load_trials(root, args.outputs)
    print(f"Total completed trial rows loaded: {len(trials)}")

    # Reference within each optimizer: normally ~1,000 trials per dataset/space.
    by_opt = build_reference(trials, spaces, pool_optimizers=False)
    by_opt.to_csv(out_dir / "reference_best_by_optimizer.csv", index=False)

    # Stronger pooled reference: normally ~2,000 trials per dataset/space.
    pooled = build_reference(trials, spaces, pool_optimizers=True)
    pooled.to_csv(out_dir / "reference_best_pooled.csv", index=False)

    # Normalize pooled label so opportunity table can merge.
    pooled["reference_pool"] = "random+tpe"
    opp_pooled = opportunity_table(pooled, spaces[0], spaces[1])
    opp_pooled.to_csv(out_dir / "opportunity_gain_pooled.csv", index=False)

    # Optimizer-specific opportunity tables.
    by_opt["reference_pool"] = by_opt["optimizer"]
    opp_opt = opportunity_table(by_opt, spaces[0], spaces[1])
    opp_opt.to_csv(out_dir / "opportunity_gain_by_optimizer.csv", index=False)

    print("\nPooled reference results:")
    cols = [
        "dataset", "reference_pool",
        f"{spaces[0]}_ref_val_auc", f"{spaces[1]}_ref_val_auc",
        "opportunity_gain_val",
        f"{spaces[0]}_ref_test_auc", f"{spaces[1]}_ref_test_auc",
        "reference_test_difference",
    ]
    print(opp_pooled[cols].to_string(index=False))

    print(f"\nSaved results to: {out_dir.resolve()}")
    print("\nInterpretation:")
    print("  opportunity_gain_val > 0  => expanded space found a better validation optimum in the pooled reference.")
    print("  opportunity_gain_val <= 0 => the pooled search did not empirically improve the best validation score.")
    print("The reference is diagnostic; theoretical non-degradation still follows from S3 being a subset of S4.")


if __name__ == "__main__":
    main()
