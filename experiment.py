from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import psutil
import sklearn
import xgboost
from optuna.samplers import RandomSampler, TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Candidate OpenML names. The loader tries each in order so that minor naming
# differences across OpenML versions do not require editing the experiment code.
DATASETS = {
    "adult": ["adult"],
    "bank": ["bank-marketing", "bank_marketing"],
    "credit": ["credit-g", "GermanCredit"],
    "magic": ["MagicTelescope", "magic-telescope", "MAGIC-Gamma-Telescope"],
    "spambase": ["spambase", "Spambase"],
}

SPACE_MODELS = {
    "S1": ["logreg", "rf"],
    "S2": ["logreg", "rf", "xgb"],
    "S3": ["logreg", "rf", "xgb", "svm", "knn"],
    "S4": ["logreg", "rf", "xgb", "svm", "knn", "dt", "extratrees", "mlp"],
}

DEFAULT_TRIAL_CHECKPOINTS = [10, 25, 50, 75, 100]
DEFAULT_TIME_CHECKPOINTS = [60, 180, 300, 600]


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    positive_class: str


def _fetch_openml_candidates(candidates: list[str]):
    last_error = None
    for name in candidates:
        try:
            return fetch_openml(name=name, version="active", as_frame=True, parser="auto")
        except Exception as exc:  # try the next alias
            last_error = exc
    raise RuntimeError(
        f"Unable to load OpenML dataset using aliases {candidates}. "
        f"Last error: {last_error}"
    )


def load_dataset(name: str, split_seed: int = 2026) -> DataBundle:
    """Load a binary-classification dataset and create a fixed 70/15/15 split."""
    if name == "smoke":
        bunch = load_breast_cancer(as_frame=True)
        X = bunch.data.copy()
        y = pd.Series(bunch.target, name="target").astype(str)
    else:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Options: {list(DATASETS)} + ['smoke']")
        bunch = _fetch_openml_candidates(DATASETS[name])
        X = bunch.data.copy()
        y = pd.Series(bunch.target).copy()

    # Remove rows with missing targets, if any.
    valid = ~pd.isna(y)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True).astype(str)

    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Dataset '{name}' is not binary: classes={classes}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    positive_class = str(le.classes_[1])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.15,
        stratify=y_enc,
        random_state=split_seed,
    )
    # 15 / 85 of the remaining data gives an overall 15% validation split.
    val_fraction_of_trainval = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=split_seed,
    )

    return DataBundle(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train),
        y_val=np.asarray(y_val),
        y_test=np.asarray(y_test),
        positive_class=positive_class,
    )


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )


def suggest_model(trial: optuna.Trial, space: str, seed: int):
    allowed = SPACE_MODELS[space]
    model = trial.suggest_categorical("model", allowed)

    if model == "logreg":
        return LogisticRegression(
            C=trial.suggest_float("logreg_C", 1e-4, 1e2, log=True),
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )

    if model == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 20),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=seed,
        )

    if model == "xgb":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 8),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.30, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )

    if model == "svm":
        kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        kwargs: dict[str, Any] = {
            "C": trial.suggest_float("svm_C", 1e-3, 1e2, log=True),
            "kernel": kernel,
            "probability": False,
            "random_state": seed,
        }
        if kernel == "rbf":
            kwargs["gamma"] = trial.suggest_float("svm_gamma", 1e-5, 1.0, log=True)
        return SVC(**kwargs)

    if model == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 3, 50),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            p=trial.suggest_categorical("knn_p", [1, 2]),
            n_jobs=1,
        )

    if model == "dt":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("dt_max_depth", 2, 30),
            min_samples_split=trial.suggest_int("dt_min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("dt_min_samples_leaf", 1, 15),
            criterion=trial.suggest_categorical("dt_criterion", ["gini", "entropy", "log_loss"]),
            random_state=seed,
        )

    if model == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("et_max_depth", 3, 25),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 20),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=seed,
        )

    if model == "mlp":
        hidden = trial.suggest_categorical("mlp_hidden", ["32", "64", "64x32", "128x64"])
        hidden_tuple = tuple(int(v) for v in hidden.split("x"))
        return MLPClassifier(
            hidden_layer_sizes=hidden_tuple,
            alpha=trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            max_iter=250,
            early_stopping=True,
            random_state=seed,
        )

    raise RuntimeError(f"Unhandled model: {model}")


def model_from_params(params: dict[str, Any], seed: int):
    """Reconstruct a model from saved Optuna parameters."""
    model = params["model"]

    if model == "logreg":
        return LogisticRegression(C=params["logreg_C"], solver="liblinear", max_iter=1000, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=params["rf_n_estimators"], max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"], max_features=params["rf_max_features"],
            n_jobs=1, random_state=seed,
        )
    if model == "xgb":
        return XGBClassifier(
            n_estimators=params["xgb_n_estimators"], max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"], subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample_bytree"], reg_lambda=params["xgb_reg_lambda"],
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1,
            random_state=seed,
        )
    if model == "svm":
        kwargs = dict(C=params["svm_C"], kernel=params["svm_kernel"], probability=False, random_state=seed)
        if params["svm_kernel"] == "rbf":
            kwargs["gamma"] = params["svm_gamma"]
        return SVC(**kwargs)
    if model == "knn":
        return KNeighborsClassifier(
            n_neighbors=params["knn_n_neighbors"], weights=params["knn_weights"], p=params["knn_p"], n_jobs=1
        )
    if model == "dt":
        return DecisionTreeClassifier(
            max_depth=params["dt_max_depth"], min_samples_split=params["dt_min_samples_split"],
            min_samples_leaf=params["dt_min_samples_leaf"], criterion=params["dt_criterion"], random_state=seed
        )
    if model == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=params["et_n_estimators"], max_depth=params["et_max_depth"],
            min_samples_split=params["et_min_samples_split"], max_features=params["et_max_features"],
            n_jobs=1, random_state=seed,
        )
    if model == "mlp":
        hidden_tuple = tuple(int(v) for v in params["mlp_hidden"].split("x"))
        return MLPClassifier(
            hidden_layer_sizes=hidden_tuple, alpha=params["mlp_alpha"], learning_rate_init=params["mlp_lr"],
            max_iter=250, early_stopping=True, random_state=seed,
        )
    raise RuntimeError(f"Unhandled model in saved params: {model}")


def continuous_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return np.asarray(estimator.predict(X)).ravel()


def fit_eval_params(params: dict[str, Any], data: DataBundle, seed: int) -> float:
    """Refit selected config on train+validation and evaluate once on untouched test data."""
    X_fit = pd.concat([data.X_train, data.X_val], axis=0, ignore_index=True)
    y_fit = np.concatenate([data.y_train, data.y_val])
    pipe = Pipeline([
        ("prep", make_preprocessor(X_fit)),
        ("model", model_from_params(params, seed)),
    ])
    pipe.fit(X_fit, y_fit)
    scores = continuous_scores(pipe, data.X_test)
    return float(roc_auc_score(data.y_test, scores))


def make_sampler(name: str, seed: int):
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "tpe":
        return TPESampler(seed=seed, n_startup_trials=10)
    raise ValueError(f"Unknown optimizer: {name}")


def best_trial_at_trial_checkpoint(study: optuna.Study, checkpoint: int):
    eligible = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number < checkpoint]
    return max(eligible, key=lambda t: t.value) if eligible else None


def best_trial_at_time_checkpoint(study: optuna.Study, checkpoint_sec: float):
    eligible = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and float(t.user_attrs.get("elapsed_end_sec", np.inf)) <= checkpoint_sec
    ]
    return max(eligible, key=lambda t: t.value) if eligible else None


def run_one(
    dataset: str,
    space: str,
    optimizer: str,
    seed: int,
    budget_mode: str,
    output_dir: Path,
    trial_checkpoints: list[int],
    time_checkpoints: list[int],
    split_seed: int,
) -> list[dict[str, Any]]:
    data = load_dataset(dataset, split_seed=split_seed)
    sampler = make_sampler(optimizer, seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        model = suggest_model(trial, space=space, seed=seed)
        pipe = Pipeline([
            ("prep", make_preprocessor(data.X_train)),
            ("model", model),
        ])
        t0 = time.perf_counter()
        pipe.fit(data.X_train, data.y_train)
        val_scores = continuous_scores(pipe, data.X_val)
        value = float(roc_auc_score(data.y_val, val_scores))
        t1 = time.perf_counter()
        trial.set_user_attr("fit_eval_sec", t1 - t0)
        trial.set_user_attr("elapsed_end_sec", t1 - start)
        return value

    if budget_mode == "trials":
        study.optimize(objective, n_trials=max(trial_checkpoints), n_jobs=1, show_progress_bar=False, catch=(Exception,))
        checkpoints = trial_checkpoints
    elif budget_mode == "time":
        study.optimize(
            objective,
            n_trials=100000,
            timeout=max(time_checkpoints),
            n_jobs=1,
            show_progress_bar=False,
            catch=(Exception,),
        )
        checkpoints = time_checkpoints
    else:
        raise ValueError("budget_mode must be 'trials' or 'time'")

    total_elapsed = time.perf_counter() - start

    # Persist per-trial information before test evaluation.
    trial_rows = []
    for t in study.trials:
        trial_rows.append({
            "dataset": dataset,
            "space": space,
            "optimizer": optimizer,
            "seed": seed,
            "budget_mode": budget_mode,
            "trial_number": t.number,
            "state": t.state.name,
            "val_auc": t.value,
            "elapsed_end_sec": t.user_attrs.get("elapsed_end_sec"),
            "fit_eval_sec": t.user_attrs.get("fit_eval_sec"),
            "params_json": json.dumps(t.params, sort_keys=True),
        })
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_file = trials_dir / f"{dataset}__{space}__{optimizer}__seed{seed}__{budget_mode}.csv"
    pd.DataFrame(trial_rows).to_csv(trial_file, index=False)

    results = []
    for checkpoint in checkpoints:
        if budget_mode == "trials":
            bt = best_trial_at_trial_checkpoint(study, checkpoint)
            completed = sum(
                t.state == optuna.trial.TrialState.COMPLETE and t.number < checkpoint
                for t in study.trials
            )
        else:
            bt = best_trial_at_time_checkpoint(study, checkpoint)
            completed = sum(
                t.state == optuna.trial.TrialState.COMPLETE
                and float(t.user_attrs.get("elapsed_end_sec", np.inf)) <= checkpoint
                for t in study.trials
            )

        if bt is None:
            continue

        test_auc = fit_eval_params(bt.params, data=data, seed=seed)
        results.append({
            "dataset": dataset,
            "space": space,
            "optimizer": optimizer,
            "seed": seed,
            "budget_mode": budget_mode,
            "checkpoint": checkpoint,
            "n_completed_trials": completed,
            "best_trial_number": bt.number,
            "selected_model": bt.params.get("model"),
            "best_val_auc": float(bt.value),
            "test_auc": test_auc,
            "positive_class": data.positive_class,
            "run_total_hpo_sec": total_elapsed,
            "best_params_json": json.dumps(bt.params, sort_keys=True),
        })

    return results


def write_metadata(output_dir: Path, args: argparse.Namespace):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "optuna": optuna.__version__,
            "xgboost": xgboost.__version__,
        },
        "args": vars(args),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def append_results(rows: list[dict[str, Any]], result_file: Path):
    df = pd.DataFrame(rows)
    if result_file.exists():
        df.to_csv(result_file, mode="a", header=False, index=False)
    else:
        df.to_csv(result_file, index=False)


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Braess-like HPO search-space expansion experiments")
    parser.add_argument("--datasets", default="adult,bank,credit,magic,spambase")
    parser.add_argument("--spaces", default="S1,S2,S3,S4")
    parser.add_argument("--optimizers", default="random,tpe")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--budget-mode", choices=["trials", "time"], default="trials")
    parser.add_argument("--trial-checkpoints", default=",".join(map(str, DEFAULT_TRIAL_CHECKPOINTS)))
    parser.add_argument("--time-checkpoints", default=",".join(map(str, DEFAULT_TIME_CHECKPOINTS)))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    spaces = [x.strip() for x in args.spaces.split(",") if x.strip()]
    optimizers = [x.strip() for x in args.optimizers.split(",") if x.strip()]
    seeds = parse_int_list(args.seeds)
    trial_checkpoints = parse_int_list(args.trial_checkpoints)
    time_checkpoints = parse_int_list(args.time_checkpoints)

    output_dir = Path(args.output)
    write_metadata(output_dir, args)
    result_file = output_dir / "checkpoint_results.csv"

    for dataset in datasets:
        for space in spaces:
            for optimizer in optimizers:
                for seed in seeds:
                    print(f"RUN dataset={dataset} space={space} optimizer={optimizer} seed={seed} mode={args.budget_mode}", flush=True)
                    try:
                        rows = run_one(
                            dataset=dataset,
                            space=space,
                            optimizer=optimizer,
                            seed=seed,
                            budget_mode=args.budget_mode,
                            output_dir=output_dir,
                            trial_checkpoints=trial_checkpoints,
                            time_checkpoints=time_checkpoints,
                            split_seed=args.split_seed,
                        )
                        append_results(rows, result_file)
                    except Exception as exc:
                        error_file = output_dir / "errors.log"
                        with error_file.open("a", encoding="utf-8") as fh:
                            fh.write(f"dataset={dataset} space={space} optimizer={optimizer} seed={seed} mode={args.budget_mode}: {repr(exc)}\n")
                        print(f"ERROR: {exc}", flush=True)

    print(f"Finished. Checkpoint results: {result_file}")


if __name__ == "__main__":
    main()
