# A Braess-Like Effect in Hyperparameter Optimization

**Search-Space Expansion, Dilution, and Recovery Under Computational Constraints**

This repository contains the code and analysis used to study how progressively expanding a hyperparameter optimization (HPO) search space affects realized predictive performance under finite computational budgets.

The central question is simple:

> If a larger nested search space contains every configuration available in a smaller space, can the larger space still produce worse realized performance when optimization resources are limited?

The experiments evaluate this question under both **fixed trial-count budgets** and **fixed wall-clock budgets**.

---

## Experimental Setup

### Datasets

The study uses four public binary-classification datasets:

- Adult Income
- German Credit
- MAGIC Gamma Telescope
- Spambase

Datasets are obtained from public UCI/OpenML sources as implemented in `experiment.py`.

### Search Spaces

Four nested HPO spaces are evaluated:

- **S1:** Logistic Regression, Random Forest
- **S2:** S1 + XGBoost
- **S3:** S2 + Support Vector Machine, K-Nearest Neighbors
- **S4:** S3 + Decision Tree, Extra Trees, Multi-Layer Perceptron

Thus,

```text
S1 ⊂ S2 ⊂ S3 ⊂ S4
```

Existing model branches and hyperparameter ranges are preserved as the search space expands.

### Optimizers

Two optimization strategies are evaluated:

- Random Search
- Tree-structured Parzen Estimator (TPE), implemented with Optuna

### Primary Metric

The primary predictive metric is **ROC-AUC**.

The data are split using a fixed stratified:

```text
70% training / 15% validation / 15% test
```

The validation set is used for HPO. The selected configuration is then refit using the training and validation partitions before final evaluation on the held-out test set.

---

## Repository Structure

A typical repository layout is:

```text
.
├── experiment.py
├── analyze_results.py
├── reference_analysis.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── scripts/
│   ├── run_trial_experiments.ps1
│   ├── run_wallclock_experiments.ps1
│   ├── run_analysis.ps1
│   ├── run_trial_experiments.sh
│   ├── run_wallclock_experiments.sh
│   └── run_analysis.sh
│
├── results/
│   ├── trial_expansion_summary.csv
│   ├── wallclock_expansion_summary.csv
│   ├── opportunity_gain_pooled.csv
│   ├── opportunity_gain_by_optimizer.csv
│   ├── reference_best_pooled.csv
│   └── reference_best_by_optimizer.csv
│
└── figures/
    ├── expansion_penalty_random_trials.pdf
    ├── expansion_penalty_tpe_trials.pdf
    ├── expansion_penalty_random_time.pdf
    └── expansion_penalty_tpe_time.pdf
```

Raw per-trial outputs may be excluded from Git because they can become large.

---

## Software Environment

The reported experiments were implemented using:

```text
Python          3.13.14
scikit-learn    1.9.1
Optuna          5.0.0
XGBoost         3.4.1
```

No GPU acceleration was used.

Model-level parallelism was restricted to a single worker where applicable (`n_jobs=1`) to reduce nested parallelism and uncontrolled resource contention.

For wall-clock experiments, numerical-library thread counts are also restricted in the supplied run scripts.

---

# Installation

## Windows

### 1. Clone the repository

```powershell
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you may temporarily allow locally created scripts for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Linux

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running the Experiments

## 1. Trial-Count Experiment

The primary experiment evaluates:

```text
Datasets:     adult, credit, magic, spambase
Spaces:       S1, S2, S3, S4
Optimizers:   random, tpe
Seeds:        0-9
Checkpoints:  10, 25, 50, 75, 100 completed trials
```

### Windows

From the repository root:

```powershell
.\scripts\run_trial_experiments.ps1
```

### Linux

First make the script executable:

```bash
chmod +x scripts/run_trial_experiments.sh
```

Then run:

```bash
./scripts/run_trial_experiments.sh
```

The trial-count experiment is the primary hardware-comparative analysis because each optimizer receives the same number of completed evaluations.

---

## 2. Wall-Clock Experiment

The wall-clock analysis focuses on the S3-to-S4 expansion:

```text
Datasets:     adult, credit, magic, spambase
Spaces:       S3, S4
Optimizers:   random, tpe
Seeds:        0-9
Checkpoints:  30, 60, 120, 180 seconds
```

### Windows

```powershell
.\scripts\run_wallclock_experiments.ps1
```

### Linux

```bash
chmod +x scripts/run_wallclock_experiments.sh
./scripts/run_wallclock_experiments.sh
```

For reproducibility, wall-clock experiments should ideally be run on an otherwise lightly loaded system.

Absolute wall-clock results are hardware-dependent. They should therefore be interpreted as measurements under the reported computational environment rather than universal time thresholds.

---

# Analysis

The analysis code computes quantities including:

- search-space expansion penalty
- median expansion penalty
- paradox incidence
- confidence intervals
- paired Wilcoxon signed-rank tests
- Holm-adjusted p-values
- sustained recovery behavior
- evaluation throughput
- aggregate high-budget opportunity estimates

## Run the Main Analysis

The supplied analysis scripts process both the trial-count and wall-clock
checkpoint files.

### Windows

```powershell
.\scripts\run_analysis.ps1
```

The script runs the equivalent of:

```powershell
python analyze_results.py `
  --input trial_results/checkpoint_results.csv `
  --output analysis/trial

python analyze_results.py `
  --input wallclock_results/checkpoint_results.csv `
  --output analysis/wallclock
```

### Linux

First make the script executable:

```bash
chmod +x scripts/run_analysis.sh
```

Then run:

```bash
./scripts/run_analysis.sh
```

The script runs the equivalent of:

```bash
python3 analyze_results.py \
  --input trial_results/checkpoint_results.csv \
  --output analysis/trial

python3 analyze_results.py \
  --input wallclock_results/checkpoint_results.csv \
  --output analysis/wallclock
```

A typical analysis output structure is:

```text
analysis/
├── trial/
│   ├── expansion_summary.csv
│   ├── expansion_by_dataset.csv
│   ├── recovery_budget.csv
│   └── ...
│
└── wallclock/
    ├── expansion_summary.csv
    ├── expansion_by_dataset.csv
    ├── recovery_budget.csv
    └── ...
```

The recovery analysis should use **sustained recovery**: the first tested
budget after which the mean expansion penalty remains non-positive at all
subsequent tested checkpoints. If recovery does not occur within the
observed range, it should be reported as exceeding the largest tested
budget.

To inspect the available command-line options directly:

### Windows

```powershell
python analyze_results.py --help
python reference_analysis.py --help
```

### Linux

```bash
python3 analyze_results.py --help
python3 reference_analysis.py --help
```

The main processed result files used for the manuscript are stored in `results/`.

---

# Key Quantities

## Search-Space Expansion Penalty

For nested spaces \(S_i \subset S_j\), the expansion penalty at budget \(B\) is defined as

\[
\Delta_{\mathrm{exp}}(S_i,S_j,B)
=
\mathbb{E}[P_A(S_i,B)]
-
\mathbb{E}[P_A(S_j,B)].
\]

A positive value indicates that the larger search space produced lower realized performance.

## Paradox Incidence

\[
\pi_{ij}(B)
=
\Pr\left[P_A(S_j,B) < P_A(S_i,B)\right].
\]

This measures how often the larger space performs worse than the smaller one across paired experimental runs.

## Recovery

Recovery is evaluated using the smallest tested budget after which the expansion penalty remains non-positive for all larger tested budgets.

If no such point occurs within the tested range, recovery is reported as exceeding the maximum tested budget.

---

# Aggregate High-Budget Reference Analysis

The repository also includes an aggregate reference analysis that pools evaluated configurations from the completed Random Search and TPE runs.

The main files are:

```text
opportunity_gain_pooled.csv
opportunity_gain_by_optimizer.csv
reference_best_pooled.csv
reference_best_by_optimizer.csv
```

The pooled reference combines configurations generated across both optimizers and is used as a diagnostic estimate of the incremental opportunity available in the larger search space.

This analysis is **not** treated as an exhaustive search or as a single sequential optimization run.

Because the reference pool is finite, an empirical pooled estimate for S4 may occasionally fall below the corresponding S3 estimate even though the theoretical optimum of a nested search space cannot decrease.

---

# Reproducibility Notes

1. Use the same software versions when reproducing the reported results.
2. Keep model-level parallelism restricted to one worker where applicable.
3. Avoid running unrelated CPU-intensive jobs during wall-clock experiments.
4. Use the same random seeds when reproducing the manuscript tables.
5. Trial-count results are more portable across hardware than absolute wall-clock results.
6. The first TPE trials may behave similarly to random exploration because of Optuna's startup phase.
7. Internet access may be required the first time public datasets are downloaded.

---

# Processed Results

The repository retains the compact processed outputs used to generate the manuscript tables and figures.

Large intermediate files, caches, failed pilot runs, and temporary experiment folders are excluded through `.gitignore`.

---

# Paper

The code accompanies the manuscript:

> **A Braess-Like Effect in Hyperparameter Optimization: Search-Space Expansion, Dilution, and Recovery Under Computational Constraints**

If you use this repository or build on the experimental framework, please cite the corresponding paper.

---

# License

Add the license selected for the repository here, for example MIT, BSD-3-Clause, or Apache-2.0.
