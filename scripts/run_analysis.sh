#!/usr/bin/env bash
set -euo pipefail

echo "Analyzing trial-count results..."

python3 analyze_results.py \
  --input trial_results/checkpoint_results.csv \
  --output analysis/trial

echo "Analyzing wall-clock results..."

python3 analyze_results.py \
  --input wallclock_results/checkpoint_results.csv \
  --output analysis/wallclock

echo "Analysis completed."
echo "Trial results: analysis/trial"
echo "Wall-clock results: analysis/wallclock"