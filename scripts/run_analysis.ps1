# Analyze both trial-count and wall-clock experiment outputs

$ErrorActionPreference = "Stop"

Write-Host "Analyzing trial-count results..."

python analyze_results.py `
  --input trial_results/checkpoint_results.csv `
  --output analysis/trial

Write-Host "Analyzing wall-clock results..."

python analyze_results.py `
  --input wallclock_results/checkpoint_results.csv `
  --output analysis/wallclock

Write-Host "Analysis completed."
Write-Host "Trial results: analysis/trial"
Write-Host "Wall-clock results: analysis/wallclock"