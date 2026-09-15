# Trial-count experiment
# 4 datasets x 4 search spaces x 2 optimizers x 10 seeds
# Checkpoints: 10, 25, 50, 75, 100 trials

$ErrorActionPreference = "Stop"

Write-Host "Starting trial-count experiments..."

python experiment.py `
  --datasets adult,credit,magic,spambase `
  --spaces S1,S2,S3,S4 `
  --optimizers random,tpe `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --budget-mode trials `
  --trial-checkpoints 10,25,50,75,100 `
  --output trial_results

Write-Host "Trial-count experiments completed."