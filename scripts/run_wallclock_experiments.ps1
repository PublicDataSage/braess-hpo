# Wall-clock experiment
# 4 datasets x 2 search spaces x 2 optimizers x 10 seeds
# Checkpoints: 30, 60, 120, 180 seconds

$ErrorActionPreference = "Stop"

# Restrict underlying numerical libraries to one thread per process
# to reduce uncontrolled nested parallelism.
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Write-Host "Starting wall-clock experiments..."

python experiment.py `
  --datasets adult,credit,magic,spambase `
  --spaces S3,S4 `
  --optimizers random,tpe `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --budget-mode time `
  --time-checkpoints 30,60,120,180 `
  --output wallclock_results

Write-Host "Wall-clock experiments completed."