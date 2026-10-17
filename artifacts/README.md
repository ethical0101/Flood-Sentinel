# artifacts/

Training and evaluation outputs. Checkpoints (`*.pt`) are git-ignored. JSON and PNG results are small and can be committed.

```
runs/<run_name>/
  config.yaml          exact configuration used (seed, data, model, optimiser)
  environment.json     Python / torch / CUDA / GPU / driver versions
  stats.json           train-split normalisation statistics used for this run's inputs
  history.json         one record per epoch (losses, validation metrics, LR, time, peak GPU memory)
  status.json          live state (running / completed / interrupted / failed), read by the dashboard
  train.log            human-readable training log
  checkpoints/best.pt  best validation-IoU weights (used for evaluation and serving)
  checkpoints/last.pt  latest weights + optimiser/scheduler/scaler state (for --resume)
  evaluation.json      held-out metrics (valid/test/bolivia), per-event metrics, latency
  curves.png           training curves
  samples/*.png        deterministic test-chip prediction panels
baselines.json         non-learned baselines under the same protocol
summary.json           cross-run comparison
figures/               comparison plots
dataset_verification.json, system_info.json, experiments.log
```
