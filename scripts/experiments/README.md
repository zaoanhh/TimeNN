# Additional experiments

## Motivation study (Fig. 3)

```bash
uv run python scripts/experiments/run_motivation_experiment.py \
  --widar-root /path/to/widar3/CSI_Processed/20181130 \
  --xrf55-root /path/to/xrf55-all \
  --output-csv runs/motivation/results.csv \
  --devices cuda:0 cuda:1
```

* `run_motivation_experiment.py` queues `dataset × model × ratio` runs and
  skips combinations already present in the CSV.
* `train_motivation_model.py` trains one baseline and appends clean and warped
  test accuracy.

The seven baselines are `resnet18`, `resnet34`, `cnn`, `inception`,
`transformer`, `lstm` and `gru`, using the model definitions in
`timenn.models`.
