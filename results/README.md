# Result Files

## `seed_metrics_summary.csv`

Three-seed controlled comparison using sequence-grouped checkpoints.
Seeds are 42, 0, and 1. Values are sample mean and sample standard deviation.

## `seed_metrics_detailed.json`

Per-seed values used to compute the summary.

## `resource_benchmark.csv`

End-to-end latency and memory measurements for the sequence-grouped YOLOv8n
and YOLOv8s checkpoints. Protocol:

- 50 validation images
- 640x640 input
- batch size 1
- confidence 0.25
- IoU 0.50
- 15 warm-up predictions
- 10 repeated trials
- CUDA synchronization before and after each trial
- timing boundary includes image decoding, preprocessing, inference, NMS and postprocessing

Full per-model JSON records are stored in `results/benchmarks/`.

## `resolution_sweep_historical.csv`

Historical inference-only resolution sweep. These measurements used the
earlier `runs/visdrone` checkpoints, not the sequence-grouped checkpoints used
for the main three-seed comparison. The 640-pixel values are therefore not
directly comparable with `seed_metrics_summary.csv`.

## `experiment_log.csv`

Compact index linking each reported result to its split, run type and metric.
