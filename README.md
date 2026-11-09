# Accuracy-Efficiency Trade-offs of Lightweight YOLO Models for UAV Small-Object Detection

This repository contains the implementation, configurations, representative results, and documentation for a study of the accuracy-efficiency trade-off between YOLOv8n and YOLOv8s on VisDrone UAV imagery. The project compares detection accuracy, inference latency, model size, and GPU memory usage under documented hardware constraints.

## Research objective

The study asks how lightweight YOLO models trade detection accuracy for deployment cost in dense aerial scenes. The main comparisons use:

- VisDrone-DET
- YOLOv8n and YOLOv8s trained from official pretrained weights
- 100 epochs
- 640x640 input
- batch size 4
- seeds 42, 0, and 1
- a sequence-grouped train/validation split with zero overlapping video sequences

The repository connects every reported result to a data split, training configuration, checkpoint identity, and evaluation protocol.

## Repository structure

```text
baha-yolo-repro/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── AI_DISCLOSURE.md
├── configs/
├── data/
│   ├── README.md
│   └── split_manifest_recorded.json
├── src/
│   ├── common.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_seeds.py
│   ├── benchmark_resources.py
│   └── resolution_sweep.py
├── scripts/
├── results/
├── figures/
└── docs/
```

## Hardware requirements

### Recommended course hardware

- NVIDIA RTX A5000 with 24 GB VRAM

### Hardware used for the reported measurements

- Intel Core i7-10750H
- NVIDIA GeForce GTX 1650 Ti with 4 GB VRAM
- Windows 11

The reported measurements were not collected on an RTX A5000. The A5000 is listed as the recommended course environment, while the GTX 1650 Ti is the actual measurement environment.

## Software and dependencies

- Python 3.11
- PyTorch 2.5.1+cu121
- Torchvision 0.20.1+cu121
- Ultralytics 8.4.152
- OpenCV, NumPy, Pandas, Matplotlib, PyYAML, tqdm

The reported environment used CUDA 12.1.

## Installation

```bash
git clone https://github.com/miao123rui/baha-yolo-repro.git
cd baha-yolo-repro

conda create -n baha-yolo python=3.11 -y
conda activate baha-yolo

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On Windows, keep `workers: 0` in the provided configs to avoid Jupyter/DataLoader multiprocessing issues.

## Dataset

The project uses VisDrone-DET. The dataset is not redistributed here.

- Official source: https://github.com/VisDrone/VisDrone-Dataset
- Dataset instructions: [data/README.md](data/README.md)
- Recorded split manifest: [data/split_manifest_recorded.json](data/split_manifest_recorded.json)

After obtaining and arranging the official files:

```bash
python src/prepare_data.py \
  --visdrone-root data/raw/VisDrone \
  --output-root data/VisDrone_subset_seq_v2 \
  --train-images 1000 \
  --val-images 150 \
  --seed 42
```

The preprocessing script groups frames by their video-sequence prefix and refuses to continue if a sequence appears in both training and validation.

## Data preprocessing

The preprocessing pipeline performs the following operations:

1. Parse VisDrone bounding-box annotations.
2. Map VisDrone categories 1-10 to YOLO classes 0-9.
3. Omit category 0 ignored regions from exported YOLO labels.
4. Group images by video sequence.
5. Select 1,000 training images and 150 validation images.
6. Verify that training and validation sequence overlap is zero.
7. Write the YOLO data YAML and `split_manifest.json`.

Recorded split:

- 48 training sequences
- 19 validation sequences
- zero sequence overlap

### Ignored-region limitation

Standard Ultralytics validation does not apply an additional VisDrone ignore mask after category 0 has been omitted. Predictions inside ignored regions may therefore contribute to false positives. Reported precision and recall should be interpreted with this limitation.

## Training

The main configurations are:

- `configs/yolov8n_seq.yaml`
- `configs/yolov8s_seq.yaml`

Train the seed-42 models:

```bash
python src/train.py --config configs/yolov8n_seq.yaml
python src/train.py --config configs/yolov8s_seq.yaml
```

Train the additional seeds by overriding the seed and run name:

```bash
python src/train.py --config configs/yolov8n_seq.yaml --seed 0 --name yolov8n_seq_seed0
python src/train.py --config configs/yolov8n_seq.yaml --seed 1 --name yolov8n_seq_seed1
python src/train.py --config configs/yolov8s_seq.yaml --seed 0 --name yolov8s_seq_seed0
python src/train.py --config configs/yolov8s_seq.yaml --seed 1 --name yolov8s_seq_seed1
```

The pretrained starting weights are the official Ultralytics `yolov8n.pt` and `yolov8s.pt` files:

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt

The resulting checkpoints are under `runs/train/<run_name>/weights/` and are excluded from Git.

## Evaluation

After placing the six trained checkpoints at the paths in `configs/seed_evaluation.yaml`:

```bash
python src/evaluate_seeds.py --config configs/seed_evaluation.yaml
```

This evaluates all six checkpoints with the same validation split and writes:

- `results/seed_evaluation/seed_metrics_raw.csv`
- `results/seed_evaluation/seed_metrics_summary.csv`

To evaluate one checkpoint directly:

```bash
python src/evaluate.py \
  --config configs/yolov8n_seq.yaml \
  --checkpoint runs/train/yolov8n_seq_seed42/weights/best.pt \
  --output results/yolov8n_single_eval.json
```

## Resource benchmark

The controlled benchmark uses sequence-grouped checkpoints:

```bash
python src/benchmark_resources.py --config configs/benchmark_yolov8n.yaml
python src/benchmark_resources.py --config configs/benchmark_yolov8s.yaml
```

Protocol:

- sequence-grouped validation split
- first 50 validation images
- 640x640 input
- batch size 1
- confidence 0.25 and IoU 0.50
- 15 warm-up predictions
- 10 repeated trials
- CUDA synchronization before and after each trial
- end-to-end timing boundary

Memory fields are reported separately:

- parameter memory
- model-file size
- allocated memory after model loading
- peak allocated memory
- incremental peak allocated memory
- peak reserved memory
- incremental peak reserved memory

## Reported results

### Three-seed accuracy

The values below are sample means and sample standard deviations across seeds 42, 0, and 1.

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.3440 ± 0.0055 | 0.2761 ± 0.0108 | 0.2405 ± 0.0076 | 0.1363 ± 0.0042 |
| YOLOv8s | 0.3942 ± 0.0067 | 0.3288 ± 0.0115 | 0.2915 ± 0.0031 | 0.1670 ± 0.0008 |

Detailed files:

- `results/seed_metrics_summary.csv`
- `results/seed_metrics_detailed.json`
- `results/experiment_log.csv`

### Controlled resource benchmark

| Model | Latency mean ± SD | p95 | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| YOLOv8n | 26.96 ± 1.65 ms | 29.05 ms | 35.40 MiB | 50.00 MiB |
| YOLOv8s | 27.56 ± 3.10 ms | 32.83 ms | 107.64 MiB | 118.00 MiB |

The distributions overlap, so the small latency difference does not establish a reliable speed ordering. The main resource difference is the substantially higher parameter, model and memory footprint of YOLOv8s.

Full measurements: `results/resource_benchmark.csv`.

### Historical resolution sweep

`results/resolution_sweep_historical.csv` is an exploratory inference-only sweep at 512, 640 and 768 pixels. It used earlier `runs/visdrone` checkpoints, not the sequence-grouped checkpoints used for the main three-seed comparison. Its 640-pixel values are therefore not directly comparable with the main seed results.

The generic sweep script is `src/resolution_sweep.py`. The historical checkpoint files are not distributed here.

## Checkpoints and pretrained models

This repository does not include trained checkpoints or the pretrained base weights. The base weights are downloaded by Ultralytics by name. To reproduce the reported results, train the configured runs or host the resulting checkpoints on GitHub Releases or another public storage service and update the paths in `configs/seed_evaluation.yaml`.

## Reproducibility chain

The intended chain is:

```text
public repository
→ documented environment
→ VisDrone download and sequence-grouped preprocessing
→ YAML configuration
→ YOLOv8n/YOLOv8s training
→ saved best.pt checkpoint
→ identical validation protocol
→ results CSV/JSON
→ final paper table
```

## Citation and attribution

If you use this repository, cite the project and the VisDrone dataset. See:

- `THIRD_PARTY_NOTICES.md`
- `data/README.md`
- `CITATION.cff`

## AI-use disclosure

See `AI_DISCLOSURE.md`. AI tools were used for repository organization, documentation, code debugging and language review. The author verified the experiments, measurements and reported results.

## License

The author's source code is released under the MIT License. Third-party software and datasets retain their own licenses and terms.
