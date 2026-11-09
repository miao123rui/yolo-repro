# Dataset Preparation

## Source

The experiments use the VisDrone-DET benchmark:

- Official repository: https://github.com/VisDrone/VisDrone-Dataset
- The dataset is not redistributed in this repository.

Download the training and validation detection archives from the official
source. Do not use hidden test labels.

## Expected raw directory

After extracting, arrange the files as:

```text
data/raw/VisDrone/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── train/
    └── val/
```

Image filenames should retain the VisDrone sequence prefix, for example:

```text
0000006_00159_d_0000007.jpg
```

The sequence ID is the prefix before the first underscore (`0000006`).

## Build the sequence-grouped subset

From the repository root:

```bash
python src/prepare_data.py \
  --visdrone-root data/raw/VisDrone \
  --output-root data/VisDrone_subset_seq_v2 \
  --train-images 1000 \
  --val-images 150 \
  --seed 42
```

The script:

1. Reads VisDrone bounding-box annotations.
2. Converts categories 1-10 to YOLO classes 0-9.
3. Omits category 0 ignored regions from exported YOLO labels.
4. Groups frames by video sequence.
5. Selects 1,000 training images and 150 validation images.
6. Verifies that training and validation sequence IDs do not overlap.
7. Writes `split_manifest.json` and the YOLO dataset YAML.

Expected result:

```text
48 training sequences and 19 validation sequences
Sequence overlap: 0
```

The exact recorded split is in `data/split_manifest_recorded.json`.

## Ignored regions

Category 0 regions are omitted from the exported labels. Standard Ultralytics
validation does not apply an additional ignore mask. Predictions inside those
regions can therefore contribute to false positives. This limitation should be
stated when interpreting precision and recall.
