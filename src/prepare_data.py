"""Convert VisDrone annotations and create a leakage-controlled subset.

Expected raw layout:
  VISDRONE_ROOT/images/train/*.jpg
  VISDRONE_ROOT/images/val/*.jpg
  VISDRONE_ROOT/annotations/train/*.txt
  VISDRONE_ROOT/annotations/val/*.txt

VisDrone annotations use:
  left, top, width, height, score, category, truncation, occlusion
Category 0 marks ignored regions and is omitted from exported YOLO labels.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def sequence_id(path: Path) -> str:
    return path.stem.split("_")[0]


def group_by_sequence(image_dir: Path):
    groups = defaultdict(list)
    for path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        groups[sequence_id(path)].append(path)
    return groups


def select_sequences(groups, count, seed, exclude=None):
    exclude = set(exclude or [])
    sequences = [s for s in groups if s not in exclude]
    random.Random(seed).shuffle(sequences)
    selected = []
    for sequence in sequences:
        selected.extend(groups[sequence])
        if len(selected) >= count:
            break
    return selected[:count]


def convert_annotation(annotation_path: Path, width: int, height: int) -> list[str]:
    if not annotation_path.exists():
        return []
    rows = []
    for line in annotation_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            parts = line.split()
        if len(parts) < 6:
            continue
        x, y, w, h = map(float, parts[:4])
        category = int(float(parts[5]))
        if category == 0 or not 1 <= category <= 10:
            continue
        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + w))
        y2 = max(0.0, min(float(height), y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        rows.append(f"{category - 1} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    return rows


def write_split(images, raw_root: Path, output_root: Path, split: str):
    image_out = output_root / "images" / split
    label_out = output_root / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_annotations = 0
    for image_path in images:
        annotation = raw_root / "annotations" / split / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        rows = convert_annotation(annotation, width, height)
        if not annotation.exists():
            missing_annotations += 1
        shutil.copy2(image_path, image_out / image_path.name)
        (label_out / f"{image_path.stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        written += 1
    return written, missing_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=1000)
    parser.add_argument("--val-images", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = args.visdrone_root.resolve()
    output_root = args.output_root.resolve()

    required = [
        raw_root / "images" / "train",
        raw_root / "images" / "val",
        raw_root / "annotations" / "train",
        raw_root / "annotations" / "val",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_groups = group_by_sequence(raw_root / "images" / "train")
    val_groups = group_by_sequence(raw_root / "images" / "val")

    train_images = select_sequences(train_groups, args.train_images, args.seed)
    train_sequences = {sequence_id(path) for path in train_images}
    val_images = select_sequences(
        val_groups,
        args.val_images,
        args.seed + 1,
        exclude=train_sequences,
    )
    val_sequences = {sequence_id(path) for path in val_images}

    overlap = sorted(train_sequences & val_sequences)
    if overlap:
        raise RuntimeError("Sequence leakage detected: " + ", ".join(overlap))

    train_written, train_missing = write_split(train_images, raw_root, output_root, "train")
    val_written, val_missing = write_split(val_images, raw_root, output_root, "val")

    manifest = {
        "seed": args.seed,
        "train_images": train_written,
        "val_images": val_written,
        "train_sequences": sorted(train_sequences),
        "val_sequences": sorted(val_sequences),
        "overlap_sequences": overlap,
        "missing_annotation_files": {
            "train": train_missing,
            "val": val_missing,
        },
        "ignored_regions": "VisDrone category 0 omitted from exported YOLO labels",
        "size_bins_pixels": {
            "small": "<= 32^2",
            "medium": "<= 96^2",
            "large": "> 96^2",
        },
    }
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    data_yaml = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"names:\n{names_yaml}\n"
    )
    (output_root / "VisDrone_subset_seq_v2.yaml").write_text(data_yaml, encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Dataset YAML:", output_root / "VisDrone_subset_seq_v2.yaml")


if __name__ == "__main__":
    main()
