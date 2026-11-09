"""Evaluate one checkpoint and save its metrics as JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import load_yaml, resolve_path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg, _ = load_yaml(args.config)
    data_path = resolve_path(cfg["data"])
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    result = YOLO(str(checkpoint)).val(
        data=str(data_path),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("eval_batch", cfg.get("batch", 4))),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 0)),
        plots=False,
        verbose=False,
    )

    metrics = result.results_dict
    row = {
        "name": args.name or checkpoint.stem,
        "checkpoint": str(checkpoint),
        "data": str(data_path),
        "precision": float(metrics["metrics/precision(B)"]),
        "recall": float(metrics["metrics/recall(B)"]),
        "mAP50": float(metrics["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics["metrics/mAP50-95(B)"]),
    }

    output = resolve_path(args.output or "results/evaluate_one.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
