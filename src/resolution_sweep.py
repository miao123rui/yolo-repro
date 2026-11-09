"""Run an inference-only resolution sweep for configured checkpoints."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import load_yaml, resolve_path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, _ = load_yaml(args.config)
    data_path = resolve_path(cfg["data"])
    output_dir = resolve_path(cfg.get("output_dir", "results/resolution_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in cfg["runs"]:
        checkpoint = resolve_path(run["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for imgsz in cfg["imgsz"]:
            print(f"Evaluating {run['name']} at imgsz={imgsz}")
            result = YOLO(str(checkpoint)).val(
                data=str(data_path),
                imgsz=int(imgsz),
                batch=int(cfg.get("batch", 1)),
                device=cfg.get("device", 0),
                workers=int(cfg.get("workers", 0)),
                plots=False,
                verbose=False,
            )
            metrics = result.results_dict
            rows.append({
                "model": run["name"],
                "checkpoint": str(checkpoint),
                "imgsz": int(imgsz),
                "precision": float(metrics["metrics/precision(B)"]),
                "recall": float(metrics["metrics/recall(B)"]),
                "mAP50": float(metrics["metrics/mAP50(B)"]),
                "mAP50-95": float(metrics["metrics/mAP50-95(B)"]),
            })

    output = output_dir / "resolution_sweep_reproduced.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", output)


if __name__ == "__main__":
    main()
