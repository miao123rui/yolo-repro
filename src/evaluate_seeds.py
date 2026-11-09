"""Evaluate all seed checkpoints and calculate mean and sample standard deviation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict

import torch
from common import load_yaml, resolve_path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/seed_evaluation")
    args = parser.parse_args()

    cfg, _ = load_yaml(args.config)
    data_path = resolve_path(cfg["data"])
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    grouped = defaultdict(list)
    for run in cfg["runs"]:
        checkpoint = resolve_path(run["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"Evaluating {run['model']} seed={run['seed']}: {checkpoint}")
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
            "model": run["model"],
            "seed": int(run["seed"]),
            "checkpoint": str(checkpoint),
            "precision": float(metrics["metrics/precision(B)"]),
            "recall": float(metrics["metrics/recall(B)"]),
            "mAP50": float(metrics["metrics/mAP50(B)"]),
            "mAP50-95": float(metrics["metrics/mAP50-95(B)"]),
        }
        raw_rows.append(row)
        grouped[run["model"]].append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_path = output_dir / "seed_metrics_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for model_name, rows in grouped.items():
        for metric in ["precision", "recall", "mAP50", "mAP50-95"]:
            values = [row[metric] for row in rows]
            summary_rows.append({
                "model": model_name,
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            })

    summary_csv = output_dir / "seed_metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "seed_metrics_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
