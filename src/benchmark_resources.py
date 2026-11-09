"""Benchmark end-to-end latency and memory for one sequence-grouped checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from common import load_yaml, resolve_path
from ultralytics import YOLO


def percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, _ = load_yaml(args.config)
    checkpoint = resolve_path(cfg["checkpoint"])
    data_dir = resolve_path(cfg["images_dir"])
    images = sorted(data_dir.glob("*.jpg"))[: int(cfg.get("images", 50))]
    if not images:
        raise FileNotFoundError(f"No JPG images found in {data_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    imgsz = int(cfg.get("imgsz", 640))
    warmup = int(cfg.get("warmup", 15))
    trials = int(cfg.get("trials", 10))
    confidence = float(cfg.get("confidence", 0.25))
    iou = float(cfg.get("iou", 0.5))

    torch.backends.cudnn.benchmark = True
    model = YOLO(str(checkpoint)).to("cuda")

    for image in images[:warmup]:
        model.predict(str(image), imgsz=imgsz, conf=confidence, iou=iou,
                      device=0, workers=0, verbose=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    allocated_after_load = torch.cuda.memory_allocated()
    reserved_after_load = torch.cuda.memory_reserved()

    latency_samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for image in images:
            model.predict(str(image), imgsz=imgsz, conf=confidence, iou=iou,
                          device=0, workers=0, verbose=False)
        torch.cuda.synchronize()
        latency_samples.append((time.perf_counter() - start) * 1000.0 / len(images))

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.model.parameters())
    mean_latency = statistics.mean(latency_samples)
    mib = 1024 ** 2

    result = {
        "model": cfg["name"],
        "checkpoint": str(checkpoint),
        "images_dir": str(data_dir),
        "images": len(images),
        "imgsz": imgsz,
        "batch": 1,
        "confidence": confidence,
        "iou": iou,
        "warmup": warmup,
        "trials": trials,
        "measurement_boundary": "image decoding, preprocessing, inference, NMS and postprocessing",
        "latency_ms_mean": mean_latency,
        "latency_ms_sample_std": statistics.stdev(latency_samples),
        "latency_ms_median": statistics.median(latency_samples),
        "latency_ms_p95": percentile(latency_samples, 0.95),
        "fps_mean": 1000.0 / mean_latency,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "parameter_memory_mib": parameter_bytes / mib,
        "model_file_size_mib": Path(checkpoint).stat().st_size / mib,
        "allocated_after_load_mib": allocated_after_load / mib,
        "reserved_after_load_mib": reserved_after_load / mib,
        "peak_allocated_total_mib": peak_allocated / mib,
        "peak_allocated_incremental_mib": (peak_allocated - allocated_after_load) / mib,
        "peak_reserved_total_mib": peak_reserved / mib,
        "peak_reserved_incremental_mib": (peak_reserved - reserved_after_load) / mib,
        "latency_samples_ms": latency_samples,
    }

    output = resolve_path(cfg.get("output", f"results/{cfg['name']}_benchmark.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
