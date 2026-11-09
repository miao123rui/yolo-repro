"""Train YOLOv8n or YOLOv8s from a YAML configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_yaml, resolve_path, set_seed
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Override the dataset YAML path")
    parser.add_argument("--weights", help="Override the pretrained model, e.g. yolov8n.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    cfg, config_path = load_yaml(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    data_path = resolve_path(args.data or cfg["data"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run src/prepare_data.py first."
        )

    weights = args.weights or cfg.get("model", "yolov8n.pt")
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 100))
    batch = args.batch if args.batch is not None else int(cfg.get("batch", 4))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 640))
    device = args.device if args.device is not None else cfg.get("device", 0)
    project = resolve_path(args.project or cfg.get("project", "runs/train"))
    name = args.name or cfg.get("name", Path(config_path).stem)

    print(f"Config: {config_path}")
    print(f"Dataset: {data_path}")
    print(f"Model: {weights}; epochs={epochs}; imgsz={imgsz}; batch={batch}; seed={seed}")

    model = YOLO(weights)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        device=device,
        workers=int(cfg.get("workers", 0)),
        amp=bool(cfg.get("amp", True)),
        pretrained=bool(cfg.get("pretrained", True)),
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
