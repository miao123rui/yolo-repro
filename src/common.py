"""Shared utilities for the UAV YOLO reproducibility repository."""
from pathlib import Path
import random

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data, path


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_named_paths(values):
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing name in: {value}")
        parsed.append((name, resolve_path(path)))
    return parsed
