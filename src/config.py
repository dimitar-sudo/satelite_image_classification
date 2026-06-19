"""Lightweight configuration loader.

The whole project is driven by ``configs/default.yaml``. This module loads it
once and exposes the values both as a nested dict (``CONFIG``) and as a handful
of convenience constants used across the code base.
"""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

# Repository root = parent of the directory holding this file (``src/``).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")


@lru_cache(maxsize=None)
def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and cache the YAML configuration as a nested dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str) -> str:
    """Resolve a config-relative path against the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


CONFIG = load_config()

# Frequently used constants ------------------------------------------------- #
SEED: int = CONFIG["seed"]

CLASS_NAMES: list[str] = CONFIG["data"]["class_names"]
NUM_CLASSES: int = CONFIG["data"]["num_classes"]
IMAGE_SIZE: int = CONFIG["data"]["image_size"]
VAL_SPLIT: float = CONFIG["data"]["val_split"]
MEAN: list[float] = CONFIG["data"]["normalize"]["mean"]
STD: list[float] = CONFIG["data"]["normalize"]["std"]

TRAIN_DIR: str = resolve(CONFIG["data"]["train_dir"])
TEST_DIR: str = resolve(CONFIG["data"]["test_dir"])

WEIGHTS_PATH: str = resolve(CONFIG["paths"]["weights"])
HISTORY_PATH: str = resolve(CONFIG["paths"]["history"])
METRICS_PATH: str = resolve(CONFIG["paths"]["metrics"])
PLOTS_DIR: str = resolve(CONFIG["paths"]["plots_dir"])
SUBMISSION_PATH: str = resolve(CONFIG["paths"]["submission"])
