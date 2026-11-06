from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import read_json


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config.json"
DATASET_NAMES = ("Amazon", "Google", "Yelp")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("method") != "TopoPoison":
        raise ValueError(f"Unexpected method in {path}")
    missing = [name for name in DATASET_NAMES if name not in config.get("datasets", {})]
    if missing:
        raise ValueError(f"Missing dataset configuration: {', '.join(missing)}")
    return config


def select_datasets(config: dict[str, Any], names: Iterable[str] | None) -> list[str]:
    selected = list(names) if names else list(DATASET_NAMES)
    unknown = [name for name in selected if name not in config["datasets"]]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return selected


def package_path(value: str) -> Path:
    path = (PACKAGE_ROOT / value).resolve()
    if path != PACKAGE_ROOT and PACKAGE_ROOT not in path.parents:
        raise ValueError(f"Path leaves the package root: {value}")
    return path
