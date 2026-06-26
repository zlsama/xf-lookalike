"""Config loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.data.paths import project_root, resolve_split


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_splits(cfg: dict, root: Path | None = None):
    root = root or project_root()
    data_cfg = cfg["data"]
    return {
        "train": resolve_split(root, {**data_cfg["splits"]["train"], "name": "train"}),
        "dev": resolve_split(root, {**data_cfg["splits"]["dev"], "name": "dev"}),
        "test": resolve_split(root, {**data_cfg["splits"]["test"], "name": "test"}),
    }
