"""Path helpers for competition data splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitPaths:
    name: str
    task_id: str
    pool_dir: Path
    seed_current_dir: Path
    seed_next_dir: Path | None = None

    @property
    def has_labels(self) -> bool:
        return self.seed_next_dir is not None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "data"


def resolve_split(root: Path, split_cfg: dict) -> SplitPaths:
    base = data_root(root)
    seed_next = split_cfg.get("seed_next")
    return SplitPaths(
        name=split_cfg.get("name", split_cfg["task_id"]),
        task_id=split_cfg["task_id"],
        pool_dir=base / split_cfg["pool"],
        seed_current_dir=base / split_cfg["seed_current"],
        seed_next_dir=base / seed_next if seed_next else None,
    )
