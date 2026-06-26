"""Label construction for lookalike tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.load import read_seed_ids
from src.data.paths import SplitPaths


@dataclass
class TaskStats:
    task_id: str
    n_pool: int
    n_seed_current: int
    n_seed_next: int
    n_positive: int
    n_candidates: int
    positive_rate: float


def build_positive_ids(split: SplitPaths) -> set[int]:
    if split.seed_next_dir is None:
        raise ValueError(f"Split {split.name} has no next seed labels")
    seed_current = read_seed_ids(split.seed_current_dir)
    seed_next = read_seed_ids(split.seed_next_dir)
    return seed_next - seed_current


def build_seed_current_ids(split: SplitPaths) -> set[int]:
    return read_seed_ids(split.seed_current_dir)


def attach_labels(
    df: pd.DataFrame,
    positive_ids: set[int],
    seed_current_ids: set[int],
    exclude_seed: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["masked_id"].isin(positive_ids).astype(int)
    out["is_seed_current"] = out["masked_id"].isin(seed_current_ids)
    if exclude_seed:
        out = out.loc[~out["is_seed_current"]].reset_index(drop=True)
    return out


def sample_training_frame(
    df: pd.DataFrame,
    neg_ratio: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep all positives; downsample negatives."""
    import numpy as np

    pos = df.loc[df["label"] == 1]
    neg = df.loc[df["label"] == 0]
    if len(pos) == 0:
        raise ValueError("No positive samples in frame")
    n_neg = min(len(neg), len(pos) * neg_ratio)
    neg_sample = neg.sample(n=n_neg, random_state=seed) if n_neg < len(neg) else neg
    out = pd.concat([pos, neg_sample], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
