"""Streaming dataset construction for full-scale training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.labels import attach_labels, build_positive_ids, build_seed_current_ids
from src.data.load import count_pool_rows, read_pool_iter
from src.data.paths import SplitPaths


def _estimate_neg_count(split: SplitPaths) -> int:
    seed_current = build_seed_current_ids(split)
    n_pool = count_pool_rows(split.pool_dir)
    return max(n_pool - len(seed_current), 1)


def build_training_sample(
    split: SplitPaths,
    neg_ratio: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stream the full candidate pool without loading everything into memory.

    - Keep all positive samples.
    - Vectorized per-chunk negative sampling to target len(positive) * neg_ratio.
    """
    positive_ids = build_positive_ids(split)
    seed_current_ids = build_seed_current_ids(split)
    k_neg = len(positive_ids) * neg_ratio

    rng = np.random.default_rng(seed)
    pos_parts: list[pd.DataFrame] = []
    neg_parts: list[pd.DataFrame] = []

    n_neg_est = _estimate_neg_count(split)
    # Slightly oversample per chunk, trim at the end.
    frac = min(1.0, (k_neg * 1.1) / n_neg_est)

    for chunk in read_pool_iter(split.pool_dir, sample_frac=1.0):
        labeled = attach_labels(chunk, positive_ids, seed_current_ids, exclude_seed=True)
        pos = labeled.loc[labeled["label"] == 1]
        neg = labeled.loc[labeled["label"] == 0]
        if len(pos):
            pos_parts.append(pos)
        if len(neg):
            mask = rng.random(len(neg)) < frac
            sampled = neg.loc[mask]
            if len(sampled):
                neg_parts.append(sampled)

    if not pos_parts:
        raise ValueError(f"No positive samples found for split {split.task_id}")

    pos_df = pd.concat(pos_parts, ignore_index=True)
    neg_df = pd.concat(neg_parts, ignore_index=True) if neg_parts else pd.DataFrame()

    if len(neg_df) > k_neg:
        neg_df = neg_df.sample(n=k_neg, random_state=seed).reset_index(drop=True)
    elif len(neg_df) < k_neg and len(neg_df) > 0:
        # rare: boost with replacement if under-sampled
        extra = neg_df.sample(n=k_neg - len(neg_df), replace=True, random_state=seed + 1)
        neg_df = pd.concat([neg_df, extra], ignore_index=True)

    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_validation_sample(
    split: SplitPaths,
    neg_ratio: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Smaller dev sample for LightGBM early stopping."""
    return build_training_sample(split, neg_ratio=neg_ratio, seed=seed + 1)
