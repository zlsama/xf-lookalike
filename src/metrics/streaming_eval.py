"""Streaming evaluation on full candidate pools."""

from __future__ import annotations

import heapq

import lightgbm as lgb
import numpy as np

from src.data.labels import attach_labels, build_positive_ids, build_seed_current_ids
from src.data.load import read_pool_iter
from src.data.paths import SplitPaths
from src.features.encode import apply_category_mappings
from src.features.flatten import flatten_features
from src.metrics.eval import EvalResult, evaluate_scores
from src.models.lgb_trainer import predict_scores


def streaming_evaluate(
    model: lgb.Booster,
    split: SplitPaths,
    feature_cols: list[str],
    categories: dict[str, dict[str, int]],
    k: int = 200_000,
    target: float = 200.0,
) -> EvalResult:
    """
    Score the full dev/test pool partition-by-partition.

    Uses a min-heap to track global Top-K without storing all scores.
    """
    positive_ids = build_positive_ids(split)
    seed_current_ids = build_seed_current_ids(split)

    heap: list[tuple[float, int]] = []
    n_candidates = 0
    n_positive = len(positive_ids)

    for chunk in read_pool_iter(split.pool_dir, sample_frac=1.0):
        labeled = attach_labels(chunk, positive_ids, seed_current_ids, exclude_seed=True)
        if labeled.empty:
            continue

        feat = flatten_features(labeled)
        feat_enc = apply_category_mappings(feat, categories)
        scores = predict_scores(model, feat_enc[feature_cols])

        labels = labeled["label"].to_numpy()
        mids = labeled["masked_id"].to_numpy()
        n_candidates += len(labeled)

        for mid, score, label in zip(mids.tolist(), scores.tolist(), labels.tolist()):
            if len(heap) < k:
                heapq.heappush(heap, (float(score), int(mid)))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (float(score), int(mid)))

    ranked = sorted(heap, key=lambda x: -x[0])
    top_ids = {mid for _, mid in ranked}
    n_hit = len(top_ids & positive_ids)

    recall = n_hit / n_positive if n_positive else 0.0
    precision = n_hit / k if k else 0.0
    precision_random = n_positive / n_candidates if n_candidates else 0.0
    efficiency = min(precision / (target * precision_random), 1.0) if precision_random > 0 else 0.0
    final_score = 0.5 * recall + 0.5 * efficiency

    return EvalResult(
        recall_at_k=recall,
        precision_at_k=precision,
        precision_random=precision_random,
        efficiency_at_k=efficiency,
        final_score=final_score,
        k=k,
        n_candidates=n_candidates,
        n_positive=n_positive,
        n_hit=n_hit,
    )
