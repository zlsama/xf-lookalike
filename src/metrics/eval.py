"""Competition evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    recall_at_k: float
    precision_at_k: float
    precision_random: float
    efficiency_at_k: float
    final_score: float
    k: int
    n_candidates: int
    n_positive: int
    n_hit: int


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, int]:
    if labels.sum() == 0:
        return 0.0, 0
    order = np.argsort(-scores)
    topk = order[:k]
    hits = int(labels[topk].sum())
    return hits / float(labels.sum()), hits


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    topk = order[:k]
    return float(labels[topk].sum()) / float(k)


def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 200_000,
    target: float = 200.0,
) -> EvalResult:
    labels = labels.astype(int)
    n_candidates = len(labels)
    n_positive = int(labels.sum())
    recall, n_hit = recall_at_k(scores, labels, k)
    precision = precision_at_k(scores, labels, k)
    precision_random = n_positive / n_candidates if n_candidates else 0.0
    if precision_random <= 0:
        efficiency = 0.0
    else:
        efficiency = min(precision / (target * precision_random), 1.0)
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
