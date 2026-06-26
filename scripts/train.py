"""Train LightGBM baseline on train split, validate on dev split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.labels import attach_labels, build_positive_ids, build_seed_current_ids, sample_training_frame
from src.data.load import read_pool_iter
from src.data.paths import project_root
from src.features.flatten import encode_categories, flatten_features
from src.metrics.eval import evaluate_scores
from src.models.lgb_trainer import predict_scores, train_lgb_classifier
from src.utils.config import get_splits, load_config


def build_labeled_frame(split, sample_frac: float) -> pd.DataFrame:
    positive_ids = build_positive_ids(split)
    seed_current_ids = build_seed_current_ids(split)
    chunks = []
    for chunk in read_pool_iter(split.pool_dir, sample_frac=sample_frac):
        labeled = attach_labels(chunk, positive_ids, seed_current_ids, exclude_seed=True)
        chunks.append(labeled)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    cfg = load_config()
    splits = get_splits(cfg, project_root())
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    sample_frac = float(train_cfg.get("sample_frac", 0.05))
    neg_ratio = int(train_cfg.get("neg_ratio", 20))

    print(f"Loading train split (sample_frac={sample_frac})...")
    train_raw = build_labeled_frame(splits["train"], sample_frac=sample_frac)
    print(f"  raw train rows after exclude seed: {len(train_raw):,}, positives: {train_raw['label'].sum():,}")

    train_sampled = sample_training_frame(train_raw, neg_ratio=neg_ratio)
    print(f"  sampled train rows: {len(train_sampled):,}, positives: {train_sampled['label'].sum():,}")

    print(f"Loading dev split (sample_frac={sample_frac})...")
    dev_raw = build_labeled_frame(splits["dev"], sample_frac=sample_frac)
    print(f"  raw dev rows: {len(dev_raw):,}, positives: {dev_raw['label'].sum():,}")

    train_feat = flatten_features(train_sampled)
    dev_feat = flatten_features(dev_raw)

    train_enc, [dev_enc], feature_cols, categories = encode_categories(train_feat, [dev_feat])
    x_train = train_enc[feature_cols]
    y_train = train_sampled["label"]
    x_dev = dev_enc[feature_cols]
    y_dev = dev_raw["label"]

    print("Training LightGBM...")
    model = train_lgb_classifier(
        x_train,
        y_train,
        x_dev,
        y_dev,
        params=train_cfg.get("lgb", {}),
    )

    model_dir = project_root() / cfg["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lgb_baseline.txt"
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    dev_scores = predict_scores(model, x_dev)
    result = evaluate_scores(
        dev_scores,
        y_dev.to_numpy(),
        k=int(eval_cfg["k"]),
        target=float(eval_cfg["target"]),
    )
    print("\n=== Dev Evaluation ===")
    print(f"Recall@{result.k:,}: {result.recall_at_k:.6f}")
    print(f"Precision@{result.k:,}: {result.precision_at_k:.6f}")
    print(f"Efficiency@{result.k:,}: {result.efficiency_at_k:.6f}")
    print(f"FinalScore: {result.final_score:.6f}")
    print(f"Hits: {result.n_hit}/{result.n_positive} positives in top-{result.k:,}")

    meta = {
        "sample_frac": sample_frac,
        "neg_ratio": neg_ratio,
        "feature_cols": feature_cols,
        "categories": categories,
        "dev_eval": result.__dict__,
    }
    with open(model_dir / "lgb_baseline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
