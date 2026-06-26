"""Train LightGBM with streaming full-data pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.paths import project_root
from src.data.streaming import build_training_sample, build_validation_sample
from src.features.flatten import encode_categories, flatten_features
from src.metrics.streaming_eval import streaming_evaluate
from src.models.lgb_trainer import train_lgb_classifier
from src.utils.config import get_splits, load_config


def main() -> None:
    cfg = load_config()
    splits = get_splits(cfg, project_root())
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    neg_ratio = int(train_cfg.get("neg_ratio", 20))
    use_streaming = bool(train_cfg.get("streaming", True))

    print(f"Building training sample (streaming={use_streaming}, neg_ratio={neg_ratio})...")
    train_sampled = build_training_sample(splits["train"], neg_ratio=neg_ratio)
    print(
        f"  train rows: {len(train_sampled):,}, "
        f"positives: {int(train_sampled['label'].sum()):,}"
    )

    print("Building validation sample for early stopping...")
    valid_sampled = build_validation_sample(splits["dev"], neg_ratio=neg_ratio)
    print(
        f"  valid rows: {len(valid_sampled):,}, "
        f"positives: {int(valid_sampled['label'].sum()):,}"
    )

    train_feat = flatten_features(train_sampled)
    valid_feat = flatten_features(valid_sampled)

    train_enc, [valid_enc], feature_cols, categories = encode_categories(train_feat, [valid_feat])
    x_train = train_enc[feature_cols]
    y_train = train_sampled["label"]
    x_valid = valid_enc[feature_cols]
    y_valid = valid_sampled["label"]

    print("Training LightGBM...")
    model = train_lgb_classifier(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params=train_cfg.get("lgb", {}),
    )

    model_dir = project_root() / cfg["output"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lgb_full.txt"
    model.save_model(str(model_path))
    print(f"Model saved: {model_path}")

    print("Streaming full dev evaluation...")
    result = streaming_evaluate(
        model,
        splits["dev"],
        feature_cols,
        categories,
        k=int(eval_cfg["k"]),
        target=float(eval_cfg["target"]),
    )
    print("\n=== Dev Evaluation (full pool, streaming) ===")
    print(f"Candidates: {result.n_candidates:,}, Positives: {result.n_positive:,}")
    print(f"Recall@{result.k:,}: {result.recall_at_k:.6f}")
    print(f"Precision@{result.k:,}: {result.precision_at_k:.6f}")
    print(f"Efficiency@{result.k:,}: {result.efficiency_at_k:.6f}")
    print(f"FinalScore: {result.final_score:.6f}")
    print(f"Hits: {result.n_hit}/{result.n_positive} positives in top-{result.k:,}")

    meta = {
        "version": "v1-stream-full",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "streaming": use_streaming,
        "neg_ratio": neg_ratio,
        "train_rows": len(train_sampled),
        "train_positives": int(train_sampled["label"].sum()),
        "valid_rows": len(valid_sampled),
        "valid_positives": int(valid_sampled["label"].sum()),
        "feature_cols": feature_cols,
        "categories": categories,
        "dev_eval_full": result.__dict__,
    }
    meta_path = model_dir / "lgb_full_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
