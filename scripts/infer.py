"""Generate submission JSON for test split (streaming, Top-K only)."""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb

from src.data.labels import build_seed_current_ids
from src.data.load import read_pool_iter
from src.data.paths import project_root
from src.features.encode import apply_category_mappings
from src.features.flatten import flatten_features
from src.models.lgb_trainer import predict_scores
from src.utils.config import get_splits, load_config


def main() -> None:
    cfg = load_config()
    splits = get_splits(cfg, project_root())
    test_split = splits["test"]
    model_dir = project_root() / cfg["output"]["model_dir"]
    model_path = model_dir / "lgb_full.txt"
    meta_path = model_dir / "lgb_full_meta.json"

    if not model_path.exists():
        # fallback to baseline model name
        model_path = model_dir / "lgb_baseline.txt"
        meta_path = model_dir / "lgb_baseline_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found under {model_dir}. Run scripts/train.py first.")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    categories = meta["categories"]

    # Keep Top-K by score. 评测最大 K=200W，超出部分永远不会命中。
    top_k = int(cfg.get("submission", {}).get("top_k", 2_000_000))

    model = lgb.Booster(model_file=str(model_path))
    seed_current = build_seed_current_ids(test_split)

    # min-heap of (score, masked_id); keep largest top_k
    heap: list[tuple[float, int]] = []
    total_scored = 0

    print(f"Inferencing test split task_id={test_split.task_id}, top_k={top_k:,} ...")
    for i, chunk in enumerate(read_pool_iter(test_split.pool_dir)):
        chunk = chunk.loc[~chunk["masked_id"].isin(seed_current)].reset_index(drop=True)
        if chunk.empty:
            continue
        feat = flatten_features(chunk)
        feat_enc = apply_category_mappings(feat, categories)
        scores = predict_scores(model, feat_enc[feature_cols])

        mids = chunk["masked_id"].to_numpy()
        for mid, score in zip(mids.tolist(), scores.tolist()):
            if len(heap) < top_k:
                heapq.heappush(heap, (float(score), int(mid)))
            else:
                # replace smallest if current is larger
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (float(score), int(mid)))
        total_scored += len(chunk)
        print(f"  partition {i+1}: scored {len(chunk):,} users, total scored {total_scored:,}, heap size {len(heap):,}")

    # sort descending by score
    ranked = sorted(heap, key=lambda x: -x[0])
    records = [
        {
            "task_id": test_split.task_id,
            "masked_id": mid,
            "score": score,
        }
        for score, mid in ranked
    ]

    out_dir = project_root() / cfg["output"]["submission_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"submit_{test_split.task_id}_top{top_k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Submission saved: {out_path} ({len(records):,} rows, top_k={top_k:,})")


if __name__ == "__main__":
    main()
