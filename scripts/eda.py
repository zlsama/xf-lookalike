"""EDA script: inspect data scale, schema, and label statistics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import build_positive_ids, build_seed_current_ids
from src.data.load import count_pool_rows, read_pool
from src.data.paths import project_root
from src.utils.config import get_splits, load_config


def main() -> None:
    cfg = load_config()
    splits = get_splits(cfg, project_root())

    print("=== Lookalike-AC EDA ===\n")
    for name, split in splits.items():
        print(f"[{name}] task_id={split.task_id}")
        n_pool = count_pool_rows(split.pool_dir)
        seed_cur = build_seed_current_ids(split)
        print(f"  pool rows: {n_pool:,}")
        print(f"  seed_current: {len(seed_cur):,}")
        if split.has_labels:
            pos = build_positive_ids(split)
            print(f"  seed_next(new positives): {len(pos):,}")
            print(f"  positive rate (vs pool): {len(pos)/n_pool:.6%}")
        print()

    # sample one partition schema
    sample = read_pool(splits["train"].pool_dir, max_rows=3)
    print("Sample columns:", list(sample.columns))
    print(sample.head(2).T)


if __name__ == "__main__":
    main()
