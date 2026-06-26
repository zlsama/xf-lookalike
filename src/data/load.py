"""Parquet loading for pool and seed tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def list_parquet_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return files


def read_seed_ids(seed_dir: Path) -> set[int]:
    files = list_parquet_files(seed_dir)
    ids: set[int] = set()
    for path in files:
        table = pq.read_table(path, columns=["masked_id"])
        ids.update(table.column("masked_id").to_pylist())
    return ids


def read_pool_iter(
    pool_dir: Path,
    columns: list[str] | None = None,
    sample_frac: float = 1.0,
    seed: int = 42,
):
    """Yield pool DataFrames partition by partition."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for path in list_parquet_files(pool_dir):
        table = pq.read_table(path, columns=columns)
        df = table.to_pandas()
        if sample_frac < 1.0:
            mask = rng.random(len(df)) < sample_frac
            df = df.loc[mask].reset_index(drop=True)
        if len(df):
            yield df


def read_pool(
    pool_dir: Path,
    columns: list[str] | None = None,
    sample_frac: float = 1.0,
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    total = 0
    for df in read_pool_iter(pool_dir, columns=columns, sample_frac=sample_frac, seed=seed):
        chunks.append(df)
        total += len(df)
        if max_rows is not None and total >= max_rows:
            break
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    if max_rows is not None:
        out = out.iloc[:max_rows].reset_index(drop=True)
    return out


def count_pool_rows(pool_dir: Path) -> int:
    total = 0
    for path in list_parquet_files(pool_dir):
        meta = pq.read_metadata(path)
        total += meta.num_rows
    return total
