"""Flatten nested map/list columns into tabular features for tree models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _map_sum(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, dict):
        return float(sum(value.values()))
    if isinstance(value, list):
        return float(sum(v for _, v in value))
    return 0.0


def _map_len(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, dict):
        return float(len(value))
    if isinstance(value, list):
        return float(len(value))
    return 0.0


def _map_max(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, dict):
        return float(max(value.values())) if value else 0.0
    if isinstance(value, list):
        return float(max(v for _, v in value)) if value else 0.0
    return 0.0


def _nested_map_sum(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    total = 0.0
    if isinstance(value, dict):
        for inner in value.values():
            total += _map_sum(inner)
        return total
    if isinstance(value, list):
        for _, inner in value:
            total += _map_sum(inner)
        return total
    return 0.0


def _tags_len(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(len(value))
    return 0.0


MAP_COLS = [
    "adunit_req_map",
    "adunit_imp_map",
    "plat_rsp_7d",
]
NESTED_MAP_COLS = [
    "adunit_req_series",
    "adunit_imp_series",
    "avg_bidprice",
]
CAT_COLS = ["make", "model", "province", "city"]


def flatten_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"masked_id": df["masked_id"].values})
    for col in CAT_COLS:
        if col in df.columns:
            out[col] = df[col].fillna("__MISSING__").astype(str)
    if "tags" in df.columns:
        out["tags_len"] = df["tags"].map(_tags_len)
    for col in MAP_COLS:
        if col in df.columns:
            out[f"{col}_sum"] = df[col].map(_map_sum)
            out[f"{col}_len"] = df[col].map(_map_len)
            out[f"{col}_max"] = df[col].map(_map_max)
    for col in NESTED_MAP_COLS:
        if col in df.columns:
            out[f"{col}_sum"] = df[col].map(_nested_map_sum)
            out[f"{col}_len"] = df[col].map(_map_len)
    if "no_rsp" in df.columns:
        out["no_rsp"] = df["no_rsp"].fillna(0).astype(int)
    return out


def encode_categories(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    cat_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str], dict[str, dict[str, int]]]:
    cat_cols = cat_cols or [c for c in CAT_COLS if c in train_df.columns]
    categories: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        values = pd.concat([train_df[col], *[d[col] for d in other_dfs if col in d.columns]], ignore_index=True)
        uniq = sorted(values.astype(str).unique().tolist())
        categories[col] = {v: i for i, v in enumerate(uniq)}

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in cat_cols:
            out[col] = out[col].astype(str).map(categories[col]).fillna(-1).astype(int)
        return out

    encoded_train = _transform(train_df)
    encoded_others = [_transform(d) for d in other_dfs]
    feature_cols = [c for c in encoded_train.columns if c != "masked_id"]
    return encoded_train, encoded_others, feature_cols, categories
