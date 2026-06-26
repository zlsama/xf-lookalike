"""Apply saved category mappings during inference."""

from __future__ import annotations

import pandas as pd

from src.features.flatten import CAT_COLS


def apply_category_mappings(
    df: pd.DataFrame,
    categories: dict[str, dict[str, int]],
) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in categories.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)
    return out
