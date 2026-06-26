"""LightGBM trainer."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def train_lgb_classifier(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame | None = None,
    valid_y: pd.Series | None = None,
    params: dict | None = None,
) -> lgb.Booster:
    cfg = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 8,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    if params:
        cfg.update(params)

    n_estimators = int(cfg.pop("n_estimators", 2000))
    early_stopping = int(cfg.pop("early_stopping_rounds", 100))

    pos = max(int(train_y.sum()), 1)
    neg = max(len(train_y) - pos, 1)
    cfg["scale_pos_weight"] = neg / pos

    train_set = lgb.Dataset(train_x, label=train_y)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=100)]

    if valid_x is not None and valid_y is not None:
        valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set)
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping, verbose=True))

    model = lgb.train(
        cfg,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def predict_scores(model: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)
