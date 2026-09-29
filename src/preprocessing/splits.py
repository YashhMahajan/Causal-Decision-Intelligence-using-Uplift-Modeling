"""Train/val/test split stratified on treatment x outcome so every split keeps the arm ratio
and outcome rate in each arm (the leakage/imbalance guard required by the design doc, section 8.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED, SPLIT_FRACTIONS


def assign_splits(df: pd.DataFrame, treatment: str, outcome: str,
                  fractions: dict[str, float] = SPLIT_FRACTIONS, seed: int = SEED) -> pd.Series:
    strata = df[treatment].astype(str) + "_" + (df[outcome] > 0).astype(int).astype(str)
    idx = np.arange(len(df))
    hold = fractions["val"] + fractions["test"]
    train_idx, rest_idx = train_test_split(idx, test_size=hold, stratify=strata, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=fractions["test"] / hold, stratify=strata.iloc[rest_idx], random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "val", "test"
    return pd.Series(split, index=df.index, name="split")
