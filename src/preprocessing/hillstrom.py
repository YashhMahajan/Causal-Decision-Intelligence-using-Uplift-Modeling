"""Hillstrom / MineThatData preprocessing.

Causal roles (fixed here, used by every downstream module):
  X (pre-treatment)  recency, history, log_history, mens, womens, newbie, both_categories,
                     zip_code_*, channel_*
  T                  t_binary (any email vs none), arm (0 control / 1 mens / 2 womens)
  Y                  conversion (first), spend (business outcome), visit (secondary)
Dropped: history_segment - a binned copy of `history`, redundant.
visit / conversion / spend are POST-treatment and must never be used as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HILLSTROM_CSV
from src.preprocessing.splits import assign_splits

RAW_COLUMNS = ["recency", "history_segment", "history", "mens", "womens", "zip_code",
               "newbie", "channel", "segment", "visit", "conversion", "spend"]
ARM_MAP = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
OUTCOMES = ["conversion", "spend", "visit"]
ZIP_LEVELS = ["Rural", "Suburban", "Urban"]           # 'Surburban' typo in raw data is fixed
CHANNEL_LEVELS = ["Phone", "Web", "Multichannel"]


def load_raw(path=HILLSTROM_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (processed frame, feature column list)."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Hillstrom columns missing: {sorted(missing)}")
    df = raw.copy()
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    if not set(df["zip_code"]) <= set(ZIP_LEVELS) or not set(df["channel"]) <= set(CHANNEL_LEVELS):
        raise ValueError("Unexpected categorical level in Hillstrom data")
    if not set(df["segment"]) <= set(ARM_MAP):
        raise ValueError("Unexpected treatment arm in Hillstrom data")

    out = pd.DataFrame(index=df.index)
    out["recency"] = df["recency"]
    out["history"] = df["history"]
    out["log_history"] = np.log1p(df["history"])
    out["mens"], out["womens"], out["newbie"] = df["mens"], df["womens"], df["newbie"]
    out["both_categories"] = df["mens"] * df["womens"]
    for lvl in ZIP_LEVELS[1:]:                        # Rural = reference level
        out[f"zip_code_{lvl}"] = (df["zip_code"] == lvl).astype(int)
    for lvl in CHANNEL_LEVELS[1:]:                    # Phone = reference level
        out[f"channel_{lvl}"] = (df["channel"] == lvl).astype(int)
    features = out.columns.tolist()

    out["arm"] = df["segment"].map(ARM_MAP)
    out["t_binary"] = (out["arm"] > 0).astype(int)
    for y in OUTCOMES:
        out[y] = df[y]
    out["split"] = assign_splits(out, "t_binary", "conversion")
    return out, features
