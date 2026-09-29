"""Covariate balance (standardized mean difference). |SMD| < 0.1 is the usual 'balanced' bar."""
from __future__ import annotations

import numpy as np
import pandas as pd


def smd(df: pd.DataFrame, treatment: str, features: list[str]) -> pd.Series:
    t, c = df[df[treatment] == 1], df[df[treatment] == 0]
    out = {}
    for f in features:
        pooled = np.sqrt((t[f].var(ddof=1) + c[f].var(ddof=1)) / 2)
        out[f] = 0.0 if pooled == 0 else float((t[f].mean() - c[f].mean()) / pooled)
    return pd.Series(out, name="smd")


def imbalanced(df: pd.DataFrame, treatment: str, features: list[str], threshold: float = 0.1) -> list[str]:
    s = smd(df, treatment, features)
    return s[s.abs() > threshold].index.tolist()
