"""Data validation layer: schema, missingness, duplicates, quality score, RCT sanity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def quality_report(df: pd.DataFrame, expected_columns: list[str] | None = None,
                   id_col: str | None = None) -> dict:
    """Health report. `duplicate_rows` is only meaningful as an error when an id column exists;
    without one (Hillstrom) identical rows are legitimate distinct customers."""
    missing = df.isna().mean()
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_columns": sorted(set(expected_columns or []) - set(df.columns)),
        "missing_pct": {c: round(float(v) * 100, 3) for c, v in missing.items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()) if id_col else None,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    penalties = (
        min(float(missing.mean()) * 100, 40)
        + (30 if report["missing_columns"] else 0)
        + (20 if report["duplicate_ids"] else 0)
    )
    report["quality_score"] = round(max(0.0, 100 - penalties), 1)
    return report


def treatment_outcome_summary(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    """Arm sizes/rates plus the naive (unadjusted) difference in means with a 95% CI.
    In an RCT this is the ATE estimate every learner must be sanity-checked against."""
    g = df.groupby(treatment)[outcome]
    n, mean, var = g.count(), g.mean(), g.var(ddof=1)
    if not {0, 1} <= set(n.index):
        raise ValueError(f"{treatment} must contain 0 and 1")
    ate = float(mean[1] - mean[0])
    se = float(np.sqrt(var[1] / n[1] + var[0] / n[0]))
    return {
        "n_control": int(n[0]), "n_treated": int(n[1]),
        "treated_share": round(float(n[1] / (n[0] + n[1])), 4),
        "mean_control": float(mean[0]), "mean_treated": float(mean[1]),
        "naive_ate": ate, "se": se, "ci95": [ate - 1.96 * se, ate + 1.96 * se],
    }
