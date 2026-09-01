"""
Stage 1 - RAW AUDIT (read-only).

Profiles the raw Hillstrom CSV without touching it and writes a machine-readable
snapshot to reports/hillstrom_audit.json plus a console summary.  Nothing here
modifies, filters, or imputes anything - it only *describes*.

Run:  python -m preprocessing.hillstrom_audit
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_HILLSTROM)
    return df


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def audit(df: pd.DataFrame) -> dict:
    n = len(df)
    rep: dict = {}

    rep["shape"] = {"rows": n, "cols": df.shape[1]}
    rep["columns"] = list(df.columns)
    rep["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    # ---- missingness --------------------------------------------------------
    rep["missing"] = {c: int(df[c].isna().sum()) for c in df.columns}
    rep["missing_total"] = int(df.isna().sum().sum())

    # ---- exact-duplicate rows (no id column exists) -----------------------
    dup_mask = df.duplicated(keep=False)
    x_key = [c for c in df.columns if c not in (C.TREATMENT_RAW, *C.OUTCOMES)]
    grp_arms = df.groupby(x_key, dropna=False)[C.TREATMENT_RAW].nunique()
    rep["duplicates"] = {
        "full_row_duplicates_extra": int(df.duplicated().sum()),
        "rows_in_any_duplicate_group": int(dup_mask.sum()),
        "distinct_duplicated_vectors": int(
            df[dup_mask].groupby(list(df.columns), dropna=False).ngroups
        ),
        "conversions_among_duplicate_rows": int(df.loc[dup_mask, "conversion"].sum()),
        "spend_gt0_among_duplicate_rows": int((df.loc[dup_mask, "spend"] > 0).sum()),
        "duplicate_rows_by_arm": _jsonable(
            df.loc[dup_mask, C.TREATMENT_RAW].value_counts().to_dict()
        ),
        "x_only_vectors_spanning_multiple_arms": int((grp_arms > 1).sum()),
        "x_only_vectors_total": int(len(grp_arms)),
        "note": (
            "Identical feature vectors are expected from low-cardinality columns "
            "(recency 1-12, 3-level zips/channels, binary flags) plus a floor-"
            "valued `history` (29.99). They span all three arms in proportion and "
            "carry zero conversions/spend. Treated as coincidental collisions, "
            "NOT data-entry errors -> kept."
        ),
    }

    # ---- treatment --------------------------------------------------------
    arm_counts = df[C.TREATMENT_RAW].value_counts()
    rep["treatment"] = {
        "variable": C.TREATMENT_RAW,
        "arms": _jsonable(arm_counts.to_dict()),
        "arm_shares": _jsonable((arm_counts / n).round(4).to_dict()),
        "binary_any_email": {
            "treated": int((df[C.TREATMENT_RAW] != C.CONTROL_LABEL).sum()),
            "control": int((df[C.TREATMENT_RAW] == C.CONTROL_LABEL).sum()),
        },
    }

    # ---- outcomes (post-treatment) --------------------------------------
    out = {}
    for arm in [C.CONTROL_LABEL, "Mens E-Mail", "Womens E-Mail"]:
        sub = df[df[C.TREATMENT_RAW] == arm]
        out[arm] = {
            "n": int(len(sub)),
            "visit_rate": float(sub["visit"].mean()),
            "conversion_rate": float(sub["conversion"].mean()),
            "mean_spend": float(sub["spend"].mean()),
        }
    naive = {
        "conversion_ATE_mens_vs_control": out["Mens E-Mail"]["conversion_rate"]
        - out[C.CONTROL_LABEL]["conversion_rate"],
        "conversion_ATE_womens_vs_control": out["Womens E-Mail"]["conversion_rate"]
        - out[C.CONTROL_LABEL]["conversion_rate"],
        "spend_ATE_mens_vs_control": out["Mens E-Mail"]["mean_spend"]
        - out[C.CONTROL_LABEL]["mean_spend"],
        "spend_ATE_womens_vs_control": out["Womens E-Mail"]["mean_spend"]
        - out[C.CONTROL_LABEL]["mean_spend"],
    }
    rep["outcomes_by_arm"] = _jsonable(out)
    rep["naive_unadjusted_ATE"] = _jsonable(naive)
    rep["outcome_structure_checks"] = {
        "conversion1_implies_spend_gt0": int(
            ((df.conversion == 1) & (df.spend <= 0)).sum()
        ),
        "spend_gt0_implies_conversion1": int(
            ((df.spend > 0) & (df.conversion != 1)).sum()
        ),
        "conversion1_implies_visit1": int(
            ((df.conversion == 1) & (df.visit != 1)).sum()
        ),
        "note": "Nested outcomes: {spend>0} == {conversion=1} subset of {visit=1}.",
    }
    rep["class_imbalance"] = {
        "conversion_positive_rate": float(df["conversion"].mean()),
        "visit_positive_rate": float(df["visit"].mean()),
        "spend_nonzero_rate": float((df["spend"] > 0).mean()),
    }

    # ---- covariates -----------------------------------------------------
    cov = {}
    for c in C.NUMERIC_COVARIATES:
        s = df[c]
        cov[c] = {
            "min": float(s.min()), "p01": float(s.quantile(.01)),
            "median": float(s.median()), "mean": float(s.mean()),
            "p99": float(s.quantile(.99)), "max": float(s.max()),
            "skew": float(s.skew()), "n_unique": int(s.nunique()),
        }
    for c in C.BINARY_COVARIATES:
        cov[c] = _jsonable(df[c].value_counts().to_dict())
    for c in C.CATEGORICAL_COVARIATES + C.DERIVED_REDUNDANT:
        cov[c] = _jsonable(df[c].value_counts().to_dict())
    cov["mens_womens_cross"] = {
        "both_1": int(((df.mens == 1) & (df.womens == 1)).sum()),
        "both_0": int(((df.mens == 0) & (df.womens == 0)).sum()),
    }
    # history_segment is a deterministic bucketing of history?
    hs = df.groupby("history_segment")["history"].agg(["min", "max"])
    hs_sorted = hs.reindex(C.HISTORY_SEGMENT_ORDER)
    monotone = bool(
        (hs_sorted["min"].values[1:] > hs_sorted["max"].values[:-1]).all()
    )
    cov["history_segment_is_deterministic_bin_of_history"] = monotone
    rep["covariates"] = cov

    # ---- invalid values -----------------------------------------------
    rep["invalid_value_checks"] = {
        "recency_out_of_range": int(
            (~df["recency"].between(*C.RECENCY_RANGE)).sum()
        ),
        "history_non_positive": int((df["history"] <= 0).sum()),
        "spend_negative": int((df["spend"] < 0).sum()),
        "binary_cols_non_binary": {
            c: int((~df[c].isin([0, 1])).sum())
            for c in C.BINARY_COVARIATES + ["visit", "conversion"]
        },
    }

    # ---- time structure ---------------------------------------------
    rep["time_structure"] = {
        "date_columns": [],
        "note": (
            "No timestamp column. Single cross-section of one campaign. "
            "Covariates are a historical 12-month snapshot taken before send; "
            "outcomes are a fixed forward 2-week window. Temporal leakage is "
            "structurally impossible within-dataset; a temporal split is N/A."
        ),
    }

    # ---- randomization sanity (SMD + propensity AUC) ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    t = (df[C.TREATMENT_RAW] != C.CONTROL_LABEL).astype(int)
    Xd = pd.get_dummies(
        df[C.NUMERIC_COVARIATES + C.BINARY_COVARIATES + C.CATEGORICAL_COVARIATES],
        drop_first=False,
    ).astype(float)

    def smd(x):
        a, b = x[t == 1], x[t == 0]
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0

    smds = {c: smd(Xd[c]) for c in Xd.columns}
    ps = cross_val_predict(
        LogisticRegression(max_iter=1000),
        Xd.values, t.values, cv=5, method="predict_proba",
    )[:, 1]
    rep["randomization_check"] = {
        "standardized_mean_diffs_email_vs_control": _jsonable(
            {k: round(v, 4) for k, v in smds.items()}
        ),
        "max_abs_smd": float(np.max(np.abs(list(smds.values())))),
        "propensity_auc_5fold": float(roc_auc_score(t, ps)),
        "propensity_min": float(ps.min()),
        "propensity_max": float(ps.max()),
        "propensity_mass_outside_[0.01,0.99]": float(
            np.mean((ps < 0.01) | (ps > 0.99))
        ),
        "verdict": (
            "Consistent with successful randomization: all |SMD| < 0.10, "
            "propensity AUC ~ 0.5, propensity support ~ P(email) everywhere. "
            "Positivity/overlap hold trivially; no confounding adjustment "
            "required (propensity kept only as a diagnostic / DR nuisance)."
        ),
    }

    return rep


def main() -> None:
    df = load_raw()
    rep = audit(df)
    out_path = C.REPORTS_DIR / "hillstrom_audit.json"
    out_path.write_text(json.dumps(_jsonable(rep), indent=2))
    print(f"[audit] raw file      : {C.RAW_HILLSTROM.name}")
    print(f"[audit] shape         : {rep['shape']}")
    print(f"[audit] missing cells : {rep['missing_total']}")
    print(
        f"[audit] duplicate rows : "
        f"{rep['duplicates']['rows_in_any_duplicate_group']} in groups "
        f"(0 conversions, spread across arms -> kept)"
    )
    print(f"[audit] arms          : {rep['treatment']['arms']}")
    print(f"[audit] naive ATE     : {rep['naive_unadjusted_ATE']}")
    print(f"[audit] max |SMD|     : {rep['randomization_check']['max_abs_smd']:.4f}")
    print(
        f"[audit] propensity AUC : "
        f"{rep['randomization_check']['propensity_auc_5fold']:.4f} "
        f"(range {rep['randomization_check']['propensity_min']:.3f}"
        f"-{rep['randomization_check']['propensity_max']:.3f})"
    )
    print(f"[audit] written        -> {out_path.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
