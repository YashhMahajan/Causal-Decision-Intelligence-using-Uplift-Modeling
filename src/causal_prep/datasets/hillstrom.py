"""
Hillstrom / MineThatData — 3-arm e-mail marketing RCT (~64k customers).

Genuine randomized experiment: Men's e-mail / Women's e-mail / No e-mail.
Primary development dataset per ``docs/dataset_guide.md`` §3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..common import DatasetSpec

NAME = "hillstrom"

_HISTORY_SEGMENT_ORDER = [
    "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
    "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
]
_ARM_MAP = {"No E-Mail": "control", "Mens E-Mail": "mens_email",
            "Womens E-Mail": "womens_email"}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_FILES["hillstrom"])


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # surrogate unit id (Hillstrom ships without one) — traceability only
    df.insert(0, "customer_uid", np.arange(len(df), dtype=np.int64))

    # fix the well-known source typo; category membership unchanged
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})

    # canonical treatment encodings (3 arms preserved + binary any-email)
    df["treatment_arm"] = df["segment"].map(_ARM_MAP).astype("category")
    df["T"] = (df["treatment_arm"] != "control").astype("int8")
    df["T_mens"] = (df["treatment_arm"] == "mens_email").astype("int8")
    df["T_womens"] = (df["treatment_arm"] == "womens_email").astype("int8")

    # ordinal view of the redundant history bucket (EDA / grouping only)
    df["history_segment_ord"] = (
        pd.Categorical(df["history_segment"], categories=_HISTORY_SEGMENT_ORDER,
                       ordered=True).codes.astype("int8") + 1)

    # conservative, causally-safe engineered covariates
    df["history_log1p"] = np.log1p(df["history"].astype("float64"))
    # NOTE: `mens + womens` is a pure linear combination of two features already
    # in X (and every customer bought men's or women's, so it is 1 or 2) -> it
    # carries no new information and is deliberately NOT added. `bought_both` is
    # the men's x women's interaction, which a linear learner cannot form itself.
    df["bought_both"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype("int8")

    # explicit dtypes
    df["recency"] = df["recency"].astype("int16")
    df["history"] = df["history"].astype("float64")
    for c in ["mens", "womens", "newbie", "visit", "conversion"]:
        df[c] = df[c].astype("int8")
    df["spend"] = df["spend"].astype("float64")
    for c in ["zip_code", "channel"]:
        df[c] = df[c].astype("category")
    return df


SPEC = DatasetSpec(
    name=NAME,
    unit_id="customer_uid",
    unit_description="individual customer (one row; surrogate customer_uid = raw row index)",
    treatment_primary="T",
    treatment_all=["treatment_arm", "T", "T_mens", "T_womens"],
    treatment_arm_col="treatment_arm",
    arms=["control", "mens_email", "womens_email"],
    control_arm="control",
    outcomes=["visit", "conversion", "spend"],
    primary_outcome="conversion",
    x_numeric=["recency", "history", "history_log1p"],
    x_binary=["mens", "womens", "newbie", "bought_both"],
    x_categorical=["zip_code", "channel"],
    x_scale_cols=["recency", "history", "history_log1p"],
    stratify_cols=["treatment_arm", "conversion"],
    excluded_from_x={
        "visit": "post-treatment outcome (2-week window) and a mediator on e-mail->visit->purchase",
        "conversion": "post-treatment outcome — the primary label",
        "spend": "post-treatment outcome — zero-inflated; spend>0 iff conversion=1",
        "history_segment": "deterministic non-overlapping bucketing of `history` — collinear, information-losing",
        "history_segment_ord": "ordinal recoding of the above; kept for grouped diagnostics only",
        "segment": "raw treatment label — mapped to treatment_arm / T",
        "customer_uid": "surrogate identifier — zero information, would memorise rows",
    },
    invalid_value_checks={"recency": (1, 12), "history": (0.01, 1e9),
                          "spend": (0.0, 1e9)},
    notes=[
        "Genuine 3-arm RCT; randomization verified (max|SMD|~0.01, propensity AUC~0.5).",
        "No timestamp column: covariates are a pre-send 12-month snapshot, outcomes a "
        "fixed forward 2-week window -> temporal leakage structurally impossible; "
        "no temporal split applicable.",
        "7,634 rows share an identical feature vector with another row; no customer id "
        "exists to prove identity. They spread evenly across arms and carry 0 "
        "conversions -> kept as coincidental collisions (no row dropping).",
        "conversion rate ~0.9% -> NO resampling/SMOTE; stratified split on "
        "treatment_arm x conversion preserves the rare cell in both folds.",
        "history / spend right tails are genuine high-value customers -> NOT winsorized.",
        "`visit` is available as a stand-alone uplift target but is a mediator; never a feature.",
    ],
)


def feature_classification() -> pd.DataFrame:
    rows = [
        ("customer_uid", "identifier (surrogate)", "exclude from X (keep for joins)",
         "Row-index id we attach; Hillstrom has none. Zero information."),
        ("recency", "pre-treatment covariate (numeric)", "keep as-is",
         "Months since last purchase, 1-12, pre-send. Bounded, low skew."),
        ("history", "pre-treatment covariate (numeric)", "keep + derive history_log1p",
         "Historical 12-month $ spend, strictly positive, right-skewed. Not winsorized."),
        ("history_segment", "derived-redundant of history", "exclude from X (ordinal kept for EDA)",
         "Deterministic non-overlapping bucketing of `history`. Collinear."),
        ("mens", "pre-treatment covariate (binary)", "keep",
         "Bought men's merchandise in prior 12 months (not gender)."),
        ("womens", "pre-treatment covariate (binary)", "keep",
         "Bought women's merchandise in prior 12 months. Not exclusive with `mens`."),
        ("newbie", "pre-treatment covariate (binary)", "keep",
         "New customer in prior 12 months."),
        ("zip_code", "pre-treatment covariate (categorical)", "fix typo + one-hot (fit on train)",
         "Urban / Suburban / Rural. 'Surburban' corrected."),
        ("channel", "pre-treatment covariate (categorical)", "one-hot (fit on train)",
         "Prior-year purchase channel: Phone / Web / Multichannel."),
        ("history_log1p", "derived covariate (numeric)", "keep (engineered)",
         "log1p(history) — skew control for linear/propensity models; monotone."),
        ("bought_both", "derived covariate (binary)", "keep (engineered, optional)",
         "mens AND womens — the interaction a linear learner cannot form itself."),
        ("segment", "TREATMENT (raw)", "map -> treatment_arm / T / T_mens / T_womens",
         "Randomized 3-arm assignment. Native arms preserved; never collapsed."),
        ("visit", "OUTCOME — post-treatment (mediator)", "target only — EXCLUDE from X",
         "Site visit in the 2 weeks after send. Caused by the e-mail."),
        ("conversion", "OUTCOME — post-treatment (primary)", "target only — EXCLUDE from X",
         "Purchase in the 2 weeks after send. Primary uplift label."),
        ("spend", "OUTCOME — post-treatment (monetary)", "target only — EXCLUDE from X",
         "Revenue in the 2 weeks after send; zero-inflated."),
        ("history_segment_ord", "EDA helper (ordinal 1-7)", "exclude from X",
         "Redundant with `history`; used for grouped balance tables."),
    ]
    return pd.DataFrame(rows, columns=["feature", "category", "action", "reason"])


def extra_raw_audit(df_clean: pd.DataFrame) -> dict:
    dup_mask = df_clean.duplicated(
        subset=[c for c in df_clean.columns
                if c not in ("customer_uid", "treatment_arm", "T", "T_mens",
                             "T_womens", "history_segment_ord")], keep=False)
    x_key = ["recency", "history", "mens", "womens", "newbie", "zip_code",
             "channel", "history_segment"]
    arms_per_x = df_clean.groupby(x_key, observed=True)["treatment_arm"].nunique()
    hs = df_clean.groupby("history_segment", observed=True)["history"].agg(["min", "max"])
    hs = hs.reindex(_HISTORY_SEGMENT_ORDER)
    return {
        "coincidental_duplicates": {
            "rows_in_duplicate_groups": int(dup_mask.sum()),
            "conversions_among_them": int(df_clean.loc[dup_mask, "conversion"].sum()),
            "by_arm": df_clean.loc[dup_mask, "treatment_arm"].value_counts().to_dict(),
            "x_vectors_spanning_multiple_arms": int((arms_per_x > 1).sum()),
            "decision": "kept — coincidental, not data-entry errors",
        },
        "history_segment_is_deterministic_bin_of_history": bool(
            (hs["min"].values[1:] > hs["max"].values[:-1]).all()),
        "outcome_nesting": {
            "conversion1_and_spend0": int(((df_clean.conversion == 1) &
                                           (df_clean.spend <= 0)).sum()),
            "spend_pos_and_conversion0": int(((df_clean.spend > 0) &
                                              (df_clean.conversion != 1)).sum()),
            "conversion1_and_visit0": int(((df_clean.conversion == 1) &
                                           (df_clean.visit != 1)).sum()),
        },
    }
