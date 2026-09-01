"""
X5 RetailHero — retail loyalty-programme uplift RCT.

Provided files (``data/raw/x5/x5-retail-hero-uplift-raw-data/``):
  clients.csv       demographics + loyalty-card dates (400,162 clients)
  products.csv      product dictionary (43,038)            -> UNUSED (see notes)
  uplift_train.csv  client_id, treatment_flg, target       (200,039 — the RCT)
  uplift_test.csv   client_id only                         (200,123 — no labels)

Treatment  : `treatment_flg`  (a marketing communication / SMS)  — single binary arm
Outcome    : `target`         (purchase during the promo period) — binary
Unit       : client (loyalty-card holder)

Key causal hazard handled here
------------------------------
`first_redeem_date` runs to 2019-11-20 while the campaign / enrolment boundary
is 2019-03-15 (max `first_issue_date`).  ~11.6% of redemptions post-date the
campaign, so ANY use of raw `first_redeem_date` leaks the future.  We censor it
at ``REF_DATE`` = 2019-03-16: redemptions on/after that date are treated exactly
like "never redeemed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..common import DatasetSpec

NAME = "x5"
REF_DATE = pd.Timestamp("2019-03-16")   # day after the last first_issue_date


def _paths():
    d = C.RAW_FILES["x5_dir"]
    return {k: d / f"{k}.csv" for k in
            ["clients", "products", "uplift_train", "uplift_test"]}


def load_raw() -> pd.DataFrame:
    """Return the labelled RCT frame (uplift_train ⨝ clients).  Unlabelled
    scoring clients are attached in :func:`load_score_frame`."""
    p = _paths()
    tr = pd.read_csv(p["uplift_train"])
    cl = pd.read_csv(p["clients"], parse_dates=["first_issue_date",
                                                "first_redeem_date"])
    return tr.merge(cl, on="client_id", how="left")


def load_score_frame() -> pd.DataFrame:
    p = _paths()
    te = pd.read_csv(p["uplift_test"])
    cl = pd.read_csv(p["clients"], parse_dates=["first_issue_date",
                                                "first_redeem_date"])
    return clean(te.merge(cl, on="client_id", how="left"), _for_score=True)


def clean(df_raw: pd.DataFrame, _for_score: bool = False) -> pd.DataFrame:
    df = df_raw.copy()

    # ---- treatment / outcome canonical encodings -----------------------
    if not _for_score:
        df["treatment_flg"] = df["treatment_flg"].astype("int8")
        df["treatment_arm"] = np.where(df["treatment_flg"] == 1,
                                       "treated", "control")
        df["treatment_arm"] = df["treatment_arm"].astype("category")
        df["target"] = df["target"].astype("int8")

    # ---- age: clip implausible -> NaN, keep an invalid flag -----------
    age = pd.to_numeric(df["age"], errors="coerce")
    df["age_invalid"] = (~age.between(14, 99)).astype("int8")
    df["age"] = age.where(age.between(14, 99)).astype("float64")

    # ---- gender: F / M / unknown (NaN and 'U' both -> unknown) --------
    df["gender"] = (df["gender"].replace({"U": "unknown"})
                    .fillna("unknown").astype("category"))

    # ---- loyalty tenure: days since first_issue_date as of REF_DATE ---
    #      every first_issue_date <= REF_DATE, so this is pre-treatment.
    df["tenure_days"] = (REF_DATE - df["first_issue_date"]).dt.days.astype("float64")
    df["issue_month"] = df["first_issue_date"].dt.month.astype("float64")

    # ---- first_redeem_date: CENSOR at REF_DATE (anti-leakage) --------
    redeem = df["first_redeem_date"]
    valid_pre = redeem.notna() & (redeem <= REF_DATE) & (redeem >= df["first_issue_date"])
    df["has_redeemed_pre"] = valid_pre.astype("int8")
    dsr = (REF_DATE - redeem.where(valid_pre)).dt.days
    df["days_since_first_redeem"] = dsr.astype("float64")     # NaN -> imputed + indicator
    df["redeem_info_missing"] = (~valid_pre).astype("int8")   # never / post-REF / invalid

    # ---- unit id / dtypes ------------------------------------------
    df = df.rename(columns={"client_id": "client_uid"})
    return df


SPEC = DatasetSpec(
    name=NAME,
    unit_id="client_uid",
    unit_description="loyalty-card client (one row per client; native client_id)",
    treatment_primary="treatment_flg",
    treatment_all=["treatment_arm", "treatment_flg"],
    treatment_arm_col="treatment_arm",
    arms=["control", "treated"],
    control_arm="control",
    outcomes=["target"],
    primary_outcome="target",
    x_numeric=["age", "tenure_days", "issue_month", "days_since_first_redeem"],
    x_binary=["age_invalid", "has_redeemed_pre", "redeem_info_missing"],
    x_categorical=["gender"],
    x_scale_cols=["age", "tenure_days", "days_since_first_redeem"],
    stratify_cols=["treatment_flg", "target"],
    excluded_from_x={
        "target": "post-treatment outcome — the label",
        "first_issue_date": "raw timestamp — replaced by pre-treatment `tenure_days` / `issue_month`",
        "first_redeem_date": "raw timestamp — POST-treatment for ~11.6% of clients; replaced by "
                             "REF_DATE-censored `has_redeemed_pre` / `days_since_first_redeem`",
        "client_uid": "native identifier — not a feature",
        "treatment_flg": "treatment — kept as T, not a covariate",
    },
    invalid_value_checks={"age": (14, 99), "tenure_days": (0, 1500),
                          "issue_month": (1, 12)},
    notes=[
        "RCT: treatment_flg ~50/50 (99,981 / 100,058); randomization verified "
        "(max|SMD|<0.01 on available covariates, propensity AUC~0.50).",
        "Feature set is THIN: the raw `purchases` transaction table (X5's main value "
        "per docs §6.2) is NOT in the repo, so covariates are limited to demographics "
        "+ loyalty-card dates. products.csv is not joinable without purchases -> unused.",
        "`first_redeem_date` censored at REF_DATE=2019-03-16 to prevent post-treatment "
        "leakage (raw values extend to 2019-11; 11.6% post-date the campaign).",
        "age has implausible values (<14 or >99, incl. negatives / years like 1901): "
        "~885 in the labelled RCT frame, 1,404 across all 400k clients. Set to NaN + "
        "`age_invalid` flag + median impute on train (rows NOT dropped).",
        "target base rate ~62% (not rare) -> no resampling.",
        "Provided uplift_test.csv has NO labels: it is transformed to `score.parquet` "
        "for later submission scoring, and is NOT used as an evaluation split. The "
        "train/test split is carved from uplift_train only.",
        "No continuous / monetary outcome is available (only binary `target`).",
    ],
    numeric_add_indicator=False,
    diag_sample=None,
)


def feature_classification() -> pd.DataFrame:
    rows = [
        ("client_id", "identifier (native)", "rename -> client_uid, exclude from X",
         "Unit id. Not a feature."),
        ("treatment_flg", "TREATMENT", "keep as T (+ treatment_arm label)",
         "Randomized marketing communication. Single binary arm."),
        ("target", "OUTCOME — post-treatment (primary)", "target only — EXCLUDE from X",
         "Purchase during the promo period. Binary uplift label."),
        ("age", "pre-treatment covariate (numeric)", "clip [14,99] -> NaN + flag + impute (train)",
         "~885 out-of-range values in the RCT frame (negatives, 1852, ...). "
         "Rows kept; invalidity flagged, not dropped."),
        ("age_invalid", "derived covariate (binary)", "keep (engineered)",
         "1 where raw age is out of [14,99] or missing. Missing-data signal."),
        ("gender", "pre-treatment covariate (categorical)", "map U/NaN -> 'unknown', one-hot",
         "F / M / unknown. 'unknown' is 46% and kept as its own level."),
        ("first_issue_date", "pre-treatment timestamp", "derive tenure_days / issue_month, then EXCLUDE",
         "Loyalty-card issue date; all <= REF_DATE. Raw datetime not a model feature."),
        ("tenure_days", "derived covariate (numeric)", "keep (engineered)",
         "(REF_DATE - first_issue_date) in days. Pre-treatment loyalty tenure."),
        ("issue_month", "derived covariate (numeric 1-12)", "keep (engineered, optional)",
         "Enrolment-cohort seasonality."),
        ("first_redeem_date", "MIXED timestamp (pre/post-treatment)", "CENSOR at REF_DATE, then EXCLUDE",
         "~11.6% of values post-date the campaign -> raw use is temporal leakage."),
        ("has_redeemed_pre", "derived covariate (binary)", "keep (engineered)",
         "1 iff a valid redemption occurred on/before REF_DATE."),
        ("days_since_first_redeem", "derived covariate (numeric)", "keep (engineered) + impute (train)",
         "Days from first valid pre-REF redemption to REF_DATE. NaN when none."),
        ("redeem_info_missing", "derived covariate (binary)", "keep (engineered)",
         "1 when redemption is absent / post-REF / invalid (redeem < issue)."),
        ("products.csv (all columns)", "external dictionary", "EXCLUDE entirely",
         "Not joinable to clients without the absent `purchases` table."),
    ]
    return pd.DataFrame(rows, columns=["feature", "category", "action", "reason"])


def extra_raw_audit(df_clean: pd.DataFrame) -> dict:
    raw = load_raw()
    redeem = raw["first_redeem_date"]
    return {
        "leakage_control_first_redeem_date": {
            "REF_DATE": str(REF_DATE.date()),
            "redemptions_after_REF_(post_treatment)": int((redeem > REF_DATE).sum()),
            "share_post_treatment": float((redeem > REF_DATE).mean()),
            "redeem_before_issue_(invalid)": int((redeem < raw["first_issue_date"]).sum()),
            "redeem_missing_raw": int(redeem.isna().sum()),
            "resolution": "censored at REF_DATE -> has_redeemed_pre / days_since_first_redeem",
        },
        "age_cleaning": {
            "raw_min": float(pd.to_numeric(raw["age"], errors="coerce").min()),
            "raw_max": float(pd.to_numeric(raw["age"], errors="coerce").max()),
            "set_invalid_to_nan": int(df_clean["age_invalid"].sum()),
        },
        "gender_levels": df_clean["gender"].value_counts().to_dict(),
        "products_table": "present but UNUSED (no purchases table to join through)",
        "provided_test_has_labels": False,
    }
