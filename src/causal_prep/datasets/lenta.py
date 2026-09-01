"""
Lenta Uplift — Russian grocery-retailer marketing RCT.

Source: scikit-uplift's public copy (``lenta_dataset.csv.gz``), fetched into
``data/raw/lenta/``.  687,029 clients x 195 columns.

Treatment : `group`         {'test' -> treated, 'control'}   — single binary arm
Outcome   : `response_att`   (binary attributed response)     — primary
Unit      : client (NO id column in this file -> surrogate row index)

Covariates: `age`, `gender`, `children`, `months_from_register`, `main_format`
plus ~188 pre-campaign purchase-behaviour aggregates over 15d/1m/3m/6m/12m
windows (cheque counts, sale sums, discount shares, coefficient-of-variation
`k_var_*`, `stdev_*`, `crazy_purchases_*`, `food_share_*`, `promo_share_*`).
All are historical by construction -> valid pre-treatment X.

`response_sms` / `response_viber` are channel-level delivery/response fractions
recorded *during* the campaign -> post-treatment, excluded from X (and not used
as outcomes: they are fractional and ill-defined for the control arm).
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from .. import config as C
from ..common import DatasetSpec

NAME = "lenta"

_TARGET = "response_att"
_TREAT = "group"
_POST_TREATMENT = ["response_sms", "response_viber"]
_CATEGORICAL = ["gender"]
_BINARY_RAW = ["main_format"]


def _header() -> list[str]:
    with gzip.open(C.RAW_FILES["lenta"], "rt") as f:
        return f.readline().strip().split(",")


def _numeric_covariates() -> list[str]:
    cols = _header()
    drop = {_TARGET, _TREAT, *_POST_TREATMENT, *_CATEGORICAL, *_BINARY_RAW}
    return [c for c in cols if c not in drop]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_FILES["lenta"])


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.insert(0, "client_uid", np.arange(len(df), dtype=np.int64))

    # ---- treatment ------------------------------------------------------
    df["treatment_arm"] = np.where(df[_TREAT] == "test", "treated", "control")
    df["treatment_arm"] = df["treatment_arm"].astype("category")
    df["T"] = (df["treatment_arm"] == "treated").astype("int8")

    # ---- outcome ------------------------------------------------------
    df["response_att"] = df["response_att"].astype("int8")

    # ---- gender: Ж/М/'Не определен'/NaN -> F / M / unknown -----------
    df["gender"] = (df["gender"]
                    .map({"Ж": "F", "М": "M", "Не определен": "unknown"})
                    .fillna("unknown").astype("category"))

    # ---- age: clip implausible -> NaN + flag (impute on train) -------
    age = pd.to_numeric(df["age"], errors="coerce")
    df["age_invalid"] = (~age.between(14, 99)).astype("int8")
    df["age"] = age.where(age.between(14, 99)).astype("float64")

    df["main_format"] = pd.to_numeric(df["main_format"], errors="coerce").astype("float64")

    # ~188 behavioural aggregates: leave NaN in place — the ColumnTransformer
    # median-imputes AND appends train-fitted missing-indicators
    # (numeric_add_indicator=True), because NaN here is structural: a
    # coefficient-of-variation / stdev feature is undefined when the client had
    # too few baskets in that window -> "insufficient activity" signal.
    return df


_NUM = _numeric_covariates()

SPEC = DatasetSpec(
    name=NAME,
    unit_id="client_uid",
    unit_description="grocery client (one row; surrogate client_uid — no id column in source)",
    treatment_primary="T",
    treatment_all=["treatment_arm", "T"],
    treatment_arm_col="treatment_arm",
    arms=["control", "treated"],
    control_arm="control",
    outcomes=["response_att"],
    primary_outcome="response_att",
    x_numeric=[c for c in _NUM if c != "main_format"],
    x_binary=["main_format", "age_invalid"],
    x_categorical=["gender"],
    x_scale_cols=[c for c in _NUM if c != "main_format"],
    stratify_cols=["group", "response_att"],
    excluded_from_x={
        "group": "raw treatment label — mapped to treatment_arm / T",
        "response_att": "post-treatment outcome — the primary label",
        "response_sms": "post-treatment campaign-delivery fraction — leakage; also ill-defined for control",
        "response_viber": "post-treatment campaign-delivery fraction — leakage; also ill-defined for control",
        "client_uid": "surrogate identifier — not a feature",
    },
    invalid_value_checks={"age": (14, 99)},
    notes=[
        "RCT with UNEQUAL allocation: group='test' 75.1% vs 'control' 24.9% "
        "(this is the dataset's holdout design, not a defect). Randomization "
        "verified: max|SMD|~0.03 across 190 covariates, propensity AUC~0.51.",
        "Propensity centres at ~0.75 (= P(treated)). An unregularized logistic fit "
        "on 189 partly-collinear covariates produces a few near-0/near-1 scores, "
        "but 0% of mass falls outside [0.01,0.99] and the scaled/regularized fit "
        "gives AUC 0.497 with tight support -> positivity holds.",
        "No id column and no timestamps in this file: unit id is a surrogate row "
        "index; covariates are pre-campaign window aggregates -> no temporal split.",
        "113 numeric covariates have >5% missing (max 72%); NaN is structural "
        "(CoV/stdev undefined for low basket counts). Handled by median impute + "
        "train-fitted missing-indicators (SimpleImputer add_indicator=True). "
        "Rows are NOT dropped.",
        "response_att base rate ~10.8% -> no resampling.",
        "Full dataset (687k rows) is processed; a 10k stratified subsample is "
        "trivial to draw from the processed train split if a benchmark size is wanted.",
        "This file has no `CardHolder` column (unlike the notebook's description) "
        "and no monetary outcome (only binary response_att).",
    ],
    numeric_add_indicator=True,
    diag_sample=60000,
)


def feature_classification() -> pd.DataFrame:
    rows = [
        ("(row index)", "identifier (surrogate)", "client_uid — exclude from X",
         "No id column in the source file."),
        ("group", "TREATMENT", "map -> treatment_arm / T",
         "'test' = treated (communication), 'control'. 75/25 split."),
        ("response_att", "OUTCOME — post-treatment (primary)", "target only — EXCLUDE from X",
         "Attributed binary response. The uplift label."),
        ("response_sms / response_viber", "post-treatment (channel delivery)", "EXCLUDE from X and from outcomes",
         "Fractional per-channel values recorded during the campaign; ill-defined for control."),
        ("age", "pre-treatment covariate (numeric)", "clip [14,99] -> NaN + flag + impute (train)",
         "~335 out-of-range (min 0) + ~11.8k already-missing. Rows kept."),
        ("age_invalid", "derived covariate (binary)", "keep (engineered)",
         "1 where raw age is out of [14,99] or missing (~12.1k rows)."),
        ("gender", "pre-treatment covariate (categorical)", "map Ж/М/Не определен/NaN -> F/M/unknown, one-hot",
         "'unknown' kept as its own level (~1.4%)."),
        ("children", "pre-treatment covariate (numeric count)", "keep + impute (train) + missing-indicator",
         "0-9; ~8.6k missing."),
        ("months_from_register", "pre-treatment covariate (numeric)", "keep + impute (train) + missing-indicator",
         "Loyalty tenure in months; ~8.6k missing."),
        ("main_format", "pre-treatment covariate (binary)", "keep",
         "Primary store-format indicator (0/1)."),
        ("cheque_count_* / sale_sum_* / sale_count_* / disc_sum_*",
         "pre-treatment covariate (numeric, historical windows)", "keep + impute (train) + missing-indicator",
         "Basket / spend / discount aggregates over 3m/6m/12m. Historical => valid X."),
        ("k_var_* / stdev_* / *_share_* / crazy_purchases_*",
         "pre-treatment covariate (numeric, historical)", "keep + impute (train) + missing-indicator",
         "Dispersion / share / anomaly aggregates over 15d..12m. NaN = insufficient "
         "activity (structural) -> indicator preserves that signal."),
    ]
    return pd.DataFrame(rows, columns=["feature", "category", "action", "reason"])


def extra_raw_audit(df_clean: pd.DataFrame) -> dict:
    nn = df_clean[SPEC.x_numeric].isna().mean().sort_values(ascending=False)
    return {
        "treatment_allocation": df_clean["treatment_arm"].value_counts().to_dict(),
        "n_numeric_covariates": len(SPEC.x_numeric),
        "numeric_cols_missing_gt_5pct": int((nn > 0.05).sum()),
        "numeric_cols_missing_gt_30pct": int((nn > 0.30).sum()),
        "max_missing_fraction": float(nn.max()),
        "post_treatment_cols_excluded": _POST_TREATMENT,
        "gender_levels": df_clean["gender"].value_counts().to_dict(),
        "age_set_invalid_to_nan": int(df_clean["age_invalid"].sum()),
        "id_column_present": False,
        "timestamp_columns_present": False,
    }
