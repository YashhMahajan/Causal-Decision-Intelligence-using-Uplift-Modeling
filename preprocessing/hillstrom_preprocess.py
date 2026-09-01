"""
Stage 2-5 - CLEAN -> CAUSAL FEATURE AUDIT -> SPLIT -> FIT-ON-TRAIN -> TRANSFORM -> SAVE.

Design contract
---------------
X  = pre-treatment covariates only  (historical 12-month customer snapshot)
T  = randomized treatment           (segment: control / mens_email / womens_email)
Y  = post-treatment outcomes        (visit, conversion, spend - 2-week window)

Hard rules enforced here
------------------------
* Raw file is never written to.
* No post-treatment variable (visit / conversion / spend) can enter X.
* `history_segment` (a deterministic bin of `history`) is excluded from X.
* Every *learned* transform (one-hot categories, imputer stats, optional scaler)
  is fit on TRAIN ONLY, then applied to train and test.
* No row dropping, no winsorizing, no resampling (see config switches + report).

Run:  python -m preprocessing.hillstrom_preprocess
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from . import config as C


# --------------------------------------------------------------------------- #
# 2. CLEAN  (deterministic, no fitting, no row loss)
# --------------------------------------------------------------------------- #
def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 2.1 surrogate unit id (traceability only - never a feature)
    df.insert(0, "customer_uid", np.arange(len(df), dtype=np.int64))

    # 2.2 fix the source spelling typo in zip_code ('Surburban' -> 'Suburban').
    #     Pure relabelling - category membership is unchanged.
    df["zip_code"] = df["zip_code"].replace(C.ZIP_SPELLING_FIX)

    # 2.3 canonical treatment columns -------------------------------------
    arm_map = {
        C.CONTROL_LABEL: "control",
        "Mens E-Mail": "mens_email",
        "Womens E-Mail": "womens_email",
    }
    df["treatment_arm"] = df[C.TREATMENT_RAW].map(arm_map).astype("category")
    df["T"] = (df["treatment_arm"] != "control").astype("int8")          # any email
    df["T_mens"] = (df["treatment_arm"] == "mens_email").astype("int8")
    df["T_womens"] = (df["treatment_arm"] == "womens_email").astype("int8")

    # 2.4 ordinal view of the redundant history bucket (EDA / grouping only)
    df["history_segment_ord"] = (
        pd.Categorical(
            df["history_segment"], categories=C.HISTORY_SEGMENT_ORDER, ordered=True
        ).codes.astype("int8")
        + 1
    )

    # 2.5 conservative, causally-safe derived covariates -----------------
    if C.ADD_HISTORY_LOG:
        # history is strictly positive and right-skewed (~13x P99/median);
        # log1p keeps it monotone and readable for linear / propensity models.
        df["history_log1p"] = np.log1p(df["history"].astype("float64"))
    if C.ADD_MW_INTERACTIONS:
        df["mw_count"] = (df["mens"] + df["womens"]).astype("int8")       # 1 or 2
        df["bought_both"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype("int8")

    # 2.6 explicit typing
    df["recency"] = df["recency"].astype("int16")
    df["history"] = df["history"].astype("float64")
    for c in C.BINARY_COVARIATES + C.OUTCOMES[:2]:
        df[c] = df[c].astype("int8")
    df["spend"] = df["spend"].astype("float64")
    for c in C.CATEGORICAL_COVARIATES:
        df[c] = df[c].astype("category")

    return df


# --------------------------------------------------------------------------- #
# 3. CAUSAL FEATURE AUDIT  (classify every column; produce the decision table)
# --------------------------------------------------------------------------- #
def causal_feature_audit(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Return the Feature | Category | Action | Reason table as a DataFrame."""
    rows = [
        ("customer_uid", "identifier (surrogate)", "exclude from X (keep for join)",
         "Row-position id we attach because Hillstrom has none. Carries no "
         "information; using it as a feature would memorise rows."),
        ("recency", "pre-treatment covariate (numeric)", "keep as-is",
         "Months since last purchase, 1-12, measured before send. Bounded, "
         "low skew - no transform needed. Strong effect modifier candidate."),
        ("history", "pre-treatment covariate (numeric)", "keep + derive history_log1p",
         "Historical 12-month $ spend, strictly positive, right-skewed. Raw kept "
         "for tree/uplift learners; log1p added for linear/propensity models. "
         "NOT winsorized - the tail is real high-value heterogeneity."),
        ("history_segment", "derived-redundant of history", "exclude from X (keep as ordinal for EDA)",
         "Deterministic 7-way bucketing of `history` (verified non-overlapping "
         "bin ranges). Collinear and information-losing; kept only as "
         "history_segment_ord for grouped diagnostics."),
        ("mens", "pre-treatment covariate (binary)", "keep",
         "Bought men's merchandise in prior 12 months. Pre-send behaviour."),
        ("womens", "pre-treatment covariate (binary)", "keep",
         "Bought women's merchandise in prior 12 months. Not mutually exclusive "
         "with `mens` (6448 customers have both)."),
        ("newbie", "pre-treatment covariate (binary)", "keep",
         "New customer in the prior 12 months. Pre-send."),
        ("zip_code", "pre-treatment covariate (categorical)", "fix typo + one-hot (fit on train)",
         "Urban / Suburban / Rural. 'Surburban' spelling corrected. K dummies, "
         "handle_unknown='ignore'."),
        ("channel", "pre-treatment covariate (categorical)", "one-hot (fit on train)",
         "Purchase channel(s) in prior year: Phone / Web / Multichannel. K dummies."),
        ("history_log1p", "derived covariate (numeric)", "keep (engineered)",
         "log1p(history). Skew reduction for linear models; monotone => no "
         "ranking distortion for tree models."),
        ("mw_count", "derived covariate (numeric 1-2)", "keep (engineered, optional)",
         "mens + womens. Cheap interpretable interaction proxy."),
        ("bought_both", "derived covariate (binary)", "keep (engineered, optional)",
         "mens AND womens. Flags broad-basket shoppers."),
        ("segment", "TREATMENT (raw)", "map -> treatment_arm / T / T_mens / T_womens",
         "The randomized 3-arm assignment. Native arms preserved; binary "
         "any-email T provided for standard uplift. Never collapsed away."),
        ("visit", "OUTCOME - post-treatment (mediator)", "target only - EXCLUDE from X",
         "Site visit in the 2 weeks AFTER send. Caused by the e-mail => using it "
         "as a feature is textbook post-treatment leakage / mediator bias."),
        ("conversion", "OUTCOME - post-treatment (primary)", "target only - EXCLUDE from X",
         "Purchase in the 2 weeks after send. Primary uplift label."),
        ("spend", "OUTCOME - post-treatment (monetary)", "target only - EXCLUDE from X",
         "Revenue in the 2 weeks after send; zero-inflated, spend>0 iff "
         "conversion=1. Business/ROI outcome."),
        ("treatment_arm/T/T_mens/T_womens", "TREATMENT (engineered)", "keep as T",
         "Canonical treatment encodings derived from `segment`."),
        ("history_segment_ord", "EDA helper (ordinal)", "exclude from X",
         "Integer 1-7 view of history_segment for grouped balance/overlap "
         "tables. Redundant with history for modelling."),
        ("customer_uid/split", "metadata", "keep outside X",
         "Bookkeeping columns in the clean file."),
    ]
    return pd.DataFrame(rows, columns=["feature", "category", "action", "reason"])


# --------------------------------------------------------------------------- #
# 4. SPLIT  (stratified on treatment x rare outcome; no time axis exists)
# --------------------------------------------------------------------------- #
def split(df_clean: pd.DataFrame) -> pd.DataFrame:
    strat = (
        df_clean["treatment_arm"].astype(str)
        + "|"
        + df_clean["conversion"].astype(str)
    )
    train_idx, test_idx = train_test_split(
        df_clean.index,
        test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED,
        shuffle=True,
        stratify=strat,
    )
    df = df_clean.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


# --------------------------------------------------------------------------- #
# 5. FIT-ON-TRAIN preprocessors + TRANSFORM
# --------------------------------------------------------------------------- #
def build_model_matrix_transformer() -> ColumnTransformer:
    """Learned steps: median-impute numerics, most-frequent-impute + one-hot
    categoricals.  On Hillstrom the imputers are no-ops (zero missing) but are
    kept so the identical pipeline is safe on the phase-2 validation datasets."""
    numeric_features = C.NUMERIC_COVARIATES + (
        ["history_log1p"] if C.ADD_HISTORY_LOG else []
    )
    binary_features = list(C.BINARY_COVARIATES) + (
        ["mw_count", "bought_both"] if C.ADD_MW_INTERACTIONS else []
    )
    categorical_features = list(C.CATEGORICAL_COVARIATES)

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop=C.ONEHOT_DROP,
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.int8,
                ),
            ),
        ]
    )

    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("bin", binary_pipe, binary_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def fit_transform_save(df_split: pd.DataFrame) -> dict:
    train = df_split[df_split["split"] == "train"].copy()
    test = df_split[df_split["split"] == "test"].copy()

    # ---- model matrix (X) : fit on train only -------------------------
    ct = build_model_matrix_transformer()
    X_train = ct.fit_transform(train)
    X_test = ct.transform(test)
    feature_names = list(X_train.columns)

    # ---- optional standardizer : fit on train only, kept SEPARATE ------
    scale_cols = [c for c in C.SCALE_COLUMNS if c in X_train.columns]
    scaler = StandardScaler().fit(X_train[scale_cols])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    # ---- assemble analysis frames (X + T + Y + ids) -------------------
    keep_meta = ["customer_uid", "split"]
    keep_T = ["treatment_arm", "T", "T_mens", "T_womens"]
    keep_Y = C.OUTCOMES

    def assemble(X, src):
        return pd.concat(
            [src[keep_meta].reset_index(drop=True),
             X.reset_index(drop=True),
             src[keep_T + keep_Y].reset_index(drop=True)],
            axis=1,
        )

    train_out = assemble(X_train, train)
    test_out = assemble(X_test, test)
    train_out_scaled = assemble(X_train_scaled, train)
    test_out_scaled = assemble(X_test_scaled, test)

    # ---- persist ------------------------------------------------------
    P = C.PROCESSED_DIR
    # canonical human-readable cleaned dataset (pre-encoding, all rows)
    clean_cols = (
        keep_meta
        + ["recency", "history", "history_log1p", "mens", "womens", "newbie",
           "mw_count", "bought_both", "zip_code", "channel",
           "history_segment", "history_segment_ord"]
        + keep_T + keep_Y
    )
    clean_cols = [c for c in clean_cols if c in df_split.columns]
    df_split[clean_cols].to_csv(P / "hillstrom_clean.csv", index=False)

    train_out.to_parquet(P / "train.parquet", index=False)
    test_out.to_parquet(P / "test.parquet", index=False)
    train_out.to_csv(P / "train.csv", index=False)
    test_out.to_csv(P / "test.csv", index=False)
    train_out_scaled.to_parquet(P / "train_scaled.parquet", index=False)
    test_out_scaled.to_parquet(P / "test_scaled.parquet", index=False)

    joblib.dump(ct, P / "preprocessor.joblib")
    joblib.dump({"scaler": scaler, "columns": scale_cols}, P / "scaler.joblib")

    feature_spec = {
        "dataset": "hillstrom",
        "unit_of_analysis": "individual customer (one row; surrogate customer_uid)",
        "random_seed": C.RANDOM_SEED,
        "test_size": C.TEST_SIZE,
        "n_rows_total": int(len(df_split)),
        "n_train": int(len(train_out)),
        "n_test": int(len(test_out)),
        "X_features": feature_names,
        "X_numeric_scaled_variant_cols": scale_cols,
        "treatment_cols": keep_T,
        "treatment_primary_binary": "T",
        "treatment_native_arms": ["control", "mens_email", "womens_email"],
        "outcome_cols": keep_Y,
        "primary_outcome": C.PRIMARY_OUTCOME,
        "secondary_outcome": C.SECONDARY_OUTCOME,
        "mediator_outcome": C.MEDIATOR_OUTCOME,
        "excluded_from_X": {
            "post_treatment_outcomes": C.OUTCOMES,
            "derived_redundant": C.DERIVED_REDUNDANT + ["history_segment_ord"],
            "identifiers_metadata": ["customer_uid", "split"],
        },
        "learned_on_train_only": [
            "ColumnTransformer (one-hot categories, imputer statistics)",
            "StandardScaler (separate artifact, optional)",
        ],
        "artifacts": {
            "clean_full": "hillstrom_clean.csv",
            "train": "train.parquet / train.csv",
            "test": "test.parquet / test.csv",
            "scaled_variant": "train_scaled.parquet / test_scaled.parquet",
            "preprocessor": "preprocessor.joblib",
            "scaler": "scaler.joblib",
        },
    }
    (P / "feature_spec.json").write_text(json.dumps(feature_spec, indent=2))

    return {
        "transformer": ct,
        "scaler": scaler,
        "feature_names": feature_names,
        "train": train_out,
        "test": test_out,
        "feature_spec": feature_spec,
    }
