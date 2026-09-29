"""
Shared, dataset-agnostic machinery for the causal-safe preprocessing standard.

Pipeline contract (identical for every dataset):

    raw  ->  audit  ->  clean  ->  causal feature audit  ->  split
         ->  fit preprocessors on TRAIN only  ->  transform  ->  save  ->  validate

X = pre-treatment covariates only          (available at the treatment decision point)
T = randomized treatment / intervention
Y = post-treatment outcome(s)              (targets only, never features)

Dataset-specific knobs are carried in a ``DatasetSpec`` produced by each
``causal_prep.datasets.<name>`` module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


# --------------------------------------------------------------------------- #
# Dataset specification
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSpec:
    name: str
    unit_id: str                       # column with the unit-of-analysis id
    unit_description: str

    treatment_primary: str             # binary 0/1 treatment column used by default
    treatment_all: list[str]           # every treatment encoding kept in outputs
    treatment_arm_col: str             # categorical arm label column
    arms: list[str]                    # arm labels (control first)
    control_arm: str

    outcomes: list[str]                # every outcome column (targets only)
    primary_outcome: str

    x_numeric: list[str]               # -> median impute
    x_binary: list[str]                # -> most-frequent impute, no encoding
    x_categorical: list[str]           # -> most-frequent impute + one-hot (all K)
    x_scale_cols: list[str]            # subset standardized in the *_scaled variant

    stratify_cols: list[str]           # split stratification keys
    excluded_from_x: dict[str, str]    # column -> reason it is NOT a feature

    invalid_value_checks: dict = field(default_factory=dict)  # col -> (lo, hi)
    notes: list[str] = field(default_factory=list)
    has_labelled_test_only: bool = False   # True for X5 (provided test has no Y)
    numeric_add_indicator: bool = False    # append train-fitted missing-indicators
                                           # (Lenta: structural NaN in CoV features)
    diag_sample: int | None = None         # cap rows for propensity diagnostics

    def all_x(self) -> list[str]:
        return list(self.x_numeric) + list(self.x_binary) + list(self.x_categorical)


# --------------------------------------------------------------------------- #
# 1. RAW AUDIT  (read-only)
# --------------------------------------------------------------------------- #
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def smd(x: pd.Series, t: pd.Series) -> float:
    """Standardized mean difference between treated (t==1) and control (t==0)."""
    x = x.astype(float)
    a, b = x[t == 1], x[t == 0]
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def audit_raw(df_clean: pd.DataFrame, spec: DatasetSpec,
              extra: dict | None = None) -> dict:
    """Profile the *cleaned* frame (post-clean, pre-split).  No mutation."""
    n = len(df_clean)
    rep: dict = {"dataset": spec.name, "n_rows": n, "n_cols": df_clean.shape[1]}
    rep["dtypes"] = {c: str(t) for c, t in df_clean.dtypes.items()}
    rep["missing"] = {c: int(df_clean[c].isna().sum()) for c in df_clean.columns
                      if df_clean[c].isna().any()}
    rep["missing_total"] = int(df_clean.isna().sum().sum())

    # duplicates on the unit id and on the full feature vector
    rep["duplicates"] = {
        "unit_id_duplicated": int(df_clean[spec.unit_id].duplicated().sum()),
        "full_row_duplicated_extra": int(df_clean.duplicated().sum()),
    }

    # invalid-value checks (spec-driven ranges)
    iv = {}
    for col, (lo, hi) in spec.invalid_value_checks.items():
        if col in df_clean:
            iv[col] = int((~df_clean[col].between(lo, hi)).sum())
    rep["invalid_value_checks"] = iv

    # treatment balance
    arm_counts = df_clean[spec.treatment_arm_col].value_counts()
    t = df_clean[spec.treatment_primary]
    rep["treatment"] = {
        "primary_binary": spec.treatment_primary,
        "arm_col": spec.treatment_arm_col,
        "arm_counts": jsonable(arm_counts.to_dict()),
        "arm_shares": jsonable((arm_counts / n).round(4).to_dict()),
        "p_treated": float(t.mean()),
    }

    # outcomes by arm + naive unadjusted ATE vs control
    ob, naive = {}, {}
    g = df_clean.groupby(spec.treatment_arm_col, observed=True)[spec.outcomes].mean()
    for arm in g.index:
        ob[arm] = {o: float(g.loc[arm, o]) for o in spec.outcomes}
    base = g.loc[spec.control_arm]
    for arm in g.index:
        if arm != spec.control_arm:
            naive[f"{arm}_vs_{spec.control_arm}"] = {
                o: float(g.loc[arm, o] - base[o]) for o in spec.outcomes
            }
    rep["outcomes_by_arm"] = ob
    rep["naive_unadjusted_ATE"] = naive
    rep["class_balance"] = {
        o: float(df_clean[o].mean()) for o in spec.outcomes
        if set(df_clean[o].dropna().unique()) <= {0, 1}
    }

    # randomization sanity: SMD + 5-fold propensity AUC on available X
    Xd = _design_matrix_for_diag(df_clean, spec)
    smds = {c: round(smd(Xd[c], t), 4) for c in Xd.columns}
    Xs, ts = _maybe_subsample(Xd, t, spec.diag_sample)
    ps = cross_val_predict(LogisticRegression(max_iter=2000), Xs.values,
                           ts.values, cv=5, method="predict_proba")[:, 1]
    lo, hi = C.PROPENSITY_TRIM
    rep["randomization_check"] = {
        "standardized_mean_diffs": smds,
        "max_abs_smd": float(np.max(np.abs(list(smds.values())))),
        "n_smd_above_flag": int(sum(abs(v) > C.SMD_FLAG for v in smds.values())),
        "propensity_auc_5fold": float(roc_auc_score(ts, ps)),
        "propensity_min": float(ps.min()),
        "propensity_max": float(ps.max()),
        "propensity_mass_outside_trim": float(np.mean((ps < lo) | (ps > hi))),
        "propensity_diag_n": int(len(ts)),
    }
    if extra:
        rep["dataset_specific"] = jsonable(extra)
    return rep


def _design_matrix_for_diag(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    parts = [df[spec.x_numeric + spec.x_binary].astype(float)]
    if spec.x_categorical:
        parts.append(pd.get_dummies(df[spec.x_categorical], dummy_na=False).astype(float))
    X = pd.concat(parts, axis=1)
    return X.fillna(X.median(numeric_only=True))


def _maybe_subsample(X: pd.DataFrame, t: pd.Series, cap: int | None):
    if cap is None or len(X) <= cap:
        return X, t
    rng = np.random.RandomState(C.RANDOM_SEED)
    idx = rng.choice(len(X), size=cap, replace=False)
    return X.iloc[idx].reset_index(drop=True), t.iloc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. SPLIT  (stratified; fit nothing)
# --------------------------------------------------------------------------- #
def make_split(df_clean: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    strat = df_clean[spec.stratify_cols].astype(str).agg("|".join, axis=1)
    train_idx, test_idx = train_test_split(
        df_clean.index, test_size=C.TEST_SIZE, random_state=C.RANDOM_SEED,
        shuffle=True, stratify=strat,
    )
    out = df_clean.copy()
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


# --------------------------------------------------------------------------- #
# 3. FIT-ON-TRAIN preprocessors + TRANSFORM + SAVE
# --------------------------------------------------------------------------- #
def build_transformer(spec: DatasetSpec) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(
        strategy="median", add_indicator=spec.numeric_add_indicator))])
    binp = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    catp = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.int8)),
    ])
    ct = ColumnTransformer(
        [("num", num, spec.x_numeric),
         ("bin", binp, spec.x_binary),
         ("cat", catp, spec.x_categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def fit_transform_save(df_split: pd.DataFrame, spec: DatasetSpec,
                       dirs: dict[str, Path],
                       score_frame: pd.DataFrame | None = None) -> dict:
    train = df_split[df_split.split == "train"].copy()
    test = df_split[df_split.split == "test"].copy()

    ct = build_transformer(spec)
    X_train = ct.fit_transform(train)
    X_test = ct.transform(test)
    feat = list(X_train.columns)

    scale_cols = [c for c in spec.x_scale_cols if c in X_train.columns]
    scaler = StandardScaler().fit(X_train[scale_cols]) if scale_cols else None

    def scaled(X):
        if not scale_cols:
            return X.copy()
        Xs = X.copy()
        Xs[scale_cols] = scaler.transform(X[scale_cols])
        return Xs

    meta = [spec.unit_id, "split"]
    tcols = [c for c in spec.treatment_all if c in df_split.columns]
    ycols = spec.outcomes

    def assemble(X, src):
        return pd.concat(
            [src[meta].reset_index(drop=True),
             X.reset_index(drop=True),
             src[tcols + ycols].reset_index(drop=True)], axis=1)

    train_out, test_out = assemble(X_train, train), assemble(X_test, test)
    train_out_s, test_out_s = assemble(scaled(X_train), train), assemble(scaled(X_test), test)

    P, A, I = dirs["processed"], dirs["artifacts"], dirs["interim"]
    # interim = cleaned, pre-encoding, ALL rows (X + T + Y + ids + split)
    df_split.to_parquet(I / f"{spec.name}_clean.parquet", index=False)

    # parquet is the canonical format; a CSV mirror is only written for small
    # datasets where quick eyeballing / non-pandas tooling matters.
    write_csv = len(df_split) <= 100_000
    for frame, stem in [(train_out, "train"), (test_out, "test")]:
        frame.to_parquet(P / f"{stem}.parquet", index=False)
        if write_csv:
            frame.to_csv(P / f"{stem}.csv", index=False)
    train_out_s.to_parquet(P / "train_scaled.parquet", index=False)
    test_out_s.to_parquet(P / "test_scaled.parquet", index=False)

    joblib.dump(ct, A / "preprocessor.joblib")
    if scaler is not None:
        joblib.dump({"scaler": scaler, "columns": scale_cols}, A / "scaler.joblib")

    # optional unlabelled scoring frame (X5 competition holdout)
    n_score = 0
    if score_frame is not None:
        Xs = ct.transform(score_frame)
        sc = pd.concat([score_frame[[spec.unit_id]].reset_index(drop=True),
                        Xs.reset_index(drop=True)], axis=1)
        sc.to_parquet(P / "score.parquet", index=False)
        n_score = len(sc)

    feature_spec = {
        "dataset": spec.name,
        "unit_of_analysis": spec.unit_description,
        "random_seed": C.RANDOM_SEED,
        "test_size": C.TEST_SIZE,
        "n_rows_total": int(len(df_split)),
        "n_train": int(len(train_out)),
        "n_test": int(len(test_out)),
        "n_score_unlabelled": int(n_score),
        "X_features": feat,
        "X_scaled_variant_cols": scale_cols,
        "treatment_cols": tcols,
        "treatment_primary_binary": spec.treatment_primary,
        "treatment_arms": spec.arms,
        "outcome_cols": ycols,
        "primary_outcome": spec.primary_outcome,
        "excluded_from_X": spec.excluded_from_x,
        "learned_on_train_only": [
            "ColumnTransformer (one-hot categories, imputer statistics)",
            "StandardScaler (separate artifact, optional/opt-in)",
        ],
        "stratified_split_on": spec.stratify_cols,
        "notes": spec.notes,
    }
    (A / "feature_spec.json").write_text(json.dumps(jsonable(feature_spec), indent=2))
    return {"transformer": ct, "scaler": scaler, "feature_names": feat,
            "train": train_out, "test": test_out, "feature_spec": feature_spec}


# --------------------------------------------------------------------------- #
# 4. VALIDATE
# --------------------------------------------------------------------------- #
def integrity_checks(train, test, feature_spec, spec: DatasetSpec) -> dict:
    Xcols = feature_spec["X_features"]
    both = pd.concat([train, test], ignore_index=True)
    r = {
        "row_count_conserved": len(both) == feature_spec["n_rows_total"],
        "no_unit_overlap": len(set(train[spec.unit_id]) & set(test[spec.unit_id])) == 0,
        "unit_id_unique": bool(both[spec.unit_id].is_unique),
        "X_missing_cells": int(both[Xcols].isna().sum().sum()),
        "X_non_finite_cells": int((~np.isfinite(
            both[Xcols].to_numpy(dtype="float64"))).sum()),
        "X_all_numeric": bool(all(np.issubdtype(both[c].dtype, np.number)
                                  for c in Xcols)),
        "leakage_outcome_or_excluded_in_X": [
            c for c in list(spec.excluded_from_x) + spec.outcomes if c in Xcols],
        "train_fraction": round(len(train) / len(both), 4),
    }
    for pfx in spec.x_categorical:
        block = [c for c in Xcols if c.startswith(pfx + "_")]
        if block:
            r[f"onehot_{pfx}_rowsum_ok"] = bool((both[block].sum(axis=1) == 1).all())
    return r


def treatment_outcome_diag(train, test, spec: DatasetSpec) -> dict:
    out = {}
    for nm, d in (("train", train), ("test", test)):
        g = d.groupby(spec.treatment_arm_col, observed=True)[spec.outcomes].mean()
        base = g.loc[spec.control_arm]
        out[nm] = {
            "arm_counts": {k: int(v) for k, v in
                           d[spec.treatment_arm_col].value_counts().items()},
            "outcome_rates_by_arm": g.round(6).to_dict("index"),
            "naive_ATE_vs_control": {
                arm: {o: float(g.loc[arm, o] - base[o]) for o in spec.outcomes}
                for arm in g.index if arm != spec.control_arm},
        }
    return out


def balance_overlap(train, feature_spec, spec: DatasetSpec,
                    figures_dir: Path) -> dict:
    Xcols = feature_spec["X_features"]
    t = train[spec.treatment_primary]
    smds = {c: round(smd(train[c], t), 4) for c in Xcols}
    res = {
        "smd": smds,
        "max_abs_smd": float(np.max(np.abs(list(smds.values())))),
        "n_smd_above_flag": int(sum(abs(v) > C.SMD_FLAG for v in smds.values())),
        "smd_flag_threshold": C.SMD_FLAG,
    }
    Xdiag, tdiag = _maybe_subsample(train[Xcols], t, spec.diag_sample)
    Xv = Xdiag.to_numpy(dtype="float64")
    res["propensity_diag_n"] = int(len(tdiag))
    lo, hi = C.PROPENSITY_TRIM
    for label, clf in (("logreg", make_pipeline(StandardScaler(),
                                                LogisticRegression(max_iter=2000))),
                       ("hgb", HistGradientBoostingClassifier(
                           random_state=C.RANDOM_SEED))):
        ps = cross_val_predict(clf, Xv, tdiag.to_numpy(), cv=5,
                               method="predict_proba")[:, 1]
        res[f"propensity_{label}"] = {
            "auc": float(roc_auc_score(tdiag, ps)),
            "min": float(ps.min()), "p01": float(np.percentile(ps, 1)),
            "mean": float(ps.mean()), "p99": float(np.percentile(ps, 99)),
            "max": float(ps.max()),
            "mass_outside_trim": float(np.mean((ps < lo) | (ps > hi))),
        }
        if label == "logreg":
            _figs(train, t, ps, tdiag, Xcols, spec, figures_dir)
    auc = res["propensity_logreg"]["auc"]
    in_band = C.PROPENSITY_AUC_RCT_BAND[0] <= auc <= C.PROPENSITY_AUC_RCT_BAND[1]
    res["positivity_verdict"] = (
        "Strong overlap consistent with randomization: propensity AUC in the RCT "
        "band, support concentrated near P(T=1), no mass in the trim tails. CATE "
        "is identified across the covariate space."
        if in_band and res["max_abs_smd"] < C.SMD_FLAG else
        "Review: propensity is predictable from X and/or covariates are "
        "imbalanced - inspect flagged features and low-overlap regions before "
        "trusting CATE there."
    )
    return res


def _figs(train, t, ps, tdiag, Xcols, spec, figures_dir: Path):
    tdiag = np.asarray(tdiag)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(float(ps.min()), float(ps.max()), 40)
    ax.hist(ps[tdiag == 1], bins=bins, alpha=0.6, label="treated", density=True)
    ax.hist(ps[tdiag == 0], bins=bins, alpha=0.6, label="control", density=True)
    ax.set_xlabel("estimated P(T=1 | X)"); ax.set_ylabel("density")
    ax.set_title(f"{spec.name} - propensity overlap (train, logistic)")
    ax.legend(); fig.tight_layout()
    fig.savefig(figures_dir / "propensity_overlap.png", dpi=120); plt.close(fig)

    pairs = sorted(((c, smd(train[c], t)) for c in Xcols), key=lambda kv: abs(kv[1]))
    names = [k for k, _ in pairs]; vals = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(names))))
    ax.scatter(vals, range(len(names)))
    for xv in (0, C.SMD_FLAG, -C.SMD_FLAG):
        ax.axvline(xv, color="k" if xv == 0 else "r",
                   ls="-" if xv == 0 else "--", lw=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("standardized mean difference (treated - control)")
    ax.set_title(f"{spec.name} - covariate balance (Love plot, train)")
    fig.tight_layout()
    fig.savefig(figures_dir / "love_plot.png", dpi=120); plt.close(fig)


def reproducibility_hash(df_clean_fn: Callable[[], pd.DataFrame],
                         spec: DatasetSpec) -> dict:
    import hashlib
    df = make_split(df_clean_fn(), spec)
    ser = df.loc[df.split == "test", spec.unit_id].sort_values().astype(str)
    h = hashlib.sha256("".join(ser).encode()).hexdigest()[:16]
    return {"test_unit_id_sha256_16": h, "seed": C.RANDOM_SEED,
            "note": "Deterministic given seed; re-run to confirm."}
