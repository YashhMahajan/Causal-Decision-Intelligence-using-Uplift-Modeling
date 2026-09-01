# `causal_prep` — pipeline internals

Standard, dataset-agnostic preprocessing for uplift / HTE modelling. One code
path; per-dataset differences live only in `datasets/<name>.py` and are surfaced
in the reports, never hidden.

## Flow

```
run.run_one(name)
  ├─ mod.load_raw()                      dataset-specific I/O (X5 also joins clients)
  ├─ mod.clean(raw)                      deterministic, row-count preserving:
  │                                        surrogate/native unit id, canonical T
  │                                        (arm label + binary + one-vs-rest),
  │                                        anti-leakage censoring, engineered
  │                                        covariates, explicit dtypes
  ├─ common.audit_raw(clean, spec)       → reports/<ds>/audit.json
  │                                        missingness, dupes, range checks,
  │                                        arm balance, outcome-by-arm, naive ATE,
  │                                        SMD + 5-fold propensity AUC
  ├─ mod.feature_classification()        → reports/<ds>/feature_classification.csv
  ├─ common.make_split(clean, spec)      stratified 80/20, seeded; adds `split`
  ├─ common.fit_transform_save(...)      ColumnTransformer FIT ON TRAIN ONLY
  │                                        (median-impute numeric [+ indicators],
  │                                         most-frequent-impute binary,
  │                                         most-frequent-impute + one-hot categ.)
  │                                        optional StandardScaler (separate art.)
  │                                        → data/interim/<ds>/, data/processed/<ds>/,
  │                                          artifacts/<ds>/
  └─ validate: integrity_checks, treatment_outcome_diag, balance_overlap,
               reproducibility_hash  → reports/<ds>/{validation.json,
               data_quality.md, balance_overlap.md, figures/}
```

## `DatasetSpec` (in `common.py`)

The single declarative object each dataset module fills in:

| field | meaning |
|---|---|
| `unit_id`, `unit_description` | unit-of-analysis id column (never a feature) |
| `treatment_primary`, `treatment_all`, `treatment_arm_col`, `arms`, `control_arm` | canonical treatment encodings kept in outputs |
| `outcomes`, `primary_outcome` | post-treatment targets (never features) |
| `x_numeric`, `x_binary`, `x_categorical` | the ONLY columns allowed into X, by transform family |
| `x_scale_cols` | subset standardized in the `*_scaled` variant |
| `stratify_cols` | split stratification keys (treatment × rare outcome) |
| `excluded_from_x` | `column → reason` for everything deliberately kept out of X |
| `invalid_value_checks` | `column → (lo, hi)` range assertions for the audit |
| `numeric_add_indicator` | append train-fitted missing-indicators (Lenta) |
| `diag_sample` | cap rows for the propensity diagnostics on large data (Lenta) |
| `notes` | dataset-specific caveats, printed into `data_quality.md` |

## Outputs per dataset

```
data/interim/<ds>/<ds>_clean.parquet     cleaned, pre-encoding, ALL rows (X+T+Y+ids+split)
data/processed/<ds>/train.parquet        model matrix: X (numeric) + T cols + Y cols + unit id + split
data/processed/<ds>/test.parquet
data/processed/<ds>/train_scaled.parquet x_scale_cols standardized (train-fit); for linear learners
data/processed/<ds>/test_scaled.parquet
data/processed/<ds>/{train,test}.csv     mirror, only for datasets ≤ 100k rows
data/processed/x5/score.parquet          transformed unlabelled competition holdout (X5 only)
artifacts/<ds>/preprocessor.joblib       fitted ColumnTransformer
artifacts/<ds>/scaler.joblib             fitted StandardScaler + column list
artifacts/<ds>/feature_spec.json         X / T / Y column lists, exclusions, provenance, counts
```

## Conventions

- **Never drop rows** for missingness, duplicates, outliers, or class imbalance.
  Flag (`*_invalid`, missing-indicators), impute on train, document.
- **Never resample / SMOTE** — it distorts the RCT base rate that Qini/AUUC need.
- **Never winsorize** genuine heavy tails (high-value customers are the point).
- **Scaling is opt-in** — the canonical matrix is unscaled (tree/uplift learners
  don't need it and it kills interpretability); `*_scaled` is provided for linear
  S/T/X/DR learners.
- Anything computed from data (encoder categories, impute stats, scaler, missing-
  indicator set) is `.fit()` on **train only**. Row-wise deterministic derivations
  (`log1p`, date arithmetic, `a & b`) may run pre-split — they have no parameters.
