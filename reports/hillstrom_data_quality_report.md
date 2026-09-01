# Hillstrom - Preprocessing & Data-Quality Report

_Seed 20240501; test fraction 0.2; pipeline `preprocessing/`._

## 1. Integrity checks (processed data)

- `row_count_conserved` = `True`
- `no_uid_overlap` = `True`
- `uid_unique` = `True`
- `X_missing_cells` = `0`
- `X_non_finite_cells` = `0`
- `outcome_or_redundant_in_X` = `[]`
- `X_dtypes_numeric` = `True`
- `onehot_zip_code_rowsum_ok` = `True`
- `onehot_channel_rowsum_ok` = `True`
- `train_test_split_ratio` = `0.8`

## 2. Raw-audit highlights

- raw shape: `{'rows': 64000, 'cols': 12}`; missing cells: **0**
- duplicate rows: **7634** across all 3 arms (`{'Womens E-Mail': 2575, 'No E-Mail': 2555, 'Mens E-Mail': 2504}`), 0 conversions among them -> **kept** (coincidental collisions, no id to disprove identity)
- invalid values: recency out of range 0, history <= 0 0, spend < 0 0
- class imbalance: conversion positive rate **0.0090**, visit 0.1468 - **no resampling / SMOTE applied** (would distort the RCT base rate)
- randomization: max |SMD| 0.0088, propensity AUC 0.4966

## 3. Reproducibility

- `test_uid_set_sha256_16` = `5acb3bbca25c3571`
- `seed` = `20240501`
- `note` = `Deterministic given seed; re-run this function to confirm.`

## 4. Cleaning decisions (deterministic, no row loss)

- attached surrogate `customer_uid` (row position) - metadata only
- corrected `zip_code` spelling `Surburban -> Suburban` (relabel only)
- mapped `segment` -> `treatment_arm` {control, mens_email, womens_email} and derived binary `T`, plus `T_mens`, `T_womens`
- excluded `history_segment` from X (deterministic bin of `history`); kept `history_segment_ord` for grouped diagnostics
- excluded `visit`, `conversion`, `spend` from X (post-treatment outcomes)
- derived `history_log1p` (skew), `mw_count`, `bought_both` (interpretable)
- no winsorizing, no duplicate drop, no imputation needed (0 missing)
- learned steps (one-hot categories, imputer stats, optional scaler) fit on **train only**