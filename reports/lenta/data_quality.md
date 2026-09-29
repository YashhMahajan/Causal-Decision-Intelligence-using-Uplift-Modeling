# lenta — Preprocessing & Data-Quality Report

_Seed 20240501; stratified split on `group , response_att`; test fraction 0.2._

## 1. Shape & missingness

- cleaned frame: **687029 rows × 199 cols**
- total missing cells (post-clean): **25631223**
- columns with missing values (top 12): `k_var_sku_price_15d_g49`=496259, `k_var_disc_share_15d_g49`=496159, `k_var_count_per_cheque_15d_g34`=468551, `k_var_sku_price_15d_g34`=468551, `k_var_disc_share_15d_g34`=468467, `k_var_count_per_cheque_15d_g24`=442121, `k_var_disc_share_15d_g24`=442054, `k_var_count_per_cheque_1m_g49`=414473, `k_var_sku_price_1m_g49`=414473, `k_var_disc_share_1m_g49`=414369, `k_var_sku_price_1m_g54`=388217, `k_var_disc_share_1m_g54`=388139

## 2. Duplicates & invalid values

- unit-id duplicates: **0**
- full-row duplicates (extra): **0**
- range checks (count outside allowed range): `age`=12100

## 3. Treatment & outcomes (cleaned frame)

- treatment `T` — arm counts `{'treated': 515892, 'control': 171137}`; P(treated) = 0.7509
- outcome rates by arm:
  - **control**: response_att 0.1026
  - **treated**: response_att 0.1101
- naive (unadjusted) ATE vs control:
  - **treated_vs_control**: response_att +0.0075
- binary-outcome base rates: `response_att` 0.1082  → **no resampling / SMOTE applied**

## 4. Randomization sanity

- max |SMD| (treated vs control) = **0.0263** (flag 0.1); features above flag: 0
- 5-fold propensity AUC = **0.5094** (n=60000); support [0.0262, 0.9923]; mass outside trim 0.0000

## 5. Dataset-specific audit

```json
{
  "treatment_allocation": {
    "treated": 515892,
    "control": 171137
  },
  "n_numeric_covariates": 189,
  "numeric_cols_missing_gt_5pct": 113,
  "numeric_cols_missing_gt_30pct": 60,
  "max_missing_fraction": 0.7223261317935633,
  "post_treatment_cols_excluded": [
    "response_sms",
    "response_viber"
  ],
  "gender_levels": {
    "F": 433448,
    "M": 243910,
    "unknown": 9671
  },
  "age_set_invalid_to_nan": 12100,
  "id_column_present": false,
  "timestamp_columns_present": false
}
```

## 6. Integrity checks (processed data)

- `row_count_conserved` = `True`
- `no_unit_overlap` = `True`
- `unit_id_unique` = `True`
- `X_missing_cells` = `0`
- `X_non_finite_cells` = `0`
- `X_all_numeric` = `True`
- `leakage_outcome_or_excluded_in_X` = `[]`
- `train_fraction` = `0.8`
- `onehot_gender_rowsum_ok` = `True`

## 7. Reproducibility

- `test_unit_id_sha256_16` = `71cfa82a34e59195`
- `seed` = `20240501`
- `note` = `Deterministic given seed; re-run to confirm.`

## 8. Causal-safety decisions

- RCT with UNEQUAL allocation: group='test' 75.1% vs 'control' 24.9% (this is the dataset's holdout design, not a defect). Randomization verified: max|SMD|~0.03 across 190 covariates, propensity AUC~0.51.
- Propensity centres at ~0.75 (= P(treated)). An unregularized logistic fit on 189 partly-collinear covariates produces a few near-0/near-1 scores, but 0% of mass falls outside [0.01,0.99] and the scaled/regularized fit gives AUC 0.497 with tight support -> positivity holds.
- No id column and no timestamps in this file: unit id is a surrogate row index; covariates are pre-campaign window aggregates -> no temporal split.
- 113 numeric covariates have >5% missing (max 72%); NaN is structural (CoV/stdev undefined for low basket counts). Handled by median impute + train-fitted missing-indicators (SimpleImputer add_indicator=True). Rows are NOT dropped.
- response_att base rate ~10.8% -> no resampling.
- Full dataset (687k rows) is processed; a 10k stratified subsample is trivial to draw from the processed train split if a benchmark size is wanted.
- This file has no `CardHolder` column (unlike the notebook's description) and no monetary outcome (only binary response_att).

### Excluded from X (and why)
- `group` — raw treatment label — mapped to treatment_arm / T
- `response_att` — post-treatment outcome — the primary label
- `response_sms` — post-treatment campaign-delivery fraction — leakage; also ill-defined for control
- `response_viber` — post-treatment campaign-delivery fraction — leakage; also ill-defined for control
- `client_uid` — surrogate identifier — not a feature