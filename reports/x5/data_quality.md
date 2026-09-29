# x5 — Preprocessing & Data-Quality Report

_Seed 20240501; stratified split on `treatment_flg , target`; test fraction 0.2._

## 1. Shape & missingness

- cleaned frame: **200039 rows × 14 cols**
- total missing cells (post-clean): **59490**
- columns with missing values (top 12): `days_since_first_redeem`=41059, `first_redeem_date`=17546, `age`=885

## 2. Duplicates & invalid values

- unit-id duplicates: **0**
- full-row duplicates (extra): **0**
- range checks (count outside allowed range): `age`=885, `tenure_days`=0, `issue_month`=0

## 3. Treatment & outcomes (cleaned frame)

- treatment `treatment_flg` — arm counts `{'control': 100058, 'treated': 99981}`; P(treated) = 0.4998
- outcome rates by arm:
  - **control**: target 0.6033
  - **treated**: target 0.6365
- naive (unadjusted) ATE vs control:
  - **treated_vs_control**: target +0.0332
- binary-outcome base rates: `target` 0.6199  → **no resampling / SMOTE applied**

## 4. Randomization sanity

- max |SMD| (treated vs control) = **0.0085** (flag 0.1); features above flag: 0
- 5-fold propensity AUC = **0.4998** (n=200039); support [0.4845, 0.5137]; mass outside trim 0.0000

## 5. Dataset-specific audit

```json
{
  "leakage_control_first_redeem_date": {
    "REF_DATE": "2019-03-16",
    "redemptions_after_REF_(post_treatment)": 23268,
    "share_post_treatment": 0.11631731812296602,
    "redeem_before_issue_(invalid)": 245,
    "redeem_missing_raw": 17546,
    "resolution": "censored at REF_DATE -> has_redeemed_pre / days_since_first_redeem"
  },
  "age_cleaning": {
    "raw_min": -7491.0,
    "raw_max": 1852.0,
    "set_invalid_to_nan": 885
  },
  "gender_levels": {
    "unknown": 92832,
    "F": 73696,
    "M": 33511
  },
  "products_table": "present but UNUSED (no purchases table to join through)",
  "provided_test_has_labels": false
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

- `test_unit_id_sha256_16` = `9573264c1085e3b6`
- `seed` = `20240501`
- `note` = `Deterministic given seed; re-run to confirm.`

## 8. Causal-safety decisions

- RCT: treatment_flg ~50/50 (99,981 / 100,058); randomization verified (max|SMD|<0.01 on available covariates, propensity AUC~0.50).
- Feature set is THIN: the raw `purchases` transaction table (X5's main value per docs §6.2) is NOT in the repo, so covariates are limited to demographics + loyalty-card dates. products.csv is not joinable without purchases -> unused.
- `first_redeem_date` censored at REF_DATE=2019-03-16 to prevent post-treatment leakage (raw values extend to 2019-11; 11.6% post-date the campaign).
- age has implausible values (<14 or >99, incl. negatives / years like 1901): ~885 in the labelled RCT frame, 1,404 across all 400k clients. Set to NaN + `age_invalid` flag + median impute on train (rows NOT dropped).
- target base rate ~62% (not rare) -> no resampling.
- Provided uplift_test.csv has NO labels: it is transformed to `score.parquet` for later submission scoring, and is NOT used as an evaluation split. The train/test split is carved from uplift_train only.
- No continuous / monetary outcome is available (only binary `target`).

### Excluded from X (and why)
- `target` — post-treatment outcome — the label
- `first_issue_date` — raw timestamp — replaced by pre-treatment `tenure_days` / `issue_month`
- `first_redeem_date` — raw timestamp — POST-treatment for ~11.6% of clients; replaced by REF_DATE-censored `has_redeemed_pre` / `days_since_first_redeem`
- `client_uid` — native identifier — not a feature
- `treatment_flg` — treatment — kept as T, not a covariate