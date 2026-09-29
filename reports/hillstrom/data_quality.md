# hillstrom — Preprocessing & Data-Quality Report

_Seed 20240501; stratified split on `treatment_arm , conversion`; test fraction 0.2._

## 1. Shape & missingness

- cleaned frame: **64000 rows × 20 cols**
- total missing cells (post-clean): **0**

## 2. Duplicates & invalid values

- unit-id duplicates: **0**
- full-row duplicates (extra): **0**
- range checks (count outside allowed range): `recency`=0, `history`=0, `spend`=0

## 3. Treatment & outcomes (cleaned frame)

- treatment `T` — arm counts `{'womens_email': 21387, 'mens_email': 21307, 'control': 21306}`; P(treated) = 0.6671
- outcome rates by arm:
  - **control**: visit 0.1062, conversion 0.0057, spend 0.6528
  - **mens_email**: visit 0.1828, conversion 0.0125, spend 1.4226
  - **womens_email**: visit 0.1514, conversion 0.0088, spend 1.0772
- naive (unadjusted) ATE vs control:
  - **mens_email_vs_control**: visit +0.0766, conversion +0.0068, spend +0.7698
  - **womens_email_vs_control**: visit +0.0452, conversion +0.0031, spend +0.4244
- binary-outcome base rates: `visit` 0.1468, `conversion` 0.0090  → **no resampling / SMOTE applied**

## 4. Randomization sanity

- max |SMD| (treated vs control) = **0.0088** (flag 0.1); features above flag: 0
- 5-fold propensity AUC = **0.4962** (n=64000); support [0.6483, 0.7319]; mass outside trim 0.0000

## 5. Dataset-specific audit

```json
{
  "coincidental_duplicates": {
    "rows_in_duplicate_groups": 7634,
    "conversions_among_them": 0,
    "by_arm": {
      "womens_email": 2575,
      "control": 2555,
      "mens_email": 2504
    },
    "x_vectors_spanning_multiple_arms": 380,
    "decision": "kept \u2014 coincidental, not data-entry errors"
  },
  "history_segment_is_deterministic_bin_of_history": true,
  "outcome_nesting": {
    "conversion1_and_spend0": 0,
    "spend_pos_and_conversion0": 0,
    "conversion1_and_visit0": 0
  }
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
- `onehot_zip_code_rowsum_ok` = `True`
- `onehot_channel_rowsum_ok` = `True`

## 7. Reproducibility

- `test_unit_id_sha256_16` = `58c31bcd96b5ace8`
- `seed` = `20240501`
- `note` = `Deterministic given seed; re-run to confirm.`

## 8. Causal-safety decisions

- Genuine 3-arm RCT; randomization verified (max|SMD|~0.01, propensity AUC~0.5).
- No timestamp column: covariates are a pre-send 12-month snapshot, outcomes a fixed forward 2-week window -> temporal leakage structurally impossible; no temporal split applicable.
- 7,634 rows share an identical feature vector with another row; no customer id exists to prove identity. They spread evenly across arms and carry 0 conversions -> kept as coincidental collisions (no row dropping).
- conversion rate ~0.9% -> NO resampling/SMOTE; stratified split on treatment_arm x conversion preserves the rare cell in both folds.
- history / spend right tails are genuine high-value customers -> NOT winsorized.
- `visit` is available as a stand-alone uplift target but is a mediator; never a feature.

### Excluded from X (and why)
- `visit` — post-treatment outcome (2-week window) and a mediator on e-mail->visit->purchase
- `conversion` — post-treatment outcome — the primary label
- `spend` — post-treatment outcome — zero-inflated; spend>0 iff conversion=1
- `history_segment` — deterministic non-overlapping bucketing of `history` — collinear, information-losing
- `history_segment_ord` — ordinal recoding of the above; kept for grouped diagnostics only
- `segment` — raw treatment label — mapped to treatment_arm / T
- `customer_uid` — surrogate identifier — zero information, would memorise rows