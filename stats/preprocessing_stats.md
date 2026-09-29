# Preprocessing Stats — What Changed, What's Cleaned, What's New

**Purpose:** a single reusable reference for explaining *what the preprocessing
did to each dataset* and *why the data is now better* for uplift / heterogeneous
treatment-effect modelling.

**Covers:** Hillstrom, X5 RetailHero, Lenta.
**Pipeline:** `src/causal_prep/` · regenerate with `python -m causal_prep.run all`.
**Seed:** `20240501` · **Rows dropped anywhere:** `0`.

---

## 1. TL;DR

| Question | Answer |
|---|---|
| Were any rows deleted? | **No.** Every raw row survives — missingness/outliers/duplicates/imbalance are flagged and handled, never dropped. |
| Was raw data modified? | **No.** `data/raw/` is read-only; all outputs are new files under `data/{interim,processed}/`, `artifacts/`, `reports/`. |
| Biggest correctness win | **Leakage removed / neutralised.** Post-treatment columns kept out of `X`; X5's future-dated `first_redeem_date` censored; redundant encodings dropped. |
| Biggest usability win | Each dataset now has an explicit **X / T / Y** contract, a fitted train-only preprocessor, a reproducible train/test split, and verified randomisation + overlap. |
| Is it model-ready? | **Yes** — `data/processed/<ds>/train.parquet` + `test.parquet` feed S/T/X/DR-learners and causal forests directly. No modelling done yet. |

---

## 2. At-a-glance: raw → processed

| | **Hillstrom** | **X5 RetailHero** | **Lenta** |
|---|---:|---:|---:|
| Domain | 3-arm e-mail RCT | Retail loyalty RCT | Grocery RCT |
| Raw rows | 64,000 | 200,039 ¹ | 687,029 |
| Raw columns | 12 | 7 ¹ | 195 |
| Rows after cleaning | 64,000 | 200,039 | 687,029 |
| **Rows dropped** | **0** | **0** | **0** |
| Cleaned-frame columns (pre-split) | 20 | 14 | 199 |
| **Model matrix `X`** | **13** | **10** | **344** |
| Treatment cols `T` | 4 (`treatment_arm`,`T`,`T_mens`,`T_womens`) | 2 (`treatment_arm`,`treatment_flg`) | 2 (`treatment_arm`,`T`) |
| Outcome cols `Y` | 3 (`visit`,`conversion`,`spend`) | 1 (`target`) | 1 (`response_att`) |
| Train / test | 51,200 / 12,800 | 160,031 / 40,008 | 549,623 / 137,406 |
| Extra scoring frame | — | 200,123 (unlabelled holdout) | — |
| Raw missing cells | 0 | 17,546 | 25,639,469 |
| Missing cells in `X` (post-transform) | **0** | **0** | **0** |
| Duplicate rows removed | 0 (7,634 kept, see §3) | 0 | 0 |
| Treatment split | 33/33/33 % | 50/50 % | 75/25 % |
| Primary-outcome base rate | 0.90 % (`conversion`) | 61.99 % (`target`) | 10.82 % (`response_att`) |
| Naive ATE (primary) | +0.68 pp / +0.31 pp ² | **+3.32 pp** | +0.75 pp |
| Covariate balance max \|SMD\| (train) | 0.011 | 0.011 | 0.028 |
| Propensity AUC (logreg / hgb) | 0.497 / 0.500 | 0.499 / 0.500 | 0.497 / 0.500 |
| Positivity mass outside [0.01, 0.99] | 0.000 | 0.000 | 0.000 |
| Reproducibility hash (test unit ids) | `58c31bcd96b5ace8` | `9573264c1085e3b6` | `71cfa82a34e59195` |

¹ X5 raw = `uplift_train.csv` (3 cols) ⨝ `clients.csv` (5 cols) → 7 distinct cols on 200,039 labelled clients.
² Mens-email vs control / Womens-email vs control.

---

## 3. Hillstrom — detail

### 3.1 Shape

| Stage | Rows | Cols | Note |
|---|---:|---:|---|
| Raw CSV | 64,000 | 12 | 0 missing cells |
| Cleaned (interim) | 64,000 | 21 | +9 engineered/canonical, +`split` |
| Processed `train` | 51,200 | 22 | `X`(13) + `T`(4) + `Y`(3) + `customer_uid` + `split` |
| Processed `test` | 12,800 | 22 | same schema |

### 3.2 What was cleaned

| Issue | Raw state | Action | Result |
|---|---|---|---|
| Category spelling | `zip_code = "Surburban"` | Relabel → `"Suburban"` | Cosmetic only; membership unchanged |
| Redundant encoding | `history_segment` (7 bins) **is** a deterministic non-overlapping bucketing of `history` | Excluded from `X`; kept as ordinal `history_segment_ord` for grouped diagnostics | No collinear, information-losing feature in `X` |
| Skew | `history` right-skewed (P99 ≈ 13× median), min 29.99 | Add `history_log1p`; **no winsorising** | Linear/propensity models get a tame scale; tails preserved for heterogeneity |
| Exact-duplicate rows | 7,634 rows share a full feature vector with another row | **Kept.** Spread evenly across arms (2,575 / 2,555 / 2,504); 0 conversions among them; no ID to prove identity | ~12 % of sample retained; no bias toward high-value customers |
| Rare outcome | `conversion` = 0.90 % positive | **No SMOTE / no resampling**; stratified split on `treatment_arm × conversion` | True base rate preserved for Qini/AUUC |
| Missing values | none | none needed | — |

### 3.3 What's new (engineered / canonical columns)

| New column | Type | Built from | Why |
|---|---|---|---|
| `customer_uid` | id (surrogate) | row index | Hillstrom has no ID; needed for join/trace, never a feature |
| `treatment_arm` | categorical | `segment` | Readable 3-arm label (`control`/`mens_email`/`womens_email`) |
| `T` | binary | `segment` | Any-email vs none — standard binary uplift |
| `T_mens`, `T_womens` | binary | `segment` | One-vs-rest / multi-arm analysis |
| `history_log1p` | numeric | `log1p(history)` | Skew control for linear learners |
| `bought_both` | binary | `mens & womens` | `mens × womens` interaction a linear model can't form itself |
| `history_segment_ord` | ordinal 1–7 | `history_segment` | Grouped balance/overlap tables (not a feature) |
| `split` | label | stratified sampler | Reproducible train/test membership |

### 3.4 Excluded from `X` (leakage / redundancy control)

| Column | Category | Reason |
|---|---|---|
| `visit` | post-treatment outcome (mediator) | Caused by the e-mail (e-mail → visit → purchase); 2-week window |
| `conversion` | post-treatment outcome (primary) | The label |
| `spend` | post-treatment outcome (monetary) | 2-week window; `spend>0 ⟺ conversion=1` |
| `history_segment`, `history_segment_ord` | redundant | Deterministic function of `history` |
| `segment` | raw treatment | Mapped to `treatment_arm` / `T` |
| `customer_uid` | identifier | Zero information; would memorise rows |

### 3.5 Change vs the earlier Phase-1 version

| | Phase-1 | Now | Why |
|---|---|---|---|
| `X` columns | 14 | **13** | Removed `mw_count` (= `mens + womens`) — a pure linear combination of two features already in `X`, i.e. redundant and collinear. `bought_both` kept. |
| Cleaned frame location | `data/processed/hillstrom/hillstrom_clean.csv` | `data/interim/hillstrom/hillstrom_clean.parquet` | It's a pre-encoding frame, not a model matrix |
| Fitted artifacts | mixed into `data/processed/` | `artifacts/hillstrom/` | Separate regenerable data from artifacts |

Everything else (treatment/outcome/covariate definitions, split policy, diagnostics) is unchanged.

---

## 4. X5 RetailHero — detail

### 4.1 Shape

| Stage | Rows | Cols | Note |
|---|---:|---:|---|
| Raw (`uplift_train` ⨝ `clients`) | 200,039 | 7 | 17,546 missing (`first_redeem_date`) |
| Cleaned (interim) | 200,039 | 15 | +8 engineered/canonical, +`split` |
| Processed `train` | 160,031 | 15 | `X`(10) + `T`(2) + `Y`(1) + `client_uid` + `split` |
| Processed `test` | 40,008 | 15 | same |
| Processed `score` | 200,123 | 11 | unlabelled competition holdout (`X` + id only) |

### 4.2 What was cleaned

| Issue | Raw state | Action | Result |
|---|---|---|---|
| **Temporal leakage** | `first_redeem_date` runs to **2019-11-20**; campaign/enrolment boundary is **2019-03-15**. **23,268 clients (11.6 %)** redeemed *after* the campaign | **Censor at `REF_DATE = 2019-03-16`**: redemptions on/after that date treated as "never redeemed" | All redeem-derived features are strictly pre-treatment |
| Invalid redemptions | 245 clients have `first_redeem_date < first_issue_date` | Folded into `redeem_info_missing = 1` | Impossible values neutralised, rows kept |
| Invalid age | `age` range **[-7491, 1852]**; 885 out-of-range in RCT frame | Clip to [14, 99] → NaN, add `age_invalid` flag, median-impute on **train** | Plausible values only; invalidity is itself a signal |
| Unknown gender | `gender ∈ {F, M, U}`, `U` = 46 % | Map `U`/NaN → `"unknown"`, keep as its own level | No information discarded |
| Raw timestamps | `first_issue_date`, `first_redeem_date` | Replace with `tenure_days`, `issue_month`, `has_redeemed_pre`, `days_since_first_redeem` (censored) | Model-usable numerics; raw datetimes excluded |
| Missing `first_redeem_date` | 17,546 (never redeemed) | `has_redeemed_pre = 0`, `redeem_info_missing = 1`, `days_since_first_redeem` imputed | Informative missingness preserved via flags |
| Unusable table | `products.csv` (43,038 rows) | **Excluded entirely** — not joinable without the absent `purchases` table | No dangling/irrelevant columns |
| Provided test set | `uplift_test.csv` has **no labels** | Transformed to `score.parquet`; train/test split carved from `uplift_train` only | Evaluation stays honest |
| Class balance | `target` = 62 % | No resampling | True base rate kept |

### 4.3 What's new

| New column | Type | Built from | Why |
|---|---|---|---|
| `client_uid` | id | rename of `client_id` | Consistent naming; not a feature |
| `treatment_arm` | categorical | `treatment_flg` | Readable `control`/`treated` label |
| `age_invalid` | binary | `age ∉ [14,99]` or missing | Missing/implausible-age signal |
| `tenure_days` | numeric | `REF_DATE − first_issue_date` | Pre-treatment loyalty tenure |
| `issue_month` | numeric 1–12 | `first_issue_date.month` | Enrolment-cohort seasonality |
| `has_redeemed_pre` | binary | valid redemption ≤ `REF_DATE` | Pre-treatment engagement |
| `days_since_first_redeem` | numeric | `REF_DATE − first_redeem_date` (censored) | Recency of first redemption |
| `redeem_info_missing` | binary | redemption absent / post-REF / invalid | Carries the "no usable redeem info" signal after imputation |

### 4.4 Excluded from `X`

| Column | Category | Reason |
|---|---|---|
| `target` | post-treatment outcome | The label |
| `first_issue_date` | raw timestamp | Replaced by `tenure_days` / `issue_month` |
| `first_redeem_date` | mixed pre/post-treatment timestamp | 11.6 % post-date the campaign → censored & replaced |
| `treatment_flg` | treatment | Kept as `T`, not a covariate |
| `client_uid` | identifier | Not a feature |
| all of `products.csv` | external dictionary | Not joinable without `purchases` |

### 4.5 Quality metrics

| Metric | Before | After |
|---|---|---|
| Rows with post-treatment `first_redeem_date` used as-is | 23,268 (leak) | **0** (censored) |
| `age` values outside [14, 99] | 885 | 0 (→ NaN → imputed, flagged) |
| Missing cells in `X` | n/a | **0** |
| Covariate balance max \|SMD\| | — | 0.011 (all < 0.10 flag) |
| Propensity AUC | — | 0.499 (RCT band) → randomisation confirmed |

---

## 5. Lenta — detail

### 5.1 Shape

| Stage | Rows | Cols | Note |
|---|---:|---:|---|
| Raw CSV.GZ | 687,029 | 195 | 25.6 M missing cells; **no ID column**, no timestamps |
| Cleaned (interim) | 687,029 | 200 | +4 engineered/canonical, +`split` |
| Processed `train` | 549,623 | 349 | `X`(344) + `T`(2) + `Y`(1) + `client_uid` + `split` |
| Processed `test` | 137,406 | 349 | same |

### 5.2 What was cleaned

| Issue | Raw state | Action | Result |
|---|---|---|---|
| **Post-treatment columns** | `response_sms`, `response_viber` — fractional per-channel delivery values recorded *during* the campaign, ill-defined for control | **Excluded from `X` and from the outcome set** | No campaign-execution leakage; `response_att` is the sole label |
| **Massive structural missingness** | 153 columns have missing values; **113 > 5 % missing, 60 > 30 %, max 72 %**. `k_var_*` / `stdev_*` are undefined at low basket counts | Median-impute on **train** + append **150 train-fitted missing-indicators** (`SimpleImputer(add_indicator=True)`) | "Insufficient activity" signal preserved; 0 missing in `X`; no rows dropped |
| Invalid age | `age` min 0; 335 out-of-range + 11,765 already missing (12,100 total) | Clip [14, 99] → NaN, `age_invalid` flag, median-impute on train | Plausible ages only; invalidity flagged |
| Cyrillic / unknown gender | `gender ∈ {Ж, М, Не определен, NaN}` | Map → `{F, M, unknown}`, NaN → `unknown` | ASCII, explicit unknown level (~1.4 %) |
| No ID column | none in source | Surrogate `client_uid` = row index | Traceable units; dedup limited to full-row identity (0 found) |
| No timestamps | none | — | No temporal split possible; window aggregates are pre-campaign by construction |
| Unequal allocation | `group` = 75 % test / 25 % control | Kept as-is (dataset's holdout design); stratified split on `group × response_att` | Both arms proportionally represented in train & test |
| Class balance | `response_att` = 10.8 % | No resampling | True base rate kept |

### 5.3 What's new

| New column(s) | Type | Built from | Why |
|---|---|---|---|
| `client_uid` | id (surrogate) | row index | No ID in source |
| `treatment_arm` | categorical | `group` | Readable `control`/`treated` label |
| `T` | binary | `group == "test"` | Standard binary treatment |
| `age_invalid` | binary | `age ∉ [14,99]` or missing | Missing/implausible-age signal |
| `<feature>_missingindicator` ×150 | binary | train missingness of each numeric covariate | Preserves structural-NaN signal after imputation |

### 5.4 `X` column accounting (344)

| Block | Count |
|---|---:|
| Numeric behavioural + demographic covariates (pre-campaign windows) | 189 |
| Train-fitted missing-indicators | 150 |
| `gender` one-hot (`F`, `M`, `unknown`) | 3 |
| `main_format` (binary) | 1 |
| `age_invalid` (binary) | 1 |
| **Total** | **344** |

### 5.5 Excluded from `X`

| Column | Category | Reason |
|---|---|---|
| `response_att` | post-treatment outcome | The label |
| `response_sms`, `response_viber` | post-treatment (channel delivery) | Recorded during the campaign; ill-defined for control |
| `group` | raw treatment | Mapped to `treatment_arm` / `T` |
| `client_uid` | identifier | Not a feature |

### 5.6 Quality metrics

| Metric | Before | After |
|---|---|---|
| Missing cells (total) | 25,639,469 | 0 in `X` (imputed + flagged) |
| Numeric covariates with >5 % missing | 113 | 0 unusable (all imputed; missingness retained as 150 indicators) |
| Post-treatment columns reachable by a model | 3 (`response_att/sms/viber` all present) | 0 (all excluded from `X`) |
| Covariate balance max \|SMD\| across ~190 covariates | — | 0.028 (all < 0.10 flag) |
| Propensity AUC | — | 0.497 (RCT band) → randomisation confirmed |

---

## 6. Cross-dataset: why the data is *better* now

| Dimension | Before (raw) | After (processed) | Benefit for the project |
|---|---|---|---|
| **Causal structure** | Columns mixed together; treatment/outcome/covariate roles implicit | Explicit `X` / `T` / `Y` with a written reason for every column (`reports/<ds>/feature_classification.csv`) | Meta-learners & causal forests can be pointed straight at `X`, `T`, `Y` |
| **Treatment leakage** | Raw treatment label only | Canonical `T` (binary + arm label + one-vs-rest); arms never collapsed | Multi-arm and binary uplift both supported from one file |
| **Outcome leakage** | Outcomes sit next to covariates | All post-treatment outcomes excluded from `X` | Estimated effects are not inflated by the outcome itself |
| **Post-treatment leakage** | X5 `first_redeem_date` future-dated; Lenta `response_sms/viber` present | X5 censored at `REF_DATE`; Lenta channel columns dropped | CATE reflects the treatment decision point, not hindsight |
| **Temporal leakage** | X5 had a hidden time axis in the redeem date | Handled via censoring; other datasets have no time axis | No "peeking into the future" via dates |
| **Redundant features** | Hillstrom `history_segment` (bin of `history`); `mw_count` (sum of two flags) | Both excluded from `X` | Cleaner design matrix, less collinearity for linear learners |
| **Missing data** | X5 & Lenta had large gaps | Median-impute on **train only** + missing-indicators; **0 rows dropped** | Full sample retained; "missing" kept as information |
| **Invalid values** | Ages of -7491, 1852, 0 | Clipped → NaN → imputed, with an `age_invalid` flag | No absurd values driving splits; the anomaly is still learnable |
| **Duplicates** | Hillstrom 7,634 look-alike rows | Investigated, shown coincidental, **kept** | No silent 12 % sample loss / value bias |
| **Class imbalance** | conversion 0.9 %, response 10.8 % | **No SMOTE / no resampling**; stratified split | Qini/AUUC and policy value stay calibrated to reality |
| **Train/test discipline** | none | Stratified on `treatment × rare-outcome`, seeded, hashed | Reproducible, leakage-free evaluation; preprocessors fitted on train only |
| **Randomisation evidence** | assumed | Verified: max \|SMD\| ≤ 0.028, propensity AUC ≈ 0.50, full overlap | Unconfoundedness & positivity are demonstrated, not asserted |
| **Reusability** | one-off | `preprocessor.joblib` + `feature_spec.json` per dataset; `python -m causal_prep.run all` | Anyone can regenerate or apply the exact transform to new data |

---

## 7. Output inventory

### Model-ready data — `data/processed/<ds>/`

| File | Hillstrom | X5 | Lenta | Contents |
|---|---:|---:|---:|---|
| `train.parquet` | 1.04 MB | 2.48 MB | 150.8 MB | `X` (numeric) + `T` cols + `Y` cols + unit id + `split` |
| `test.parquet` | 0.30 MB | 0.69 MB | 42.5 MB | same schema |
| `train_scaled.parquet` | 1.13 MB | 2.48 MB | 166.2 MB | numeric `X` standardised (train-fit) — for linear S/T/X/DR learners |
| `test_scaled.parquet` | 0.33 MB | 0.69 MB | 49.2 MB | same |
| `{train,test}.csv` | ✅ | ✅ | — (skipped >100k rows) | plain-text mirror |
| `score.parquet` | — | 2.96 MB | — | X5 unlabelled holdout, transformed |

### Cleaned pre-encoding frames — `data/interim/<ds>/<ds>_clean.parquet`
All rows, human-readable, `X` + `T` + `Y` + ids + `split` (Hillstrom 64,000×21 · X5 200,039×15 · Lenta 687,029×200).

### Fitted artifacts — `artifacts/<ds>/`

| File | Hillstrom | X5 | Lenta |
|---|---:|---:|---:|
| `preprocessor.joblib` (ColumnTransformer, train-fitted) | 4.9 KB | 4.6 KB | 32.7 KB |
| `scaler.joblib` (StandardScaler + column list) | 1.0 KB | 1.0 KB | 14.6 KB |
| `feature_spec.json` (`X`/`T`/`Y` lists, exclusions, counts, provenance) | 2.6 KB | 2.6 KB | 20.7 KB |

### Reports — `reports/<ds>/`
`audit.json` · `data_quality.md` · `feature_classification.csv` · `balance_overlap.md` · `validation.json` · `figures/propensity_overlap.png` · `figures/love_plot.png`
Master: `reports/causal_preprocessing_overview.md`.

---

## 8. How to reuse / regenerate

```bash
pip install -e .                     # pinned deps (pyproject.toml)
bash scripts/fetch_lenta.sh          # one-time, ~145 MB
python -m causal_prep.run all        # ~1m45s; deterministic (seed 20240501)
# or one dataset:
python -m causal_prep.run x5
```

To apply the **exact fitted transform** to new raw rows:

```python
import joblib, pandas as pd
from causal_prep.datasets import x5
ct = joblib.load("artifacts/x5/preprocessor.joblib")
X_new = ct.transform(x5.clean(new_raw_df))   # same columns as feature_spec.json["X_features"]
```

Numbers in this file come from `reports/<ds>/audit.json` and
`reports/<ds>/validation.json`; re-run the pipeline to refresh them.
