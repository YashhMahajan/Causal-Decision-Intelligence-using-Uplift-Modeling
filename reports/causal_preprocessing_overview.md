# Causal-Safe Preprocessing — Master Overview (Phase 1 + Phase 2)

**Project:** Causal Decision Intelligence using Uplift Modeling
**Deliverable state:** three datasets audited, causally preprocessed, validated,
documented, and modeling-ready for uplift / heterogeneous-treatment-effect
estimation. **No uplift models built.**

| Dataset | Domain | Rows | X cols | T | Y (primary) | Naive ATE | max\|SMD\| | Propensity AUC |
|---|---|---:|---:|---|---|---:|---:|---:|
| **Hillstrom** | E-mail retail RCT (3-arm) | 64,000 | 13 | `T` = any e-mail (+ `T_mens`,`T_womens`) | `conversion` | +0.0068 / +0.0031 | 0.011 | 0.497 |
| **X5 RetailHero** | Retail loyalty RCT | 200,039 | 10 | `treatment_flg` | `target` | +0.0332 | 0.011 | 0.499 |
| **Lenta** | Grocery RCT | 687,029 | 344 | `T` = communication | `response_att` | +0.0075 | 0.028 | 0.497 |

All three: randomization confirmed, positivity holds, no treatment/outcome/
post-treatment/temporal leakage, preprocessors fit on train only, splits
reproducible from a fixed seed.

---

## Part 1 — Phase-1 (Hillstrom) independent audit & corrections

Phase 1 was re-inspected against the data, not the summary. Findings:

### Verified correct (no change)
- **Treatment**: `segment` → 3 native arms preserved (`control` / `mens_email` /
  `womens_email`) plus binary `T` (any e-mail) and `T_mens` / `T_womens`. Not
  collapsed. ✔
- **Outcomes**: `visit`, `conversion`, `spend` — all post-treatment (2-week
  window), used as targets only, excluded from X. Nesting `{spend>0} ≡
  {conversion=1} ⊂ {visit=1}` holds exactly (0 violations). ✔
- **X**: only the historical 12-month snapshot (`recency`, `history`, `mens`,
  `womens`, `newbie`, `zip_code`, `channel`). `history_segment` excluded as a
  deterministic non-overlapping bucketing of `history`. ✔
- **Leakage**: no post-treatment column in X; no timestamp column so no temporal
  leakage path; surrogate `customer_uid` never a feature. ✔
- **Split**: 80/20 stratified on `treatment_arm × conversion`. Stratifying on the
  rare outcome preserves the (T, Y) cell sizes that Qini/AUUC variance depends on;
  it uses no feature information and does not leak. ✔
- **Fit-on-train**: one-hot categories, imputer statistics, optional scaler — all
  `.fit()` on train, applied to both splits; persisted to `artifacts/hillstrom/`. ✔
- **Duplicates**: 7,634 rows share a feature vector with another row. Spread
  evenly across arms, 0 conversions among them, no ID to disprove identity →
  kept as coincidental collisions (not data-entry errors). ✔
- **Positivity**: propensity ≈ 2/3 everywhere, AUC ≈ 0.5, 0 mass in the trim
  tails. ✔

### Corrected in Phase 2
| Issue | Phase-1 state | Fix | Justification |
|---|---|---|---|
| Redundant engineered feature | `mw_count` (= `mens + womens`) was in X | **Removed** `mw_count` | It is a pure linear combination of two features already in X and is constant-ish (every customer bought men's or women's, so it is 1 or 2) → zero new information, adds collinearity. `bought_both` (the `mens × womens` interaction, which a linear learner cannot form itself) is kept. |
| Repo layout | `preprocessing/` + fitted artifacts mixed into `data/processed/` | **Restructured** to `src/causal_prep/` + `data/{raw,interim,processed}` + `artifacts/` + `reports/<ds>/` | Required for Phase-2 consistency across 3 datasets; separates regenerable data from source and from small fitted artifacts. |
| Interim vs processed | cleaned full table sat in `data/processed/` | Moved to `data/interim/hillstrom/hillstrom_clean.parquet` | It is the pre-encoding cleaned frame, not a model matrix. |

Methodology was **not** otherwise changed. Hillstrom numbers are unchanged
except X drops from 14 → 13 columns (`mw_count` removed).

---

## Part 2 — Per-dataset causal setup

### 2A. Hillstrom (primary)
- **Unit**: customer, one row. Surrogate `customer_uid` (raw row index), metadata only.
- **T**: `segment`, 3 randomized arms; binary `T` = any e-mail vs none.
- **Y**: `visit` (mediator), `conversion` (primary), `spend` (monetary, zero-inflated).
- **X (13)**: `recency`, `history`, `history_log1p`, `mens`, `womens`, `newbie`,
  `bought_both`, `zip_code_{Rural,Suburban,Urban}`, `channel_{Multichannel,Phone,Web}`.
- **Time**: none. Covariates pre-send; outcomes fixed forward 2-week window →
  temporal leakage impossible, no temporal split.
- **Imbalance**: conversion 0.9% → no resampling; stratified split.
- **Cleaning**: `Surburban`→`Suburban` (relabel only); no winsorizing, no row drops,
  no imputation needed (0 missing).

### 2B. X5 RetailHero
- **Unit**: loyalty-card client, one row. Native `client_id` → `client_uid`.
- **T**: `treatment_flg` (a marketing communication), single binary arm, ≈ 50/50.
- **Y**: `target` (purchase in the promo period), binary. **No monetary outcome.**
- **X (10)**: `age` (clipped+imputed), `age_invalid`, `gender_{F,M,unknown}`,
  `tenure_days`, `issue_month`, `has_redeemed_pre`, `days_since_first_redeem`
  (censored+imputed), `redeem_info_missing`.
- **Temporal-leakage control (the key move)**: `first_redeem_date` runs to
  2019-11-20 while the campaign/enrolment boundary is 2019-03-15 (max
  `first_issue_date`). **11.6%** of redemptions post-date the campaign. All
  redeem-derived features are **censored at `REF_DATE` = 2019-03-16**: a
  redemption on/after that date is treated exactly like "never redeemed".
  `first_redeem_date` and `first_issue_date` raw datetimes are excluded.
- **Invalid values**: `age` has ~885 out-of-range values in the RCT frame
  (negatives, `1852`, …) → NaN + `age_invalid` flag + train-median impute. Rows
  kept.
- **Provided `uplift_test.csv` has no labels** → transformed to
  `data/processed/x5/score.parquet` for later submission scoring; **not** an
  evaluation split. Train/test (80/20) is carved from `uplift_train` only,
  stratified on `treatment_flg × target`.
- **`products.csv` unused**: not joinable to clients without the (absent)
  `purchases` transaction table. See limitations.
- **Imbalance**: `target` ≈ 62% → no resampling.

### 2C. Lenta
- **Unit**: grocery client, one row. **No ID column in the source** → surrogate
  `client_uid` (row index).
- **T**: `group` — `test` (treated, a communication) vs `control`. **Unequal
  allocation 75.1% / 24.9%** (the dataset's holdout design, not a defect).
- **Y**: `response_att` (attributed binary response). `response_sms` /
  `response_viber` are fractional per-channel campaign-delivery values recorded
  *during* the campaign and ill-defined for control → **excluded from X and from
  the outcome set** (post-treatment).
- **X (344)**: `age` (clipped+imputed), `age_invalid`, `gender_{F,M,unknown}`,
  `children`, `months_from_register`, `main_format`, ~183 pre-campaign
  purchase-behaviour aggregates (cheque counts, sale sums/counts, discount
  shares, coefficient-of-variation `k_var_*`, `stdev_*`, `crazy_purchases_*`,
  `*_share_*`) over 15d–12m windows, **plus ~150 train-fitted missing-indicators**.
- **Missingness (large & structural)**: 113 numeric covariates have >5% missing
  (max 72%). A `k_var_*` / `stdev_*` feature is *undefined* when the client had
  too few baskets in the window, so NaN carries an "insufficient activity"
  signal. Handled by **median impute + `SimpleImputer(add_indicator=True)`** (all
  fit on train). Rows are **not** dropped.
- **Time**: no timestamps → no temporal split; the window aggregates are
  pre-campaign by construction.
- **Imbalance**: `response_att` ≈ 10.8% → no resampling.
- **Scale**: the full 687k-row dataset is processed. A 10k stratified benchmark
  subsample (docs §4) is a one-liner on the processed train split if wanted.

---

## Part 3 — Validation (every dataset)

Machine-readable: `reports/<ds>/validation.json`. Human-readable:
`reports/<ds>/data_quality.md` and `reports/<ds>/balance_overlap.md`.

### Hard gates — all pass for all three
- row count conserved (train + test == cleaned frame)
- no unit-id overlap between train and test; unit-id unique
- **0** missing cells and **0** non-finite cells in the X block after transform
- **no** outcome / excluded / post-treatment column present in X
- every X column numeric; one-hot blocks row-sum to 1
- reproducibility: test-set unit-id fingerprint recorded and **stable across
  re-runs** — `hillstrom 58c31bcd96b5ace8`, `x5 9573264c1085e3b6`,
  `lenta 71cfa82a34e59195` (see `reports/<ds>/validation.json`)

### Balance / positivity
| Dataset | max\|SMD\| (train) | Propensity AUC (logreg / hgb) | Mass outside [0.01,0.99] | Verdict |
|---|---:|---:|---:|---|
| Hillstrom | 0.011 | 0.497 / 0.500 | 0.000 | Strong overlap; CATE identified everywhere |
| X5 | 0.011 | 0.499 / 0.500 | 0.000 | Strong overlap; CATE identified everywhere |
| Lenta | 0.028 | 0.497 / 0.500 | 0.000 | Strong overlap (propensity centred at 0.75); CATE identified |

All `|SMD|` well under the 0.10 flag; propensity AUCs in the RCT band
[0.45, 0.55]. Figures per dataset: `reports/<ds>/figures/propensity_overlap.png`,
`reports/<ds>/figures/love_plot.png`.

### Causal-assumption ledger
| Assumption | Hillstrom | X5 | Lenta |
|---|---|---|---|
| **SUTVA / no interference** | Plausible (independent households, 1 e-mail each) | Plausible (per-client SMS) | Plausible (per-client communication) |
| **Positivity / overlap** | Holds strongly | Holds strongly | Holds strongly (unequal but full-support allocation) |
| **Unconfoundedness** | By design (randomized; AUC≈0.5) | By design (randomized; AUC≈0.5) | By design (randomized; AUC≈0.5) |
| **Consistency** | "the campaign e-mail" is one concrete intervention | "the communication" is one intervention | "the communication" is one intervention |
| **No post-treatment conditioning** | Enforced (visit/conversion/spend out of X) | Enforced (`first_redeem_date` censored; `target` out of X) | Enforced (`response_*` out of X) |

---

## Part 4 — Deliverables index

| # | Deliverable | Location |
|---|---|---|
| 1 | Processed datasets (modeling-ready) | `data/processed/<ds>/train.parquet`, `test.parquet` (+ `*_scaled.parquet`; `x5/score.parquet`; CSV mirror for datasets ≤ 100k rows) |
| — | Cleaned pre-encoding frames | `data/interim/<ds>/<ds>_clean.parquet` |
| 2 | Reproducible preprocessing code | `src/causal_prep/` — `config.py`, `common.py`, `reporting.py`, `run.py`, `datasets/{hillstrom,x5,lenta}.py`; `pyproject.toml`; `scripts/fetch_lenta.sh` |
| 3 | Feature classification tables | `reports/<ds>/feature_classification.csv` |
| 4 | Preprocessing / data-quality reports | `reports/<ds>/data_quality.md`, `reports/<ds>/audit.json` |
| 5 | Treatment/control & overlap diagnostics | `reports/<ds>/balance_overlap.md`, `reports/<ds>/validation.json`, `reports/<ds>/figures/*.png` |
| 6 | Fitted preprocessors + spec | `artifacts/<ds>/preprocessor.joblib`, `scaler.joblib`, `feature_spec.json` |
| 7 | Remaining uncertainties / limitations | Part 5 below |

Regenerate everything: `python -m causal_prep.run all` (Hillstrom ~30 s, X5 ~20 s,
Lenta ~2 min).

---

## Part 5 — Remaining uncertainties & causal limitations

### Cross-cutting
1. **No individual ground-truth ITE** in any of the three (all real RCTs):
   `Y(1)` and `Y(0)` are never both observed for a unit. Report population /
   subgroup effects, uplift ranking (Qini/AUUC), and policy value — **not**
   per-unit ITE error. Route ITE-accuracy claims to the synthetic / IHDP / ACIC
   benchmarks (`docs/dataset_guide.md` §7–9), which are not yet in the repo.
2. **One-hot keeps all K levels** → exact collinearity for an unpenalised linear
   model. The `*_scaled` matrices target penalised / tree learners; drop one
   reference dummy per block for plain logistic/OLS.
3. **Small effects.** Hillstrom conversion uplift ≈ 0.3–0.7 pp on a 0.9% base;
   Lenta ≈ 0.75 pp on 10.8%. S-learner regularisation can shrink these toward
   zero — a modeling-phase concern.

### Hillstrom
4. `visit` is a **mediator** (e-mail → visit → purchase): fine as a stand-alone
   uplift target, never a covariate; mediation analysis needs extra assumptions.
5. `spend` is **99.1% exact zeros** → treat as continuous for ATE, but CATE on
   spend likely needs a two-part / hurdle model.
6. The 7,634 coincidental duplicates cannot be *proved* distinct (no ID); impact
   on estimates is negligible (0 conversions, arm-balanced) but noted.

### X5
7. **Feature-poor.** The `purchases` transaction table — X5's entire "rich retail"
   value proposition (`docs/dataset_guide.md` §6.2) — is **not in the repo**. X
   is limited to demographics + two loyalty-card dates (10 columns). Adding
   `purchases` later would roughly 10–50× the covariate space and warrants
   re-running this pipeline.
8. **`REF_DATE = 2019-03-16` is inferred** from `max(first_issue_date)`, not
   documented. If the true campaign send date differs, the redeem-censoring
   boundary should move with it. The direction of caution is correct (censoring
   can only *remove* leakage), but tenure/recency magnitudes depend on it.
9. `days_since_first_redeem` is median-imputed for the 79%→ / never-redeemed
   rows; the `has_redeemed_pre` + `redeem_info_missing` flags carry the signal,
   but a model that ignores the flags will see a spurious mode at the median.
10. Provided `uplift_test.csv` cannot be used for evaluation (no labels).

### Lenta
11. **Anonymised behavioural features** (`k_var_count_per_cheque_3m_g34`, …):
    causally valid as pre-treatment history, but low interpretability for SHAP /
    business narrative. Group-level (`g24`, `g34`, …) meanings are unknown.
12. **~150 missing-indicators** roughly double the matrix width (344 columns).
    They are information-preserving and can be dropped by the modeler; several
    will be near-constant.
13. **No monetary outcome** (only binary `response_att`) → no incremental-revenue
    / ROI optimisation from Lenta; use Hillstrom `spend` for that.
14. **No `CardHolder` / ID column** in this file (the reference notebook assumes
    one). Deduplication is limited to full-row identity (0 found).
15. **Unequal 75/25 allocation** inflates control-arm variance; effective sample
    for the control counterfactual is ~171k, not ~344k.
16. `response_sms` / `response_viber` were discarded entirely. If a future
    analysis wants channel-specific treatment effects, they would need careful
    re-derivation (they are fractions, not clean binaries, and populated oddly
    for control).

### Not addressed (out of Phase-2 scope)
- MegaFon: no local data (only a reference notebook). Synthetic CATE / IHDP /
  ACIC / Criteo: not in the repo. `Digital Marketing Campaign Dataset.xlsx`:
  synthetic, exposure ≠ randomization, post-treatment columns — **not** a valid
  causal dataset; left as a schema fixture only.
