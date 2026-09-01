# Causal-Safe Preprocessing — Overview & Decisions

**Project:** Causal Decision Intelligence using Uplift Modeling
**Scope of this document:** repository inspection, causal problem definition,
preprocessing decisions, validation results, and remaining limitations for the
**primary dataset (Hillstrom / MineThatData)**. Secondary datasets present in
the repo are audited at the end with go/no-go verdicts.
**Status:** preprocessing only — *no modeling performed*.

---

## 0. Repository inspection (what is actually here)

| Path | Contents | Role |
|---|---|---|
| `docs/dataset_guide.md` | Dataset playbook — designates **Hillstrom 64K** as the primary development dataset; Lenta / MegaFon / X5 as generalization; synthetic/IHDP/ACIC for causal truth; Criteo for scale. | Source of truth for dataset strategy |
| `docs/knowledge_base_1.md` | Platform spec: propensity → S/T/X/DR-learner + Causal Forest → 4 segments → Qini/AUUC → budget optimizer. Explicit guardrails: no post-treatment features, overlap is a gate, evaluation is causal-native. | Source of truth for method + guardrails |
| `datasets/phase 1 - Main Development/Kevin_Hillstrom_…2008.03.20.csv` | 64,000 × 12. The Hillstrom RCT. | **Primary — this pipeline** |
| `datasets/phase 1 - Main Development/Digital Marketing Campaign Dataset.xlsx` | 5,000 × 15. Not referenced anywhere in the docs. | Audited §7 — **not RCT-clean; fixture only** |
| `datasets/phase 1 - Main Development/x5-retail-hero-uplift-raw-data.csv` | 400,162 × 5. Byte-identical to `phase 2/.../clients.csv`. | X5 client table (duplicate copy) |
| `datasets/phase 2 - generalization/x5-retail-hero-uplift-raw-data/` | `clients.csv` (400,162), `products.csv` (43,039), `uplift_train.csv` (200,039), `uplift_test.csv`, `uplift_sample_submission.csv`. **No `purchases` table.** | Audited §7 — validation dataset, feature-limited |
| `datasets/phase 2 - generalization/*.ipynb` | Kaggle reference notebooks (MegaFon, Lenta) using `scikit-uplift` meta-learners. Not tied to local files; no preprocessing logic to reuse. | Context only |

No pre-existing preprocessing code exists in the repo.

---

## 1. Causal problem definition — Hillstrom

| Element | Value | Notes |
|---|---|---|
| **Unit of analysis** | Individual customer, one row. | No native ID. We attach `customer_uid` = raw row position — **metadata, never a feature**. |
| **Treatment `T`** | `segment` ∈ {`No E-Mail`, `Mens E-Mail`, `Womens E-Mail`} | Randomized 3-arm campaign. Kept natively as `treatment_arm` {control, mens_email, womens_email}; binary `T` = any e-mail vs none for standard uplift; `T_mens`, `T_womens` for one-vs-rest / multi-arm. **Arms never collapsed away** (per `docs/dataset_guide.md` §3.3). |
| **Outcomes `Y`** (post-treatment, 2-week window) | `visit` (0/1), `conversion` (0/1), `spend` (\$, continuous) | `visit` is an **intermediate outcome / mediator** (a visit is itself caused by the e-mail). `conversion` is the primary uplift label. `spend` is the ROI outcome, zero-inflated. Structural nesting confirmed in data: `{spend>0} ≡ {conversion=1} ⊂ {visit=1}` (0 violations). |
| **Pre-treatment covariates `X`** | `recency`, `history`, `mens`, `womens`, `newbie`, `zip_code`, `channel` (+ engineered `history_log1p`, `mw_count`, `bought_both`) | Historical 12-month customer snapshot — available at send time. |
| **Derived-redundant (excluded from X)** | `history_segment` | Deterministic 7-way bucketing of `history` — verified non-overlapping bin ranges. Kept as `history_segment_ord` for grouped diagnostics only. |
| **Identifiers / metadata** | `customer_uid`, `split` | Never in X. |
| **Post-treatment variables** | `visit`, `conversion`, `spend` | Targets only. Excluded from X — using any of them as a feature is post-treatment / mediator leakage. |
| **Potential leakage variables** | none beyond the outcomes | `mens`/`womens` are *prior-year purchase* flags (not the campaign, not gender), so they are legitimate pre-treatment covariates. |
| **Time / date structure** | none | Single cross-section of one campaign. Covariates are a pre-send snapshot; outcomes a fixed forward 2-week window. **Temporal leakage is structurally impossible within-dataset; a temporal split is not applicable.** |

### Data-quality summary (raw audit — `reports/hillstrom_audit.json`)

| Check | Result | Decision |
|---|---|---|
| Missing values | **0** across all 64,000 × 12 | No imputation needed. Imputers kept in the pipeline as no-ops for reuse on phase-2 datasets. |
| Exact-duplicate rows | 7,634 rows in duplicate groups; concentrated at `history = 29.99` (floor value) + low-cardinality combos; spread across arms `{W:2575, C:2555, M:2504}`; **0 conversions / 0 spend** among them | **Kept.** No ID to establish true identity; collisions are expected from coarse features; dropping them would delete ~12 % of the sample, preferentially remove low-value customers, and bias CATE toward high-value — with no offsetting benefit. Flagged as a limitation (§6). |
| Invalid values | recency all in [1,12]; `history` all ≥ 29.99 (> 0); `spend` all ≥ 0; binaries all {0,1} | Nothing to fix. |
| Outliers | `history` P99≈1,219, max 3,345; `spend` max 499 (zero-inflated: 578 non-zero) | **Not removed / not winsorized.** Right tails are genuine high-value customers — exactly the heterogeneity CATE must capture. `history_log1p` added for linear/propensity models that are sensitive to skew. |
| Class imbalance | conversion positive rate **0.90 %** (578/64,000); visit 14.7 % | **No SMOTE / no resampling.** Uplift metrics (Qini, AUUC) and honest ATE depend on the true base rate. Handle at model stage via stratified splits + class weights. |
| Source typo | `zip_code` value `Surburban` | Relabelled to `Suburban` — pure string fix, category membership unchanged. |
| Treatment balance | arms 21,306 / 21,307 / 21,387 (33.3 % each) | Consistent with randomization. |
| Randomization sanity | max \|SMD\| (e-mail vs control) = **0.009**; 5-fold propensity AUC = **0.497**; propensity support 0.649–0.708 (≈ P(e-mail)=2/3) | Clean RCT. No confounding adjustment required; propensity retained only as a diagnostic and as a DR-learner nuisance. |

### Naive (unadjusted) treatment effects — a sanity anchor for later models

| Contrast | Δ visit | Δ conversion | Δ spend (\$) |
|---|---:|---:|---:|
| Mens e-mail − control | +0.076 | **+0.0068** | +0.77 |
| Womens e-mail − control | +0.045 | **+0.0031** | +0.42 |

(Stable across train and test — see `reports/hillstrom_balance_overlap.md`.)

---

## 2. Feature classification table

Full machine-readable version: `reports/hillstrom_feature_classification.csv`.

| Feature | Category | Action | Reason |
|---|---|---|---|
| `customer_uid` | Identifier (surrogate) | **Exclude from X** (keep for joins) | Row-position id we attach because Hillstrom has none. Zero information; as a feature it would memorise rows. |
| `recency` | Pre-treatment covariate (numeric) | **Keep as-is** | Months since last purchase, 1–12, pre-send. Bounded, low skew — no transform. Prime effect-modifier. |
| `history` | Pre-treatment covariate (numeric) | **Keep** + derive `history_log1p` | Historical 12-month \$ spend, strictly positive, right-skewed. Raw for tree/uplift learners; log for linear/propensity. **Not winsorized.** |
| `history_segment` | Derived-redundant of `history` | **Exclude from X** (keep as `history_segment_ord` for EDA) | Deterministic non-overlapping bucketing of `history`. Collinear + information-losing. |
| `mens` | Pre-treatment covariate (binary) | **Keep** | Bought men's merchandise in prior 12 months. Pre-send behaviour (not gender). |
| `womens` | Pre-treatment covariate (binary) | **Keep** | Bought women's merchandise in prior 12 months. Not mutually exclusive with `mens` (6,448 have both). |
| `newbie` | Pre-treatment covariate (binary) | **Keep** | New customer in prior 12 months. Pre-send. |
| `zip_code` | Pre-treatment covariate (categorical) | **Fix typo → one-hot** (fit on train) | Urban / Suburban / Rural. K dummies, `handle_unknown='ignore'`. |
| `channel` | Pre-treatment covariate (categorical) | **One-hot** (fit on train) | Prior-year purchase channel: Phone / Web / Multichannel. K dummies. |
| `history_log1p` | Derived covariate (numeric) | **Keep** (engineered) | log1p(history). Skew reduction for linear models; monotone → no ranking distortion for trees. |
| `mw_count` | Derived covariate (0/1/2) | **Keep** (engineered, optional) | `mens + womens`. Cheap interpretable interaction proxy. |
| `bought_both` | Derived covariate (binary) | **Keep** (engineered, optional) | `mens AND womens`. Flags broad-basket shoppers. |
| `segment` | **Treatment** (raw) | **Map** → `treatment_arm` / `T` / `T_mens` / `T_womens` | The randomized 3-arm assignment. Native arms preserved; binary `T` provided. Never collapsed. |
| `visit` | **Outcome** — post-treatment (mediator) | **Target only — exclude from X** | Site visit in the 2 weeks after send. Caused by the e-mail → post-treatment / mediator leakage if used as X. |
| `conversion` | **Outcome** — post-treatment (primary) | **Target only — exclude from X** | Purchase in the 2 weeks after send. Primary uplift label. |
| `spend` | **Outcome** — post-treatment (monetary) | **Target only — exclude from X** | Revenue in the 2 weeks after send; zero-inflated, `spend>0 ⟺ conversion=1`. ROI outcome. |
| `history_segment_ord` | EDA helper (ordinal 1–7) | **Exclude from X** | Redundant with `history` for modelling; used for grouped balance/overlap tables. |
| `split` | Metadata | **Keep outside X** | Bookkeeping. |

**Final X (14 columns):** `recency, history, history_log1p, mens, womens, newbie,
mw_count, bought_both, zip_code_{Rural,Suburban,Urban},
channel_{Multichannel,Phone,Web}`.

---

## 3. Pipeline & leakage controls

```
Raw ─▶ Audit ─▶ Clean ─▶ Causal Feature Audit ─▶ Split (stratified) ─▶
Fit ColumnTransformer + Scaler on TRAIN ─▶ Transform train & test ─▶ Save ─▶ Validate
```

* **Raw is read-only** — nothing is ever written back to `datasets/`.
* **Fit-on-train-only:** one-hot categories, imputer statistics, and the optional
  `StandardScaler` are `.fit()` on the training split, then applied to both
  splits. Persisted as `preprocessor.joblib` / `scaler.joblib`.
* **Split:** 80/20, stratified on `treatment_arm × conversion` so the 0.9 %
  converter rate and the 3-arm balance are preserved in both folds. Seed
  `20240501`. Model selection should use k-fold *within* train.
* **Scaling is opt-in.** Tree-based CATE learners (Causal Forest, gradient-boosted
  meta-learners) neither need nor benefit from it, and it destroys the
  interpretability of the linear propensity diagnostic. The canonical matrix is
  unscaled; a `*_scaled` variant is provided for linear S/T/X/DR learners.
* **Positivity is a gate, not a footnote** (per `knowledge_base_1.md` guardrail 4)
  — diagnostics in §4.

---

## 4. Validation results (`reports/hillstrom_validation.json`)

### Integrity — all hard checks pass

| Check | Result |
|---|---|
| Row count conserved (train + test = 64,000) | ✅ |
| No `customer_uid` overlap between train and test | ✅ |
| Missing cells in X | **0** |
| Non-finite cells in X | **0** |
| Outcome / redundant column present in X | **none** |
| One-hot `zip_code_*` / `channel_*` row-sums = 1 | ✅ / ✅ |
| Train fraction | 0.80 |

### Treatment / control & outcome rates (train)

| Arm | n | visit | conversion | mean spend |
|---|---:|---:|---:|---:|
| control | 17,045 | 0.107 | 0.0057 | 0.69 |
| mens_email | 17,046 | 0.183 | 0.0126 | 1.47 |
| womens_email | 17,109 | 0.153 | 0.0088 | 1.12 |

Test split reproduces the same ordering and magnitudes (`hillstrom_balance_overlap.md`).

### Covariate overlap / positivity

* Max \|SMD\| (e-mail vs control, train) = **0.011** — 0 features above the 0.10 flag.
* 5-fold propensity AUC: **0.497** (logistic), **0.500** (gradient boosting).
* Propensity support entirely within ≈ [0.65, 0.73]; **0 %** mass outside the
  [0.01, 0.99] trim band.
* **Verdict:** strong overlap across the whole covariate space. CATE is
  identified everywhere; no low-overlap region needs flagging in the product.
* Figures: `reports/figures/hillstrom_propensity_overlap.png`,
  `reports/figures/hillstrom_love_plot.png`.

### Reproducibility

Deterministic given the seed. Test-set `customer_uid` fingerprint recorded
(`test_uid_set_sha256_16 = 5acb3bbca25c3571`); `run_all` regenerates every
artifact from the raw file.

### Causal-assumption ledger (per `knowledge_base_1.md` §8.1)

| Assumption | Status for Hillstrom |
|---|---|
| **SUTVA / no interference** | Plausible — customers are independent households; one e-mail per customer, no shared-treatment spillover mechanism. |
| **Positivity / overlap** | **Holds strongly** — see above. |
| **Unconfoundedness / ignorability** | **Holds by design** — treatment is randomized (balance + propensity AUC ≈ 0.5 confirm it). This is the dataset's key strength. |
| **Consistency / well-defined treatment** | Holds — "the campaign e-mail" is a single concrete intervention per arm. |
| **No post-treatment conditioning** | Enforced — `visit`/`conversion`/`spend` excluded from X; no filtering on any post-send variable. |

---

## 5. Deliverables index

| # | Deliverable | Location |
|---|---|---|
| 1 | Final processed dataset | `data/processed/hillstrom/` — `hillstrom_clean.csv` (all rows, readable), `train/test.parquet+.csv` (model matrix), `train/test_scaled.parquet` (optional), `feature_spec.json` |
| 2 | Reproducible preprocessing code | `preprocessing/` (`config.py`, `hillstrom_audit.py`, `hillstrom_preprocess.py`, `hillstrom_validate.py`, `run_all.py`, `README.md`), `requirements.txt` |
| 3 | Feature classification table | `reports/hillstrom_feature_classification.csv` (+ §2 here) |
| 4 | Preprocessing / data-quality report | `reports/hillstrom_data_quality_report.md`, `reports/hillstrom_audit.json` |
| 5 | Treatment/control & overlap diagnostics | `reports/hillstrom_balance_overlap.md`, `reports/hillstrom_validation.json`, `reports/figures/*.png` |
| 6 | Remaining uncertainties / causal limitations | §6 below |
| — | Fitted preprocessors | `data/processed/hillstrom/preprocessor.joblib`, `scaler.joblib` |

---

## 6. Remaining uncertainties & causal limitations

1. **No individual ground-truth ITE.** Hillstrom is a real RCT: `Y(1)` and `Y(0)`
   are never both observed for the same customer. Only population/subgroup
   effects and policy value are estimable. Report AUUC/Qini/policy value, **not**
   per-customer ITE error, on this dataset. Use the synthetic / IHDP / ACIC
   benchmarks (per `docs/dataset_guide.md`) for ITE-accuracy claims.
2. **No customer ID → "duplicate" rows are unresolved.** 7,634 rows share an
   identical feature vector with another row. We keep them as coincidental
   collisions (they carry 0 conversions and spread evenly across arms, so the
   impact on any estimate is negligible), but we cannot *prove* they are distinct
   customers. If a later data drop with IDs appears, re-audit.
3. **`visit` is a mediator, not a clean secondary outcome.** It sits on the
   causal path e-mail → visit → purchase. It is fine as a stand-alone uplift
   target, but must not be used as a covariate, and mediation analyses that
   condition on it need front-door / sequential-ignorability assumptions that are
   out of scope here.
4. **`spend` is zero-inflated** (99.1 % exact zeros). Treating it as a plain
   continuous outcome is valid for ATE, but CATE models on `spend` should expect
   heavy-tailed residuals; a two-part (hurdle) formulation or a
   conversion-conditional spend model may be needed later. Not decided at the
   preprocessing stage.
5. **Effect sizes are small relative to outcome variance** (conversion uplift
   ≈ 0.3–0.7 pp on a 0.9 % base). S-learner regularisation can shrink such
   effects toward zero — flagged for the modeling phase, not fixable here.
6. **External validity.** One retailer, one campaign, 2008, US. Generalization
   claims require the phase-2 datasets (Lenta / MegaFon / X5) — see §7.
7. **`recency` granularity.** Integer months only; finer recency (days) is not
   available, which caps the resolution of recency-driven heterogeneity.
8. **One-hot with all K levels** introduces exact collinearity in the intercept
   for unregularised linear models. The `*_scaled` matrix is intended for
   penalised/tree learners; drop a reference level per block if fitting plain OLS
   logistic regression.

---

## 7. Secondary datasets — audit & verdicts (not preprocessed here)

### 7.1 `Digital Marketing Campaign Dataset.xlsx` — ⚠️ not causally usable as-is

* 5,000 × 15. `user_id` 4,988 unique. Clean, no missing — looks **synthetic/toy**.
* **Not referenced in `docs/`** anywhere; not part of the documented suite.
* `treatment_exposed` is *exposure*, not random assignment — endogenous
  (exposed users differ systematically from non-exposed). Treating it as `T`
  invites selection bias.
* Severe **post-treatment columns**: `impressions`, `clicks`, `spend_usd`,
  `revenue_usd`, `roi` are all realised *after* exposure; `roi` is a
  deterministic post-hoc function of `revenue_usd`/`spend_usd`.
* Valid pre-treatment covariates would be only `channel`, `device`, `country`,
  `segment`, `prior_visits_30d`, `prior_spend_180d`.
* **Verdict:** use at most as a schema/pipeline **test fixture**. Do not report
  causal estimates from it without an explicit randomization/ignorability story.

### 7.2 X5 RetailHero (`phase 2 - generalization/x5-retail-hero-uplift-raw-data/`) — ✅ usable, feature-limited

* `uplift_train.csv`: 200,039 rows, `treatment_flg` ≈ 50/50 (99,981 / 100,058),
  `target` binary. Naive ATE ≈ **+3.3 pp** (63.7 % treated vs 60.3 % control).
* `clients.csv`: 400,162 rows. `age` has **invalid values** (min −7,491, max
  1,901) → needs range-clipping / to-missing (~a few hundred rows outside
  [10, 100]). `gender` ∈ {F, M, U} — `U` (46 %) is "unknown", keep as its own
  level. `first_redeem_date` missing for **35,469** clients (8.9 %) — informative
  ("never redeemed"), model as a flag; **do not drop**.
* **`purchases` table is absent** — the raw transaction history that X5's value
  proposition rests on (`docs/dataset_guide.md` §6.2) is not in the repo. Feature
  engineering is limited to demographics + card issue/redeem dates.
* **Temporal caution:** `first_redeem_date` can post-date the campaign; only the
  pre-campaign portion is a valid covariate. Needs the campaign date (not in the
  provided files) to adjudicate — treat as a flag + pre-campaign-only derived
  features until confirmed.
* **Verdict:** valid RCT for cross-domain generalization (a separate pipeline,
  mirroring this one, is warranted). Lower feature richness than documented until
  the purchases table is added.

### 7.3 MegaFon / Lenta

Only referenced via the phase-2 notebooks; **no local data files**. Out of scope
until the datasets are added. The notebooks confirm the intended modeling stack
(`scikit-uplift` S-/T-learners, Qini/AUUC) but contain no preprocessing logic to
port.

---

## 8. One-line summary

Hillstrom is a clean, well-powered 3-arm marketing RCT: randomization holds,
overlap is total, there is no missingness and no temporal axis to leak through.
The only real preprocessing risks are **post-treatment leakage** (handled:
`visit`/`conversion`/`spend` are targets only) and **redundant encoding**
(handled: `history_segment` dropped in favour of continuous `history`). The
dataset is ready for uplift / CATE estimation; individual-ITE accuracy claims
must be routed to the synthetic benchmarks instead.
