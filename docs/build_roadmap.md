# Build roadmap — what to build next, in order, in detail

**Baseline (done):** `src/causal_prep` (teammates) produces audited, leakage-safe, model-ready data for Hillstrom, X5 and Lenta:
`data/processed/<ds>/{train,test}[_scaled].parquet`, `artifacts/<ds>/feature_spec.json`, `reports/<ds>/`.
Regenerate: `python -m causal_prep.run all`. It is the **single source of truth for data** — do not add another preprocessing path.
Shared utilities kept in `src/`: `ingestion/validate.py` (quality report, naive ATE + CI), `eda/balance.py` (SMD), `preprocessing/splits.py` (train/val/test stratified on treatment x outcome).

**Rules for every step (design doc §8/§9):** nothing is "done" until its verification gate passes; features come only from `feature_spec.json` (never infer from column names); evaluation is causal-native (Qini/AUUC/policy value), never accuracy/F1; log every run to MLflow from the first learner on.

**Known facts that shape every step**
- Hillstrom conversion is rare (0.9%, +0.5pp naive effect) -> noisy CATE; bootstrap everything. Treated share is 0.667, not 0.5.
- Hillstrom has no cost data: ROI uses an *assumed* cost/contact (`ASSUMED_COST_PER_CONTACT`, $0.10) and must be reported next to any ROI number.
- X5 has only demographic/loyalty features (no `purchases.csv`); Lenta is large (687K, 190 covariates, heavy missingness). MegaFon, Synthetic, IHDP, ACIC, Criteo are not in the repo.
- Splits: `causal_prep` is 80/20. Model selection needs a validation set: carve `val` out of train with `assign_splits` (or CV), and touch test once.

---

## Step 1 — Get the remaining data
**Why first:** generalization and true-ITE validation depend on it. **Needs your approval** (downloads).
- **MegaFon** (10K benchmark): `sklift.datasets.fetch_megafon`; add `src/causal_prep/datasets/megafon.py` filling a `DatasetSpec` like the others (treatment col `treatment_group`, outcome `conversion`, features `X_1..X_50`, all numeric, already standardized). Run through `causal_prep.run`.
- **Synthetic CATE**: write our own generator `src/data/synthetic.py` rather than depend on an external repo: covariates X~N(0,1)^d, propensity e(x) (constant for RCT, or logistic for observational variant), baseline mu0(x), effect tau(x) (nonlinear, includes a negative-effect region so Sleeping Dogs exist), outcome noise. Return `tau` as ground truth. Provide RCT and confounded variants, n=2K and 20K, seeded.
- **IHDP** (672x25, semi-synthetic, has `mu0/mu1`): load from a public copy; use for quick estimator debugging.
- **X5 purchases.csv** (optional, large): enables RFM/basket features; add to `x5.py` with the same REF_DATE censoring so no purchase after the campaign date enters X.
- **Gate:** each new dataset passes `causal_prep` integrity checks; synthetic tau has known mean/variance recorded in a test.

## Step 2 — EDA and propensity/overlap diagnostics
**Build:** `src/eda/` (plots + auto insights) and `src/propensity/` (model + diagnostics).
- EDA: treatment/outcome rates per arm, feature distributions by arm, correlation matrix, outcome-by-segment tables; auto-insight strings (e.g. flag any feature with |SMD|>0.1). Save figures to `reports/<ds>/figures/`.
- Propensity: logistic regression + gradient boosting (`P(T=1|X)`), cross-fitted so predictions are out-of-fold. Outputs: overlap histogram by arm, propensity AUC, Love plot (SMD before/after weighting), share of units outside [0.01, 0.99].
- Expected on RCT data: AUC ~0.5 and flat propensity. If not, something is wrong with the data or the leakage guards.
- Return the propensity as a reusable object (`fit`, `predict`, `clip`) — DR/X-learners consume it. For Hillstrom binary treatment pass the *known* 0.667 as an alternative.
- **Gate:** overlap/positivity is reported per dataset and low-overlap units are flagged (design doc §9.4).

## Step 3 — Baselines (non-causal reference)
**Build:** `src/evaluation/baselines.py`.
- Predictive model of `P(Y|X)` (GBM), ranked by predicted response — the "who is likely to buy" approach the project argues against.
- Policy baselines used later: **random targeting**, **target everyone**, **propensity-only** ranking.
- **Gate:** baseline metrics exist and are stored before any learner is trained.

## Step 4 — Meta-learners and Causal Forest
**Build:** `src/learners/{s_learner,t_learner,x_learner,dr_learner,causal_forest}.py` behind one interface: `fit(X, T, Y, propensity=None)`, `predict_cate(X)`, plus bootstrap CIs. Base learners pluggable (LightGBM / sklearn GBM / logistic).
- **S**: one model on [X, T]; CATE = f(X,1) - f(X,0). Weak for small effects — expected on Hillstrom.
- **T**: two outcome models per arm; CATE = mu1 - mu0.
- **X**: T-learner imputed effects, then two effect models, blended with the propensity. Useful for Hillstrom's 2:1 imbalance.
- **DR**: cross-fitted outcome + propensity models; pseudo-outcome regression. Requires out-of-fold nuisance predictions.
- **Causal Forest**: `econml` CausalForestDML (add `econml`, `lightgbm`, `mlflow` to requirements); provides intervals.
- Multi-arm Hillstrom: run per-arm-vs-control (men's, women's) as well as "any email".
- Wire MLflow now: params, metrics, model artifacts per run.
- **Gate:** mean predicted CATE must match the naive ATE (within its CI) on the RCT datasets; if a learner disagrees, understand why before continuing. Also validate against synthetic + IHDP with PEHE, ATE bias, ITE correlation.

## Step 5 — Causal evaluation and model selection
**Build:** `src/evaluation/{qini,auuc,curves,policy,bootstrap}.py`.
- Uplift curve, Qini curve/coefficient, AUUC, uplift@k, calibration by CATE decile (predicted vs observed treated-minus-control within decile).
- Policy value: estimated outcome if we treat top-k% by CATE, via IPW/DR estimator using the known propensity; compare with random, target-all, propensity-only, and the predictive baseline.
- Bootstrap CIs (>=200 resamples) for all metrics; with 0.9% conversion, differences between learners will often be insignificant — report that honestly instead of picking a winner on noise.
- Choose the best model with written reasoning; run identical evaluation on X5, Lenta, MegaFon for the generalization claim.
- **Gate (§8.3):** beating random by a trivial margin is not "working" — investigate. Test set used once.

## Step 6 — Segmentation, sensitivity and stability
**Build:** `src/segmentation/`.
- Four quadrants need both potential outcomes: from the chosen model take `p1(x)`, `p0(x)`, `tau(x)`. Persuadables (tau > +eps), Sleeping Dogs (tau < -eps), Sure Things (tau ~ 0, p0 high), Lost Causes (tau ~ 0, p0 low). Choose eps/thresholds from the CATE confidence interval, not arbitrarily.
- Validate: segment sizes stable across bootstrap and across train/test; observed treated-minus-control effect inside each segment has the predicted sign; segment proportions compared across datasets.
- Subgroup/sensitivity analysis: flag low-overlap regions and unstable CATE estimates in the output rather than reporting a number (§8.5).

## Step 7 — Decision layer
**Build:** `src/optimization/`, `src/recommendation/`, `src/explainability/`, counterfactual module.
- **Budget optimizer:** value_i = tau_revenue(x_i) - cost; maximize total incremental value s.t. sum(cost) <= budget (greedy by value/cost is optimal for the knapsack-with-equal-cost case; use a solver if costs vary). Only treat units with positive value. Use the Hillstrom `spend` outcome (train a CATE model on spend, zero-inflated: try a two-part model or DR on raw spend). All cost/revenue assumptions are inputs and echoed in the output.
- **Policy evaluation:** ROI and incremental revenue vs random and target-all across budgets (curve, not one point).
- **Recommendation engine:** per customer: send/don't send, CATE, interval, overlap flag, segment.
- **Counterfactual simulator:** change one feature (recency, channel, ...), re-predict CATE, show delta; only vary features that are actually modifiable and warn that it is a model-implied, not experimental, effect.
- **Explainability:** SHAP on the CATE model (for meta-learners, explain the effect model / difference of outcome models; for forests, SHAP on the forest CATE). Translate feature names to business language for Hillstrom.

## Step 8 — Productization
**Build in this order:** FastAPI -> Streamlit -> Docker -> CI -> monitoring -> report -> deploy.
- **FastAPI** (`api/main.py`): `/predict_uplift`, `/recommend`, `/counterfactual`, `/metrics`, `/customers`; loads the fitted preprocessor + chosen model from `artifacts/`; request validation reuses `feature_spec.json`.
- **Streamlit** (`dashboard/streamlit_app.py`): Overview, EDA, Propensity, Treatment Effect, Customer Explorer, Budget Optimizer, Policy Simulator, Counterfactual Explorer, Model Comparison, Business Insights. Must call the same library code as the API.
- **Docker + compose** (app, API, dashboard, Postgres, MLflow).
- **PostgreSQL + auth**: store predictions, campaigns, experiment logs; Admin/Analyst/Read-only roles.
- **CI/CD** (GitHub Actions): lint, unit tests, data-contract tests, small end-to-end run on Hillstrom, image build.
- **Monitoring**: Evidently drift on features and predicted CATE; retraining trigger on drift.
- **PDF report**: auto-generated from stored results (EDA, assumptions, effects, segments, ROI vs baselines, limitations).
- **Cloud deploy** (Cloud Run / EC2 / App Service).
- **Optional LLM analyst layer:** explains outputs only; never computes or modifies estimates.
- **Gate:** raw data in -> recommendation + ROI number out through the deployed API/dashboard; consistency check that dashboard, API and offline numbers agree; failure/load handling tested.

## Step 9 — Robustness and scale
- **ACIC 2016**: run all learners; report PEHE and bias under confounding (this needs the observational variant — RCT-only methods may fail, which is a finding).
- **Criteo**: 100K -> 500K -> 1M -> 5M -> 14M; record train time, RAM, inference latency, model size. Use LightGBM-based learners; Causal Forest will not scale.

## Step 10 — Final validation and write-up
Limitations and failure conditions documented; final report tying EDA -> assumptions -> effects -> segments -> ROI; explicit statement that Hillstrom/X5/Lenta have no individual ground truth (ITE accuracy claims come only from synthetic/IHDP/ACIC).

## Suggested parallel work for 3 people
Data (Step 1) + EDA/propensity (2) | learners (4) | evaluation harness (5), after agreeing the learner interface and the `feature_spec.json` contract. Baselines (3) can be done by whoever finishes first. Steps 6-8 wait on 4-5.
