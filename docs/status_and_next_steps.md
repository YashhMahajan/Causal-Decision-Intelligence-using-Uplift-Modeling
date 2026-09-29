# Status and next steps (as of 2026-09-29)

## Audit of what existed
Only the two docs, Hillstrom CSV, and X5 clients/uplift files. **No code and no preprocessing existed.**
Lenta and MegaFon have only third-party notebooks (no data). X5 `purchases.csv` is missing. Phase 3/4 dataset folders are empty.

## Done in this commit (data foundation)
- `src/config.py`, `src/ingestion/validate.py` (quality report, naive ATE + CI), `src/eda/balance.py` (SMD)
- `src/preprocessing/{hillstrom,x5,splits}.py`; `scripts/prepare_data.py` -> `data/processed/{hillstrom,x5,x5_10k}.parquet`
- Stratified (treatment x outcome) 60/20/20 splits, fixed seed; 12 passing tests (leakage guard, randomization, split stratification)
- `docs/causal_assumptions.md`; `scripts/fetch_uplift_benchmarks.py` for Lenta/MegaFon (**written but untested — needs downloads**)

## Next (in dependency order, per design doc §8)
1. **Fetch Lenta + MegaFon** (needs your OK for the downloads), optionally X5 purchases.csv for real feature engineering.
2. **EDA + propensity/overlap** (`src/eda`, `src/propensity`): plots, Love plot, overlap. Expect flat propensity on RCT data.
3. **Baseline non-causal model** for comparison.
4. **Learners** S/T/X/DR (own implementations, `src/learners/`) + CausalForestDML (add `econml`, `lightgbm`, `mlflow` to requirements). Gate: ATE vs naive ATE above.
5. **Evaluation** (`src/evaluation/`): Qini, AUUC, uplift curves, policy value; baselines = random, target-all, propensity-only. Decide up front how to handle Hillstrom's rare conversion (bootstrap CIs; pool arms vs. per-arm).
6. **Synthetic CATE / IHDP** for true ITE error (PEHE) — needed to validate the learners themselves.
7. Segmentation (4 quadrants), counterfactual simulator, budget optimizer (use `spend` outcome + cost assumption), SHAP.
8. Productization: FastAPI, Streamlit, Docker, MLflow, CI, drift monitoring, PDF report, DB/auth, deploy. Then ACIC robustness, Criteo scale test.

## Open decisions for the team
- Binary treatment = "any email" vs per-arm (men's vs control, women's vs control). Both are supported by `arm`.
- Cost per contact and revenue-per-conversion assumptions for ROI.
- Whether X5 full 200K or the 10K sample is the working set.
