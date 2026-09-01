# `preprocessing/` — causal-safe preprocessing pipeline

Research-grade, leakage-aware preprocessing for the **Causal Decision
Intelligence / Uplift Modeling** project. The primary dataset is the
**Kevin Hillstrom / MineThatData** 3-arm e-mail marketing RCT
(`datasets/phase 1 - Main Development/…2008.03.20.csv`), designated the primary
development dataset in `docs/dataset_guide.md`.

The pipeline **does not begin modeling**. Its only job is to turn the raw CSV
into a dataset that is *precise, defensible, and valid for uplift / CATE
estimation*, with every decision justified.

## Pipeline stages

```
Raw ─▶ Audit ─▶ Clean ─▶ Causal Feature Audit ─▶ Split
    ─▶ Fit preprocessors on TRAIN ─▶ Transform ─▶ Save ─▶ Validate
```

| Stage | Module | What it does |
|---|---|---|
| Audit | `hillstrom_audit.py` | Read-only profile of the raw file → `reports/hillstrom_audit.json`. Missingness, duplicates, invalid values, class imbalance, treatment balance, randomization sanity (SMD + propensity AUC). |
| Clean | `hillstrom_preprocess.clean` | Deterministic, **no row loss**: surrogate `customer_uid`, fix `Surburban→Suburban` typo, map `segment→treatment_arm / T / T_mens / T_womens`, ordinal `history_segment_ord`, derived `history_log1p / mw_count / bought_both`, explicit dtypes. |
| Causal Feature Audit | `hillstrom_preprocess.causal_feature_audit` | Classifies every column (Feature \| Category \| Action \| Reason) → `reports/hillstrom_feature_classification.csv`. |
| Split | `hillstrom_preprocess.split` | 80/20, **stratified on `treatment_arm × conversion`** (preserves the 0.9 % converter rate in both folds). Seeded. No time axis exists → no temporal split. |
| Fit / Transform / Save | `hillstrom_preprocess.fit_transform_save` | `ColumnTransformer` (median-impute numerics, most-frequent-impute + one-hot categoricals) **fit on train only**; optional `StandardScaler` kept as a **separate** artifact. Writes datasets + `feature_spec.json` + `preprocessor.joblib` + `scaler.joblib`. |
| Validate | `hillstrom_validate.py` | Hard asserts (row count conserved, no uid overlap, 0 missing/non-finite in X, no outcome/redundant column in X, one-hot row-sums = 1); balance/overlap diagnostics + figures; reproducibility hash. |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m preprocessing.run_all          # end-to-end, ~30 s
# or individually:
python -m preprocessing.hillstrom_audit
python -m preprocessing.hillstrom_preprocess   # (call from run_all; not standalone-main)
python -m preprocessing.hillstrom_validate
```

## Outputs

```
data/processed/hillstrom/
  hillstrom_clean.csv     all 64,000 rows, pre-encoding, human-readable (X + T + Y + ids + split)
  train.parquet/.csv      51,200 rows — model matrix (14 numeric X cols) + T cols + Y cols + ids
  test.parquet/.csv       12,800 rows — same schema
  train_scaled.parquet    optional variant: recency/history/history_log1p standardized (train-fit)
  test_scaled.parquet
  preprocessor.joblib     fitted ColumnTransformer (apply to any new raw-cleaned frame)
  scaler.joblib           fitted StandardScaler + column list
  feature_spec.json       X / T / Y column lists, exclusions, provenance
reports/
  hillstrom_audit.json
  hillstrom_data_quality_report.md
  hillstrom_feature_classification.csv
  hillstrom_balance_overlap.md
  hillstrom_validation.json
  figures/hillstrom_propensity_overlap.png, hillstrom_love_plot.png
```

## The causal contract

* **X** = pre-treatment covariates only (historical 12-month customer snapshot).
* **T** = randomized treatment — native 3 arms preserved (`treatment_arm`), plus
  binary `T` (any e-mail vs none) and one-vs-rest `T_mens`, `T_womens`.
* **Y** = post-treatment outcomes measured in the 2-week window after send:
  `visit` (mediator/intermediate), `conversion` (primary), `spend` (monetary).
  **Never** used as features.
* `history_segment` is a deterministic bucketing of `history` → excluded from X.
* Nothing that would be unavailable at send time enters X.

See `reports/causal_preprocessing_overview.md` for assumptions, the full feature
table, diagnostics, and remaining causal limitations. Configuration (paths,
seed, feature switches, thresholds) lives in `preprocessing/config.py`.
