# Causal Decision Intelligence using Uplift Modeling

Causal-safe, reproducible **preprocessing** for uplift / heterogeneous-treatment-
effect modelling across three real marketing RCTs. **No models are trained yet** —
this repo currently delivers audited, leakage-safe, modelling-ready datasets plus
the diagnostics that justify them.

See `docs/knowledge_base_1.md` (platform spec) and `docs/dataset_guide.md`
(dataset strategy) for project context.

## Datasets

| Name | Source | Rows | Treatment | Outcome | Notes |
|---|---|---:|---|---|---|
| **Hillstrom** | `datasets/phase 1 - Main Development/…2008.03.20.csv` (in repo) | 64,000 | 3-arm e-mail (`segment`) + binary any-e-mail | `conversion` (+ `visit`, `spend`) | Primary development dataset |
| **X5 RetailHero** | `datasets/phase 2 - generalization/x5-retail-hero-uplift-raw-data/` (in repo) | 200,039 | `treatment_flg` (SMS) | `target` | Feature-poor: no `purchases` table |
| **Lenta** | `sklift` public mirror → `scripts/fetch_lenta.sh` | 687,029 | `group` (communication) | `response_att` | 190 covariates, heavy structural missingness |

## Repository layout

```
data/
  raw/            immutable inputs (symlinks into datasets/ for Hillstrom & X5; downloaded .csv.gz for Lenta)
  interim/        cleaned, pre-encoding frames  (git-ignored, regenerable)
  processed/      model-ready train/test parquet (git-ignored, regenerable)
src/causal_prep/  the pipeline
  config.py       paths, seed, thresholds
  common.py       shared machinery: DatasetSpec, audit, split, fit/transform/save, validation, diagnostics
  reporting.py    markdown report writers
  run.py          CLI:  python -m causal_prep.run {hillstrom|x5|lenta|all}
  datasets/
    hillstrom.py  load_raw / clean / SPEC / feature_classification  (dataset-specific logic, isolated)
    x5.py
    lenta.py
artifacts/<ds>/   fitted preprocessor.joblib + scaler.joblib + feature_spec.json  (committed; small)
reports/<ds>/     audit.json, data_quality.md, feature_classification.csv, balance_overlap.md,
                  validation.json, figures/*.png
reports/causal_preprocessing_overview.md   ← start here
scripts/fetch_lenta.sh
datasets/         original project drop (unchanged, treated as read-only raw)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # installs pinned deps from pyproject.toml
bash scripts/fetch_lenta.sh      # ~145 MB, one-time
python -m causal_prep.run all    # Hillstrom ~30s, X5 ~20s, Lenta ~2min
```

Outputs land in `data/processed/<ds>/`, `artifacts/<ds>/`, `reports/<ds>/`.
Every run is deterministic (`RANDOM_SEED = 20240501`).

## The causal contract (enforced for every dataset)

```
X = pre-treatment covariates only   (available at the treatment decision point)
T = randomized treatment
Y = post-treatment outcome(s)        (targets only — never features)

raw → audit → clean → causal feature audit → stratified split
    → fit preprocessors on TRAIN only → transform → save → validate
```

Hard gates checked on every output: row count conserved, no train/test unit
overlap, zero missing/non-finite cells in X, **no outcome / post-treatment /
excluded column in X**, one-hot integrity, reproducible split hash. Raw files are
never written to.

## What is NOT here yet

Uplift/CATE models; the synthetic / IHDP / ACIC / Criteo / MegaFon datasets (not
in the repo). `datasets/phase 1 - Main Development/Digital Marketing Campaign
Dataset.xlsx` is synthetic with post-treatment columns and non-random exposure —
**not** a valid causal dataset; ignored except as a schema fixture.
