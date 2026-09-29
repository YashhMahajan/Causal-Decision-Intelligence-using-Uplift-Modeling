# `data/raw/` — immutable inputs

Nothing in this tree is ever written to by the pipeline. Each dataset resolves to
a canonical path in `causal_prep/config.py::RAW_FILES`.

| Dataset | `data/raw/` entry | Actual source |
|---|---|---|
| **Hillstrom** | `hillstrom/hillstrom.csv` → symlink | `datasets/phase 1 - Main Development/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` (committed with the project) |
| **X5 RetailHero** | `x5/x5-retail-hero-uplift-raw-data/` → symlink | `datasets/phase 2 - generalization/x5-retail-hero-uplift-raw-data/` (committed): `clients.csv`, `products.csv`, `uplift_train.csv`, `uplift_test.csv` |
| **Lenta** | `lenta/lenta_dataset.csv.gz` | Downloaded by `scripts/fetch_lenta.sh` from the scikit-uplift public mirror (`https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz`, ~145 MB). See `docs/dataset_guide.md` §4. **Not committed** (git-ignored). |

The symlinks keep "raw is untouched" literally true while giving every dataset the
same `data/raw/<ds>/…` shape. If symlinks do not survive your checkout, point
`RAW_FILES` at the `datasets/…` paths above.

## Not acquired (out of current scope)

MegaFon (only a reference notebook is in the repo), and the causal-benchmark
datasets (Synthetic CATE, IHDP, ACIC, Criteo) named in `docs/dataset_guide.md`.
