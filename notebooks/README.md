# `notebooks/`

Scratch space for exploration. **No pipeline logic lives in notebooks** — the
preprocessing standard is entirely in `src/causal_prep/` and is driven by
`python -m causal_prep.run`.

## Reference notebooks shipped with the project

Two external Kaggle-style notebooks are in `datasets/phase 2 - generalization/`
(left in place, not modified):

- `study-series-uplift-modeling.ipynb` — MegaFon data; S-/T-learner walkthrough
  with `scikit-uplift`. MegaFon data itself is **not** in the repo.
- `uplift-modelling-lenta-dataset.ipynb` — Lenta; assumes a `CardHolder` id
  column and a Kaggle path. The `sklift` mirror we actually use
  (`scripts/fetch_lenta.sh`) has **no** `CardHolder` column — see
  `reports/lenta/data_quality.md`.

They informed the intended modelling stack (meta-learners, Qini/AUUC) but contain
no preprocessing to port.
