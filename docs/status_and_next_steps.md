# Status (2026-09-29)

- **Data:** done via `src/causal_prep` (Hillstrom, X5, Lenta) — see `stats/preprocessing_stats.md` and `reports/`.
- **Utilities:** `src/ingestion/validate.py`, `src/eda/balance.py`, `src/preprocessing/splits.py` (tested).
- **Not started:** everything from propensity modelling onward.
- **Next steps, in order and in detail:** [build_roadmap.md](build_roadmap.md).

Note: [causal_assumptions.md](causal_assumptions.md) was written before the merge; where its X5 details (e.g. redemption features)
differ from `stats/preprocessing_stats.md`, the latter is correct (X5 `first_redeem_date` is censored at the campaign date).
