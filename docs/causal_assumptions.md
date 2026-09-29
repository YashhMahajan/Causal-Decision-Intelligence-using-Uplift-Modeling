# Causal assumptions and known data limitations (design doc §8.1)

## Hillstrom (primary)
- **Design:** randomized 3-arm email experiment, 64,000 customers (21.3K control / 21.3K men's / 21.4K women's).
  Verified after preprocessing: max |SMD| across all features = 0.007 (threshold 0.1) -> randomization looks intact.
- **Unconfoundedness:** holds by design. A propensity model here is a *diagnostic* (should be ~constant), not a confounding correction.
- **Positivity:** by design (each arm ~1/3). `t_binary` treated share is 0.667, not 0.5 — X/DR-learners and IPW must use the true propensity.
- **SUTVA:** plausible (email to one customer should not affect another) but not testable.
- **Outcomes:** `conversion` is rare (0.9%; 0.57% control vs 1.07% treated) -> CATE estimates are noisy; expect wide intervals and weak Qini signal. `spend` is zero-inflated and heavy-tailed (max $499).
- **Feature timing:** `mens`/`womens` are *past* purchases (pre-treatment); `visit`, `conversion`, `spend` are post-treatment and excluded from X.
- **No ID column:** 6,562 fully identical rows exist; they are distinct customers, not duplicates, so nothing is dropped.
- **No cost data:** ROI needs a contact-cost assumption (`ASSUMED_COST_PER_CONTACT` = $0.10 in `src/config.py`). Treat as an assumption in every report.
- **Ground truth:** no individual ITE labels. Evaluate with Qini/AUUC/policy value only; use synthetic data for ITE error.
- **Naive ATE (the sanity target for all learners):** conversion +0.50pp (95% CI 0.35–0.64pp); spend +$0.60 (CI 0.38–0.82).

## X5 RetailHero (generalization)
- 200,039 labeled clients, randomized (treated share 0.4998, max |SMD| 0.023). Outcome rate is high (~62%): naive ATE +3.3pp (CI 2.9–3.7pp).
- **Only clients.csv features exist** (age, gender, loyalty-card dates). `purchases.csv` is not in the repo, so the "rich purchase history" use case in the dataset guide is not yet possible.
- `age` is corrupted (min -7491, max 1901; 1,779 rows outside 14–100): set to NaN with `age_invalid` flag. 35,469 clients never redeemed (`first_redeem_date` NaN) -> `ever_redeemed` flag.
- `uplift_test.csv` has no labels (competition hold-out) and is unused.

## Not yet available
Lenta, MegaFon (raw data not in repo — see status doc), Synthetic CATE, IHDP, ACIC, Criteo.
