# lenta — Treatment/Control & Overlap Diagnostics

## train

- arm counts: `{'treated': 412713, 'control': 136910}`

| arm | response_att |
|---|---|
| control | 0.1026 |
| treated | 0.1101 |

- naive ATE vs control:
  - **treated**: response_att +0.0075

## test

- arm counts: `{'treated': 103179, 'control': 34227}`

| arm | response_att |
|---|---|
| control | 0.1026 |
| treated | 0.1101 |

- naive ATE vs control:
  - **treated**: response_att +0.0075

## Covariate balance (train)

- max |SMD| = **0.0283** (flag 0.1); features above flag: **0**

| feature (top 20 by |SMD|) | SMD |
|---|---:|
| crazy_purchases_cheque_count_12m | +0.0283 |
| crazy_purchases_goods_count_12m | +0.0279 |
| gender_M | -0.0263 |
| gender_F | +0.0257 |
| crazy_purchases_cheque_count_6m | +0.0250 |
| crazy_purchases_goods_count_6m | +0.0231 |
| crazy_purchases_cheque_count_3m | +0.0218 |
| age | +0.0213 |
| promo_share_15d | +0.0190 |
| k_var_disc_share_3m_g26 | +0.0188 |
| k_var_count_per_cheque_1m_g34 | +0.0187 |
| mean_discount_depth_15d | +0.0184 |
| k_var_disc_share_3m_g34 | +0.0181 |
| k_var_sku_price_3m_g33 | +0.0178 |
| k_var_count_per_cheque_3m_g34 | +0.0174 |
| k_var_disc_share_1m_g34 | +0.0174 |
| cheque_count_12m_g38 | +0.0166 |
| k_var_sku_price_3m_g27 | +0.0164 |
| k_var_disc_share_3m_g27 | +0.0162 |
| k_var_sku_price_6m_g44 | +0.0162 |

## Propensity / positivity

(diagnostic n = 60000)

- **logreg**: AUC 0.4968; support [0.309, 0.989] (p01 0.676, p99 0.836); mass outside trim 0.0000
- **hgb**: AUC 0.4996; support [0.548, 0.853] (p01 0.698, p99 0.792); mass outside trim 0.0000

**Verdict:** Strong overlap consistent with randomization: propensity AUC in the RCT band, support concentrated near P(T=1), no mass in the trim tails. CATE is identified across the covariate space.

![propensity overlap](figures/propensity_overlap.png)

![love plot](figures/love_plot.png)
