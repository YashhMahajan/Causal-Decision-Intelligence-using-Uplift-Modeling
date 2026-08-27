# Causal Decision Intelligence Platform — Dataset Knowledge Base

> **Purpose:** This document is the dataset playbook for the Causal Decision Intelligence Platform for Personalized Marketing. It defines **what each dataset contains, what it is good for, when to use it, at what scale to use it, and what not to claim from it**.
>
> **Project focus:** marketing uplift modeling / causal ML, heterogeneous treatment effects, customer-level counterfactuals, causal segmentation, causal-native evaluation, and budget-constrained targeting.

---

## 1. Executive Summary

The project should **not rely on one dataset**.

Different datasets answer different validation questions:

| Dataset | Approx. practical size | Marketing | Randomized treatment | Individual ground truth | Primary role |
|---|---:|:---:|:---:|:---:|---|
| **Hillstrom / MineThatData** | 64,000 | Yes | Yes | No | **Primary development + demo** |
| **Lenta Uplift** | 10,000 benchmark subset | Yes | Yes | No | **Small-scale validation** |
| **MegaFon Uplift** | 10,000 benchmark subset | Yes | Yes | No | **Cross-domain validation** |
| **X5 RetailHero** | 10,000 benchmark subset | Yes | Yes | No | **Retail/customer-policy validation** |
| **Synthetic CATE benchmark** | ~2,000 | No / synthetic | Controlled | **Yes** | **ITE/CATE correctness** |
| **IHDP** | 672 | No / healthcare | Semi-synthetic | **Yes / semi-synthetic** | **Tiny causal benchmark** |
| **ACIC 2016** | ~4,800 | No / synthetic | Semi-synthetic | **Yes** | **Robustness research** |
| **Criteo Uplift** | 13,979,592 | Yes | Yes / incrementality | Not directly observed | **Large-scale stress test** |

### Recommended hierarchy

```text
                    PROJECT DATASET STRATEGY

             ┌───────────────────────────────┐
             │ Hillstrom — 64K               │
             │ PRIMARY DEVELOPMENT DATASET    │
             └───────────────┬───────────────┘
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
           Lenta 10K    MegaFon 10K    X5 10K
           validation    validation    validation
                │            │            │
                └────────────┼────────────┘
                             ↓
                  Cross-dataset testing
                             │
                ┌────────────┴────────────┐
                ↓                         ↓
        Synthetic CATE 2K             IHDP 672
        true ITE/CATE tests           tiny benchmark
                │
                ↓
             ACIC 4.8K
          robustness testing
                │
                ↓
          Criteo 100K → 1M+
             scale testing
```

---

# 2. What Our Dataset Actually Needs

The platform needs data that can support the following causal structure:

```text
Pre-treatment customer information X
                  │
                  ↓
          Treatment T
       ┌──────────┴──────────┐
       ↓                     ↓
    Control               Treatment
       │                     │
       └──────────┬──────────┘
                  ↓
             Outcome Y
                  │
                  ↓
      Individual treatment effect
          τ(x) = Y(1) − Y(0)
                  │
                  ↓
        Customer-level uplift
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   Target customer      Do not target
        │
        ↓
Incremental outcome / revenue
        │
        ↓
Budget-constrained policy
```

A useful dataset should therefore contain:

1. **Customer/entity ID**
2. **Treatment/intervention indicator**
3. **Outcome**
4. **Pre-treatment covariates**
5. Enough customer heterogeneity to estimate CATE
6. A meaningful treatment/control comparison
7. Sufficient overlap/positivity
8. Ideally, randomized treatment assignment
9. Ideally, revenue or monetary outcome for ROI optimization

### Critical distinction

There are three different kinds of "ground truth":

#### A. Real randomized experiment

You know treatment was assigned experimentally.

You can estimate causal effects from the experiment, but you **still do not observe both potential outcomes for the same individual**.

#### B. Semi-synthetic benchmark

Real covariates are combined with simulated outcomes/treatments.

The benchmark can provide known treatment effects.

#### C. Fully synthetic benchmark

The data-generating process itself is known.

You can know the true:

- ATE
- ITE
- CATE
- treatment-response function

For rigorous ITE validation, B/C are essential.

---

# 3. Primary Dataset — Hillstrom / MineThatData

## 3.1 Identity

**Name:** Kevin Hillstrom MineThatData E-Mail Analytics and Data Mining Challenge

**Official dataset page:**

https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

**CSV download:**

https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

---

## 3.2 Why this is the project's primary dataset

This is the closest match to the actual product.

It is a customer-level marketing experiment involving email campaigns.

The dataset contains approximately:

**64,000 customers**

with customers randomly assigned to marketing treatments/control.

The original challenge is specifically about understanding whether email marketing causes additional purchasing behavior.

This maps almost directly to our platform:

```text
Customer
   ↓
Email treatment
   ↓
Purchase behavior
   ↓
Incremental response
   ↓
Uplift score
   ↓
Targeting policy
```

---

## 3.3 Treatment

The original experiment contains:

- Men's email
- Women's email
- No email/control

For a simple binary uplift pipeline, a common approach is to construct:

```text
Treatment = relevant campaign
Control = no email
```

For a multi-treatment analysis, preserve the original treatment categories.

### Important

Do **not** blindly collapse Men's and Women's email into one treatment if the project's experiment is supposed to study treatment-specific effects.

The three-arm structure can also be useful for demonstrating that the platform can compare interventions.

---

## 3.4 Typical variables

Important variables include customer attributes and historical behavior such as:

- recency
- historical purchase frequency
- historical monetary value
- gender
- purchase/category indicators
- treatment assignment
- visit behavior
- conversion/purchase behavior
- post-campaign spend

Exact column names should be verified after ingestion rather than hard-coded into the documentation.

---

## 3.5 Outcomes

The dataset supports outcomes such as:

### Binary outcome

```text
purchase = 0 / 1
```

Useful for:

- conversion uplift
- Qini
- AUUC
- uplift curves
- customer segmentation

### Continuous outcome

```text
spend / revenue
```

This is especially valuable for:

- incremental revenue
- expected monetary value
- policy value
- ROI
- budget-constrained targeting

### Recommended project setup

Use **purchase/conversion** as the first outcome.

Then add **spend** as the business optimization outcome.

---

## 3.6 Pre-treatment features

Historical customer behavior is particularly useful because it can serve as X:

```text
X = historical customer characteristics
T = email intervention
Y = future response
```

This is the correct causal direction.

Do not include variables that were generated after the email campaign as model features.

---

## 3.7 Ground truth

Hillstrom is a genuine marketing experiment, but:

**It does not give us both Y(1) and Y(0) for every individual.**

Therefore:

```text
True individual treatment effect
        ❌ directly observed

Experimental population effect
        ✅ estimable

Uplift ranking / policy value
        ✅ evaluable from experimental data
```

This distinction must remain explicit in the research paper.

---

## 3.8 What Hillstrom should power

### Phase 1

- ingestion
- EDA
- treatment balance
- missing-value analysis
- feature engineering

### Phase 2

- propensity estimation
- overlap diagnostics
- S-learner
- T-learner
- X-learner
- DR-learner
- causal forest

### Phase 3

- ITE/CATE prediction
- uplift scores
- four causal segments
- Qini
- AUUC
- uplift curves

### Phase 4

- counterfactual simulation
- treatment policy
- budget optimization
- incremental revenue
- incremental ROI

### Phase 5

- explainability
- SHAP uplift explanations
- customer-level recommendations

**Verdict: ⭐⭐⭐⭐⭐ — Primary dataset**

---

# 4. Lenta Uplift Dataset

## 4.1 Identity

**Dataset:** Lenta Uplift Modeling Dataset

**Documentation/download:**

https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html

A manageable **10K benchmark version** is used in uplift benchmarking.

---

## 4.2 Domain

Retail / grocery marketing.

This is important because it gives the platform a second marketing domain.

```text
Hillstrom → email marketing
Lenta     → retail/grocery marketing
```

---

## 4.3 Data characteristics

The dataset contains customer-level information associated with a marketing campaign.

It includes treatment/control information and customer/shopper characteristics useful for uplift modeling.

A **10,000-row benchmark subset** is particularly attractive for this project because it is easy to run repeatedly during model experimentation.

---

## 4.4 Why it matters

Lenta should answer:

> Does the causal engine generalize beyond email marketing?

That is a much stronger validation story than simply training multiple models on the same dataset.

---

## 4.5 Best uses

Use Lenta for:

- treatment balance
- overlap diagnostics
- model comparison
- Qini/AUUC
- uplift segmentation
- policy evaluation
- cross-dataset generalization

### Not ideal for

Very detailed revenue/ROI claims unless the selected outcome and monetary semantics are explicitly validated from the dataset documentation.

**Verdict: ⭐⭐⭐⭐⭐ — Best small marketing validation dataset**

---

# 5. MegaFon Uplift Dataset

## 5.1 Identity

**Dataset:** MegaFon Uplift Modeling Dataset

**Documentation/download:**

https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html

A **10,000-row benchmark version** is used in uplift benchmarking.

---

## 5.2 Domain

Telecommunications / marketing.

This creates another domain shift:

```text
Email / retail
      ↓
Telecom marketing
```

---

## 5.3 Practical characteristics

The benchmark version contains approximately:

- 10,000 observations
- around 50 features
- binary treatment
- binary outcome

The treatment/control structure makes it suitable for standard uplift workflows.

---

## 5.4 Best uses

MegaFon is particularly useful for:

- S/T/X/DR learner comparison
- causal forest comparison
- uplift ranking
- Qini
- AUUC
- cross-domain validation
- pipeline reproducibility

---

## 5.5 Limitations

It is less naturally aligned with the project's eventual **revenue optimization** story than Hillstrom.

Therefore:

```text
Hillstrom → business/product demo
MegaFon   → causal-model generalization
```

**Verdict: ⭐⭐⭐⭐½ — Excellent validation dataset**

---

# 6. X5 RetailHero Uplift Dataset

## 6.1 Identity

**Dataset:** X5 RetailHero Uplift Modeling

**Competition/data page:**

https://ods.ai/competitions/x5-retailhero-uplift-modeling/data

**Alternative dataset documentation:**

https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html

---

## 6.2 Domain

Retail/customer marketing.

Unlike a simple tabular marketing dataset, X5 has richer customer and purchase-history information.

The competition data includes components such as:

```text
clients
products
purchases
uplift_train
uplift_test
```

---

## 6.3 Why it is valuable

It can demonstrate a more realistic data-engineering pipeline:

```text
Customer profile
       +
Historical transactions
       ↓
Feature engineering
       ↓
Treatment
       ↓
Future outcome
       ↓
Uplift
       ↓
Targeting policy
```

This is particularly relevant if the final platform is supposed to resemble a real decision system rather than a single CSV notebook.

---

## 6.4 Size warning

The complete raw dataset is much larger than the other datasets.

The full purchase data is roughly:

- hundreds of MB compressed
- several GB uncompressed
- tens of millions of purchase records

That is unnecessary during initial development.

A **10K benchmark subset** is available/used in uplift benchmarking.

### Recommended usage

```text
Development:
10K

Feature-engineering experiments:
10K–100K

Scale test:
full dataset
```

---

## 6.5 Best uses

- customer feature engineering
- treatment-effect modeling
- uplift ranking
- policy evaluation
- retail targeting
- pipeline scalability

**Verdict: ⭐⭐⭐⭐½ — Excellent richer retail benchmark**

---

# 7. Synthetic CATE Benchmark

## 7.1 Purpose

This dataset is not intended to demonstrate marketing realism.

It exists to answer a different question:

> **Are our causal estimators actually recovering treatment effects correctly?**

A benchmark version of approximately **2,000 observations** with known treatment-effect structure is available through uplift benchmarking resources.

**Benchmark repository:**

https://github.com/binshuangli/uplift-bench

---

## 7.2 Why this is essential

In observational/RCT marketing data:

```text
Y(1) observed for treated customer
Y(0) observed for control customer
```

but never both for the same customer.

Therefore individual treatment-effect error cannot be directly measured.

Synthetic data solves this.

You can compare:

```text
True ITE
     vs
Predicted ITE
```

---

## 7.3 Metrics

Use:

- PEHE
- ITE MSE
- ITE correlation
- ATE bias
- CATE calibration
- treatment-effect ranking quality

This gives the research paper a stronger methodological evaluation.

**Verdict: ⭐⭐⭐⭐⭐ — Required for causal-estimator validation**

---

# 8. IHDP

## 8.1 Identity

**Infant Health and Development Program (IHDP)** is a classic causal inference benchmark.

A common benchmark version contains:

**672 observations × 25 covariates**

It is not marketing data.

---

## 8.2 Why keep it?

It is extremely small.

It can be used to quickly validate:

```text
S-Learner
T-Learner
X-Learner
DR-Learner
Causal Forest
```

without waiting for large datasets.

---

## 8.3 What it is good for

- debugging causal pipelines
- checking treatment-effect estimation
- comparing estimators
- testing CATE metrics
- reproducibility

---

## 8.4 What it is NOT good for

Do not use IHDP to claim:

> "Our marketing platform works on customers."

It is a healthcare benchmark.

Use it only as a methodological benchmark.

**Verdict: ⭐⭐⭐⭐ — Tiny research/debugging benchmark**

---

# 9. ACIC 2016

## 9.1 Identity

**Atlantic Causal Inference Conference 2016 challenge**

Information:

https://www.acicdatachallenge.org/

A commonly used benchmark configuration contains roughly:

**4,800 observations × 79 covariates**

---

## 9.2 Purpose

ACIC is useful because it provides semi-synthetic causal inference scenarios where treatment effects can be evaluated against known simulated truth.

---

## 9.3 Best uses

- robustness testing
- nonlinear treatment effects
- confounding stress tests
- CATE estimation
- estimator comparison
- research-paper evaluation

---

## 9.4 Role in our project

ACIC should be:

```text
Research / robustness benchmark
```

not:

```text
Main marketing demonstration
```

**Verdict: ⭐⭐⭐⭐ — Strong causal robustness benchmark**

---

# 10. Criteo Uplift Prediction Dataset

## 10.1 Identity

**Criteo Uplift Prediction Dataset**

**Official Criteo AI Lab page:**

https://ailab.criteo.com/criteo-uplift-prediction-dataset/

**Research repository:**

https://github.com/criteo-research/large-scale-ITE-UM-benchmark

---

## 10.2 Size

Approximately:

**13,979,592 rows**

with:

**12 anonymized features**

plus treatment/exposure/outcome-related variables.

It is several orders of magnitude larger than the small datasets.

---

## 10.3 Domain

Online advertising / incrementality.

This is highly relevant to causal marketing.

---

## 10.4 Why it matters

Criteo is useful for demonstrating:

> The causal decision engine can scale beyond small academic datasets.

---

## 10.5 Major limitation

The features are anonymized.

Conceptually you may have:

```text
f0
f1
f2
...
f11
```

rather than meaningful business variables.

That makes customer-facing explanations less intuitive.

For example:

Bad demo:

> f3 increased uplift by 0.18.

Better demo:

> Recent purchasing behavior and historical engagement increased predicted incremental response.

Hillstrom supports the latter style much more naturally.

---

## 10.6 Recommended scale

Do not start with 14M.

Use:

```text
100K
   ↓
500K
   ↓
1M
   ↓
5M
   ↓
13.98M
```

Measure:

- training time
- memory usage
- inference throughput
- storage
- model scalability

**Verdict: ⭐⭐⭐⭐⭐ scientifically / ⭐⭐⭐ practically for initial development**

---

# 11. Dataset Selection by Project Module

This is the most important operational section.

## 11.1 Data ingestion / EDA

### Use

**Hillstrom 64K**

Why:

- manageable
- meaningful marketing context
- customer-level
- randomized campaign
- easy to visualize

---

## 11.2 Treatment balance

### Primary

Hillstrom

### Validation

Lenta / MegaFon / X5

Check:

```text
P(T=1)
P(T=0)
```

and covariate balance.

---

# 11.3 Propensity modeling

### Use

Hillstrom first.

Then Lenta/MegaFon/X5.

Even in an RCT, a propensity model can still be useful as a diagnostic/modeling component.

For a perfectly randomized binary treatment:

```text
P(T=1 | X)
```

should ideally be approximately constant with respect to X.

Do not artificially invent confounding just because a propensity model exists.

---

# 11.4 Positivity / overlap

Use:

- Hillstrom
- Lenta
- MegaFon
- X5

Plot:

```text
Propensity distribution by treatment
```

and identify regions with weak overlap.

---

# 11.5 S/T/X/DR learners

Run on:

1. Hillstrom
2. Lenta
3. MegaFon
4. X5

Use synthetic CATE for ground-truth comparison.

---

# 11.6 Causal Forest

Primary:

**Hillstrom**

Validation:

**Lenta + X5**

Ground-truth methodological test:

**Synthetic CATE / IHDP / ACIC**

---

# 11.7 ITE/CATE accuracy

Do NOT rely solely on Hillstrom.

Use:

**Synthetic CATE**

and optionally:

**IHDP + ACIC**

because their simulated/semi-synthetic truth allows direct treatment-effect evaluation.

---

# 11.8 Four customer segments

Your project requires:

```text
                 Predicted Response
                    Treatment
                       ↑
                       │
     Lost Causes       │      Persuadables
                       │
───────────────────────┼────────────────────→
                       │
     Sleeping Dogs     │      Sure Things
                       │
```

Use:

**Hillstrom first**

because the marketing interpretation is strongest.

Validate segmentation on:

- Lenta
- MegaFon
- X5

---

# 11.9 Qini / AUUC

Best datasets:

1. Hillstrom
2. Lenta
3. MegaFon
4. X5
5. Criteo

Use the same evaluation pipeline across all datasets.

---

# 11.10 Counterfactual simulation

Best:

**Hillstrom**

because the treatment is an actual marketing intervention and the customer context is intuitive.

Example:

```text
Customer A

Treatment:
Send email

Predicted:
P(purchase | treatment) = 0.18
P(purchase | control)   = 0.07

Estimated uplift = +0.11
```

---

# 11.11 Budget optimization

### Best

**Hillstrom**

Especially if using monetary outcomes.

Example:

```text
Budget = ₹100,000

Customer i:
Expected incremental revenue = ₹X
Treatment cost = ₹Y

Select customers maximizing:

Σ incremental revenue
---------------------
Σ campaign cost
```

The same policy engine can then be tested on other datasets where monetary semantics are available.

---

# 11.12 SHAP uplift explanations

Best:

**Hillstrom / X5**

Meaningful customer variables are preferable to anonymized variables.

Criteo is less attractive here.

---

# 11.13 Scale testing

Use:

**Criteo**

Recommended progression:

```text
10K
 ↓
50K
 ↓
100K
 ↓
500K
 ↓
1M
 ↓
5M
 ↓
14M
```

Record:

- training runtime
- RAM
- inference latency
- model size
- feature-processing time

---

# 12. Recommended Experimental Matrix

| Experiment | Dataset | Size | Purpose |
|---|---|---:|---|
| Pipeline development | Hillstrom | 64K | Main system |
| Fast debugging | IHDP | 672 | Causal pipeline |
| CATE truth | Synthetic | ~2K | ITE accuracy |
| Robustness | ACIC | ~4.8K | Causal robustness |
| Marketing validation | Lenta | 10K | Cross-domain |
| Marketing validation | MegaFon | 10K | Cross-domain |
| Retail validation | X5 | 10K | Richer customer data |
| Large-scale test | Criteo | 100K | Initial scale |
| Large-scale test | Criteo | 1M | Serious scale |
| Maximum stress | Criteo | 14M | Final scalability |

---

# 13. Recommended Development Stages

## Stage 0 — Causal pipeline debugging

Dataset:

**IHDP — 672**

Goal:

Make sure:

```text
data → treatment → outcome → estimator → CATE → metrics
```

actually works.

---

## Stage 1 — Ground-truth causal validation

Dataset:

**Synthetic CATE — ~2K**

Goal:

Validate:

- ITE
- CATE
- ATE
- PEHE
- calibration
- treatment-effect ranking

---

## Stage 2 — Main product development

Dataset:

**Hillstrom — 64K**

Build the entire platform.

---

## Stage 3 — Generalization

Run the exact same pipeline on:

```text
Lenta 10K
MegaFon 10K
X5 10K
```

Compare:

- model performance
- Qini
- AUUC
- policy value
- segment proportions
- overlap

---

## Stage 4 — Robustness

Use:

**ACIC**

Test how the methods behave under more difficult causal structures.

---

## Stage 5 — Scale

Use:

**Criteo**

```text
100K
500K
1M
5M
14M
```

Test system performance.

---

# 14. What NOT to Do

## Don't use only one dataset

One dataset can hide:

- overfitting
- dataset-specific treatment behavior
- feature-specific effects
- poor generalization

---

## Don't claim individual causal truth from Hillstrom

You cannot observe:

```text
Y_i(1)
and
Y_i(0)
```

for the same customer.

Therefore, don't report individual ITE error as if Hillstrom provides exact ITE labels.

---

## Don't use post-treatment variables

Anything generated after treatment can create leakage.

For example:

```text
Email sent
   ↓
Customer opens email
   ↓
Customer visits website
```

If "website visit" happens after treatment, it must not be used as a pre-treatment X.

---

## Don't train on the full Criteo dataset initially

Start with a sample.

Large data is useful only after the causal pipeline is stable.

---

## Don't judge models using only prediction metrics

This is a causal project.

Do not make:

```text
Accuracy
F1
RMSE
```

the primary success criteria.

Prioritize:

- ATE error
- CATE/ITE quality
- Qini
- AUUC
- uplift curve
- policy value
- incremental gain
- calibration
- ROI

---

# 15. Recommended Final Dataset Suite

If the project must stay manageable, use exactly this:

## Core

### 1. Hillstrom — 64K

**Role:** Main product dataset

---

### 2. Lenta — 10K

**Role:** Small marketing validation

---

### 3. MegaFon — 10K

**Role:** Cross-domain validation

---

### 4. X5 — 10K

**Role:** Retail/customer-data validation

---

## Causal research

### 5. Synthetic CATE — ~2K

**Role:** Known-ground-truth ITE/CATE validation

---

### 6. IHDP — 672

**Role:** Tiny causal benchmark/debugging

---

### 7. ACIC 2016 — ~4.8K

**Role:** Robustness benchmark

---

## Scale

### 8. Criteo — 14M

**Role:** Large-scale stress test

---

# 16. Acquisition Order

Do not download everything at once.

### Download first

```text
1. Hillstrom
2. Synthetic CATE
3. Lenta
```

Build the initial pipeline.

### Then

```text
4. MegaFon
5. X5
```

Test generalization.

### Then

```text
6. IHDP
7. ACIC
```

Strengthen research evaluation.

### Finally

```text
8. Criteo
```

Only when the system is already stable.

---

# 17. Quick Decision Table

| If you want to... | Use |
|---|---|
| Build the actual application | **Hillstrom** |
| Work with the smallest marketing dataset | **Lenta 10K** |
| Test another industry | **MegaFon** |
| Test richer retail data | **X5** |
| Know the true ITE | **Synthetic CATE** |
| Debug causal models quickly | **IHDP** |
| Stress-test causal assumptions | **ACIC** |
| Test millions of customers | **Criteo** |
| Demonstrate marketing realism | **Hillstrom** |
| Demonstrate ROI optimization | **Hillstrom** |
| Demonstrate cross-dataset generalization | **Lenta + MegaFon + X5** |
| Demonstrate causal-estimator correctness | **Synthetic + IHDP + ACIC** |
| Demonstrate scalability | **Criteo** |

---

# 18. Final Recommendation

The project should be presented as a **single causal decision platform evaluated across complementary datasets**, not as a collection of unrelated datasets.

The strongest architecture is:

```text
                         CAUSAL PLATFORM
                               │
                ┌──────────────┴──────────────┐
                │                             │
         MARKETING REALISM              CAUSAL TRUTH
                │                             │
          Hillstrom 64K                 Synthetic ~2K
                │                         IHDP 672
        ┌───────┼────────┐                ACIC ~4.8K
        ↓       ↓        ↓
      Lenta  MegaFon     X5
       10K     10K      10K
        │       │        │
        └───────┼────────┘
                │
         Generalization
                │
                ↓
        Criteo 100K → 14M
                │
                ↓
          Scale testing
```

### The core message for the research paper

> **Marketing RCT datasets establish practical causal decision performance, while synthetic and semi-synthetic benchmarks establish treatment-effect estimation validity; a large-scale dataset then establishes computational scalability.**

That is much more defensible than trying to force one dataset to satisfy every requirement.

---

# 19. Sources / Download Links

### Primary marketing datasets

- **Hillstrom / MineThatData:**  
  https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html  
  CSV: https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv

- **Lenta:**  
  https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html

- **MegaFon:**  
  https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html

- **X5 RetailHero:**  
  https://ods.ai/competitions/x5-retailhero-uplift-modeling/data  
  https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html

### Causal benchmarks

- **Uplift benchmark repository:**  
  https://github.com/binshuangli/uplift-bench

- **ACIC:**  
  https://www.acicdatachallenge.org/

### Large-scale

- **Criteo AI Lab:**  
  https://ailab.criteo.com/criteo-uplift-prediction-dataset/

- **Criteo research repository:**  
  https://github.com/criteo-research/large-scale-ITE-UM-benchmark

---

## One-line operating rule

**Hillstrom builds the product → Lenta/MegaFon/X5 prove generalization → Synthetic/IHDP/ACIC prove causal methodology → Criteo proves scale.**
