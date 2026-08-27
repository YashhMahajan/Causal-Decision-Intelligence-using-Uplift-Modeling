# AI-Powered Causal Decision Intelligence Platform for Personalized Marketing

**Area of work:** Core Machine Learning, Orchestration, Decision-Making Infrastructure
**Category:** Uplift Modeling / Causal ML / Applied Decision Systems
**Team:** 3 members, working jointly across all parts — no fixed role split by expertise.
**Bar:** every module must be verified correct before being treated as done — see §8.

---

## 1. Problem Statement

Traditional personalized marketing systems use predictive models to identify customers with a high *probability* of purchasing or responding to a campaign. But high conversion probability ≠ the intervention *caused* the conversion.

This creates three failure modes:

| Customer type | What happens if targeted | Cost |
|---|---|---|
| **Sure Things** | Would have bought anyway | Wasted spend |
| **Lost Causes** | Won't buy regardless | Wasted spend |
| **Sleeping Dogs** | React negatively to contact | Negative ROI (churn/unsubscribe) |
| **Persuadables** | Buy *because* of the campaign | Only this group is worth targeting |

The gap: existing systems answer **"Who is likely to purchase?"** (a prediction problem). The correct question for marketing decisions is **"Who will purchase *because of* this specific action?"** (a causal inference problem).

**Core thesis of the project:** apply uplift modeling / causal ML to estimate Individual Treatment Effects (ITE) and Conditional Average Treatment Effects (CATE), then use those estimates to drive a constrained optimization decision (who to target, under a budget) — not just a ranked list.

---

## 2. Core Conceptual Shift

```
Predictive ML:        P(Y | X)                                → "will they convert?"
Causal ML (this):     P(Y | X, do(T=1)) − P(Y | X, do(T=0))   → "does the treatment change the outcome?"
```

Example:
```
Customer A, predictive probability of buying = 82%   → not actionable
Customer A, P(buy | email) − P(buy | no email) = 82% − 70% = +12% uplift → actionable
```

Business framing: 10M customers, $500K budget → you cannot email everyone, and you shouldn't — the optimization target is incremental revenue per dollar spent, not conversion count.

---

## 3. System Architecture (End-to-End Pipeline)

```
Dataset (CSV / SQL / API)
        │
Data Validation Layer (schema, missingness, duplicates, quality score)
        │
Preprocessing + Feature Engineering
        │
Exploratory Dashboard (treatment/outcome distributions, imbalance)
        │
Propensity Score Model  →  P(Treatment | X)
        │
Covariate Balance Check (SMD, Love Plot, Overlap Plot)
        │
Treatment Effect Estimation
   ├── S-Learner
   ├── T-Learner
   ├── X-Learner
   ├── DR-Learner (Doubly Robust)
   └── Causal Forest (EconML / CausalForestDML)
        │
Individual Treatment Effect (ITE) / CATE
        │
Customer Segmentation Engine
   (Persuadables / Sure Things / Lost Causes / Sleeping Dogs)
        │
Evaluation Layer (Qini, AUUC, Uplift Curve, Policy Risk)
        │
Explainability (SHAP on uplift drivers)
        │
Counterfactual Simulator (what-if per customer)
        │
Budget-Constrained Optimizer (maximize incremental ROI s.t. budget)
        │
Recommendation Engine (send / don't send + confidence)
        │
API (FastAPI) + Dashboard (Streamlit) + Optional LLM analyst layer
```

**Key architectural principle:** the LLM (if added) sits *downstream* of the causal engine and only explains outputs in natural language. It never generates or determines causal estimates — that boundary is both a technical and an academic integrity requirement.

---

## 4. Module Breakdown

### 4.1 Data Ingestion & Validation
- Sources: CSV, Excel, SQL, PostgreSQL, API
- Checks: missing columns, wrong dtypes, duplicates, data quality score
- Output: dataset health report (rows, columns, missing %, outliers, duplicate %, quality score)

### 4.2 Automatic EDA
- Distribution plots, correlation matrix, target/treatment distribution, class imbalance
- Auto-generated insights (e.g., "treatment assignment correlates with income → confounding risk")

### 4.3 Propensity Score Estimation
- `P(Treatment | X)` via Logistic Regression, Random Forest, XGBoost
- Diagnostics: overlap plot, propensity distribution, covariate balance (SMD, Love Plot)
- Purpose: verify positivity assumption before trusting any downstream causal estimate

### 4.4 Treatment Effect Estimation (Meta-Learners + Causal Forest)
| Method | When it's the right tool |
|---|---|
| S-Learner | Baseline, simple, weak when treatment effect is small relative to outcome variance |
| T-Learner | Separate models per arm; better with distinct treatment/control regimes |
| X-Learner | Best under strong treatment/control imbalance |
| DR-Learner | Doubly robust — protects against propensity *or* outcome model misspecification |
| Causal Forest (EconML) | Non-parametric CATE, best for heterogeneity discovery + confidence intervals |

### 4.5 Customer Segmentation (4 quadrants, driven by ITE sign × baseline behavior)
- **Persuadables** — positive uplift → target
- **Sure Things** — will convert regardless → don't waste spend
- **Lost Causes** — won't convert regardless → don't waste spend
- **Sleeping Dogs** — negative uplift → actively avoid contact

### 4.6 Evaluation (causal-specific, not accuracy-based)
- **Qini Curve** — industry standard for uplift model ranking quality
- **AUUC** (Area Under Uplift Curve)
- **Uplift Curve**, **Incremental Gain Curve**
- **Calibration Plot**, **Policy Risk**
- **ATE** (Average Treatment Effect), **CATE** (Conditional ATE)

### 4.7 Explainability
- SHAP applied to the uplift model (not the outcome model) → "why is uplift high for this segment"

### 4.8 Counterfactual Simulator
- Per-customer what-if: adjust age/income/region, observe how ITE changes
- Most demo-friendly module — makes the causal reasoning tangible

### 4.9 Budget-Constrained Optimizer
- Input: budget, cost/customer
- Output: ranked customer selection maximizing incremental ROI subject to budget constraint
- This is a constrained optimization problem layered on top of the causal estimates — the actual "decision intelligence" part, distinct from pure modeling

### 4.10 Recommendation Engine
- Per-customer: send/don't-send + uplift estimate + confidence

### 4.11 Dashboard (Streamlit) Pages
Overview, EDA, Propensity, Treatment Effect, Customer Explorer, Budget Optimizer, Policy Simulator, Counterfactual Explorer, Model Comparison, Business Insights

### 4.12 Auto Report Generation (PDF)
Dataset summary, EDA, treatment effects, recommendations, ROI, visualizations, business conclusions — for non-technical stakeholders.

---

## 5. Production / MLOps Layer (what elevates this from "notebook project" to "system")

| Component | Role |
|---|---|
| **MLflow** | Experiment tracking — params, metrics, model versions, artifacts |
| **DVC** | Data + model + pipeline versioning |
| **Great Expectations** | Automated incoming-data validation |
| **Evidently AI** | Data/feature/prediction drift monitoring |
| **Docker** | Packaging, reproducibility |
| **GitHub Actions (CI/CD)** | Automated test → train → build → deploy |
| **FastAPI** | Serving layer: `/predict_uplift`, `/recommend`, `/counterfactual`, `/metrics`, `/customers` |
| **PostgreSQL** | Store predictions, users, campaigns, reports, experiment logs |
| **Auth** | Admin / Analyst / Read-only stakeholder roles |
| **Cloud deploy** | Docker → AWS EC2 / GCP Cloud Run / Azure App Service |
| **Automated retraining** | Triggered on detected drift |
| **LLM Assistant (optional, downstream-only)** | Natural-language explanation layer over causal outputs — not a causal estimator |

---

## 6. Repository Structure

```text
causal-intelligence-platform/
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── eda/
│   ├── feature_engineering/
│   ├── propensity/
│   ├── learners/
│   │   ├── s_learner.py
│   │   ├── t_learner.py
│   │   ├── x_learner.py
│   │   ├── dr_learner.py
│   │   └── causal_forest.py
│   ├── evaluation/
│   ├── explainability/
│   ├── optimization/
│   ├── recommendation/
│   ├── reports/
│   └── monitoring/
├── api/
│   └── main.py
├── dashboard/
│   └── streamlit_app.py
├── mlruns/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 7. Differentiation vs. Standard ML Portfolio Projects

| Capability | Typical ML project (churn, house price, recs) | This project |
|---|---|---|
| Prediction | ✓ | ✓ |
| Causal reasoning | ✗ | ✓ |
| Multiple estimator architectures | Limited | ✓ (5 learners) |
| Statistical validity checks (overlap, SMD) | ✗ | ✓ |
| Business-constrained optimization | Rare | ✓ |
| Counterfactual analysis | ✗ | ✓ |
| Explainability | Optional | ✓ |
| Experiment tracking | Rare | ✓ |
| Data validation | Rare | ✓ |
| Drift monitoring | Rare | ✓ |
| API + Dashboard | Sometimes | ✓ |
| Deployment | Sometimes | ✓ |

Most student/portfolio ML projects stop at prediction. This project demonstrates the full research → causal ML → decision optimization → productionization → monitoring lifecycle, which is closer to what teams at Uber, DoorDash, Meta, Netflix, Airbnb actually build for growth/marketing decisioning.

---

## 8. Complete Task Set (no phases, no fixed ownership — worked jointly, correctness-gated)

There is no week-by-week schedule and no per-member expertise split here. All three members work across the stack together. What matters is: **every task below has a dependency it sits on, and none of it should be marked done until it's actually verified correct** — a causal platform is uniquely easy to make *look* finished while being statistically wrong underneath (silent confounding, broken overlap, leakage), so verification is not optional polish, it's the deliverable.

### 8.1 Foundational (must exist before anything downstream is trustworthy)
- [ ] Define treatment, outcome, and feature set explicitly; document dataset structure and known limitations
- [ ] Data profiling: missing values, duplicates, categorical vars, outliers, treatment/control distribution
- [ ] Preprocessing pipeline: missing-value handling, encoding, transformations, feature engineering
- [ ] Full EDA: treatment imbalance, feature distributions, outcome differences by arm
- [ ] Document causal assumptions explicitly: SUTVA, positivity/overlap, unconfoundedness — state where they're plausible and where they're shaky for this dataset
- [ ] Repo scaffolding, Docker base environment, reproducible dependency management
- [ ] Baseline predictive model (non-causal) as a reference point for later comparison

**Verification gate:** treatment/control distributions and data quality report reviewed by the whole team before modeling starts — garbage in here invalidates everything after it.

### 8.2 Causal Modeling
- [ ] Propensity score model (logistic regression + tree-based) — `P(Treatment | X)`
- [ ] Overlap plot, propensity distribution, covariate balance (SMD, Love Plot) — verify positivity holds
- [ ] S-Learner implementation
- [ ] T-Learner implementation
- [ ] X-Learner implementation
- [ ] DR-Learner (doubly robust) implementation
- [ ] Causal Forest (EconML / CausalForestDML) implementation
- [ ] Experiment tracking wired in (MLflow) from the first learner onward, not retrofitted later

**Verification gate:** for every learner, sanity-check ATE sign/magnitude against domain intuition and against the naive (unadjusted) treatment-control difference — if a "sophisticated" learner disagrees wildly with the naive estimate, understand *why* before trusting it, don't just take the fancier model on faith.

### 8.3 Evaluation + Decision System
- [ ] Train/val/test split with explicit leakage prevention around treatment assignment
- [ ] Model comparison via Qini, AUUC, uplift curves, CATE distributions (not accuracy/F1)
- [ ] Select best model by causal evaluation criteria, with reasoning documented
- [ ] Customer segmentation into the 4 quadrants, with statistical validation that segments are stable/meaningful, not noise
- [ ] Counterfactual simulator (per-customer what-if)
- [ ] Budget-constrained optimizer (maximize incremental ROI under spend cap)
- [ ] Policy evaluation: compare the model's targeting policy against baselines — **random targeting** and **target-everyone** — report the actual incremental lift over both. This is non-negotiable; an uplift model without this comparison hasn't proven anything.

**Verification gate:** Qini/AUUC results reviewed against baseline policies before the model is called "working." A model that beats random by a trivial margin is not done — investigate.

### 8.4 Productization
- [ ] FastAPI endpoints: `/predict_uplift`, `/recommend`, `/counterfactual`, `/metrics`, `/customers`
- [ ] Streamlit dashboard: Overview, EDA, Propensity, Treatment Effect, Customer Explorer, Budget Optimizer, Policy Simulator, Counterfactual Explorer, Model Comparison, Business Insights
- [ ] PostgreSQL for predictions/campaigns/experiment logs
- [ ] Auth roles (Admin / Analyst / Read-only)
- [ ] Dockerize full application
- [ ] CI/CD via GitHub Actions (test → build → deploy)
- [ ] Cloud deployment (AWS EC2 / GCP Cloud Run / Azure App Service)
- [ ] Monitoring: API health, data/model drift (Evidently AI), logs
- [ ] Automated retraining trigger on detected drift
- [ ] Auto-generated PDF report for stakeholders
- [ ] **Optional, non-mandatory:** LLM analyst layer — natural-language explanation of causal outputs, strictly downstream, never computing estimates

**Verification gate:** end-to-end integration test — raw data in, recommendation + ROI number out, through the actual deployed API/dashboard, not just in a notebook. Load/failure-handling check before calling deployment done.

### 8.5 Final Validation (applies continuously, not just at the end)
- [ ] Subgroup / sensitivity analysis: identify which estimated effects are unstable or shouldn't be trusted, and say so explicitly in the output (flag low-overlap regions rather than silently reporting a number)
- [ ] Documented model limitations and failure conditions
- [ ] Consistency check: dashboard numbers, API responses, and offline analysis all agree
- [ ] Final report tying together EDA → assumptions → treatment effects → segments → business recommendation → ROI

---

## 9. Explicit Design Decisions / Guardrails

1. **LLM is explanatory, never inferential.** It narrates causal outputs; it does not compute or adjust treatment effects. Keep this boundary hard.
2. **Always benchmark policy against baselines** (random targeting, target-everyone, target-by-propensity-alone). Uplift-model ROI claims without this comparison are not credible.
3. **Evaluation must be causal-native** (Qini/AUUC/policy risk), not classification accuracy.
4. **Overlap/positivity checks are a gate, not a footnote.** Where propensity distributions don't overlap, treatment effect estimates there are untrustworthy — surface this in the product (flagged segments/confidence bounds), don't hide it.
5. **Sleeping Dogs segment is a first-class output**, not an afterthought — it's what turns "more marketing" into negative ROI, and it's what differentiates causal targeting from propensity-based targeting.
6. **Nothing is "done" until verified.** Each subsystem above has a stated verification gate — a working demo that hasn't cleared its gate (naive-estimate sanity check, baseline-beating policy, overlap check, end-to-end integration test) is not considered complete. This project's entire value proposition is statistical correctness; a broken causal pipeline that runs without errors is worse than an obviously broken one.
