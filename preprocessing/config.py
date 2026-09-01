"""
Central configuration for the Hillstrom causal-safe preprocessing pipeline.

Everything that another module needs to know about *where* data lives, *what*
each column means causally, and *which* knobs are tunable is declared here so
the pipeline stays declarative and auditable.

Primary dataset: Kevin Hillstrom / MineThatData E-Mail Analytics challenge.
It is a genuine 3-arm marketing RCT (Men's e-mail / Women's e-mail / No e-mail)
with ~64k customers. See docs/dataset_guide.md sections 3 and 14.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_HILLSTROM = (
    REPO_ROOT
    / "datasets"
    / "phase 1 - Main Development"
    / "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "hillstrom"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 20240501
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Causal column map  ---  the heart of the pipeline
# --------------------------------------------------------------------------- #
# `unit`      : identifier / metadata.  NEVER a model feature.
# `treatment` : the randomized intervention.
# `outcome`   : measured in the 2 weeks AFTER the campaign -> targets only.
# `covariate` : measured BEFORE the campaign (historical 12-month snapshot) -> X.
# `derived_redundant` : a deterministic function of another covariate; kept for
#                       EDA / grouping only, excluded from the model matrix.
#
# Hillstrom ships with no customer id.  We attach a surrogate `customer_uid`
# equal to the 0-based row position of the raw file.  It exists purely for
# traceability / joining diagnostics back to rows and is tagged `unit`.

RAW_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

TREATMENT_RAW = "segment"  # {'Mens E-Mail','Womens E-Mail','No E-Mail'}
CONTROL_LABEL = "No E-Mail"

# Outcomes (all POST-treatment, 2-week observation window).
OUTCOMES = ["visit", "conversion", "spend"]
PRIMARY_OUTCOME = "conversion"          # binary conversion uplift
SECONDARY_OUTCOME = "spend"             # monetary / ROI outcome (zero-inflated)
MEDIATOR_OUTCOME = "visit"              # intermediate outcome; also a mediator

# Pre-treatment covariates (the only things allowed into X).
NUMERIC_COVARIATES = ["recency", "history"]
BINARY_COVARIATES = ["mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel"]

# Deterministic coarsening of `history` -> excluded from X, kept for EDA.
DERIVED_REDUNDANT = ["history_segment"]

# --------------------------------------------------------------------------- #
# Cleaning / validation expectations (asserted by the audit + validate steps)
# --------------------------------------------------------------------------- #
EXPECTED_ROWS = 64000
RECENCY_RANGE = (1, 12)          # months since last purchase
HISTORY_MIN_POSITIVE = True      # historical $ spend must be > 0
SPEND_NON_NEGATIVE = True
ZIP_SPELLING_FIX = {"Surburban": "Suburban"}  # fix source typo, values unchanged

HISTORY_SEGMENT_ORDER = [
    "1) $0 - $100",
    "2) $100 - $200",
    "3) $200 - $350",
    "4) $350 - $500",
    "5) $500 - $750",
    "6) $750 - $1,000",
    "7) $1,000 +",
]

# --------------------------------------------------------------------------- #
# Feature-engineering switches (kept conservative & explicit)
# --------------------------------------------------------------------------- #
ADD_HISTORY_LOG = True     # history_log1p : tame right-skew for linear/propensity
ADD_MW_INTERACTIONS = True # mw_count (0/1/2), bought_both : cheap, interpretable
WINSORIZE = False          # never: history / spend tails are genuine heterogeneity
DROP_DUPLICATES = False    # never: identical feature vectors are coincidental,
                           #        not data-entry errors (no id to prove identity)
RESAMPLE_IMBALANCE = False # never: SMOTE / undersampling distorts the RCT and the
                           #        base rate that uplift metrics depend on

# One-hot: keep every level (K dummies).  Tree/uplift learners handle it; linear
# models can regularize or drop a reference downstream.  handle_unknown='ignore'.
ONEHOT_DROP = None

# Optional standardization: delivered as a SEPARATE fitted artifact, never baked
# into the canonical model matrix (tree-based CATE learners neither need nor want
# it, and it destroys coefficient interpretability for the linear propensity
# diagnostic that we *do* want readable).
SCALE_COLUMNS = ["recency", "history", "history_log1p"]

# Overlap / balance diagnostic thresholds.
SMD_FLAG = 0.10                 # |SMD| above this = covariate imbalance worth noting
PROPENSITY_TRIM = (0.01, 0.99)  # positivity: flag mass outside this band
