"""Central paths, seeds and dataset constants. Everything downstream imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets"
PROCESSED_DIR = ROOT / "data" / "processed"
VALIDATION_DIR = ROOT / "data" / "validation"

HILLSTROM_CSV = (
    RAW_DIR / "phase 1 - Main Development"
    / "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
X5_DIR = RAW_DIR / "phase 2 - generalization" / "x5-retail-hero-uplift-raw-data"

SEED = 42
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}

# Business assumptions for the decision layer. Hillstrom ships no cost data, so these are
# ASSUMPTIONS, not facts. Override at the call site; always report them next to any ROI number.
ASSUMED_COST_PER_CONTACT = 0.10  # USD per email sent
