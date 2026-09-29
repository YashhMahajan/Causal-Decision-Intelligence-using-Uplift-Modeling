"""X5 RetailHero preprocessing (clients + uplift_train only).

NOTE: purchases.csv (the large transaction log) is NOT in the repo, so purchase-history
features cannot be built yet. Only demographic / loyalty-card features are available.
Only uplift_train has labels; uplift_test is the competition hold-out and is unused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEED, X5_DIR
from src.preprocessing.splits import assign_splits

AGE_MIN, AGE_MAX = 14, 100   # raw ages include -7491 and 1901; treated as missing


def load_raw(directory=X5_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(directory / "clients.csv", parse_dates=["first_issue_date", "first_redeem_date"])
    train = pd.read_csv(directory / "uplift_train.csv")
    return clients, train


def preprocess(clients: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if train["client_id"].duplicated().any():
        raise ValueError("Duplicate client_id in uplift_train")
    df = train.merge(clients, on="client_id", how="left", validate="one_to_one")
    if df["first_issue_date"].isna().any():
        raise ValueError("uplift_train clients missing from clients.csv")

    ref = clients["first_issue_date"].max()           # fixed reference date, same for every row
    age_ok = df["age"].between(AGE_MIN, AGE_MAX)
    out = pd.DataFrame({"client_id": df["client_id"]})
    out["age"] = df["age"].where(age_ok)              # NaN kept; tree learners handle it, else impute on train only
    out["age_invalid"] = (~age_ok).astype(int)
    out["gender_F"] = (df["gender"] == "F").astype(int)
    out["gender_M"] = (df["gender"] == "M").astype(int)   # U (unknown) = reference
    out["days_since_issue"] = (ref - df["first_issue_date"]).dt.days
    out["ever_redeemed"] = df["first_redeem_date"].notna().astype(int)
    out["days_issue_to_redeem"] = (df["first_redeem_date"] - df["first_issue_date"]).dt.days
    features = [c for c in out.columns if c != "client_id"]

    out["t_binary"] = df["treatment_flg"].astype(int)
    out["conversion"] = df["target"].astype(int)
    out["split"] = assign_splits(out, "t_binary", "conversion")
    return out, features


def sample_10k(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Stratified (treatment x outcome) 10K benchmark subset, per the dataset guide."""
    frac = 10_000 / len(df)
    key = df["t_binary"].astype(str) + "_" + df["conversion"].astype(str)
    return df.groupby(key, group_keys=False).sample(frac=frac, random_state=seed).reset_index(drop=True)
