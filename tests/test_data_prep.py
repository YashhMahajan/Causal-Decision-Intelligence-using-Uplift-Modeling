import numpy as np
import pandas as pd
import pytest

from src.config import HILLSTROM_CSV, X5_DIR
from src.eda.balance import smd
from src.ingestion.validate import quality_report, treatment_outcome_summary
from src.preprocessing import hillstrom, x5

POST_TREATMENT = {"visit", "conversion", "spend", "segment", "arm", "t_binary"}


@pytest.fixture(scope="module")
def hs():
    return hillstrom.preprocess(hillstrom.load_raw())


@pytest.fixture(scope="module")
def x5d():
    return x5.preprocess(*x5.load_raw())


def test_hillstrom_shape_and_no_nans(hs):
    df, feats = hs
    assert len(df) == 64000 and not df[feats].isna().any().any()


def test_no_post_treatment_features(hs, x5d):
    for _, feats in (hs, x5d):
        assert not POST_TREATMENT & set(feats)


def test_typo_fixed(hs):
    assert "zip_code_Surburban" not in hs[0].columns and "zip_code_Suburban" in hs[0].columns


def test_arms_and_binary_treatment_consistent(hs):
    df, _ = hs
    assert set(df.arm) == {0, 1, 2}
    assert (df.t_binary == (df.arm > 0)).all()


def test_hillstrom_is_randomized(hs):
    df, feats = hs
    assert smd(df, "t_binary", feats).abs().max() < 0.05


def test_naive_ate_matches_known_result(hs):
    s = treatment_outcome_summary(hs[0], "t_binary", "conversion")
    assert 0.003 < s["naive_ate"] < 0.007 and s["ci95"][0] > 0


def test_splits_stratified(hs, x5d):
    for df, _ in (hs, x5d):
        assert set(df.split) == {"train", "val", "test"}
        rates = df.groupby("split").t_binary.mean()
        assert rates.max() - rates.min() < 0.005
        y = df.groupby(["split", "t_binary"]).conversion.apply(lambda s: (s > 0).mean()).unstack()
        assert (y.max() - y.min()).max() < 0.01


def test_x5_invalid_ages_flagged(x5d):
    df, _ = x5d
    assert df.age.dropna().between(14, 100).all()
    assert df.age_invalid.sum() == df.age.isna().sum() > 0


def test_x5_unique_clients_and_balance(x5d):
    df, feats = x5d
    assert not df.client_id.duplicated().any()
    assert smd(df.fillna(df.median(numeric_only=True)), "t_binary", feats).abs().max() < 0.05


def test_x5_10k_sample_keeps_rates(x5d):
    df, _ = x5d
    s = x5.sample_10k(df)
    assert abs(len(s) - 10000) < 10
    assert abs(s.conversion.mean() - df.conversion.mean()) < 0.01


def test_missing_column_raises():
    with pytest.raises(ValueError):
        hillstrom.preprocess(hillstrom.load_raw().drop(columns=["channel"]))


def test_quality_report_flags_dup_ids():
    r = quality_report(pd.DataFrame({"id": [1, 1], "a": [1, np.nan]}), ["id", "a", "b"], "id")
    assert r["missing_columns"] == ["b"] and r["duplicate_ids"] == 1 and r["quality_score"] < 100
