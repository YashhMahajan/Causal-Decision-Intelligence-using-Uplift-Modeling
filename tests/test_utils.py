import numpy as np
import pandas as pd
import pytest

from src.eda.balance import smd
from src.ingestion.validate import quality_report, treatment_outcome_summary
from src.preprocessing.splits import assign_splits


def _rct(n=20000, seed=0):
    r = np.random.default_rng(seed)
    t = r.integers(0, 2, n)
    return pd.DataFrame({"x": r.normal(size=n), "t": t, "y": (r.random(n) < 0.05 + 0.03 * t).astype(int)})


def test_naive_ate_recovers_effect():
    s = treatment_outcome_summary(_rct(), "t", "y")
    assert 0.02 < s["naive_ate"] < 0.04 and s["ci95"][0] > 0


def test_smd_small_when_randomized():
    assert abs(smd(_rct(), "t", ["x"]).iloc[0]) < 0.05


def test_splits_stratified_and_disjoint():
    df = _rct()
    s = assign_splits(df, "t", "y")
    assert set(s) == {"train", "val", "test"} and s.notna().all()
    d = df.assign(split=s)
    assert d.groupby("split").t.mean().pipe(lambda r: r.max() - r.min()) < 0.01
    assert d.groupby(["split", "t"]).y.mean().unstack().pipe(lambda r: (r.max() - r.min()).max()) < 0.01


def test_quality_report_flags_dup_ids():
    r = quality_report(pd.DataFrame({"id": [1, 1], "a": [1, np.nan]}), ["id", "a", "b"], "id")
    assert r["missing_columns"] == ["b"] and r["duplicate_ids"] == 1 and r["quality_score"] < 100


def test_treatment_summary_needs_both_arms():
    with pytest.raises(ValueError):
        treatment_outcome_summary(pd.DataFrame({"t": [1, 1], "y": [0, 1]}), "t", "y")
