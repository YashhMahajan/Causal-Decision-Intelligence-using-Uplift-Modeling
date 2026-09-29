"""Download Lenta and MegaFon (NOT in the repo) into data/raw/. Needs network + scikit-uplift.
    .venv/bin/python -m scripts.fetch_uplift_benchmarks
Both are then reduced to a stratified 10K benchmark subset (data/processed/{lenta,megafon}_10k.parquet)
with columns: features..., t_binary, conversion, split.
"""
from __future__ import annotations

import json

from sklift.datasets import fetch_lenta, fetch_megafon

from src.config import PROCESSED_DIR, ROOT, SEED
from src.preprocessing.splits import assign_splits


def _prep(name, bunch):
    df = bunch.data.copy()
    treat = bunch.treatment
    if treat.dtype == object:  # Lenta: 'test'/'control'; MegaFon: 'treatment'/'control'
        treat = (treat != "control").astype(int)
    df["t_binary"] = treat.astype(int).values
    df["conversion"] = bunch.target.astype(int).values
    key = df["t_binary"].astype(str) + "_" + df["conversion"].astype(str)
    df = df.groupby(key, group_keys=False).sample(frac=min(1, 10_000 / len(df)), random_state=SEED)
    df = df.reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("t_binary", "conversion")]
    cats = [c for c in feats if df[c].dtype == object]
    df = df.join(__import__("pandas").get_dummies(df[cats], drop_first=True, dtype=int)).drop(columns=cats)
    feats = [c for c in df.columns if c not in ("t_binary", "conversion")]
    df["split"] = assign_splits(df, "t_binary", "conversion")
    df.to_parquet(PROCESSED_DIR / f"{name}_10k.parquet", index=False)
    (PROCESSED_DIR / f"{name}_10k.meta.json").write_text(json.dumps(
        {"features": feats, "treatment": "t_binary", "outcomes": ["conversion"], "n_rows": len(df)}, indent=2))
    print(name, df.shape, "treated share", df.t_binary.mean(), "conv", df.conversion.mean())


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, fetch in (("lenta", fetch_lenta), ("megafon", fetch_megafon)):
        _prep(name, fetch(data_home=str(ROOT / "data" / "raw" / name)))
