"""Raw -> data/processed. Run:  .venv/bin/python -m scripts.prepare_data"""
from __future__ import annotations

import json

from src.config import PROCESSED_DIR, VALIDATION_DIR
from src.eda.balance import smd
from src.ingestion.validate import quality_report, treatment_outcome_summary
from src.preprocessing import hillstrom, x5


def _save(name, df, features, extra):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    meta = {"features": features, "n_rows": len(df), **extra}
    (PROCESSED_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"{name}: {df.shape}, {len(features)} features")


def main():
    raw = hillstrom.load_raw()
    q = quality_report(raw, hillstrom.RAW_COLUMNS)
    df, feats = hillstrom.preprocess(raw)
    binary = treatment_outcome_summary(df, "t_binary", "conversion")
    spend = treatment_outcome_summary(df, "t_binary", "spend")
    bal = smd(df, "t_binary", feats)
    _save("hillstrom", df, feats, {
        "treatment": "t_binary", "arm": "arm", "outcomes": hillstrom.OUTCOMES,
        "quality": q, "naive_conversion": binary, "naive_spend": spend,
        "max_abs_smd": float(bal.abs().max()),
        "split_counts": df["split"].value_counts().to_dict()})

    clients, train = x5.load_raw()
    xdf, xfeats = x5.preprocess(clients, train)
    _save("x5", xdf, xfeats, {
        "treatment": "t_binary", "outcomes": ["conversion"],
        "quality": quality_report(train, ["client_id", "treatment_flg", "target"], "client_id"),
        "naive_conversion": treatment_outcome_summary(xdf, "t_binary", "conversion"),
        "max_abs_smd": float(smd(xdf.fillna(xdf.median(numeric_only=True)), "t_binary", xfeats).abs().max())})
    s = x5.sample_10k(xdf)
    _save("x5_10k", s, xfeats, {"treatment": "t_binary", "outcomes": ["conversion"]})


if __name__ == "__main__":
    main()
