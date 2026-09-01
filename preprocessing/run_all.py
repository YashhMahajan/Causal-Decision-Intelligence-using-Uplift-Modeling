"""
One-shot reproducible driver:

    Raw  ->  Audit  ->  Clean  ->  Causal Feature Audit  ->  Split
         ->  Fit preprocessors on TRAIN  ->  Transform  ->  Save  ->  Validate

Usage:
    python -m preprocessing.run_all

Outputs:
    data/processed/hillstrom/   (processed datasets + fitted artifacts + spec)
    reports/                    (audit json, data-quality report, feature table,
                                 balance/overlap report, figures)

The files under datasets/ are never written to.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .hillstrom_audit import audit, load_raw
from .hillstrom_preprocess import (
    causal_feature_audit,
    clean,
    fit_transform_save,
    split,
)
from . import hillstrom_audit as _audit_mod
from . import hillstrom_validate as _validate_mod


def main() -> None:
    np.random.seed(C.RANDOM_SEED)

    print("=" * 72)
    print("STEP 1/6  RAW AUDIT (read-only)")
    print("=" * 72)
    _audit_mod.main()

    print("\n" + "=" * 72)
    print("STEP 2/6  CLEAN (deterministic, no row loss)")
    print("=" * 72)
    raw = load_raw()
    df_clean = clean(raw)
    print(f"[clean] rows in={len(raw)} rows out={len(df_clean)} "
          f"(delta {len(df_clean) - len(raw)})")
    print(f"[clean] new cols: "
          f"{[c for c in df_clean.columns if c not in raw.columns]}")

    print("\n" + "=" * 72)
    print("STEP 3/6  CAUSAL FEATURE AUDIT")
    print("=" * 72)
    fc = causal_feature_audit(df_clean)
    print(fc.to_string(index=False, max_colwidth=60))

    print("\n" + "=" * 72)
    print("STEP 4/6  SPLIT (stratified: treatment_arm x conversion)")
    print("=" * 72)
    df_split = split(df_clean)
    vc = df_split.groupby(["split", "treatment_arm"], observed=True).size().unstack()
    print(vc)

    print("\n" + "=" * 72)
    print("STEP 5/6  FIT-ON-TRAIN + TRANSFORM + SAVE")
    print("=" * 72)
    res = fit_transform_save(df_split)
    print(f"[save] X features ({len(res['feature_names'])}): "
          f"{res['feature_names']}")
    print(f"[save] artifacts -> {C.PROCESSED_DIR.relative_to(C.REPO_ROOT)}/")

    print("\n" + "=" * 72)
    print("STEP 6/6  VALIDATE")
    print("=" * 72)
    _validate_mod.main()

    print("\nDONE.  See reports/ for the written deliverables.")


if __name__ == "__main__":
    main()
