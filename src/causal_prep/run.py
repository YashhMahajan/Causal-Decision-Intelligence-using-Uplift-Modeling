"""
Reproducible driver for the causal-safe preprocessing standard.

    python -m causal_prep.run hillstrom
    python -m causal_prep.run x5
    python -m causal_prep.run lenta
    python -m causal_prep.run all

For each dataset:
    raw -> audit -> clean -> causal feature audit -> stratified split
        -> fit preprocessors on TRAIN only -> transform -> save -> validate

Raw files under data/raw/ are never written to.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import config as C
from . import common, reporting
from .common import jsonable
from .datasets import REGISTRY


def run_one(name: str) -> dict:
    mod = REGISTRY[name]
    spec = mod.SPEC
    dirs = C.ensure_dirs(name)
    print("=" * 78)
    print(f"DATASET: {name}")
    print("=" * 78)

    # 1. load + clean (deterministic, no row loss) -----------------------
    raw = mod.load_raw()
    df_clean = mod.clean(raw)
    assert len(df_clean) == len(raw), "clean() changed row count!"
    print(f"[clean]  rows={len(df_clean)}  cols={df_clean.shape[1]}  "
          f"(raw cols={raw.shape[1]})")

    # 2. audit (on cleaned frame) -------------------------------------
    extra = mod.extra_raw_audit(df_clean) if hasattr(mod, "extra_raw_audit") else None
    audit = common.audit_raw(df_clean, spec, extra)
    (dirs["reports"] / "audit.json").write_text(json.dumps(jsonable(audit), indent=2))
    rc = audit["randomization_check"]
    print(f"[audit]  missing_total={audit['missing_total']}  "
          f"max|SMD|={rc['max_abs_smd']:.4f}  propensityAUC={rc['propensity_auc_5fold']:.4f}")
    print(f"[audit]  naive ATE: {audit['naive_unadjusted_ATE']}")

    # 3. causal feature audit ---------------------------------------
    reporting.write_feature_classification(mod, dirs["reports"])
    fc = mod.feature_classification()
    print(f"[feataudit] {len(fc)} columns classified -> "
          f"{dirs['reports'].relative_to(C.REPO_ROOT)}/feature_classification.csv")

    # 4. stratified split -----------------------------------------
    df_split = common.make_split(df_clean, spec)
    n_tr = int((df_split.split == "train").sum())
    print(f"[split]  train={n_tr}  test={len(df_split)-n_tr}  "
          f"stratified on {spec.stratify_cols}")

    # 5-7. fit-on-train + transform + save --------------------------
    score_frame = mod.load_score_frame() if hasattr(mod, "load_score_frame") else None
    res = common.fit_transform_save(df_split, spec, dirs, score_frame)
    print(f"[save]   X features: {len(res['feature_names'])}  "
          f"-> data/processed/{name}/  artifacts/{name}/")

    # 8. validate -------------------------------------------------
    train, test = res["train"], res["test"]
    integ = common.integrity_checks(train, test, res["feature_spec"], spec)
    tox = common.treatment_outcome_diag(train, test, spec)
    bal = common.balance_overlap(train, res["feature_spec"], spec, dirs["figures"])
    repro = common.reproducibility_hash(lambda: mod.clean(mod.load_raw()), spec)

    (dirs["reports"] / "validation.json").write_text(
        json.dumps(jsonable({"integrity": integ, "treatment_outcome": tox,
                             "balance_overlap": bal, "reproducibility": repro}),
                   indent=2))
    reporting.write_data_quality(name, audit, integ, repro, dirs["reports"], spec)
    reporting.write_balance_overlap(name, tox, bal, dirs["reports"], spec)

    # hard gates
    problems = []
    if not integ["row_count_conserved"]:
        problems.append("row count not conserved")
    if not integ["no_unit_overlap"]:
        problems.append("unit id overlap between train and test")
    if integ["X_missing_cells"] != 0:
        problems.append(f"{integ['X_missing_cells']} missing cells in X")
    if integ["X_non_finite_cells"] != 0:
        problems.append(f"{integ['X_non_finite_cells']} non-finite cells in X")
    if integ["leakage_outcome_or_excluded_in_X"]:
        problems.append(f"leakage: {integ['leakage_outcome_or_excluded_in_X']} in X")
    if not integ["X_all_numeric"]:
        problems.append("non-numeric column in X")
    print(f"[validate] integrity: {integ}")
    print(f"[validate] max|SMD|(train)={bal['max_abs_smd']:.4f}  "
          f"propensityAUC logreg={bal['propensity_logreg']['auc']:.4f} "
          f"hgb={bal['propensity_hgb']['auc']:.4f}")
    if problems:
        print("[validate] !!! FAILED GATES: " + "; ".join(problems))
        raise SystemExit(1)
    print(f"[validate] ALL HARD GATES PASSED for {name}\n")
    return {"audit": audit, "integrity": integ, "balance": bal,
            "feature_spec": res["feature_spec"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=list(REGISTRY) + ["all"])
    args = ap.parse_args(argv)
    np.random.seed(C.RANDOM_SEED)
    targets = list(REGISTRY) if args.dataset == "all" else [args.dataset]
    summary = {}
    for name in targets:
        summary[name] = run_one(name)
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, s in summary.items():
        fs = s["feature_spec"]
        print(f"  {name:10s}  n={fs['n_rows_total']:>7}  X={len(fs['X_features']):>3}  "
              f"train/test={fs['n_train']}/{fs['n_test']}  "
              f"max|SMD|={s['balance']['max_abs_smd']:.3f}  "
              f"propAUC={s['balance']['propensity_logreg']['auc']:.3f}")


if __name__ == "__main__":
    main()
