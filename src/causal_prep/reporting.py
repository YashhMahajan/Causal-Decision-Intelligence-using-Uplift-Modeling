"""Markdown report writers — one consistent set of documents per dataset."""

from __future__ import annotations

import json
from pathlib import Path

from . import config as C


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_feature_classification(module, reports_dir: Path) -> None:
    module.feature_classification().to_csv(
        reports_dir / "feature_classification.csv", index=False)


def write_data_quality(name, audit, integrity, repro, reports_dir: Path,
                       spec) -> None:
    L = [f"# {name} — Preprocessing & Data-Quality Report", "",
         f"_Seed {C.RANDOM_SEED}; stratified split on "
         f"`{' , '.join(spec.stratify_cols)}`; test fraction {C.TEST_SIZE}._", "",
         "## 1. Shape & missingness", "",
         f"- cleaned frame: **{audit['n_rows']} rows × {audit['n_cols']} cols**",
         f"- total missing cells (post-clean): **{audit['missing_total']}**"]
    if audit["missing"]:
        top = sorted(audit["missing"].items(), key=lambda kv: -kv[1])[:12]
        L.append("- columns with missing values (top 12): "
                 + ", ".join(f"`{k}`={v}" for k, v in top))
    L += ["", "## 2. Duplicates & invalid values", "",
          f"- unit-id duplicates: **{audit['duplicates']['unit_id_duplicated']}**",
          f"- full-row duplicates (extra): **{audit['duplicates']['full_row_duplicated_extra']}**"]
    if audit["invalid_value_checks"]:
        L.append("- range checks (count outside allowed range): "
                 + ", ".join(f"`{k}`={v}" for k, v in
                             audit["invalid_value_checks"].items()))
    L += ["", "## 3. Treatment & outcomes (cleaned frame)", "",
          f"- treatment `{audit['treatment']['primary_binary']}` — arm counts "
          f"`{audit['treatment']['arm_counts']}`; P(treated) = "
          f"{_fmt(audit['treatment']['p_treated'])}",
          "- outcome rates by arm:"]
    for arm, d in audit["outcomes_by_arm"].items():
        L.append(f"  - **{arm}**: " + ", ".join(f"{k} {_fmt(v)}" for k, v in d.items()))
    L.append("- naive (unadjusted) ATE vs control:")
    for k, d in audit["naive_unadjusted_ATE"].items():
        L.append(f"  - **{k}**: " + ", ".join(f"{o} {v:+.4f}" for o, v in d.items()))
    if audit["class_balance"]:
        L.append("- binary-outcome base rates: "
                 + ", ".join(f"`{k}` {_fmt(v)}" for k, v in audit["class_balance"].items())
                 + "  → **no resampling / SMOTE applied**")
    rc = audit["randomization_check"]
    L += ["", "## 4. Randomization sanity", "",
          f"- max |SMD| (treated vs control) = **{_fmt(rc['max_abs_smd'])}** "
          f"(flag {C.SMD_FLAG}); features above flag: {rc['n_smd_above_flag']}",
          f"- 5-fold propensity AUC = **{_fmt(rc['propensity_auc_5fold'])}** "
          f"(n={rc.get('propensity_diag_n','?')}); support "
          f"[{_fmt(rc['propensity_min'])}, {_fmt(rc['propensity_max'])}]; "
          f"mass outside trim {_fmt(rc['propensity_mass_outside_trim'])}"]
    if "dataset_specific" in audit:
        L += ["", "## 5. Dataset-specific audit", "", "```json",
              json.dumps(audit["dataset_specific"], indent=2), "```"]
    L += ["", "## 6. Integrity checks (processed data)", ""]
    for k, v in integrity.items():
        L.append(f"- `{k}` = `{v}`")
    L += ["", "## 7. Reproducibility", ""]
    for k, v in repro.items():
        L.append(f"- `{k}` = `{v}`")
    L += ["", "## 8. Causal-safety decisions", ""]
    for n in spec.notes:
        L.append(f"- {n}")
    L.append("")
    L.append("### Excluded from X (and why)")
    for col, why in spec.excluded_from_x.items():
        L.append(f"- `{col}` — {why}")
    (reports_dir / "data_quality.md").write_text("\n".join(L))


def write_balance_overlap(name, tox, bal, reports_dir: Path, spec) -> None:
    L = [f"# {name} — Treatment/Control & Overlap Diagnostics", ""]
    for sp in ("train", "test"):
        L += [f"## {sp}", "",
              f"- arm counts: `{tox[sp]['arm_counts']}`", "",
              "| arm | " + " | ".join(spec.outcomes) + " |",
              "|---" * (len(spec.outcomes) + 1) + "|"]
        for arm, d in tox[sp]["outcome_rates_by_arm"].items():
            L.append(f"| {arm} | " + " | ".join(f"{d[o]:.4f}" for o in spec.outcomes) + " |")
        L.append("")
        L.append("- naive ATE vs control:")
        for arm, d in tox[sp]["naive_ATE_vs_control"].items():
            L.append(f"  - **{arm}**: " + ", ".join(f"{o} {v:+.4f}" for o, v in d.items()))
        L.append("")
    L += ["## Covariate balance (train)", "",
          f"- max |SMD| = **{bal['max_abs_smd']:.4f}** (flag {bal['smd_flag_threshold']}); "
          f"features above flag: **{bal['n_smd_above_flag']}**", ""]
    worst = sorted(bal["smd"].items(), key=lambda kv: -abs(kv[1]))[:20]
    L += ["| feature (top 20 by |SMD|) | SMD |", "|---|---:|"]
    for k, v in worst:
        L.append(f"| {k} | {v:+.4f} |")
    L += ["", "## Propensity / positivity", "",
          f"(diagnostic n = {bal.get('propensity_diag_n','all')})", ""]
    for label in ("logreg", "hgb"):
        d = bal[f"propensity_{label}"]
        L.append(f"- **{label}**: AUC {d['auc']:.4f}; support "
                 f"[{d['min']:.3f}, {d['max']:.3f}] "
                 f"(p01 {d['p01']:.3f}, p99 {d['p99']:.3f}); "
                 f"mass outside trim {d['mass_outside_trim']:.4f}")
    L += ["", f"**Verdict:** {bal['positivity_verdict']}", "",
          "![propensity overlap](figures/propensity_overlap.png)", "",
          "![love plot](figures/love_plot.png)", ""]
    (reports_dir / "balance_overlap.md").write_text("\n".join(L))
