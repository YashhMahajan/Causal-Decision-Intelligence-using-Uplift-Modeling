"""
causal_prep — causal-safe, reproducible preprocessing for uplift / HTE modelling.

Datasets: hillstrom (3-arm e-mail RCT), x5 (retail loyalty RCT), lenta (grocery RCT).

Standard pipeline (identical for every dataset, dataset-specific logic isolated in
``causal_prep.datasets.<name>``):

    raw -> audit -> clean -> causal feature audit -> stratified split
        -> fit preprocessors on TRAIN only -> transform -> save -> validate

Entry point:  python -m causal_prep.run {hillstrom|x5|lenta|all}
"""

__all__ = ["config", "common", "reporting", "run", "datasets"]
