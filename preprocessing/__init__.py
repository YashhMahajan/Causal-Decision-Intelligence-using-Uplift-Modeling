"""Causal-safe preprocessing pipeline for the Causal Decision Intelligence project.

Pipeline order:  Raw -> Audit -> Clean -> Causal Feature Audit -> Split
                 -> Fit preprocessors on Train -> Transform -> Validate.

The raw datasets under ``datasets/`` are treated as read-only.
"""
