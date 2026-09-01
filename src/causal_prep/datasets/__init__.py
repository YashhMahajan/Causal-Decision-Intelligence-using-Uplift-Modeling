"""Dataset-specific modules. Each exposes: NAME, load_raw(), clean(), SPEC,
feature_classification(), and (optionally) extra_raw_audit()."""

from . import hillstrom, x5, lenta

REGISTRY = {m.NAME: m for m in (hillstrom, x5, lenta)}
