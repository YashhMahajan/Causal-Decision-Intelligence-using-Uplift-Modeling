# Causal Decision Intelligence using Uplift Modeling

Estimate *who a campaign actually changes* (ITE/CATE), then choose whom to target under a budget.
Design: [docs/knowledge_base_1.md](docs/knowledge_base_1.md) · Datasets: [docs/dataset_guide.md](docs/dataset_guide.md) ·
Status / roadmap: [docs/status_and_next_steps.md](docs/status_and_next_steps.md) · Assumptions: [docs/causal_assumptions.md](docs/causal_assumptions.md)

## Setup
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m scripts.prepare_data      # datasets/ (raw) -> data/processed/*.parquet + *.meta.json
.venv/bin/python -m pytest -q
```

## Data contract (every processed dataset)
`<name>.parquet` + `<name>.meta.json`. Columns: features (listed in meta), `t_binary` (0/1 treatment),
`conversion` (binary outcome), `split` (train/val/test, stratified on treatment x outcome).
Hillstrom also has `arm` (0 control/1 men's/2 women's), `spend`, `visit`. Downstream code must read the feature list
from meta — never infer features from column names, or outcomes leak in.
