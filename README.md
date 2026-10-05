# Defiende el servidor — UNSW-NB15

MLOps project: detect attack vs. normal traffic on UNSW-NB15, with reproducible
data, experiments, models and pipeline (Git + DVC + MLflow + Prefect).

## Status
Scaffolding only. No data downloaded and no models trained yet.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # exact versions: requirements.lock.txt
```

## Layout
- `src/pipeline/` — Prefect flow (prepare → train → compare → register)
- `data/raw`, `data/processed`, `models/` — tracked with DVC (not Git)
- `params.yaml` — central configuration
- `reports/` — experiment table and final decision
- `notebooks/` — exploration
- `tests/` — tests

## Tool responsibilities
| Tool | Role |
|---|---|
| Git | code and decisions |
| DVC | data and model versions |
| MLflow | experiments and model registry |
| Prefect | pipeline orchestration |

## Data
UNSW-NB15 (UNSW Canberra). Source, version and usage conditions: to be documented
when downloaded.
