# Defiende el servidor — UNSW-NB15

Proyecto de MLOps: distinguir tráfico normal de ataques en UNSW-NB15, con datos,
experimentos, modelos y pipeline reproducibles (Git + DVC + MLflow + Prefect).

## Estado
Solo la estructura base. Aún no se han descargado datos ni entrenado modelos.

## Instalación
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # versiones exactas: requirements.lock.txt
```

## Estructura
- `src/pipeline/` — flujo de Prefect (preparar → entrenar → comparar → registrar)
- `data/raw/unsw-nb15/` (`particion_oficial/`, `completo/`, `metadatos/`), `data/processed`, `models/` — versionados con DVC (no con Git)
- `params.yaml` — configuración central
- `reports/` — tabla de experimentos y decisión final
- `notebooks/` — exploración
- `tests/` — pruebas

## Responsabilidad de cada herramienta
| Herramienta | Función |
|---|---|
| Git | código y decisiones |
| DVC | versiones de datos y modelos |
| MLflow | experimentos y registro de modelos |
| Prefect | orquestación del pipeline |

## Datos
UNSW-NB15 (UNSW Canberra). La fuente, la versión y las condiciones de uso se
documentarán cuando se descarguen.
