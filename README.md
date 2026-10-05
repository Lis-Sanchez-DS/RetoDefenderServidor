# Defiende el servidor — UNSW-NB15

Proyecto de MLOps: distinguir tráfico normal de ataques en UNSW-NB15, con datos,
experimentos, modelos y pipeline reproducibles (Git + DVC + MLflow + Prefect).

## Estado
Hecho: exploración y documentación de datos, línea base, 6 experimentos (regresión logística y XGBoost, validación cruzada por bloques y por grupos, búsqueda con Optuna, elección de umbral), seguimiento en MLflow y versionado de datos y modelos con DVC.

Modelo candidato: XGBoost ajustado con Optuna (`cv-grupos-xgboost-optuna`), umbral 0,6.

Pendiente: evaluación final única en la prueba, pipeline de Prefect real, remoto de DVC y demostración de recuperación, decisión final.

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
UNSW-NB15 (UNSW Canberra). Fuente, versión, condiciones de uso y sumas de verificación en `docs/datos.md`. Los datos no están en el repositorio público: se descargan a mano y se versionan con DVC (`data/raw/unsw-nb15/`).

## Documentación
- `docs/datos.md` — fuente, términos, citas y sumas de verificación.
- `docs/exploracion.md` — exploración: tamaños, capturas, variables, categóricas.
- `docs/decisiones.md` — decisiones y su justificación (costos, codificación, `proto`, piso de recall, validación cruzada, Optuna, umbral).
- `docs/duplicados.md` — filas casi idénticas, intentos y la decisión de pliegues por grupos.
- `reports/experimentos.md` — los experimentos con la hipótesis escrita antes de entrenar, resultados y conclusiones.

## Cómo reproducir
```bash
.venv/bin/python -m src.baseline                          # línea base
.venv/bin/python -m src.experimentos                      # exp. 1 y 2 (regresión logística)
.venv/bin/python -m src.validacion_cruzada                # exp. 3 (por bloques)
.venv/bin/python -m src.validacion_cruzada --grupos       # exp. 4 (por grupos)
.venv/bin/python -m src.optuna_xgboost 50                 # exp. 5 (Optuna, ~45 min)
.venv/bin/python -m src.evaluar_test <modelo>             # evaluación en la prueba (una sola vez)
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/pytest
```
Las bases `mlflow.db` y `optuna.db` son locales y no se versionan.
