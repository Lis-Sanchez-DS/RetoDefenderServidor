# Pipeline de Prefect

Archivo: `src/pipeline/flow.py` (flujo `defender-servidor`). Configuración en la sección `pipeline` de `params.yaml`.

## Etapas
| Etapa (tarea) | Qué hace | Cuándo falla (y qué dice) |
|---|---|---|
| `preparar-datos` | Comprueba que `UNSW_NB15_training-set.csv` existe y que su md5 coincide con el `.dvc`; carga sin los estados excluidos; construye los 5 pliegues por grupos | Falta el `.dvc` o el archivo (sugiere `dvc pull`); el md5 no coincide con la versión de DVC; falta una clase o hay columnas vacías |
| `entrenar-modelo` (x3) | Validación cruzada por grupos y reajuste final de cada candidato: regresión logística, XGBoost por defecto y XGBoost con los parámetros de Optuna (`reports/optuna_mejores_parametros.json`). Registra cada uno en MLflow con el commit de Git, el md5 de los datos y el id de la ejecución de Prefect | Cualquier excepción del entrenamiento, con el nombre del modelo y el error |
| `comparar-experimentos` | Con las predicciones fuera de muestra y el umbral congelado (0,6), calcula recall, precisión, macro-F1 y MCC por pliegue; solo admite modelos con recall >= 0,95 en **todos** los pliegues y elige el de mayor precisión con recall >= 0,95. Guarda `reports/pipeline_comparacion.csv` | Ningún modelo cumple el piso de recall (indica el mejor recall mínimo obtenido) |
| `registrar-modelo` | Registra el elegido en el registro de MLflow como `defender-servidor`, con las etiquetas `umbral` y `origen` y el alias `candidato` | No se puede registrar el modelo |

Si una tarea falla, Prefect marca la tarea y el flujo como `Failed` y el mensaje de la excepción (clase `EtapaFallida`) explica la causa.

## Decisiones de diseño
- **No evalúa en la prueba.** La evaluación en la prueba se hizo una sola vez (`src/evaluar_test.py`) con el modelo y el umbral congelados; repetirla en cada ejecución del pipeline la convertiría en parte del ajuste.
- **No repite la búsqueda de Optuna** (42 min): usa los mejores parámetros guardados. La búsqueda sigue siendo un experimento aparte.
- **Modelos con prefijo `pipeline-`**: no pisan los modelos congelados de los experimentos, cuyos `.joblib` tienen su md5 en DVC.
- **El umbral viene de `params.yaml`** (0,6, decisión del experimento 6), no se vuelve a elegir en cada ejecución.
- **Ejecución rápida** (`--muestra 0.05`): usa una fracción de las filas para comprobar el flujo; los modelos llevan el sufijo `-muestra` y se registran como `defender-servidor-muestra`, así no contaminan el registro real.
- **Demostración de fallo**: `--piso-recall 0.999` hace fallar `comparar-experimentos` con un mensaje claro (ningún modelo cumple el piso). Las pruebas (`tests/test_pipeline.py`) cubren el md5 que no coincide, el archivo ausente y el piso de recall.

## Infraestructura
Prefect 3 arranca un servidor temporal local en cada ejecución (no hace falta instalar nada más). En este entorno (WSL) arrancarlo puede tardar, por lo que se usa `PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=120`; con `ulimit -v` el servidor no arranca. Para ver las ejecuciones en la interfaz web se puede levantar un servidor persistente (`prefect server start`, en el puerto 4200) y apuntar a él con `PREFECT_API_URL=http://127.0.0.1:4200/api`.

## Cómo ejecutarlo
```bash
export PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=120
.venv/bin/python -m src.pipeline.flow                    # completo (~5 min)
.venv/bin/python -m src.pipeline.flow --muestra 0.05     # rápido
.venv/bin/python -m src.pipeline.flow --piso-recall 0.999  # demostración de fallo
```

## Resultado de la ejecución completa (2026-10-05, 5 min)
Precisión con recall >= 0,95 (media de los 5 pliegues) y métricas con el umbral 0,6:

| Modelo | Precisión con recall ≥ 0,95 | Recall (umbral 0,6) | Recall mínimo por pliegue | Precisión | MCC |
|---|---|---|---|---|---|
| `pipeline-xgboost-optuna` (elegido) | 0,9797 | 0,966 | 0,964 | 0,971 | 0,902 |
| `pipeline-xgboost` | 0,9770 | 0,961 | 0,960 | 0,971 | 0,895 |
| `pipeline-regresion-logistica` | 0,9508 | 0,976 | 0,975 | 0,933 | 0,850 |

- **Reproducibilidad:** los valores de XGBoost con Optuna (0,9797) y XGBoost por defecto (0,977) coinciden con los de los experimentos 4 y 5, y recall/precisión/MCC con umbral 0,6 coinciden con los del experimento 6: el pipeline reproduce los resultados.
- Modelo registrado: `defender-servidor` con alias `candidato` (la versión 2; la versión 1 es de una ejecución rápida previa, hecha antes de separar las ejecuciones con muestra en `defender-servidor-muestra`).
- Comparación completa: `reports/pipeline_comparacion.csv`. Los modelos y predicciones fuera de muestra de `pipeline-*` están versionados con DVC.
