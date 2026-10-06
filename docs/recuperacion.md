# Reconstruir el proyecto desde cero

Pregunta de la guía: "Si perdieras tu entorno local, ¿qué necesitarías para reconstruirlo?"

## Qué vive dónde
| Qué | Dónde | Cómo se recupera |
|---|---|---|
| Código, documentos, parámetros, resultados (`reports/`), punteros `.dvc` | GitHub (público) | `git clone` |
| Versiones exactas de las dependencias (Python 3.14) | `requirements.lock.txt` | `pip install -r requirements.lock.txt` |
| Datos (CSV de UNSW-NB15), modelos, predicciones fuera de muestra | Remoto de DVC: bucket privado `pipeline-mlops` en Cloudflare R2 | `dvc pull` con credenciales |
| Credenciales del bucket | Solo en `.dvc/config.local` (ignorado por Git) | El dueño las vuelve a crear en Cloudflare (R2 → API Tokens) |
| Historial de MLflow (`mlflow.db`) y estudio de Optuna (`optuna.db`) | Solo local, no versionado | No se recuperan. Los resultados quedan en `reports/experimentos.md` y los mejores parámetros en `reports/optuna_mejores_parametros.json`; el pipeline vuelve a generar el historial y el registro |
| Datos originales | Descarga manual de UNSW (ver `docs/datos.md`, con las sumas de verificación) | Plan B si el bucket se pierde: se vuelven a descargar y se comprueba el md5 |

## Pasos
```bash
git clone https://github.com/Lis-Sanchez-DS/RetoDefenderServidor.git && cd RetoDefenderServidor
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
# credenciales del bucket (no van a Git):
.venv/bin/dvc remote modify --local almacen access_key_id <ID>
.venv/bin/dvc remote modify --local almacen secret_access_key <SECRETO>
.venv/bin/dvc pull $(find . -type f -name "*.dvc" -not -path "./.dvc/*")   # datos, modelos y predicciones
.venv/bin/python -m src.predecir entrada.csv --modelo models/cv-grupos-xgboost-optuna.joblib --umbral 0.6
.venv/bin/python -m src.pipeline.flow                                        # reconstruye el historial de MLflow (~5 min)
```

## Comprobación hecha (2026-10-05)
Se clonó el repositorio en una carpeta vacía, se copiaron las credenciales y se hizo `dvc pull`: se recuperaron los 33 archivos (729 MB de datos y 24 MB de modelos) y `dvc status` quedó sin diferencias. El md5 del modelo congelado `cv-grupos-xgboost-optuna.joblib` es idéntico al original (`554b165e…`), el CSV de entrenamiento coincide con `docs/datos.md` y `src.predecir` produjo predicciones con el modelo recuperado.

## Detalle que conviene recordar
`dvc push` sin argumentos subió solo 9 de los archivos la primera vez; la recuperación en el clon lo descubrió (faltaba el modelo). Se resolvió subiendo con todos los `.dvc` como destino (`dvc push $(find . -type f -name "*.dvc" -not -path "./.dvc/*")`), tras lo cual `dvc status -c` quedó sincronizado. Por eso se usa siempre esa forma y se verifica con un `dvc pull` en un clon.
