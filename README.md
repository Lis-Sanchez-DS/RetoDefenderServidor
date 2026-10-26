# Defiende el servidor — MLOps 2026-II

Clasificación binaria de conexiones de red (0 = normal, 1 = ataque) con UNSW-NB15, para demostrar que se pueden **identificar, comparar y recuperar** versiones de código, datos, experimentos y modelos. Taller evaluable del curso MLOps (Universidad Externado de Colombia). Dataset histórico de laboratorio (2015): no demuestra eficacia sobre redes actuales.

Cadena de trazabilidad: **commit Git → datos (DVC) → run de MLflow → revisión de Hugging Face**. El índice completo está en [`entrega.md`](entrega.md).

| Versión | Qué es | Código (tag) | Run de MLflow | Revisión en Hugging Face |
|---|---|---|---|---|
| v1 | XGBoost estándar (C2), umbral 0,5 | `v1` | `18fbcc33e83041e7ad99c7937583159d` | `053f3e62f144e78e055533f82bce14c9b54e3030` |
| v2 | XGBoost con búsqueda de Optuna (C4), umbral 0,66 | `v2` | `6d7cb635a3104e838d81b205f2d51b49` | `c521d5ccf41ad404998e3364c685363ffa620bbb` |

Modelos: https://huggingface.co/LisSanchez-DS/defender-servidor (repositorio privado; se necesita acceso de lectura).

## Requisitos
- Python **3.14** (probado con 3.14.4), Linux/WSL, `git`. Dependencias fijadas en [`requirements.lock.txt`](requirements.lock.txt) (`requirements.txt` lista las directas).

## Instalación
```bash
git clone https://github.com/Lis-Sanchez-DS/RetoDefenderServidor.git && cd RetoDefenderServidor
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```
Para el código de una versión concreta: `git checkout v1` o `git checkout v2` (etiquetas anotadas).

## Recuperar los datos con DVC
Los datos y los modelos de trabajo no están en Git: los guarda DVC en un bucket privado de Cloudflare R2 (`.dvc/config`). Las credenciales de lectura **no** están en el repositorio; se entregan por separado y se guardan en el archivo local ignorado por Git:
```bash
.venv/bin/dvc remote modify --local almacen access_key_id <ID>
.venv/bin/dvc remote modify --local almacen secret_access_key <SECRETO>
RAW=data/raw/unsw-nb15
.venv/bin/dvc pull $RAW/particion_oficial/UNSW_NB15_training-set.csv.dvc $RAW/particion_oficial/UNSW_NB15_testing-set.csv.dvc \
  data/processed/particion_entrenamiento_validacion.csv.dvc data/processed/validacion_taller-C2.csv.dvc data/processed/validacion_taller-C4.csv.dvc
sha256sum $RAW/particion_oficial/*.csv   # deben coincidir con evidencias/auditoria.json y con el anexo del taller
```

## Ejecutar un experimento
Requiere los datos del paso anterior. Cada configuración (C0 línea base, C1 regresión logística, C2 XGBoost estándar, C3 y C4 con Optuna) se entrena con la partición fija 70/30 y queda en MLflow (experimento `taller-evaluable-70-30`):
```bash
.venv/bin/python -m src.taller_experimentos C2      
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
.venv/bin/python -m src.exportar_mlflow                # regenera evidencias/experimentos.csv y evidencias/mlflow/
```
Las búsquedas de Optuna (C3: ~35 min, C4: ~15 min en 16 núcleos): `python -m src.optuna_taller` y `python -m src.optuna_taller --umbral`. Las hipótesis (escritas antes de entrenar) están en [`docs/taller_hipotesis.md`](docs/taller_hipotesis.md).

## Descargar cada modelo y predecir
Hace falta acceso de lectura al repositorio de Hugging Face (`.venv/bin/hf auth login` o `export HF_TOKEN=<token de lectura>`). Las revisiones salen de `evidencias/seleccion.json`; el script verifica el hash del modelo y aplica el umbral publicado:
```bash
.venv/bin/python -m src.predecir v1 evidencias/recuperacion/entrada_validacion.csv salida_v1.csv
.venv/bin/python -m src.predecir v2 evidencias/recuperacion/entrada_validacion.csv salida_v2.csv
```
Descarga manual por SHA completo:
```python
from huggingface_hub import hf_hub_download
ruta = hf_hub_download("LisSanchez-DS/defender-servidor", "model.joblib", revision="053f3e62f144e78e055533f82bce14c9b54e3030")  # v1
```
La prueba de recuperación completa (entorno nuevo, `dvc pull`, v1 y v2 por SHA, 50 filas fijas) está en `evidencias/recuperacion/` (`recuperar.sh`, `log.txt`, `resultado.json`).

## Resultados
Validación (52.605 filas; ataque = clase positiva):

| Config | Familia | Umbral | Recall | Precisión | FPR | Macro-F1 | Run MLflow |
|---|---|---|---|---|---|---|---|
| C0 | linea-base | 0.5 | 1.0000 | 0.6806 | 1.0000 | 0.4050 | `6ae04f9efb8e47d0908f8f9bfa03a381` |
| C1 | regresion-logistica | 0.5 | 0.9876 | 0.9217 | 0.1789 | 0.9212 | `c3deb2a3e927442fa4f209f7d7770838` |
| C2 | arboles-xgboost | 0.5 | 0.9748 | 0.9608 | 0.0847 | 0.9487 | `18fbcc33e83041e7ad99c7937583159d` |
| C3 | arboles-xgboost | 0.5 | 0.9727 | 0.9613 | 0.0834 | 0.9476 | `7ca31411665944ea8e7d9927e8b752ba` |
| C4 | arboles-xgboost | 0.660669 | 0.9537 | 0.9738 | 0.0547 | 0.9443 | `6d7cb635a3104e838d81b205f2d51b49` |

Prueba (evaluación única de v2 sobre las 82,332 filas, umbral 0.66): recall 0.965, precisión 0.860, FPR 0.193, macro-F1 0.891. La precisión baja mucho frente a la validación; ver el informe y la tarjeta del modelo.

## Estructura
`src/` código · `configs/` hiperparámetros de Optuna · `docs/` hipótesis y documentación de datos · `evidencias/` auditoría, experimentos, selección, métricas y predicciones de prueba, exportación de MLflow y prueba de recuperación · `tests/` pruebas (`.venv/bin/python -m pytest -q`) · `*.dvc` punteros de DVC.

## Limitaciones
Benchmark histórico; la validación estima rendimiento en distribución (filas parecidas a ambos lados); 10,4 % de las filas de prueba tienen una entrada idéntica en entrenamiento; la mejora de v2 frente a v1 proviene del umbral, no de los hiperparámetros; Fuzzers se detectan peor (recall 0,77 en prueba).

## Datos y atribución
UNSW-NB15 (Moustafa y Slay, UNSW Canberra), uso académico; los datos no se redistribuyen aquí. Fuente, huellas y citas en [`docs/datos.md`](docs/datos.md).
