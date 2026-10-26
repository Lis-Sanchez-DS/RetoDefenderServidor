# Entrega — MLOps · Defiende el servidor


## 1. Integrantes y repositorio
- Integrante (entrega individual): **Lis Alejandra Sánchez Moreno** — Grupo B (4:00 p. m. – 6:00 p. m.)
- Repositorio de código: https://github.com/Lis-Sanchez-DS/RetoDefenderServidor (público)
- Commit final de la entrega: se indica en el mensaje de entrega (un commit no puede contener su propio hash). Las versiones se identifican con las etiquetas `v1` y `v2`.
- Repositorio de modelos: https://huggingface.co/LisSanchez-DS/defender-servidor (público: se descarga sin iniciar sesión ni usar tokens)

## 2. Versiones
### v1 — C2
- Código: el modelo se entrenó en el commit `4c3fb7c` (el que registró MLflow); la etiqueta `v1` apunta a `f40c9cc`, que ya incluye el puntero DVC del modelo y la evidencia — https://github.com/Lis-Sanchez-DS/RetoDefenderServidor/tree/v1
- Referencia DVC: datos originales (training) md5 `e55caabaa6cd4a8f1c06a227bcfababc`; datos preparados (partición) md5 `a869e919c9a8a014fa6c7aec00446ce5`; puntero del modelo `models/taller-C2.joblib.dvc`
- Run de MLflow: `18fbcc33e83041e7ad99c7937583159d` (exportado en `evidencias/mlflow/18fbcc33e83041e7ad99c7937583159d.json`)
- Hugging Face: https://huggingface.co/LisSanchez-DS/defender-servidor · revisión `053f3e62f144e78e055533f82bce14c9b54e3030` (tag `v1`)
- Hash del modelo (SHA-256 de `model.joblib`): `7fef2e3afa957bd67cd48fe5f72eb4d1eb49285832480632a37ed4012e9bb8fd`
- Umbral: **0.5** · métricas de validación: recall 0.9748, precisión 0.9608, FPR 0.0847, macro-F1 0.9487

### v2 — C4
- Código: el modelo se entrenó en el commit `f235480` (el que registró MLflow), al que apunta la etiqueta `v2` — https://github.com/Lis-Sanchez-DS/RetoDefenderServidor/tree/v2
- Referencia DVC: datos originales (training) md5 `e55caabaa6cd4a8f1c06a227bcfababc`; datos preparados (partición) md5 `a869e919c9a8a014fa6c7aec00446ce5`; puntero del modelo `models/taller-C4.joblib.dvc`
- Run de MLflow: `6d7cb635a3104e838d81b205f2d51b49` (exportado en `evidencias/mlflow/6d7cb635a3104e838d81b205f2d51b49.json`)
- Hugging Face: https://huggingface.co/LisSanchez-DS/defender-servidor · revisión `c521d5ccf41ad404998e3364c685363ffa620bbb` (tag `v2`)
- Hash del modelo (SHA-256 de `model.joblib`): `e7f7ddc749f2a22c66007031110f9658b1803b3f306f8e3c4e032c52cdac7ed4`
- Umbral: **0.66** · métricas de validación: recall 0.9537, precisión 0.9738, FPR 0.0548, macro-F1 0.9443

**Qué cambia entre v1 y v2:** los hiperparámetros del XGBoost (búsqueda de Optuna solo con entrenamiento) y el umbral (0,5 → 0,66). Con umbral 0,5 el modelo de v2 no supera a v1; la mejora en precisión (0,9608 → 0,9738) y en falsas alarmas (1.423 → 920) proviene del umbral, a costa de más ataques perdidos (903 → 1.656).
**Selección congelada antes de la prueba:** commit `4fda613` (`evidencias/seleccion.json`); la prueba se evaluó después (commit `ea7cc48`). Verificable con `git log -- evidencias/seleccion.json evidencias/metricas-test.json`.

**Versiones de datos en DVC** (referencias de Git, documentadas en `docs/transformacion_datos.md`): datos originales = etiqueta `datos-originales` (commit `7e1b3ac`); datos preparados (partición 70/30) = etiqueta `datos-preparados` (commit `6f6399e`).

## 3. Evidencias (carpeta `evidencias/`)
| Archivo | Contenido |
|---|---|
| `auditoria.json` | Fuentes y SHA-256, filas/columnas, clases, faltantes, duplicados y contradicciones, política de limpieza, variables, semilla, huellas de la partición, cruces entre particiones |
| `experimentos.csv` | Una fila por configuración (C0-C4): hipótesis, familia, parámetros, semilla, commit, referencias DVC, run ID, umbral, métricas y conclusión de validación |
| `conclusiones_validacion.json` | Veredictos del usuario sobre cada hipótesis |
| `seleccion.json` | v1/v2: run, referencias Git/DVC/Hugging Face, hash del modelo, umbral, razón y marca de congelación |
| `metricas-test.json` | Versión y hash del modelo evaluado (v2), hash del dataset, 82,332 filas, umbral, clase positiva, matriz de confusión y métricas |
| `predicciones-test.csv` | `row_id` (`testing:<id>`), `y_true`, `score_attack`, `y_pred`; regla: `y_pred = 1` si `score_attack >= umbral` |
| `mlflow/` | Exportación de los runs (parámetros, métricas, etiquetas); `curvas/` curvas de umbral en validación |
| `informe/informe.pdf` (se entrega aparte, fuera del ZIP de código) | Informe (3 páginas de contenido más una de referencias), con el formato de la plantilla del taller. `informe/generar_informe.py` lo genera desde estos archivos (fuentes Source Sans y Bitter, licencia OFL, en `informe/fuentes/`; el logo se toma del PDF del taller y no se versiona) y `tests/test_informe.py` comprueba que sus cifras coinciden |
| `recuperacion/` | 50 filas fijas de validación, salidas de v1/v2 antes y después, hashes, comandos (`recuperar.sh`), log y `resultado.json` (**APROBADA**, diferencia máxima de probabilidad 0.0) |

## 4. Almacenamiento DVC
Hay dos vías, sin compartir credenciales en el ZIP ni en el informe (el taller pide no compartir tokens):
- **Respaldo del almacenamiento DVC** (`respaldo-dvc.zip`, 17 MB), entregado **aparte del ZIP de código**: contiene los 14 archivos versionados (datos originales de entrenamiento y prueba, metadatos, partición preparada, puntuaciones de validación y los 5 modelos de trabajo). Se usa como remoto local: `unzip respaldo-dvc.zip`, `dvc remote add --local respaldo ./respaldo-dvc`, `dvc pull -r respaldo` (instrucciones completas en el README). Probado desde un clon limpio de GitHub, sin credenciales: las huellas coinciden con las del anexo.
- **Bucket privado `pipeline-mlops` en Cloudflare R2** (remoto `almacen`, endpoint en `.dvc/config`): es el almacenamiento original; el acceso requiere credenciales de la autora y no se comparte con el docente.

## 5. Comandos y recursos
Instalación, recuperación de datos, entrenamiento y predicción: ver `README.md`. Recursos medidos (equipo de desarrollo: WSL2, 16 núcleos, 7,6 GB de RAM): búsqueda de Optuna C3 ≈ 35 min y C4 ≈ 15 min (25 pruebas cada una); prueba de recuperación completa en entorno nuevo ≈ 5 min (23:43:49–23:49:05 UTC del 25-oct-2026, incluye instalar dependencias y `dvc pull`). El entrenamiento de un modelo final no se midió por separado.

## 6. Limitaciones conocidas
- Benchmark histórico de laboratorio (2015). La validación estima rendimiento en distribución; el 10,4 % de las filas de prueba tiene una entrada idéntica en entrenamiento.
- En la prueba la precisión de v2 cae a 0.860 (0,974 en validación) y la tasa de falsos positivos sube a 0.193; la causa no está demostrada.
- El conjunto de prueba ya se había evaluado antes, con otro modelo, en una versión anterior del trabajo (otra guía); no se usó para elegir el modelo ni el umbral actuales. Los commits registrados en MLflow para C0-C2 son anteriores a mover funciones auxiliares a `src/metricas.py` (movidas sin cambios).
- **Fe de erratas:** las tarjetas inmutables de v1 y v2 en Hugging Face (revisiones `053f3e62…` y `c521d5cc…`) indican por error los tamaños de una partición 80/20 descartada (140.271 y 35.069 filas); los tamaños reales son 122.735 y 52.605 (70/30). La tarjeta principal del repositorio ya está corregida y trae la fe de erratas; los modelos, sus hashes y los SHA no cambian. `evidencias/auditoria.json` se regeneró con la partición vigente.
- Solo dos familias (lineal y árboles potenciados); sin garantía de que XGBoost sea el mejor modelo posible. La búsqueda de C4 podó 16 de 25 pruebas.

## 7. Uso de IA, código externo y contribuciones
- Herramienta de IA: Claude Code (Anthropic, modelo Claude Sonnet 5.5). Se usó para escribir y refactorizar el código de `src/` y `tests/`, redactar la documentación y las tarjetas de modelo, ejecutar las pruebas y la prueba de recuperación y verificar hashes y resultados.
- **Informe PDF:** fue redactado con Claude Code a partir de los archivos de `evidencias/` y del perfil de estilo de escritura de la integrante, y luego revisado por ella. Las decisiones, hipótesis y veredictos que aparecen en el informe son de la integrante; las cifras se tomaron de los archivos de evidencia y se contrastaron con ellos (hay una prueba automática que verifica que el informe cita las mismas cifras).
- Decisiones del equipo (no de la IA): partición 70/30, las cinco configuraciones, la definición de v1/v2, los umbrales, **todas las hipótesis** (`docs/taller_hipotesis.md`) y los veredictos sobre ellas.
- Verificado: resultados recalculados desde las predicciones guardadas (pruebas en `tests/`), hashes de modelos y datos, prueba de recuperación en entorno nuevo.
- Código externo: scikit-learn, XGBoost, Optuna, MLflow, DVC, huggingface_hub (licencias de cada proyecto). Dataset UNSW-NB15, ver `docs/datos.md`.
- Contribución: Lis Alejandra Sánchez Moreno realizó todo el trabajo del taller (diseño, decisiones, hipótesis, ejecución, revisión y entrega), con la asistencia de IA descrita arriba.
