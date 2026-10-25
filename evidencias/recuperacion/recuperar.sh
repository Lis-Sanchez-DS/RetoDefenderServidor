#!/usr/bin/env bash
# Prueba de recuperación en una carpeta y un entorno nuevos (sin cachés de datos, modelos ni pip).
# Uso:  HF_TOKEN=<token de lectura> bash recuperar.sh <carpeta_nueva_vacía> <url_o_ruta_del_repositorio> <commit>
# Antes: copiar las credenciales de lectura del remoto DVC a <carpeta>/proyecto/.dvc/config.local (no van a Git);
#        con el script: DVC_CONFIG_LOCAL=<ruta del config.local>.
set -euo pipefail
DEST=${1:?carpeta nueva}; REPO=${2:?repositorio}; COMMIT=${3:?commit}
ANEXO_TRAIN=bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa
ANEXO_TEST=734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559
export HF_HOME="$DEST/hf_cache" PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
echo "== 1. Código: clonar y fijar la revisión"; date -u
git clone -q "$REPO" "$DEST/proyecto"; cd "$DEST/proyecto"; git checkout -q "$COMMIT"; git rev-parse HEAD
echo "== 2. Entorno nuevo con dependencias fijadas"; python3 --version
python3 -m venv .venv; .venv/bin/pip install -q --no-cache-dir -r requirements.lock.txt; .venv/bin/pip check | tail -1
echo "== 3. Datos con DVC desde el remoto (caché nueva)"
cp "${DVC_CONFIG_LOCAL:?ruta a config.local}" .dvc/config.local
test ! -e .dvc/cache || echo "caché preexistente: $(ls .dvc/cache)"
RAW=data/raw/unsw-nb15
.venv/bin/dvc pull $RAW/particion_oficial/UNSW_NB15_training-set.csv.dvc $RAW/particion_oficial/UNSW_NB15_testing-set.csv.dvc \
  $RAW/metadatos/NUSW-NB15_features.csv.dvc data/processed/particion_entrenamiento_validacion.csv.dvc \
  data/processed/validacion_taller-C2.csv.dvc data/processed/validacion_taller-C4.csv.dvc
.venv/bin/dvc status
echo "== 4. Huellas de los datos originales frente al anexo del taller"
sha256sum $RAW/particion_oficial/*.csv
[ "$(sha256sum $RAW/particion_oficial/UNSW_NB15_training-set.csv | cut -d' ' -f1)" = "$ANEXO_TRAIN" ] && echo "training: coincide con el anexo"
[ "$(sha256sum $RAW/particion_oficial/UNSW_NB15_testing-set.csv | cut -d' ' -f1)" = "$ANEXO_TEST" ] && echo "testing: coincide con el anexo"
echo "== 5. Modelos desde Hugging Face por SHA completo, y predicción de la misma muestra"
.venv/bin/python -m src.recuperacion recuperar evidencias/recuperacion 2>&1 | grep -v -E "INFO|Warning"
echo "== 6. Comparación con las salidas guardadas antes"
.venv/bin/python -m src.recuperacion comparar evidencias/recuperacion
echo "== fin"; date -u
