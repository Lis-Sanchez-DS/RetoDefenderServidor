"""Publica v1 y v2 en un repositorio de modelos de Hugging Face Hub (una revisión por versión).

Requiere haber iniciado sesión (`hf auth login`) o HF_TOKEN en el entorno; el token nunca se
guarda en el repositorio. Cada revisión contiene: model.joblib, config.json (umbral, variables,
hashes y referencias Git/DVC/MLflow), requirements.txt y README.md (model card). Los SHA de commit
resultantes se escriben en evidencias/seleccion.json y se crean los tags v1 y v2.
Uso: .venv/bin/python -m src.publicar_hf <usuario/repositorio> [--publico]
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

TARJETA = """---
license: other
library_name: scikit-learn
tags: [intrusion-detection, unsw-nb15, tabular-classification, mlops]
---
# Defiende el servidor — {version} ({config_id})

Clasificador binario de conexiones de red: **0 = normal, 1 = ataque** (UNSW-NB15). Proyecto académico del curso MLOps 2026-II (Universidad Externado de Colombia).

## Propósito y uso previsto
Práctica de versionado de código, datos, experimentos y modelos. **No** demuestra eficacia sobre redes actuales: es un benchmark de laboratorio de 2015.

## Contenido de esta revisión
`model.joblib` (pipeline completo de scikit-learn: preprocesamiento + estimador), `config.json` (umbral, variables, hashes, referencias), `requirements.txt`.
- Modelo: {descripcion}
- Umbral: **{umbral}** (se predice ataque si `score_attack >= {umbral}`)
- SHA-256 de `model.joblib`: `{sha}`

## Datos
Partición oficial `UNSW_NB15_training-set.csv` (175.341 filas; SHA-256 `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`). Entrenamiento y validación se crearon solo con ese archivo (140.271 / 35.069 filas), con grupos de entradas idénticas sin repartir (semilla 42). Entradas: 42 variables; se excluyen `id`, `label`, `attack_cat`. Datos versionados con DVC.

## Métricas de validación (ataque = clase positiva)
| Métrica | Valor |
|---|---|
| Recall | {recall:.4f} |
| Precisión | {precision:.4f} |
| Tasa de falsos positivos | {fpr:.4f} |
| Macro-F1 | {macro_f1:.4f} |
| Matriz (TN / FP / FN / TP) | {tn:.0f} / {fp:.0f} / {fn:.0f} / {tp:.0f} |

{extra}
## Limitaciones
- Dataset histórico de laboratorio; la validación estima el rendimiento en distribución (hay filas parecidas a ambos lados) y es optimista.
- Un 10 % de las filas de test tienen una entrada idéntica en entrenamiento.
- Un ataque no detectado cuesta más que una falsa alarma: el umbral prioriza recall >= 0,95 en validación a costa de falsas alarmas.
- El archivo es un pickle de joblib: cárgalo solo de fuentes de confianza y con las versiones de `requirements.txt`.

## Cómo usarlo
```python
import json, joblib
from huggingface_hub import hf_hub_download
rev = "{{SHA_COMPLETO_DE_ESTA_REVISION}}"
modelo = joblib.load(hf_hub_download("{repo}", "model.joblib", revision=rev))
umbral = json.load(open(hf_hub_download("{repo}", "config.json", revision=rev)))["umbral"]
# X = DataFrame con las columnas originales del CSV (puede incluir id/label/attack_cat: quítalas antes)
score = modelo.predict_proba(X)[:, 1]; pred = (score >= umbral).astype(int)
```

## Trazabilidad
Código: commit Git `{commit}` · Datos DVC: originales `{dvc_orig}`, preparados `{dvc_prep}` · MLflow: run `{run_id}` (experimento `taller-evaluable`).
"""


def preparar(carpeta, version, sel, e, repo):
    ficha = sel[version]
    f = e.loc[ficha["config_id"]]
    cid = ficha["config_id"]
    desc = {"C2": "XGBoost estándar (300 árboles, profundidad 6, tasa 0,1) con variables numéricas y categóricas (one-hot)",
            "C4": "XGBoost con hiperparámetros y umbral de una búsqueda de Optuna hecha solo con entrenamiento, con categóricas (one-hot)",
            "C3": "XGBoost de Optuna con umbral 0,5"}[cid]
    extra = ""
    if version == "v2":
        v1 = sel["v1"]["metricas_validacion"]
        v2 = ficha["metricas_validacion"]
        extra = ("## Cambios frente a v1 (validación)\n"
                 f"v2 cambia los hiperparámetros (búsqueda de Optuna hecha solo con entrenamiento) y el umbral ({sel['v1']['umbral']} → {ficha['umbral']}). "
                 f"Precisión {v1['precision']:.4f} → {v2['precision']:.4f}; recall {v1['recall']:.4f} → {v2['recall']:.4f}; "
                 f"falsas alarmas {v1['fp']:.0f} → {v2['fp']:.0f}; ataques perdidos {v1['fn']:.0f} → {v2['fn']:.0f}. "
                 "La mejora en precisión proviene sobre todo del umbral: con umbral 0,5 el modelo de v2 no supera a v1. "
                 "Se aceptan más ataques perdidos a cambio de menos falsas alarmas. Ver `evidencias/experimentos.csv` del repositorio de código.\n\n")
    shutil.copy(ficha["archivo_modelo"], carpeta / "model.joblib")
    shutil.copy("requirements.lock.txt", carpeta / "requirements.txt")
    json.dump({"version": version, "config_id": cid, "umbral": ficha["umbral"], "clase_positiva": "ataque (label=1)",
               "regla": "y_pred = 1 si score_attack >= umbral", "modelo_sha256": ficha["modelo_sha256"],
               "columnas_excluidas_de_la_entrada": ["id", "label", "attack_cat"], "mlflow_run_id": ficha["run_id"],
               "git_commit_codigo": ficha["git_commit_codigo_que_lo_produjo"],
               "dvc_datos_originales_md5": ficha["dvc_datos_originales_md5"], "dvc_datos_preparados_md5": ficha["dvc_datos_preparados_md5"],
               "metricas_validacion": ficha["metricas_validacion"]}, open(carpeta / "config.json", "w"), indent=2)
    m = {**ficha["metricas_validacion"], "fpr": ficha["metricas_validacion"]["tasa_falsos_positivos"]}
    (carpeta / "README.md").write_text(TARJETA.format(
        version=version, config_id=cid, descripcion=desc, umbral=ficha["umbral"], sha=ficha["modelo_sha256"], extra=extra, repo=repo,
        commit=ficha["git_commit_codigo_que_lo_produjo"], dvc_orig=ficha["dvc_datos_originales_md5"],
        dvc_prep=ficha["dvc_datos_preparados_md5"], run_id=ficha["run_id"], **m))


def main():
    import pandas as pd
    repo = sys.argv[1]
    sel = json.load(open("evidencias/seleccion.json"))
    e = pd.read_csv("evidencias/experimentos.csv").set_index("config_id")
    api = HfApi()
    api.create_repo(repo, repo_type="model", private="--publico" not in sys.argv, exist_ok=True)
    api.update_repo_settings(repo, private="--publico" not in sys.argv, repo_type="model")  # fuerza la visibilidad pedida
    for version in ("v1", "v2"):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            preparar(carpeta, version, sel, e, repo)
            info = api.upload_folder(folder_path=str(carpeta), repo_id=repo, repo_type="model",
                                     commit_message=f"{version}: {sel[version]['config_id']} (umbral {sel[version]['umbral']})")
        sha = info.oid
        api.create_tag(repo, tag=version, revision=sha, repo_type="model", exist_ok=True)
        sel[version]["huggingface"] = {"repositorio": repo, "revision_sha": sha, "tag": version}
        print(version, sha)
    json.dump(sel, open("evidencias/seleccion.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
