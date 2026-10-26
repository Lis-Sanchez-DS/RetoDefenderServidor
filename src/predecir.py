"""Descarga v1 o v2 de Hugging Face por su SHA completo y predice las filas de un CSV.

Las revisiones salen de evidencias/seleccion.json. Se comprueba el hash del modelo y se usa el umbral
publicado. La entrada puede tener las columnas originales de UNSW-NB15 (se ignoran id, label y attack_cat).
Salida: row_id, score_attack (probabilidad de ataque) y y_pred (1 = ataque si score_attack >= umbral).
El repositorio de Hugging Face es público: no hace falta iniciar sesión.
Uso: .venv/bin/python -m src.predecir {v1|v2} entrada.csv salida.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from src.datos import COLUMNAS_EXCLUIDAS


def cargar_version(version: str):
    """Devuelve (modelo, umbral) de la revisión publicada de `version`, verificando su hash."""
    ficha = json.load(open("evidencias/seleccion.json"))[version]
    hf = ficha["huggingface"]
    ruta = hf_hub_download(hf["repositorio"], "model.joblib", revision=hf["revision_sha"])
    cfg = json.load(open(hf_hub_download(hf["repositorio"], "config.json", revision=hf["revision_sha"])))
    if hashlib.sha256(Path(ruta).read_bytes()).hexdigest() != ficha["modelo_sha256"]:
        raise ValueError("el modelo descargado no coincide con el hash registrado")
    return joblib.load(ruta), cfg["umbral"]


def predecir(modelo, datos: pd.DataFrame, umbral: float) -> pd.DataFrame:
    X = datos.drop(columns=[c for c in COLUMNAS_EXCLUIDAS if c in datos.columns])
    score = modelo.predict_proba(X)[:, 1]
    ids = datos["id"] if "id" in datos.columns else range(len(datos))
    return pd.DataFrame({"row_id": list(ids), "score_attack": score, "y_pred": (score >= umbral).astype(int)})


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("v1", "v2"):
        raise SystemExit(__doc__)
    modelo, umbral = cargar_version(sys.argv[1])
    salida = predecir(modelo, pd.read_csv(sys.argv[2]), umbral)
    salida.to_csv(sys.argv[3], index=False)
    print(f"{sys.argv[1]}: umbral {umbral}; {len(salida)} filas; {int(salida.y_pred.sum())} marcadas como ataque -> {sys.argv[3]}")
