"""Congela la selección de v1 y v2 (decisiones del usuario) usando SOLO validación.

v1 = C2 (XGBoost estándar) con umbral 0,5; v2 = C4 (XGBoost de la búsqueda de Optuna con umbral)
con umbral 0,66. Las métricas se recalculan desde las puntuaciones de validación guardadas.
Se ejecuta y se commitea ANTES de evaluar en prueba. Escribe evidencias/seleccion.json.
Uso: .venv/bin/python -m src.seleccion
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.metricas import metricas

VERSIONES = {"v1": ("C2", 0.5), "v2": ("C4", 0.66)}  # decisiones del usuario
RAZON = {
    "v1": "XGBoost estándar (300 árboles, profundidad 6, tasa 0,1) con umbral 0,5: primera solución funcional.",
    "v2": "Modelo de la búsqueda de Optuna (solo entrenamiento) con umbral 0,66 elegido por el usuario a partir del umbral hallado (0,6607): "
          "más precisión y menos falsas alarmas que v1 a costa de más ataques perdidos (recall 0,954 frente a 0,975).",
}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main():
    if Path("evidencias/seleccion.json").exists() and "--rehacer" not in sys.argv:
        raise SystemExit("la selección ya está congelada (evidencias/seleccion.json); no se rehace. --rehacer solo para rehacerla a propósito")
    e = pd.read_csv("evidencias/experimentos.csv").set_index("config_id")

    def ficha(version):
        cid, umbral = VERSIONES[version]
        f = e.loc[cid]
        v = pd.read_csv(f"data/processed/validacion_taller-{cid}.csv")
        m = metricas(v.y_true.values, (v.score_attack >= umbral).astype(int).values)
        return {"config_id": cid, "run_id": f.run_id, "modelo_sha256": f.modelo_sha256, "umbral": umbral,
                "archivo_modelo": f"models/taller-{cid}.joblib", "dvc_modelo": f"models/taller-{cid}.joblib.dvc",
                "git_commit_codigo_que_lo_produjo": f.git_commit,
                "dvc_datos_originales_md5": f.referencia_dvc_datos_originales, "dvc_datos_preparados_md5": f.referencia_dvc_datos_preparados,
                "razon": RAZON[version],
                "metricas_validacion": {k: float(x) for k, x in m.items() if k != "accuracy"},
                "huggingface": {"repositorio": None, "revision_sha": None, "tag": None,
                                "nota": "se completa tras publicar (src/publicar_hf.py); no modifica los campos congelados"}}

    sel = {"congelado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "evidencia_de_congelacion": "esta selección se commitea antes de que existan evidencias/metricas-test.json y predicciones-test.csv; ver `git log -- evidencias/seleccion.json evidencias/metricas-test.json`",
           "criterio": "decisión del usuario con datos de validación únicamente (partición fija 70/30); la prueba no se usó",
           "solo_validacion": True, "v1": ficha("v1"), "v2": ficha("v2"),
           "modelo_para_test": "v2",
           "razon_modelo_para_test": "v2 es la solución final; v1 no se evalúa en la prueba (la comparación v1/v2 es en validación)"}
    json.dump(sel, open("evidencias/seleccion.json", "w"), indent=2, ensure_ascii=False)
    for v in ("v1", "v2"):
        print(v, sel[v]["config_id"], sel[v]["umbral"], {k: round(x, 4) for k, x in sel[v]["metricas_validacion"].items()})


if __name__ == "__main__":
    main()
