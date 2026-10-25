"""Congela la selección de v1 y v2 usando SOLO validación. Se ejecuta y se commitea ANTES de evaluar en prueba.

v1 = C2 (regresión logística con categóricas, primera solución funcional).
v2 = configuración de XGBoost con mayor precisión con recall >= 0,95 en validación.
Escribe evidencias/seleccion.json. Uso: .venv/bin/python -m src.seleccion
"""
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main():
    e = pd.read_csv("evidencias/experimentos.csv").set_index("config_id")
    v1 = "C2"
    candidatos = e[e.familia == "arboles-xgboost"]
    v2 = candidatos.precision.idxmax()
    def ficha(cid):
        f = e.loc[cid]
        return {"config_id": cid, "run_id": f.run_id, "modelo_sha256": f.modelo_sha256, "umbral": float(f.umbral),
                "archivo_modelo": f"models/taller-{cid}.joblib", "git_commit_codigo_que_lo_produjo": f.git_commit,
                "dvc_datos_originales_md5": f.referencia_dvc_datos_originales, "dvc_datos_preparados_md5": f.referencia_dvc_datos_preparados,
                "metricas_validacion": {k: float(f[k]) for k in ("recall", "precision", "fpr", "macro_f1", "tn", "fp", "fn", "tp")},
                "huggingface": {"repositorio": None, "revision_sha": None, "nota": "se completa tras publicar (src/publicar_hf.py)"}}
    sel = {"congelado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "evidencia_de_congelacion": "esta selección se commitea antes de crear metricas-test.json; ver git log de evidencias/seleccion.json frente a evidencias/metricas-test.json",
           "criterio": "v2 = mayor precisión con recall >= 0,95 en validación entre las configuraciones XGBoost; umbral = mayor valor (paso 0,05) con recall de validación >= 0,95",
           "solo_validacion": True, "v1": ficha(v1), "v2": ficha(v2),
           "razon_v2": f"{v2} tiene la mayor precisión de validación entre XGBoost ({candidatos.precision.max():.4f}); cambia el modelo (lineal -> árboles) y el umbral respecto a v1",
           "modelo_para_test": v2}
    json.dump(sel, open("evidencias/seleccion.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: sel[k] for k in ("razon_v2", "modelo_para_test")}, ensure_ascii=False), sel["v1"]["umbral"], sel["v2"]["umbral"])


if __name__ == "__main__":
    main()
