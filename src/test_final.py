"""Evaluación única del modelo v2 congelado sobre las 82.332 filas del archivo de prueba.

Comprueba que el hash del modelo coincide con evidencias/seleccion.json. Escribe
evidencias/metricas-test.json y evidencias/predicciones-test.csv. Se ejecuta UNA vez.
Regla: y_pred = 1 (ataque) si score_attack >= umbral. Uso: .venv/bin/python -m src.test_final
"""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from src.baseline import metricas
from src.datos import RUTA, separar_xy
from src.evaluar_test import todas_las_metricas


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def main():
    sel = json.load(open("evidencias/seleccion.json"))
    cid = sel["modelo_para_test"]
    ficha = next(sel[v] for v in ("v1", "v2") if sel[v]["config_id"] == cid)
    version = next(v for v in ("v1", "v2") if sel[v]["config_id"] == cid)
    ruta = ficha["archivo_modelo"]
    assert sha256(ruta) == ficha["modelo_sha256"], "el modelo no coincide con el congelado"
    if Path("evidencias/metricas-test.json").exists():
        raise SystemExit("la prueba ya se evaluó: no se repite")
    te = pd.read_csv(RUTA.format("testing"))
    X, y = separar_xy(te)
    prob = joblib.load(ruta).predict_proba(X)[:, 1]
    umbral = ficha["umbral"]
    pred = (prob >= umbral).astype(int)
    pd.DataFrame({"row_id": [f"testing:{i}" for i in te["id"]], "y_true": y.values,
                  "score_attack": prob, "y_pred": pred}).to_csv("evidencias/predicciones-test.csv", index=False)
    res = {k: float(v) for k, v in todas_las_metricas(y.values, prob, umbral).items()}
    por_familia = te.assign(acierto=(pred == y.values)).query("label == 1").groupby("attack_cat").acierto.mean().round(4).to_dict()
    json.dump({"version": version, "config_id": cid, "run_id": ficha["run_id"], "modelo_sha256": ficha["modelo_sha256"],
               "hash_dataset_sha256": sha256(RUTA.format("testing")), "filas_evaluadas": len(te),
               "umbral": umbral, "regla": "y_pred = 1 si score_attack >= umbral",
               "clase_positiva": "ataque (label = 1)",
               "matriz_confusion": {"tn": int(res["tn"]), "fp": int(res["fp"]), "fn": int(res["fn"]), "tp": int(res["tp"])},
               "metricas": {k: v for k, v in res.items() if k not in ("tn", "fp", "fn", "tp")},
               "recall_por_familia_de_ataque": por_familia,
               "denominadores_cero": "precision/recall con zero_division=0; FPR = FP/(FP+TN)",
               "nota": "evaluación única; las filas con estados no vistos (no, ACC, CLO) se codifican con ceros",
               "limite": "10,4 % de las filas de test tienen un vector de entrada idéntico en entrenamiento (evidencias/auditoria.json)"},
              open("evidencias/metricas-test.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"filas": len(te), "matriz": [int(res[k]) for k in ("tn", "fp", "fn", "tp")],
                      "recall": res["recall"], "precision": res["precision"], "fpr": res["tasa_falsos_positivos"]}))


if __name__ == "__main__":
    main()
