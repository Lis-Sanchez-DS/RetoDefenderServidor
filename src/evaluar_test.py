"""Evaluación única en el conjunto de prueba de un modelo guardado (congelado).

Carga el modelo desde `models/`, comprueba que coincide con la versión registrada en
DVC, predice con el umbral indicado y registra el resultado en MLflow. No ajusta nada.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from src.baseline import metricas
from src.datos import cargar, separar_xy

UMBRAL = 0.6  # elegido con predicciones fuera de muestra (reports/experimentos.md, experimento 6)


def todas_las_metricas(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    res = metricas(y, pred)
    res["mcc"] = matthews_corrcoef(y, pred)
    res["roc_auc"] = roc_auc_score(y, prob)
    res["pr_auc"] = average_precision_score(y, prob)
    res["fp_por_10000"] = res["fp"] / len(y) * 10000
    res["fn_por_10000"] = res["fn"] / len(y) * 10000
    return res


def metricas_validacion(nombre, umbral):
    """Media (y desviación) por pliegue de las predicciones fuera de muestra."""
    oof = pd.read_csv(f"data/processed/oof_{nombre}.csv")
    filas = [todas_las_metricas(g.y.values, g.prob.values, umbral) for _, g in oof.groupby("pliegue")]
    tabla = pd.DataFrame(filas)
    return tabla.mean(), tabla.std()


def main(nombre):
    ruta = Path(f"models/{nombre}.joblib")
    md5_dvc = yaml.safe_load(Path(f"{ruta}.dvc").read_text())["outs"][0]["md5"]
    md5_real = hashlib.md5(ruta.read_bytes()).hexdigest()
    assert md5_dvc == md5_real, "el modelo no coincide con la versión registrada en DVC"
    modelo = joblib.load(ruta)

    prueba = cargar("testing")
    X, y = separar_xy(prueba)
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= UMBRAL).astype(int)
    res = todas_las_metricas(y, prob, UMBRAL)
    media_val, desv_val = metricas_validacion(nombre, UMBRAL)
    comparacion = pd.DataFrame({"validacion_media": media_val, "validacion_desv": desv_val, "prueba": pd.Series(res)})
    Path("reports").mkdir(exist_ok=True)
    comparacion.to_csv("reports/test_vs_validacion.csv")
    recall_familia = prueba.assign(acierto=(pred == y)).query("label == 1").groupby("attack_cat").acierto.agg(["mean", "size"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("unsw-nb15")
    with mlflow.start_run(run_name=f"test-{nombre}"):
        mlflow.set_tags({"etapa": "evaluacion_test", "modelo_md5_dvc": md5_dvc,
                         "nota": "Evaluacion en prueba; no se uso para elegir modelo ni umbral."})
        mlflow.log_params({"modelo": nombre, "umbral": UMBRAL, "filas_prueba": len(prueba)})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in res.items() if k not in ("tn", "fp", "fn", "tp")})
    print(comparacion.round(4).to_string())
    print(recall_familia.round(3).to_string())
    print("ataques en prueba:", int(y.sum()), "normales:", int((y == 0).sum()))


if __name__ == "__main__":
    main(sys.argv[1])
