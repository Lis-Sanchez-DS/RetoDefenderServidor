"""Cuatro configuraciones del taller evaluable (docs/taller_hipotesis.md), dos familias.

Misma partición fija, mismo preprocesamiento (ajustado solo con entrenamiento) y mismas
métricas de validación. Cada ejecución queda en MLflow (experimento `taller-evaluable-70-30`)
con hipótesis, parámetros, semilla, versión de datos, commit, métricas y el modelo
completo; además se guarda `models/taller-<id>.joblib` y sus probabilidades de validación.
El conjunto de prueba no se toca. Uso: .venv/bin/python -m src.taller_experimentos C0 [C1 C2 ...]
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline import metricas
from src.validacion_cruzada import punto_con_recall_minimo
from src.datos import separar_xy
from src.particion import SALIDA, cargar_particion
from src.preprocesamiento import columnas_numericas, construir_preprocesamiento

PISO_RECALL = 0.95
EXPERIMENTO = "taller-evaluable-70-30"
REJILLA = np.round(np.arange(0.05, 0.96, 0.05), 2)
PARAMETROS_OPTUNA = "reports/optuna_taller_mejores_parametros.json"


def configuraciones():
    mejores = json.load(open(PARAMETROS_OPTUNA))["mejores_parametros"] if Path(PARAMETROS_OPTUNA).exists() else None
    xgb = lambda **p: (lambda s: XGBClassifier(tree_method="hist", random_state=s, n_jobs=8, **p))
    cfgs = {
        "C0": dict(nombre="linea-base", familia="linea-base", categoricas=False, escalar=False, criterio="recall",
                   clasificador=lambda s: DummyClassifier(strategy="most_frequent"),
                   parametros={"modelo": "DummyClassifier(most_frequent)"}),
        "C1": dict(nombre="regresion-logistica", familia="regresion-logistica", categoricas=True, escalar=True, criterio="recall",
                   clasificador=lambda s: LogisticRegression(max_iter=2000, random_state=s),
                   parametros={"modelo": "LogisticRegression", "max_iter": 2000}),
        "C2": dict(nombre="xgboost-defecto", familia="arboles-xgboost", categoricas=True, escalar=False, criterio="precision",
                   clasificador=xgb(n_estimators=300, max_depth=6, learning_rate=0.1),
                   parametros={"modelo": "XGBClassifier", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}),
        "C3": dict(nombre="xgboost-profundo", familia="arboles-xgboost", categoricas=True, escalar=False, criterio="precision",
                   clasificador=xgb(n_estimators=600, max_depth=10, learning_rate=0.05),
                   parametros={"modelo": "XGBClassifier", "n_estimators": 600, "max_depth": 10, "learning_rate": 0.05}),
    }
    if mejores:
        cfgs["C4"] = dict(nombre="xgboost-optuna", familia="arboles-xgboost", categoricas=True, escalar=False, criterio="precision",
                          clasificador=xgb(**mejores), parametros={"modelo": "XGBClassifier", "origen": "optuna", **mejores})
    return cfgs


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def md5_dvc(ruta):
    return yaml.safe_load(Path(ruta).read_text())["outs"][0]["md5"]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def hipotesis(cid):
    """Texto de la hipótesis de `cid` tal como se commiteó en docs/taller_hipotesis.md."""
    if cid == "C0":
        return "Línea base: predice siempre la clase dominante (ataque); recall 1,00 y tasa de falsos positivos 1,00 por construcción."
    texto = Path("docs/taller_hipotesis.md").read_text()
    m = re.search(rf"\*\*{cid} — [^\n]*\n(.*?)(?:\n\n|\Z)", texto, re.S)
    return m.group(1).strip()


def curva(y, prob):
    filas = []
    for u in REJILLA:
        r = metricas(y, (prob >= u).astype(int))
        filas.append({"umbral": u, **{k: r[k] for k in ("recall", "precision", "tasa_falsos_positivos", "macro_f1", "tn", "fp", "fn", "tp")}})
    return pd.DataFrame(filas)


def main():
    seed = yaml.safe_load(open("params.yaml"))["seed"]
    entrenamiento, validacion = cargar_particion()
    X_tr, y_tr = separar_xy(entrenamiento)
    X_val, y_val = separar_xy(validacion)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENTO)
    cfgs = configuraciones()
    elegidos = sys.argv[1:]
    ref_datos = {"dvc_md5_datos_originales_training": md5_dvc("data/raw/unsw-nb15/particion_oficial/UNSW_NB15_training-set.csv.dvc"),
                 "dvc_md5_datos_preparados_particion": md5_dvc(f"{SALIDA}.dvc")}
    for cid in elegidos:
        cfg = cfgs[cid]
        modelo = Pipeline([
            ("preprocesamiento", construir_preprocesamiento(columnas_numericas(X_tr), cfg["categoricas"], cfg["escalar"])),
            ("clasificador", cfg["clasificador"](seed)),
        ]).fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_val)[:, 1]
        y = y_val.values
        r05 = metricas(y, (prob >= 0.5).astype(int))
        p95 = punto_con_recall_minimo(y, prob, PISO_RECALL)
        rp = metricas(y, (prob >= p95["umbral"]).astype(int)) if not np.isnan(p95["umbral"]) else {}
        ruta = Path(f"models/taller-{cid}.joblib")
        ruta.parent.mkdir(exist_ok=True)
        joblib.dump(modelo, ruta)
        pd.DataFrame({"row_id": [f"training:{i}" for i in validacion["id"]], "y_true": y, "score_attack": prob}).to_csv(
            f"data/processed/validacion_taller-{cid}.csv", index=False)
        Path("evidencias/curvas").mkdir(parents=True, exist_ok=True)
        curva(y, prob).to_csv(f"evidencias/curvas/curva_validacion_{cid}.csv", index=False)
        with mlflow.start_run(run_name=f"taller-{cid}-{cfg['nombre']}"):
            mlflow.set_tags({"config_id": cid, "familia": cfg["familia"], "hipotesis": hipotesis(cid),
                             "criterio": cfg["criterio"], "git_commit": git("rev-parse", "HEAD"),
                             "git_cambios_sin_commit": str(bool(git("status", "--porcelain", "src", "params.yaml"))),
                             "modelo_sha256": sha256(ruta), "etapa": "validacion",
                             "denominador_cero": "precision y recall con zero_division=0; FPR = FP/(FP+TN), nunca 0/0 (hay normales)",
                             **ref_datos})
            mlflow.log_params({**cfg["parametros"], "seed": seed, "categoricas": cfg["categoricas"], "umbral_reporte": 0.5,
                               "piso_recall": PISO_RECALL, "filas_entrenamiento": len(X_tr), "filas_validacion": len(X_val)})
            mlflow.log_metrics({
                **{f"val_{k}": float(v) for k, v in r05.items()},
                "val_roc_auc": roc_auc_score(y, prob), "val_pr_auc": average_precision_score(y, prob),
                "val_p95_precision": p95["precision"], "val_p95_umbral": p95["umbral"],
                **{f"val_p95_{k}": float(v) for k, v in rp.items()}})
            mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle")
            run_id = mlflow.active_run().info.run_id
        print(cid, cfg["nombre"], run_id)
        print("umbral 0,5:", {k: round(float(v), 4) for k, v in r05.items()})
        print("max precisión con recall >= 0,95:", {k: round(float(v), 4) for k, v in {**p95, **{"recall": rp.get("recall", float("nan"))}}.items()})
        print("roc_auc", round(roc_auc_score(y, prob), 4))


if __name__ == "__main__":
    main()
