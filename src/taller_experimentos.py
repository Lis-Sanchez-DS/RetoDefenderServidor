"""Cuatro configuraciones del taller evaluable (docs/taller_hipotesis.md), dos familias.

Misma partición fija, mismo preprocesamiento (ajustado solo con entrenamiento) y mismas
métricas de validación. Cada ejecución queda en MLflow (experimento `taller-evaluable`)
con hipótesis, parámetros, semilla, versión de datos, commit, métricas y el modelo
completo; además se guarda `models/taller-<id>.joblib` y sus probabilidades de validación.
El conjunto de prueba no se toca. Uso: .venv/bin/python -m src.taller_experimentos [C1 C2 ...]
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline import metricas
from src.datos import separar_xy
from src.particion import SALIDA, cargar_particion
from src.preprocesamiento import columnas_numericas, construir_preprocesamiento

PISO_RECALL = 0.95
REJILLA = np.round(np.arange(0.05, 0.96, 0.05), 2)
PARAMETROS_OPTUNA = "reports/optuna_taller_mejores_parametros.json"


def configuraciones():
    mejores = json.load(open(PARAMETROS_OPTUNA))["mejores_parametros"] if Path(PARAMETROS_OPTUNA).exists() else None
    cfgs = {
        "C1": dict(familia="regresion-logistica", categoricas=False, escalar=True,
                   clasificador=lambda s: LogisticRegression(max_iter=2000, random_state=s),
                   parametros={"modelo": "LogisticRegression", "max_iter": 2000},
                   hipotesis="Solo numéricas: recall alto y muchas falsas alarmas (FPR > 0,15 con umbral 0,5)."),
        "C2": dict(familia="regresion-logistica", categoricas=True, escalar=True,
                   clasificador=lambda s: LogisticRegression(max_iter=2000, random_state=s),
                   parametros={"modelo": "LogisticRegression", "max_iter": 2000},
                   hipotesis="Añadir categóricas baja la FPR al menos 3 puntos frente a C1; sigue por debajo de XGBoost. Será v1."),
        "C3": dict(familia="arboles-xgboost", categoricas=True, escalar=False,
                   clasificador=lambda s: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                        tree_method="hist", random_state=s, n_jobs=8),
                   parametros={"modelo": "XGBClassifier", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
                   hipotesis="XGBoost supera a C2 en precisión (recall >= 0,95) por más de 0,01 y reduce las falsas alarmas a la mitad o menos."),
    }
    if mejores:
        cfgs["C4"] = dict(familia="arboles-xgboost", categoricas=True, escalar=False,
                          clasificador=lambda s: XGBClassifier(tree_method="hist", random_state=s, n_jobs=8, **mejores),
                          parametros={"modelo": "XGBClassifier", "origen": "optuna", **mejores},
                          hipotesis="Optuna (solo con entrenamiento) mejora a C3 menos de 0,01 en precisión con recall >= 0,95. Candidato a v2.")
    return cfgs


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def md5_dvc(ruta):
    return yaml.safe_load(Path(ruta).read_text())["outs"][0]["md5"]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def umbral_por_validacion(y, prob):
    """Mayor umbral de la rejilla (paso 0,05) con recall de validación >= piso; 0,5 si ninguno."""
    ok = [u for u in REJILLA if (prob[y == 1] >= u).mean() >= PISO_RECALL]
    return float(max(ok)) if ok else 0.5


def main():
    seed = yaml.safe_load(open("params.yaml"))["seed"]
    entrenamiento, validacion = cargar_particion()
    X_tr, y_tr = separar_xy(entrenamiento)
    X_val, y_val = separar_xy(validacion)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("taller-evaluable")
    cfgs = configuraciones()
    elegidos = sys.argv[1:] or list(cfgs)
    ref_datos = {"dvc_md5_datos_originales_training": md5_dvc("data/raw/unsw-nb15/particion_oficial/UNSW_NB15_training-set.csv.dvc"),
                 "dvc_md5_datos_preparados_particion": md5_dvc(f"{SALIDA}.dvc") if Path(f"{SALIDA}.dvc").exists() else "sin-dvc"}
    for cid in elegidos:
        cfg = cfgs[cid]
        modelo = Pipeline([
            ("preprocesamiento", construir_preprocesamiento(columnas_numericas(X_tr), cfg["categoricas"], cfg["escalar"])),
            ("clasificador", cfg["clasificador"](seed)),
        ]).fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_val)[:, 1]
        y = y_val.values
        umbral = umbral_por_validacion(y, prob)
        r05 = metricas(y, (prob >= 0.5).astype(int))
        ru = metricas(y, (prob >= umbral).astype(int))
        ruta = Path(f"models/taller-{cid}.joblib")
        ruta.parent.mkdir(exist_ok=True)
        joblib.dump(modelo, ruta)
        pd.DataFrame({"row_id": [f"training:{i}" for i in validacion["id"]], "y_true": y, "score_attack": prob}).to_csv(
            f"data/processed/validacion_taller-{cid}.csv", index=False)
        with mlflow.start_run(run_name=f"taller-{cid}"):
            mlflow.set_tags({"config_id": cid, "familia": cfg["familia"], "hipotesis": cfg["hipotesis"],
                             "git_commit": git("rev-parse", "HEAD"),
                             "git_cambios_sin_commit": str(bool(git("status", "--porcelain", "src", "params.yaml"))),
                             "modelo_sha256": sha256(ruta), "etapa": "validacion",
                             "denominador_cero": "precision y recall con zero_division=0; FPR = FP/(FP+TN), nunca 0/0 (hay normales)",
                             **ref_datos})
            mlflow.log_params({**cfg["parametros"], "seed": seed, "categoricas": cfg["categoricas"],
                               "umbral": umbral, "regla_umbral": "mayor umbral (paso 0,05) con recall de validacion >= 0,95",
                               "filas_entrenamiento": len(X_tr), "filas_validacion": len(X_val)})
            mlflow.log_metrics({
                **{f"val_{k}": float(v) for k, v in ru.items()},
                **{f"val_u05_{k}": float(v) for k, v in r05.items()},
                "val_roc_auc": roc_auc_score(y, prob), "val_pr_auc": average_precision_score(y, prob)})
            mlflow.sklearn.log_model(modelo, name="modelo", serialization_format="cloudpickle")
            run_id = mlflow.active_run().info.run_id
        print(cid, run_id, "umbral", umbral, {k: round(v, 4) for k, v in ru.items() if k not in ("tn", "fp", "fn", "tp")},
              "| u0.5 FPR", round(r05["tasa_falsos_positivos"], 4))


if __name__ == "__main__":
    main()
