"""Experimentos de regresión logística con y sin variables categóricas codificadas.

Misma partición de entrenamiento/validación en ambos; el conjunto de prueba no se usa.
Cada ejecución queda en MLflow (con el modelo registrado) y el modelo completo
(preprocesamiento + clasificador) se guarda en `models/` para versionarlo con DVC.
"""
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.baseline import metricas
from src.datos import cargar, dividir_entrenamiento_validacion, separar_xy
from src.preprocesamiento import columnas_numericas, construir_preprocesamiento

EXPERIMENTOS = {
    "regresion-logistica-sin-categoricas": dict(
        incluir_categoricas=False,
        hipotesis="Con solo las variables numericas la regresion logistica supera la linea base "
                  "y deja de alarmar todo el trafico normal.",
    ),
    "regresion-logistica-con-categoricas": dict(
        incluir_categoricas=True,
        hipotesis="Agregar service, state y proto codificados mejora el macro-F1 y reduce los "
                  "falsos positivos frente a usar solo numericas.",
    ),
}


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def _hash_datos():
    """MD5 de DVC del archivo de entrenamiento (identifica la versión exacta de los datos)."""
    ruta = Path("data/raw/unsw-nb15/particion_oficial/UNSW_NB15_training-set.csv.dvc")
    return yaml.safe_load(ruta.read_text())["outs"][0]["md5"]


def ejecutar(nombre, incluir_categoricas, hipotesis, params):
    entrenamiento, validacion = dividir_entrenamiento_validacion(
        cargar("training"), params["split"]["val_size"], seed=params["seed"]
    )
    X_tr, y_tr = separar_xy(entrenamiento)
    X_val, y_val = separar_xy(validacion)

    modelo = Pipeline([
        ("preprocesamiento", construir_preprocesamiento(columnas_numericas(X_tr), incluir_categoricas)),
        ("clasificador", LogisticRegression(max_iter=2000, random_state=params["seed"])),
    ])
    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags({
            "hipotesis": hipotesis,
            "particion": "validacion (bloques de 500 filas del entrenamiento oficial)",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_cambios_sin_commit": str(bool(_git("status", "--porcelain", "src", "params.yaml"))),
            "datos_dvc_md5_entrenamiento": _hash_datos(),
        })
        modelo.fit(X_tr, y_tr)
        mlflow.log_params({
            "modelo": "LogisticRegression", "incluir_categoricas": incluir_categoricas,
            "escalado": "StandardScaler", "umbral": 0.5, "max_iter": 2000, "seed": params["seed"],
            "val_size": params["split"]["val_size"], "columnas_entrada": len(modelo[:-1].get_feature_names_out()),
            "filas_entrenamiento": len(X_tr), "filas_validacion": len(X_val),
        })
        prob = modelo.predict_proba(X_val)[:, 1]
        res = metricas(y_val, (prob >= 0.5).astype(int))
        res["roc_auc"] = roc_auc_score(y_val, prob)
        res["pr_auc"] = average_precision_score(y_val, prob)
        res["macro_f1_entrenamiento"] = metricas(y_tr, modelo.predict(X_tr))["macro_f1"]
        res["iteraciones"] = int(modelo[-1].n_iter_[0])
        mlflow.log_metrics({k: float(v) for k, v in res.items() if k not in ("tn", "fp", "fn", "tp")})
        mlflow.sklearn.log_model(modelo, name="modelo", registered_model_name=nombre)
        Path("models").mkdir(exist_ok=True)
        joblib.dump(modelo, f"models/{nombre}.joblib")
    return res


def main():
    params = yaml.safe_load(open("params.yaml"))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("unsw-nb15")
    for nombre, cfg in EXPERIMENTOS.items():
        res = ejecutar(nombre, cfg["incluir_categoricas"], cfg["hipotesis"], params)
        print(nombre, {k: round(float(v), 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
