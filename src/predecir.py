"""Carga un modelo guardado y predice sobre filas nuevas. No es parte del pipeline de Prefect.

El modelo guardado es la solución completa (preprocesamiento + clasificador), así que la
entrada son las columnas originales del conjunto de datos. Por defecto se carga el modelo
registrado en MLflow con el alias `candidato`, y el umbral sale de su etiqueta `umbral`
(0,6, decisión del experimento 6). También se puede cargar un archivo de `models/`,
comprobando antes su md5 contra DVC.

Uso:
    .venv/bin/python -m src.predecir entrada.csv [salida.csv]
    .venv/bin/python -m src.predecir entrada.csv --modelo models/cv-grupos-xgboost-optuna.joblib --umbral 0.6
"""
import argparse
import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient

from src.datos import COLUMNAS_EXCLUIDAS, ESTADOS_EXCLUIDOS

MLFLOW_URI = "sqlite:///mlflow.db"


def cargar_modelo(ruta: str | None = None, nombre="defender-servidor", alias="candidato"):
    """Devuelve (modelo, umbral, descripcion) desde un archivo de DVC o desde el registro de MLflow."""
    if ruta:
        archivo = Path(ruta)
        dvc = Path(f"{archivo}.dvc")
        if dvc.exists():
            esperado = yaml.safe_load(dvc.read_text())["outs"][0]["md5"]
            if hashlib.md5(archivo.read_bytes()).hexdigest() != esperado:
                raise ValueError(f"{archivo} no coincide con la versión registrada en DVC")
        return joblib.load(archivo), None, f"archivo {archivo}"
    mlflow.set_tracking_uri(MLFLOW_URI)
    version = MlflowClient().get_model_version_by_alias(nombre, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@{alias}")
    umbral = float(version.tags["umbral"])
    return modelo, umbral, f"registro MLflow {nombre} v{version.version} (alias {alias}, origen {version.tags.get('origen')})"


def predecir(modelo, datos: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Probabilidad y decisión (1 = ataque) con el umbral dado; quita columnas que no son entradas."""
    X = datos.drop(columns=[c for c in COLUMNAS_EXCLUIDAS if c in datos.columns])
    prob = modelo.predict_proba(X)[:, 1]
    return pd.DataFrame({"prob_ataque": prob, "prediccion": (prob >= umbral).astype(int)}, index=datos.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", help="CSV con las columnas originales (se ignoran id, label y attack_cat)")
    ap.add_argument("salida", nargs="?", default=None, help="CSV de salida (opcional)")
    ap.add_argument("--modelo", default=None, help="archivo de models/ en lugar del registro de MLflow")
    ap.add_argument("--umbral", type=float, default=None, help="obligatorio con --modelo; con el registro sale de la etiqueta")
    a = ap.parse_args()

    modelo, umbral, origen = cargar_modelo(a.modelo)
    umbral = a.umbral if a.umbral is not None else umbral
    if umbral is None:
        raise SystemExit("con --modelo hay que indicar --umbral")
    datos = pd.read_csv(a.entrada)
    datos = datos[~datos["state"].isin(ESTADOS_EXCLUIDOS)]  # mismos estados que en el entrenamiento
    res = predecir(modelo, datos, umbral)
    print(f"modelo: {origen}; umbral {umbral}; {len(res)} filas; {res.prediccion.mean():.1%} marcadas como ataque")
    if "label" in datos.columns:
        y = datos["label"]
        print(f"(con etiquetas) recall {(res.prediccion[y == 1] == 1).mean():.3f}, "
              f"tasa de falsos positivos {(res.prediccion[y == 0] == 1).mean():.3f}")
    if a.salida:
        res.to_csv(a.salida, index_label="fila")
        print("guardado en", a.salida)
    else:
        print(res.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
