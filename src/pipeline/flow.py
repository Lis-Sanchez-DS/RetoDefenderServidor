"""Flujo de Prefect: preparar -> entrenar -> comparar -> registrar.

Reproduce, con los mismos pliegues por grupos y el mismo preprocesamiento, el
entrenamiento de los tres candidatos finales (regresión logística, XGBoost por
defecto y XGBoost con los hiperparámetros de Optuna), elige el mejor por
validación y lo registra en MLflow con el umbral congelado.

No evalúa en el conjunto de prueba: esa evaluación se hace una sola vez, aparte
(`src/evaluar_test.py`). Cada etapa falla con un mensaje que dice qué falló y por qué.

Uso:
    .venv/bin/python -m src.pipeline.flow                  # ejecución completa (~5 min)
    .venv/bin/python -m src.pipeline.flow --muestra 0.05   # ejecución rápida con 5 % de las filas
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.exceptions import MissingContextError
from prefect.runtime import flow_run
from xgboost import XGBClassifier

from src.datos import RUTA, cargar, pliegues_por_grupos, separar_xy
from src.validacion_cruzada import MODELOS, ejecutar
from src.evaluar_test import todas_las_metricas

MLFLOW_URI = "sqlite:///mlflow.db"


def _log():
    """Registro de Prefect dentro de una ejecución; uno normal fuera (por ejemplo, en pruebas)."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


class EtapaFallida(RuntimeError):
    """Error de una etapa del pipeline, con la causa explicada."""


def _md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


@task(name="preparar-datos")
def preparar_datos(params: dict, muestra: float | None):
    """Comprueba el archivo contra DVC, carga el entrenamiento y construye los pliegues."""
    log = _log()
    ruta = Path(RUTA.format("training"))
    ruta_dvc = Path(f"{ruta}.dvc")
    if not ruta_dvc.exists():
        raise EtapaFallida(f"falta {ruta_dvc}: el repositorio no está completo")
    if not ruta.exists():
        raise EtapaFallida(f"falta {ruta}; recupérelo con `dvc pull` (ver README)")
    esperado = yaml.safe_load(ruta_dvc.read_text())["outs"][0]["md5"]
    real = _md5(ruta)
    if real != esperado:
        raise EtapaFallida(f"{ruta} no coincide con la versión de DVC (md5 {real} frente a {esperado})")

    datos = cargar("training")
    if muestra:
        datos = datos.sample(frac=muestra, random_state=params["seed"]).reset_index(drop=True)
    X, y = separar_xy(datos)
    if y.nunique() != 2 or X.isna().all().any():
        raise EtapaFallida("el entrenamiento no tiene las dos clases o tiene columnas vacías")
    pliegues = pliegues_por_grupos(X, y, n_pliegues=params["pipeline"]["n_pliegues"], seed=params["seed"])
    log.info(f"datos verificados (md5 {real[:8]}): {len(X)} filas, {y.mean():.1%} ataques, {len(pliegues)} pliegues por grupos")
    return X, y, pliegues, real


def _candidatos(params: dict) -> dict:
    """Los tres candidatos finales; el XGBoost de Optuna usa los parámetros guardados."""
    ruta = Path(params["pipeline"]["parametros_optuna"])
    if not ruta.exists():
        raise EtapaFallida(f"falta {ruta}: ejecute la búsqueda de Optuna (src/optuna_xgboost.py)")
    mejor = json.loads(ruta.read_text())["mejores_parametros"]
    optuna = dict(
        escalar=False,
        clasificador=lambda s: XGBClassifier(tree_method="hist", random_state=s, n_jobs=4, **mejor),
        parametros={"modelo": "XGBClassifier", "origen": "optuna", **mejor},
    )
    p = params["pipeline"]["prefijo_modelos"]
    return {
        f"{p}-regresion-logistica": MODELOS["cv-regresion-logistica"],
        f"{p}-xgboost": MODELOS["cv-xgboost"],
        f"{p}-xgboost-optuna": optuna,
    }


@task(name="entrenar-modelo")
def entrenar_modelo(nombre, cfg, X, y, pliegues, params, md5_datos, muestra=None):
    """Validación cruzada por grupos y reajuste final de un candidato; todo en MLflow."""
    etiquetas = {"pipeline": "prefect", "prefect_flow_run": str(flow_run.id), "datos_md5": md5_datos,
                 "muestra": str(muestra or 1.0)}
    try:
        d, resumen = ejecutar(nombre, cfg, X, y, pliegues, params, "grupos", etiquetas)
    except Exception as e:
        raise EtapaFallida(f"falló el entrenamiento de {nombre}: {type(e).__name__}: {e}") from e
    _log().info(f"{nombre}: precisión con recall>=0,95 = {resumen['precision_con_recall_095_media']:.4f}")
    return nombre, resumen


@task(name="comparar-experimentos")
def comparar_experimentos(resultados: list, params: dict):
    """Elige el modelo por validación y comprueba el piso de recall con el umbral congelado."""
    log = _log()
    piso, umbral = params["pipeline"]["piso_recall"], params["pipeline"]["umbral"]
    filas = []
    for nombre, resumen in resultados:
        oof = pd.read_csv(f"data/processed/oof_{nombre}.csv")
        por_pliegue = pd.DataFrame([todas_las_metricas(g.y.values, g.prob.values, umbral)
                                    for _, g in oof.groupby("pliegue")])
        filas.append(dict(modelo=nombre, run_id=resumen["run_id"],
                          precision_con_recall_095=resumen["precision_con_recall_095_media"],
                          recall_umbral=por_pliegue.recall.mean(), recall_min_umbral=por_pliegue.recall.min(),
                          precision_umbral=por_pliegue.precision.mean(), mcc_umbral=por_pliegue.mcc.mean(),
                          macro_f1_umbral=por_pliegue.macro_f1.mean()))
    tabla = pd.DataFrame(filas).sort_values("precision_con_recall_095", ascending=False)
    Path("reports").mkdir(exist_ok=True)
    tabla.drop(columns="run_id").to_csv("reports/pipeline_comparacion.csv", index=False)
    log.info("comparación (validación, umbral %.2f):\n%s", umbral, tabla.drop(columns="run_id").round(4).to_string(index=False))
    admisibles = tabla[tabla.recall_min_umbral >= piso]
    if admisibles.empty:
        raise EtapaFallida(f"ningún modelo mantiene recall >= {piso} en todos los pliegues con el umbral {umbral}; "
                           f"el mejor recall mínimo fue {tabla.recall_min_umbral.max():.4f}")
    return admisibles.iloc[0].to_dict()


@task(name="registrar-modelo")
def registrar_modelo(elegido: dict, params: dict, muestra: float | None = None):
    """Registra el modelo elegido en MLflow con el umbral y el alias `candidato`."""
    p = dict(params["pipeline"])
    if muestra:  # las ejecuciones rápidas no tocan el modelo registrado de verdad
        p["modelo_registrado"] += "-muestra"
    mlflow.set_tracking_uri(MLFLOW_URI)
    cliente = MlflowClient()
    try:
        version = mlflow.register_model(f"runs:/{elegido['run_id']}/modelo", p["modelo_registrado"])
        cliente.set_model_version_tag(p["modelo_registrado"], version.version, "umbral", str(p["umbral"]))
        cliente.set_model_version_tag(p["modelo_registrado"], version.version, "origen", elegido["modelo"])
        cliente.set_registered_model_alias(p["modelo_registrado"], "candidato", version.version)
    except Exception as e:
        raise EtapaFallida(f"no se pudo registrar el modelo: {type(e).__name__}: {e}") from e
    _log().info(f"registrado {p['modelo_registrado']} v{version.version} (origen {elegido['modelo']}, umbral {p['umbral']})")
    return version.version


@flow(name="defender-servidor", log_prints=True)
def main_flow(muestra: float | None = None, piso_recall: float | None = None):
    params = yaml.safe_load(open("params.yaml"))
    if piso_recall is not None:
        params["pipeline"]["piso_recall"] = piso_recall
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("unsw-nb15")
    X, y, pliegues, md5 = preparar_datos(params, muestra)
    resultados = [entrenar_modelo(f"{n}-muestra" if muestra else n, c, X, y, pliegues, params, md5, muestra) for n, c in _candidatos(params).items()]
    elegido = comparar_experimentos(resultados, params)
    return registrar_modelo(elegido, params, muestra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=float, default=None, help="fracción de filas (ejecución rápida)")
    ap.add_argument("--piso-recall", type=float, default=None, help="sustituye el piso de params.yaml")
    a = ap.parse_args()
    main_flow(muestra=a.muestra, piso_recall=a.piso_recall)
