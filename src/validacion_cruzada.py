"""Experimento 3: regresión logística y XGBoost con validación cruzada por bloques.

Mismos pliegues y mismo preprocesamiento para ambos. Objetivo: mayor precisión con
recall >= 0,90. El umbral no se elige aquí (se reporta el umbral 0,5 y la precisión
máxima con recall >= 0,90). El conjunto de prueba no se usa.
Tras la validación cruzada, cada modelo se reajusta con todo el entrenamiento y se guarda.
"""
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.baseline import metricas
from src.datos import cargar, pliegues_por_bloques, pliegues_por_grupos, separar_xy
from src.preprocesamiento import columnas_numericas, construir_preprocesamiento

PISO_RECALL = 0.90

MODELOS = {
    "cv-regresion-logistica": dict(
        escalar=True,
        clasificador=lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
        parametros={"modelo": "LogisticRegression", "max_iter": 2000},
    ),
    "cv-xgboost": dict(
        escalar=False,  # los árboles no necesitan estandarizar
        clasificador=lambda seed: XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
            random_state=seed, n_jobs=4),
        parametros={"modelo": "XGBClassifier", "n_estimators": 300, "max_depth": 6,
                    "learning_rate": 0.1, "tree_method": "hist"},
    ),
}


def precision_con_recall_minimo(y, prob, piso=PISO_RECALL):
    """Mayor precisión alcanzable con recall >= piso, y el umbral que la produce."""
    precision, recall, umbrales = precision_recall_curve(y, prob)
    ok = recall[:-1] >= piso
    if not ok.any():
        return float("nan"), float("nan")
    i = np.argmax(np.where(ok, precision[:-1], -1))
    return float(precision[i]), float(umbrales[i])


def construir(cfg, X, seed):
    return Pipeline([
        ("preprocesamiento", construir_preprocesamiento(columnas_numericas(X), True, cfg["escalar"])),
        ("clasificador", cfg["clasificador"](seed)),
    ])


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def ejecutar(nombre, cfg, X, y, pliegues, params, regimen="bloques"):
    seed = params["seed"]
    oof = pd.DataFrame({"y": y.values, "prob": np.nan, "pliegue": -1})
    filas = []
    with mlflow.start_run(run_name=nombre):
        md5 = yaml.safe_load(Path("data/raw/unsw-nb15/particion_oficial/UNSW_NB15_training-set.csv.dvc").read_text())["outs"][0]["md5"]
        mlflow.set_tags({
            "etapa": "validacion_cruzada",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_cambios_sin_commit": str(bool(_git("status", "--porcelain", "src", "params.yaml"))),
            "datos_dvc_md5_entrenamiento": md5,
            "criterio": f"max precision con recall >= {PISO_RECALL}",
            "regimen_pliegues": regimen,
        })
        mlflow.log_params({**cfg["parametros"], "n_pliegues": len(pliegues), "tamano_bloque": 500,
                           "seed": seed, "umbral": 0.5, "piso_recall": PISO_RECALL,
                           "categoricas": True})
        for k, (i_tr, i_val) in enumerate(pliegues):
            modelo = construir(cfg, X, seed).fit(X.iloc[i_tr], y.iloc[i_tr])
            prob = modelo.predict_proba(X.iloc[i_val])[:, 1]
            yv = y.iloc[i_val]
            r = metricas(yv, (prob >= 0.5).astype(int))
            r["roc_auc"] = roc_auc_score(yv, prob)
            r["pr_auc"] = average_precision_score(yv, prob)
            r["precision_con_recall_090"], r["umbral_para_recall_090"] = precision_con_recall_minimo(yv, prob)
            oof.loc[oof.index[i_val], ["prob", "pliegue"]] = np.column_stack([prob, np.full(len(prob), k)])
            filas.append(r)
            mlflow.log_metrics({f"pliegue_{k}_{m}": float(v) for m, v in r.items()
                                if m not in ("tn", "fp", "fn", "tp")})
        d = pd.DataFrame(filas)
        resumen = {}
        for m in ["recall", "precision", "tasa_falsos_positivos", "macro_f1", "roc_auc", "pr_auc",
                  "precision_con_recall_090", "umbral_para_recall_090"]:
            resumen[f"{m}_media"], resumen[f"{m}_std"] = float(d[m].mean()), float(d[m].std())
        resumen["fp_media"], resumen["fn_media"] = float(d.fp.mean()), float(d.fn.mean())
        mlflow.log_metrics(resumen)
        # Modelo final: reajustado con todo el entrenamiento (para versionarlo)
        final = construir(cfg, X, seed).fit(X, y)
        mlflow.sklearn.log_model(final, name="modelo", registered_model_name=nombre,
                                 serialization_format="cloudpickle")
        Path("models").mkdir(exist_ok=True)
        joblib.dump(final, f"models/{nombre}.joblib")
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        oof.to_csv(f"data/processed/oof_{nombre}.csv", index_label="fila")
    return d, resumen


def main():
    params = yaml.safe_load(open("params.yaml"))
    X, y = separar_xy(cargar("training"))
    pliegues = pliegues_por_bloques(len(X), n_pliegues=5, tamano_bloque=500, seed=params["seed"])
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("unsw-nb15")
    args = sys.argv[1:]
    regimen = "grupos" if "--grupos" in args else "bloques"
    elegidos = [a for a in args if not a.startswith("--")] or list(MODELOS)  # opcionalmente, solo algunos
    if regimen == "grupos":
        pliegues = pliegues_por_grupos(X, y, n_pliegues=5, seed=params["seed"])
    for base, cfg in ((n, MODELOS[n]) for n in elegidos):
        nombre = base if regimen == "bloques" else base.replace("cv-", "cv-grupos-", 1)
        d, res = ejecutar(nombre, cfg, X, y, pliegues, params, regimen)
        print("\n", nombre)
        print(d.round(4).to_string())
        print({k: round(v, 4) for k, v in res.items() if k.endswith("_media")})


if __name__ == "__main__":
    main()
