"""Experimento 5: búsqueda bayesiana de hiperparámetros de XGBoost con Optuna.

Objetivo: maximizar la precisión media con recall >= 0,95 en 5 pliegues conscientes de
grupos (los mismos del experimento 4). Poda temprana con MedianPruner. Cada prueba queda
en MLflow como ejecución anidada. El conjunto de prueba no se usa.
Al terminar, se reajusta y registra la mejor configuración (cv-grupos-xgboost-optuna).
Uso: .venv/bin/python -m src.optuna_xgboost [n_pruebas]
"""
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import optuna
import yaml
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.datos import cargar, pliegues_por_grupos, separar_xy
from src.preprocesamiento import columnas_numericas, construir_preprocesamiento
from src.validacion_cruzada import PISO_RECALL_NUEVO, ejecutar, punto_con_recall_minimo

N_PRUEBAS = 50
TIMEOUT_S = 3 * 3600  # tope de seguridad


def parametros(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def modelo(X, seed, **p):
    return Pipeline([
        ("preprocesamiento", construir_preprocesamiento(columnas_numericas(X), True, False)),
        ("clasificador", XGBClassifier(tree_method="hist", random_state=seed, n_jobs=4, **p)),
    ])


def main():
    n_pruebas = int(sys.argv[1]) if len(sys.argv) > 1 else N_PRUEBAS
    params = yaml.safe_load(open("params.yaml"))
    seed = params["seed"]
    X, y = separar_xy(cargar("training"))
    pliegues = pliegues_por_grupos(X, y, n_pliegues=5, seed=seed)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("unsw-nb15")

    def objetivo(trial):
        p = parametros(trial)
        with mlflow.start_run(run_name=f"optuna-prueba-{trial.number}", nested=True):
            mlflow.set_tags({"etapa": "busqueda_optuna", "criterio": f"max precision con recall >= {PISO_RECALL_NUEVO}"})
            mlflow.log_params(p)
            precisiones, fps, fprs = [], [], []
            for k, (i_tr, i_val) in enumerate(pliegues):
                m = modelo(X, seed, **p).fit(X.iloc[i_tr], y.iloc[i_tr])
                r = punto_con_recall_minimo(y.iloc[i_val], m.predict_proba(X.iloc[i_val])[:, 1], PISO_RECALL_NUEVO)
                precisiones.append(r["precision"]); fps.append(r["fp"]); fprs.append(r["fpr"])
                mlflow.log_metrics({f"pliegue_{k}_precision_recall095": r["precision"], f"pliegue_{k}_fp_recall095": r["fp"]})
                trial.report(float(np.mean(precisiones)), k)
                if trial.should_prune():
                    mlflow.set_tag("estado", "podada")
                    raise optuna.TrialPruned()
            res = dict(precision_recall095_media=float(np.mean(precisiones)), precision_recall095_std=float(np.std(precisiones, ddof=1)),
                       fp_recall095_media=float(np.mean(fps)), fpr_recall095_media=float(np.mean(fprs)))
            mlflow.log_metrics(res); mlflow.set_tag("estado", "completa")
            trial.set_user_attr("fp_media", res["fp_recall095_media"])
            return res["precision_recall095_media"]

    estudio = optuna.create_study(
        study_name="xgboost-recall095", direction="maximize", storage="sqlite:///optuna.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    with mlflow.start_run(run_name="optuna-xgboost-busqueda"):
        mlflow.set_tags({"etapa": "busqueda_optuna", "regimen_pliegues": "grupos"})
        mlflow.log_params({"n_pruebas": n_pruebas, "piso_recall": PISO_RECALL_NUEVO, "sampler": "TPE", "pruner": "MedianPruner"})
        pendientes = max(0, n_pruebas - len(estudio.trials))
        estudio.optimize(objetivo, n_trials=pendientes, timeout=TIMEOUT_S)
        completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        podadas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.PRUNED]
        mlflow.log_metrics({"pruebas_completas": len(completas), "pruebas_podadas": len(podadas),
                            "mejor_precision_recall095": estudio.best_value})
        mlflow.log_params({f"mejor_{k}": v for k, v in estudio.best_params.items()})
    Path("reports").mkdir(exist_ok=True)
    json.dump({"mejor_precision_recall095": estudio.best_value, "mejores_parametros": estudio.best_params,
               "pruebas_completas": len(completas), "pruebas_podadas": len(podadas)},
              open("reports/optuna_mejores_parametros.json", "w"), indent=2)
    print("completas", len(completas), "podadas", len(podadas), "| mejor", round(estudio.best_value, 4), estudio.best_params)

    # Reajuste y registro de la mejor configuración (mismos pliegues; guarda el modelo y las predicciones fuera de muestra)
    mejor = estudio.best_params
    cfg = dict(escalar=False,
               clasificador=lambda s: XGBClassifier(tree_method="hist", random_state=s, n_jobs=4, **mejor),
               parametros={"modelo": "XGBClassifier", "origen": "optuna", **{k: v for k, v in mejor.items()}})
    d, res = ejecutar("cv-grupos-xgboost-optuna", cfg, X, y, pliegues, params, "grupos")
    print(d.round(4).to_string())
    print({k: round(v, 4) for k, v in res.items() if k.endswith("_media")})


if __name__ == "__main__":
    main()
