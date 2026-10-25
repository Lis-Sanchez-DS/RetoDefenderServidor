"""Búsqueda de hiperparámetros de XGBoost para la configuración C4 del taller.

Usa SOLO el conjunto de entrenamiento de la partición fija (validación cruzada de 3
pliegues por grupos dentro de él), de modo que la validación y la prueba no influyen.
Objetivo (decisión del usuario): mayor precisión media con umbral fijo 0,5, contando solo pruebas con recall medio >= 0,95
(las que no llegan devuelven recall - 1, un valor negativo). Uso: .venv/bin/python -m src.optuna_taller [n_pruebas]
"""
import json
import sys

import numpy as np
import optuna
import yaml

from src.datos import pliegues_por_grupos, separar_xy
from src.optuna_xgboost import modelo, parametros
from src.particion import cargar_particion
from src.baseline import metricas
from src.validacion_cruzada import PISO_RECALL_NUEVO

UMBRAL = 0.5


def main(n_pruebas=25):
    seed = yaml.safe_load(open("params.yaml"))["seed"]
    entrenamiento, _ = cargar_particion()
    X, y = separar_xy(entrenamiento)
    pliegues = pliegues_por_grupos(X, y, n_pliegues=3, seed=seed)

    def objetivo(trial):
        p = parametros(trial)
        precisiones, recalls = [], []
        for k, (i_tr, i_val) in enumerate(pliegues):
            m = modelo(X, seed, **p).set_params(clasificador__n_jobs=8).fit(X.iloc[i_tr], y.iloc[i_tr])
            pred = (m.predict_proba(X.iloc[i_val])[:, 1] >= UMBRAL).astype(int)
            r = metricas(y.iloc[i_val], pred)
            precisiones.append(r["precision"])
            recalls.append(r["recall"])
            trial.report(float(np.mean(precisiones)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        recall = float(np.mean(recalls))
        return float(np.mean(precisiones)) if recall >= PISO_RECALL_NUEVO else recall - 1

    estudio = optuna.create_study(
        study_name="xgboost-taller", direction="maximize", storage="sqlite:///optuna_taller.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    estudio.optimize(objetivo, n_trials=max(0, n_pruebas - len(estudio.trials)), timeout=2 * 3600)
    completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    json.dump({"fuente": "solo conjunto de entrenamiento de la partición fija; CV de 3 pliegues por grupos",
               "mejor_precision_umbral05_con_recall_minimo_095": estudio.best_value, "mejores_parametros": estudio.best_params,
               "pruebas_completas": len(completas), "pruebas_totales": len(estudio.trials)},
              open("reports/optuna_taller_mejores_parametros.json", "w"), indent=2)
    print("mejor", round(estudio.best_value, 4), estudio.best_params, "| completas", len(completas))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
