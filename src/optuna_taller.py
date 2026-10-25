"""Búsqueda de hiperparámetros de XGBoost para la configuración C4 del taller.

Usa SOLO el conjunto de entrenamiento de la partición fija (validación cruzada de 3
pliegues por grupos dentro de él), de modo que la validación y la prueba no influyen.
Objetivo: mayor precisión media con recall >= 0,95. Uso: .venv/bin/python -m src.optuna_taller [n_pruebas]
"""
import json
import sys

import numpy as np
import optuna
import yaml

from src.datos import pliegues_por_grupos, separar_xy
from src.optuna_xgboost import modelo, parametros
from src.particion import cargar_particion
from src.validacion_cruzada import PISO_RECALL_NUEVO, punto_con_recall_minimo


def main(n_pruebas=25):
    seed = yaml.safe_load(open("params.yaml"))["seed"]
    entrenamiento, _ = cargar_particion()
    X, y = separar_xy(entrenamiento)
    pliegues = pliegues_por_grupos(X, y, n_pliegues=3, seed=seed)

    def objetivo(trial):
        p = parametros(trial)
        precisiones = []
        for k, (i_tr, i_val) in enumerate(pliegues):
            m = modelo(X, seed, **p).set_params(clasificador__n_jobs=8).fit(X.iloc[i_tr], y.iloc[i_tr])
            r = punto_con_recall_minimo(y.iloc[i_val], m.predict_proba(X.iloc[i_val])[:, 1], PISO_RECALL_NUEVO)
            precisiones.append(r["precision"])
            trial.report(float(np.mean(precisiones)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(precisiones))

    estudio = optuna.create_study(
        study_name="xgboost-taller", direction="maximize", storage="sqlite:///optuna_taller.db", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    estudio.optimize(objetivo, n_trials=max(0, n_pruebas - len(estudio.trials)), timeout=2 * 3600)
    completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    json.dump({"fuente": "solo conjunto de entrenamiento de la partición fija; CV de 3 pliegues por grupos",
               "mejor_precision_recall095": estudio.best_value, "mejores_parametros": estudio.best_params,
               "pruebas_completas": len(completas), "pruebas_totales": len(estudio.trials)},
              open("reports/optuna_taller_mejores_parametros.json", "w"), indent=2)
    print("mejor", round(estudio.best_value, 4), estudio.best_params, "| completas", len(completas))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
