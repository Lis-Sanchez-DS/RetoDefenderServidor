"""Exporta las ejecuciones del experimento `taller-evaluable-70-30` a evidencias/ (sin servidor).

Genera evidencias/experimentos.csv (una fila por configuración, con la última ejecución de cada una)
y evidencias/mlflow/<run_id>.json con parámetros, métricas, etiquetas y metadatos del artefacto.
Las conclusiones de validación salen de evidencias/conclusiones_validacion.json (veredictos del usuario).
Uso: .venv/bin/python -m src.exportar_mlflow
"""
import json
from pathlib import Path

import mlflow
import pandas as pd


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    conclusiones = json.load(open("evidencias/conclusiones_validacion.json"))
    runs = mlflow.search_runs(experiment_names=["taller-evaluable-70-30"], output_format="list")
    ultimas = {}
    for r in sorted(runs, key=lambda r: r.info.start_time):
        ultimas[r.data.tags["config_id"]] = r  # la última de cada configuración
    Path("evidencias/mlflow").mkdir(parents=True, exist_ok=True)
    for viejo in Path("evidencias/mlflow").glob("*.json"):
        viejo.unlink()
    filas = []
    for cid, r in sorted(ultimas.items()):
        d, m, t = r.data.params, r.data.metrics, r.data.tags
        json.dump({"run_id": r.info.run_id, "run_name": r.info.run_name, "estado": r.info.status,
                   "inicio_ms": r.info.start_time, "parametros": d, "metricas": m, "etiquetas": t,
                   "artefacto": f"{r.info.artifact_uri}/modelo", "modelo_sha256": t.get("modelo_sha256")},
                  open(f"evidencias/mlflow/{r.info.run_id}.json", "w"), indent=2, ensure_ascii=False)
        propio = d.get("umbral_propio", "ninguno") != "ninguno"
        filas.append({"config_id": cid, "run_id": r.info.run_id, "hipotesis": t["hipotesis"], "familia": t["familia"],
                      "parametros": json.dumps({k: v for k, v in d.items() if k not in ("seed", "umbral_reporte", "umbral_propio", "piso_recall")}, ensure_ascii=False),
                      "semilla": d["seed"], "git_commit": t["git_commit"], "git_cambios_sin_commit": t["git_cambios_sin_commit"],
                      "referencia_dvc_datos_originales": t["dvc_md5_datos_originales_training"],
                      "referencia_dvc_datos_preparados": t["dvc_md5_datos_preparados_particion"],
                      "umbral": float(d["umbral_propio"]) if propio else float(d["umbral_reporte"]),
                      "recall": m["val_upropio_recall"] if propio else m["val_recall"],
                      "precision": m["val_upropio_precision"] if propio else m["val_precision"],
                      "fpr": m["val_upropio_tasa_falsos_positivos"] if propio else m["val_tasa_falsos_positivos"],
                      "macro_f1": m["val_upropio_macro_f1"] if propio else m["val_macro_f1"],
                      "tn": m["val_upropio_tn"] if propio else m["val_tn"], "fp": m["val_upropio_fp"] if propio else m["val_fp"],
                      "fn": m["val_upropio_fn"] if propio else m["val_fn"], "tp": m["val_upropio_tp"] if propio else m["val_tp"],
                      "roc_auc": m["val_roc_auc"], "modelo_sha256": t["modelo_sha256"],
                      "conclusion_validacion": conclusiones.get(cid, "")})
    df = pd.DataFrame(filas)
    df.to_csv("evidencias/experimentos.csv", index=False)
    print(df.drop(columns=["hipotesis", "parametros", "referencia_dvc_datos_originales", "referencia_dvc_datos_preparados", "modelo_sha256", "conclusion_validacion", "git_commit"]).round(4).to_string())


if __name__ == "__main__":
    main()
