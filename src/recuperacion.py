"""Prueba de recuperación de v1 y v2 (taller evaluable).

  muestra    elige 50 filas fijas de validación (25 normales, 25 ataques; semilla 42) -> entrada_validacion.csv
  original   predice esa muestra con los modelos locales de models/ -> salidas_originales_v{1,2}.csv (ANTES)
  recuperar  (entorno nuevo) descarga v1 y v2 de Hugging Face por su SHA completo, carga modelo y umbral,
             y predice la misma muestra -> salidas_recuperadas_v{1,2}.csv (DESPUÉS)
  comparar   compara predicciones y puntuaciones de antes y después, con hashes -> resultado.json

Uso: .venv/bin/python -m src.recuperacion {muestra|original|recuperar|comparar} [carpeta]
La carpeta por defecto es evidencias/recuperacion.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.datos import RUTA, separar_xy
from src.particion import SALIDA

TOLERANCIA = 1e-9  # mismas versiones de librerías: se espera igualdad exacta; se declara esta tolerancia numérica
N_POR_CLASE = 25


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def predecir(modelo, entrada, umbral):
    X, _ = separar_xy(entrada)
    score = modelo.predict_proba(X)[:, 1]
    return pd.DataFrame({"row_id": [f"training:{i}" for i in entrada["id"]], "score_attack": score,
                         "y_pred": (score >= umbral).astype(int)})


def main(orden, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    sel = json.load(open("evidencias/seleccion.json"))
    entrada_csv = carpeta / "entrada_validacion.csv"
    if orden == "muestra":
        p = pd.read_csv(SALIDA)
        val = p[p.conjunto == "validacion"]
        df = pd.read_csv(RUTA.format("training"))
        df = df[df["id"].isin(val["id"])]
        rng = np.random.default_rng(42)
        partes = [g.loc[rng.choice(g.index, N_POR_CLASE, replace=False)] for _, g in df.groupby("label")]
        pd.concat(partes).sort_values("id").to_csv(entrada_csv, index=False)
        print("muestra:", len(pd.read_csv(entrada_csv)), "filas ->", entrada_csv)
        return 0
    entrada = pd.read_csv(entrada_csv)
    if orden in ("original", "recuperar"):
        p = pd.read_csv(SALIDA)  # la partición viene de DVC: la muestra debe estar en validación
        assert set(entrada["id"]) <= set(p.loc[p.conjunto == "validacion", "id"]), "la muestra no es de validación"
        for v in ("v1", "v2"):
            ficha = sel[v]
            if orden == "original":
                ruta_modelo, umbral = ficha["archivo_modelo"], ficha["umbral"]
                assert sha256(ruta_modelo) == ficha["modelo_sha256"]
            else:
                from huggingface_hub import hf_hub_download
                hf = ficha["huggingface"]
                assert len(hf["revision_sha"]) == 40, "hace falta el SHA completo de la revisión"
                ruta_modelo = hf_hub_download(hf["repositorio"], "model.joblib", revision=hf["revision_sha"])
                cfg = json.load(open(hf_hub_download(hf["repositorio"], "config.json", revision=hf["revision_sha"])))
                umbral = cfg["umbral"]
                assert sha256(ruta_modelo) == ficha["modelo_sha256"] == cfg["modelo_sha256"], "el hash no coincide"
                assert umbral == ficha["umbral"]
                print(f"{v}: revisión {hf['revision_sha']} descargada; hash del modelo coincide; umbral {umbral}")
            salida = predecir(joblib.load(ruta_modelo), entrada, umbral)
            nombre = "originales" if orden == "original" else "recuperadas"
            salida.to_csv(carpeta / f"salidas_{nombre}_{v}.csv", index=False)
            print(f"{v}: {len(salida)} filas, {int(salida.y_pred.sum())} marcadas como ataque -> salidas_{nombre}_{v}.csv")
        return 0
    if orden == "comparar":
        res = {"entrada_sha256": sha256(entrada_csv), "filas": len(entrada), "tolerancia_probabilidades": TOLERANCIA, "versiones": {}}
        ok = True
        for v in ("v1", "v2"):
            a, b = (pd.read_csv(carpeta / f"salidas_{n}_{v}.csv") for n in ("originales", "recuperadas"))
            iguales_pred = bool((a.y_pred == b.y_pred).all()) and (a.row_id == b.row_id).all()
            dif = float((a.score_attack - b.score_attack).abs().max())
            aprobado = bool(iguales_pred and dif <= TOLERANCIA)
            ok &= aprobado
            res["versiones"][v] = {"revision_hf": sel[v]["huggingface"]["revision_sha"], "modelo_sha256": sel[v]["modelo_sha256"],
                                   "umbral": sel[v]["umbral"], "predicciones_identicas": bool(iguales_pred),
                                   "diferencia_maxima_en_probabilidad": dif, "aprobado": aprobado,
                                   "salidas_originales_sha256": sha256(carpeta / f"salidas_originales_{v}.csv"),
                                   "salidas_recuperadas_sha256": sha256(carpeta / f"salidas_recuperadas_{v}.csv")}
            print(v, "predicciones idénticas:", iguales_pred, "| diferencia máxima de probabilidad:", dif, "|", "OK" if aprobado else "FALLA")
        res["resultado"] = "APROBADA" if ok else "FALLIDA"
        json.dump(res, open(carpeta / "resultado.json", "w"), indent=2)
        print("prueba de recuperación:", res["resultado"])
        return 0 if ok else 1
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "evidencias/recuperacion"))
