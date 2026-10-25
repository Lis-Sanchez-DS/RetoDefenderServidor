"""Partición fija entrenamiento/validación (taller evaluable).

Se crea solo con el archivo de entrenamiento oficial. Las filas con el mismo vector de
entrada (sin `id`, `label` ni `attack_cat`) forman un grupo y nunca se reparten entre
entrenamiento y validación. La asignación se guarda en un CSV que versiona DVC y que
usan los cuatro experimentos. Uso: .venv/bin/python -m src.particion
"""
import hashlib
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.datos import COLUMNAS_EXCLUIDAS, RUTA

SALIDA = Path("data/processed/particion_entrenamiento_validacion.csv")
ESTADOS_EXCLUIDOS_ENTRENAMIENTO = ["no", "ACC", "CLO"]  # ver docs/decisiones.md


def row_id(archivo: str, ids) -> pd.Series:
    """Identificador inequívoco de archivo y fila: `training:<id>` o `testing:<id>`."""
    return pd.Series([f"{archivo}:{i}" for i in ids])


def grupos_de_entradas(df: pd.DataFrame):
    entradas = df.drop(columns=COLUMNAS_EXCLUIDAS)
    return entradas.groupby(list(entradas.columns), sort=False, dropna=False).ngroup().values


def crear(seed: int = 42, val_size: float = 0.2) -> pd.DataFrame:
    df = pd.read_csv(RUTA.format("training"))
    df = df[~df["state"].isin(ESTADOS_EXCLUIDOS_ENTRENAMIENTO)].reset_index(drop=True)
    grupo = grupos_de_entradas(df)
    n_pliegues = round(1 / val_size)
    cv = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=seed)
    _, i_val = next(iter(cv.split(df, df["label"], grupo)))
    conjunto = pd.Series("entrenamiento", index=df.index)
    conjunto.iloc[i_val] = "validacion"
    return pd.DataFrame({"row_id": row_id("training", df["id"]), "id": df["id"], "conjunto": conjunto.values,
                         "grupo": grupo, "label": df["label"]})


def huella(particion: pd.DataFrame) -> dict:
    """SHA-256 de la lista ordenada de row_id de cada conjunto."""
    return {c: hashlib.sha256("\n".join(sorted(g.row_id)).encode()).hexdigest()
            for c, g in particion.groupby("conjunto")}


def cargar_particion():
    """Devuelve (entrenamiento, validacion) como DataFrames completos del archivo de entrenamiento."""
    p = pd.read_csv(SALIDA)
    df = pd.read_csv(RUTA.format("training"))
    df = df[df["id"].isin(p["id"])].merge(p[["id", "conjunto"]], on="id")
    return (df[df.conjunto == "entrenamiento"].drop(columns="conjunto").reset_index(drop=True),
            df[df.conjunto == "validacion"].drop(columns="conjunto").reset_index(drop=True))


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    p = crear(params["seed"], params["split"]["val_size"])
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(SALIDA, index=False)
    cruces = set(p[p.conjunto == "entrenamiento"].grupo) & set(p[p.conjunto == "validacion"].grupo)
    print(p.groupby("conjunto").label.agg(["size", "mean"]).round(4).to_string())
    print("grupos repartidos entre ambos conjuntos:", len(cruces))
    print(huella(p))
