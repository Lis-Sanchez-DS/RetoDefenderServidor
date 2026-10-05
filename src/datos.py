"""Carga de datos y partición de entrenamiento/validación.

El archivo de entrenamiento oficial está ordenado en bloques (primero normales,
al final solo ataques), así que ni las primeras filas ni los últimos `id` sirven
como validación. Se parte por bloques contiguos asignados al azar: filas vecinas
(que comparten ráfagas de tráfico) quedan en el mismo lado de la partición.
"""
import numpy as np
import pandas as pd

RUTA = "data/raw/unsw-nb15/particion_oficial/UNSW_NB15_{}-set.csv"
# Estados que aparecen en una sola partición (5 filas en prueba, 1 en entrenamiento):
# no aportan y complicarían la codificación. Ver docs/decisiones.md.
ESTADOS_EXCLUIDOS = ["no", "ACC", "CLO"]
# `id` solo refleja el orden del archivo; `label` y `attack_cat` son la respuesta.
COLUMNAS_EXCLUIDAS = ["id", "label", "attack_cat"]


def cargar(particion: str) -> pd.DataFrame:
    """Lee 'training' o 'testing' de la partición oficial, sin los estados excluidos."""
    df = pd.read_csv(RUTA.format(particion))
    return df[~df["state"].isin(ESTADOS_EXCLUIDOS)].reset_index(drop=True)


def separar_xy(df: pd.DataFrame):
    return df.drop(columns=COLUMNAS_EXCLUIDAS), df["label"]


def dividir_entrenamiento_validacion(df, val_size=0.2, tamano_bloque=500, seed=42):
    """Divide `df` (en el orden original) en entrenamiento y validación por bloques."""
    bloque = np.arange(len(df)) // tamano_bloque
    ids = np.unique(bloque)
    rng = np.random.default_rng(seed)
    en_val = rng.choice(ids, size=int(round(len(ids) * val_size)), replace=False)
    mascara = np.isin(bloque, en_val)
    return df[~mascara], df[mascara]


def pliegues_por_bloques(n_filas, n_pliegues=5, tamano_bloque=500, seed=42):
    """Validación cruzada que conserva el orden: bloques contiguos completos por pliegue.

    Devuelve una lista de (índices_entrenamiento, índices_validación) sobre posiciones.
    """
    from sklearn.model_selection import KFold

    bloque = np.arange(n_filas) // tamano_bloque
    ids = np.unique(bloque)
    pliegues = []
    for tr_b, val_b in KFold(n_pliegues, shuffle=True, random_state=seed).split(ids):
        en_val = np.isin(bloque, ids[val_b])
        pliegues.append((np.where(~en_val)[0], np.where(en_val)[0]))
    return pliegues


def pliegues_por_grupos(X, y, n_pliegues=5, seed=42):
    """Validación cruzada en la que las filas con el mismo vector de entrada van al mismo pliegue.

    Estratificada por clase. Devuelve (índices_entrenamiento, índices_validación) por posición.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    grupos = X.groupby(list(X.columns), sort=False).ngroup().values
    cv = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=seed)
    return [(tr, val) for tr, val in cv.split(X, y, grupos)]
