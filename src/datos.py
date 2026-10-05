"""Carga de datos y partición de entrenamiento/validación.

El archivo de entrenamiento oficial está ordenado en bloques (primero normales,
al final solo ataques), así que ni las primeras filas ni los últimos `id` sirven
como validación. Se parte por bloques contiguos asignados al azar: filas vecinas
(que comparten ráfagas de tráfico) quedan en el mismo lado de la partición.
"""
import numpy as np
import pandas as pd

RUTA = "data/raw/unsw-nb15/particion_oficial/UNSW_NB15_{}-set.csv"
# `id` solo refleja el orden del archivo; `label` y `attack_cat` son la respuesta.
COLUMNAS_EXCLUIDAS = ["id", "label", "attack_cat"]


def cargar(particion: str) -> pd.DataFrame:
    """Lee 'training' o 'testing' de la partición oficial."""
    return pd.read_csv(RUTA.format(particion))


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
