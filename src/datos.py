"""Constantes de los datos y utilidades de partición (taller evaluable)."""
RUTA = "data/raw/unsw-nb15/particion_oficial/UNSW_NB15_{}-set.csv"
# `id` solo refleja el orden del archivo; `label` y `attack_cat` son la respuesta.
COLUMNAS_EXCLUIDAS = ["id", "label", "attack_cat"]


def separar_xy(df):
    return df.drop(columns=COLUMNAS_EXCLUIDAS), df["label"]


def pliegues_por_grupos(X, y, n_pliegues=5, seed=42):
    """Validación cruzada en la que las filas con el mismo vector de entrada van al mismo pliegue.

    Estratificada por clase. Devuelve (índices_entrenamiento, índices_validación) por posición.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    grupos = X.groupby(list(X.columns), sort=False).ngroup().values
    cv = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True, random_state=seed)
    return [(tr, val) for tr, val in cv.split(X, y, grupos)]
