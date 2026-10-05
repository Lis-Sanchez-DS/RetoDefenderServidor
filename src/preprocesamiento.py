"""Transformaciones de entrada: one-hot de `service` y `state`.

`proto` queda fuera por ahora, hasta decidir cómo tratarlo (ver docs/decisiones.md).
Las demás columnas son numéricas y pasan sin cambios. El transformador se ajusta
solo con entrenamiento y se guarda dentro del pipeline del modelo.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAS = ["service", "state"]
EXCLUIDAS_POR_AHORA = ["proto"]


def construir_preprocesamiento() -> ColumnTransformer:
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
         ("descartar", "drop", EXCLUIDAS_POR_AHORA)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
