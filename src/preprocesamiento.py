"""Transformaciones de entrada: one-hot de `service`, `state` y `proto`.

`proto` agrupa en una categoría "infrecuente" todos los valores que aparecen en
menos del 1 % de las filas de entrenamiento (la frecuencia se calcula al ajustar,
solo con entrenamiento) y luego aplica one-hot. Los valores nuevos en prueba van
a esa misma categoría. Las demás columnas son numéricas y pasan sin cambios.
El transformador se guarda dentro del pipeline del modelo.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAS = ["service", "state"]
FRECUENCIA_MINIMA_PROTO = 0.01


def construir_preprocesamiento() -> ColumnTransformer:
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
         ("onehot_proto", OneHotEncoder(min_frequency=FRECUENCIA_MINIMA_PROTO,
                                        handle_unknown="infrequent_if_exist",
                                        sparse_output=False), ["proto"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
