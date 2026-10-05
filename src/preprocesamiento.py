"""Transformaciones de entrada.

- Numéricas: estandarización (necesaria para la regresión logística).
- `service`, `state`: one-hot; los valores nuevos se codifican con ceros.
- `proto`: los valores con menos del 1 % de las filas de entrenamiento (la frecuencia
  se calcula al ajustar, solo con entrenamiento) se agrupan en una categoría
  "infrecuente" y luego se aplica one-hot; los valores nuevos en prueba van a esa categoría.

`incluir_categoricas=False` descarta las tres columnas categóricas.
El transformador se guarda dentro del pipeline del modelo.
"""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAS = ["service", "state"]
FRECUENCIA_MINIMA_PROTO = 0.01
TODAS_CATEGORICAS = CATEGORICAS + ["proto"]


def construir_preprocesamiento(incluir_categoricas: bool = True, escalar: bool = True) -> ColumnTransformer:
    numericas = make_column_selector(dtype_include="number")
    pasos = [("numericas", StandardScaler() if escalar else "passthrough", numericas)]
    if incluir_categoricas:
        pasos += [
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("onehot_proto", OneHotEncoder(min_frequency=FRECUENCIA_MINIMA_PROTO,
                                           handle_unknown="infrequent_if_exist",
                                           sparse_output=False), ["proto"]),
        ]
    return ColumnTransformer(pasos, remainder="drop", verbose_feature_names_out=False)
