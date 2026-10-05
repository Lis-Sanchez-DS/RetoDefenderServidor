import pandas as pd

from src.datos import COLUMNAS_EXCLUIDAS, dividir_entrenamiento_validacion, separar_xy


def test_particion_sin_solapamiento_y_completa():
    df = pd.DataFrame({"id": range(5000), "x": range(5000), "label": [0, 1] * 2500, "attack_cat": "a"})
    tr, val = dividir_entrenamiento_validacion(df)
    assert len(tr) + len(val) == len(df)
    assert set(tr.id).isdisjoint(val.id)


def test_separar_xy_quita_id_y_respuesta():
    df = pd.DataFrame({"id": [1], "x": [2], "label": [0], "attack_cat": ["Normal"]})
    X, y = separar_xy(df)
    assert not set(COLUMNAS_EXCLUIDAS) & set(X.columns)
    assert list(y) == [0]


def test_preprocesamiento_onehot_y_sin_proto():
    from src.preprocesamiento import construir_preprocesamiento

    df = pd.DataFrame({"service": ["dns", "-", "http"], "state": ["INT", "FIN", "CON"],
                       "proto": ["tcp", "udp", "tcp"], "dur": [1.0, 2.0, 3.0]})
    Xt = construir_preprocesamiento().fit_transform(df)
    assert Xt.shape == (3, 3 + 3 + 1)  # 3 servicios + 3 estados + dur
