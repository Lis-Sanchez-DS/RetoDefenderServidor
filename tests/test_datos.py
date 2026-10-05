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


def test_preprocesamiento_agrupa_proto_infrecuente():
    from src.preprocesamiento import construir_preprocesamiento

    protos = ["tcp"] * 120 + ["udp"] * 78 + ["raro1", "raro2"]  # cada raro: 0,5 %
    df = pd.DataFrame({"service": ["-"] * 200, "state": ["INT"] * 200, "proto": protos, "dur": 1.0})
    pp = construir_preprocesamiento().fit(df)
    nombres = list(pp.get_feature_names_out())
    assert "proto_infrequent_sklearn" in nombres
    assert "proto_raro1" not in nombres
    nuevo = pd.DataFrame({"service": ["-"], "state": ["INT"], "proto": ["otro_nuevo"], "dur": [1.0]})
    fila = dict(zip(nombres, pp.transform(nuevo)[0]))
    assert fila["proto_infrequent_sklearn"] == 1.0
