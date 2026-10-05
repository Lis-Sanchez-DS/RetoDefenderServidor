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
    pp = construir_preprocesamiento(["dur"], escalar=False).fit(df)
    nombres = list(pp.get_feature_names_out())
    assert "proto_infrequent_sklearn" in nombres
    assert "proto_raro1" not in nombres
    nuevo = pd.DataFrame({"service": ["-"], "state": ["INT"], "proto": ["otro_nuevo"], "dur": [1.0]})
    fila = dict(zip(nombres, pp.transform(nuevo)[0]))
    assert fila["proto_infrequent_sklearn"] == 1.0


def test_preprocesamiento_sin_categoricas():
    from src.preprocesamiento import construir_preprocesamiento

    df = pd.DataFrame({"service": ["dns", "-"], "state": ["INT", "FIN"], "proto": ["tcp", "udp"],
                       "dur": [1.0, 2.0], "sbytes": [10, 20]})
    assert construir_preprocesamiento(["dur", "sbytes"], incluir_categoricas=False).fit_transform(df).shape == (2, 2)


def test_pliegues_por_bloques_conservan_orden_y_cubren_todo():
    import numpy as np
    from src.datos import pliegues_por_bloques

    pliegues = pliegues_por_bloques(5000, n_pliegues=5, tamano_bloque=500)
    todas_val = np.concatenate([v for _, v in pliegues])
    assert sorted(todas_val) == list(range(5000))  # cada fila valida exactamente una vez
    for tr, val in pliegues:
        assert set(tr).isdisjoint(val)
        assert (np.diff(val)[np.diff(val) > 1] > 1).all()  # bloques completos y en orden
        assert np.array_equal(val, np.sort(val))


def test_pliegues_por_grupos_no_separan_copias():
    import numpy as np
    from src.datos import pliegues_por_grupos

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 50, 2000), "b": rng.integers(0, 3, 2000)})
    y = pd.Series(rng.integers(0, 2, 2000))
    pliegues = pliegues_por_grupos(X, y, n_pliegues=5)
    clave = X.astype(str).agg("-".join, axis=1)
    for tr, val in pliegues:
        assert set(clave.iloc[tr]).isdisjoint(set(clave.iloc[val]))
    assert sorted(np.concatenate([v for _, v in pliegues])) == list(range(2000))
