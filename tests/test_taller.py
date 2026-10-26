"""Pruebas del código del taller evaluable: datos, partición, métricas y coherencia de las evidencias."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datos import COLUMNAS_EXCLUIDAS, pliegues_por_grupos, separar_xy
from src.metricas import metricas, punto_con_recall_minimo
from src.particion import SALIDA, grupos_de_entradas, huella, row_id
from src.preprocesamiento import construir_preprocesamiento


def test_separar_xy_quita_id_y_respuesta():
    df = pd.DataFrame({"id": [1], "x": [2], "label": [0], "attack_cat": ["Normal"]})
    X, y = separar_xy(df)
    assert not set(COLUMNAS_EXCLUIDAS) & set(X.columns)
    assert list(y) == [0]


def test_preprocesamiento_agrupa_proto_infrecuente():
    protos = ["tcp"] * 120 + ["udp"] * 78 + ["raro1", "raro2"]  # cada raro: 0,5 %
    df = pd.DataFrame({"service": ["-"] * 200, "state": ["INT"] * 200, "proto": protos, "dur": 1.0})
    pp = construir_preprocesamiento(["dur"], escalar=False).fit(df)
    nombres = list(pp.get_feature_names_out())
    assert "proto_infrequent_sklearn" in nombres and "proto_raro1" not in nombres
    nuevo = pd.DataFrame({"service": ["-"], "state": ["ACC"], "proto": ["otro_nuevo"], "dur": [1.0]})  # estado no visto
    fila = dict(zip(nombres, pp.transform(nuevo)[0]))
    assert fila["proto_infrequent_sklearn"] == 1.0


def test_grupos_de_entradas_ignora_id_y_etiquetas():
    df = pd.DataFrame({"id": [1, 2, 3], "dur": [1.0, 1.0, 2.0], "label": [0, 1, 0], "attack_cat": ["a", "b", "a"]})
    g = grupos_de_entradas(df)
    assert g[0] == g[1] != g[2]  # mismas entradas con etiquetas contradictorias: mismo grupo


def test_pliegues_por_grupos_no_reparten_grupos():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 50, 2000), "b": rng.integers(0, 3, 2000)})
    y = pd.Series(rng.integers(0, 2, 2000))
    g = X.groupby(["a", "b"]).ngroup().values
    for tr, val in pliegues_por_grupos(X, y, n_pliegues=5):
        assert set(g[tr]).isdisjoint(g[val])


def test_metricas_y_denominador_cero():
    r = metricas([0, 0, 1, 1], [0, 1, 1, 1])
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 0, 2)
    assert r["tasa_falsos_positivos"] == 0.5 and r["recall"] == 1.0
    assert metricas([0, 0], [0, 0])["precision"] == 0.0  # sin predicciones positivas: zero_division=0


def test_punto_con_recall_minimo_respeta_el_piso():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.concatenate([np.linspace(0, 0.6, 50), np.linspace(0.4, 1, 50)])
    p = punto_con_recall_minimo(y, prob, 0.95)
    assert metricas(y, (prob >= p["umbral"]).astype(int))["recall"] >= 0.95


def test_row_id_identifica_archivo_y_fila():
    assert list(row_id("testing", [7, 9])) == ["testing:7", "testing:9"]
    assert len(huella(pd.DataFrame({"row_id": ["training:1", "training:2"], "conjunto": ["entrenamiento", "validacion"]}))) == 2


@pytest.mark.skipif(not Path(SALIDA).exists(), reason="la partición se recupera con dvc pull")
def test_particion_guardada_no_reparte_grupos():
    p = pd.read_csv(SALIDA)
    assert p.row_id.is_unique
    ent, val = p[p.conjunto == "entrenamiento"], p[p.conjunto == "validacion"]
    assert set(ent.grupo).isdisjoint(val.grupo)
    assert abs(len(val) / len(p) - 0.3) < 0.01


@pytest.mark.skipif(not Path("evidencias/predicciones-test.csv").exists(), reason="sin evidencias de prueba")
def test_metricas_de_prueba_se_recalculan_desde_las_predicciones():
    p = pd.read_csv("evidencias/predicciones-test.csv")
    m = json.load(open("evidencias/metricas-test.json"))
    assert len(p) == m["filas_evaluadas"] == 82332 and p.row_id.is_unique
    assert (p.y_pred == (p.score_attack >= m["umbral"]).astype(int)).all()
    r = metricas(p.y_true.values, p.y_pred.values)
    assert {k: int(r[k]) for k in ("tn", "fp", "fn", "tp")} == m["matriz_confusion"]
    assert r["recall"] == pytest.approx(m["metricas"]["recall"])


@pytest.mark.skipif(not Path("evidencias/seleccion.json").exists(), reason="sin selección")
def test_seleccion_hashes_coinciden_con_los_modelos():
    import hashlib
    s = json.load(open("evidencias/seleccion.json"))
    for v in ("v1", "v2"):
        ruta = Path(s[v]["archivo_modelo"])
        if ruta.exists():
            assert hashlib.sha256(ruta.read_bytes()).hexdigest() == s[v]["modelo_sha256"]


@pytest.mark.skipif(not Path(SALIDA).exists(), reason="la partición se recupera con dvc pull")
def test_auditoria_describe_la_particion_vigente():
    import hashlib
    a = json.load(open("evidencias/auditoria.json"))["particion"]
    p = pd.read_csv(SALIDA)
    assert a["sha256_archivo"] == hashlib.sha256(Path(SALIDA).read_bytes()).hexdigest()
    assert a["filas"] == {k: int(n) for k, n in p.conjunto.value_counts().items()}
