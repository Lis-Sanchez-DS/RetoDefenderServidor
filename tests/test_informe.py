"""El informe PDF debe citar las mismas cifras que los archivos de evidencias y respetar el límite de páginas."""
import json
import re
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")
RUTA = Path("informe/informe.pdf")
pytestmark = pytest.mark.skipif(not RUTA.exists(), reason="el informe se genera con informe/generar_informe.py")


def texto():
    return " ".join(p.extract_text() for p in pypdf.PdfReader(str(RUTA)).pages)


def coma(x, d):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def test_limite_de_paginas_y_sin_rayas_largas():
    paginas = len(pypdf.PdfReader(str(RUTA)).pages)
    assert paginas <= 6  # máximo 5 de contenido más las referencias, que no cuentan
    assert "—" not in texto()  # sin rayas largas (preferencia de estilo de la autora)


def test_cifras_del_informe_coinciden_con_las_evidencias():
    t = texto()
    m = json.load(open("evidencias/metricas-test.json"))
    sel = json.load(open("evidencias/seleccion.json"))
    aud = json.load(open("evidencias/auditoria.json"))
    for k in ("recall", "precision", "tasa_falsos_positivos", "macro_f1"):
        assert coma(m["metricas"][k], 4) in t, k
    for k in ("tn", "fp", "fn", "tp"):
        assert coma(m["matriz_confusion"][k], 0) in t, k
    assert coma(m["filas_evaluadas"], 0) in t
    for v in ("v1", "v2"):
        for k in ("precision", "recall"):
            assert coma(sel[v]["metricas_validacion"][k], 4) in t, (v, k)
    plano = re.sub(r"\s+", "", t)  # los SHA largos se parten entre líneas
    for v in ("v1", "v2"):
        assert sel[v]["huggingface"]["revision_sha"] in plano
        assert sel[v]["run_id"] in plano
    for k in ("entrenamiento", "validacion"):
        assert coma(aud["particion"]["filas"][k], 0) in t
