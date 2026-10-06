"""Pruebas de la carga y la predicción con un modelo guardado."""
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import predecir as p


def _modelo():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1, 0, 1, 0]})
    return Pipeline([("e", StandardScaler()), ("c", LogisticRegression())]).fit(X, [0, 0, 1, 1])


def test_predecir_ignora_columnas_de_respuesta_y_aplica_umbral():
    datos = pd.DataFrame({"id": [1, 2], "a": [0.0, 3.0], "b": [1, 0], "label": [0, 1], "attack_cat": ["Normal", "DoS"]})
    r = p.predecir(_modelo(), datos, umbral=0.5)
    assert list(r.prediccion) == [0, 1]
    assert (p.predecir(_modelo(), datos, umbral=0.0).prediccion == 1).all()  # el umbral manda


def test_cargar_modelo_rechaza_archivo_que_no_coincide_con_dvc(tmp_path):
    archivo = tmp_path / "m.joblib"
    joblib.dump(_modelo(), archivo)
    (tmp_path / "m.joblib.dvc").write_text("outs:\n- md5: 0000\n  path: m.joblib\n")
    with pytest.raises(ValueError, match="no coincide"):
        p.cargar_modelo(str(archivo))
