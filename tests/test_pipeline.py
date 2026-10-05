"""Pruebas de las etapas del pipeline que fallan con un mensaje claro."""
import pandas as pd
import pytest

from src.pipeline import flow


def test_preparar_datos_falla_si_no_coincide_con_dvc(tmp_path, monkeypatch):
    ruta = tmp_path / "{}.csv"
    (tmp_path / "training.csv").write_text("a,b\n1,2\n")
    (tmp_path / "training.csv.dvc").write_text("outs:\n- md5: 0000\n  path: training.csv\n")
    monkeypatch.setattr(flow, "RUTA", str(ruta))
    with pytest.raises(flow.EtapaFallida, match="no coincide con la versión de DVC"):
        flow.preparar_datos.fn({"seed": 42, "pipeline": {"n_pliegues": 5}}, None)


def test_preparar_datos_sugiere_dvc_pull_si_falta_el_archivo(tmp_path, monkeypatch):
    (tmp_path / "training.csv.dvc").write_text("outs:\n- md5: 0000\n  path: training.csv\n")
    monkeypatch.setattr(flow, "RUTA", str(tmp_path / "{}.csv"))
    with pytest.raises(flow.EtapaFallida, match="dvc pull"):
        flow.preparar_datos.fn({"seed": 42, "pipeline": {"n_pliegues": 5}}, None)


def test_comparar_falla_si_ningun_modelo_cumple_el_piso(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    # predicciones fuera de muestra con recall 0,5 en cada pliegue
    oof = pd.DataFrame({"y": [1, 1, 0, 0] * 5, "prob": [0.9, 0.1, 0.2, 0.1] * 5, "pliegue": [0] * 10 + [1] * 10})
    oof.to_csv("data/processed/oof_m.csv", index=False)
    params = {"pipeline": {"piso_recall": 0.95, "umbral": 0.6}}
    with pytest.raises(flow.EtapaFallida, match="ningún modelo"):
        flow.comparar_experimentos.fn([("m", {"run_id": "x", "precision_con_recall_095_media": 0.9})], params)
