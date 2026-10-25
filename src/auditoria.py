"""Genera evidencias/auditoria.json: fuentes, huellas, calidad, duplicados, cruces y particiones.

No usa etiquetas ni métricas de la prueba para decidir nada: solo audita esquema,
duplicados y cruces de entradas. Uso: .venv/bin/python -m src.auditoria
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.datos import COLUMNAS_EXCLUIDAS, RUTA
from src.particion import ESTADOS_EXCLUIDOS_ENTRENAMIENTO, SALIDA, huella

FUENTE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
FECHA_DESCARGA = "2026-10-05"


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def clave(df):
    """Huella de 64 bits del vector de entrada de cada fila (sin id ni etiquetas)."""
    return pd.util.hash_pandas_object(df.drop(columns=COLUMNAS_EXCLUIDAS), index=False)


def describir(nombre, df):
    k = clave(df)
    por_grupo = df.assign(k=k).groupby("k").label.agg(["size", "nunique"])
    return {
        "archivo": f"UNSW_NB15_{nombre}-set.csv", "sha256": sha256(RUTA.format(nombre)),
        "filas": len(df), "columnas": df.shape[1],
        "clases_label": {str(c): int(n) for c, n in df.label.value_counts().sort_index().items()},
        "clases_attack_cat": {str(c): int(n) for c, n in df.attack_cat.value_counts().items()},
        "faltantes_total": int(df.isna().sum().sum()),
        "faltantes_por_columna": {c: int(n) for c, n in df.isna().sum().items() if n},
        "tipos": {str(t): int(n) for t, n in df.dtypes.astype(str).value_counts().items()},
        "duplicados_por_entradas": {
            "vectores_distintos": int(len(por_grupo)),
            "filas_copia_de_otra": int(len(df) - len(por_grupo)),
            "grupos_con_mas_de_una_fila": int((por_grupo["size"] > 1).sum()),
            "grupos_con_etiquetas_contradictorias": int((por_grupo["nunique"] > 1).sum()),
            "filas_en_grupos_contradictorios": int(por_grupo.loc[por_grupo["nunique"] > 1, "size"].sum()),
        },
    }


def main():
    seed = yaml.safe_load(open("params.yaml"))["seed"]
    tr, te = pd.read_csv(RUTA.format("training")), pd.read_csv(RUTA.format("testing"))
    assert tr.shape[1] == te.shape[1] and list(tr.columns) == list(te.columns)
    assert len(tr) == 175341 and len(te) == 82332, "las particiones no son las publicadas"
    p = pd.read_csv(SALIDA)
    tr = tr.assign(k=clave(tr))
    te = te.assign(k=clave(te))
    k_tr = set(tr.k)
    en_val = set(tr.loc[tr.id.isin(p.loc[p.conjunto == "validacion", "id"]), "k"])
    en_ent = set(tr.loc[tr.id.isin(p.loc[p.conjunto == "entrenamiento", "id"]), "k"])
    contra_te = te.k.isin(k_tr)
    # etiqueta contradictoria entre archivos (misma entrada, etiqueta distinta)
    et_tr = tr.groupby("k").label.agg(lambda s: set(s))
    et_te = te[contra_te].groupby("k").label.agg(lambda s: set(s))
    contradictorios_cruce = int(sum(1 for k, s in et_te.items() if s != et_tr[k] or len(s) > 1))
    entradas = [c for c in tr.columns if c not in COLUMNAS_EXCLUIDAS + ["k"]]
    auditoria = {
        "fuente": {"url": FUENTE, "fecha_descarga": FECHA_DESCARGA,
                   "nota": "descarga manual de la carpeta oficial; huellas coinciden con las del anexo del taller (mirror)"},
        "archivos": {"training": describir("training", tr.drop(columns="k")),
                     "testing": describir("testing", te.drop(columns="k"))},
        "variables": {"usadas_como_entrada": entradas, "n_entradas": len(entradas),
                      "excluidas": COLUMNAS_EXCLUIDAS,
                      "motivo": {"label": "respuesta", "attack_cat": "revela la respuesta",
                                 "id": "identifica filas; solo para auditoría y relacionar predicciones"}},
        "politica_de_limpieza": {
            "duplicados": "no se eliminan; las filas con el mismo vector de entrada forman un grupo y se asignan juntas a entrenamiento o validación",
            "contradicciones": "grupos con etiquetas distintas se conservan (error irreducible) y van juntos a un mismo conjunto",
            "estados_excluidos_en_entrenamiento": ESTADOS_EXCLUIDOS_ENTRENAMIENTO,
            "filas_excluidas_en_entrenamiento": int(tr.state.isin(ESTADOS_EXCLUIDOS_ENTRENAMIENTO).sum()),
            "test": "no se elimina ninguna fila; los estados no vistos se codifican con ceros (OneHotEncoder handle_unknown)",
            "imputacion": "no hay faltantes",
        },
        "particion": {
            "semilla": seed, "metodo": "StratifiedGroupKFold(5, shuffle) fold 0 como validación; grupo = vector de entrada idéntico",
            "archivo": str(SALIDA), "sha256_archivo": sha256(SALIDA),
            "filas": {c: int(n) for c, n in p.conjunto.value_counts().items()},
            "ataques_por_conjunto": {c: round(float(m), 4) for c, m in p.groupby("conjunto").label.mean().items()},
            "huellas_row_id": huella(p),
            "grupos_repartidos_entre_entrenamiento_y_validacion": len(en_val & en_ent),
        },
        "cruces_entre_particiones": {
            "filas_test_con_vector_identico_en_entrenamiento_oficial": int(contra_te.sum()),
            "porcentaje_test": round(float(contra_te.mean()) * 100, 2),
            "filas_test_con_vector_identico_en_validacion": int(te.k.isin(en_val).sum()),
            "vectores_test_con_etiqueta_distinta_a_la_de_entrenamiento": contradictorios_cruce,
            "limite": "solapamiento medido solo con entradas; el rendimiento en test es optimista en las filas con gemela en entrenamiento",
        },
    }
    Path("evidencias").mkdir(exist_ok=True)
    json.dump(auditoria, open("evidencias/auditoria.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: auditoria[k] for k in ("politica_de_limpieza", "cruces_entre_particiones")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
