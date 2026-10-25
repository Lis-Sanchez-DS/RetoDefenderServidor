"""Métricas de clasificación binaria (ataque = clase positiva), compartidas por los scripts del taller.

Denominadores cero: precisión y recall usan zero_division=0; la tasa de falsos positivos es FP/(FP+TN)
(siempre hay filas normales en las particiones usadas).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

PISO_RECALL_NUEVO = 0.95  # piso de recall (ver docs/taller_hipotesis.md)


def metricas(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "tasa_falsos_positivos": fp / (fp + tn),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def punto_con_recall_minimo(y, prob, piso):
    """Mejor punto con recall >= piso: precisión, umbral, falsos positivos y su tasa."""
    precision, recall, umbrales = precision_recall_curve(y, prob)
    ok = recall[:-1] >= piso
    if not ok.any():
        return dict(precision=float("nan"), umbral=float("nan"), fp=float("nan"), fpr=float("nan"))
    i = np.argmax(np.where(ok, precision[:-1], -1))
    tp = recall[i] * float(np.sum(y))
    fp = tp * (1 - precision[i]) / precision[i]
    return dict(precision=float(precision[i]), umbral=float(umbrales[i]), fp=float(fp),
                fpr=float(fp / np.sum(np.asarray(y) == 0)))


def todas_las_metricas(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    res = metricas(y, pred)
    res["mcc"] = matthews_corrcoef(y, pred)
    res["roc_auc"] = roc_auc_score(y, prob)
    res["pr_auc"] = average_precision_score(y, prob)
    res["fp_por_10000"] = res["fp"] / len(y) * 10000
    res["fn_por_10000"] = res["fn"] / len(y) * 10000
    return res
