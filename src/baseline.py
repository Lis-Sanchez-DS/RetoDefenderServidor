"""Línea base trivial: predice siempre la clase dominante del entrenamiento.

Solo se evalúa en validación; el conjunto de prueba no se toca.
"""
import mlflow
import mlflow.sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.datos import cargar, dividir_entrenamiento_validacion, separar_xy


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


def main():
    params = yaml.safe_load(open("params.yaml"))
    entrenamiento, validacion = dividir_entrenamiento_validacion(
        cargar("training"), params["split"]["val_size"], seed=params["seed"]
    )
    X_tr, y_tr = separar_xy(entrenamiento)
    X_val, y_val = separar_xy(validacion)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("unsw-nb15")
    with mlflow.start_run(run_name="baseline-clase-dominante"):
        mlflow.set_tags({
            "hipotesis": "Predecir siempre la clase dominante da accuracy alta "
                         "pero recall perfecto con 100% de falsas alarmas y macro-F1 bajo.",
            "particion": "validacion (bloques de 500 filas del entrenamiento oficial)",
        })
        modelo = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        mlflow.log_params({
            "modelo": "DummyClassifier(most_frequent)",
            "seed": params["seed"],
            "val_size": params["split"]["val_size"],
            "filas_entrenamiento": len(X_tr),
            "filas_validacion": len(X_val),
            "clase_predicha": int(modelo.classes_[modelo.class_prior_.argmax()]),
            "ataques_en_entrenamiento": round(float(y_tr.mean()), 4),
            "ataques_en_validacion": round(float(y_val.mean()), 4),
            "columnas_excluidas": "id,label,attack_cat",
        })
        res = metricas(y_val, modelo.predict(X_val))
        mlflow.log_metrics({k: v for k, v in res.items() if k not in ("tn", "fp", "fn", "tp")})
        mlflow.sklearn.log_model(modelo, name="modelo")
    print(res)


if __name__ == "__main__":
    main()
