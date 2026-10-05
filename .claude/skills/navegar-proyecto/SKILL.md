---
name: navegar-proyecto
description: Guía para orientarse en el proyecto Defiende el servidor (UNSW-NB15, MLOps con Git, DVC, MLflow y Prefect). Úsala al buscar dónde va o dónde está algo, cómo ejecutar el entorno, o qué falta por hacer.
---

# Navegar el proyecto "Defiende el servidor"

Objetivo: distinguir tráfico normal de ataques en UNSW-NB15 con una solución reproducible. Todo el contenido del repositorio (README, comentarios, docs) va en español.

## Mapa del repositorio
| Ruta | Contenido |
|---|---|
| `src/pipeline/flow.py` | Flujo de Prefect: preparar → entrenar → comparar → registrar |
| `params.yaml` | Configuración central (semilla, particiones, versión de datos) |
| `data/raw/`, `data/processed/` | Datos; los versiona DVC, no Git |
| `models/` | Modelos guardados; los versiona DVC |
| `reports/` | Tabla de experimentos y decisión final |
| `notebooks/` | Exploración |
| `tests/` | Pruebas con pytest |
| `.dvc/` | Configuración de DVC |
| `requirements.txt` / `requirements.lock.txt` | Dependencias y versiones exactas |
| `guia-conceptual-defiende-servidor-2026II.pdf` | Enunciado del reto (ignorado por Git, solo local) |

## Comandos
Usar siempre el entorno local `.venv`:
```bash
.venv/bin/python -m pytest -q          # pruebas
.venv/bin/python -m src.pipeline.flow  # ejecutar el pipeline de Prefect
.venv/bin/dvc status                   # estado de datos y modelos
.venv/bin/mlflow ui                    # interfaz de experimentos
```

## Reglas del proyecto
- Datos y modelos nunca van a Git: se versionan con DVC (`dvc add`).
- Experimentos: mínimo 4 configuraciones con 2 tipos de modelo; escribir la hipótesis antes de entrenar, mantener las mismas particiones y cambiar una decisión a la vez. Registrar todo en MLflow.
- Evaluación: recall, precision, tasa de falsos positivos y macro-F1. Elegir modelo y umbral con validación; evaluar una sola vez en test y conservar el resultado aunque empeore.
- El modelo guardado incluye las transformaciones de entrada (pipeline completo) y debe poder cargarse para predecir.
- El pipeline debe mostrar si una etapa falla y por qué.
- Evitar fuga de información: duplicados entre particiones y columnas de la respuesta (etiquetas) fuera de las variables de entrada; revisar si el identificador tiene significado.

## Entregables (ver el PDF)
Repositorio reproducible, comparación de experimentos, decisión final breve (modelo, umbral, errores, resultado en test, limitaciones), y demostración (Prefect, cargar modelo, predecir, recuperar datos y modelo con DVC).

## Estado actual
Solo la estructura base. Aún sin datos descargados ni modelos entrenados. Actualiza esta sección cuando avance el trabajo.
