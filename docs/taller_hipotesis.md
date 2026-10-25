# Taller evaluable: hipótesis de las cuatro configuraciones (escritas antes de entrenar)

Hipótesis dictadas por el usuario (tomadas de `reports/experimentos.md`; los cambios de redacción los indicó el usuario). Partición fija 70/30 (semilla 42, grupos de entradas idénticas sin repartir), mismo preprocesamiento ajustado solo con entrenamiento, ataque = clase positiva.

## Línea base (referencia de C1)
Es el `DummyClassifier(strategy="most_frequent")` de `src/baseline.py`: ignora todas las variables y predice siempre la clase más frecuente del entrenamiento, que es **ataque** (68 % de las filas). Por construcción tiene recall 1,00 (no pierde ningún ataque), tasa de falsos positivos 1,00 (alarma todo el tráfico normal) y macro-F1 bajo.
Evidencia de que esas son sus métricas: ejecución `baseline-clase-dominante` en MLflow (`mlflow.db`, experimento `unsw-nb15`; run `ad077fbe0e4b4fd0841a560cc2a94bf6`), validación de la partición anterior (bloques 80/20): recall 1,0, precisión 0,7095, tasa de falsos positivos 1,0, macro-F1 0,415, clase predicha 1. El valor 0,415 de la hipótesis viene de esa partición; en la partición actual se vuelve a medir y se reporta en `evidencias/` (ejecución `taller-C0-linea-base`).

## Criterio de evaluación
- C1: se busca maximizar el recall (un ataque perdido cuesta más que una falsa alarma); se reporta con umbral 0,5.
- C2 y C3: **maximizar la precisión con recall de al menos 0,95**. Por ahora todas las comparaciones usan umbral 0,5; la elección del umbral es el análisis C4.
- Se descartó la configuración «XGBoost más profundo» (a petición del usuario): su hipótesis y su ejecución se eliminaron; en la corrida solo se vio que difería de C2 en menos de 0,001 de precisión.

## Hipótesis
**C1 — Regresión logística** (variables numéricas estandarizadas y categóricas codificadas)
Con todas las variables numéricas estandarizadas y las categóricas codificadas, la regresión logística supera con claridad a la línea base (macro-F1 0,415) y, sobre todo, deja de alarmar todo el tráfico normal (tasa de falsos positivos muy por debajo de 1,00).

**C2 — XGBoost con hiperparámetros por defecto (300 árboles, profundidad 6, tasa 0,1) — v1**
Al enfocarse en la precisión, XGBoost lo hará mejor que la regresión logística gracias a su mayor complejidad. La interpretabilidad no es prioritaria en este problema, así que se aceptan modelos más complejos siempre que funcionen.

**C3 — XGBoost con búsqueda de Optuna (25 pruebas, solo con la parte de entrenamiento) — v2**
Este espacio de búsqueda permite encontrar una configuración de XGBoost que mantenga un buen recall (≥ 0,95) y reduzca de forma significativa el número de falsos positivos.
(Qué cuenta como "significativa" lo decide el usuario al ver los datos de C3.)

**C4 — Análisis de umbral**
(Pendiente: hipótesis del usuario, antes de ejecutar.)

## v1 frente a v2 (hipótesis del usuario)
Se espera que con respecto al XGBoost base, Optuna logre obtener un mejor desempeño al lograr optimizar los hiperparámetros para el problema presente.
(v1 = C2, XGBoost con hiperparámetros por defecto; v2 = C3, XGBoost con búsqueda de Optuna. Se corrigió "CGBoost" por "XGBoost".)
