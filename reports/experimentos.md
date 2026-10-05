# Comparación de experimentos

Todos los experimentos usan la misma partición de entrenamiento/validación (bloques de 500 filas, 80/20, semilla 42, sin `id`) y se registran en MLflow (experimento `unsw-nb15`). El conjunto de prueba no se usa. Umbral de decisión: 0,5 (aún no se ha ajustado). Las hipótesis se escribieron antes de entrenar.

## Experimento 1: regresión logística sin variables categóricas
- Hipótesis: con solo las 39 variables numéricas (estandarizadas), la regresión logística supera con claridad a la línea base (macro-F1 0,415) y, sobre todo, deja de alarmar todo el tráfico normal (tasa de falsos positivos muy por debajo de 1,00).
- Cambio: respecto a la línea base, un modelo real con solo variables numéricas.

## Experimento 2: regresión logística con variables categóricas codificadas
- Hipótesis: agregar `service`, `state` y `proto` (one-hot, `proto` con categoría infrecuente) mejora el macro-F1 y reduce los falsos positivos respecto al experimento 1, porque estas variables concentran mucha información sobre la clase (p. ej. `state` INT 93 % ataques, `ssh` 0,8 %). Si no mejora, la codificación no está aportando.
- Cambio: respecto al experimento 1, solo se agregan las tres variables categóricas codificadas.

## Resultados (validación, umbral 0,5, 35.000 filas: 10.166 normales y 24.834 ataques)

| Modelo | Recall | Precisión | Tasa de falsos positivos | Macro-F1 | ROC-AUC | PR-AUC | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|---|---|
| Línea base (siempre "ataque") | 1,000 | 0,710 | 1,000 | 0,415 | - | - | 0 | 10.166 |
| Exp. 1: regresión logística, solo numéricas (39 col.) | 0,987 | 0,927 | 0,190 | 0,918 | 0,970 | 0,983 | 326 | 1.935 |
| Exp. 2: regresión logística, con categóricas (66 col.) | 0,988 | 0,935 | 0,169 | 0,927 | 0,986 | 0,994 | 307 | 1.719 |

## Conclusiones
- Experimento 1: la hipótesis se cumple. Sube el macro-F1 de 0,415 a 0,918 y la tasa de falsos positivos baja de 1,00 a 0,19, con el recall casi intacto (0,987).
- Experimento 2: la hipótesis se cumple, pero la mejora es modesta: macro-F1 +0,009, falsos positivos 1.935 → 1.719 (-11 %), falsos negativos 326 → 307, y el ROC-AUC sube de 0,970 a 0,986. La codificación funciona (el modelo la usa y mejora el orden de las probabilidades), aunque con umbral 0,5 se nota poco en las métricas de decisión. La mejora mayor en AUC que en macro-F1 sugiere que el umbral puede ajustarse después.
- Aún con las categóricas, el 17 % del tráfico normal se marca como ataque; con el criterio de que un ataque no detectado cuesta más, es un punto de partida aceptable, pero hay que bajar esa cifra.

## Precauciones
- El macro-F1 de entrenamiento (0,911 y 0,921) es menor que el de validación (0,918 y 0,927): la validación resulta algo más fácil que el entrenamiento. Probable causa: los duplicados (no se han eliminado) y que bloques vecinos son parecidos. No debe extrapolarse a prueba, donde además hay datos de otra sesión de captura.
- Misma partición, mismos datos y mismo umbral en ambos; solo cambian las columnas categóricas.
- Las dos ejecuciones se identifican en MLflow por su `git_commit` y por el hash DVC de los datos.

## Trazabilidad
- MLflow (`sqlite:///mlflow.db`, local): experimento `unsw-nb15`. Modelos registrados: `regresion-logistica-sin-categoricas` v1 y `regresion-logistica-con-categoricas` v1.
- Modelos completos (preprocesamiento + clasificador) en `models/*.joblib`, versionados con DVC (`models/*.joblib.dvc`).
- Reproducir: `.venv/bin/python -m src.experimentos`.
- En MLflow quedan además dos ejecuciones de la línea base (la primera con 175.341 filas, antes de excluir 1 fila con estado `no`; la vigente es la segunda) y una ejecución fallida de la regresión sin categóricas, causada por un error al guardar el modelo (selector de columnas no serializable), ya corregido.
