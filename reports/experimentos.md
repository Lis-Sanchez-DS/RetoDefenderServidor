# Comparación de experimentos

Todos los experimentos usan la misma partición de entrenamiento/validación (bloques de 500 filas, 80/20, semilla 42, sin `id`) y se registran en MLflow (experimento `unsw-nb15`). El conjunto de prueba no se usa. Umbral de decisión: 0,5 (aún no se ha ajustado). Las hipótesis se escribieron antes de entrenar.

## Experimento 1: regresión logística sin variables categóricas
- Hipótesis: con solo las 39 variables numéricas (estandarizadas), la regresión logística supera con claridad a la línea base (macro-F1 0,415) y, sobre todo, deja de alarmar todo el tráfico normal (tasa de falsos positivos muy por debajo de 1,00).
- Cambio: respecto a la línea base, un modelo real con solo variables numéricas.

## Experimento 2: regresión logística con variables categóricas codificadas
- Hipótesis: las variables categóricas (`service`, `state` y `proto`, con one-hot y `proto` con categoría infrecuente) impactan de forma muy marcada los resultados en validación. El objetivo es comprobar si vale la pena conservarlas.
- Cambio: respecto al experimento 1, solo se agregan las tres variables categóricas codificadas.

## Resultados (validación, umbral 0,5, 35.000 filas: 10.166 normales y 24.834 ataques)

| Modelo | Recall | Precisión | Tasa de falsos positivos | Macro-F1 | ROC-AUC | PR-AUC | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|---|---|
| Línea base (siempre "ataque") | 1,000 | 0,710 | 1,000 | 0,415 | - | - | 0 | 10.166 |
| Exp. 1: regresión logística, solo numéricas (39 col.) | 0,987 | 0,927 | 0,190 | 0,918 | 0,970 | 0,983 | 326 | 1.935 |
| Exp. 2: regresión logística, con categóricas (66 col.) | 0,988 | 0,935 | 0,169 | 0,927 | 0,986 | 0,994 | 307 | 1.719 |

## Conclusiones
- Experimento 1: la hipótesis se cumple. Sube el macro-F1 de 0,415 a 0,918 y la tasa de falsos positivos baja de 1,00 a 0,19, con el recall casi intacto (0,987).
- Experimento 2: la hipótesis (impacto muy marcado en validación) no se cumple del todo. Las variables categóricas sí ayudan al modelo a clasificar (macro-F1 +0,009, falsos positivos 1.935 → 1.719, un 11 % menos, falsos negativos 326 → 307, ROC-AUC de 0,970 a 0,986), pero no dominan el resultado en validación: el modelo con solo numéricas ya llega a 0,918 de macro-F1. Decisión: por ahora se conservan, porque ayudan y la codificación funciona, y se revisará con otros modelos y con el ajuste del umbral.
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

## Evaluación en prueba: regresión logística con categóricas
Pedida por el equipo como primera mirada al conjunto de prueba (82.327 filas: 36.998 normales y 45.329 ataques). Modelo `regresion-logistica-con-categoricas` v1, tal como se guardó (verificado contra su hash de DVC), umbral 0,5 sin ajustar. Se ejecutó una vez con `.venv/bin/python -m src.evaluar_test regresion-logistica-con-categoricas`, queda en MLflow (`etapa=evaluacion_test`) y se conserva tal cual.

| | Recall | Precisión | Tasa de falsos positivos | Macro-F1 | ROC-AUC | PR-AUC | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|---|---|
| Validación | 0,988 | 0,935 | 0,169 | 0,927 | 0,986 | 0,994 | 307 | 1.719 |
| Prueba | 0,972 | 0,753 | 0,392 | 0,795 | 0,955 | 0,966 | 1.270 | 14.483 |

- El modelo se degrada en prueba: el macro-F1 baja de 0,927 a 0,795 y la tasa de falsos positivos sube de 0,17 a 0,39 (39 % del tráfico normal se marca como ataque). El recall se mantiene alto (0,972).
- El deterioro coincide con lo esperado por la partición: la prueba incluye datos de otra sesión de captura (22 de enero) que el entrenamiento no tiene, tiene menos ataques proporcionalmente (55 % frente a 68 %) y la validación salía de la misma distribución que el entrenamiento. No se ha comprobado qué parte del deterioro se debe a cada causa.
- Recall por familia de ataque (umbral 0,5): Generic 0,999, Reconnaissance 0,996, Shellcode 0,989, Backdoor 0,988, Worms 0,977, DoS 0,975, Exploits 0,940, Fuzzers 0,935, Analysis 0,914.
- Esta evaluación ya se usó: no deben tomarse decisiones de modelo ni de umbral ajustando sobre este resultado. Solo se evaluó este modelo; el de solo numéricas no se evaluó en prueba.
