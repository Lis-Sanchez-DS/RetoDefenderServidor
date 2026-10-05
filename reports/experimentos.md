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

## Experimento 3: XGBoost frente a regresión logística, con validación cruzada por bloques y enfoque en precisión
- Hipótesis (definida por el equipo): al enfocarse en la precisión, XGBoost lo hará mejor que la regresión logística gracias a su mayor complejidad. La interpretabilidad no es prioritaria en este problema, así que se aceptan modelos más complejos siempre que funcionen.
- Cambio: tipo de modelo (regresión logística → XGBoost) y régimen de evaluación (5 pliegues por bloques, ver `docs/decisiones.md`). La regresión logística se vuelve a entrenar con los mismos pliegues para que la comparación sea justa. Ambos usan las variables categóricas codificadas.
- Criterio: mayor precisión con recall ≥ 0,90 (en cada pliegue y promedio); además se vigilan el macro-F1, la tasa de falsos positivos y el ROC-AUC. El umbral aún no se elige.
- Configuración de XGBoost: valores moderados sin ajuste (300 árboles, profundidad máxima 6, tasa de aprendizaje 0,1, `hist`, sin ponderar clases), para cambiar una decisión a la vez.

### Resultados del experimento 3 (validación cruzada por bloques, 5 pliegues, media ± desviación entre pliegues)

| Métrica | Regresión logística | XGBoost |
|---|---|---|
| Precisión máxima con recall ≥ 0,90 | 0,969 ± 0,008 | **0,985 ± 0,008** |
| Precisión (umbral 0,5) | 0,920 ± 0,014 | **0,946 ± 0,017** |
| Recall (umbral 0,5) | **0,988 ± 0,002** | 0,977 ± 0,003 |
| Tasa de falsos positivos (umbral 0,5) | 0,185 ± 0,031 | **0,119 ± 0,027** |
| Macro-F1 (umbral 0,5) | 0,919 ± 0,010 | **0,936 ± 0,012** |
| ROC-AUC | 0,982 ± 0,003 | **0,990 ± 0,003** |
| Falsas alarmas por pliegue (umbral 0,5) | 2.033 | **1.319** |
| Ataques no detectados por pliegue (umbral 0,5) | **295** | 559 |

Conclusiones:
- La hipótesis se cumple: XGBoost supera a la regresión logística en el criterio principal (precisión con recall ≥ 0,90) en los 5 pliegues, con una mejora media de 0,016 (de 0,969 a 0,985); es decir, la proporción de alertas incorrectas baja de 3,1 % a 1,5 %. También gana en precisión, falsas alarmas, macro-F1 y ROC-AUC en todos los pliegues.
- Con el umbral 0,5 XGBoost reduce las falsas alarmas un 35 % (2.033 → 1.319), a costa de más ataques no detectados (295 → 559) y un recall algo menor (0,977 frente a 0,988). Ambos quedan por encima del piso de 0,90, y el umbral aún no se ha elegido.
- La diferencia entre modelos (0,016) es del mismo tamaño que la variación entre pliegues (0,008 de desviación), pero el signo es el mismo en los 5 pliegues.

Precauciones:
- "Precisión máxima con recall ≥ 0,90" usa el mejor umbral de cada pliegue conociendo sus etiquetas, así que es un techo optimista del criterio; el umbral real deberá fijarse sin ver el pliegue evaluado (por ejemplo con las predicciones fuera de muestra de los otros pliegues).
- Persisten los duplicados y las filas vecinas parecidas, y la validación no refleja el cambio de sesión de captura que sí tiene la prueba; el resultado en prueba de la regresión logística (macro-F1 0,795 frente a 0,927 en validación) recuerda que estas cifras serán optimistas. XGBoost no se ha evaluado en prueba.
- Solo se probó una configuración de XGBoost (sin ajuste de hiperparámetros ni ponderación de clases).
- Trazabilidad: MLflow, experimento `unsw-nb15`, ejecuciones `cv-regresion-logistica` y `cv-xgboost` (modelos registrados con esos nombres, versión 1); código en `src/validacion_cruzada.py` (`.venv/bin/python -m src.validacion_cruzada`); modelos reajustados con todo el entrenamiento en `models/cv-*.joblib` y predicciones fuera de muestra en `data/processed/oof_cv-*.csv`, versionados con DVC. Las etiquetas `git_commit` de estas ejecuciones marcan `git_cambios_sin_commit=True` porque el código se consolidó en Git después de entrenar. Hay además una ejecución fallida de `cv-xgboost` (error al serializar el modelo con el formato por defecto de MLflow; se corrigió usando `cloudpickle`).
