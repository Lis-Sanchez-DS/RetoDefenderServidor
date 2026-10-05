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

## Experimento 4: regresión logística y XGBoost con pliegues conscientes de grupos
- Contexto: ver `docs/duplicados.md`. Se repite el experimento 3 con 5 pliegues en los que las filas con el mismo vector de entrada van al mismo pliegue. Mismas variables, mismo preprocesamiento, mismos modelos y mismos hiperparámetros.
- Hipótesis (formulada por el asistente antes de entrenar): al quitar la fuga por copias, el recall y el macro-F1 bajarán respecto a los pliegues por bloques, sobre todo el recall de los ataques con muchas copias; las falsas alarmas cambiarán poco, porque casi ninguna fila normal tiene gemela; y XGBoost seguirá por encima de la regresión logística en la precisión con recall ≥ 0,90, aunque la diferencia puede reducirse.
- Criterio: igual que el experimento 3 (mayor precisión con recall ≥ 0,90; se vigilan macro-F1, tasa de falsos positivos y ROC-AUC). El umbral aún no se elige.

### Resultados del experimento 4 (5 pliegues conscientes de grupos, media ± desviación entre pliegues)

| Métrica | Reg. logística (grupos) | XGBoost (grupos) | Reg. logística (bloques, exp. 3) | XGBoost (bloques, exp. 3) |
|---|---|---|---|---|
| Precisión máx. con recall ≥ 0,90 | 0,973 ± 0,002 | **0,992 ± 0,001** | 0,969 ± 0,008 | 0,985 ± 0,008 |
| Precisión (umbral 0,5) | 0,922 ± 0,002 | **0,960 ± 0,002** | 0,920 ± 0,014 | 0,946 ± 0,017 |
| Recall (umbral 0,5) | **0,988 ± 0,001** | 0,976 ± 0,001 | 0,988 ± 0,002 | 0,977 ± 0,003 |
| Tasa de falsos positivos | 0,178 ± 0,004 | **0,086 ± 0,003** | 0,185 ± 0,031 | 0,119 ± 0,027 |
| Macro-F1 | 0,922 ± 0,001 | **0,949 ± 0,002** | 0,919 ± 0,010 | 0,936 ± 0,012 |
| ROC-AUC | 0,984 ± 0,001 | **0,993 ± 0,000** | 0,982 ± 0,003 | 0,990 ± 0,003 |
| Falsas alarmas por pliegue | 1.989 | **963** | 2.033 | 1.319 |
| Ataques no detectados por pliegue | **282** | 576 | 295 | 559 |

Conclusiones:
- La hipótesis del experimento 4 no se cumple: al separar las copias exactas, el recall no baja (0,988 y 0,976, igual que con bloques) y el macro-F1 y las falsas alarmas mejoran un poco, sobre todo en XGBoost (tasa de falsos positivos 0,119 → 0,086). Sí se cumple que XGBoost sigue por encima: precisión con recall ≥ 0,90 de 0,992 frente a 0,973, en los 5 pliegues.
- Las desviaciones entre pliegues caen mucho (por ejemplo 0,017 → 0,002 en la precisión de XGBoost). Una causa probable es que los pliegues estratificados tienen todos la misma mezcla de clases, mientras que los de bloques tenían mezclas distintas.
- No se debe concluir que las copias no importan: quitar las copias exactas entre pliegues no elimina las filas vecinas parecidas (no idénticas), y en esta partición esas filas pueden estar en entrenamiento. Las filas parecidas no son una fuga en sentido estricto: ver la sección "Lectura revisada" de `docs/duplicados.md`. Lo más probable es que la mejora en falsas alarmas venga de que las ráfagas de tráfico normal ya no quedan completas fuera del entrenamiento, como ocurría con los bloques; no se ha comprobado. Por eso estos pliegues no son necesariamente más fiables que los de bloques: eliminan un tipo de fuga y dejan otro.
- Las cifras de validación siguen siendo optimistas respecto a la prueba (la regresión logística tuvo macro-F1 0,795 en prueba frente a 0,927 en validación).
- Trazabilidad: ejecuciones `cv-grupos-regresion-logistica` y `cv-grupos-xgboost` en MLflow (modelos registrados con esos nombres, versión 1; etiqueta `regimen_pliegues=grupos`), código `.venv/bin/python -m src.validacion_cruzada --grupos`, modelos en `models/cv-grupos-*.joblib` y predicciones fuera de muestra en `data/processed/oof_cv-grupos-*.csv`, versionados con DVC.

## Experimento 5: búsqueda de hiperparámetros de XGBoost con Optuna (optimización bayesiana)
- Hipótesis (definida por el equipo): este espacio de búsqueda permite encontrar una configuración de XGBoost que mantenga un buen recall (≥ 0,95) y reduzca de forma significativa el número de falsos positivos.
- Método: Optuna con el muestreador TPE (semilla 42), 50 pruebas y poda temprana (`MedianPruner`: una prueba se detiene si, tras entrenar algunos pliegues, va por debajo de la mediana de las pruebas anteriores). Los 5 pliegues conscientes de grupos del experimento 4. El conjunto de prueba no se usa.
- Objetivo: maximizar la precisión media (sobre los pliegues) con recall ≥ 0,95. Con el recall fijado en el piso, maximizar la precisión equivale a minimizar las falsas alarmas.
- Espacio de búsqueda: `n_estimators` 100–800; `max_depth` 3–10; `learning_rate` 0,01–0,3 (escala log); `min_child_weight` 1–20; `subsample` 0,5–1,0; `colsample_bytree` 0,5–1,0; `reg_alpha` 1e-8–10 (log); `reg_lambda` 1e-3–10 (log). Sin ponderación de clases.
- Referencia previa a la búsqueda (XGBoost por defecto del experimento 4, calculada con sus predicciones fuera de muestra con recall ≥ 0,95 en cada pliegue): precisión 0,977 ± 0,002; falsas alarmas 534 por pliegue (± 37), tasa de falsos positivos 0,048 ± 0,003.
- Interpretación de "reducción significativa" (fijada por el asistente antes de ejecutar; ajustable): al menos un 20 % menos de falsas alarmas que la referencia (≤ 427 por pliegue en media) con el recall ≥ 0,95, y que la reducción se dé en los 5 pliegues. Si la mejor configuración queda por debajo de eso, la hipótesis no se cumple.
- Precaución: la precisión con recall ≥ 0,95 usa el mejor umbral de cada pliegue conociendo sus etiquetas (techo optimista); sirve para comparar configuraciones. Probar muchas configuraciones sobre los mismos pliegues puede ajustar el resultado a esa validación: la prueba mostrará si la mejora es real, y el modelo se congelará antes.

### Resultados del experimento 5 (50 pruebas: 46 completas y 4 podadas; 42 min en total, 54 s por prueba completa)

Mejor configuración (prueba 45): `n_estimators` 706, `max_depth` 10, `learning_rate` 0,0347, `min_child_weight` 1, `subsample` 0,81, `colsample_bytree` 0,62, `reg_alpha` 5,8e-6, `reg_lambda` 0,0018 (en `reports/optuna_mejores_parametros.json`).

Con recall ≥ 0,95 en cada pliegue, frente a la referencia (XGBoost por defecto):

| Pliegue | Falsas alarmas, referencia | Falsas alarmas, mejor Optuna | Reducción |
|---|---|---|---|
| 0 | 482 | 406 | 15,8 % |
| 1 | 531 | 496 | 6,6 % |
| 2 | 576 | 553 | 4,0 % |
| 3 | 561 | 484 | 13,7 % |
| 4 | 518 | 410 | 20,8 % |
| Media | 534 | 470 | **12,0 %** |

Precisión media con recall ≥ 0,95: 0,980 frente a 0,977. Al umbral 0,5: recall 0,977, precisión 0,962, tasa de falsos positivos 0,082, macro-F1 0,951 (referencia: 0,976; 0,960; 0,086; 0,949).

Conclusiones:
- Hipótesis: se cumple a medias. El recall ≥ 0,95 se mantiene (es el piso del objetivo) y las falsas alarmas bajan, pero un 12 % de media, por debajo del 20 % fijado como "reducción significativa", y solo en 1 de los 5 pliegues se llega al 20 %. Con el criterio acordado, la hipótesis no se cumple.
- La mejora es pequeña frente al ruido: +0,003 de precisión con una desviación entre pliegues de 0,002. Las 8 mejores pruebas están entre 0,9793 y 0,9797 (la mediana de las completas es 0,9785): el paisaje es plano y elegir "la mejor" de 46 pruebas sobre los mismos pliegues sobrestima algo la mejora real.
- Las mejores pruebas caen en el borde del espacio: 9 de las 10 mejores tienen `max_depth` 9 o 10 (el máximo permitido) y `min_child_weight` bajo (1 a 4). Es posible que modelos más profundos mejoren algo más; no se probó.
- Trazabilidad: ejecución padre `optuna-xgboost-busqueda` y 50 ejecuciones anidadas `optuna-prueba-N` en MLflow; el estudio completo queda en `optuna.db` (local, no versionado); modelo reajustado con la mejor configuración y registrado como `cv-grupos-xgboost-optuna` (modelo en `models/` y predicciones fuera de muestra en `data/processed/`, versionados con DVC). No se usó la prueba.

## Experimento 6: elección del umbral de decisión (hipótesis escrita antes de ejecutar)

Modelo: la mejor configuración de Optuna (`cv-grupos-xgboost-optuna`). Datos: solo las predicciones fuera de muestra de los 5 pliegues por grupos (`data/processed/oof_cv-grupos-xgboost-optuna.csv`). La prueba no se usa.

**Hipótesis (del usuario):** como el objetivo es maximizar la precisión con un recall alto fijado (≥ 0,95), el umbral óptimo estará cerca de 0,5. Además, al elegir se favorecerán umbrales más bajos, porque se prefieren falsas alarmas a ataques sin revisar.

Operacionalización (fijada por el asistente antes de ejecutar; ajustable):
- Se barren umbrales de 0,05 a 0,95 en pasos de 0,01 y se calculan recall, precisión y falsas alarmas por pliegue y su media.
- Un umbral es admisible si el recall medio es ≥ 0,95 y el recall de cada pliegue es ≥ 0,95.
- "Cerca de 0,5" significa que el umbral elegido está entre 0,40 y 0,60.
- Regla de elección con sesgo hacia umbrales bajos: entre los admisibles, se toma el umbral más bajo cuya precisión media esté a menos de 0,002 de la máxima (0,002 es la desviación entre pliegues observada en el experimento 5, es decir, diferencias dentro del ruido).
- Precaución: el umbral se elige con las mismas predicciones fuera de muestra con que se eligió la configuración; el modelo final se reajusta con todo el entrenamiento y sus probabilidades pueden diferir algo de las de los modelos de pliegue (entrenados con el 80 %). La prueba mostrará si el umbral se sostiene.

### Resultados del experimento 6 (predicciones fuera de muestra, 5 pliegues, media por pliegue)

| Umbral | Recall | Recall mínimo (pliegue) | Precisión | Falsas alarmas por pliegue |
|---|---|---|---|---|
| 0,30 | 0,9915 | 0,9906 | 0,9419 | 1459 |
| 0,40 | 0,9850 | 0,9838 | 0,9530 | 1160 |
| 0,50 | 0,9769 | 0,9745 | 0,9619 | 923 |
| 0,60 | 0,9663 | 0,9636 | 0,9707 | 696 |
| 0,68 | 0,9550 | 0,9534 | 0,9770 | 536 |
| 0,70 | 0,9516 | 0,9501 | 0,9789 | 490 |

- La precisión sube de forma continua con el umbral y el recall baja; el piso de 0,95 es lo que frena la subida. El umbral admisible más alto es 0,70 (recall mínimo por pliegue 0,9501, al borde del piso) y es también el de máxima precisión (0,9789).
- Con la regla fijada (el más bajo dentro de 0,002 de la precisión máxima): **umbral 0,68**, precisión 0,977, recall 0,955.
- Hipótesis: **no se cumple.** El umbral óptimo con el piso de 0,95 no está cerca de 0,5 (fuera del rango 0,40–0,60), sino cerca de 0,7. Con 0,5 el recall es 0,977, es decir, mucho más alto que el piso, y se pagan unas 390 falsas alarmas más por pliegue (923 frente a 536) que con 0,68. El umbral 0,5 solo sería el óptimo con un piso de recall de ~0,977.
- El sesgo hacia umbrales bajos es una decisión de costos, no estadística: cada paso hacia abajo compra recall con falsas alarmas (de 0,68 a 0,50: +2,2 puntos de recall a cambio de +387 falsas alarmas por pliegue).
- Cautela: 0,68 y 0,70 dejan muy poco margen sobre el piso (recall mínimo 0,953 y 0,950), y el modelo final reajustado con todo el entrenamiento puede dar probabilidades algo distintas. No se usó la prueba.
