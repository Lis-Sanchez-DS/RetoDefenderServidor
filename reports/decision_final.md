# Decisión final

## Modelo y umbral
- **Modelo:** XGBoost (preprocesamiento + clasificador en un solo pipeline de scikit-learn) con los hiperparámetros de la búsqueda con Optuna: 706 árboles, profundidad 10, tasa de aprendizaje 0,035. Registrado como `defender-servidor` (alias `candidato`); archivo `models/cv-grupos-xgboost-optuna.joblib`, versionado con DVC.
- **Umbral:** 0,6 (probabilidad de ataque ≥ 0,6 se marca como ataque).
- **Criterio:** un ataque no detectado cuesta más que una falsa alarma. Por eso el recall es un piso (≥ 0,95 en cada pliegue de validación) y, sujeto a él, se maximiza la precisión.
- **Congelados:** modelo y umbral se fijaron con validación y no se tocaron tras ver la prueba.

## Por qué este modelo (evidencia, no fama)
Matriz completa en `docs/evidencia_modelos.md`.
- **XGBoost frente a regresión logística: evidencia sólida.** Con el mismo preprocesamiento y los mismos pliegues, XGBoost gana en precisión con recall ≥ 0,90 en los 5 pliegues, tanto por bloques (0,985 frente a 0,969) como por grupos (0,992 frente a 0,973). Las falsas alarmas por pliegue bajan de 1.989 a 963.
- **Optuna frente a XGBoost por defecto: evidencia débil.** +0,003 de precisión con recall ≥ 0,95 (0,9797 frente a 0,9770) y 12 % menos falsas alarmas, por debajo del 20 % fijado de antemano. Se conserva porque no es peor y no cuesta más, pero la diferencia es del tamaño del ruido.
- **Umbral 0,6:** con el piso de 0,95, la precisión máxima está cerca de 0,7 (la hipótesis de 0,5 no se cumplió). Se eligió 0,6 porque subir a 0,68 solo gana 0,006 de precisión y deja un margen mínimo sobre el piso.

## Resultado en la prueba (una sola evaluación)
Prueba: 82.327 filas (45.329 ataques, 36.998 normales). Validación: media de 5 pliegues por grupos, mismo umbral.

| Métrica | Validación | Prueba |
|---|---|---|
| Recall | 0,966 | **0,974** |
| Precisión | 0,971 | **0,845** |
| Tasa de falsos positivos | 0,062 | **0,219** |
| Macro-F1 | 0,951 | 0,884 |
| Exactitud | 0,957 | 0,888 |
| MCC | 0,902 | 0,780 |
| ROC-AUC | 0,994 | 0,983 |

Matriz de confusión en la prueba: TN 28.900, FP 8.098, FN 1.166, TP 44.163. El resultado se conserva aunque empeore (guía, sección 3).

## Errores relevantes
- **Falsas alarmas (8.098; tasa 0,219):** casi una de cada cinco filas normales se marca como ataque. Es el error que se aceptó a propósito, por el criterio de costos. Es mayor que en validación (0,062); es coherente con que la prueba incluya una sesión de captura (enero) ausente en el entrenamiento, pero esto no se ha demostrado.
- **Ataques no detectados (1.166, 2,6 %):** el piso de recall de 0,95 acepta perder una pequeña fracción. En validación el recall es 0,966.
- **Fuzzers:** recall 0,827 en la prueba (6.061 ataques); es la debilidad principal y no se vio en validación. Analysis: 0,963. Las demás familias están por encima de 0,98. Esto es una limitación, no una compensación elegida.
- **El recall no bajó en la prueba (subió 0,008):** no implica mejor generalización; la mezcla de ataques cambia (Generic es el 42 % de los ataques de la prueba y se detecta al 100 %). El deterioro se concentra en la precisión.

## Limitaciones del conjunto de datos
- Benchmark histórico (2015), con tráfico generado en laboratorio: practica el ciclo de ML, no prueba eficacia en una red actual.
- Dos sesiones de captura (22 de enero y 18 de febrero de 2015); la partición oficial pone la sesión de enero solo en la prueba. La validación (solo entrenamiento) no refleja ese cambio de condiciones y es optimista.
- Filas casi idénticas por ráfagas de tráfico (`docs/duplicados.md`): se usaron pliegues por grupos de entradas idénticas, pero las filas vecinas parecidas siguen a ambos lados. Son datos reales, no una fuga estricta, pero la validación mide rendimiento "en distribución".
- Se excluyeron 6 filas con estados que aparecen en una sola partición (`no`, `ACC`, `CLO`) y `id`, `label`, `attack_cat` no se usan como entradas.
- Solo se compararon dos familias de modelos (lineal y árboles potenciados); no se puede afirmar que XGBoost sea el mejor posible.
- El umbral 0,6 no está calibrado para las condiciones de la prueba (ROC-AUC 0,983: el modelo ordena bien, pero se pasa en falsas alarmas); no se cambió tras verla.

## Preguntas de la defensa
**¿Qué experimento cambió tu opinión?** El experimento 2 (regresión logística con categóricas): en validación tenía recall 0,988 pero 1.719 falsas alarmas sobre 10.166 normales (17 %). Eso, visible antes de mirar la prueba y registrado en `docs/decisiones.md`, nos llevó a replantear la optimización: recall como piso y precisión como objetivo. La prueba posterior (14.483 falsas alarmas) fue coherente pero no se usó para ajustar. Otros experimentos que cambiaron decisiones: el 4 (pliegues por grupos en lugar de bloques) y el 6 (el umbral óptimo estaba cerca de 0,7, no de 0,5).

**¿Qué errores aceptaste y por qué?** Falsas alarmas, porque una alarma innecesaria cuesta menos que un ataque sin revisar (tasa de 0,219 en la prueba). También aceptamos perder cerca del 3 % de los ataques por el piso de recall de 0,95, y quedó como limitación que Fuzzers solo llegue a 0,827.

**¿Cómo sabes que tu comparación fue justa?** Mismos datos, mismos pliegues (semilla 42), mismo preprocesamiento y mismas métricas para todos los modelos de cada experimento, con las hipótesis escritas antes de entrenar. Límites: los experimentos 3 y 4 usan regímenes de pliegues distintos y no se comparan entre sí; Optuna reutilizó los mismos pliegues (sesgo de selección); la precisión con recall mínimo usa el mejor umbral de cada pliegue (techo optimista); y la prueba no compara limpiamente los dos modelos (la regresión logística se evaluó con otro umbral y otra partición).

**Si perdieras tu entorno local, ¿qué necesitarías para reconstruirlo?** El repositorio de GitHub (código, documentos, punteros `.dvc`), `requirements.lock.txt`, y las credenciales del bucket privado de Cloudflare R2 para hacer `dvc pull` (datos, modelos, predicciones). El historial de MLflow y el estudio de Optuna son locales y se pierden, pero los resultados y mejores parámetros están en `reports/`, y el pipeline reconstruye el historial en unos 5 minutos. Se comprobó en un clon vacío. Pasos en `docs/recuperacion.md`.

## Dónde está cada cosa
Datos y fuente: `docs/datos.md`; decisiones: `docs/decisiones.md`; experimentos 1-6 y prueba: `reports/experimentos.md`; evidencia: `docs/evidencia_modelos.md`; pipeline: `docs/pipeline.md`; recuperación: `docs/recuperacion.md`; predecir: `src/predecir.py`.
