# ¿Elegimos por evidencia o por ser un algoritmo más conocido y complejo?

Pregunta de la guía (sección 4): "¿estás eligiendo por evidencia o porque un algoritmo es más conocido, más complejo o el último que ejecutaste?". Cada fila resume una decisión, la alternativa, la evidencia y qué tan fuerte es. Las cifras salen de `reports/experimentos.md` (validación cruzada, nunca de la prueba, salvo la fila de la prueba).

## Matriz de decisiones

| Decisión | Alternativa | Evidencia | Fuerza | ¿Evidencia o costumbre? |
|---|---|---|---|---|
| Usar un modelo mejor que la línea base | Predecir siempre "ataque" | Macro-F1 0,415 → 0,918; falsos positivos 1,00 → 0,19 (exp. 1) | Fuerte | Evidencia |
| Incluir `service`, `state`, `proto` codificadas | Solo variables numéricas | Macro-F1 +0,009; falsos positivos 1.935 → 1.719 (−11 %) (exp. 2). La hipótesis del usuario ("impacto masivo") se cumplió solo a medias | Débil a moderada | Evidencia, pero modesta: se mantienen "por ahora", con costo casi nulo |
| XGBoost en lugar de regresión logística | Regresión logística | Precisión con recall ≥ 0,90: 0,985 frente a 0,969 (bloques) y 0,992 frente a 0,973 (grupos). Gana en los 5 pliegues, en ambos regímenes; falsas alarmas por pliegue 1.989 → 963; macro-F1 0,949 frente a 0,922 | Fuerte | Evidencia |
| ...a pesar de que XGBoost pierde en recall bruto | | Con umbral 0,5 la regresión logística detecta más (ataques perdidos por pliegue 282 frente a 576; recall 0,988 frente a 0,976). Con un piso de recall común (el criterio acordado) XGBoost gana en precisión, pero la regresión logística es mejor si solo se mira recall sin piso | Matiz importante | La comparación justa es a igual recall (piso), no a igual umbral |
| Pliegues por grupos | Pliegues por bloques | Con bloques, filas casi idénticas quedaban a ambos lados; con grupos la desviación entre pliegues baja (0,017 → 0,002) y las falsas alarmas de XGBoost mejoran (tasa 0,119 → 0,086) (exp. 4) | Moderada | Evidencia (aunque la causa exacta del cambio no está aislada) |
| XGBoost con hiperparámetros de Optuna en lugar del XGBoost por defecto | XGBoost por defecto | Precisión con recall ≥ 0,95: 0,9797 frente a 0,9770 (+0,003, con desviación entre pliegues de 0,002); falsas alarmas 534 → 470 (−12 %, menos del 20 % fijado de antemano; solo 1 de 5 pliegues alcanza el 20 %); paisaje plano (las 8 mejores pruebas entre 0,9793 y 0,9797); sesgo de selección: la mejor de 46 pruebas sobre los mismos pliegues | **Débil** | **Zona gris.** La hipótesis previa no se cumplió. Se conserva porque no es peor, es el mismo tipo de modelo y no cuesta nada más, pero la evidencia no distingue claramente entre ambos |
| Umbral 0,6 | 0,5 u otro | Con piso de recall 0,95 el umbral óptimo estaba cerca de 0,7 (no de 0,5); 0,6 deja recall 0,966 (mínimo por pliegue 0,964); subir a 0,68 gana solo 0,006 de precisión | Moderada | Evidencia + criterio de costos (se prefiere una falsa alarma a un ataque sin revisar); el margen sobre el piso es la razón explícita |
| Congelar modelo y umbral tras la prueba | Reajustar tras ver la prueba | La guía lo exige ("conserva el resultado aunque empeore"). Prueba: recall 0,974, precisión 0,845, tasa de falsos positivos 0,219 | n/a | Regla del procedimiento, no del algoritmo |

## Respuesta

**Entre regresión logística y XGBoost: evidencia.** Es la decisión más sólida. XGBoost gana en el criterio fijado de antemano en los 10 pares de pliegues (5 por bloques, 5 por grupos), con diferencias mayores que el ruido en el régimen por grupos. No elegimos XGBoost por fama: se probó contra una alternativa más simple, con las mismas particiones, y la hipótesis se escribió antes. Que sea más complejo no se penalizó porque la interpretabilidad no es una prioridad en este problema (decisión del usuario), pero tampoco explica la elección: la prueba la hacen los números.

**Entre XGBoost por defecto y XGBoost con Optuna: la evidencia es débil.** La mejora (+0,003 de precisión, −12 % de falsas alarmas) es del tamaño del ruido y no cumplió el criterio previo. Se queda el modelo de Optuna porque es el mejor en validación y no cuesta nada, pero sería honesto defender cualquiera de los dos. También es el modelo con 706 árboles y profundidad 10, es decir, el más complejo, y esa complejidad no está respaldada por una mejora clara.

## Límites de la evidencia (lo que no sabemos)
- **Solo se compararon dos familias de modelos** (lineal y árboles potenciados), que es el mínimo de la guía. No se probaron bosques aleatorios, LightGBM ni redes; no podemos decir que XGBoost sea el mejor posible, solo que supera a la regresión logística.
- **La prueba no compara limpiamente los dos modelos.** La regresión logística se evaluó en prueba con otro umbral (0,5) y otra versión de entrenamiento (partición por bloques), y no se repitió con las mismas condiciones que XGBoost para no gastar la prueba en elegir candidatos. Las cifras de la prueba (falsos positivos 0,392 frente a 0,219) apuntan en la misma dirección, pero no son una comparación justa.
- **La validación es optimista:** las filas casi idénticas de una ráfaga están a ambos lados de los pliegues y la prueba incluye una sesión de captura (enero) que el entrenamiento no tiene. Esto afecta a ambos modelos por igual, pero no sabemos si de forma igual.
- **"Precisión con recall ≥ piso" usa el mejor umbral de cada pliegue** (techo optimista); es válido para comparar modelos, no como estimación del rendimiento real. La estimación real es la del umbral 0,6 sobre predicciones fuera de muestra y, sobre todo, la prueba.
- **Fuzzers (recall 0,827 en la prueba)** no se vio en validación: la validación no anticipó esta debilidad.
