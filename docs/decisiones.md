# Decisiones

## Criterio de utilidad: un ataque no detectado cuesta más que una falsa alarma
Postura del equipo: un falso negativo (ataque que el modelo no identifica) es más costoso que un falso positivo (tráfico normal marcado como ataque). Por eso:
- Se prioriza el recall (ataques detectados) al elegir modelo y umbral de alerta.
- El recall no se optimiza solo: se vigilan la tasa de falsos positivos y, sobre todo, el macro-F1, que impide aceptar un modelo que alarma todo.

## Línea base trivial: predecir siempre "ataque"
La línea base predice siempre la clase dominante del entrenamiento (ataque, 68 %). Se eligió así porque coincide con el criterio anterior: es la política "ante la duda, alerta", que maximiza el recall.
Resultado en validación: recall 1,00, precisión 0,71, tasa de falsos positivos 1,00, macro-F1 0,415, accuracy 0,71.
Lectura: el recall perfecto no es mérito del modelo, y el macro-F1 de 0,415 y la tasa de falsos positivos de 1,00 muestran que no sirve. Cualquier modelo real debe mantener un recall alto y mejorar ese macro-F1 y reducir las falsas alarmas.

## Partición de validación
Se parte el entrenamiento oficial en bloques contiguos de 500 filas asignados al azar (80 % entrenamiento, 20 % validación, semilla 42). El conjunto de prueba no se usa para elegir nada.

## Codificación de `service` y `state`
Se aplica one-hot a `service` y `state` (ver `src/preprocesamiento.py`); el transformador se ajusta solo con entrenamiento y se guarda dentro del pipeline del modelo.

Se eliminan las filas cuyo `state` es `ACC` o `CLO` (5 filas en prueba, ninguna en entrenamiento) o `no` (1 fila en entrenamiento, ninguna en prueba). Son valores que aparecen en una sola partición: el modelo no los vio al entrenar, o no se pueden evaluar. Son 6 de 257.673 filas (0,002 %), por lo que su efecto en la población estudiada es insignificante y solo complicarían la codificación. Tras eliminarlas: 175.340 filas de entrenamiento y 82.327 de prueba.

## `proto`: agrupar los infrecuentes y aplicar one-hot
Decisión: los valores de `proto` que aparecen en menos del 1 % de las filas de entrenamiento se agrupan en una categoría "infrecuente" y luego se aplica one-hot (`min_frequency=0.01` en `src/preprocesamiento.py`). La frecuencia se calcula al ajustar el transformador, solo con entrenamiento; los valores nuevos en prueba caen en la misma categoría.
Resultado: 6 columnas (`tcp`, `udp`, `unas`, `arp`, `ospf` e infrecuente; esta última reúne 128 protocolos y 14.573 filas). Con esto la entrada tiene 66 columnas.
Justificación del umbral de 1 %:
- Aparecer en menos del 1 % de las filas tiene una fuerte correlación con ser ataque: la categoría infrecuente reúne 14.573 filas de entrenamiento y es casi todo ataque (126 de los 133 protocolos son 100 % ataque).
- Deja un buen equilibrio: se conserva la variación de `proto` que importa (`tcp` y `udp`, que son mixtos, más `arp`, `unas` y `ospf`, cada uno con un comportamiento distinto) y se absorbe en una sola categoría la mayoría de las clases pequeñas (128 de 133 valores), evitando decenas de columnas casi vacías.
- Advertencia: justamente por esa correlación, la categoría infrecuente puede ser un atajo del laboratorio y no un comportamiento transferible a una red real; por eso se evaluará con y sin `proto`.

Hallazgos que motivaron el análisis:
- Los protocolos distintos de tcp/udp (32.111 filas, 91 % ataques) casi siempre tienen `state` INT y `service` "-". Esa celda (INT + "-") es mixta por sí sola (86,7 % ataques), así que `state` y `service` no explican el patrón: dentro de ella, `unas`, `ospf` y los demás protocolos raros son ~100 % ataque, `arp` 0 % y `udp` 76 %.
- Una regla basada solo en `state` y `service` acierta el 3 % de las filas de `arp`; es decir, `proto` aporta información adicional. El acierto de una tabla de búsqueda en muestra sube de 0,794 (state+service) a 0,818 (con proto).
- El patrón se repite en la partición de prueba (arp 0 %, unas 100 %, ospf 99,8 %), pero en prueba la celda INT + "-" baja a 45 % de ataques para udp.
- Riesgo: es probable que refleje cómo se generó el ataque en el laboratorio (herramientas que usan protocolos inusuales) y no un comportamiento transferible a una red real. Se tratará como experimento: con y sin `proto`.

## Enfoque en precisión con piso de recall
Decisión: de aquí en adelante se busca mejorar la precisión (menos falsas alarmas) sin sacrificar la detección. Regla: **solo se mejora la precisión mientras el recall se mantenga por encima de 0,90**. El recall sigue siendo la prioridad principal por el criterio de costo (un ataque no detectado cuesta más que una falsa alarma); 0,90 es un piso provisional, que se puede subir.

Origen de la decisión: ya en validación la regresión logística con categóricas mostraba un recall alto (0,988) pero todavía más de 1.000 falsas alarmas (1.719 de 10.166 normales, 17 %). Ese patrón, visible antes de mirar la prueba, es lo que motiva el enfoque. El resultado en prueba (precisión 0,75, 14.483 falsas alarmas) es coherente con él pero no se usa para ajustar nada.

Cómo se mide: además de las métricas al umbral 0,5, se reporta la **precisión máxima con recall ≥ 0,90** de cada pliegue (sobre su curva precisión-recall). Así se compara la capacidad de cada modelo para el objetivo sin elegir todavía el umbral. El umbral se elegirá más adelante, con las predicciones fuera de muestra de la validación cruzada, nunca con la prueba.

## Validación cruzada por bloques (sin mezclar filas)
Se usa validación cruzada de 5 pliegues sobre el entrenamiento oficial, conservando el orden del archivo: las filas se agrupan en bloques contiguos de 500 y se asignan bloques completos a cada pliegue (los bloques se reparten al azar con semilla 42, las filas dentro de un bloque no se mezclan).

Por qué no pliegues contiguos puros ni validación hacia adelante en el tiempo:
- El archivo está ordenado en bloques por clase: por décimas de `id` la proporción de ataques es 0 %, 0 %, 27 %, 95 %, 86 %, 88 %, 85 %, 100 %, 100 %, 100 %. Un pliegue contiguo tendría una sola clase y la precisión o la tasa de falsos positivos no estarían definidas; el primer modelo de una validación hacia adelante se entrenaría solo con tráfico normal.
- El orden del archivo no coincide exactamente con el tiempo de captura (correlación 0,96 en las filas que pude fechar), así que tampoco sería una validación temporal real.
- Los bloques mantienen juntas las filas vecinas (que comparten ráfagas de tráfico y contadores `ct_*` parecidos), evitando que filas casi idénticas queden a ambos lados y inflen la validación. Es la misma lógica de la partición de entrenamiento/validación usada hasta ahora.
Limitación: no se han eliminado los duplicados exactos, que pueden seguir cruzando pliegues; y los pliegues no reproducen el cambio de sesión de captura que sí aparece en prueba.

Ambos modelos (regresión logística y XGBoost) se entrenan con los mismos pliegues y el mismo preprocesamiento para que la comparación sea justa.

## Piso de recall elevado a 0,95
Desde este punto, el piso de recall del criterio "mayor precisión con recall mínimo" pasa de 0,90 a **0,95** (decisión del equipo). Los experimentos 1 a 4 se evaluaron con el piso de 0,90 y así se conservan; la columna `precision_con_recall_090` de sus ejecuciones sigue siendo válida, y desde la búsqueda de hiperparámetros se registra también `precision_con_recall_095`. El recall sigue siendo la prioridad principal por el criterio de costo; el piso puede volver a subirse.
