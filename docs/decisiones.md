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
