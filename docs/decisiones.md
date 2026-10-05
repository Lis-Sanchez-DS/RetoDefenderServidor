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
