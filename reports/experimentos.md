# Comparación de experimentos

Todos los experimentos usan la misma partición de entrenamiento/validación (bloques de 500 filas, 80/20, semilla 42, sin `id`) y se registran en MLflow (experimento `unsw-nb15`). El conjunto de prueba no se usa. Umbral de decisión: 0,5 (aún no se ha ajustado). Las hipótesis se escribieron antes de entrenar.

## Experimento 1: regresión logística sin variables categóricas
- Hipótesis: con solo las 39 variables numéricas (estandarizadas), la regresión logística supera con claridad a la línea base (macro-F1 0,415) y, sobre todo, deja de alarmar todo el tráfico normal (tasa de falsos positivos muy por debajo de 1,00).
- Cambio: respecto a la línea base, un modelo real con solo variables numéricas.

## Experimento 2: regresión logística con variables categóricas codificadas
- Hipótesis: agregar `service`, `state` y `proto` (one-hot, `proto` con categoría infrecuente) mejora el macro-F1 y reduce los falsos positivos respecto al experimento 1, porque estas variables concentran mucha información sobre la clase (p. ej. `state` INT 93 % ataques, `ssh` 0,8 %). Si no mejora, la codificación no está aportando.
- Cambio: respecto al experimento 1, solo se agregan las tres variables categóricas codificadas.

## Resultados
(se completan tras entrenar)
