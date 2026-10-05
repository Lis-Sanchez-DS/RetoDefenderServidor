# Exploración inicial (partición oficial de entrenamiento y prueba)

Hallazgos del 5 de octubre de 2026, sobre `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv`.

- Tamaño: 175.341 filas de entrenamiento y 82.332 de prueba; 45 columnas cada uno. No hay valores nulos.
- Balance de clases (`label`): entrenamiento 68,1 % ataque / 31,9 % normal; prueba 55,1 % ataque / 44,9 % normal. La proporción cambia entre particiones.
- Familias (`attack_cat`): muy desbalanceadas (Worms: 130 en entrenamiento y 44 en prueba; Generic: 40.000 en entrenamiento).
- Repeticiones: 74.301 filas de entrenamiento (42 %) y 28.386 de prueba (34 %) son duplicados exactos en las variables de entrada. Además, 10.279 filas de entrenamiento aparecen también en prueba. Esto puede inflar las métricas y hay que decidir cómo tratarlo.
- Identificador: `id` va de 1 a N en cada archivo y solo refleja el orden del archivo; no debe usarse como variable.
- Variables de respuesta: `label` y `attack_cat` (consistentes entre sí) no deben entrar como entradas.
- Categóricas: `proto` (133 valores), `service` (13; 53,7 % es "-"), `state` (9). En prueba aparecen estados (`ACC`, `CLO`) que no están en entrenamiento.
- La partición oficial no tiene conjunto de validación: hay que crearlo desde el entrenamiento.
