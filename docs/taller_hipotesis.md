# Taller evaluable: hipótesis de las cuatro configuraciones (escritas antes de entrenar)

Operacionalización del asistente a partir de los experimentos 1-6 (hipótesis y resultados del usuario en `reports/experimentos.md`). Todas usan la misma partición fija (`data/processed/particion_entrenamiento_validacion.csv`, semilla 42, grupos de entradas idénticas sin repartir), el mismo preprocesamiento ajustado solo con entrenamiento y las mismas métricas en validación. Ataque es la clase positiva. Criterio: recall como piso (>= 0,95), luego precisión.

| ID | Familia | Cambio respecto a la anterior | Hipótesis |
|---|---|---|---|
| C1 | Regresión logística | Referencia: solo variables numéricas, estandarizadas | Con umbral 0,5 tendrá recall alto y muchas falsas alarmas (FPR > 0,15), porque sin protocolo, servicio ni estado no distingue bien el tráfico normal. |
| C2 | Regresión logística | + variables categóricas (`service`, `state`, `proto` agrupado) | Mejora la precisión y baja la FPR respecto a C1 al menos 3 puntos de FPR; sigue siendo lineal, así que no alcanza a XGBoost. **Será v1** (primera solución funcional). |
| C3 | Árboles (XGBoost) | Cambia la familia: 300 árboles, profundidad 6, tasa 0,1 | Con recall >= 0,95 su precisión supera a C2 por más de 0,01 y las falsas alarmas se reducen a la mitad o más. |
| C4 | Árboles (XGBoost) | Hiperparámetros por búsqueda con Optuna en el entrenamiento (sin validación) | Mejora a C3 en precisión con recall >= 0,95 menos de 0,01 (diferencia del tamaño del ruido, como en el experimento 5). **Candidato a v2.** |

Reglas fijadas antes de ver resultados:
- Umbral de cada versión: el mayor valor que, en validación, cumple recall >= 0,95 (el que maximiza la precisión), redondeado hacia abajo a 0,05 para dejar margen; no se toca después.
- v2 = la configuración con mayor precisión con recall >= 0,95 en validación; v2 debe diferir de v1 en modelo y umbral.
- Test: una sola evaluación del modelo v2 congelado, sobre las 82.332 filas.
- Antecedente a declarar: en el desarrollo previo (guía anterior) se evaluó en test un modelo parecido (regresión logística con categóricas, y luego el XGBoost con Optuna). Esos resultados no se usaron para elegir el modelo ni el umbral actuales, pero se conocen.
