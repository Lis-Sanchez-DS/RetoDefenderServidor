# De los datos originales a los datos preparados

El taller pide conservar con DVC **dos versiones de datos** y documentar la transformación entre ellas. Cada una tiene una referencia de Git (etiqueta) y sus punteros `.dvc`.

| Versión | Qué es | Referencia Git | Puntero DVC (md5) |
|---|---|---|---|
| Datos originales | `UNSW_NB15_training-set.csv` y `UNSW_NB15_testing-set.csv` tal como se descargaron, más `NUSW-NB15_features.csv` | etiqueta `datos-originales` (commit `7e1b3ac`) | training `e55caabaa6cd4a8f1c06a227bcfababc` (SHA-256 en `docs/datos.md`) |
| Datos preparados | `data/processed/particion_entrenamiento_validacion.csv`: la asignación de cada fila de entrenamiento a entrenamiento o validación, con su grupo | etiqueta `datos-preparados` (commit `6f6399e`) | `a869e919c9a8a014fa6c7aec00446ce5` |

## Transformación (`python -m src.particion`)
1. Se lee el archivo de entrenamiento original (175.341 filas, sin modificarlo).
2. Se excluye 1 fila cuyo estado (`no`) casi no existe en el entrenamiento (en la prueba no se elimina ninguna fila; los estados no vistos se codifican con ceros). Quedan 175.340 filas.
3. Se calcula el **grupo** de cada fila: todas las filas con el mismo vector de las 42 variables de entrada (sin `id`, `label` ni `attack_cat`) comparten grupo. Tras la exclusión quedan 101.039 vectores distintos y 74.301 filas son copia de otra.
4. Se hace un `StratifiedGroupKFold` de 10 pliegues (mezclado, semilla 42, estratificado por `label`) y se unen 3 pliegues como validación: **70/30**, con grupos completos.
5. Se guarda un CSV con `row_id` (`training:<id>`), `id`, `conjunto`, `grupo` y `label`. Resultado: 122.735 filas de entrenamiento y 52.605 de validación, 68,06 % de ataques en ambas y 0 grupos repartidos. Las huellas de cada conjunto están en `evidencias/auditoria.json`.

Los valores de las variables no se copian: el CSV preparado solo asigna filas, y los datos siguen leyéndose del archivo original. El **preprocesamiento** (estandarización, `one-hot` de `service` y `state`, `proto` con categoría infrecuente) se ajusta solo con entrenamiento y viaja dentro del pipeline de cada modelo (`src/preprocesamiento.py`). La prueba no interviene.

## Recuperar cada versión
```bash
git checkout datos-originales && .venv/bin/dvc pull -r respaldo data/raw/unsw-nb15/particion_oficial/UNSW_NB15_training-set.csv.dvc   # o el remoto almacen
git checkout datos-preparados && .venv/bin/dvc pull -r respaldo data/processed/particion_entrenamiento_validacion.csv.dvc
git checkout main
```
(`respaldo` es el almacenamiento entregado; ver `README.md`.) Después de cada `pull`, el SHA-256 o el md5 del archivo debe coincidir con la tabla.
