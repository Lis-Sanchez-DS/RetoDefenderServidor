# Datos: UNSW-NB15

## Fuente oficial
- Página del proyecto: https://research.unsw.edu.au/projects/unsw-nb15-dataset (última actualización indicada: 2 de junio de 2021; consultada el 5 de octubre de 2026).
- Descarga: carpeta de SharePoint de UNSW enlazada desde esa página. No permite descarga por script (responde 403), por lo que la descarga es manual.
- Creadores: Nour Moustafa y Jill Slay, UNSW Canberra. Tráfico generado con la herramienta IXIA PerfectStorm (~100 GB de paquetes capturados); características extraídas con Argus y Bro-IDS.

## Contenido
- 2.540.044 registros en 4 CSV (`UNSW-NB15_1.csv` a `UNSW-NB15_4.csv`), sin encabezado; los nombres de columnas están en `NUSW-NB15_features.csv`.
- Partición propuesta por los autores: `UNSW_NB15_training-set.csv` (175.341 registros) y `UNSW_NB15_testing-set.csv` (82.332 registros).
- 49 características más la etiqueta (`label`: 0 normal, 1 ataque; `attack_cat`: familia del ataque).
- Nueve familias de ataque: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms.

## Condiciones de uso
- Licencia (texto de la fuente): "Free use of the UNSW-NB15 dataset for academic research purposes is hereby granted in perpetuity." El uso comercial requiere acuerdo con los autores.
- Este proyecto es de uso académico (curso de MLOps, Universidad Externado de Colombia).
- Los datos NO se redistribuyen en este repositorio público: están excluidos de Git y se versionan con DVC.
- Contacto de los autores: nour.moustafa@unsw.edu.au

## Citas requeridas por la fuente
1. Moustafa, N. y Slay, J. "UNSW-NB15: a comprehensive data set for network intrusion detection systems." MilCIS, IEEE, 2015.
2. Moustafa, N. y Slay, J. "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset." Information Security Journal: A Global Perspective, 2016, 1-14.
3. Moustafa, N. et al. "Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks." IEEE Transactions on Big Data, 2017.
4. Moustafa, N. et al. "Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models." Data Analytics and Decision Support for Cybersecurity, Springer, 2017, 127-156.
5. Sarhan, M. et al. "NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems." BDTA 2020, Springer.

## Versión utilizada
Archivos descargados manualmente de la carpeta oficial de UNSW el 5 de octubre de 2026 (página actualizada por última vez el 2 de junio de 2021). Se guardan en `data/raw/` y los versiona DVC (archivos `.dvc`).

| Archivo | Filas de datos | Tamaño (bytes) | SHA-256 |
|---|---|---|---|
| `UNSW_NB15_training-set.csv` | 175.341 | 32.293.018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `UNSW_NB15_testing-set.csv` | 82.332 | 15.380.800 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| `NUSW-NB15_features.csv` | 49 (descripción de variables) | 4.044 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` |
| `NUSW-NB15_GT.csv` | 188.913 | 86.426.111 | `6d27542cb6457db599e0a78274ac141fec0296531f00ac5569551a276c5ab1a8` |

Las filas de entrenamiento y prueba coinciden con las cifras publicadas por los autores. Los cuatro CSV crudos (`UNSW-NB15_1..4.csv`) no se descargaron por ahora.

## Limitaciones conocidas (a verificar en la exploración)
Benchmark histórico de laboratorio (2015): no demuestra eficacia sobre redes actuales.
