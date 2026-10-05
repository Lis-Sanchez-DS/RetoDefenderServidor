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
Pendiente: se completará con los nombres de archivo, tamaños y sumas SHA-256 reales una vez descargados manualmente en `data/raw/`.

## Limitaciones conocidas (a verificar en la exploración)
Benchmark histórico de laboratorio (2015): no demuestra eficacia sobre redes actuales.
