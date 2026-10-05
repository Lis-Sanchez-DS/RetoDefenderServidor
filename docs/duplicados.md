# Filas duplicadas: problema, intentos de solución y decisión

## El problema
En el entrenamiento oficial (175.340 filas tras excluir 1 fila con estado raro), 74.301 filas (42,4 %) son copias exactas de otra fila en las 42 variables de entrada (101.039 vectores distintos). Se concentran en los ataques: son copia el 58,8 % de las filas de ataque frente al 7,5 % de las normales (Generic 90 %, DoS 76 %, Analysis 73 %, Backdoor 71 %, Exploits 44 %). Los grupos más grandes (hasta 415 filas) son sondeos de un solo paquete (conexiones de duración casi nula y sin respuesta), que son idénticos porque no hay nada más que los distinga. Además, 229 grupos (940 filas) tienen vectores idénticos con etiquetas distintas: error imposible de evitar.

Efecto en la validación: con pliegues por bloques, el 44,5 % de las filas de validación tenía una fila idéntica en el entrenamiento del mismo pliegue. XGBoost fallaba el 0,7 % de esas filas frente al 9,1 % de las que no tenían gemela (ataques: 0,2 % frente a 5,9 %), así que el recall de la validación por bloques es optimista. Las falsas alarmas no se ven afectadas: solo el 5 % de las filas normales tiene gemela y su error es el mismo con o sin ella (12,8 % frente a 11,7 %). La prueba también las tiene: 34,5 % de copias extra dentro de la prueba y 10,4 % de sus filas con un vector idéntico en entrenamiento (15,6 % de los ataques de prueba).

## Intento 1 (insuficiente): pliegues por bloques
Los bloques contiguos de 500 filas no impiden la fuga: solo el 21 % de los grupos de copias queda dentro de un bloque (mediana de 11.276 filas entre la primera y la última copia), y 7.068 de 9.748 grupos tienen copias en más de un pliegue.

## Intento 2 (fallido): reponer las copias con filas nuevas del conjunto completo
Idea: quitar las copias extra y reemplazarlas por el mismo número de filas muestreadas de las 2,54 millones del conjunto crudo, con condiciones parecidas (sesión de captura de febrero, misma clase). No es viable:
- La partición oficial ya contiene todas las filas de ataque del conjunto crudo, salvo las de Generic. Para Exploits, DoS, Reconnaissance, Fuzzers, Analysis, Backdoor, Shellcode y Worms (unas 34.000 copias extra) no queda ninguna fila nueva que muestrear.
- Para Generic (36.076 copias) quedan 81.479 filas sin usar, pero solo unos 7.331 vectores distintos (todos de febrero): el resto serían de nuevo copias.
- Solo Normal (4.185 copias) tiene reserva amplia (cerca de 1,7 millones de vectores distintos).
- En total, unas 11.500 de las 74.301 copias se podrían reponer con filas nuevas; el resto seguiría duplicado, y al reponer solo Normal y Generic la mezcla de clases cambiaría. Se descartó.
Nota: este análisis fue exploratorio (guion no versionado) y compara filas con 25 de las 42 columnas para ahorrar memoria, así que las cifras de vectores distintos son aproximadas.

## Intento 3 (descartado): agrupar a la vez por bloque y por vector idéntico
Para conservar a la vez el orden por bloques y la separación de las copias, se probó unir las filas que comparten bloque o vector. Resultado: un único componente conexo con el 89,4 % de las filas; imposible repartirlo en pliegues.

## Decisión: pliegues "conscientes de grupos" por vector idéntico
Se usa validación cruzada de 5 pliegues en la que todas las filas con el mismo vector de entrada caen en el mismo pliegue (`StratifiedGroupKFold`, con mezcla y semilla 42, estratificada por clase). Así ninguna fila de validación tiene una copia exacta en el entrenamiento, sin eliminar ni cambiar filas (la población y las frecuencias reales se conservan).
Limitaciones:
- Se renuncia a la separación por bloques: filas vecinas parecidas (no idénticas) pueden quedar a ambos lados.
- Las copias siguen pesando en el entrenamiento (un sondeo repetido 400 veces cuenta 400 veces).
- La prueba mantiene sus copias y su solapamiento con el entrenamiento: sus números seguirán siendo algo optimistas en el recall.
