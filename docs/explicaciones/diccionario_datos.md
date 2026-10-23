# Diccionario de datos

Referencia unica de todas las columnas y variables del proyecto.

---

## dataset_fusionado.csv (7.719 filas x 25 columnas)

El dataset principal. Cada fila es un incendio con sus condiciones meteorologicas.

### Columnas del incendio

| Columna | Tipo | Descripcion | Ejemplo |
|---|---|---|---|
| `id` | string | Identificador unico del incendio (EGIF) | 2017320145 |
| `fecha` | date | Fecha de deteccion del incendio | 2017-08-15 |
| `lat` | float | Latitud WGS84 del punto del incendio | 42.1523 |
| `lng` | float | Longitud WGS84 del punto del incendio | -7.8451 |
| `municipio` | string | Nombre del municipio de Ourense | VIANA DO BOLO |
| `superficie` | float | Hectareas quemadas (forestal) | 23.5 |
| `causa` | int | Codigo de causa del incendio | 1-5 |
| `causa_supuesta` | int | 1 si la causa es sospechada, no confirmada | 0 o 1 |

### Columnas de asignacion espacial

| Columna | Tipo | Descripcion | Ejemplo |
|---|---|---|---|
| `estacion_id` | int | ID de la estacion meteorologica asignada | 10116 |
| `dist_estacion_m` | float | Distancia al incendio en metros | 8432.5 |
| `estacion_rank` | int | Si fue la 1a, 2a, 3a... estacion mas cercana con datos | 1 |
| `estacion_viento_id` | int | ID de la estacion usada para datos de viento | 10116 |
| `dist_viento_m` | float | Distancia a la estacion de viento en metros | 8432.5 |

### Variables meteorologicas (12)

| Columna | Descripcion | Unidad | Rango fisico |
|---|---|---|---|
| `TA_AVG_1.5m` | Temperatura media del dia | °C | -15 a 50 |
| `TA_MAX_1.5m` | Temperatura maxima del dia | °C | -15 a 50 |
| `TA_MIN_1.5m` | Temperatura minima del dia | °C | -15 a 50 |
| `HR_AVG_1.5m` | Humedad relativa media | % | 0 a 100 |
| `HR_MIN_1.5m` | Humedad relativa minima | % | 0 a 100 |
| `PP_SUM_1.5m` | Precipitacion total (lluvia) | L/m² | 0 a 500 |
| `VV_AVG_10m` | Velocidad media del viento | m/s | 0 a 50 |
| `VV_MAX_10m` | Racha maxima de viento | m/s | 0 a 80 |
| `IRD_SUM_1.5m` | Irradiacion solar diaria | 10kJ/(m²·dia) | 0 a 4.000 |
| `HSOL_SUM_1.5m` | Horas de sol | h | 0 a 18 |
| `TO_AVG_1.5m` | Temperatura de punto de rocio | °C | -25 a 40 |
| `PR_AVG_1.5m` | Presion atmosferica media | hPa | 750 a 1.080 |

**Nota**: El sufijo `1.5m` indica que el sensor esta a 1,5 metros del suelo. El sufijo `10m` indica sensor a 10 metros (anemometros).

---

## incendios_ourense.csv (7.719 filas x 11 columnas)

Incendios de Ourense desde 2005, limpios. Columnas: `id`, `fecha`, `lat`, `lng`, `latlng_explicit`, `idprovincia`, `idmunicipio`, `municipio`, `superficie`, `causa`, `causa_supuesta`.

## meteo_ourense.csv (106.928 filas x 14 columnas)

Datos meteorologicos diarios de las 20 estaciones. Columnas: `fecha`, `estacion_id`, + las 12 variables meteorologicas.

## estaciones_ourense.csv (20 filas x 5 columnas)

| Columna | Descripcion |
|---|---|
| `estacion_id` | ID de la estacion MeteoGalicia |
| `nombre` | Nombre de la estacion |
| `utmx` | Coordenada UTM X (zona 29N, metros) |
| `utmy` | Coordenada UTM Y (zona 29N, metros) |
| `dias` | Numero de dias con datos descargados |

---

## Codigos de causa de incendio (campo `causa`)

| Codigo | Descripcion |
|---|---|
| 1 | Rayo |
| 2 | Negligencia o causa accidental |
| 3 | Intencionado |
| 4 | Desconocida |
| 5 | Reproduccion de incendio anterior |

## Codigos de validacion MeteoGalicia (usados en la limpieza)

| Codigo | Significado | Accion |
|---|---|---|
| 1 | Dato valido original | Aceptado |
| 5 | Dato valido interpolado | Aceptado |
| 0 | Sin validar | Descartado |
| 2 | Sospechoso | Descartado |
| 3 | Erroneo | Descartado |
| 9 | No registrado | Descartado |
