# Pasos 1 a 7: Obtencion, evaluacion y preparacion de los datos

Documento explicativo de los primeros pasos del proyecto, desde la obtencion de los datos hasta la evaluacion previa a la limpieza y fusion. Escrito para que Lua pueda defender cada decision ante el tribunal.

---

## Paso 1: Obtencion del dataset de incendios forestales

### Que se hizo

Se descargo el dataset historico de todos los incendios forestales registrados en España desde la Fundacion Civio.

### Fuente de los datos

Los datos provienen de la **Estadistica General de Incendios Forestales (EGIF)**, elaborada por el Ministerio para la Transicion Ecologica. El EGIF es el registro oficial de incendios forestales de España, pero sus datos no estan facilmente accesibles en formato descargable. La **Fundacion Civio**, una organizacion dedicada a la transparencia y los datos abiertos, recopila y publica estos datos en formato CSV.

- URL: https://datos.civio.es/dataset/todos-los-incendios-forestales/
- Fichero obtenido: `fires-all.csv` (~31 MB)

### Que contiene

- **292.181 registros** de incendios forestales en toda España
- Rango temporal: **1968 a 2023** (55 anos)
- Campos principales: identificador unico (`id`), fecha del incendio, coordenadas geograficas (latitud/longitud), provincia, municipio, superficie quemada, causa (real o supuesta), tiempos de deteccion/control/extincion, personal desplegado, medios aereos, gastos y perdidas

### Por que esta fuente y no otra

1. Es la unica fuente publica que ofrece incendios de toda España en formato CSV con fecha exacta y coordenadas geograficas — ambos campos son imprescindibles para cruzar con datos meteorologicos
2. Los datos originales del Ministerio se publican en informes PDF agregados, no como datos individuales descargables
3. Civio ya ha hecho el trabajo de limpieza y estructuracion del EGIF, ahorrando meses de preprocesamiento

### Limitaciones conocidas

- **Geolocalizacion**: Historicamente, ~54% de los incendios no tienen coordenadas precisas (se usa el centroide del municipio). Sin embargo, desde 2005 el 99,99% de los registros de Ourense SI tienen coordenadas exactas, asi que esta limitacion no nos afecta.
- **Causa**: Es "supuesta" en mas del 76% de los casos. No afecta al proyecto porque nuestro modelo predice riesgo a partir de meteorologia, no a partir de la causa.
- **Datos economicos**: Gastos (~19% sin datos) y perdidas (~34% sin datos) estan incompletos. Tampoco los usamos.

---

## Paso 2: Evaluacion de viabilidad del cruce de datos

### Objetivo

Antes de invertir tiempo en descargar y procesar datos meteorologicos, se verifico que el cruce espacial-temporal entre incendios y meteorologia era viable. Es decir: hay suficientes incendios con coordenadas, las estaciones meteorologicas cubren el mismo rango temporal, y la API funciona correctamente.

### Analisis del CSV de incendios

Se analizaron los 292.181 registros para responder tres preguntas clave:

**Pregunta 1: Cuantos incendios hay en Ourense?**

Se filtro por `idprovincia == 32` (codigo de Ourense). Resultado:
- Total en España: 292.181
- Solo Ourense: 36.813
- Ourense desde 2005 (solapamiento con MeteoGalicia): **7.720 incendios**

Se usa 2005 como ano de inicio porque es cuando las estaciones meteorologicas de MeteoGalicia empiezan a tener datos disponibles.

**Pregunta 2: Tienen coordenadas?**

- Ourense total: 32.460 con coordenadas (88,2%)
- Ourense desde 2005: **7.719 con coordenadas (99,99%)**

Solo 1 registro no tiene coordenadas. Esto es critico: sin coordenadas no podemos saber que estacion meteorologica estaba mas cerca del incendio.

**Pregunta 3: Como se distribuyen en el tiempo?**

| Ano  | Incendios | Ano  | Incendios |
|------|-----------|------|-----------|
| 2005 | 1.539     | 2014 | 93        |
| 2006 | 569       | 2015 | 368       |
| 2007 | 493       | 2016 | 220       |
| 2008 | 469       | 2017 | 480       |
| 2009 | 614       | 2018 | 101       |
| 2010 | 463       | 2019 | 170       |
| 2011 | 1.020     | 2020 | 172       |
| 2012 | 512       | 2021 | 86        |
| 2013 | 351       |      |           |

Nota: los datos del CSV llegan hasta 2021, no hasta 2023 como indicaba Civio. Los ultimos anos (2018-2021) tienen menos registros, posiblemente por retraso en la publicacion del EGIF.

### Prueba de la API de MeteoGalicia

Se realizaron llamadas de prueba al endpoint de datos diarios:

```
GET https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action
    ?idEst={ID}&dataIni=dd/MM/yyyy&dataFin=dd/MM/yyyy
```

Resultados:
- La API **no requiere autenticacion** ni clave
- Responde en formato JSON limpio y bien estructurado
- Las estaciones serie 10xxx (antiguas) tienen datos desde ~2005
- Las estaciones serie 19xxx (nuevas) solo desde ~2015-2017
- Formato de fechas: `dd/MM/yyyy` (no ISO), detalle importante para evitar errores

### Conclusion: el cruce es viable

1. Hay 7.720 incendios en Ourense desde 2005, practicamente todos con coordenadas
2. Las estaciones 10xxx cubren desde 2005, solapando con todo el rango de incendios
3. Hay al menos 20 estaciones repartidas por la provincia
4. Las variables necesarias (temperatura, humedad, precipitacion, viento) estan disponibles
5. La API es publica, gratuita y funciona correctamente

**Rango util del dataset final: 2005-2021 (17 anos).**

---

## Paso 3: Preparacion del entorno Python

### Que se hizo

Se creo un entorno virtual Python y se instalaron las dependencias del proyecto.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn requests
```

### Por que un entorno virtual

- **Aislamiento**: Las librerias del proyecto no interfieren con las del sistema operativo ni con otros proyectos
- **Reproducibilidad**: Se pueden congelar las versiones exactas en `requirements.txt` para que cualquiera pueda replicar el entorno
- **Compatibilidad**: macOS bloquea la instalacion global de paquetes Python con pip desde la version Ventura (PEP 668), asi que el entorno virtual es obligatorio

### Librerias instaladas y su funcion

| Libreria | Version | Para que se usa |
|---|---|---|
| pandas | >= 2.0 | Manipulacion de datos tabulares (DataFrames) |
| numpy | >= 1.24 | Operaciones numericas y arrays |
| scikit-learn | >= 1.5 | Modelos de machine learning y metricas |
| xgboost | >= 2.0 | Modelo de gradient boosting |
| imbalanced-learn | >= 0.12 | Herramientas para datos desbalanceados |
| matplotlib | >= 3.7 | Graficos y visualizaciones |
| seaborn | >= 0.12 | Graficos estadisticos (sobre matplotlib) |
| requests | >= 2.31 | Peticiones HTTP a la API de MeteoGalicia |
| jupyter | >= 1.0 | Notebooks para exploracion interactiva |

---

## Paso 4: Identificacion de estaciones meteorologicas objetivo

### Que se hizo

Se consulto la API de MeteoGalicia para listar todas las estaciones de Ourense y se verifico la fecha de inicio de datos de cada una.

### Proceso

1. Se consulto el endpoint de listado de estaciones:
   ```
   GET https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action
   ```

2. Se encontraron **34 estaciones** en la provincia de Ourense:
   - 20 estaciones serie 10xxx (antiguas, datos mas largos)
   - 14 estaciones serie 19xxx (nuevas, datos desde ~2015-2017)

3. Para cada estacion 10xxx, se probo la API pidiendo datos de enero de anos sucesivos hasta encontrar el primer ano con datos

### Decision: usar las 20 estaciones 10xxx

Las estaciones 19xxx se descartaron por tener un rango temporal demasiado corto (6-7 anos frente a 16-17 anos de las 10xxx). Incluirlas habria aportado muy pocos datos adicionales y complicado el analisis.

### Las 20 estaciones seleccionadas

| ID    | Nombre              | Concello                 | Datos desde |
|-------|---------------------|--------------------------|-------------|
| 10048 | O Invernadeiro      | Vilarino de Conso        | 2005        |
| 10057 | Alto do Rodicio     | Maceda                   | 2005        |
| 10058 | Verin-Vilamaior     | Verin                    | 2005        |
| 10109 | Amiudal             | Avion                    | 2006        |
| 10110 | Baltar              | Baltar                   | 2006        |
| 10111 | Entrimo             | Entrimo                  | 2006        |
| 10112 | Gandarela           | Celanova                 | 2006        |
| 10113 | Monte Medo          | Banos de Molgas          | 2006        |
| 10114 | As Petarelas        | Rubia                    | 2006        |
| 10115 | Serra do Eixe       | A Veiga                  | 2006        |
| 10116 | Viana do Bolo       | Viana do Bolo            | 2006        |
| 10117 | San Xoan de Rio     | San Xoan de Rio          | 2006        |
| 10119 | Xures               | Muinos                   | 2006        |
| 10130 | Lardeira            | Carballeda de Valdeorras | 2008        |
| 10131 | Cabeza de Manzaneda | Manzaneda                | 2007        |
| 10138 | Xares               | A Veiga                  | 2008        |
| 10148 | Ourense             | Ourense                  | 2012        |
| 10155 | Ourense-Estacions   | Ourense                  | 2015        |
| 10204 | Casaio              | Carballeda de Valdeorras | 2013        |
| 10500 | Verin-Vilela        | Verin                    | 2009        |

Las 3 estaciones mas antiguas (10048, 10057, 10058) tienen datos desde 2005. La mayoria arranca en 2006. Las mas recientes (10148, 10155, 10204) empiezan entre 2012 y 2015.

---

## Paso 5: Descarga masiva de datos meteorologicos

### Que se hizo

Se escribio el script `src/descargar_meteo.py` que descarga automaticamente todos los datos diarios de las 20 estaciones, mes a mes, desde su ano de inicio hasta 2021.

### Como funciona el script

1. Itera sobre las 20 estaciones con su ano de inicio
2. Para cada estacion, recorre **todos los meses** del rango (ano_inicio a 2021)
3. Hace una peticion GET por mes al endpoint de datos diarios de MeteoGalicia
4. Espera **0.5 segundos entre peticiones** (cortesia con el servidor para no saturarlo)
5. Guarda un fichero JSON por estacion en `data/raw/meteogalicia/estacion_{ID}.json`

### Volumen de la descarga

Aproximadamente **3.200 peticiones HTTP** (20 estaciones x ~160 meses de media). La descarga tarda varias horas por la pausa entre peticiones.

### Resultado

La descarga se completo con **0 errores** en todas las estaciones.

| Metrica | Valor |
|---|---|
| Estaciones descargadas | 20 |
| Dias totales | 106.928 |
| Tamano total | 285,7 MB |
| Errores HTTP | 0 |

Cada JSON contiene:
- Metadatos de la estacion (id, nombre, concello, rango)
- Array de dias (`listDatosDiarios`), cada uno con las mediciones del dia organizadas por variable

Los ficheros se almacenaron en `data/raw/meteogalicia/` y no se modifican en pasos posteriores (los datos raw se preservan siempre intactos).

---

## Paso 6: Verificacion de calidad de los datos

### Objetivo

Antes de limpiar y fusionar, se analizo la calidad de ambos datasets para confirmar que sirven para el modelado.

### Calidad del dataset de incendios

- 7.720 registros, rango 2005-2021
- 7.719 con coordenadas (99,99%)
- Todas las fechas validas
- 0 nulos significativos
- Superficie: mediana 3 ha, media 23,8 ha, maximo 5.970,7 ha

**Estacionalidad** (incendios por mes):

| Mes | Incendios | Mes | Incendios |
|---|---|---|---|
| Enero | 276 | Julio | 472 |
| Febrero | 1.063 | Agosto | 1.483 |
| Marzo | 1.264 | Septiembre | 1.059 |
| Abril | 461 | Octubre | 873 |
| Mayo | 178 | Noviembre | 282 |
| Junio | 131 | Diciembre | 178 |

Se observa un patron de dos picos, coherente con la realidad gallega:
- **Febrero-marzo**: quemas agricolas de invierno (rozas) que se descontrolan
- **Agosto-septiembre**: sequia estival combinada con altas temperaturas

### Calidad de los datos meteorologicos

- 106.928 dias totales descargados
- 99,5% con datos validados (codigo de validacion 1 o 5 de MeteoGalicia)
- 72,1% con las 4 variables basicas simultaneamente (temperatura, humedad, lluvia, viento)

**Problema detectado: 4 estaciones no miden viento**

| ID    | Nombre       | Temp | Humedad | Lluvia | Viento |
|-------|--------------|------|---------|--------|--------|
| 10111 | Entrimo      | SI   | SI      | SI     | NO     |
| 10113 | Monte Medo   | SI   | SI      | SI     | NO     |
| 10114 | As Petarelas | SI   | SI      | SI     | NO     |
| 10148 | Ourense      | SI   | SI      | SI     | NO     |

Estas 4 estaciones no tienen anemometro (sensor de viento). Las otras 16 tienen todas las variables completas.

**Decision**: Se mantienen las 4 estaciones en el dataset. Para los incendios asignados a estas estaciones, se usaran sus datos de temperatura/humedad/lluvia, pero el viento se obtendra de la estacion mas cercana que SI tenga anemometro (estrategia "Opcion B", implementada en el paso 8).

### Conclusion

Ambos datasets son de buena calidad y sirven para el cruce. Los datos de MeteoGalicia tienen una tasa de validacion del 99,5%, lo que indica un buen mantenimiento de las estaciones.

---

## Paso 7: Evaluacion previa a la limpieza y fusion

### Objetivo

Antes de ejecutar la limpieza y fusion, se simulo todo el proceso en modo lectura (sin modificar nada) para anticipar problemas y tomar decisiones informadas.

### Simulacion de limpieza de incendios

Se aplico la cadena de filtros prevista:

| Paso | Registros | Perdidos |
|---|---|---|
| CSV total | 292.181 | - |
| Filtro Ourense | 36.813 | - |
| Filtro desde 2005 | 7.720 | - |
| Sin coordenadas | 7.719 | 1 |
| Duplicados | 7.719 | 0 |

**Hallazgo importante: descarte del bounding box**. Se habia previsto un filtro por rectangulo geografico para eliminar coordenadas erroneas. La simulacion revelo que 347 registros caian fuera del rectangulo, pero al investigarlos resultaron ser municipios reales de Ourense situados en los limites irregulares de la provincia (Pinor, O Irixo, Carballeda de Valdeorras...). La provincia no es un rectangulo, asi que el filtro habria descartado datos validos. Decision: **no usar bounding box**. El filtro por `idprovincia == 32` ya es suficiente.

### Seleccion de las 12 variables meteorologicas

Se evaluaron las 14 variables disponibles por su disponibilidad (porcentaje de dias con dato valido):

| Variable | Descripcion | Disponibilidad | Decision |
|---|---|---|---|
| TA_AVG_1.5m | Temperatura media | 99,5% | Incluir |
| TA_MAX_1.5m | Temperatura maxima | 99,5% | Incluir |
| TA_MIN_1.5m | Temperatura minima | 99,5% | Incluir |
| HR_AVG_1.5m | Humedad relativa media | 96,2% | Incluir |
| HR_MIN_1.5m | Humedad relativa minima | 96,2% | Incluir |
| PP_SUM_1.5m | Precipitacion | 99,5% | Incluir |
| VV_AVG_10m | Velocidad viento media | 66,9% | Incluir (con Opcion B) |
| VV_MAX_10m | Racha maxima viento | 66,5% | Incluir (con Opcion B) |
| IRD_SUM_1.5m | Irradiacion global | 86,8% | Incluir |
| HSOL_SUM_1.5m | Horas de sol | 86,8% | Incluir |
| TO_AVG_1.5m | Temperatura de rocio | 95,7% | Incluir |
| PR_AVG_1.5m | Presion atmosferica | 62,9% | Incluir con cuidado |
| ET0_SUM_1.5m | Evapotranspiracion | 55,2% | **Descartar** |
| BH_SUM_1.5m | Balance hidrico | 58,7% | **Descartar** |

**Criterio de descarte**: Variables con disponibilidad inferior al 60% se descartan porque introducirian demasiados valores nulos, perjudicando la capacidad del modelo para aprender patrones. Se incluyen **12 variables**, se descartan 2.

### Simulacion de la fusion espacial-temporal

**Problema descubierto: coordenadas en sistemas distintos.**

- Los incendios tienen coordenadas en **lat/lon WGS84** (grados)
- Las estaciones tienen coordenadas en **UTM zona 29N** (metros)

Para calcular distancias, hay que convertir todo al mismo sistema. Se eligio UTM porque las distancias euclidianas en metros son exactas dentro de la misma zona UTM (mas preciso que la formula haversine para distancias cortas como las nuestras, de menos de 30 km).

**Distancia de cada incendio a la estacion mas cercana:**

| Estadistico | Distancia |
|---|---|
| Minima | 0,2 km |
| Mediana | 9,4 km |
| Media | 10,0 km |
| Percentil 95 | 18,8 km |
| Maxima | 25,9 km |

Una mediana de 9,4 km es buena cobertura espacial. Ningun incendio queda a mas de 26 km de una estacion. Para contexto: las variables meteorologicas como temperatura y humedad varian poco en distancias inferiores a 20-30 km (a altitud similar), asi que la aproximacion es razonable.

### Estrategia para estaciones sin viento (Opcion B)

Las 4 estaciones sin anemometro afectan a **925 incendios (12%)**. La estrategia:

1. Para cada incendio, se asigna la estacion mas cercana para todas las variables excepto viento
2. Si esa estacion no tiene anemometro, se busca la **segunda estacion mas cercana que SI mida viento**
3. La distancia mediana a esa estacion alternativa es 12,6 km (maximo 21,1 km)

Esto es aceptable porque el viento varia menos espacialmente que la precipitacion — una estacion a 12 km puede dar una estimacion razonable del viento.

### Plan de ejecucion validado

Con toda la simulacion completa y las decisiones tomadas, se definio el plan para el paso 8:

1. Limpiar incendios: filtrar Ourense >= 2005, eliminar sin coordenadas
2. Parsear los 20 JSONs meteorologicos en un DataFrame con las 12 variables
3. Filtrar por codigo de validacion, capar outliers, interpolar huecos aislados
4. Convertir coordenadas de incendios de lat/lon a UTM zona 29N
5. Asignar estacion mas cercana a cada incendio (con fallback de viento)
6. Merge por fecha + estacion
7. Guardar dataset fusionado

---

## Resumen: que se tiene al final del paso 7

Al terminar estos primeros pasos, el proyecto tiene:

- **7.719 incendios** de Ourense (2005-2021) con coordenadas y fecha exacta
- **106.928 registros meteorologicos** de 20 estaciones con 12 variables cada uno
- Un **plan detallado y validado** para limpiar, fusionar y construir el dataset de clasificacion
- Decisiones fundamentadas sobre variables a incluir, tratamiento de estaciones sin viento, y rechazo del filtro por bounding box

Todo esta listo para ejecutar el pipeline de limpieza y fusion (paso 8).
