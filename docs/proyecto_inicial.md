# Proyecto STEMbach: Predicción de incendios forestales en Galicia mediante IA

> **Nota**: Este documento es la **investigacion inicial** del proyecto (25 de marzo de 2026). Contiene la exploracion de fuentes de datos, endpoints de la API, listado de estaciones y plan metodologico original. El registro autoritativo de lo que realmente se ejecuto es `paso_a_paso.md`. Consultar este documento para referencia de endpoints, variables, estaciones y contexto del programa STEMbach.

## Contexto del alumno

- Estudiante de último curso de Bachillerato (STEMbach)
- Nivel Python: intermedio/avanzado
- Tiempo disponible: 1-2 meses
- Ubicación: Ourense, Galicia
- Tema sugerido por tutor: detectar incendios por condiciones climatológicas
- Proyecto individual con defensa pública ante tribunal universitario

---

## 1. Programa STEMbach — Qué es y qué se espera

### Descripción general

STEMbach (Bachillerato de Excelencia en Ciencias y Tecnología) es un programa de la Xunta de Galicia, enmarcado en la Estrategia Edudixital 2030, que promueve competencias STEM mediante investigación científica. Funciona desde el curso 2018/2019.

- Duración: 2 cursos académicos (bienio)
- Abierto a cualquier modalidad de Bachillerato
- En el curso 2024/2025: 53 centros participantes, más de 1.000 titulados
- El alumno debe desarrollar un **proyecto de investigación** codirigido por un docente universitario y titorizado por un profesor del centro
- El proyecto se defiende públicamente ante un tribunal

### Reconocimiento

- Hasta 6 créditos ECTS de libre configuración en universidades públicas gallegas (1.5 por curso)
- Reconocimiento retroactivo para titulados de cursos anteriores
- Los mejores proyectos (9 en toda Galicia, 3 por universidad) reciben premios en la Ciudad de la Cultura

### Universidades colaboradoras

| Universidad | Web STEMbach                                                      | Email              |
| ----------- | ----------------------------------------------------------------- | ------------------ |
| USC         | -                                                                 | stembach@usc.gal   |
| UDC         | https://www.udc.es/gl/stembach/                                   | stembach@udc.gal   |
| UVigo       | https://www.uvigo.gal/ven-uvigo/centros-secundaria/steam/stembach | stembach@uvigo.gal |

---

## 2. Criterios de evaluación del tribunal

### Distribución de la calificación (modelo UVigo, representativo)

| Componente                  | Peso | Descripción                                                    |
| --------------------------- | ---- | -------------------------------------------------------------- |
| **Exposición oral**         | 30%  | Máximo 15 min de exposición + 15 min de preguntas del tribunal |
| **Memoria escrita**         | 40%  | Documento formal del proyecto                                  |
| **Desarrollo del proyecto** | 30%  | Evaluado por el director universitario y el tutor del centro   |

### Qué valora el tribunal (rúbrica)

**En la exposición oral (30%):**

- Claridad al explicar la labor realizada
- Metodología aplicada
- Contenido y conclusiones
- Capacidad de responder a las preguntas del tribunal

**En la memoria escrita (40%):**

- Rigor metodológico
- Calidad de redacción y estructura
- Marco teórico fundamentado
- Análisis de resultados
- Conclusiones bien argumentadas

**En el desarrollo (30%):**

- Grado de dificultad del proyecto
- Autonomía del alumno
- Proceso de trabajo a lo largo del bienio

### Composición del tribunal

- Director/a del centro educativo (o delegado)
- Coordinador/a del programa STEMbach en el centro
- Una persona de la universidad (departamento/centro de investigación) que NO haya participado en la tutorización

### Fechas

- Defensa antes del 1 de mayo (UVigo) o 31 de mayo según la convocatoria
- Período lectivo, sesión pública

### Qué impresiona a un tribunal STEMbach

1. **Relevancia local**: Un proyecto que aplica a Galicia/Ourense tiene más peso que uno genérico
2. **Honestidad con las limitaciones**: Explicar por qué un modelo no funciona bien demuestra más madurez que presentar resultados inflados
3. **Metodología clara**: Que se entienda el proceso paso a paso
4. **Conexión con la realidad**: Vincular el trabajo con sistemas reales (p.ej. el IRDI de la Xunta usa el mismo índice FWI canadiense)
5. **Construcción propia de datos**: Si el alumno construye su propio dataset cruzando fuentes, eso eleva mucho el nivel percibido

---

## 3. Propuesta elegida: Dataset propio Galicia (incendios + meteorología)

### Idea central

Construir un dataset propio cruzando datos históricos de incendios forestales en Ourense/Galicia con datos meteorológicos diarios de MeteoGalicia, y entrenar un modelo de clasificación que prediga si un día con unas condiciones meteorológicas determinadas tiene riesgo de incendio.

### Por qué es una buena propuesta

- Ourense es la provincia más afectada por incendios en toda España (70 de los 100 municipios con más incendios entre 1983-2017 están en Galicia)
- El índice de riesgo diario de la Xunta (IRDI) usa el sistema canadiense FWI — mismo concepto que el proyecto replica a escala educativa
- La construcción del dataset es en sí misma una parte valiosa del trabajo (demuestra competencias en ciencia de datos)
- Los datos son 100% públicos y accesibles

---

## 4. Fuentes de datos — Exploración realizada

### 4.1 Datos de incendios: Civio (EGIF)

**Fuente**: Fundación Civio — "Todos los incendios forestales"
**URL**: https://datos.civio.es/dataset/todos-los-incendios-forestales/
**Formato**: CSV, 28.3 MB
**Rango temporal**: 1968 - 2023
**Origen real**: Estadística General de Incendios Forestales (EGIF) del Ministerio

**Descarga**: Requiere formulario manual (nombre + email). No se puede automatizar.
**Registro**: Lua Piñeiro Cabanelas (lua@cabanelas.info)
**Archivo descargado**: `fires-all.csv` (~31 MB)

**Campos disponibles:**

| Campo             | Descripción                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `id`              | Identificador del incendio                                        |
| `superficie`      | Superficie forestal quemada (hectáreas)                           |
| `fecha`           | Fecha de detección (formato yyyy-mm-dd) ← **CLAVE para el merge** |
| `lat`             | Latitud WGS84                                                     |
| `lng`             | Longitud WGS84                                                    |
| `latlng_explicit` | 1 = coordenadas reales, 0 = coordenadas del municipio             |
| `comunidad`       | Código CC.AA.                                                     |
| `provincia`       | Código provincia (32 = Ourense)                                   |
| `municipio`       | Nombre del municipio                                              |
| `causa`           | Código de causa                                                   |
| `causa_supuesta`  | 1 si la causa es supuesta                                         |
| `causa_desc`      | Código descripción de causa                                       |
| `muertos`         | Número de muertos                                                 |
| `heridos`         | Número de heridos                                                 |
| `time_ctrl`       | Tiempo hasta control (minutos)                                    |
| `time_ext`        | Tiempo hasta extinción (minutos)                                  |
| `personal`        | Personas en la extinción                                          |
| `medios`          | Medios terrestres y aéreos                                        |
| `gastos`          | Gastos de extinción (euros)                                       |
| `perdidas`        | Pérdidas económicas (euros)                                       |

**Tablas de códigos adicionales**: https://civio.box.com/s/6jcg8g9iku245jkrv2rmxfls3pknlpyt

**Limitaciones conocidas:**

- Geolocalización no especificada en ~54% de incendios (se usa ubicación del municipio)
- Causa supuesta en >76% de los casos
- Datos de gastos/pérdidas incompletos (~19% sin gastos, ~34% sin pérdidas)

### 4.2 Datos de incendios: IEPNB SPARQL (alternativa)

**Endpoint SPARQL**: https://datos.iepnb.es/sparql
**Datos verificados para Ourense**: 72.154 incendios (1983-2015)

**Incendios por año en Ourense (consulta SPARQL verificada):**

| Año  | Incendios | Año  | Incendios |
| ---- | --------- | ---- | --------- |
| 1983 | 182       | 2000 | 5.317     |
| 1985 | 920       | 2001 | 3.559     |
| 1989 | 1.615     | 2005 | 4.292     |
| 1994 | 3.542     | 2006 | 1.613     |
| 1995 | 3.605     | 2010 | 1.416     |
| 1996 | 3.994     | 2011 | 2.609     |
| 1998 | 4.431     | 2015 | 1.203     |

**Limitación**: Solo tiene año + municipio + causa + superficie. **NO tiene fecha exacta ni coordenadas**. Por eso no sirve para cruzar con meteorología diaria. Solo útil como referencia o para análisis agregado.

**Consulta SPARQL funcional para contar por año:**

```sparql
PREFIX inc: <https://datos.iepnb.es/def/sector-publico/medio-ambiente/incendios-forestales#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?year (COUNT(DISTINCT ?incendio) AS ?total)
WHERE {
  ?incendio a inc:IncendioForestal .
  ?incendio geo:sfWithin <http://datos.gob.es/recurso/sector-publico/territorio/Provincia/Ourense> .
  ?incendio prov:wasMemberOf ?yearUri .
  BIND(REPLACE(STR(?yearUri), ".*/", "") AS ?year)
}
GROUP BY ?year ORDER BY ?year
```

### 4.3 Datos meteorológicos: MeteoGalicia API

**Documentación oficial**: https://www.meteogalicia.gal/datosred/infoweb/meteo/docs/rss/JSON_EstacionsDiarios_es.pdf

**Endpoints verificados y funcionales:**

#### Listar todas las estaciones

```
GET https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action
```

Devuelve JSON con: `idEstacion`, `estacion` (nombre), `concello`, `provincia`, `lat`, `lon`, `altitude`

**Total estaciones**: 155 en toda Galicia

| Provincia  | Estaciones |
| ---------- | ---------- |
| A Coruña   | 36         |
| Lugo       | 42         |
| Ourense    | 34         |
| Pontevedra | 43         |

#### Datos diarios por estación y rango de fechas

```
GET https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action?idEst={ID}&dataIni={dd/MM/yyyy}&dataFin={dd/MM/yyyy}
```

**Parámetros opcionales:**

- `idEst`: ID de estación (separados por coma para varias)
- `idParam`: código de parámetro (separados por coma)
- `dataIni` / `dataFin`: rango de fechas en formato `dd/MM/yyyy`

#### Estaciones en Ourense (34 estaciones verificadas)

| ID    | Nombre              | Concello                 | Altitud |
| ----- | ------------------- | ------------------------ | ------- |
| 10148 | Ourense             | OURENSE                  | 143m    |
| 10155 | Ourense-Estacións   | OURENSE                  | 139m    |
| 10057 | Alto do Rodicio     | MACEDA                   | 981m    |
| 10109 | Amiudal             | AVIÓN                    | 553m    |
| 10110 | Baltar              | BALTAR                   | 807m    |
| 10112 | Gandarela           | CELANOVA                 | 623m    |
| 10113 | Monte Medo          | BAÑOS DE MOLGAS          | 604m    |
| 10058 | Verín-Vilamaior     | VERÍN                    | 546m    |
| 10500 | Verín-Vilela        | VERÍN                    | 392m    |
| 10116 | Viana do Bolo       | VIANA DO BOLO            | 851m    |
| 10131 | Cabeza de Manzaneda | MANZANEDA                | 1758m   |
| 10114 | As Petarelas        | RUBIÁ                    | 577m    |
| 10117 | San Xoán de Río     | SAN XOÁN DE RÍO          | 1026m   |
| 10130 | Lardeira            | CARBALLEDA DE VALDEORRAS | 1620m   |
| 10111 | Entrimo             | ENTRIMO                  | 763m    |
| 10119 | Xurés               | MUÍÑOS                   | 1059m   |
| 10048 | O Invernadeiro      | VILARIÑO DE CONSO        | 1026m   |
| 10115 | Serra do Eixe       | A VEIGA                  | 1229m   |
| 10138 | Xares               | A VEIGA                  | 1762m   |
| 19026 | EVEGA Leiro         | LEIRO                    | 105m    |
| 19027 | Xinzo               | XINZO DE LIMIA           | 616m    |
| 19030 | Larouco             | LAROUCO                  | 522m    |
| 19031 | Pazo de Fontefiz    | COLES                    | 403m    |
| 19032 | Prado               | CASTRELO DE MIÑO         | 122m    |
| 19033 | Ponte Boga          | CASTRO CALDELAS          | 469m    |
| 19037 | Corzos              | A VEIGA                  | 1217m   |
| 19038 | Laza                | LAZA                     | 575m    |
| 19039 | Riós                | RIÓS                     | 839m    |
| 19040 | A Trabe             | VILARDEVÓS               | 778m    |
| 19041 | A Mezquita          | A MEZQUITA               | 1025m   |
| 19042 | Calvos              | CALVOS DE RANDÍN         | 890m    |
| 19044 | Remuíño             | A ARNOIA                 | 120m    |
| 19073 | A Portela           | VILAMARTÍN DE VALDEORRAS | 408m    |
| 10204 | Casaio              | CARBALLEDA DE VALDEORRAS | 1280m   |

#### Variables disponibles por estación

**Estación completa (Ourense-Estacións, 10155) — 20 variables:**

| Código parámetro     | Descripción                       | Unidad        |
| -------------------- | --------------------------------- | ------------- |
| TA_AVG_1.5m          | Temperatura media                 | ºC            |
| TA_MAX_1.5m          | Temperatura máxima                | ºC            |
| TA_MIN_1.5m          | Temperatura mínima                | ºC            |
| HR_AVG_1.5m          | Humedad relativa media            | %             |
| HR_MAX_1.5m          | Humedad relativa máxima           | %             |
| HR_MIN_1.5m          | Humedad relativa mínima           | %             |
| PP_SUM_1.5m          | Precipitación (lluvia)            | L/m²          |
| VV_AVG_10m           | Velocidad del viento media        | m/s           |
| VV_MAX_10m           | Racha máxima de viento            | m/s           |
| DV_CONDICION_10m     | Dirección de la racha             | º             |
| DVP_MODA_10m         | Dirección del viento predominante | º             |
| PR_AVG_1.5m          | Presión atmosférica media         | hPa           |
| PRED_AVG_1.5m        | Presión reducida al nivel del mar | hPa           |
| IRD_SUM_1.5m         | Irradiación global diaria         | 10kJ/(m².día) |
| HSOL_SUM_1.5m        | Horas de sol                      | h             |
| INS_RATIO_1.5m       | Insolación (ratio)                | %             |
| ET0_SUM_1.5m         | Evapotranspiración de referencia  | L/m²          |
| BH_SUM_1.5m          | Balance hídrico                   | L/m²          |
| TO_AVG_1.5m          | Temperatura de rocío              | ºC            |
| HFRIO7_RECUENTO_1.5m | Horas de frío (≤7ºC)              | h             |

**Estación agrometeorológica (Pazo de Fontefiz, 19031) — 24 variables (añade):**

| Código parámetro | Descripción                    | Unidad |
| ---------------- | ------------------------------ | ------ |
| HS_CV_AVG_-0.2m  | Humedad del suelo              | m³/m³  |
| TS_AVG_-0.1m     | Temperatura del suelo          | ºC     |
| HF_SUM_2m        | Horas de humedad foliar diaria | h      |
| TA_AVG_0.1m      | Temperatura media a 0.1m       | ºC     |

**Código de validación de datos:**

| Código | Significado             |
| ------ | ----------------------- |
| 0      | Dato sin validar        |
| 1      | Dato válido original    |
| 2      | Dato sospechoso         |
| 3      | Dato erróneo            |
| 5      | Dato válido interpolado |
| 9      | Dato no registrado      |

#### Disponibilidad temporal verificada

| Estación               | ID    | Datos desde               |
| ---------------------- | ----- | ------------------------- |
| Alto do Rodicio        | 10057 | ~2005                     |
| Ourense                | 10148 | ~2012                     |
| Ourense-Estacións      | 10155 | ~2014                     |
| Estaciones serie 19xxx | 19xxx | Varía, muchas desde ~2017 |

**Importante**: Las estaciones 10xxx (serie antigua) suelen tener datos desde ~2005. Las 19xxx (serie nueva) desde ~2015-2017. Para maximizar el solapamiento con los datos de incendios, priorizar estaciones 10xxx.

#### Ejemplo de respuesta JSON (datos diarios)

```json
{
  "listDatosDiarios": [
    {
      "data": "2023-07-01T00:00:00",
      "listaEstacions": [
        {
          "estacion": "Ourense",
          "idEstacion": 10148,
          "concello": "OURENSE",
          "provincia": "Ourense",
          "utmx": "...",
          "utmy": "...",
          "listaMedidas": [
            {
              "codigoParametro": "TA_AVG_1.5m",
              "nomeParametro": "Temperatura media a 1.5m",
              "lnCodigoValidacion": 1,
              "unidade": "ºC",
              "valor": 23.19
            }
          ]
        }
      ]
    }
  ]
}
```

### 4.4 Dato relevante: AEMET y el índice FWI

La AEMET publica estadísticas del índice meteorológico de riesgo de incendios (FWI) por provincia:
https://www.aemet.es/es/datos_abiertos/estadisticas/riesgo_incendios

El IRDI (Índice de Riesgo Diario de Incendio) de la Xunta de Galicia también se basa en el sistema canadiense FWI. Variables necesarias para calcularlo: temperatura, humedad relativa, velocidad del viento, precipitación. **Todas disponibles en MeteoGalicia**.

Esto es relevante para el marco teórico: el alumno puede explicar que su modelo intenta replicar, con machine learning, lo que el sistema FWI calcula con fórmulas empíricas.

---

## 5. Metodología de construcción del dataset

### Fase 1: Obtención de datos (2-3 días)

**Datos de incendios (Civio CSV):**

1. Cargar CSV con pandas
2. Filtrar: `provincia == 32` (Ourense) o toda Galicia
3. Filtrar: fecha >= 2005 (para solapar con MeteoGalicia)
4. Limpiar fechas a formato datetime

**Datos meteorológicos (MeteoGalicia API):**

1. Obtener lista de estaciones de Ourense
2. Para cada estación, descargar datos diarios desde su fecha de inicio hasta 2023
3. Llamadas al API en rangos mensuales para no sobrecargar
4. Parsear JSON y consolidar en un DataFrame de pandas

### Fase 2: Limpieza de cada fuente (2-3 días)

**Incendios:**

- Convertir `fecha` a datetime
- Convertir `lat`, `lng` a float
- Filtrar incendios sin coordenadas o con coordenadas fuera de Galicia
- Eliminar duplicados por `id`

**Meteorología:**

- Filtrar datos con `lnCodigoValidacion` en [1, 5] (válidos)
- Descartar datos con código 3 (erróneo) o 9 (no registrado)
- Detectar outliers por rangos físicos razonables:
  - Temperatura: -15°C a 50°C
  - Humedad relativa: 0% a 100%
  - Precipitación: >= 0
  - Velocidad viento: >= 0
- Interpolar valores faltantes aislados con `df.interpolate(method='time')`

### Fase 3: Fusión espacial-temporal (3-4 días)

**Paso clave: asignar estación meteorológica más cercana a cada incendio**

1. Para cada incendio (lat, lon), calcular distancia haversine a todas las estaciones de MeteoGalicia
2. Asignar la estación más cercana
3. Hacer merge por fecha (`incendio.fecha == meteo.fecha`) y estación asignada
4. Opcionalmente añadir features derivadas:
   - Precipitación acumulada últimos 7 días
   - Temperatura media últimos 3 días
   - Días consecutivos sin lluvia

```python
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))
```

### Fase 4: Generar "no-incendios" (1-2 días)

Solo tenemos días CON incendio. Para clasificación binaria necesitamos también días SIN incendio.

**Estrategia de muestreo de negativos:**

1. Para cada estación, identificar todos los días del período donde NO hubo ningún incendio cercano
2. Muestrear aleatoriamente N días sin incendio (ratio 1:1 o 1:2 con los positivos)
3. Para esos días, obtener los datos meteorológicos correspondientes
4. Etiquetar como clase `no_fire`

### Fase 5: Modelado (1 semana)

**Herramientas**: scikit-learn, pandas, matplotlib/seaborn

**Pipeline:**

1. Train/test split (80/20) estratificado
2. Entrenar varios modelos de clasificación:
   - Logistic Regression (baseline)
   - Random Forest
   - KNN
   - SVM (si el dataset no es demasiado grande)
3. Evaluar con métricas relevantes:
   - **Recall** (prioridad: no queremos falsos negativos = incendios no detectados)
   - Precision
   - F1-score
   - Matriz de confusión
   - Curva ROC / AUC
4. Analizar **feature importance** (qué variables meteorológicas predicen mejor)
5. Comparar modelos y seleccionar el mejor

### Fase 6: Memoria y presentación (última semana)

**Estructura sugerida de la memoria:**

1. Introducción y motivación (Ourense = provincia más afectada)
2. Marco teórico (sistema FWI, IRDI, machine learning basics)
3. Fuentes de datos y obtención
4. Metodología de limpieza y fusión
5. Análisis exploratorio (EDA) con gráficas
6. Modelado y resultados
7. Conclusiones y limitaciones
8. Bibliografía

---

## 6. Estructura de archivos del proyecto

```
stembach_incendios/
├── data/
│   ├── raw/
│   │   ├── fires-all.csv                # CSV descargado de Civio (292.181 incendios)
│   │   └── meteogalicia/                # 20 JSONs descargados de la API
│   └── processed/
│       ├── incendios_ourense.csv        # 7.719 incendios filtrados y limpios
│       ├── meteo_ourense.csv            # 106.928 dias meteorologicos limpios
│       ├── estaciones_ourense.csv       # Coordenadas UTM de las 20 estaciones
│       └── dataset_fusionado.csv        # Incendios + meteorologia cruzados
├── src/
│   ├── config.py                        # Constantes compartidas del proyecto
│   ├── descargar_meteo.py               # Descarga masiva API MeteoGalicia
│   ├── limpiar_incendios.py             # Limpieza CSV de incendios
│   ├── limpiar_meteo.py                 # Parseo JSONs + limpieza meteorologica
│   └── fusionar_datasets.py             # Conversion UTM + asignacion + merge
├── memoria/                             # Documentos explicativos para la defensa
├── _copiaseguridad/                     # Backups (zip con timestamp)
├── STEMBACH_PROYECTO_IA_INCENDIOS.md    # Documento maestro (investigacion)
├── paso_a_paso.md                       # Diario del proyecto
├── CLAUDE.md                            # Contexto para agentes IA
├── README.md
└── requirements.txt
```

---

## 7. Dependencias Python

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
requests>=2.31
jupyter>=1.0
```

---

## 8. Referencias y enlaces

### Datos

- Civio - Todos los incendios forestales: https://datos.civio.es/dataset/todos-los-incendios-forestales/
- MeteoGalicia API (datos diarios): https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action
- MeteoGalicia (lista estaciones): https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action
- MeteoGalicia documentación API JSON: https://www.meteogalicia.gal/datosred/infoweb/meteo/docs/rss/JSON_EstacionsDiarios_es.pdf
- MeteoGalicia parámetros: https://www.meteogalicia.gal/web/observacion/parametros
- IEPNB SPARQL: https://datos.iepnb.es/sparql
- IGE - Incendios por distrito forestal: https://www.ige.gal/igebdt/selector.jsp?idioma=es&COD=3726
- AEMET OpenData: https://opendata.aemet.es/
- AEMET estadísticas riesgo incendios (FWI): https://www.aemet.es/es/datos_abiertos/estadisticas/riesgo_incendios
- EGIF buscador público: https://servicio.mapa.gob.es/incendios/Search/Publico
- Civio - España en Llamas (mapa): https://civio.es/espana-en-llamas/mapa-de-incendios-forestales/

### STEMbach

- Xunta de Galicia - STEMbach: https://www.edu.xunta.gal/portal/es/node/27960
- UVigo STEMbach (criterios evaluación): https://www.uvigo.gal/es/ven-uvigo/centros-secundaria/steam/stembach/stembach-uvigo
- UDC STEMbach: https://www.udc.es/es/stembach/stembach-na-udc/
- Resolución reguladora 2024/25: DOG, Resolución do 7 de xuño de 2024

### Referencia científica

- Cortez, P. & Morais, A. (2007). A Data Mining Approach to Predict Forest Fires using Meteorological Data.
- UCI Forest Fires Dataset (Portugal): https://archive.ics.uci.edu/ml/datasets/forest+fires
- UCI Algerian Forest Fires Dataset: https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
- IRDI Xunta de Galicia (basado en FWI canadiense): https://mediorural.xunta.gal/es/temas/defensa-monte/irdi
- Existe un paquete Python en PyPI: MeteoGalicia-API (v0.1.1)

### Contexto

- Galicia acumula ~49% de la superficie quemada intencionadamente en España (1983-2017)
- Ourense fue la provincia más afectada
- 70 de los 100 municipios con más incendios en España están en Galicia
- Fuente: Civio / EGIF

---

## 9. Notas técnicas para Claude Code

### Archivo de incendios Civio

- **Archivo**: `fires-all.csv`
- **Tamaño**: ~31 MB
- **Codificación**: probablemente UTF-8
- **Separador**: probablemente coma (verificar)

### API MeteoGalicia — Notas prácticas

- **No requiere API key**
- **Formato fechas**: `dd/MM/yyyy` (ojo: día/mes/año, NO ISO)
- **Sin límite aparente** de peticiones, pero hacer pausas entre llamadas (1-2 seg)
- **Rango recomendado por petición**: 1 mes (no exceder rangos muy largos)
- **Las estaciones serie 10xxx tienen datos más antiguos** (~2005) que las 19xxx (~2015-2017)
- **Dato vacío / sin registrar**: si `valor` es null o `lnCodigoValidacion` = 9, no hay dato para ese día

### Prioridades de implementación

1. ~~Descargar CSV de Civio~~ (HECHO: `fires-all.csv`)
2. Explorar el CSV de Civio y verificar su estructura
3. Descargar datos de MeteoGalicia para las estaciones principales
4. Fusión espacial-temporal
5. EDA y modelado
6. Memoria y presentación

### Datasets alternativos (plan B si el merge sale mal)

- **UCI Algerian Forest Fires** (244 instancias, clasificación binaria fire/not-fire): https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset
- **UCI Forest Fires Portugal** (517 instancias, regresión de área quemada): https://archive.ics.uci.edu/ml/datasets/forest+fires
- Ambos son datasets limpios y listos para usar si la construcción del dataset propio consume demasiado tiempo

---

*Documento generado el 25 de marzo de 2026 a partir de investigación realizada en Claude.ai*
*Uso: contexto completo para continuar el desarrollo del proyecto con Claude Code*
