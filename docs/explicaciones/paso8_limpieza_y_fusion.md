# Paso 8: Limpieza y fusion de datos — Explicacion detallada

> Este documento explica, sin asumir conocimientos previos, todo lo que se hizo en el paso 8 del proyecto: tomar los datos "en bruto" de incendios y de meteorologia, limpiarlos, y combinarlos en un unico fichero listo para entrenar un modelo de inteligencia artificial.

---

## Que teniamos antes de empezar

Despues de los pasos anteriores del proyecto, teniamos dos fuentes de datos completamente separadas:

**1. Un fichero de incendios** (`fires-all.csv`, 31 MB):
- Descargado de la Fundacion Civio, que recopila datos oficiales del Ministerio
- Contiene **292.181 incendios forestales** de toda España desde 1968 hasta 2023
- Cada incendio tiene: fecha, coordenadas geograficas (latitud y longitud), provincia, municipio, superficie quemada, causa, etc.

**2. Veinte ficheros de meteorologia** (286 MB en total):
- Descargados de la API de MeteoGalicia, el servicio meteorologico de la Xunta de Galicia
- Cada fichero corresponde a una estacion meteorologica de la provincia de Ourense
- Contienen datos diarios: temperatura, humedad, lluvia, viento, etc.
- Rango: desde 2005-2006 hasta 2021 (depende de cuando se instalo cada estacion)

**El problema**: Estos dos conjuntos de datos no estan conectados. Un incendio dice "ocurrio el 15 de agosto de 2017 en estas coordenadas", y la meteorologia dice "en esta estacion, el 15 de agosto de 2017 la temperatura fue de 35 grados". Pero no hay nada que los una. El objetivo del paso 8 es **unirlos**: que cada incendio tenga asociadas las condiciones meteorologicas de ese dia.

---

## Por que es necesario limpiar los datos

Los datos del mundo real nunca vienen perfectos. Antes de combinarlos, hay que prepararlos:

- El fichero de incendios tiene datos de toda España, pero solo nos interesa Ourense
- Algunos registros pueden tener errores: fechas mal escritas, coordenadas que faltan, registros duplicados
- Los datos meteorologicos tienen sensores que a veces fallan: un dia no se registro la temperatura, o el sensor de humedad dio un valor imposible (como 105% de humedad, que no puede existir)
- Los sistemas de validacion de MeteoGalicia marcan cada dato con un codigo que indica si es fiable o no

Si entrenamos un modelo de inteligencia artificial con datos sucios o erroneos, el modelo aprendera patrones falsos y sus predicciones seran malas. La limpieza es un paso fundamental en cualquier proyecto de ciencia de datos.

---

## Paso 8.1 — Limpieza del fichero de incendios

### Que se hizo

Se tomo el fichero original con 292.181 incendios de toda España y se fue filtrando paso a paso hasta quedarnos solo con los que nos interesan y son validos:

**Filtro 1 — Solo Ourense**: De los 292.181 incendios, nos quedamos solo con los de la provincia de Ourense (codigo de provincia 32). Esto deja 36.813 incendios.

**Filtro 2 — Solo desde 2005**: Las estaciones meteorologicas de MeteoGalicia empezaron a funcionar en torno a 2005-2006. No tiene sentido conservar incendios de 1968 si no tenemos datos del tiempo de esos años. Nos quedamos con los 7.720 incendios desde el 1 de enero de 2005.

**Filtro 3 — Coordenadas validas**: Para poder cruzar un incendio con una estacion meteorologica, necesitamos saber *donde* ocurrio exactamente. De los 7.720 incendios, solo 1 no tenia coordenadas. Se elimino.

**Filtro 4 — Duplicados**: Se comprobo si habia incendios repetidos (mismo identificador). No habia ninguno.

**Seleccion de columnas**: El fichero original tenia 20 columnas, pero muchas no son utiles para nuestro modelo. Por ejemplo, los "gastos de extincion" o el "numero de medios aereos" son datos que se conocen *despues* del incendio, no antes. Nuestro modelo quiere predecir *antes* de que pase, asi que solo conservamos las columnas relevantes:

- `id`, `fecha`, `lat`, `lng` — para identificar y localizar el incendio
- `municipio`, `superficie` — informacion geografica y de magnitud
- `causa`, `causa_supuesta` — por si queremos analizar causas mas adelante
- `latlng_explicit` — indica si las coordenadas son exactas o aproximadas

### Resultado

Un fichero limpio con **7.719 incendios** en Ourense, desde 2005 hasta 2021, todos con coordenadas validas y sin duplicados.

### Decision importante: por que no se uso un "bounding box"

Inicialmente se habia previsto descartar incendios cuyas coordenadas cayeran fuera de un rectangulo que encierra la provincia de Ourense. La idea era detectar errores en las coordenadas.

Sin embargo, al probarlo, 347 incendios caian fuera del rectangulo. Al investigarlos, resulto que eran incendios reales en municipios como Piñor, O Irixo o San Cristovo de Cea, que estan en los bordes de la provincia. El problema es que una provincia no tiene forma de rectangulo — tiene bordes irregulares — y un rectangulo no la representa bien. Se descarto este filtro para no perder datos validos.

---

## Paso 8.2 — Parseo de los datos meteorologicos

### Que es "parsear"

Los datos de MeteoGalicia estan guardados en formato JSON, que es una forma de estructurar informacion usada en internet. Cada fichero JSON tiene una estructura anidada (datos dentro de datos dentro de datos). "Parsear" significa leer esa estructura y extraer solo la informacion que necesitamos, organizandola en una tabla plana con filas y columnas.

### Que se hizo

Se leyeron los 20 ficheros JSON (uno por estacion meteorologica) y se extrajeron 12 variables para cada dia:

| Variable | Que mide | Unidad |
|---|---|---|
| TA_AVG_1.5m | Temperatura media del dia | °C |
| TA_MAX_1.5m | Temperatura maxima del dia | °C |
| TA_MIN_1.5m | Temperatura minima del dia | °C |
| HR_AVG_1.5m | Humedad relativa media | % |
| HR_MIN_1.5m | Humedad relativa minima | % |
| PP_SUM_1.5m | Precipitacion total (lluvia) | Litros por metro cuadrado |
| VV_AVG_10m | Velocidad media del viento | Metros por segundo |
| VV_MAX_10m | Racha maxima de viento | Metros por segundo |
| IRD_SUM_1.5m | Irradiacion solar del dia | Energia por superficie |
| HSOL_SUM_1.5m | Horas de sol | Horas |
| TO_AVG_1.5m | Temperatura de rocio | °C |
| PR_AVG_1.5m | Presion atmosferica media | hPa (hectopascales) |

**Por que estas 12 y no otras**: En el paso 7 (evaluacion previa) se analizaron todas las variables disponibles. Se incluyeron las que tenian al menos un 60% de disponibilidad. Dos variables (evapotranspiracion y balance hidrico) se descartaron porque solo estaban disponibles en el 55-58% de los casos — demasiados huecos.

**El "1.5m" y "10m" en los nombres** se refieren a la altura del sensor sobre el suelo. Los termometros e higrometros (sensores de humedad) se colocan a 1,5 metros dentro de una garita. Los anemometros (sensores de viento) se colocan a 10 metros de altura, donde el viento no esta bloqueado por arboles ni edificios.

### Filtrado por codigo de validacion

MeteoGalicia marca cada dato con un codigo que indica su fiabilidad:

| Codigo | Significado | Que hicimos con el |
|---|---|---|
| 1 | Dato valido original | **Aceptado** |
| 5 | Dato valido, interpolado por MeteoGalicia | **Aceptado** |
| 0 | Sin validar | Descartado |
| 2 | Sospechoso | Descartado |
| 3 | Erroneo | Descartado |
| 9 | No registrado | Descartado |

De 1.165.763 medidas evaluadas, **1.129.123 (96,9%) pasaron el filtro**. Solo el 3,1% se descarto. Esto indica que los datos de MeteoGalicia son de buena calidad.

### Resultado

Una tabla con **106.928 filas** (una fila por cada dia y estacion) y 14 columnas (fecha, identificador de estacion, y las 12 variables meteorologicas).

---

## Paso 8.3 — Deteccion de outliers e interpolacion

### Que es un outlier

Un outlier (valor atipico) es un dato que esta fuera de lo fisicamente posible o razonable. Por ejemplo, una temperatura de 200°C en Ourense, o una humedad relativa del 150%. Estos valores, aunque hayan pasado el filtro de validacion de MeteoGalicia, son errores del sensor.

### Que se hizo

Se definieron rangos razonables para cada variable y se buscaron valores fuera de ellos:

| Variable | Rango aceptado | Outliers encontrados |
|---|---|---|
| Temperaturas | -15°C a 50°C | 0 |
| Humedad relativa | 0% a 100% | 0 |
| Precipitacion | 0 a 500 L/m² | 0 |
| Velocidad viento | 0 a 50 m/s | 0 |
| Irradiacion | 0 a 4.000 | 1 |
| Temperatura de rocio | -25°C a 40°C | 6 |
| Presion atmosferica | 750 a 1.080 hPa | 0 |

**Solo se encontraron 7 outliers** en mas de un millon de medidas. Los datos son muy fiables.

Los outliers no se eliminaron (perderiamos todos los datos de ese dia), sino que se "caparon": si un valor era mayor que el maximo aceptado, se fijo al maximo; si era menor que el minimo, al minimo. Esto es mas conservador que borrar el registro entero.

**Nota sobre la presion atmosferica**: Inicialmente se habia puesto un limite inferior de 870 hPa (presion normal al nivel del mar). Pero al probarlo, miles de valores de estaciones de montana saltaban como "outliers". La razon: la presion atmosferica baja con la altitud. En Cabeza de Manzaneda (1.758 metros sobre el nivel del mar), la presion media es de 826 hPa, muy por debajo de 870. Se ajusto el limite a 750 hPa para no descartar datos validos de las estaciones de alta montana.

### Interpolacion de huecos

A veces un sensor falla un dia aislado: tiene datos el lunes y el miercoles, pero no el martes. En estos casos se aplico **interpolacion lineal**: se calcula el valor del martes como el promedio entre el lunes y el miercoles. Esto tiene sentido fisico porque la temperatura o la humedad no cambian bruscamente de un dia a otro.

**Limites de seguridad**:
- Solo se rellenaron huecos de **maximo 2 dias consecutivos**. Si faltan 3 o mas dias seguidos, probablemente el sensor estaba averiado y no tiene sentido inventar datos.
- La interpolacion se hizo **dentro de cada estacion**. Nunca se mezclan datos de estaciones distintas.

**Resultado**: Se rellenaron **7.684 huecos** aislados, manteniendo la integridad de los datos.

---

## Paso 8.4 — Conversion de coordenadas

### El problema

Los incendios y las estaciones meteorologicas usan sistemas de coordenadas distintos:

- **Incendios**: latitud y longitud (el sistema del GPS, que mide angulos sobre la esfera terrestre)
- **Estaciones MeteoGalicia**: UTM zona 29N (un sistema que mide distancias en metros sobre un plano)

Para saber que estacion esta mas cerca de cada incendio, necesitamos que ambos esten en el mismo sistema.

### Que es UTM

Imagina que pelas una naranja e intentas aplanar la piel sobre una mesa. Es imposible hacerlo sin deformarla, porque una esfera no se puede convertir en un plano sin distorsion. Los sistemas de coordenadas son formas matematicas de hacer esa conversion minimizando la deformacion.

**UTM** (Universal Transverse Mercator) divide el mundo en 60 zonas verticales. Galicia esta en la zona 29N. Dentro de cada zona, las coordenadas se miden en metros: el eje X mide la distancia este-oeste, y el eje Y la distancia norte-sur. La ventaja es que calcular la distancia entre dos puntos es tan simple como aplicar el teorema de Pitagoras:

```
distancia = raiz cuadrada de ((X2-X1)² + (Y2-Y1)²)
```

Esto da la distancia en metros, directamente, sin necesidad de trigonometria esferica ni formulas complicadas.

### Que se hizo

Se convirtieron las 7.719 coordenadas de incendios de latitud/longitud a UTM zona 29N usando las formulas matematicas exactas de la proyeccion (publicadas por Snyder en 1987 para el servicio geologico de Estados Unidos). La conversion se implemento directamente en Python, sin depender de librerias externas.

---

## Paso 8.5 — Asignacion de estacion meteorologica a cada incendio

Este es el paso mas importante y el que requirio mas ingenieria.

### La idea basica

Para cada incendio, necesitamos saber: ¿que tiempo hacia ese dia en esa zona? La respuesta mas fiable viene de la **estacion meteorologica mas cercana**. Asi que para cada incendio:

1. Se calcula la distancia desde el punto del incendio hasta cada una de las 20 estaciones
2. Se ordenan de mas cercana a mas lejana
3. Se buscan los datos meteorologicos de la estacion mas cercana para la fecha del incendio

### El problema que surgio

En la primera version, simplemente se asignaba la estacion mas cercana. Pero **1.933 incendios (25%) se quedaron sin datos meteorologicos**. ¿Por que?

Porque no todas las estaciones funcionaban en todas las fechas:

- La mayoria de estaciones empezaron a medir en 2006, pero tenemos incendios desde 2005
- Algunas estaciones (como Ourense-Estacións) no empezaron hasta 2015
- Algunos dias concretos, un sensor podia estar averiado

Ejemplo concreto: Imaginemos un incendio el 3 de marzo de 2005 cerca de Celanova. La estacion mas cercana es Gandarela (10112), pero esta estacion no empezo a funcionar hasta 2006. Al buscar datos del 3 de marzo de 2005 en Gandarela, no hay nada. El incendio se queda sin datos.

### La solucion: estrategia de fallback

"Fallback" significa "plan alternativo". En vez de rendirse cuando la estacion mas cercana no tiene datos, el programa prueba la siguiente:

1. ¿La estacion mas cercana tiene datos para esta fecha? Si -> usar esos datos
2. ¿No? Probar la 2a estacion mas cercana
3. ¿Tampoco? Probar la 3a, 4a, 5a... hasta encontrar una que si tenga datos

Con esta estrategia, **el 100% de los incendios encontraron datos meteorologicos**:

- El 75% uso la estacion mas cercana (la ideal)
- El 7,2% uso la 2a mas cercana
- El 6% uso la 3a mas cercana
- El 11,8% restante uso estaciones mas lejanas (hasta la 10a)

La distancia mediana fue de 10,8 km entre el incendio y su estacion asignada. En el peor caso, la distancia fue de 64,4 km. Aunque no es ideal, tener datos meteorologicos aproximados es mucho mejor que no tener datos.

### El caso especial del viento

4 de las 20 estaciones no tienen anemometro (el instrumento que mide la velocidad del viento). Si un incendio se asigna a una de esas estaciones, obtiene todos los datos (temperatura, humedad, lluvia...) excepto el viento.

Para resolver esto, se aplico un proceso separado: para los incendios asignados a estaciones sin anemometro, se busco la estacion mas cercana **que si midiera el viento** en esa fecha, y se tomaron solo los datos de viento de ahi. Asi, un incendio puede tener la temperatura de una estacion cercana y el viento de otra un poco mas lejana. Esto es razonable porque el viento varia menos que la lluvia de un punto a otro.

---

## Paso 8.6 — El merge (la union de los datos)

### Que es un merge

Un merge (union o cruce) en ciencia de datos es como buscar en dos tablas distintas y pegar la informacion. Es similar a lo que haria un humano con dos hojas de calculo:

- **Tabla A** (incendios): "El 15 de agosto de 2017 hubo un incendio cerca de la estacion 10116"
- **Tabla B** (meteorologia): "En la estacion 10116, el 15 de agosto de 2017 hizo 35°C, 22% de humedad, 0 litros de lluvia, viento de 8 m/s..."

El merge busca las filas que coinciden en fecha y estacion, y las junta en una sola fila:

"El 15 de agosto de 2017 hubo un incendio cerca de la estacion 10116, donde hizo 35°C, 22% de humedad, 0 litros de lluvia, viento de 8 m/s..."

### Resultado

Cada uno de los 7.719 incendios tiene ahora asociadas 12 variables meteorologicas del dia en que ocurrio. Las temperaturas y la precipitacion tienen cobertura del 100%. Otras variables como el viento (84,6%) o las horas de sol (74,6%) tienen algunos huecos en dias/estaciones concretas donde ese sensor no estaba disponible.

---

## Paso 8.7 — El dataset fusionado

El resultado final de todo este proceso es un unico fichero CSV (`dataset_fusionado.csv`) con la siguiente estructura:

### Columnas del dataset (25 en total)

**Informacion del incendio (8 columnas):**
- `id` — Identificador unico del incendio
- `fecha` — Dia en que se detecto
- `lat`, `lng` — Coordenadas donde ocurrio
- `municipio` — Municipio de Ourense donde ocurrio
- `superficie` — Hectareas quemadas
- `causa` — Codigo de la causa (rayo, intencionado, negligencia, etc.)
- `causa_supuesta` — Si la causa es confirmada o solo sospechada

**Informacion de la asignacion espacial (5 columnas):**
- `estacion_id` — Que estacion meteorologica se uso
- `dist_estacion_m` — A cuantos metros estaba esa estacion del incendio
- `estacion_rank` — Si fue la 1a, 2a, 3a... estacion mas cercana
- `estacion_viento_id` — Que estacion se uso para el viento (puede ser distinta)
- `dist_viento_m` — A cuantos metros estaba la estacion de viento

**Datos meteorologicos del dia (12 columnas):**
- Temperatura: media, maxima, minima
- Humedad relativa: media, minima
- Precipitacion (lluvia)
- Viento: velocidad media, racha maxima
- Irradiacion solar, horas de sol
- Temperatura de rocio
- Presion atmosferica

### Numeros clave

| Metrica | Valor |
|---|---|
| Registros | 7.719 incendios |
| Periodo | 2005 a 2021 (17 años) |
| Municipios | 93 de la provincia de Ourense |
| Estaciones meteorologicas usadas | 20 |
| Tamano del fichero | 1,40 MB |
| Tasa de cruce exitoso | 100% |

---

## Resumen visual del proceso completo

```
fires-all.csv                    20 x estacion_XXXXX.json
(292.181 incendios, España)      (106.928 dias de meteorologia)
         |                                   |
    [Filtrar Ourense]                  [Parsear JSON]
    [Filtrar >= 2005]                  [Filtrar codigo validacion]
    [Eliminar sin coords]              [Capar outliers]
    [Seleccionar columnas]             [Interpolar huecos]
         |                                   |
         v                                   v
  incendios_ourense.csv            meteo_ourense.csv
  (7.719 incendios)                (106.928 dias x 12 variables)
         |                                   |
         |     [Convertir coords a UTM]      |
         |     [Calcular distancias]         |
         |     [Asignar estacion + fallback] |
         |     [Buscar estacion viento]      |
         |                                   |
         +------ MERGE por fecha + estacion -+
                          |
                          v
               dataset_fusionado.csv
               (7.719 filas x 25 columnas)
               Cada incendio + su meteorologia
```

---

## Que falta

Este dataset contiene solo dias en los que **SI hubo incendio**. Para entrenar un modelo de clasificacion (que prediga si un dia tiene riesgo de incendio o no), tambien necesitamos ejemplos de dias en los que **NO hubo incendio** pero tenemos los datos meteorologicos. Generar esos ejemplos negativos y crear variables adicionales (como la lluvia acumulada en los ultimos 7 dias, o cuantos dias consecutivos lleva sin llover) sera el proximo paso del proyecto.
