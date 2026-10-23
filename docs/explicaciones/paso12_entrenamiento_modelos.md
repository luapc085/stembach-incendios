# Paso 12: Entrenamiento de modelos — Explicacion detallada

> Este documento explica, sin asumir conocimientos previos, todo el proceso de entrenamiento de modelos de machine learning sobre nuestro dataset de incendios. Cada termino tecnico se define la primera vez que aparece.

---

## Que tenemos antes de empezar

### El dataset: una tabla de numeros

Nuestro fichero `dataset_modelo.csv` es una tabla de **92.844 filas y 25 columnas**. En machine learning, a esta tabla se le llama **dataset** o **matriz de datos**.

- Cada **fila** es una observacion: un dia concreto en una estacion meteorologica concreta. Por ejemplo, la fila "1 de agosto de 2017, estacion Viana do Bolo" contiene las condiciones meteorologicas de ese dia en ese lugar.
- Cada **columna** es una variable. Hay tres tipos:
  - **Metadatos** (3 columnas): `fecha`, `estacion_id`, `incendio` (la etiqueta)
  - **Variables meteorologicas** (12 columnas): temperatura media, maxima, minima, humedad, lluvia, viento, irradiacion, horas de sol, punto de rocio, presion
  - **Variables derivadas** (10 columnas): calculadas a partir de las anteriores — dias consecutivos sin lluvia, lluvia acumulada en 7/14/30 dias, tendencias de temperatura y humedad, amplitud termica, mes y dia del año

### Features y target

En machine learning se usan dos terminos constantemente:

- **Features** (o **X**): las columnas que el modelo usa como entrada para hacer su prediccion. En nuestro caso, las 22 columnas meteorologicas y derivadas. Son las "pistas" que el modelo recibe.
- **Target** (o **y**): la columna que queremos predecir. En nuestro caso, la columna `incendio`, que vale 1 (hubo incendio) o 0 (no hubo incendio).

El modelo nunca ve la columna `fecha` ni `estacion_id` como features — esas son solo para identificar la fila. Lo que el modelo recibe son 22 numeros (temperatura, humedad, lluvia...) y tiene que decidir: ¿incendio o no incendio?

---

## Division de los datos: train y test

### Por que no podemos usar todos los datos para entrenar

Imaginemos que un estudiante estudia para un examen con un libro de 100 preguntas y respuestas. El dia del examen, le ponen las mismas 100 preguntas. Saca un 10. ¿Ha aprendido la materia o simplemente ha memorizado las respuestas?

Con los modelos de machine learning pasa lo mismo. Si entrenamos el modelo con todos los datos y luego lo evaluamos con esos mismos datos, puede que haya **memorizado** los patrones especificos de nuestra tabla en vez de aprender reglas generales. A esto se le llama **overfitting** (sobreajuste): el modelo funciona muy bien en los datos que ya conoce, pero falla con datos nuevos.

Para evitarlo, dividimos los datos en dos partes:

- **Train set** (conjunto de entrenamiento): los datos que el modelo usa para aprender. Es su "libro de estudio".
- **Test set** (conjunto de evaluacion): datos que el modelo **nunca ha visto**. Es su "examen sorpresa".

Si el modelo saca buena nota en el examen sorpresa, sabemos que realmente ha aprendido algo util.

### Split temporal: por que no dividimos al azar

Hay dos formas de dividir los datos:

**Split aleatorio**: mezclar todas las filas y sacar el 20% al azar para test. Es lo mas comun en machine learning general.

**Split temporal**: usar los datos mas antiguos para train y los mas recientes para test.

Nosotros elegimos **split temporal** porque nuestros datos tienen una dimension temporal importante. Si mezclaramos aleatoriamente, el modelo podria entrenar con datos de agosto de 2020 y luego testearse con datos de julio de 2020. Estaria usando informacion del futuro para predecir el pasado. Esto se llama **data leakage temporal** (fuga de datos temporal) y produce resultados artificialmente optimistas — el modelo parece muy bueno en el laboratorio pero falla en la realidad.

El split temporal simula el escenario real: "tengo datos historicos de 14 años y quiero predecir los proximos 3 años". Si funciona, es una señal mucho mas fiable.

### Nuestro split concreto

| Conjunto | Años | Filas | Incendios | % incendios | Ratio no-incendio:incendio |
|---|---|---|---|---|---|
| Train | 2005-2018 | 72.437 | 4.451 | 6,1% | 15:1 |
| Test | 2019-2021 | 20.407 | 355 | 1,7% | 56:1 |

El test set tiene muchos menos incendios porque los años 2019-2021 tuvieron menos incendios registrados en Ourense (posiblemente por mejoras en prevencion, meteorologia mas favorable, o retraso en la publicacion de datos). Esto hace que el test sea mas dificil para el modelo, pero eso es preferible: si funciona en condiciones dificiles, funcionara en condiciones normales.

---

## Preprocesamiento: preparar los datos antes de entrenar

Antes de darle los datos al modelo, hay que resolver dos problemas.

### Problema 1: Valores nulos (NaN)

Algunas celdas de la tabla estan vacias. En la jerga se llaman **NaN** (Not a Number). Esto ocurre porque:
- 4 estaciones no tienen anemometro (sensor de viento), asi que las columnas de viento estan vacias para esas estaciones (~34% de las filas)
- Algunas estaciones no miden presion atmosferica (~38%)
- Hay pequeños huecos aislados en otras variables (<3%)

Los modelos de scikit-learn no aceptan celdas vacias. Hay que rellenarlas con algun valor razonable. A esto se le llama **imputacion**.

**Estrategia elegida: mediana.** Se rellena cada celda vacia con la **mediana** de esa columna en el train set. La mediana es el valor central cuando ordenas todos los valores de menor a mayor. Por ejemplo, si la temperatura media tiene valores [2, 5, 8, 12, 35], la mediana es 8.

¿Por que la mediana y no la media (promedio)? Porque la mediana es mas robusta ante valores extremos. Si un dia tuvo 35°C (ola de calor), la media sube mucho (12,4°C) pero la mediana apenas cambia (8°C). Queremos un valor "tipico", no uno distorsionado por extremos.

**Detalle importante**: La mediana se calcula SOLO con datos de train. No usamos datos de test para calcularla, porque eso seria otra forma de data leakage (usar informacion del "examen" para preparar el "libro de estudio").

En el codigo, esto se implementa con `SimpleImputer(strategy='median')` de scikit-learn.

### Problema 2: Escalas diferentes (solo para Logistic Regression)

Las 22 variables tienen escalas muy diferentes:
- Temperatura: de -15 a 50 (grados Celsius)
- Presion: de 750 a 1080 (hectopascales)
- Dias sin lluvia: de 0 a 65
- Humedad: de 0 a 100 (porcentaje)

La Logistic Regression es un modelo **lineal**: multiplica cada variable por un peso y suma todo. Si la presion tiene numeros en el rango 750-1080 y la temperatura en el rango -15 a 50, el modelo podria pensar que la presion "importa mas" simplemente porque tiene numeros mas grandes. Pero eso no es cierto — una variacion de 5°C en temperatura puede ser mas significativa que una variacion de 50 hPa en presion.

Para solucionar esto, se aplica **estandarizacion**: a cada variable se le resta su media y se divide entre su desviacion estandar. Despues de estandarizar, todas las variables tienen media 0 y desviacion estandar 1, asi que estan en la misma escala y el modelo puede comparar su importancia de forma justa.

En el codigo, esto se implementa con `StandardScaler()` de scikit-learn.

**Nota**: Los modelos basados en arboles (Random Forest, XGBoost) **no necesitan estandarizacion** porque funcionan de otra forma: toman decisiones por umbrales ("¿la temperatura es mayor que 30?"), y un umbral funciona igual independientemente de la escala. El StandardScaler solo se aplica a la Logistic Regression.

### El Pipeline: encadenar todo sin errores

En scikit-learn, un **Pipeline** es una cadena de pasos que se ejecutan en orden. Nuestro pipeline para Logistic Regression tiene tres pasos:

1. `SimpleImputer(strategy='median')` — Rellena los NaN con la mediana
2. `StandardScaler()` — Estandariza todas las variables
3. `LogisticRegression(...)` — El modelo propiamente dicho

¿Por que usar un Pipeline en vez de hacer cada paso por separado? Porque el Pipeline garantiza que todo se hace correctamente y en el orden correcto. Cuando le dices `pipeline.fit(X_train, y_train)`:
- Calcula las medianas con X_train y rellena los NaN de X_train
- Calcula la media y desviacion estandar con X_train y estandariza X_train
- Entrena el modelo con X_train limpio

Y cuando le dices `pipeline.predict(X_test)`:
- Rellena los NaN de X_test usando las medianas calculadas con X_train (no recalcula)
- Estandariza X_test usando la media/desviacion de X_train (no recalcula)
- Predice con el modelo entrenado

Esto evita data leakage accidental: nunca se usa informacion del test para preprocesar.

---

## Logistic Regression: el modelo baseline

### Que es un baseline

Un **baseline** (linea base) es el modelo mas simple razonable para nuestro problema. Su funcion no es ser el mejor modelo, sino establecer una referencia minima. Si modelos mas complejos (Random Forest, XGBoost) no mejoran significativamente al baseline, puede significar que:
- Los datos no contienen patrones suficientes, o
- Las relaciones son lineales y no hace falta un modelo complejo

Si los modelos complejos SI mejoran al baseline, eso demuestra que las relaciones en los datos son no lineales (lo que esperamos con datos meteorologicos) y justifica el uso de modelos mas sofisticados.

### Que es la Logistic Regression

A pesar de su nombre, la Logistic Regression no es una "regresion" (predecir un numero) sino un modelo de **clasificacion** (predecir una categoria: incendio o no-incendio).

Funciona asi:

1. **Toma las 22 variables** de una fila (temperatura, humedad, lluvia, etc.)
2. **Multiplica cada variable por un peso** (coeficiente) que ha aprendido durante el entrenamiento. Por ejemplo: temperatura x 0.8 + humedad x (-0.5) + dias_sin_lluvia x 1.2 + ...
3. **Suma todo** y le aplica una funcion matematica llamada **sigmoide** que convierte el resultado en un numero entre 0 y 1
4. Ese numero se interpreta como una **probabilidad**: "hay un 72% de probabilidad de incendio"

Es un modelo **lineal** porque la decision se basa en una suma ponderada de las variables. No puede capturar relaciones complejas como "el riesgo sube mucho cuando la temperatura supera los 35°C Y la humedad baja del 20% simultaneamente" (una interaccion no lineal). Para eso necesitamos arboles de decision.

### Parametros que usamos

```python
LogisticRegression(
    class_weight='balanced',    # Compensa el desbalanceo
    max_iter=1000,              # Maximo de iteraciones para converger
    solver='lbfgs',             # Algoritmo de optimizacion
    random_state=42,            # Semilla aleatoria para reproducibilidad
)
```

- **class_weight='balanced'**: Le dice al modelo que equivocarse en un incendio (clase 1) "duele" 15 veces mas que equivocarse en un no-incendio (clase 0). Esto compensa el desbalanceo de 15:1. Sin este parametro, el modelo tenderia a decir siempre "no incendio" porque acertaria en el 94% de los casos (pero seria completamente inutil).

- **max_iter=1000**: El modelo aprende de forma iterativa — va ajustando los pesos poco a poco. 1000 iteraciones es suficiente para que converja (deje de mejorar).

- **solver='lbfgs'**: El algoritmo matematico que usa para encontrar los mejores pesos. LBFGS es un algoritmo de optimizacion eficiente y es el recomendado por defecto en scikit-learn.

- **random_state=42**: Fija la semilla del generador de numeros aleatorios. Esto garantiza que si ejecutamos el script dos veces, obtenemos exactamente los mismos resultados. El numero 42 es una convencion (referencia a "The Hitchhiker's Guide to the Galaxy"), cualquier numero fijo serviria.

---

## Evaluacion: como saber si el modelo funciona

### Prediccion de probabilidades

Una vez entrenado, el modelo recibe las 20.407 filas del test set y para cada una calcula una probabilidad entre 0 y 1. Por ejemplo:

| Dia | Estacion | Probabilidad | Realidad |
|---|---|---|---|
| 2019-08-15 | Viana do Bolo | 0.72 | Incendio (1) |
| 2019-01-03 | Baltar | 0.03 | No incendio (0) |
| 2019-09-22 | Gandarela | 0.31 | Incendio (1) |
| 2019-04-10 | Ourense | 0.08 | No incendio (0) |

### El umbral de decision

Para convertir una probabilidad en una decision (incendio si/no), necesitamos un **umbral** (threshold). El umbral por defecto es **0.5**: si la probabilidad es mayor que 0.5, el modelo dice "incendio"; si es menor, dice "no incendio".

Pero con datos desbalanceados, 0.5 suele ser un mal umbral. El modelo, al haber visto muchos mas dias sin incendio que con incendio, tiende a dar probabilidades bajas. Un dia con riesgo real podria recibir una probabilidad de 0.25 — el modelo "sabe" que hay algo, pero no llega al 0.5.

Por eso usamos **TunedThresholdClassifierCV**: prueba sistematicamente muchos umbrales (0.05, 0.10, 0.15, ..., 0.95) y mide cual da el mejor F1-score usando validacion cruzada. Quiza el mejor umbral resulte ser 0.15, lo que significaria: "si el modelo dice que hay al menos un 15% de probabilidad de incendio, clasificar como riesgo".

### Validacion cruzada (cv=5)

Para encontrar el umbral optimo de forma robusta, se usa **validacion cruzada de 5 folds** (cv=5). Esto significa:

1. Se divide el train set en 5 partes iguales (folds)
2. Se entrena el modelo con 4 partes y se evalua en la parte restante
3. Se repite 5 veces, cada vez dejando una parte diferente fuera
4. Se promedia el resultado

Esto evita que el umbral optimo dependa de una division particular de los datos. Es un umbral robusto que funciona consistentemente.

### Las cuatro metricas

#### Metrica 1: Recall (sensibilidad)

**Pregunta que responde**: "De todos los dias que REALMENTE hubo incendio, ¿cuantos detecto el modelo?"

```
Recall = Incendios detectados / Total de incendios reales
```

Si hubo 100 incendios y el modelo detecto 80, el recall es 0.80 (80%).

En nuestro contexto, **queremos recall alto**: es peor no detectar un incendio real (falso negativo) que dar una falsa alarma (falso positivo). Un sistema de alerta que detecta el 90% de los incendios aunque tenga algunas falsas alarmas es mucho mas util que uno que solo da alertas cuando esta 100% seguro pero se pierde la mitad de los incendios.

#### Metrica 2: Precision

**Pregunta que responde**: "De todos los dias que el modelo DIJO que habria incendio, ¿en cuantos realmente lo hubo?"

```
Precision = Incendios reales entre los alertados / Total de alertas emitidas
```

Si el modelo emitio 200 alertas y en 60 de ellas realmente hubo incendio, la precision es 0.30 (30%). Las otras 140 alertas fueron falsas alarmas.

Precision alta = pocas falsas alarmas. Precision baja = muchas falsas alarmas.

#### Metrica 3: F1-score

**Pregunta que responde**: "¿Cual es el equilibrio entre recall y precision?"

El F1-score es la **media armonica** de precision y recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

La media armonica penaliza valores extremos. Si el recall es 0.90 pero la precision es 0.05, el F1 sera bajo (0.095) porque tener un 5% de precision significa que el 95% de las alertas son falsas — el modelo es inutil aunque detecte muchos incendios.

El F1-score es nuestra metrica principal para comparar modelos: un buen modelo tiene **tanto** buen recall **como** precision razonable.

#### Metrica 4: PR-AUC (Area bajo la curva Precision-Recall)

**Pregunta que responde**: "¿Que tan bueno es el modelo ANTES de elegir un umbral?"

Las tres metricas anteriores dependen del umbral elegido: si bajamos el umbral, sube el recall (detectamos mas) pero baja la precision (mas falsas alarmas). Si subimos el umbral, pasa lo contrario.

La **curva Precision-Recall** dibuja todos los pares (recall, precision) posibles para todos los umbrales. El **PR-AUC** es el area debajo de esa curva: un numero entre 0 y 1 donde 1 es perfecto.

PR-AUC es la metrica mas objetiva porque no depende de la eleccion del umbral. Un modelo con PR-AUC alto tiene buen recall Y buena precision independientemente del umbral que elijamos.

**Nota importante sobre PR-AUC con datos desbalanceados**: Un clasificador aleatorio (que no sabe nada) tendria un PR-AUC igual a la proporcion de positivos. En nuestro test set, eso seria 355/20.407 = 0.017. Cualquier PR-AUC por encima de 0.017 indica que el modelo ha aprendido algo. Un PR-AUC de 0.10 parece bajo pero ya es 6 veces mejor que el azar.

### La matriz de confusion

La **matriz de confusion** es un desglose completo de los 4 tipos de resultado posibles:

|  | Modelo dice: NO incendio | Modelo dice: SI incendio |
|---|---|---|
| **Realidad: NO incendio** | Verdadero Negativo (TN) | Falso Positivo (FP) — falsa alarma |
| **Realidad: SI incendio** | Falso Negativo (FN) — incendio no detectado | Verdadero Positivo (TP) — acierto |

- **Verdadero Negativo (TN)**: El modelo dijo "no incendio" y efectivamente no hubo. Bien.
- **Falso Positivo (FP)**: El modelo dijo "incendio" pero no lo hubo. Falsa alarma. Molesto pero no peligroso.
- **Falso Negativo (FN)**: El modelo dijo "no incendio" pero SI lo hubo. **Este es el peor error** — un incendio no detectado.
- **Verdadero Positivo (TP)**: El modelo dijo "incendio" y efectivamente lo hubo. Acierto.

De estos 4 numeros se calculan las metricas:
- Recall = TP / (TP + FN) — De los incendios reales, cuantos detecto
- Precision = TP / (TP + FP) — De las alertas, cuantas fueron reales

### Coeficientes: que aprendio el modelo

La Logistic Regression asigna un **coeficiente** (peso) a cada variable:

- **Coeficiente positivo** (+): A mayor valor de esa variable, mayor probabilidad de incendio. Ejemplo: si `dias_sin_lluvia` tiene coeficiente +1.2, significa que cuantos mas dias sin llover, mas riesgo ve el modelo.
- **Coeficiente negativo** (-): A mayor valor de esa variable, menor probabilidad de incendio. Ejemplo: si `HR_AVG_1.5m` (humedad) tiene coeficiente -0.5, significa que a mas humedad, menos riesgo ve el modelo.
- **Coeficiente cercano a 0**: El modelo considera que esa variable no aporta mucha informacion.

El valor absoluto del coeficiente indica la **importancia** de la variable. Un coeficiente de +1.5 o -1.5 significa que esa variable influye mucho en la prediccion. Un coeficiente de +0.05 significa que apenas influye.

**Nota**: Los coeficientes solo son comparables entre si cuando las variables estan estandarizadas (todas en la misma escala), que es exactamente lo que hace el StandardScaler. Sin estandarizar, un coeficiente grande podria significar simplemente que esa variable tiene numeros pequeños.

---

## El grafico: curva Precision-Recall

El script genera un grafico que muestra la curva Precision-Recall del modelo. Este grafico es util para la defensa porque muestra visualmente el compromiso entre detectar mas incendios (recall alto) y tener menos falsas alarmas (precision alta).

En el grafico:
- El **eje X** es el recall (de 0 a 1)
- El **eje Y** es la precision (de 0 a 1)
- La **curva** muestra todos los pares posibles al variar el umbral
- La **linea horizontal gris** marca el rendimiento de un clasificador aleatorio (baseline absoluto)
- Cuanto mas arriba y a la derecha este la curva, mejor es el modelo

Un modelo perfecto tendria una curva que pasa por la esquina superior derecha (recall=1, precision=1): detecta todos los incendios sin ninguna falsa alarma. En la practica, siempre hay un compromiso.

---

## Resumen del flujo completo

```
dataset_modelo.csv (92.844 filas x 25 columnas)
        |
        v
[1] Dividir en train (2005-2018) y test (2019-2021)
        |
        v
[2] Pipeline de preprocesamiento:
    SimpleImputer (rellenar NaN con mediana)
    → StandardScaler (poner todo en la misma escala)
    → LogisticRegression (entrenar el modelo)
        |
        v
[3] Predecir probabilidades sobre el test set
        |
        v
[4] Evaluar con umbral 0.5 (F1, Recall, Precision, PR-AUC)
        |
        v
[5] Buscar mejor umbral con TunedThresholdClassifierCV
        |
        v
[6] Evaluar con umbral optimizado
        |
        v
[7] Mostrar coeficientes (que variables importan mas)
        |
        v
[8] Generar grafico de curva Precision-Recall
```

Todo esto se ejecuta con un solo comando:

```bash
source .venv/bin/activate
python3 src/entrenar_modelos.py
```

El script tarda unos pocos segundos en ejecutarse. El resultado es un conjunto de metricas que nos dicen si el modelo lineal es capaz de predecir incendios a partir de datos meteorologicos, y que variables contribuyen mas a esa prediccion.

Despues de analizar los resultados, se entrenara Random Forest para ver si un modelo no lineal mejora significativamente al baseline.
