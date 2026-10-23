# Paso 13: Como se entreno la Logistic Regression y por que no funciona

> Documento explicativo detallado para la defensa. Explica que se hizo, como se hizo, que resultados se obtuvieron, y por que este modelo no es adecuado para predecir incendios forestales.

---

## Que es la Logistic Regression y por que empezamos por ella

### El concepto basico

La Logistic Regression es uno de los modelos de machine learning mas antiguos y simples. A pesar de su nombre (que incluye la palabra "regresion"), es un modelo de **clasificacion**: su trabajo es decidir si algo pertenece a una categoria u otra. En nuestro caso: incendio o no incendio.

Funciona de una forma muy sencilla:

1. Recibe los 22 numeros de un dia (temperatura, humedad, lluvia, viento, etc.)
2. Multiplica cada numero por un **peso** (coeficiente) que ha aprendido
3. Suma todos los resultados
4. Pasa la suma por una funcion matematica llamada **sigmoide** que la convierte en un numero entre 0 y 1
5. Ese numero se interpreta como probabilidad: "hay un 72% de probabilidad de incendio"

Es como una receta de cocina con proporciones fijas: "toma 0.8 de temperatura + (-0.5) de humedad + 1.2 de dias sin lluvia + ... y dime si la suma es alta o baja". Si es alta, hay riesgo de incendio. Si es baja, no.

### Por que empezar por ella

En cualquier proyecto de machine learning serio, el primer modelo que se entrena es el mas simple posible. Se le llama **baseline** (linea base). Su funcion NO es ser el mejor modelo, sino responder dos preguntas:

1. **¿Los datos contienen informacion util?** Si ni siquiera un modelo simple puede predecir mejor que el azar, puede que los datos no sirvan.
2. **¿Hace falta algo mas complejo?** Si el modelo simple ya funciona bien, no necesitamos complicar el proyecto. Si funciona mal, tenemos una justificacion solida para usar modelos mas potentes.

Empezar directamente por el modelo mas potente (XGBoost) sin probar antes el mas simple seria un error metodologico: no podriamos demostrar que la complejidad extra esta justificada.

---

## Como se prepararon los datos para el entrenamiento

### Paso 1: Division temporal (train / test)

Se dividieron las 92.844 filas del dataset en dos conjuntos:

- **Train** (2005-2018): 72.437 filas con 4.451 incendios. Son los datos que el modelo usa para aprender.
- **Test** (2019-2021): 20.407 filas con 355 incendios. Son los datos para evaluar. El modelo nunca los ve durante el entrenamiento.

La division es **temporal**: todo lo anterior a 2019 para aprender, todo lo posterior para evaluar. No se mezcla aleatoriamente. ¿Por que? Porque si el modelo pudiera entrenar con datos de 2020 y evaluarse con datos de 2018, estaria usando informacion del futuro para predecir el pasado. En la realidad, no tenemos datos del futuro, asi que la evaluacion seria engañosa.

### Paso 2: Rellenar huecos (imputacion)

Algunas celdas de la tabla estan vacias (se llaman NaN, de "Not a Number"). Principalmente:

- **Viento** (~34% vacio): 4 estaciones no tienen anemometro
- **Presion** (~38% vacio): varias estaciones no miden presion
- **Irradiacion y horas de sol** (~12% vacio)

Los modelos de scikit-learn no aceptan celdas vacias, asi que hay que rellenarlas. Se usa la **mediana** de cada variable. La mediana es el valor que queda justo en la mitad cuando ordenas todos los valores de menor a mayor. Es mejor que la media (promedio) porque no se distorsiona con valores extremos.

Ejemplo concreto: si la temperatura en la columna `TA_AVG_1.5m` tiene estos valores ordenados: [2, 5, 8, 12, 35], la mediana es 8 (el del medio), mientras que la media es 12.4 (distorsionada por el 35).

**Detalle critico**: La mediana se calcula SOLO con los datos de train. Si usaramos datos de test para calcularla, estariamos haciendo trampa (el modelo "veria" indirectamente informacion del examen mientras estudia).

En el codigo, esto lo hace `SimpleImputer(strategy='median')`.

### Paso 3: Estandarizacion (poner todo en la misma escala)

Las 22 variables tienen rangos muy diferentes:

| Variable | Rango tipico | Escala |
|---|---|---|
| Temperatura media | -5 a 35 | Grados Celsius |
| Presion atmosferica | 800 a 1050 | Hectopascales |
| Dias sin lluvia | 0 a 65 | Dias |
| Humedad relativa | 20 a 100 | Porcentaje |
| Precipitacion acumulada 30d | 0 a 332 | Litros/m2 |

La Logistic Regression multiplica cada variable por un peso y suma. Si la presion tiene numeros alrededor de 900 y la temperatura alrededor de 15, el peso de la presion tendria que ser muy pequeño para compensar que sus valores son grandes. Esto confunde al algoritmo de optimizacion: le cuesta encontrar los pesos correctos cuando las escalas son tan diferentes.

La **estandarizacion** resuelve esto. Para cada variable:

```
valor_estandarizado = (valor_original - media) / desviacion_estandar
```

Despues de estandarizar, todas las variables tienen media 0 y desviacion estandar 1. Una temperatura de 30°C y una presion de 1050 hPa se convierten ambas en numeros como +1.5 o -0.8. Ahora el modelo puede comparar su importancia de forma justa.

En el codigo, esto lo hace `StandardScaler()`.

### Paso 4: El Pipeline (encadenar todo)

Los tres pasos anteriores (imputar, escalar, entrenar) se encadenan en un **Pipeline** de scikit-learn:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )),
])
```

Un Pipeline es como una cadena de montaje: los datos entran por un extremo, pasan por cada paso en orden, y salen procesados por el otro. La ventaja es que garantiza que todo se aplica correctamente: las medianas y las escalas se calculan SOLO con train, y luego se aplican tal cual al test.

---

## Como se entreno el modelo

### Que significa "entrenar"

Entrenar la Logistic Regression significa encontrar los 22 pesos (uno por cada variable) que mejor separan los dias con incendio de los dias sin incendio.

El proceso es iterativo:

1. El modelo empieza con pesos aleatorios
2. Hace predicciones con esos pesos sobre los datos de train
3. Compara sus predicciones con la realidad (la columna `incendio`)
4. Calcula cuanto se ha equivocado (una medida llamada **loss** o funcion de perdida)
5. Ajusta los pesos un poco en la direccion que reduce el error
6. Repite desde el paso 2

Cada repeticion se llama una **iteracion**. El modelo se va acercando poco a poco a los mejores pesos posibles. Cuando el error deja de bajar (o se alcanzan las 1000 iteraciones maximas), el entrenamiento termina.

El algoritmo que decide como ajustar los pesos se llama **solver**. Nosotros usamos **LBFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno), que es un algoritmo de optimizacion eficiente recomendado por scikit-learn para datasets de tamaño medio.

### El parametro class_weight='balanced'

Nuestro train set tiene 4.451 incendios y 67.986 no-incendios (ratio 15:1). Si el modelo se equivoca en un no-incendio, pierde 1 punto. Si se equivoca en un incendio, tambien pierde 1 punto. Pero como hay 15 veces mas no-incendios, el modelo minimiza su error total simplemente diciendo siempre "no incendio". Acierta en el 94% de los casos y se queda tan contento, aunque jamas detecta un incendio.

`class_weight='balanced'` cambia las reglas: ahora equivocarse en un incendio duele **15 veces mas** que equivocarse en un no-incendio. El modelo ya no puede ignorar los incendios porque cada fallo le cuesta mucho. Es como decirle: "los incendios son raros pero muy importantes, prestales mucha atencion".

El peso exacto se calcula automaticamente con la formula:

```
peso_clase = total_muestras / (2 * muestras_de_esa_clase)
```

Para la clase incendio: 72.437 / (2 x 4.451) = 8.13
Para la clase no-incendio: 72.437 / (2 x 67.986) = 0.53

### Que son los coeficientes

Una vez entrenado, el modelo tiene 22 numeros: uno por variable. Estos son los **coeficientes** (pesos). Representan cuanto influye cada variable en la prediccion:

- **Coeficiente positivo (+)**: A mayor valor de esa variable, mas probabilidad de incendio
- **Coeficiente negativo (-)**: A mayor valor de esa variable, menos probabilidad de incendio
- **Coeficiente cercano a 0**: Esa variable apenas influye

Como las variables estan estandarizadas, los coeficientes son directamente comparables: un coeficiente de +1.3 indica mas influencia que uno de +0.3, independientemente de la escala original de la variable.

---

## Los resultados y que significan

### Evaluacion con umbral 0.5

El modelo asigna a cada dia del test una probabilidad entre 0 y 1. Con el umbral por defecto de 0.5, si la probabilidad supera 0.5, dice "incendio"; si no, dice "no incendio".

**Resultados:**

| Metrica | Valor | Que significa |
|---|---|---|
| Recall | 0.7746 (77%) | De los 355 incendios reales, detecto 275 |
| Precision | 0.0542 (5%) | De las 5.077 alertas que emitio, solo 275 eran reales |
| F1-score | 0.1013 | Muy bajo — el equilibrio entre recall y precision es malo |
| PR-AUC | 0.0855 | 5 veces mejor que el azar (0.017), pero aun asi pobre |

**Desglose completo (matriz de confusion):**

De las 20.407 filas del test:
- **15.250** dias sin incendio clasificados correctamente como "no incendio"
- **4.802** dias sin incendio clasificados incorrectamente como "incendio" (falsas alarmas)
- **80** dias con incendio no detectados (el modelo dijo "no incendio" pero SI hubo)
- **275** dias con incendio detectados correctamente

**Interpretacion en lenguaje llano**: El modelo grita "fuego" 5.077 veces, pero solo tiene razon 275 veces (5% de las alertas). Es como un detector de humo que suena constantemente — detecta la mayoria de los fuegos reales, pero tambien suena cuando alguien cocina, cuando hay vapor, cuando pasa un camion... Al final la gente lo desconecta porque no se fian.

### Evaluacion con umbral optimizado (0.746)

Se uso `TunedThresholdClassifierCV` para buscar un umbral mejor. Resultado: el umbral optimo es 0.746.

Esto es inusual. Normalmente, con datos desbalanceados, el umbral optimo baja (por ejemplo, a 0.2 o 0.3) porque el modelo da probabilidades bajas a todo. Que el umbral suba a 0.746 indica que el `class_weight='balanced'` ya hace que el modelo sea muy agresivo dando probabilidades altas, y necesita un filtro mas estricto para no emitir tantas falsas alarmas.

**Resultados con umbral 0.746:**

| Metrica | Valor | Cambio respecto a 0.5 |
|---|---|---|
| Recall | 0.4732 (47%) | Baja de 77% a 47% |
| Precision | 0.0906 (9%) | Sube de 5% a 9% |
| F1-score | 0.1521 | Mejora ligeramente |

Ahora detecta solo 168 de los 355 incendios (47%), con 1.686 falsas alarmas. Menos ruido, pero se pierde la mitad de los incendios. El F1 mejora de 0.10 a 0.15, pero sigue siendo un resultado pobre.

### Los coeficientes: que "aprendio" el modelo

| Rank | Variable | Peso | Interpretacion |
|---|---|---|---|
| 1 | PP_acum_7d (lluvia 7 dias) | -1.32 | Menos lluvia reciente = mas riesgo |
| 2 | IRD_SUM_1.5m (irradiacion) | -0.90 | Menos irradiacion = mas riesgo (*) |
| 3 | dia_del_ano | -0.75 | Principio de año = mas riesgo (*) |
| 4 | TA_AVG_1.5m (temp media) | -0.74 | Menor temperatura = mas riesgo (*) |
| 5 | HSOL_SUM_1.5m (horas sol) | +0.69 | Mas horas de sol = mas riesgo |
| 6 | HR_MIN_3d (humedad min 3d) | -0.62 | Menos humedad = mas riesgo |
| 7 | mes | +0.43 | Meses mas altos = mas riesgo |
| 8 | PP_acum_14d (lluvia 14 dias) | -0.41 | Menos lluvia = mas riesgo |
| 9 | TA_MIN_1.5m (temp minima) | +0.37 | Noches mas calidas = mas riesgo |
| 10 | VV_AVG_10m (viento medio) | +0.36 | Mas viento = mas riesgo |

Algunos coeficientes tienen sentido intuitivo:
- Menos lluvia = mas riesgo (sequedad)
- Menos humedad = mas riesgo (aire seco)
- Mas viento = mas riesgo (propaga el fuego)

Pero otros (*) son contradictoriamente negativos — dicen que menos temperatura y menos irradiacion significan mas riesgo. Eso parece absurdo: ¿el frio causa incendios? La explicacion esta en la siguiente seccion.

---

## Por que la Logistic Regression no funciona para nuestros datos

### Razon 1: No puede capturar relaciones no lineales

La Logistic Regression es un modelo **lineal**: asume que la relacion entre cada variable y el riesgo de incendio es una linea recta. Si la temperatura sube 1 grado, el riesgo siempre sube la misma cantidad, da igual si pasamos de 5°C a 6°C o de 35°C a 36°C.

En la realidad, la relacion temperatura-incendio no es lineal:
- De 5°C a 15°C: el riesgo apenas cambia
- De 15°C a 25°C: empieza a subir
- De 25°C a 35°C: sube mucho
- De 35°C a 40°C: sube drasticamente

Esto es una curva, no una linea recta. La Logistic Regression no puede representar curvas. Solo puede decir "por cada grado mas, el riesgo sube X", cuando la realidad es "a partir de cierto punto, cada grado adicional importa mucho mas".

### Razon 2: No puede capturar interacciones entre variables

Un incendio forestal no depende de una sola variable, sino de la **combinacion** de varias:

- 35°C con 80% de humedad: bajo riesgo (es un dia caluroso pero humedo, tipico del tropico)
- 35°C con 15% de humedad: alto riesgo (calor seco, la vegetacion esta deshidratada)
- 35°C con 15% de humedad y 30 dias sin lluvia: riesgo extremo

La Logistic Regression no puede representar esto. Trata cada variable de forma independiente: suma el efecto de la temperatura + el efecto de la humedad + el efecto de la sequia. No puede decir "el efecto de la temperatura DEPENDE de la humedad". Para eso necesitaria multiplicar variables entre si, lo cual no esta en su formula.

Los modelos de arboles SI pueden capturar interacciones, porque toman decisiones condicionales: "SI la temperatura es > 30 Y la humedad es < 20 Y lleva > 15 dias sin llover, ENTONCES riesgo alto". Esas condiciones encadenadas son exactamente las interacciones que la Logistic Regression no puede modelar.

### Razon 3: La estacionalidad tiene dos picos

En Ourense hay incendios en dos epocas muy diferentes:

- **Febrero-marzo**: quemas agricolas (rozas) que se descontrolan. Temperatura baja, irradiacion baja, poca sequedad.
- **Agosto-septiembre**: sequia estival. Temperatura alta, irradiacion alta, mucha sequedad.

La Logistic Regression tiene que asignar UN solo peso a cada variable. Para la temperatura, tiene dos opciones:
- Peso positivo (+): "mas temperatura = mas riesgo" — captura los incendios de verano, pero falla en los de invierno
- Peso negativo (-): "menos temperatura = mas riesgo" — captura los de invierno, pero falla en los de verano

No puede hacer ambas cosas a la vez. El resultado es un compromiso incoherente: los coeficientes de irradiacion y temperatura salen negativos (favoreciendo los incendios de invierno, que son mas numerosos en Ourense) pero contradiciendose con otras variables como las horas de sol.

Un modelo de arboles resolveria esto trivialmente: una rama del arbol se especializa en incendios de invierno (temperatura < 15 + viento + epoca de rozas) y otra rama en incendios de verano (temperatura > 30 + humedad baja + sequia). Cada rama tiene sus propias condiciones. La Logistic Regression no puede tener "ramas".

### Razon 4: Demasiadas falsas alarmas

Con umbral 0.5, el modelo emite 5.077 alertas de las cuales 4.802 son falsas (95% de falsas alarmas). Esto lo hace inutilizable en la practica: ningun servicio de emergencias confiaria en un sistema que grita "fuego" 5.000 veces y solo acierta 275.

Subir el umbral a 0.746 reduce las falsas alarmas (1.686) pero a costa de perder la mitad de los incendios (solo detecta 168 de 355). No hay ninguna configuracion del umbral que consiga un equilibrio aceptable, porque el problema no esta en el umbral sino en el modelo: la Logistic Regression no sabe distinguir bien entre dias de incendio y dias normales.

---

## Entonces, ¿sirvio de algo?

Si. La Logistic Regression cumplio su funcion como baseline:

### 1. Confirmo que los datos contienen informacion

El PR-AUC de 0.085 es 5 veces mejor que el azar (0.017). Esto demuestra que las variables meteorologicas SI tienen relacion con los incendios. Si el PR-AUC hubiera salido 0.017 o menos, significaria que nuestros datos no sirven y habria que replantear todo el proyecto. No fue el caso.

### 2. Demostro que un modelo lineal es insuficiente

Los coeficientes contradictoriamente (temperatura negativa, irradiacion negativa) y el F1 bajo (0.15) demuestran matematicamente que las relaciones meteorologia-incendio no son lineales. Esto justifica de forma rigurosa el uso de modelos mas complejos.

### 3. Dio una referencia para comparar

Ahora cualquier modelo que entrenemos tiene que superar estos numeros:
- PR-AUC > 0.085
- F1 > 0.15

Si Random Forest o XGBoost no mejoran significativamente, sabriamos que el problema no es el modelo sino los datos. Si mejoran mucho (como esperamos), la mejora es cuantificable y demostrable.

### 4. Identifico las variables mas importantes (de forma aproximada)

Aunque los coeficientes tienen contradicciones, las variables con mayor peso absoluto (lluvia acumulada, humedad, viento) son coherentes con la literatura sobre incendios forestales. Esto valida que el dataset esta bien construido.

---

## Que viene despues

El siguiente paso es entrenar **Random Forest**, un modelo basado en arboles de decision que puede capturar relaciones no lineales e interacciones entre variables. Esperamos una mejora significativa porque:

1. Puede modelar la estacionalidad con dos picos (ramas separadas para invierno y verano)
2. Puede capturar interacciones (temperatura alta + humedad baja + sequia)
3. Puede modelar umbrales no lineales (el riesgo sube mucho a partir de cierta temperatura)

Ademas, Random Forest proporciona **feature importance** (importancia de variables) que, a diferencia de los coeficientes lineales, tiene en cuenta las interacciones. Esto nos permitira responder con un grafico a la pregunta: "¿que variable meteorologica predice mejor los incendios forestales en Ourense?"
