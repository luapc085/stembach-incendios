# Paso 14: Random Forest — Resultados explicados desde cero

> Este documento explica que es Random Forest, como se entreno, que resultados dio, y por que mejora a la Logistic Regression. Escrito para personas sin conocimientos previos de machine learning.

---

## Que es un Random Forest

### La idea basica: muchos arboles votando

Imaginemos que queremos saber si mañana habra incendio en Ourense. Podriamos preguntarle a un experto, pero un solo experto puede equivocarse. Si en vez de uno le preguntamos a **300 expertos** y nos quedamos con la respuesta mas votada, la decision sera mucho mas fiable.

Eso es exactamente lo que hace Random Forest (que en español significa "bosque aleatorio"): entrena **300 arboles de decision** y los hace votar.

### Que es un arbol de decision

Un arbol de decision es como un diagrama de flujo — una serie de preguntas con respuestas si/no:

```
¿Lleva mas de 20 dias sin llover?
├── SI → ¿La humedad es menor del 30%?
│        ├── SI → ¿La temperatura supera los 30°C?
│        │        ├── SI → ALTO RIESGO DE INCENDIO
│        │        └── NO → RIESGO MODERADO
│        └── NO → RIESGO BAJO
└── NO → ¿Llovio mas de 5mm esta semana?
         ├── SI → RIESGO MUY BAJO
         └── NO → RIESGO BAJO
```

Cada pregunta (nodo) divide los datos en dos grupos. El arbol va haciendo preguntas cada vez mas especificas hasta llegar a una decision final (hoja).

La ventaja de los arboles sobre la Logistic Regression es que pueden encadenar condiciones: "SI hay sequia Y ADEMAS la humedad es baja Y ADEMAS hace calor, ENTONCES riesgo alto". La Logistic Regression no puede hacer eso — trata cada variable de forma independiente.

### Por que un bosque y no un solo arbol

Un solo arbol tiene un problema: tiende a **memorizar** los datos de entrenamiento. Crea reglas muy especificas que funcionan perfectamente para los datos que ya ha visto, pero fallan con datos nuevos. A esto se le llama **overfitting** (sobreajuste).

Random Forest resuelve esto con tres trucos:

1. **Entrenar muchos arboles** (300 en nuestro caso) en vez de uno solo
2. **Cada arbol ve datos diferentes**: se le da a cada arbol un subconjunto aleatorio de las filas (como si cada experto estudiara un libro diferente)
3. **Cada arbol usa variables diferentes**: en cada pregunta, el arbol solo puede elegir entre un subconjunto aleatorio de las 22 variables

Al final, los 300 arboles votan. Si 200 dicen "incendio" y 100 dicen "no incendio", la decision es "incendio". Esta votacion colectiva es mucho mas fiable que la opinion de un solo arbol.

Es como pedir opinion a 300 meteorologos, cada uno especializado en variables diferentes (uno sabe mucho de humedad, otro de viento, otro de sequia). Su opinion conjunta es mejor que la de cualquiera individualmente.

---

## Que diferencia hay con la Logistic Regression

| Aspecto | Logistic Regression | Random Forest |
|---|---|---|
| **Como decide** | Suma ponderada de variables (una formula lineal) | Preguntas encadenadas (diagrama de flujo) |
| **Relaciones** | Solo lineales ("mas temperatura = mas riesgo, siempre") | No lineales ("el riesgo sube mucho A PARTIR de 30°C") |
| **Interacciones** | No las captura | Si ("calor + sequia + viento = riesgo extremo") |
| **Estacionalidad** | No puede tener dos picos | Si (una rama para invierno, otra para verano) |
| **Escalado** | Necesita que todas las variables esten en la misma escala | No lo necesita |

La diferencia fundamental es que la Logistic Regression busca una **linea recta** que separe incendios de no-incendios, mientras que Random Forest dibuja **regiones complejas** con formas irregulares. La realidad de los incendios forestales es compleja (depende de combinaciones de factores, no de un solo numero), asi que las regiones complejas se ajustan mucho mejor.

---

## Como se preparo y entreno

### Preparacion

Los datos son los mismos que para la Logistic Regression:
- **Train** (2005-2018): 72.437 filas con 4.451 incendios
- **Test** (2019-2021): 20.407 filas con 355 incendios

Unico preprocesamiento necesario: rellenar las celdas vacias (NaN) con la mediana. Random Forest **no necesita estandarizacion** (poner las variables en la misma escala) porque las preguntas del arbol son del tipo "¿la temperatura es > 30?" y un umbral de 30°C funciona igual independientemente de la escala.

### Parametros elegidos

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=10,
    class_weight='balanced',
)
```

En lenguaje llano:
- **300 arboles**: Cuantos expertos votan. 300 es suficiente para una votacion estable.
- **Profundidad maxima 15**: Cada arbol puede hacer como maximo 15 preguntas encadenadas. Esto evita que los arboles se hagan demasiado especificos y memoricen casos individuales.
- **Minimo 10 ejemplos por hoja**: Cada decision final del arbol necesita estar respaldada por al menos 10 dias reales. Si solo hubiera 1-2 dias, la regla seria poco fiable.
- **Pesos balanceados**: Igual que en LR, equivocarse en un incendio "duele" 15 veces mas que equivocarse en un dia normal. Esto fuerza al modelo a prestar atencion a los incendios aunque sean raros.

---

## Los resultados

### Con umbral 0.5 (por defecto)

El modelo mira los 20.407 dias del test set y para cada uno calcula una probabilidad de incendio. Si la probabilidad supera 0.5, dice "incendio".

**Que paso:**

De los 20.407 dias del test:
- **18.280** dias sin incendio clasificados correctamente
- **1.772** falsas alarmas (el modelo dijo "incendio" pero no hubo)
- **158** incendios no detectados (el modelo dijo "no incendio" pero SI hubo)
- **197** incendios detectados correctamente

**En numeros:**
- **Recall 55%**: De los 355 incendios reales, detecto 197 (algo mas de la mitad)
- **Precision 10%**: De cada 100 alertas, 10 eran incendios reales (las otras 90 eran falsas alarmas)
- **F1-score 0.17**: Equilibrio entre recall y precision
- **PR-AUC 0.14**: Capacidad general del modelo (8 veces mejor que el azar)

### Con umbral optimizado (0.591)

Se busco el umbral que mejor equilibra la deteccion de incendios con las falsas alarmas. Resultado: 0.591.

**Que paso:**

- **18.914** dias sin incendio clasificados correctamente
- **1.138** falsas alarmas (menos que antes)
- **201** incendios no detectados
- **154** incendios detectados

**En numeros:**
- **Recall 43%**: Detecta 154 de 355 incendios
- **Precision 12%**: De cada 100 alertas, 12 son reales
- **F1-score 0.19**: Mejor equilibrio que con umbral 0.5

### Comparacion directa con Logistic Regression

| Lo que medimos | Logistic Regression | Random Forest | Mejora |
|---|---|---|---|
| PR-AUC (capacidad general) | 0.085 | 0.141 | **+64%** |
| F1-score (equilibrio) | 0.152 | 0.187 | +23% |
| Precision (fiabilidad de alertas) | 9% | 12% | +32% |
| Recall (incendios detectados) | 47% | 43% | -8% |
| Falsas alarmas | 1.686 | 1.138 | **-32%** |

**Resumen**: Random Forest es claramente mejor. La mejora mas importante es en PR-AUC (+64%), que mide la capacidad general del modelo independientemente del umbral elegido. Tambien reduce las falsas alarmas un 32% (de 1.686 a 1.138).

El recall baja ligeramente (de 47% a 43%), pero es un intercambio positivo: pierde 14 incendios detectados a cambio de eliminar 548 falsas alarmas. Un sistema con menos falsas alarmas es mas confiable — la gente no lo desconecta por cansancio.

---

## Que variables predicen mejor los incendios

Esta es una de las partes mas valiosas del proyecto para la defensa. Random Forest calcula la **importancia de cada variable**: cuanto contribuye esa variable a las decisiones de los 300 arboles.

### Ranking completo

| Posicion | Variable | Importancia | Que mide |
|---|---|---|---|
| **1** | **dias_sin_lluvia** | **11,7%** | Dias consecutivos sin lluvia (< 1 L/m²) |
| **2** | **PP_acum_7d** | **11,2%** | Lluvia total de la ultima semana |
| **3** | **HR_MIN_3d** | **9,1%** | Humedad minima de los ultimos 3 dias |
| 4 | PP_acum_14d | 8,1% | Lluvia total de las ultimas 2 semanas |
| 5 | PP_acum_30d | 6,9% | Lluvia total del ultimo mes |
| 6 | HR_AVG_1.5m | 6,7% | Humedad relativa media del dia |
| 7 | HR_MIN_1.5m | 6,4% | Humedad relativa minima del dia |
| 8 | dia_del_ano | 5,8% | Epoca del año (estacionalidad) |
| 9 | IRD_SUM_1.5m | 4,0% | Irradiacion solar |
| 10 | HSOL_SUM_1.5m | 3,7% | Horas de sol |
| 11 | mes | 3,0% | Mes del año |
| 12 | amplitud_termica | 2,9% | Diferencia entre maxima y minima del dia |
| 13 | TO_AVG_1.5m | 2,8% | Temperatura de rocio |
| 14 | TA_MAX_3d | 2,4% | Temperatura maxima de los ultimos 3 dias |
| 15 | PR_AVG_1.5m | 2,3% | Presion atmosferica |
| 16 | TA_MAX_1.5m | 2,2% | Temperatura maxima del dia |
| 17 | TA_AVG_3d | 2,1% | Temperatura media de los ultimos 3 dias |
| 18 | TA_AVG_1.5m | 1,9% | Temperatura media del dia |
| 19 | PP_SUM_1.5m | 1,8% | Lluvia del dia |
| 20 | VV_AVG_10m | 1,8% | Velocidad media del viento |
| 21 | TA_MIN_1.5m | 1,7% | Temperatura minima del dia |
| 22 | VV_MAX_10m | 1,5% | Racha maxima de viento |

### Que nos dice esto

**La sequia es el factor principal.** Las 5 variables mas importantes estan todas relacionadas con el agua: dias sin lluvia, lluvia acumulada en 7/14/30 dias, y humedad minima. Juntas suman el 47% de la importancia total.

Esto tiene todo el sentido:
- Un bosque seco arde facilmente. Un bosque humedo, no.
- No importa tanto la temperatura de HOY como cuantos dias lleva sin llover.
- La humedad relativa del aire indica cuanta agua hay disponible para que la vegetacion se seque.

**La temperatura importa menos de lo esperado.** `TA_AVG_1.5m` (temperatura media) ocupa la posicion 18 de 22. Esto no significa que la temperatura no importe, sino que su efecto ya esta capturado indirectamente por otras variables: cuando hace calor, la humedad baja, la vegetacion se seca, y los dias sin lluvia se acumulan. El modelo no necesita mirar la temperatura directamente porque ya tiene esa informacion a traves de la sequia y la humedad.

**El viento es el menos importante.** Las variables de viento (posiciones 20 y 22) son las menos relevantes. El viento importa mucho para la **propagacion** de un incendio una vez iniciado, pero no tanto para **predecir** si habra un incendio. Nuestro modelo predice si un dia tiene condiciones para que se inicie un incendio, no si ese incendio sera grande.

**Comparacion con la Logistic Regression**: En LR, los coeficientes decian cosas contradictorias (menos temperatura = mas riesgo, menos irradiacion = mas riesgo). En Random Forest, la importancia de variables es coherente y se puede explicar con sentido comun. Esto es una ventaja importante para la defensa.

---

## Sigue sin ser perfecto: por que el rendimiento es modesto

A pesar de la mejora sobre LR, los numeros absolutos siguen siendo modestos: F1 de 0.19, precision del 12%. ¿Por que?

### Razon 1: El test set es muy dificil

Los años 2019-2021 tuvieron muy pocos incendios: solo 355 en 3 años (1,7% de las filas). El ratio de desbalanceo en test es 56:1, mucho peor que el 15:1 del train. Es como estudiar con un libro donde el 6% de los ejemplos son incendios, y luego hacer un examen donde solo el 1,7% lo son. El examen es mucho mas dificil que el material de estudio.

### Razon 2: Los incendios son inherentemente dificiles de predecir

Un incendio forestal necesita tres cosas: combustible (vegetacion seca), condiciones meteorologicas favorables, y una **fuente de ignicion** (un rayo, una quema agricola, un piromanía...). Nuestro modelo solo tiene datos meteorologicos. No sabe si alguien va a encender una quema ese dia. Puede identificar dias con condiciones de riesgo, pero no puede predecir la chispa.

Esto es una limitacion fundamental e inevitable del modelo: **predecimos el riesgo, no el incendio**. Muchos dias con condiciones perfectas para un incendio no tienen incendio porque nadie encendio nada. Esos dias aparecen como falsas alarmas, pero en realidad el modelo esta en lo correcto: las condiciones eran peligrosas.

### Razon 3: Las estaciones meteorologicas no estan en el punto exacto del incendio

Nuestros datos meteorologicos son de 20 estaciones repartidas por Ourense. La mediana de distancia entre un incendio y su estacion mas cercana es 9,4 km. En 9 kilometros, la lluvia, la humedad y el viento pueden variar bastante, especialmente en terreno montañoso como el de Ourense. El modelo trabaja con una aproximacion de las condiciones reales, no con las condiciones exactas del punto del incendio.

---

## Conclusion

Random Forest confirma lo que la Logistic Regression sugirio pero no pudo ejecutar: las variables meteorologicas SI predicen el riesgo de incendio, pero las relaciones son complejas y no lineales. El modelo no es perfecto (predice el riesgo, no el incendio), pero demuestra que la sequia y la humedad son los factores determinantes.

El siguiente paso es probar **XGBoost**, un modelo aun mas potente que deberia mejorar estos numeros. XGBoost entrena arboles de forma secuencial — cada nuevo arbol se especializa en corregir los errores del anterior — lo que suele dar mejor rendimiento que la votacion independiente de Random Forest.
