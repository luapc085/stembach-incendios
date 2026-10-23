# Paso 10: Seleccion de modelos de clasificacion — Explicacion detallada

> Este documento explica, sin asumir conocimientos previos, por que se eligieron los modelos de machine learning que se usaran en el proyecto, por que se descartaron otros, y como se compararan entre si para elegir el mejor.

---

## Que es un modelo de clasificacion

Un modelo de clasificacion es un programa que aprende de ejemplos pasados para hacer predicciones sobre casos nuevos. En nuestro caso:

- **Entrada**: datos meteorologicos de un dia (temperatura, humedad, lluvia, viento...)
- **Salida**: una prediccion: "riesgo de incendio" o "sin riesgo"

El modelo no sabe nada de incendios ni de meteorologia. Lo unico que hace es buscar **patrones** en los datos historicos: "cuando la temperatura fue alta, la humedad baja y no habia llovido en dias... hubo incendio". Si encuentra esos patrones de forma consistente, puede usarlos para predecir dias futuros.

---

## Los tipos de datos importan: tabulares vs no estructurados

Antes de elegir un modelo, hay que entender que tipo de datos tenemos. En inteligencia artificial hay dos grandes familias:

### Datos no estructurados
- **Imagenes**: miles de pixeles con colores
- **Texto**: palabras y frases con significado
- **Audio**: ondas sonoras
- **Video**: secuencias de imagenes

Para estos datos se usan **redes neuronales profundas** (deep learning) y **modelos de lenguaje** (LLMs como ChatGPT o Claude). Son modelos enormes con millones de parametros que necesitan grandes cantidades de datos y mucha potencia de calculo.

### Datos tabulares (los nuestros)
- Una **tabla** con filas y columnas
- Cada fila es un registro (un dia)
- Cada columna es una variable numerica (temperatura, humedad, etc.)
- Es el tipo de datos de una hoja de calculo

Nuestro dataset es una tabla de ~107.000 filas y ~22 columnas numericas. **Este es un dato tabular clasico.**

### Por que la distincion importa

La investigacion cientifica ha demostrado repetidamente que, para datos tabulares, los modelos basados en **arboles de decision** funcionan mejor que las redes neuronales profundas. Esto puede parecer contraintuitivo — ¿no son las redes neuronales "mas potentes"? — pero tiene una explicacion logica.

Las redes neuronales son extraordinarias encontrando patrones en datos donde la estructura importa: la posicion de un pixel respecto a sus vecinos (imagenes), el orden de las palabras (texto), la secuencia temporal (audio). Pero en una tabla, las columnas son independientes: la columna de temperatura no tiene "vecinos" como los pixeles de una foto. No hay estructura espacial que explotar, y las redes neuronales desperdician su potencia buscando estructuras que no existen.

Los arboles de decision, en cambio, estan diseñados exactamente para esto: dividir datos segun valores de columnas. "Si temperatura > 30 y humedad < 25, entonces riesgo alto." Es la forma natural de razonar sobre tablas.

**Evidencia cientifica:**

- **Grinsztajn et al. (2022)** — *"Why do tree-based models still outperform deep learning on tabular data?"*, presentado en NeurIPS 2022 (una de las conferencias mas importantes de IA). Compararon modelos de arboles con redes neuronales en 45 datasets tabulares. Los arboles ganaron en la mayoria.

- **Shwartz-Ziv & Armon (2022)** — *"Tabular data: Deep learning is not all you need"*, publicado en Information Fusion. Conclusión: XGBoost iguala o supera a las redes neuronales en la mayoria de problemas tabulares, siendo mucho mas rapido de entrenar.

---

## Los tres modelos elegidos

Se eligieron tres modelos, ordenados del mas simple al mas complejo. El objetivo es demostrar una **progresion**: empezar con algo basico y justificar por que se necesita algo mas avanzado.

### Modelo 1: Regresion Logistica (el baseline)

#### Que es

A pesar de su nombre, la regresion logistica es un modelo de **clasificacion**, no de regresion. Es el modelo mas simple posible para clasificacion binaria.

Funciona asi: asigna un "peso" a cada variable meteorologica y los suma. Si la suma supera un umbral, predice "incendio"; si no, predice "no incendio".

```
puntuacion = (peso_temp × temperatura) + (peso_humedad × humedad) + (peso_lluvia × lluvia) + ...

Si puntuacion > umbral → "incendio"
Si puntuacion ≤ umbral → "no incendio"
```

#### Por que se incluye

No porque esperemos que sea el mejor, sino porque es el **punto de referencia**. Si la regresion logistica ya consigue un F1-score de 0.80, quizas no necesitamos modelos mas complejos. Si solo consigue 0.40, los modelos de arboles deberian mejorar esa cifra, y la diferencia demuestra que las relaciones en los datos no son lineales (que la temperatura y la humedad interactuan de formas que una suma simple no puede capturar).

Es como en un experimento de quimica: siempre se necesita un **grupo de control**. La regresion logistica es nuestro grupo de control.

#### Limitacion

Solo puede encontrar relaciones lineales. Si el riesgo de incendio depende de que la temperatura sea alta **Y** la humedad sea baja **al mismo tiempo** (una interaccion), la regresion logistica no lo captura bien. Los arboles de decision si.

---

### Modelo 2: Random Forest (el robusto)

#### Que es

Un Random Forest (bosque aleatorio) es un conjunto de muchos **arboles de decision** que trabajan juntos.

**Un arbol de decision** es como un diagrama de flujo:

```
¿Temperatura > 30°C?
├── SI → ¿Humedad < 25%?
│         ├── SI → ¿Lluvia ultimos 7 dias < 5mm?
│         │         ├── SI → RIESGO ALTO
│         │         └── NO → RIESGO MEDIO
│         └── NO → RIESGO BAJO
└── NO → ¿Dias sin lluvia > 15?
          ├── SI → RIESGO MEDIO
          └── NO → RIESGO BAJO
```

El problema de un solo arbol es que se adapta demasiado a los datos de entrenamiento (overfitting): memoriza los ejemplos en vez de aprender patrones generales.

**Un Random Forest resuelve esto** entrenando muchos arboles (tipicamente 100-500), cada uno con:
- Un subconjunto aleatorio de los datos
- Un subconjunto aleatorio de las variables

Cuando llega un dia nuevo, cada arbol "vota": riesgo / no riesgo. La decision final es por **mayoria**. Si 350 de 500 arboles dicen "riesgo", la prediccion es "riesgo".

#### Por que se incluye

- **Robusto**: muy dificil de sobreajustar gracias al mecanismo de votacion
- **Interpretable**: permite calcular la **importancia de cada variable**. El Random Forest puede decirnos: "la humedad relativa minima es la variable que mas contribuye a la prediccion, seguida de la precipitacion acumulada y la temperatura maxima". Esto es valiosisimo para la memoria y la defensa — responde directamente a la pregunta "¿que condiciones meteorologicas predicen los incendios?"
- **Facil de explicar**: "un bosque de arboles de decision que votan" es una metafora que cualquier tribunal entiende

#### Feature importance: que variables importan mas

Una de las salidas mas utiles de Random Forest es un grafico de barras que muestra cuanto contribuye cada variable a la prediccion:

```
HR_MIN_1.5m (humedad minima)     ████████████████████  0.18
PP_acum_7d (lluvia 7 dias)       ███████████████████   0.17
TA_MAX_1.5m (temp maxima)        ████████████████      0.14
dias_sin_lluvia                  ██████████████        0.12
VV_AVG_10m (viento)              ██████████            0.09
...
```

*(Estos numeros son hipoteticos — los reales se calcularan al entrenar el modelo)*

Este grafico es probablemente la figura mas importante de toda la memoria: conecta directamente el machine learning con la meteorologia, mostrando que variables del tiempo predicen mejor los incendios.

---

### Modelo 3: XGBoost (el principal)

#### Que es

XGBoost (eXtreme Gradient Boosting) tambien usa arboles de decision, pero de forma muy distinta a Random Forest.

En Random Forest, los arboles se entrenan **en paralelo** (cada uno por su cuenta) y luego votan. En XGBoost, los arboles se entrenan **en serie** (uno detras de otro), y cada nuevo arbol intenta **corregir los errores del anterior**:

1. Se entrena el primer arbol. Comete errores en algunos registros.
2. Se entrena el segundo arbol, pero se le dice: "presta mas atencion a los registros donde el primero se equivoco".
3. Se entrena el tercero, prestando atencion a los errores acumulados.
4. Y asi sucesivamente.

La prediccion final es la suma ponderada de todos los arboles.

#### Por que es el modelo principal

- **Rendimiento**: XGBoost es consistentemente el modelo con mejor rendimiento en datos tabulares. En las competiciones de Kaggle (la mayor plataforma de ciencia de datos), XGBoost o sus variantes (LightGBM, CatBoost) ganan la mayoria de competiciones con datos tabulares.
- **Manejo nativo del desbalanceo**: El parametro `scale_pos_weight=13` le indica directamente al modelo que los incendios son 13 veces mas importantes que los no-incendios.
- **Rapidez**: Puede entrenar en 107.000 filas en segundos.

#### Diferencia con Random Forest

| Aspecto | Random Forest | XGBoost |
|---|---|---|
| Entrenamiento | Arboles en paralelo (independientes) | Arboles en serie (cada uno corrige al anterior) |
| Filosofia | "Muchas opiniones independientes" | "Aprender de los errores" |
| Riesgo de overfitting | Bajo | Medio (requiere ajustar hiperparametros) |
| Rendimiento tipico | Bueno | Ligeramente mejor |
| Explicabilidad | Muy alta (feature importance directa) | Alta (SHAP values) |

---

## Modelos descartados y por que

### KNN (K-Nearest Neighbors)

KNN funciona asi: para clasificar un dia nuevo, busca los K dias mas parecidos en el historico y "copia" su etiqueta. Si los 5 dias mas parecidos fueron todos incendios, predice incendio.

**Por que se descarta**:
- Para cada prediccion, necesita calcular la distancia a los ~107.000 registros del dataset. Es muy lento.
- Funciona mal cuando hay muchas variables (22 en nuestro caso). En alta dimensionalidad, la nocion de "distancia" se degrada — todos los puntos parecen estar a la misma distancia.
- Con desbalanceo 13:1, los "vecinos" mas cercanos casi siempre seran no-incendios, por pura probabilidad.

### SVM (Support Vector Machine)

SVM busca un hiperplano que separe las dos clases con el mayor margen posible. Puede usar "kernels" para manejar relaciones no lineales.

**Por que se descarta**:
- El entrenamiento tiene complejidad O(n²) o peor. Con 107.000 filas, seria extremadamente lento (horas en vez de segundos).
- Dificil de interpretar: no produce feature importance ni probabilidades calibradas nativamente.
- No aporta ventaja sobre XGBoost en datos tabulares segun los benchmarks.

### Redes neuronales profundas (Deep Learning)

**Por que se descartan**:
- Diseñadas para datos no estructurados (imagenes, texto), no para tablas numericas
- Peor rendimiento que arboles en datos tabulares (demostrado en NeurIPS 2022)
- Requieren mucha mas potencia de calculo y datos
- Son "cajas negras": es muy dificil explicar al tribunal POR QUE predicen lo que predicen
- Para 107.000 filas con 22 variables, es matar moscas a cañonazos

### LLMs (modelos de lenguaje tipo ChatGPT/Claude)

**Por que se descartan**:
- Los LLMs procesan **texto** (palabras, frases, documentos)
- Nuestros datos son **numeros en una tabla**: 35.2°C, 22.1% de humedad, 0.0 mm de lluvia
- Un LLM no puede (ni debe) hacer clasificacion numerica — no esta diseñado para eso
- Es como usar un diccionario para resolver una ecuacion: son herramientas para tareas distintas

---

## La estrategia completa: comparacion rigurosa

La clave del proyecto no es usar "el mejor modelo", sino **demostrar por que es el mejor**. Para eso se entrenaran los tres modelos con los mismos datos y las mismas metricas, y se presentara una tabla comparativa:

| Modelo | F1-score | Recall | PR-AUC | Tiempo |
|---|---|---|---|---|
| Logistic Regression | ? | ? | ? | ? |
| Random Forest | ? | ? | ? | ? |
| XGBoost | ? | ? | ? | ? |

*(Los valores reales se rellenaran al ejecutar el modelado)*

Esta tabla responde a varias preguntas que el tribunal podria hacer:

- "¿Por que no usaste un modelo mas simple?" → "Lo probe. Aqui estan los numeros: el F1 de la regresion logistica es X, y el de XGBoost es Y. La mejora justifica la complejidad."
- "¿Por que Random Forest y no solo XGBoost?" → "Random Forest da feature importance mas intuitiva y es mas facil de explicar. Si el rendimiento fuera igual, elegiria Random Forest por interpretabilidad."
- "¿No seria mejor una red neuronal?" → "No. La investigacion de Grinsztajn et al. (NeurIPS 2022) demuestra que en datos tabulares los arboles superan a las redes neuronales. Ademas, una red neuronal seria una caja negra que no podria explicar."

---

## Analogia final: el equipo de diagnostico medico

Elegir modelos de machine learning es como montar un equipo de diagnostico:

- **Regresion logistica** es el medico de cabecera: rapido, economico, detecta lo obvio. Si dice que estas enfermo, probablemente lo estas. Pero puede pasar por alto casos complejos.

- **Random Forest** es una junta medica: 500 medicos independientes examinan al paciente por separado y votan. Es muy fiable porque los errores individuales se compensan. Ademas, puede decirte que sintomas fueron los que mas influyeron en el diagnostico.

- **XGBoost** es un equipo de especialistas que trabajan en cadena: el primero hace un diagnostico inicial, el segundo revisa los casos donde el primero se equivoco y los estudia con mas detalle, el tercero revisa los errores del segundo, y asi sucesivamente. El resultado final es el mas preciso, pero tambien el mas dificil de explicar al paciente.

Nuestro proyecto usa los tres y compara sus resultados, igual que un hospital compara tratamientos para decidir cual funciona mejor. La diferencia es que nosotros podemos probarlo todo en segundos en un ordenador, sin riesgo para nadie.

---

## Referencias

1. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022)**. "Why do tree-based models still outperform deep learning on tabular data?" *NeurIPS 2022*. https://arxiv.org/abs/2207.08815

2. **Shwartz-Ziv, R. & Armon, A. (2022)**. "Tabular data: Deep learning is not all you need." *Information Fusion, 81*, 84-90. https://arxiv.org/abs/2106.03253

3. **Chen, T. & Guestrin, C. (2016)**. "XGBoost: A Scalable Tree Boosting System." *KDD 2016*. El paper original de XGBoost, con mas de 30.000 citas.

4. **Breiman, L. (2001)**. "Random Forests." *Machine Learning, 45*, 5-32. El paper original de Random Forest.

5. **Documentacion de scikit-learn** — RandomForestClassifier: https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees

6. **Documentacion de XGBoost** — https://xgboost.readthedocs.io/
