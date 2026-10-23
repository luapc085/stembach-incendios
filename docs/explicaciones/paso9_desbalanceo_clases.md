# Paso 9: Como manejar el desbalanceo de clases — Explicacion detallada

> Este documento explica, sin asumir conocimientos previos, el problema del desbalanceo de clases en machine learning, por que afecta a nuestro proyecto de prediccion de incendios, y que tecnicas modernas existen para resolverlo. Incluye las fuentes cientificas consultadas.

---

## El problema explicado desde cero

### Que es clasificacion binaria

Nuestro modelo debe responder una pregunta de si/no: **"¿Habra un incendio forestal hoy en esta zona?"**. Esto se llama clasificacion binaria. El modelo recibe datos meteorologicos (temperatura, humedad, lluvia, viento...) y responde con una de dos opciones:

- **Clase 1 (positiva)**: "SI, hay riesgo de incendio"
- **Clase 0 (negativa)**: "NO, no hay riesgo de incendio"

Para aprender a distinguir entre las dos clases, el modelo necesita ejemplos de ambas: dias con incendio Y dias sin incendio.

### Que es el desbalanceo de clases

En nuestro dataset tenemos aproximadamente:

- ~7.700 registros de dias CON incendio (clase 1)
- ~100.000 registros de dias SIN incendio (clase 0)

Hay unas **13 veces mas dias sin incendio que con incendio**. Esto es lo que se llama desbalanceo de clases: una clase es mucho mas frecuente que la otra.

Esto tiene sentido en la realidad: la mayoria de los dias no hay incendios forestales. Pero crea un problema para el modelo.

### Por que el desbalanceo es un problema

Imaginemos que entrenamos un modelo con estos datos desbalanceados y el modelo "decide" hacer trampa: en vez de aprender patrones meteorologicos, simplemente responde **siempre** "NO hay incendio". ¿Cual seria su tasa de acierto?

- De cada 100 registros, ~93 son "no incendio" y ~7 son "incendio"
- Si dice siempre "no incendio", acierta en 93 de 100
- **Tasa de acierto: 93%**

Un 93% parece impresionante, pero el modelo es completamente inutil: jamas detecta un incendio. Es como un detector de humo que nunca suena — su "tasa de acierto" es altisima (porque la mayoria del tiempo no hay fuego), pero el dia que hay un incendio real, no avisa.

Este es el problema fundamental del desbalanceo: los modelos tienden a ignorar la clase minoritaria porque "aciertan mas" si simplemente predicen siempre la clase mayoritaria.

---

## Enfoque clasico vs enfoque moderno

### Lo que se hacia antes: re-muestrear los datos

Historicamente, el enfoque mas comun era modificar los datos para equilibrar las clases antes de entrenar el modelo:

**Undersampling (sub-muestreo)**: Descartar datos de la clase mayoritaria. Por ejemplo, de los ~100.000 dias sin incendio, quedarnos solo con ~7.700 (elegidos al azar) para tener un ratio 1:1.

- **Ventaja**: Dataset pequeño y equilibrado
- **Problema grave**: Tiramos el 92% de los datos negativos. Estamos desperdiciando informacion valiosa — cada dia sin incendio nos dice algo sobre que condiciones meteorologicas son seguras.

**Oversampling / SMOTE**: Generar datos sinteticos de la clase minoritaria. SMOTE (Synthetic Minority Over-sampling Technique) crea incendios "inventados" interpolando entre incendios reales.

- **Ventaja**: No se pierde informacion
- **Problema**: Los datos sinteticos pueden no representar la realidad. Ademas, investigacion reciente (2024) ha demostrado que con modelos potentes como XGBoost, SMOTE no aporta mejora significativa.

### Lo que recomienda la ciencia en 2024-2025

La investigacion mas reciente ha demostrado que hay tecnicas mas efectivas que no requieren modificar los datos. Se basan en tres principios:

1. **No toques los datos**. Usa todos los ejemplos reales, tanto positivos como negativos.
2. **Dile al modelo que los incendios importan mas**. Esto se llama "class weights" (pesos de clase).
3. **Optimiza el punto de corte**. Esto se llama "threshold tuning" (ajuste del umbral).

---

## Las tres tecnicas que usaremos

### Tecnica 1: Pesos de clase (Class Weights)

#### Que es

Cuando un modelo de machine learning se entrena, aprende de sus errores. Si predice "no incendio" y la respuesta correcta era "incendio", se equivoco y se ajusta para no repetir ese error. Los pesos de clase cambian **cuanto le duele al modelo cada tipo de error**.

Sin pesos: equivocarse en un incendio y equivocarse en un no-incendio "duelen" igual.

Con pesos: equivocarse en un incendio "duele" 13 veces mas que equivocarse en un no-incendio.

El efecto es que el modelo se esfuerza mucho mas en no fallar los incendios, porque el coste de ese error es mayor. Es como si a un estudiante le dijeran: "cada pregunta del examen vale 1 punto, pero las preguntas sobre incendios valen 13 puntos". Naturalmente, se concentraria mas en esas preguntas.

#### Como se implementa

En los modelos de scikit-learn y XGBoost, es un unico parametro:

```python
# En Random Forest
modelo = RandomForestClassifier(class_weight='balanced')
# 'balanced' calcula automaticamente los pesos segun la proporcion de clases

# En XGBoost
modelo = XGBClassifier(scale_pos_weight=13)
# 13 porque hay ~13 negativos por cada positivo
```

#### Por que funciona

El estudio "Balancing the Scales" (2024), que realizo 9.000 experimentos con 15 modelos distintos y 30 datasets desbalanceados, demostro que ajustar los pesos de clase supera consistentemente al modelo base (sin ningun tratamiento del desbalanceo).

### Tecnica 2: Ajuste del umbral de decision (Threshold Tuning)

#### Que es un umbral

Un modelo de clasificacion no responde directamente "si" o "no". En realidad, calcula una **probabilidad**: "hay un 72% de probabilidad de incendio", o "hay un 8% de probabilidad de incendio".

El **umbral** es el punto de corte que convierte esa probabilidad en una decision:

- Con umbral = 0.5 (el valor por defecto): Si la probabilidad es >50%, el modelo dice "incendio". Si es <50%, dice "no incendio".
- Con umbral = 0.2: Si la probabilidad es >20%, ya dice "incendio".
- Con umbral = 0.8: Solo dice "incendio" si la probabilidad supera el 80%.

#### Por que 0.5 no es un buen umbral para datos desbalanceados

Cuando hay 13 veces mas negativos que positivos, el modelo tiende a dar probabilidades bajas a todo. Un dia que realmente tiene alto riesgo de incendio podria recibir una probabilidad del 30% — el modelo "sabe" que hay riesgo, pero como la mayoria de dias no hay incendio, no llega al 50%.

Con el umbral por defecto de 0.5, ese dia se clasificaria como "no incendio" a pesar de que el modelo detecto algo.

#### Que hace el threshold tuning

En vez de usar 0.5, se prueba sistematicamente con muchos umbrales distintos (0.1, 0.15, 0.2, 0.25... etc.) y se mide cual da el mejor resultado. Se usa validacion cruzada para que la eleccion sea robusta.

Quiza el mejor umbral para nuestros datos resulte ser 0.18. Eso significaria: "si el modelo dice que hay al menos un 18% de probabilidad de incendio, clasificar como riesgo". Esto puede parecer bajo, pero es coherente con el hecho de que los incendios son eventos raros.

#### Evidencia cientifica

El estudio "Balancing the Scales" (2024) posiciono el threshold tuning como **la tecnica mas efectiva de todas las evaluadas**:

| Tecnica | F1-score medio |
|---|---|
| **Threshold tuning** | **0.617** |
| SMOTE | 0.605 |
| Class weights | 0.594 |
| Sin tratamiento | 0.556 |

Fue la mejor tecnica en 10 de los 15 modelos y en el 40% de los datasets evaluados. Desde scikit-learn v1.5, se puede usar con una sola linea de codigo:

```python
from sklearn.model_selection import TunedThresholdClassifierCV

modelo_base = XGBClassifier(scale_pos_weight=13)
modelo_optimizado = TunedThresholdClassifierCV(modelo_base, scoring='f1', cv=5)
modelo_optimizado.fit(X_train, y_train)
print(f"Umbral optimo encontrado: {modelo_optimizado.best_threshold_}")
```

### Tecnica 3: Metricas adecuadas

#### Por que no usar accuracy

Como se explico arriba, un modelo que siempre dice "no incendio" tendria 93% de accuracy. La accuracy no sirve para evaluar modelos con datos desbalanceados.

#### Metricas que si sirven

**Recall (sensibilidad)**: De todos los dias que REALMENTE hubo incendio, ¿cuantos detecto el modelo?
- Recall = 0.90 significa que detecta el 90% de los incendios
- En nuestro caso, queremos recall alto: preferimos una falsa alarma a no detectar un incendio real

**Precision**: De todos los dias que el modelo PREDIJO como incendio, ¿en cuantos realmente hubo incendio?
- Precision = 0.50 significa que la mitad de las alertas son reales y la otra mitad son falsas alarmas

**F1-score**: Media armonica de precision y recall. Equilibra ambas metricas en un solo numero.

**PR-AUC (Area bajo la curva Precision-Recall)**: Mide el rendimiento del modelo a lo largo de todos los umbrales posibles. Es la metrica mas informativa para datos desbalanceados porque no se deja engañar por la clase mayoritaria.

#### Ejemplo con numeros

Supongamos que en un conjunto de test hay 100 dias con incendio y 1.300 dias sin incendio:

| Modelo | Dice "incendio" | Incendios detectados | Falsas alarmas | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Siempre "no" | 0 | 0/100 | 0 | 0% | - | 0 |
| Naive (umbral 0.5) | 20 | 15/100 | 5 | 15% | 75% | 0.25 |
| Con class weights | 150 | 80/100 | 70 | 80% | 53% | 0.64 |
| Con weights + threshold | 200 | 90/100 | 110 | **90%** | 45% | **0.60** |

El ultimo modelo tiene mas falsas alarmas, pero detecta el 90% de los incendios. En un contexto de prevencion de incendios forestales, eso es mucho mas util que un modelo "preciso" que solo detecta el 15%.

---

## Por que NO usar SMOTE en nuestro caso

SMOTE (Synthetic Minority Over-sampling Technique) fue durante años la tecnica mas popular para desbalanceo. Genera datos sinteticos de la clase minoritaria interpolando entre ejemplos existentes.

Sin embargo, investigacion reciente ha demostrado sus limitaciones:

1. **No mejora con modelos fuertes**: El paper "To SMOTE or not to SMOTE" (2024) demostro que con XGBoost o Random Forest (los modelos que usaremos), SMOTE no aporta mejora estadisticamente significativa sobre simplemente usar class weights.

2. **Riesgo de data leakage**: Si se aplica SMOTE antes de dividir los datos en entrenamiento/test, los datos sinteticos contaminan el test set y las metricas quedan infladas artificialmente. Es un error comun y grave.

3. **Datos sinteticos no son datos reales**: Un dia de incendio "inventado" interpolando entre dos dias reales podria tener combinaciones de variables que no existen en la realidad (por ejemplo, alta temperatura con alta precipitacion simultanea).

4. **Mayor complejidad sin beneficio**: Añade un paso extra al pipeline que puede introducir errores y hace el proyecto mas dificil de explicar.

---

## Modelos ensemble especializados: una opcion intermedia

### Antes de nada: que es un modelo y como "aprende"

Un modelo de machine learning es, en esencia, una funcion matematica que recibe unos datos de entrada (en nuestro caso: temperatura, humedad, lluvia, viento de un dia concreto) y produce una salida (en nuestro caso: "riesgo de incendio" o "sin riesgo").

Esa funcion no esta programada a mano. El modelo **aprende** la funcion a partir de ejemplos. Se le enseñan miles de dias del pasado con sus condiciones meteorologicas y se le dice: "este dia hubo incendio, este no, este si, este no...". A partir de esos ejemplos, el modelo descubre patrones. Por ejemplo, podria descubrir que cuando la temperatura supera 30°C, la humedad baja del 25% y lleva 10 dias sin llover, el riesgo de incendio es alto.

El proceso de aprendizaje funciona asi, simplificado:

1. El modelo hace una prediccion con sus parametros actuales (al principio, aleatorios)
2. Compara su prediccion con la respuesta correcta
3. Mide cuanto se equivoco (esto se llama "error" o "perdida")
4. Ajusta sus parametros internos un poco para equivocarse menos la proxima vez
5. Repite con el siguiente ejemplo

Despues de ver miles de ejemplos, los parametros se han ajustado lo suficiente para que el modelo haga predicciones razonables sobre datos nuevos que nunca ha visto.

### Que es un arbol de decision

El tipo de modelo mas intuitivo para nuestro problema es el **arbol de decision**. Funciona exactamente como un diagrama de flujo con preguntas de si/no:

```
                    ¿Temperatura > 28°C?
                    /                  \
                  SI                    NO
                  /                      \
       ¿Lluvia 7 dias < 5mm?         Bajo riesgo
        /              \
      SI                NO
      /                  \
 ¿Humedad < 30%?      Riesgo medio
  /          \
SI            NO
/              \
ALTO RIESGO   Riesgo medio
```

El arbol va haciendo preguntas sobre las variables meteorologicas, y en cada bifurcacion divide los datos. Al final de cada rama llega a una conclusion: "alto riesgo" o "bajo riesgo". El algoritmo de aprendizaje decide automaticamente que preguntas hacer, en que orden, y con que valores de corte, de forma que los dias con incendio y sin incendio queden lo mas separados posible.

**Ventajas de los arboles**: Son muy faciles de entender y de explicar. Se puede seguir el razonamiento del modelo paso a paso.

**Problema de los arboles**: Un unico arbol tiende a "memorizar" los datos de entrenamiento. Si le enseñas muchos datos, crea un arbol enorme con ramas muy especificas que funcionan perfectamente con los datos que ya vio, pero falla con datos nuevos. Esto se llama **sobreajuste** (overfitting). Es como un estudiante que memoriza las respuestas del examen del año pasado: saca buena nota si el examen se repite exactamente igual, pero suspende si las preguntas son ligeramente distintas.

### Que es un ensemble (conjunto de modelos)

La solucion al problema de un unico arbol es no usar uno solo, sino **muchos**. Si un arbol puede equivocarse, cien arboles que votan juntos se equivocan mucho menos. Esto es un **ensemble**: un grupo de modelos que trabajan juntos y combinan sus predicciones.

La analogia mas clara es el jurado popular. Una sola persona puede tener prejuicios o equivocarse. Pero si 12 personas escuchan el mismo caso y votan, la decision conjunta suele ser mas justa que la de cualquiera de ellas individualmente. Los errores de unas se compensan con los aciertos de otras.

### Random Forest: el ensemble mas conocido

**Random Forest** (Bosque Aleatorio) es un ensemble de arboles de decision. Funciona asi:

1. Se crean, por ejemplo, 200 arboles de decision
2. Cada arbol se entrena con una **muestra aleatoria** de los datos (no todos los arboles ven los mismos datos)
3. Ademas, en cada bifurcacion, cada arbol solo puede elegir entre un **subconjunto aleatorio** de las variables (no todas)
4. Para hacer una prediccion, los 200 arboles votan. Si 150 dicen "incendio" y 50 dicen "no incendio", la prediccion final es "incendio"

La doble aleatorizacion (datos distintos + variables distintas) hace que cada arbol sea diferente. Unos seran mejores detectando incendios de verano, otros los de invierno. Unos se fijaran mas en la temperatura, otros en la humedad. Al combinarlos, el bosque es mucho mas robusto que cualquier arbol individual.

**Problema con datos desbalanceados**: Cada arbol se entrena con una muestra aleatoria de los datos. Si hay 13 veces mas dias sin incendio que con incendio, la muestra aleatoria de cada arbol tambien estara desbalanceada. Los arboles tendran pocos ejemplos de incendio para aprender, y la mayoria "aprendera" a decir "no incendio" por defecto.

### XGBoost y Gradient Boosting: ensembles que aprenden de sus errores

Hay otro tipo de ensemble que funciona diferente al Random Forest. En vez de crear todos los arboles en paralelo (independientes), los crea **en secuencia**, donde cada nuevo arbol se enfoca en corregir los errores de los anteriores:

1. Se entrena el primer arbol. Comete errores.
2. Se entrena el segundo arbol, pero se le dice: "concentrate en los ejemplos que el primer arbol fallo". Asi, el segundo arbol se especializa en los casos dificiles.
3. Se entrena el tercer arbol, enfocado en lo que los dos primeros aun fallan.
4. Y asi sucesivamente, hasta tener 100 o 200 arboles.

La prediccion final es la suma ponderada de todos los arboles. Cada uno aporta una pequeña correccion sobre el anterior.

**XGBoost** (eXtreme Gradient Boosting) es la implementacion mas popular y eficiente de esta idea. Es uno de los algoritmos mas usados en competiciones de ciencia de datos y en la industria. Tiene una ventaja natural para datos desbalanceados: como cada nuevo arbol se enfoca en los errores previos, y los incendios son dificiles de predecir (la clase minoritaria), los arboles posteriores naturalmente prestan mas atencion a los incendios.

### Ahora si: los ensemble especializados para desbalanceo

Los modelos anteriores (Random Forest, XGBoost) son genericos — funcionan con cualquier tipo de datos. Pero la libreria `imbalanced-learn` ofrece versiones especificamente diseñadas para datos desbalanceados. La diferencia clave esta en **como seleccionan los datos que ve cada arbol**.

#### BalancedRandomForestClassifier

Es un Random Forest con una modificacion crucial: en vez de dar a cada arbol una muestra aleatoria (que estaria desbalanceada), le da a cada arbol una **muestra equilibrada**.

Funciona asi:

1. Hay 7.700 dias con incendio y 100.000 sin incendio
2. Para el arbol 1: se toman los 7.700 dias con incendio y se eligen 7.700 dias sin incendio al azar. El arbol ve datos equilibrados (50/50).
3. Para el arbol 2: se toman otra vez los 7.700 dias con incendio y se eligen **otros** 7.700 dias sin incendio al azar (distintos del arbol 1).
4. Para el arbol 3: lo mismo, pero con otra seleccion aleatoria de 7.700 negativos.
5. Y asi para los 200 arboles.

**El resultado clave**: Cada arbol individual ve datos equilibrados y aprende bien a distinguir incendios de no-incendios. Pero como cada arbol ve una seleccion distinta de dias sin incendio, entre todos los arboles del bosque se ha visto **toda la informacion negativa**. No se descarta nada — simplemente se reparte entre los arboles.

Es como si 200 estudiantes se repartieran un temario enorme. Cada uno estudia una parte manejable, pero entre todos cubren todo el temario. Cuando se juntan a resolver un examen, cada uno aporta su conocimiento parcial y el grupo resuelve el examen completo.

En benchmarks con 143 datasets desbalanceados, este modelo supero significativamente al Random Forest estandar.

#### BalancedBaggingClassifier

Este modelo es mas flexible: permite usar **cualquier clasificador** como base (no solo arboles de decision), y le aplica el mismo truco de muestra equilibrada.

Por ejemplo, se pueden crear 10 modelos XGBoost, cada uno entrenado con un subconjunto equilibrado distinto:

- XGBoost 1: 7.700 incendios + 7.700 no-incendios (seleccion A)
- XGBoost 2: 7.700 incendios + 7.700 no-incendios (seleccion B)
- ...
- XGBoost 10: 7.700 incendios + 7.700 no-incendios (seleccion J)

La prediccion final es el promedio de los 10 modelos. Esto combina la potencia de XGBoost con la ventaja del submuestreo equilibrado.

#### EasyEnsembleClassifier

Similar al BalancedBagging, pero usa un tipo especifico de clasificador como base: **AdaBoost**. AdaBoost (Adaptive Boosting) es un ensemble que, como XGBoost, crea modelos en secuencia donde cada uno corrige los errores del anterior.

El EasyEnsemble crea multiples AdaBoosts, cada uno con una muestra equilibrada distinta. Es un "ensemble de ensembles" — una doble capa de combinacion.

### Resumen visual: como se comparan

```
Random Forest estandar:
  Arbol 1: [muestra desbalanceada 13:1] -> prediccion
  Arbol 2: [muestra desbalanceada 13:1] -> prediccion
  Arbol 3: [muestra desbalanceada 13:1] -> prediccion
  -> Voto mayoritario (sesgado hacia "no incendio")

BalancedRandomForest:
  Arbol 1: [7.700 incendios + 7.700 no-incendios (grupo A)] -> prediccion
  Arbol 2: [7.700 incendios + 7.700 no-incendios (grupo B)] -> prediccion
  Arbol 3: [7.700 incendios + 7.700 no-incendios (grupo C)] -> prediccion
  -> Voto mayoritario (equilibrado, y entre todos ven toda la info)

XGBoost con class_weight:
  Arbol 1 -> Arbol 2 (corrige errores de 1) -> Arbol 3 (corrige errores de 1+2) -> ...
  [Todos ven todos los datos, pero los incendios "pesan" 13x mas]
  -> Suma ponderada de todos los arboles
```

### ¿Cual es mejor?

No hay una respuesta universal. Depende del dataset. Por eso en nuestro proyecto la estrategia es:

1. **Primero**: probar XGBoost y Random Forest con class weights + threshold tuning (mas simple, suele funcionar)
2. **Si no es suficiente**: probar BalancedRandomForest y BalancedBagging (mas sofisticados, diseñados para desbalanceo)
3. **Comparar** todos los modelos con las mismas metricas (PR-AUC, F1, Recall) y elegir el mejor

La ventaja de probar varios es que en la defensa del proyecto se puede presentar la comparativa: "probamos X modelos, el mejor fue Y por estas razones". Esto demuestra rigor metodologico.

---

## La estrategia elegida para nuestro proyecto

Combinando las conclusiones de la investigacion, la estrategia es:

### Paso 1: Construir el dataset con TODOS los datos

No descartar ningun dia sin incendio. El dataset tendra ~107.000 filas, con ratio natural ~13:1.

### Paso 2: Entrenar modelos con class weights

```python
# Cada modelo recibe el parametro de pesos
RandomForestClassifier(class_weight='balanced')
XGBClassifier(scale_pos_weight=13)
```

### Paso 3: Optimizar el umbral de decision

```python
# Buscar el mejor punto de corte
TunedThresholdClassifierCV(modelo, scoring='f1', cv=5)
```

### Paso 4: Evaluar con metricas adecuadas

- PR-AUC como metrica principal
- F1-score y Recall como metricas complementarias
- Nunca reportar accuracy como indicador de rendimiento

### Paso 5: Si los resultados no son satisfactorios

Probar los ensemble especializados de imblearn (BalancedRandomForest, BalancedBagging) como plan B.

---

## Analogia final: el detector de humo

Para entender por que este enfoque es el correcto, pensemos en un detector de humo:

- **Sin class weights**: El detector trata igual el error de "sonar cuando no hay fuego" (falsa alarma) y el error de "no sonar cuando si hay fuego" (fallo de deteccion). Resultado: como el 99.9% del tiempo no hay fuego, "aprende" a no sonar nunca.

- **Con class weights**: Le decimos al detector que no sonar durante un incendio es 13 veces mas grave que sonar en falso. Resultado: suena mas a menudo, pero detecta los incendios reales.

- **Con threshold tuning**: Calibramos el nivel de sensibilidad del detector. Quiza el nivel optimo no es "humo denso" (umbral alto) sino "cualquier traza de humo" (umbral bajo). Preferimos alguna falsa alarma a no detectar un incendio.

- **Con metricas adecuadas**: No evaluamos el detector por "% de tiempo que acierta" (accuracy), sino por "% de incendios reales que detecta" (recall) y "% de alarmas que son reales" (precision).

---

## Fuentes y referencias

1. **"Balancing the Scales: A Comprehensive Study on Class Imbalance"** (septiembre 2024)
   - ArXiv: https://arxiv.org/html/2409.19751v1
   - 9.000 experimentos, 15 modelos, 30 datasets
   - Conclusion principal: threshold tuning es la tecnica mas efectiva

2. **"To SMOTE or not to SMOTE"** (2024)
   - Conclusion: SMOTE no mejora con clasificadores fuertes (XGBoost, Random Forest)

3. **"Beyond Rebalancing: Benchmarking Without Rebalancing"** (2025)
   - ArXiv: https://arxiv.org/html/2509.07605v1
   - Modelos como CatBoost mantienen rendimiento sin re-muestreo

4. **"Wildfire Occurrence Prediction with Machine Learning"** (2025)
   - PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC11953322/
   - Aplicacion especifica a prediccion de incendios forestales

5. **Documentacion scikit-learn — TunedThresholdClassifierCV**
   - https://scikit-learn.org/stable/modules/classification_threshold.html
   - Disponible desde scikit-learn v1.5

6. **Documentacion imbalanced-learn — Ensemble Methods**
   - https://imbalanced-learn.org/stable/ensemble.html
   - BalancedRandomForest, EasyEnsemble, BalancedBagging

7. **"Class Imbalance: The Complete 2025 Guide"** (Medium)
   - Guia practica con implementaciones y comparativas

8. **XGBoost for Imbalanced Classification**
   - https://xgboosting.com/xgboost-for-imbalanced-classification/
   - Uso de scale_pos_weight vs SMOTE
