# Comparativa Random Forest vs FWI canadiense

Aporte cuantitativo a la **mejora 2** de la memoria: comparar el modelo Random Forest contra el sistema oficial Fire Weather Index (FWI) canadiense que usa la Xunta como base del IRDI.

## Pregunta que responde

> Si la Xunta ya usa el IRDI (basado en FWI canadiense), ¿qué aporta este modelo Random Forest?

## Cómo se compara

Calculamos el FWI desde las fórmulas oficiales de Van Wagner 1987 sobre las **mismas 20 estaciones MeteoGalicia** que usa el RF, lo evaluamos sobre el **mismo test set 2019-2021** (20.418 filas, 355 incendios), y comparamos las mismas métricas. Es una comparación apples-to-apples: cualquier diferencia es atribuible al método (índice empírico vs ML), no al input.

## Resultado

| Métrica | Random Forest | FWI canadiense | Δ |
|---|---|---|---|
| PR-AUC | 0,1485 | 0,0781 | **+90,0 %** |
| F1 (umbral óptimo) | 0,2183 | 0,1485 | +47,1 % |
| Precision | 0,2093 | 0,1045 | +100,3 % |
| Recall | 0,2282 | 0,2563 | −11,0 % |

El RF **mejora al FWI un 90 % en PR-AUC** sobre el mismo test set. La curva PR (`resultados/pr_curve_rf_vs_fwi.png`) muestra dominancia del RF en todo el rango de recall.

### El FWI sí funciona (validación cruzada del concepto)

Distribución de incendios por nivel canónico FWI:

| Nivel | Rango FWI | N días | Incendios | Tasa | Lift vs base |
|---|---|---:|---:|---:|---:|
| Baixo | 0,0–5,2 | 16.678 | 173 | 1,04 % | 0,6× |
| Moderado | 5,2–11,2 | 2.367 | 64 | 2,70 % | 1,6× |
| Alto | 11,2–21,3 | 1.228 | 90 | 7,33 % | 4,2× |
| Moi alto | 21,3–38,0 | 142 | 28 | 19,72 % | **11,3×** |
| Extremo | ≥38,0 | 3 | 0 | 0,00 % | — |

La tasa de incendio crece monotónicamente con el FWI (descartando Extremo, N=3). Eso confirma que el FWI capta una señal real — el RF no la inventa, la mejora.

## Por qué el RF gana

El FWI es un índice empírico con fórmulas y pesos fijos diseñados en Canadá en los años 70. El RF aprende:

- **No-linealidades específicas de Ourense** desde 17 años de datos.
- **Interacciones complejas** entre variables (ej: días sen choiva × humidade mínima × mes).
- **Estacionalidad gallega** completa, incluido el pico invernal de queimas agrícolas (febrero–marzo) que el FWI canadiense no estaba diseñado para captar.
- **Más variables** (22 frente a las 4 del FWI: T, HR, viento, precip).

## Estructura

```
comparativa_fwi/
├── README.md                  Este fichero
├── calcular_fwi.py            Implementación Van Wagner 1987 sobre meteo_ourense.csv
├── evaluar_fwi.py             Evaluación FWI vs RF + métricas + figura PR
├── data/
│   └── fwi_diario.csv         FWI diario por estación 2005-2021 (output)
└── resultados/
    ├── comparativa.json       Métricas y deltas estructurados
    └── pr_curve_rf_vs_fwi.png Curvas PR superpuestas
```

## Cómo reproducir

```bash
.venv/bin/python3 comparativa_fwi/calcular_fwi.py    # ~30 s
.venv/bin/python3 comparativa_fwi/evaluar_fwi.py     # ~5 s
```

## Decisiones técnicas y limitaciones

- **Implementación**: fórmulas Van Wagner 1987 directas (no librería externa) — ~100 líneas, transparente para defender en memoria.
- **Spinup**: inicialización canónica FFMC=85, DMC=6, DC=15 el día 1 de cada estación; primeros 30 días descartados como burn-in.
- **NaN**: forward-fill de T/HR/precip por estación; viento (32 % nan, 4 estaciones sin anemómetro) imputado con media mensual cross-estación.
- **Latitud**: cada estación con su latitud real (UTM huso 29N → ETRS89 lat/lon vía pyproj). Rango Ourense: 41,9–42,5 °N.
- **Umbral óptimo**: ambos modelos buscan el umbral que maximiza F1 directamente sobre test (criterio idéntico). PR-AUC es insensible al umbral y es la métrica principal.
- **Limitación**: el FWI exige idealmente meteo a las 12:00 (noon); aquí se usa media diaria — convención común en literatura europea, ligeramente conservadora pero estándar.

## Estado

- 2026-05-10: scripts creados y ejecutados. Resultados disponibles. **Pendiente** integrar conclusiones en la memoria (mejora 2).
