# Predición de incendios forestais en Ourense mediante aprendizaxe automática

Proxecto de investigación **STEMbach 2024-2026** (Bacharelato de Excelencia en Ciencias e
Tecnoloxía) do **IES As Lagoas** (Ourense), en colaboración coa **Universidade de Vigo**.

Este repositorio é o **arquivo do proxecto académico tal como se defendeu** o 15 de maio de
2026: a memoria, o código do pipeline, a comparativa co índice FWI canadense e a
documentación do método.

> O desenvolvemento continúa noutro lugar: a aplicación web é [LUMIA](https://lumia.cabanelas.info)
> e a ampliación a toda Galicia está en [lumia-investigacion](https://github.com/luapc085/lumia-investigacion).
> Aquí non se engade nada novo.

## Autoría

**Alumnado:** Lúa Piñeiro Cabanelas · Dimitrii Rapoport Shakin · Mikel Brea Gómez

**Titor do centro:** Carlos Ferreiro García (IES As Lagoas)
**Cotitor universitario:** José Ángel Cid Arújo (Dpto. de Matemáticas, Universidade de Vigo)

## A memoria

[`memoria/memoria_stembach.pdf`](memoria/memoria_stembach.pdf) — a memoria completa, a mesma
que se publica no arquivo do proxecto.

## Que se fixo

Non existía un dataset que cruzase incendios forestais con meteoroloxía diaria en Ourense,
así que había que construílo. Cada fila é un par **estación meteorolóxica × día**, etiquetado
segundo se houbo ou non incendio nesa zona ese día; sobre iso adéstrase un clasificador.

### Fontes (públicas)

| Fonte | Que achega |
|---|---|
| [Fundación Civio / EGIF](https://datos.civio.es/dataset/todos-los-incendios-forestales/) | Incendios forestais rexistrados: data, coordenadas e superficie |
| [API de MeteoGalicia](https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action) | Observacións diarias das estacións meteorolóxicas |

Os datos **non están no repositorio** (centos de MB, e rexenerables): os scripts descárganos
das fontes orixinais.

### O dataset

| | |
|---|---|
| Período | 2005-2021 |
| Estacións | 20 de Ourense (serie 10xxx), 4 delas sen anemómetro |
| Incendios en bruto | 7.719 |
| Incendios con datos completos | 6.528 (84,6 %) |
| Filas (estación × día) | 93.221 |
| Con incendio / sen incendio | 4.512 / 88.709 (1 : 19,7) |

Cada incendio asígnase á estación máis próxima **que teña datos ese día** (mediana: 9,4 km).
Arredor de cada incendio exclúese unha ventá de ±3 días, para non etiquetar como "día
tranquilo" un día inmediatamente anterior ou posterior a un lume.

### As variables

**12 meteorolóxicas**: temperatura media, máxima e mínima; humidade relativa media e mínima;
precipitación; vento medio e refacho máximo; irradiación; horas de sol; punto de rocío; presión.

**10 derivadas**, que capturan a seca e a tendencia: días consecutivos sen choiva;
precipitación acumulada a 7, 14 e 30 días; medias móbiles de temperatura e humidade;
amplitude térmica; mes e día do ano.

## Resultados

Modelo final: **Random Forest** (300 árbores, profundidade 15), reparto **temporal**
(adestra con 2005-2018, avalía con 2019-2021) para simular o escenario real de predicir o
futuro a partir do pasado.

| Métrica | Regresión loxística | **Random Forest** |
|---|---|---|
| PR-AUC | 0,0883 | **0,1485** |
| F1 | 0,159 | **0,208** |
| Recall | — | 0,42 |
| Precisión | — | 0,14 |

**0,1485 é 8,5 veces mellor que o azar**: só o 1,74 % dos pares estación-día teñen incendio,
así que un clasificador aleatorio obtería 0,0174. O Random Forest mellora un 68 % sobre a
referencia lineal.

**As variables máis determinantes son os indicadores de seca e a humidade relativa**, o que
coincide co deseño do FWI canadense (o sistema que usa a Xunta como IRDI). A comparativa
directa co FWI está en [`comparativa_fwi/`](comparativa_fwi/).

## Limitacións (as que recoñece a propia memoria)

- **Non predí incendios: predí condicións favorables a incendio.** Un risco alto non
  significa que vaia haber lume, senón que se o hai, atopará condicións para arder.
- O **83 %** dos incendios en Galicia son intencionados, e o modelo non recolle nada diso.
- O teito do rendemento vén das **variables non meteorolóxicas ausentes**: vexetación,
  topografía e uso do solo.
- A asignación incendio-estación é unha aproximación: a estación pode estar a quilómetros.

## Estrutura

```
memoria/                   A memoria do proxecto (PDF)
src/                       Pipeline, na orde en que se executa
  descargar_meteo.py         Descarga as observacións da API de MeteoGalicia
  limpiar_incendios.py       Filtra os incendios da provincia e do período
  limpiar_meteo.py           Códigos de validación, outliers e ocos
  fusionar_datasets.py       Asigna cada incendio á estación máis próxima con datos
  construir_dataset_clasificacion.py   Etiquetado, ventá de exclusión e variables derivadas
  entrenar_modelos.py        Regresión loxística (referencia)
  entrenar_random_forest.py  Modelo principal
  exportar_modelo.py         Serializa o modelo para a aplicación web
  config.py                  Constantes compartidas
comparativa_fwi/           FWI canadense e comparación co modelo
docs/explicaciones/        O método paso a paso, explicado
docs/proyecto_inicial.md   Planificación e investigación inicial
resultados/                Figuras e métricas do adestramento
```

## Como reproducilo

```bash
pip install -r requirements.txt

# 1. Baixar o CSV de incendios de Civio a data/raw/fires-all.csv
# 2. Descargar a meteoroloxía (tarda)
python3 src/descargar_meteo.py

# 3. Limpeza, fusión e etiquetado
python3 src/limpiar_incendios.py
python3 src/limpiar_meteo.py
python3 src/fusionar_datasets.py
python3 src/construir_dataset_clasificacion.py

# 4. Adestramento
python3 src/entrenar_modelos.py          # referencia
python3 src/entrenar_random_forest.py    # modelo principal
```

## Datos e reutilización

Os **datos** non son deste proxecto e teñen as súas propias condicións: os incendios proceden
da Fundación Civio, a partir da Estatística Xeral de Incendios Forestais do Ministerio, e a
meteoroloxía de MeteoGalicia (Xunta de Galicia). Cita as fontes orixinais se os reutilizas.

O **código e a memoria** son dos tres autores. Se queres reutilizalos, ponte en contacto.
