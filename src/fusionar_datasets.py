#!/usr/bin/env python3
"""
Paso 8.4-8.7 — Fusion espacial-temporal de incendios + meteorologia.

1. Convierte coordenadas de incendios de lat/lon WGS84 a UTM zona 29N
2. Para cada incendio, ordena TODAS las estaciones por distancia
3. Busca la estacion mas cercana que tenga datos para esa fecha
4. Para variables de viento, busca la estacion mas cercana CON anemometro
5. Guarda el dataset fusionado

Uso: python3 src/fusionar_datasets.py
"""

import pandas as pd
import numpy as np
import math
import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
INC_CSV = os.path.join(PROCESSED_DIR, "incendios_ourense.csv")
METEO_CSV = os.path.join(PROCESSED_DIR, "meteo_ourense.csv")
ESTACIONES_CSV = os.path.join(PROCESSED_DIR, "estaciones_ourense.csv")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "dataset_fusionado.csv")

# Estaciones que NO miden viento (identificadas en paso 7)
SIN_VIENTO = {10111, 10113, 10114, 10148}

# Variables de viento (se obtienen de estacion alternativa si la primaria no tiene)
VARS_VIENTO = ["VV_AVG_10m", "VV_MAX_10m"]

# All meteorological variables
METEO_VARS = [
    "TA_AVG_1.5m", "TA_MAX_1.5m", "TA_MIN_1.5m",
    "HR_AVG_1.5m", "HR_MIN_1.5m",
    "PP_SUM_1.5m",
    "VV_AVG_10m", "VV_MAX_10m",
    "IRD_SUM_1.5m", "HSOL_SUM_1.5m",
    "TO_AVG_1.5m", "PR_AVG_1.5m",
]


def latlon_to_utm29n(lat, lon):
    """
    Convert WGS84 lat/lon to UTM zone 29N coordinates.
    UTM zone 29N: central meridian = -9 degrees.
    Uses the exact transverse Mercator formulas (Snyder, 1987).
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0_rad = math.radians(-9)

    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e_prime2 * math.cos(lat_rad) ** 2
    A_ = (lon_rad - lon0_rad) * math.cos(lat_rad)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )

    k0 = 0.9996
    easting = k0 * N * (
        A_ + (1 - T + C) * A_**3 / 6
        + (5 - 18 * T + T**2 + 72 * C - 58 * e_prime2) * A_**5 / 120
    ) + 500000

    northing = k0 * (
        M + N * math.tan(lat_rad) * (
            A_**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A_**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * e_prime2) * A_**6 / 720
        )
    )
    return easting, northing


def dist_utm(x1, y1, x2, y2):
    """Euclidean distance in meters between two UTM points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    print("=" * 70)
    print("FUSION ESPACIAL-TEMPORAL: INCENDIOS + METEOROLOGIA")
    print("=" * 70)

    # --- Load data ---
    inc = pd.read_csv(INC_CSV, parse_dates=["fecha"])
    meteo = pd.read_csv(METEO_CSV, parse_dates=["fecha"])
    estaciones = pd.read_csv(ESTACIONES_CSV)

    print(f"\nDatos cargados:")
    print(f"  Incendios: {len(inc):,} registros")
    print(f"  Meteorologia: {len(meteo):,} filas (dias x estaciones)")
    print(f"  Estaciones: {len(estaciones)}")

    # --- STEP 1: Convert fire coordinates to UTM ---
    print(f"\n--- PASO 1: Conversion de coordenadas a UTM zona 29N ---")
    utm_coords = inc.apply(lambda r: latlon_to_utm29n(r.lat, r.lng), axis=1)
    inc["utmx"] = [c[0] for c in utm_coords]
    inc["utmy"] = [c[1] for c in utm_coords]
    print(f"  {len(inc):,} puntos convertidos")

    # --- STEP 2: Compute distance from each fire to ALL stations ---
    print(f"\n--- PASO 2: Calculo de distancias a todas las estaciones ---")
    est_coords = {
        int(row.estacion_id): (row.utmx, row.utmy)
        for _, row in estaciones.iterrows()
    }

    # For each fire, rank all stations by distance
    fire_station_ranks = []
    for _, fire in inc.iterrows():
        fx, fy = fire.utmx, fire.utmy
        dists = []
        for eid, (ex, ey) in est_coords.items():
            d = dist_utm(fx, fy, ex, ey)
            dists.append((eid, d))
        dists.sort(key=lambda x: x[1])
        fire_station_ranks.append(dists)

    print(f"  Ranking calculado para {len(fire_station_ranks):,} incendios x {len(est_coords)} estaciones")

    # --- STEP 3: Build index of available meteo data by (date, station) ---
    print(f"\n--- PASO 3: Indice de disponibilidad meteorologica ---")
    meteo["fecha_str"] = meteo["fecha"].dt.strftime("%Y-%m-%d")

    # "Tiene datos utiles" = tiene las 3 variables base presentes en TODAS las estaciones:
    # temperatura, humedad y precipitacion. El viento se trata aparte porque 4 estaciones
    # no tienen anemometro (10111, 10113, 10114, 10148) y nunca tendran ese dato.
    BASIC_VARS = ["TA_AVG_1.5m", "HR_AVG_1.5m", "PP_SUM_1.5m"]

    # Vectorizado: una sola pasada en lugar de 3 iterrows sobre 106k filas
    meteo["estacion_id_int"] = meteo["estacion_id"].astype(int)
    has_basics = meteo[BASIC_VARS].notna().all(axis=1)
    has_wind = meteo["VV_AVG_10m"].notna()

    meteo_available = set(zip(meteo.loc[has_basics, "fecha_str"],
                              meteo.loc[has_basics, "estacion_id_int"]))
    meteo_wind_available = set(zip(meteo.loc[has_wind, "fecha_str"],
                                   meteo.loc[has_wind, "estacion_id_int"]))

    # Index meteo como dict: (date_str, station_id) -> row dict
    meteo_index = {
        (row["fecha_str"], row["estacion_id_int"]): row
        for _, row in meteo.iterrows()
    }

    print(f"  Pares (fecha, estacion) con datos basicos (T+HR+PP): {len(meteo_available):,}")
    print(f"  Pares (fecha, estacion) con viento:                  {len(meteo_wind_available):,}")

    # --- STEP 4: For each fire, find nearest station WITH data for that date ---
    print(f"\n--- PASO 4: Asignacion con fallback ---")
    print(f"  Estrategia: si la estacion mas cercana no tiene datos para esa fecha,")
    print(f"  se prueba la siguiente mas cercana, y asi sucesivamente.")

    results = []
    stats = {
        "matched_1st": 0, "matched_2nd": 0, "matched_3rd_plus": 0,
        "no_match": 0, "wind_from_alt": 0,
    }

    for i, (_, fire) in enumerate(inc.iterrows()):
        fecha_str = fire.fecha.strftime("%Y-%m-%d")
        ranks = fire_station_ranks[i]

        # Find nearest station with data for this date
        matched_est = None
        matched_dist = None
        matched_rank = None

        for rank_idx, (eid, dist) in enumerate(ranks):
            if (fecha_str, eid) in meteo_available:
                matched_est = eid
                matched_dist = dist
                matched_rank = rank_idx
                break

        if matched_est is None:
            stats["no_match"] += 1
            row = {
                "id": fire.id, "fecha": fire.fecha, "lat": fire.lat, "lng": fire.lng,
                "municipio": fire.municipio, "superficie": fire.superficie,
                "causa": fire.causa, "causa_supuesta": fire.causa_supuesta,
                "estacion_id": None, "dist_estacion_m": None,
                "estacion_rank": None,
                "estacion_viento_id": None, "dist_viento_m": None,
            }
            for var in METEO_VARS:
                row[var] = None
            results.append(row)
            continue

        if matched_rank == 0:
            stats["matched_1st"] += 1
        elif matched_rank == 1:
            stats["matched_2nd"] += 1
        else:
            stats["matched_3rd_plus"] += 1

        # Get meteo data from matched station
        meteo_row = meteo_index[(fecha_str, matched_est)]

        row = {
            "id": fire.id, "fecha": fire.fecha, "lat": fire.lat, "lng": fire.lng,
            "municipio": fire.municipio, "superficie": fire.superficie,
            "causa": fire.causa, "causa_supuesta": fire.causa_supuesta,
            "estacion_id": matched_est, "dist_estacion_m": matched_dist,
            "estacion_rank": matched_rank + 1,  # 1-based
            "estacion_viento_id": matched_est, "dist_viento_m": matched_dist,
        }

        for var in METEO_VARS:
            row[var] = meteo_row.get(var)

        # If matched station has no wind, find nearest station WITH wind for this date
        if matched_est in SIN_VIENTO or pd.isna(row.get("VV_AVG_10m")):
            for eid, dist in ranks:
                if (fecha_str, eid) in meteo_wind_available:
                    wind_row = meteo_index[(fecha_str, eid)]
                    row["VV_AVG_10m"] = wind_row.get("VV_AVG_10m")
                    row["VV_MAX_10m"] = wind_row.get("VV_MAX_10m")
                    row["estacion_viento_id"] = eid
                    row["dist_viento_m"] = dist
                    stats["wind_from_alt"] += 1
                    break

        results.append(row)

    result_df = pd.DataFrame(results)

    print(f"\n  Resultados de la asignacion:")
    print(f"    Asignados a 1a estacion mas cercana:  {stats['matched_1st']:,}")
    print(f"    Asignados a 2a estacion mas cercana:  {stats['matched_2nd']:,}")
    print(f"    Asignados a 3a+ estacion mas cercana: {stats['matched_3rd_plus']:,}")
    print(f"    Sin ningun dato meteorologico:        {stats['no_match']:,}")
    print(f"    Viento desde estacion alternativa:    {stats['wind_from_alt']:,}")

    total_matched = stats["matched_1st"] + stats["matched_2nd"] + stats["matched_3rd_plus"]
    print(f"\n    TOTAL CON DATOS: {total_matched:,} de {len(inc):,} ({total_matched/len(inc)*100:.1f}%)")

    # --- STEP 5: Final stats and save ---
    print(f"\n--- PASO 5: Estadisticas finales ---")
    print(f"  Registros: {len(result_df):,}")

    print(f"\n  Disponibilidad por variable:")
    for var in METEO_VARS:
        nn = result_df[var].notna().sum()
        print(f"    {var:<20} {nn:>6,} ({nn/len(result_df)*100:.1f}%)")

    # Distance stats for matched fires
    matched_df = result_df[result_df.estacion_id.notna()]
    print(f"\n  Distancia a estacion asignada (solo los {len(matched_df):,} con datos):")
    print(f"    Mediana: {matched_df.dist_estacion_m.median()/1000:.1f} km")
    print(f"    Media:   {matched_df.dist_estacion_m.mean()/1000:.1f} km")
    print(f"    P95:     {matched_df.dist_estacion_m.quantile(0.95)/1000:.1f} km")
    print(f"    Max:     {matched_df.dist_estacion_m.max()/1000:.1f} km")

    # Rank distribution
    print(f"\n  Distribucion del ranking de estacion usada:")
    for rank, count in result_df.estacion_rank.value_counts().sort_index().items():
        pct = count / len(matched_df) * 100
        label = "mas cercana" if rank == 1 else f"{int(rank)}a mas cercana"
        print(f"    Estacion {label}: {count:,} ({pct:.1f}%)")

    # Save
    print(f"\n--- GUARDADO ---")
    result_df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"  Fichero: {OUTPUT_CSV}")
    print(f"  Tamano: {size_mb:.2f} MB")
    print(f"  Filas: {len(result_df):,}, Columnas: {len(result_df.columns)}")


if __name__ == "__main__":
    main()
