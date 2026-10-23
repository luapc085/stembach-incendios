#!/usr/bin/env python3
"""
Paso 8.2-8.3 — Parseo y limpieza de datos meteorologicos de MeteoGalicia.

Lee los 20 JSONs de estaciones, extrae las 12 variables seleccionadas,
filtra por codigo de validacion, elimina outliers, interpola huecos aislados,
y guarda el resultado en data/processed/meteo_ourense.csv.

Uso: python3 src/limpiar_meteo.py
"""

import json
import glob
import os
import pandas as pd
import numpy as np

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
METEO_DIR = os.path.join(PROJECT_DIR, "data", "raw", "meteogalicia")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "meteo_ourense.csv")

# 12 variables seleccionadas en el paso 7 (evaluacion)
VARIABLES = [
    "TA_AVG_1.5m",    # Temperatura media (C)
    "TA_MAX_1.5m",    # Temperatura maxima (C)
    "TA_MIN_1.5m",    # Temperatura minima (C)
    "HR_AVG_1.5m",    # Humedad relativa media (%)
    "HR_MIN_1.5m",    # Humedad relativa minima (%)
    "PP_SUM_1.5m",    # Precipitacion (L/m2)
    "VV_AVG_10m",     # Velocidad viento media (m/s)
    "VV_MAX_10m",     # Racha maxima viento (m/s)
    "IRD_SUM_1.5m",   # Irradiacion global (10kJ/m2/dia)
    "HSOL_SUM_1.5m",  # Horas de sol (h)
    "TO_AVG_1.5m",    # Temperatura de rocio (C)
    "PR_AVG_1.5m",    # Presion atmosferica (hPa)
]

# Codigos de validacion aceptables
# 1 = dato valido original, 5 = dato valido interpolado
VALID_CODES = {1, 5}

# Rangos fisicos razonables para deteccion de outliers
RANGOS = {
    "TA_AVG_1.5m":  (-15, 50),
    "TA_MAX_1.5m":  (-15, 50),
    "TA_MIN_1.5m":  (-15, 50),
    "HR_AVG_1.5m":  (0, 100),
    "HR_MIN_1.5m":  (0, 100),
    "PP_SUM_1.5m":  (0, 500),    # Max historico Galicia ~300 L/m2/dia
    "VV_AVG_10m":   (0, 50),     # 50 m/s = 180 km/h
    "VV_MAX_10m":   (0, 80),     # Rachas extremas
    "IRD_SUM_1.5m": (0, 4000),   # Max teorico ~3500 en verano
    "HSOL_SUM_1.5m":(0, 18),     # Max horas dia en verano Galicia ~16h
    "TO_AVG_1.5m":  (-25, 40),
    "PR_AVG_1.5m":  (750, 1080), # 750 hPa = ~2500m altitud, margen amplio
}


def parsear_estacion(filepath):
    """Parse a station JSON and extract daily measurements for selected variables."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    est_id = data["idEstacion"]
    nombre = data["nombre"]
    dias = data["listDatosDiarios"]

    # Get UTM coordinates from first day with data
    utmx, utmy = None, None
    for dia in dias:
        ests = dia.get("listaEstacions", [])
        if ests:
            utmx_raw = ests[0].get("utmx")
            utmy_raw = ests[0].get("utmy")
            if utmx_raw is not None:
                utmx = float(utmx_raw)
                utmy = float(utmy_raw)
                break

    rows = []
    stats = {"total": 0, "valid": 0, "invalid_code": 0, "null_value": 0}

    for dia in dias:
        fecha = dia.get("data", "")[:10]  # "2005-01-01T00:00:00" -> "2005-01-01"
        ests = dia.get("listaEstacions", [])
        if not ests:
            continue

        medidas = ests[0].get("listaMedidas", [])
        row = {"fecha": fecha, "estacion_id": est_id}

        for m in medidas:
            cod = m.get("codigoParametro")
            if cod not in VARIABLES:
                continue

            stats["total"] += 1
            val_code = m.get("lnCodigoValidacion", 0)
            valor = m.get("valor")

            if valor is None:
                stats["null_value"] += 1
                continue

            if val_code not in VALID_CODES:
                stats["invalid_code"] += 1
                continue

            stats["valid"] += 1
            row[cod] = valor

        rows.append(row)

    return rows, est_id, nombre, utmx, utmy, stats


def main():
    print("=" * 70)
    print("PARSEO Y LIMPIEZA DE DATOS METEOROLOGICOS")
    print("=" * 70)

    files = sorted(glob.glob(os.path.join(METEO_DIR, "estacion_*.json")))
    print(f"\nEstaciones a procesar: {len(files)}")
    print(f"Variables seleccionadas: {len(VARIABLES)}")

    # --- FASE 1: Parseo ---
    print(f"\n--- FASE 1: Parseo de JSONs ---")
    all_rows = []
    estaciones_meta = []
    total_stats = {"total": 0, "valid": 0, "invalid_code": 0, "null_value": 0}

    for fpath in files:
        rows, est_id, nombre, utmx, utmy, stats = parsear_estacion(fpath)
        all_rows.extend(rows)
        estaciones_meta.append({
            "estacion_id": est_id, "nombre": nombre,
            "utmx": utmx, "utmy": utmy, "dias": len(rows),
        })
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"  {est_id} {nombre:<25} {len(rows):>5} dias, "
              f"{stats['valid']:>6} medidas validas, "
              f"{stats['invalid_code']:>4} filtradas por codigo")

    df = pd.DataFrame(all_rows)
    df["fecha"] = pd.to_datetime(df["fecha"])
    print(f"\nDataFrame bruto: {len(df):,} filas x {len(df.columns)} columnas")
    print(f"Medidas totales evaluadas: {total_stats['total']:,}")
    print(f"  Validas (codigo 1/5): {total_stats['valid']:,} ({total_stats['valid']/total_stats['total']*100:.1f}%)")
    print(f"  Filtradas por codigo: {total_stats['invalid_code']:,}")
    print(f"  Valor null: {total_stats['null_value']:,}")

    # --- FASE 2: Outliers ---
    print(f"\n--- FASE 2: Deteccion y tratamiento de outliers ---")
    outliers_total = 0
    for var, (vmin, vmax) in RANGOS.items():
        if var not in df.columns:
            continue
        mask_outlier = df[var].notna() & ((df[var] < vmin) | (df[var] > vmax))
        n_outliers = mask_outlier.sum()
        if n_outliers > 0:
            # Cap values to range instead of deleting (more conservative)
            print(f"  {var}: {n_outliers} outliers -> capados a [{vmin}, {vmax}]")
            df.loc[df[var] < vmin, var] = vmin
            df.loc[df[var] > vmax, var] = vmax
            outliers_total += n_outliers
    if outliers_total == 0:
        print(f"  Ningun outlier detectado fuera de rangos fisicos")
    else:
        print(f"  Total outliers capados: {outliers_total}")

    # --- FASE 3: Interpolacion de huecos aislados ---
    print(f"\n--- FASE 3: Interpolacion de huecos aislados ---")
    interpolated_total = 0

    # Sort by station and date for proper interpolation
    df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

    for var in VARIABLES:
        if var not in df.columns:
            continue

        before_nans = df[var].isna().sum()

        # Interpolate within each station (not across stations)
        # Use linear interpolation with limit=2 (max 2 consecutive NaNs filled)
        # This preserves physical meaning: if 3+ days are missing, likely a real outage
        df[var] = df.groupby("estacion_id")[var].transform(
            lambda s: s.interpolate(method="linear", limit=2)
        )

        after_nans = df[var].isna().sum()
        filled = before_nans - after_nans
        if filled > 0:
            print(f"  {var}: {filled} huecos interpolados (quedan {after_nans} NaN)")
            interpolated_total += filled

    print(f"  Total huecos interpolados: {interpolated_total}")

    # --- FASE 4: Estadisticas finales ---
    print(f"\n--- RESULTADO FINAL ---")
    print(f"Filas: {len(df):,}")
    print(f"Estaciones: {df.estacion_id.nunique()}")
    print(f"Rango temporal: {df.fecha.min().date()} a {df.fecha.max().date()}")

    print(f"\nDisponibilidad por variable:")
    print(f"  {'Variable':<20} {'No-nulos':>10} {'Nulos':>8} {'%disp':>7}")
    for var in VARIABLES:
        if var in df.columns:
            nn = df[var].notna().sum()
            na = df[var].isna().sum()
            pct = nn / len(df) * 100
            print(f"  {var:<20} {nn:>10,} {na:>8,} {pct:>6.1f}%")
        else:
            print(f"  {var:<20} {'N/A':>10} {'N/A':>8} {'0.0%':>7}")

    # --- Guardar ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\nGuardado en: {OUTPUT_CSV}")
    print(f"Tamano: {size_mb:.2f} MB")

    # Also save station metadata for the fusion step
    meta_df = pd.DataFrame(estaciones_meta)
    meta_csv = os.path.join(OUTPUT_DIR, "estaciones_ourense.csv")
    meta_df.to_csv(meta_csv, index=False)
    print(f"Metadatos estaciones: {meta_csv}")


if __name__ == "__main__":
    main()
