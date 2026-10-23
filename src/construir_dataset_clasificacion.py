#!/usr/bin/env python3
"""
Paso 9 — Construccion del dataset de clasificacion binaria.

1. Carga meteo_ourense.csv (todos los dias-estacion) y dataset_fusionado.csv (incendios)
2. Etiqueta cada dia-estacion como incendio (1) o no-incendio (0)
3. Aplica ventana de exclusion ±3 dias alrededor de incendios
4. Calcula 10 features derivadas (sequia, tendencias, calendario)
5. Guarda dataset_modelo.csv listo para entrenar

Uso: python3 src/construir_dataset_clasificacion.py
"""

import pandas as pd
import numpy as np
import os

# Import from config
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PROCESSED_DIR, METEO_CSV, DATASET_FUSIONADO_CSV, ESTACIONES_CSV,
    METEO_VARIABLES, WIND_VARIABLES, STATIONS_WITHOUT_WIND,
)

OUTPUT_CSV = os.path.join(PROCESSED_DIR, "dataset_modelo.csv")

# Exclusion window: ±N days around each fire are neither positive nor negative
EXCLUSION_WINDOW_DAYS = 3

# Dry day threshold (mm)
DRY_THRESHOLD = 1.0


def consecutive_dry_days(precip_series, threshold=DRY_THRESHOLD):
    """Count consecutive days with precipitation below threshold."""
    result = np.zeros(len(precip_series), dtype=int)
    count = 0
    for i, val in enumerate(precip_series.values):
        if pd.notna(val) and val < threshold:
            count += 1
        else:
            count = 0
        result[i] = count
    return result


def compute_derived_features(df):
    """
    Compute 10 derived features for a single station's time series.
    DataFrame must be sorted by date with no duplicate dates.
    """
    # Drought indicators
    df["dias_sin_lluvia"] = consecutive_dry_days(df["PP_SUM_1.5m"])
    df["PP_acum_7d"] = df["PP_SUM_1.5m"].rolling(7, min_periods=4).sum()
    df["PP_acum_14d"] = df["PP_SUM_1.5m"].rolling(14, min_periods=7).sum()
    df["PP_acum_30d"] = df["PP_SUM_1.5m"].rolling(30, min_periods=15).sum()

    # Temperature trends
    df["TA_AVG_3d"] = df["TA_AVG_1.5m"].rolling(3, min_periods=2).mean()
    df["TA_MAX_3d"] = df["TA_MAX_1.5m"].rolling(3, min_periods=2).max()

    # Humidity trend
    df["HR_MIN_3d"] = df["HR_MIN_1.5m"].rolling(3, min_periods=2).min()

    # Thermal amplitude
    df["amplitud_termica"] = df["TA_MAX_1.5m"] - df["TA_MIN_1.5m"]

    # Calendar features
    df["mes"] = df["fecha"].dt.month
    df["dia_del_ano"] = df["fecha"].dt.dayofyear

    return df


def main():
    print("=" * 70)
    print("CONSTRUCCION DEL DATASET DE CLASIFICACION")
    print("=" * 70)

    # --- LOAD DATA ---
    meteo = pd.read_csv(METEO_CSV, parse_dates=["fecha"])
    fusionado = pd.read_csv(DATASET_FUSIONADO_CSV, parse_dates=["fecha"])
    estaciones = pd.read_csv(ESTACIONES_CSV)

    print(f"\nDatos cargados:")
    print(f"  Meteo: {len(meteo):,} filas")
    print(f"  Fusionado: {len(fusionado):,} incendios")
    print(f"  Estaciones: {len(estaciones)}")

    # --- STEP 1: Identify fire pairs (date, station) ---
    print(f"\n--- PASO 1: Identificar pares con incendio ---")

    fusionado["fecha_str"] = fusionado["fecha"].dt.strftime("%Y-%m-%d")
    fire_pairs = set()
    for _, row in fusionado.iterrows():
        if pd.notna(row.estacion_id):
            fire_pairs.add((row.fecha_str, int(row.estacion_id)))

    print(f"  Pares unicos (fecha, estacion) con incendio: {len(fire_pairs):,}")
    print(f"  ({len(fusionado):,} incendios -> {len(fire_pairs):,} pares, porque varios incendios "
          f"pueden caer en la misma estacion el mismo dia)")

    # --- STEP 2: Build exclusion window ---
    print(f"\n--- PASO 2: Ventana de exclusion ±{EXCLUSION_WINDOW_DAYS} dias ---")

    from datetime import timedelta
    exclusion_pairs = set()
    for fecha_str, est_id in fire_pairs:
        fecha = pd.Timestamp(fecha_str)
        for delta in range(-EXCLUSION_WINDOW_DAYS, EXCLUSION_WINDOW_DAYS + 1):
            excl_date = (fecha + timedelta(days=delta)).strftime("%Y-%m-%d")
            exclusion_pairs.add((excl_date, est_id))

    # Fire days themselves are NOT excluded (they're positives)
    exclusion_only = exclusion_pairs - fire_pairs
    print(f"  Pares en zona de exclusion: {len(exclusion_only):,}")

    # --- STEP 3: Label all meteo rows ---
    print(f"\n--- PASO 3: Etiquetar todas las filas ---")

    meteo["fecha_str"] = meteo["fecha"].dt.strftime("%Y-%m-%d")
    meteo["estacion_id_int"] = meteo["estacion_id"].astype(int)
    meteo["_pair"] = list(zip(meteo["fecha_str"], meteo["estacion_id_int"]))

    # Label: 1 = fire, 0 = no fire, -1 = exclusion zone
    labels = []
    for pair in meteo["_pair"]:
        if pair in fire_pairs:
            labels.append(1)
        elif pair in exclusion_only:
            labels.append(-1)
        else:
            labels.append(0)

    meteo["incendio"] = labels

    n_pos = (meteo["incendio"] == 1).sum()
    n_neg = (meteo["incendio"] == 0).sum()
    n_excl = (meteo["incendio"] == -1).sum()
    print(f"  Positivos (incendio=1):     {n_pos:,}")
    print(f"  Negativos (incendio=0):     {n_neg:,}")
    print(f"  Excluidos (ventana):        {n_excl:,}")

    # Remove excluded rows
    df = meteo[meteo["incendio"] >= 0].copy()
    print(f"  Filas tras exclusion:       {len(df):,}")
    print(f"  Ratio negativo:positivo:    {n_neg/n_pos:.1f}:1")

    # --- STEP 4: Compute derived features ---
    print(f"\n--- PASO 4: Features derivadas (por estacion) ---")

    # Las rolling features se calculan sobre la serie temporal continua
    # de cada estacion (sin huecos), y solo despues filtramos las filas
    # excluidas por la ventana ±3 dias. Si filtrasemos antes, los rolling
    # se calcularian sobre series con huecos y darian valores incorrectos.

    df_all = meteo.copy()
    df_all = df_all.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

    all_derived = []
    for est_id in sorted(df_all["estacion_id"].unique()):
        est_data = df_all[df_all["estacion_id"] == est_id].copy()
        est_data = est_data.sort_values("fecha").reset_index(drop=True)
        est_data = compute_derived_features(est_data)
        all_derived.append(est_data)

    df_all = pd.concat(all_derived, ignore_index=True)

    # Ahora filtramos filas excluidas (incendio == -1)
    df = df_all[df_all["incendio"] >= 0].copy()

    derived_cols = [
        "dias_sin_lluvia", "PP_acum_7d", "PP_acum_14d", "PP_acum_30d",
        "TA_AVG_3d", "TA_MAX_3d", "HR_MIN_3d", "amplitud_termica",
        "mes", "dia_del_ano",
    ]

    print(f"  Features calculadas sobre series continuas, luego filtradas")
    print(f"\n  Disponibilidad de features derivadas:")
    for col in derived_cols:
        nn = df[col].notna().sum()
        pct = nn / len(df) * 100
        print(f"    {col:<20} {nn:>8,} ({pct:.1f}%)")

    # --- STEP 5: Select final columns and save ---
    print(f"\n--- PASO 5: Seleccion de columnas y guardado ---")

    final_cols = (
        ["fecha", "estacion_id", "incendio"]
        + METEO_VARIABLES
        + derived_cols
    )

    result = df[final_cols].copy()

    # Final stats
    n_pos_final = (result["incendio"] == 1).sum()
    n_neg_final = (result["incendio"] == 0).sum()

    print(f"\n  Dataset final:")
    print(f"    Filas:     {len(result):,}")
    print(f"    Columnas:  {len(result.columns)}")
    print(f"    Positivos: {n_pos_final:,} ({n_pos_final/len(result)*100:.1f}%)")
    print(f"    Negativos: {n_neg_final:,} ({n_neg_final/len(result)*100:.1f}%)")
    print(f"    Ratio:     {n_neg_final/n_pos_final:.1f}:1")
    print(f"    Rango:     {result.fecha.min()} a {result.fecha.max()}")
    print(f"    Estaciones: {result.estacion_id.nunique()}")

    print(f"\n  Disponibilidad completa:")
    for col in result.columns:
        if col in ("fecha", "estacion_id", "incendio"):
            continue
        nn = result[col].notna().sum()
        pct = nn / len(result) * 100
        print(f"    {col:<20} {nn:>8,} ({pct:.1f}%)")

    # Save
    result.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\n  Guardado: {OUTPUT_CSV}")
    print(f"  Tamano:   {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
