#!/usr/bin/env python3
"""
Paso 8.1 — Limpieza del dataset de incendios forestales (Civio/EGIF).

Filtra incendios de Ourense desde 2005, elimina registros sin coordenadas,
y guarda el resultado limpio en data/processed/incendios_ourense.csv.

Uso: python3 src/limpiar_incendios.py
"""

import pandas as pd
import os

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_CSV = os.path.join(PROJECT_DIR, "data", "raw", "fires-all.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "incendios_ourense.csv")


def main():
    print("=" * 70)
    print("LIMPIEZA DEL DATASET DE INCENDIOS")
    print("=" * 70)

    # 1. Cargar CSV completo
    df = pd.read_csv(INPUT_CSV)
    print(f"\n1. CSV cargado: {len(df):,} registros totales")

    # 2. Filtrar solo provincia de Ourense (codigo 32)
    ou = df[df.idprovincia == 32].copy()
    print(f"2. Filtro Ourense (idprovincia=32): {len(ou):,} registros")

    # 3. Convertir fecha a datetime
    ou["fecha"] = pd.to_datetime(ou["fecha"], errors="coerce")
    fechas_invalidas = ou["fecha"].isna().sum()
    print(f"3. Conversion de fechas: {fechas_invalidas} fechas invalidas")

    # 4. Filtrar desde 2005 (solapamiento con MeteoGalicia)
    ou = ou[ou.fecha >= "2005-01-01"]
    print(f"4. Filtro desde 2005-01-01: {len(ou):,} registros")

    # 5. Convertir lat/lng a float y eliminar registros sin coordenadas
    ou["lat"] = pd.to_numeric(ou["lat"], errors="coerce")
    ou["lng"] = pd.to_numeric(ou["lng"], errors="coerce")
    sin_coords = ou[ou["lat"].isna() | ou["lng"].isna()]
    print(f"5. Registros sin coordenadas: {len(sin_coords)}")
    ou = ou.dropna(subset=["lat", "lng"])

    # 6. Eliminar duplicados por id
    dupes = ou.duplicated(subset=["id"]).sum()
    print(f"6. Duplicados por id: {dupes}")
    ou = ou.drop_duplicates(subset=["id"])

    # 7. Verificacion de integridad
    print(f"\n--- Verificacion final ---")
    print(f"Registros finales: {len(ou):,}")
    print(f"Rango temporal: {ou.fecha.min().date()} a {ou.fecha.max().date()}")
    print(f"Anios: {sorted(ou.fecha.dt.year.unique().tolist())}")
    print(f"Coordenadas lat: [{ou.lat.min():.4f}, {ou.lat.max():.4f}]")
    print(f"Coordenadas lng: [{ou.lng.min():.4f}, {ou.lng.max():.4f}]")
    print(f"Municipios unicos: {ou.municipio.nunique()}")
    print(f"Nulos por columna:")
    for col in ou.columns:
        n = ou[col].isna().sum()
        if n > 0:
            print(f"  {col}: {n} ({n/len(ou)*100:.1f}%)")

    # 8. Seleccionar columnas utiles (descartar las que no usaremos)
    columnas_utiles = [
        "id", "fecha", "lat", "lng", "latlng_explicit",
        "idprovincia", "idmunicipio", "municipio",
        "superficie", "causa", "causa_supuesta",
    ]
    ou = ou[columnas_utiles]
    print(f"\n8. Columnas seleccionadas ({len(columnas_utiles)}): {columnas_utiles}")

    # 9. Guardar
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ou.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\n9. Guardado en: {OUTPUT_CSV}")
    print(f"   Tamano: {size_mb:.2f} MB")
    print(f"   Registros: {len(ou):,}")


if __name__ == "__main__":
    main()
