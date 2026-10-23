#!/usr/bin/env python3
"""
Descarga datos meteorologicos diarios de MeteoGalicia para las estaciones de Ourense.
Guarda un JSON por estacion con todos los datos del rango 2005-2021.

Uso: python3 src/descargar_meteo.py
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from config import STATIONS as ESTACIONES, METEO_RAW_DIR as OUTPUT_DIR, API_URL, YEAR_END as FIN, API_PAUSE_SECONDS


def descargar_estacion(est_id, info):
    """Descarga todos los datos diarios de una estacion, mes a mes."""
    todos_los_dias = []
    inicio = info["desde"]
    errores = 0

    for year in range(inicio, FIN + 1):
        for month in range(1, 13):
            # Primer y último día del mes
            fecha_ini = f"01/{month:02d}/{year}"
            if month == 12:
                fecha_fin = f"31/12/{year}"
            else:
                # Último día del mes: primer día del mes siguiente - 1
                siguiente = datetime(year, month + 1, 1) - timedelta(days=1)
                fecha_fin = siguiente.strftime("%d/%m/%Y")

            params = {
                "idEst": est_id,
                "dataIni": fecha_ini,
                "dataFin": fecha_fin,
            }

            try:
                r = requests.get(API_URL, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                dias = data.get("listDatosDiarios", [])
                # Solo guardar días que tienen datos de estación
                dias_con_datos = [d for d in dias if d.get("listaEstacions")]
                todos_los_dias.extend(dias_con_datos)
            except Exception as e:
                errores += 1
                print(f"    ERROR {est_id} {fecha_ini}-{fecha_fin}: {e}")

            time.sleep(API_PAUSE_SECONDS)  # Pausa entre llamadas

    return todos_los_dias, errores


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Descargando datos de {len(ESTACIONES)} estaciones de Ourense")
    print(f"Rango: cada estacion desde su año de inicio hasta {FIN}")
    print(f"Destino: {OUTPUT_DIR}")
    print("=" * 60)

    resumen = []

    for est_id, info in sorted(ESTACIONES.items()):
        print(f"\n[{est_id}] {info['nombre']} ({info['concello']}) — desde {info['desde']}...")
        dias, errores = descargar_estacion(est_id, info)

        # Guardar JSON
        output_file = os.path.join(OUTPUT_DIR, f"estacion_{est_id}.json")
        resultado = {
            "idEstacion": est_id,
            "nombre": info["nombre"],
            "concello": info["concello"],
            "rango": f"{info['desde']}-{FIN}",
            "total_dias": len(dias),
            "errores": errores,
            "listDatosDiarios": dias,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False)

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  -> {len(dias)} dias guardados ({size_mb:.1f} MB), {errores} errores")
        resumen.append({"id": est_id, "nombre": info["nombre"], "dias": len(dias), "errores": errores, "mb": size_mb})

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE DESCARGA")
    print("=" * 60)
    total_dias = 0
    total_mb = 0
    for r in resumen:
        estado = "OK" if r['errores'] == 0 else f"{r['errores']} err"
        print(f"  {r['id']:>5} {r['nombre']:<25} {r['dias']:>5} dias  {r['mb']:.1f} MB  {estado}")
        total_dias += r["dias"]
        total_mb += r["mb"]
    print(f"\n  TOTAL: {total_dias} dias, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
