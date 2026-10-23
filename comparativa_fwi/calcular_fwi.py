"""Calcular FWI canadiense (Van Wagner 1987) sobre meteo_ourense.csv.

Implementación directa de las fórmulas oficiales del Servicio Forestal Canadiense
(Forest Technical Report 35, Van Wagner 1987) sin dependencias externas más allá de
pandas y pyproj.

Inputs (por estación-día):
    T   : temperatura media a 1.5 m (°C)
    HR  : humedad relativa media a 1.5 m (%)
    W   : viento medio a 10 m (km/h)
    P   : precipitación 24h (mm)
    lat : latitud (°N)
    mes : 1-12

Outputs (por estación-día):
    FFMC, DMC, DC, ISI, BUI, FWI

NaN: forward-fill por estación (gaps cortos típicos en MeteoGalicia).
Viento: 4 estaciones sin anemómetro → imputación con media mensual cross-estación.
Spinup: inicialización canónica (FFMC=85, DMC=6, DC=15) el primer día por estación,
descartando los primeros 30 días como burn-in.
"""
import math
import pandas as pd
from pyproj import Transformer
from pathlib import Path

PROJ = Path(__file__).parent.parent
METEO_CSV = PROJ / "data/processed/meteo_ourense.csv"
ESTAC_CSV = PROJ / "data/processed/estaciones_ourense.csv"
OUT_CSV = Path(__file__).parent / "data/fwi_diario.csv"

# Day length tables Van Wagner 1987 (DMC, DC) for boreal latitudes
EL_DMC = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]  # NH default
EL_DMC_S = [12.8, 12.0, 11.8, 11.6, 11.5, 11.4, 11.4, 11.5, 11.5, 11.7, 11.8, 12.6]
LF_DC = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def ffmc(t, h, w, p, ffmc_prev):
    """Fine Fuel Moisture Code (Van Wagner 1987)."""
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if p > 0.5:
        rf = p - 0.5
        if mo > 150.0:
            mo += 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf)) \
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        else:
            mo += 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 250.0:
            mo = 250.0
    ed = 0.942 * h ** 0.679 + 11.0 * math.exp((h - 100.0) / 10.0) \
         + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * h))
    if mo > ed:
        ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) \
             + 0.0694 * math.sqrt(w) * (1.0 - (h / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * t)
        m = ed + (mo - ed) * 10.0 ** -kd
    else:
        ew = 0.618 * h ** 0.753 + 10.0 * math.exp((h - 100.0) / 10.0) \
             + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * h))
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) \
                 + 0.0694 * math.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
            kw = kl * 0.581 * math.exp(0.0365 * t)
            m = ew - (ew - mo) * 10.0 ** -kw
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)


def dmc(t, h, p, dmc_prev, mes, lat):
    """Duff Moisture Code."""
    if p > 1.5:
        re = 0.92 * p - 1.27
        mo = 20.0 + math.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * math.log(mr - 20.0)
        pr = max(pr, 0.0)
    else:
        pr = dmc_prev
    el = EL_DMC[mes - 1] if lat > 0 else EL_DMC_S[mes - 1]
    if t < -1.1:
        t = -1.1
    k = 1.894 * (t + 1.1) * (100.0 - h) * el * 1e-6
    return pr + 100.0 * k


def dc(t, p, dc_prev, mes, lat):
    """Drought Code."""
    if p > 2.8:
        rd = 0.83 * p - 1.27
        qo = 800.0 * math.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * math.log(800.0 / qr)
        dr = max(dr, 0.0)
    else:
        dr = dc_prev
    lf = LF_DC[mes - 1] if lat > 0 else LF_DC[(mes + 5) % 12]
    if t < -2.8:
        t = -2.8
    v = 0.36 * (t + 2.8) + lf
    v = max(v, 0.0)
    return dr + 0.5 * v


def isi(w, ffmc_v):
    """Initial Spread Index."""
    fw = math.exp(0.05039 * w)
    m = 147.2 * (101.0 - ffmc_v) / (59.5 + ffmc_v)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * fw * ff


def bui(dmc_v, dc_v):
    """Buildup Index."""
    if dmc_v <= 0.4 * dc_v:
        b = 0.8 * dmc_v * dc_v / (dmc_v + 0.4 * dc_v + 1e-9)
    else:
        b = dmc_v - (1.0 - 0.8 * dc_v / (dmc_v + 0.4 * dc_v + 1e-9)) \
            * (0.92 + (0.0114 * dmc_v) ** 1.7)
    return max(b, 0.0)


def fwi(isi_v, bui_v):
    """Fire Weather Index."""
    if bui_v <= 80.0:
        bb = 0.1 * isi_v * (0.626 * bui_v ** 0.809 + 2.0)
    else:
        bb = 0.1 * isi_v * (1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui_v)))
    if bb > 1.0:
        return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
    return bb


def utm_a_lat(utmx, utmy, huso=29):
    """ETRS89 UTM huso 29N → lat/lon."""
    t = Transformer.from_crs(f"epsg:258{huso}", "epsg:4326", always_xy=True)
    lon, lat = t.transform(utmx, utmy)
    return lat, lon


def main():
    print("Cargando datos…")
    meteo = pd.read_csv(METEO_CSV, parse_dates=["fecha"])
    est = pd.read_csv(ESTAC_CSV)
    est["lat"], est["lon"] = zip(*[utm_a_lat(x, y) for x, y in zip(est.utmx, est.utmy)])
    print(f"  {len(meteo):,} filas meteo · {len(est)} estaciones")
    print(f"  lat estaciones: {est.lat.min():.2f}-{est.lat.max():.2f}°N")

    # Imputación viento: estaciones sin anemómetro → media mensual cross-estación
    meteo["mes"] = meteo.fecha.dt.month
    viento_mes = meteo.groupby("mes")["VV_AVG_10m"].transform("mean")
    meteo["W"] = meteo["VV_AVG_10m"].fillna(viento_mes)
    # km/h ← m/s
    meteo["W_kmh"] = meteo["W"] * 3.6

    # Forward-fill T, HR, P por estación
    meteo = meteo.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)
    for col in ["TA_AVG_1.5m", "HR_AVG_1.5m", "PP_SUM_1.5m"]:
        meteo[col] = meteo.groupby("estacion_id")[col].ffill().bfill()

    # Renombrar columnas a nombres seguros para itertuples
    meteo = meteo.rename(columns={
        "TA_AVG_1.5m": "T",
        "HR_AVG_1.5m": "H",
        "PP_SUM_1.5m": "P",
    })

    print("Calculando FWI por estación…")
    rows = []
    for sid, g in meteo.groupby("estacion_id"):
        lat = est.loc[est.estacion_id == sid, "lat"].iloc[0]
        ffmc_p, dmc_p, dc_p = 85.0, 6.0, 15.0  # spinup canónico
        for r in g.itertuples():
            t = r.T if not pd.isna(r.T) else 10.0
            h = max(min(r.H if not pd.isna(r.H) else 70.0, 100.0), 0.0)
            w = r.W_kmh if not pd.isna(r.W_kmh) else 10.0
            p = r.P if not pd.isna(r.P) else 0.0
            mes = r.fecha.month
            try:
                ffmc_v = ffmc(t, h, w, p, ffmc_p)
                ffmc_v = max(min(ffmc_v, 101.0), 0.0)
                dmc_v = dmc(t, h, p, dmc_p, mes, lat)
                dmc_v = max(dmc_v, 0.0)
                dc_v = dc(t, p, dc_p, mes, lat)
                dc_v = max(dc_v, 0.0)
                isi_v = isi(w, ffmc_v)
                bui_v = bui(dmc_v, dc_v)
                fwi_v = fwi(isi_v, bui_v)
            except (ValueError, OverflowError):
                ffmc_v = ffmc_p; dmc_v = dmc_p; dc_v = dc_p
                isi_v = bui_v = fwi_v = float("nan")
            rows.append((sid, r.fecha, ffmc_v, dmc_v, dc_v, isi_v, bui_v, fwi_v))
            ffmc_p, dmc_p, dc_p = ffmc_v, dmc_v, dc_v

    out = pd.DataFrame(rows, columns=["estacion_id", "fecha",
                                       "ffmc", "dmc", "dc", "isi", "bui", "fwi"])
    # Burn-in: descartar primeros 30 días por estación
    out["dia_idx"] = out.groupby("estacion_id").cumcount()
    out = out[out.dia_idx >= 30].drop(columns="dia_idx")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nGuardado: {OUT_CSV}")
    print(f"  {len(out):,} filas · {out.fwi.notna().sum():,} con FWI válido")
    print(f"\nFWI agosto (sanity check, debería ser ~10-30):")
    print(out.assign(mes=out.fecha.dt.month)
             .query("mes==8").groupby(out.fecha.dt.year)["fwi"]
             .agg(["mean", "median", "max"]).round(1))
    print(f"\nFWI febrero (debería ser bajo, ~1-5):")
    print(out.assign(mes=out.fecha.dt.month)
             .query("mes==2").groupby(out.fecha.dt.year)["fwi"]
             .agg(["mean", "median", "max"]).round(1))


if __name__ == "__main__":
    main()
