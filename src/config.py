"""
Constantes compartidas del proyecto STEMbach Incendios.

Todas las constantes que usan los scripts estan centralizadas aqui.
Si un valor cambia (rango de fechas, variables seleccionadas, etc.),
solo hay que modificar este fichero.
"""

import os

# === PATHS ===
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METEO_RAW_DIR = os.path.join(RAW_DIR, "meteogalicia")

# Input files
FIRES_CSV = os.path.join(RAW_DIR, "fires-all.csv")

# Output files
INCENDIOS_CSV = os.path.join(PROCESSED_DIR, "incendios_ourense.csv")
METEO_CSV = os.path.join(PROCESSED_DIR, "meteo_ourense.csv")
ESTACIONES_CSV = os.path.join(PROCESSED_DIR, "estaciones_ourense.csv")
DATASET_FUSIONADO_CSV = os.path.join(PROCESSED_DIR, "dataset_fusionado.csv")

# === TEMPORAL RANGE ===
YEAR_START = 2005       # First year with MeteoGalicia data overlap
YEAR_END = 2021         # Last year with fire data in Civio CSV
PROVINCIA_OURENSE = 32  # idprovincia code for Ourense

# === METEOGALICIA API ===
API_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action"
API_LIST_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action"
API_DATE_FORMAT = "%d/%m/%Y"  # dd/MM/yyyy (NOT ISO)
API_PAUSE_SECONDS = 0.5

# === VALIDATION ===
VALID_CODES = {1, 5}    # 1=valid original, 5=valid interpolated
INVALID_CODES = {0, 2, 3, 9}  # 0=unvalidated, 2=suspicious, 3=erroneous, 9=not recorded
INTERPOLATION_LIMIT = 2  # Max consecutive NaN days to interpolate

# === METEOROLOGICAL VARIABLES ===
# 12 variables selected in paso 7 (availability >= 60%)
METEO_VARIABLES = [
    "TA_AVG_1.5m",    # Average temperature (C)
    "TA_MAX_1.5m",    # Maximum temperature (C)
    "TA_MIN_1.5m",    # Minimum temperature (C)
    "HR_AVG_1.5m",    # Average relative humidity (%)
    "HR_MIN_1.5m",    # Minimum relative humidity (%)
    "PP_SUM_1.5m",    # Total precipitation (L/m2)
    "VV_AVG_10m",     # Average wind speed (m/s)
    "VV_MAX_10m",     # Maximum wind gust (m/s)
    "IRD_SUM_1.5m",   # Global irradiation (10kJ/m2/day)
    "HSOL_SUM_1.5m",  # Sunshine hours (h)
    "TO_AVG_1.5m",    # Dew point temperature (C)
    "PR_AVG_1.5m",    # Atmospheric pressure (hPa)
]

# Variables discarded (availability < 60%)
# ET0_SUM_1.5m (evapotranspiration, 55.2%)
# BH_SUM_1.5m (water balance, 58.7%)

WIND_VARIABLES = ["VV_AVG_10m", "VV_MAX_10m"]

# === OUTLIER RANGES (physical limits) ===
OUTLIER_RANGES = {
    "TA_AVG_1.5m":   (-15, 50),
    "TA_MAX_1.5m":   (-15, 50),
    "TA_MIN_1.5m":   (-15, 50),
    "HR_AVG_1.5m":   (0, 100),
    "HR_MIN_1.5m":   (0, 100),
    "PP_SUM_1.5m":   (0, 500),
    "VV_AVG_10m":    (0, 50),
    "VV_MAX_10m":    (0, 80),
    "IRD_SUM_1.5m":  (0, 4000),
    "HSOL_SUM_1.5m": (0, 18),
    "TO_AVG_1.5m":   (-25, 40),
    "PR_AVG_1.5m":   (750, 1080),  # 750 hPa accounts for high-altitude stations
}

# === STATIONS ===
# Stations WITHOUT wind measurement (no anemometer)
STATIONS_WITHOUT_WIND = {10111, 10113, 10114, 10148}

# All 20 Ourense stations (series 10xxx) with start year
STATIONS = {
    10048: {"nombre": "O Invernadeiro", "concello": "VILARIÑO DE CONSO", "desde": 2005},
    10057: {"nombre": "Alto do Rodicio", "concello": "MACEDA", "desde": 2005},
    10058: {"nombre": "Verín-Vilamaior", "concello": "VERÍN", "desde": 2005},
    10109: {"nombre": "Amiudal", "concello": "AVIÓN", "desde": 2006},
    10110: {"nombre": "Baltar", "concello": "BALTAR", "desde": 2006},
    10111: {"nombre": "Entrimo", "concello": "ENTRIMO", "desde": 2006},
    10112: {"nombre": "Gandarela", "concello": "CELANOVA", "desde": 2006},
    10113: {"nombre": "Monte Medo", "concello": "BAÑOS DE MOLGAS", "desde": 2006},
    10114: {"nombre": "As Petarelas", "concello": "RUBIÁ", "desde": 2006},
    10115: {"nombre": "Serra do Eixe", "concello": "A VEIGA", "desde": 2006},
    10116: {"nombre": "Viana do Bolo", "concello": "VIANA DO BOLO", "desde": 2006},
    10117: {"nombre": "San Xoán de Río", "concello": "SAN XOÁN DE RÍO", "desde": 2006},
    10119: {"nombre": "Xurés", "concello": "MUÍÑOS", "desde": 2006},
    10130: {"nombre": "Lardeira", "concello": "CARBALLEDA DE VALDEORRAS", "desde": 2008},
    10131: {"nombre": "Cabeza de Manzaneda", "concello": "MANZANEDA", "desde": 2007},
    10138: {"nombre": "Xares", "concello": "A VEIGA", "desde": 2008},
    10148: {"nombre": "Ourense", "concello": "OURENSE", "desde": 2012},
    10155: {"nombre": "Ourense-Estacións", "concello": "OURENSE", "desde": 2015},
    10204: {"nombre": "Casaio", "concello": "CARBALLEDA DE VALDEORRAS", "desde": 2013},
    10500: {"nombre": "Verín-Vilela", "concello": "VERÍN", "desde": 2009},
}

# === UTM CONVERSION (Zone 29N) ===
UTM_ZONE = "29N"
UTM_CENTRAL_MERIDIAN = -9  # degrees
