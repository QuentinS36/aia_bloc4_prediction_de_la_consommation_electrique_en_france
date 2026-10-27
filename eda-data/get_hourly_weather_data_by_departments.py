"""
get_hourly_weather_data_by_departments.py
=====================
Fetches hourly weather data (all variables) for all 94 metropolitan French
departments (excluding Corsica), using each department's prefecture as the
measurement point.

A single Open-Meteo /v1/forecast multi-location call retrieves all 94
prefectures at once. Variables that are absent or unavailable for a given
location or timestamp are represented as NaN in the output.

Sources:
  Departments : Wikidata SPARQL — with deduplication fix for departments 91 & 95
                and Paris (Collectivité de Paris) injection fix for dept 75.
                Fallback: embedded INSEE 2021 dataset (94 departments).
  Weather     : Open-Meteo /v1/forecast (free, no API key required)

Usage:
    python get_hourly_weather_data_by_departments.py                   # last 35 days, all variables
    python get_hourly_weather_data_by_departments.py --days 14
    python get_hourly_weather_data_by_departments.py --use-fallback     # skip Wikidata
"""

import argparse
import sys
import unicodedata
import pandas as pd
import requests

from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict


# ===========================================================================
# 1. OPEN-METEO settings
# ===========================================================================

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Maximum past_days supported by the /v1/forecast endpoint
MAX_PAST_DAYS = 92

# All hourly variables to request.
# Variables marked [forecast-only] may return NaN for some past timestamps
# depending on model availability. The API will always return the key in the
# response; missing individual values are represented as null in JSON and
# converted to NaN by pandas.
HOURLY_VARIABLES: list[str] = [
    "temperature_2m",               # Air temperature at 2 m (°C)
    "apparent_temperature",         # Feels-like temperature (°C)
    "relative_humidity_2m",         # Relative humidity (%)
    "dew_point_2m",                 # Dew point temperature (°C)
    "wind_speed_10m",               # Wind speed at 10 m (km/h)
    "wind_direction_10m",           # Wind direction (degrees)
    "wind_gusts_10m",               # Wind gusts at 10 m (km/h)
    "precipitation",                # Total precipitation (mm)
    "rain",                         # Rainfall (mm)
    "snowfall",                     # Snowfall (cm)
    "precipitation_probability",    # Precipitation probability (%) [forecast-only]
    "weather_code",                 # WMO weather interpretation code
    "cloud_cover",                  # Total cloud cover (%)
    "sunshine_duration",            # Sunshine duration per hour (s)
    "uv_index",                     # UV index [forecast-only]
    "uv_index_clear_sky",           # UV index under clear-sky conditions [forecast-only]
    "surface_pressure",             # Surface atmospheric pressure (hPa)
    "visibility",                   # Visibility (m)
    "is_day",                       # Daytime flag: 1 = day, 0 = night
    "et0_fao_evapotranspiration",   # Reference evapotranspiration (mm)
    "cape",                         # CAPE — convective available potential energy (J/kg) [forecast-only]
    "freezing_level_height",        # Freezing level / 0°C isotherm height (m)
]

# WMO weather code -> human-readable label
WMO_CODES: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",        2: "Partly cloudy",          3: "Overcast",
    45: "Fog",                 48: "Depositing rime fog",
    51: "Light drizzle",       53: "Moderate drizzle",      55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Slight rain",         63: "Moderate rain",         65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall",     73: "Moderate snowfall",     75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


# ===========================================================================
# 2. WIKIDATA — Department metadata source
# ===========================================================================

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Key design decisions:
#
# 1. wdt:P31/wdt:P279* wd:Q6465 — transitive subclass traversal.
#    Paris (Collectivité de Paris, Q1142326) is a subclass of Q6465 since the
#    2019 merger. Without /wdt:P279* it is silently excluded.
#
# 2. ?prefPop is fetched alongside coordinates. When Wikidata maps multiple
#    cities as prefecture for the same department (e.g. Évry and
#    Évry-Courcouronnes for dept 91, Cergy and Pontoise for dept 95), Python
#    deduplicates by keeping the most populated one.
#
# 3. ORDER BY ?deptCode DESC(?prefPop) pre-sorts so the most populated
#    prefecture appears first, making deduplication a simple "keep first seen
#    per dept_code" iteration.
WIKIDATA_DEPARTMENTS_QUERY = """
SELECT ?deptCode ?deptLabel ?prefLabel ?prefInsee
       (SAMPLE(?lat) AS ?latitude) (SAMPLE(?lon) AS ?longitude)
       (MAX(?deptPop) AS ?pop) ?prefPop
WHERE {
  ?dept wdt:P31/wdt:P279* wd:Q6465 ;
        wdt:P2586 ?deptCode ;
        wdt:P36   ?pref ;
        wdt:P1082 ?deptPop .
  ?pref wdt:P374  ?prefInsee ;
        wdt:P625  ?coords ;
        wdt:P1082 ?prefPop .
  BIND(geof:latitude(?coords)  AS ?lat)
  BIND(geof:longitude(?coords) AS ?lon)
  FILTER(?deptCode NOT IN ("2A", "2B"))
  FILTER(!STRSTARTS(?deptCode, "97"))
  FILTER(!STRSTARTS(?deptCode, "98"))
  SERVICE wikibase:label { bd:serviceParam wikibase:language "fr,en". }
}
GROUP BY ?deptCode ?deptLabel ?prefLabel ?prefInsee ?prefPop
ORDER BY ?deptCode DESC(?prefPop)
"""


# ===========================================================================
# 2b. FALLBACK — 94 metropolitan departments (INSEE Legal Populations 2021)
# ===========================================================================

FALLBACK_DEPARTMENTS: list[tuple] = [
    ("01", "Ain",                      "Bourg-en-Bresse",        "01053",  46.2057,  5.2297,    650_762),
    ("02", "Aisne",                    "Laon",                   "02408",  49.5647,  3.6244,    524_081),
    ("03", "Allier",                   "Moulins",                "03185",  46.5641,  3.3342,    334_208),
    ("04", "Alpes-de-Haute-Provence",  "Digne-les-Bains",        "04070",  44.0919,  6.2358,    165_197),
    ("05", "Hautes-Alpes",             "Gap",                    "05061",  44.5592,  6.0782,    141_284),
    ("06", "Alpes-Maritimes",          "Nice",                   "06088",  43.7102,  7.2620,  1_094_354),
    ("07", "Ardèche",                  "Privas",                 "07186",  44.7353,  4.5992,    329_899),
    ("08", "Ardennes",                 "Charleville-Mézières",   "08105",  49.7699,  4.7194,    270_348),
    ("09", "Ariège",                   "Foix",                   "09122",  42.9644,  1.6053,    154_071),
    ("10", "Aube",                     "Troyes",                 "10387",  48.2973,  4.0744,    307_953),
    ("11", "Aude",                     "Carcassonne",            "11069",  43.2130,  2.3491,    378_073),
    ("12", "Aveyron",                  "Rodez",                  "12202",  44.3514,  2.5750,    279_206),
    ("13", "Bouches-du-Rhône",         "Marseille",              "13055",  43.2965,  5.3698,  2_043_110),
    ("14", "Calvados",                 "Caen",                   "14118",  49.1829, -0.3707,    700_839),
    ("15", "Cantal",                   "Aurillac",               "15014",  44.9255,  2.4433,    144_692),
    ("16", "Charente",                 "Angoulême",              "16015",  45.6490,  0.1558,    349_915),
    ("17", "Charente-Maritime",        "La Rochelle",            "17300",  46.1603, -1.1511,    650_316),
    ("18", "Cher",                     "Bourges",                "18033",  47.0836,  2.3988,    299_673),
    ("19", "Corrèze",                  "Tulle",                  "19272",  45.2672,  1.7714,    238_765),
    ("21", "Côte-d'Or",                "Dijon",                  "21231",  47.3220,  5.0415,    535_504),
    ("22", "Côtes-d'Armor",            "Saint-Brieuc",           "22278",  48.5144, -2.7697,    600_639),
    ("23", "Creuse",                   "Guéret",                 "23096",  46.1728,  1.8706,    113_522),
    ("24", "Dordogne",                 "Périgueux",              "24322",  45.1874,  0.7207,    413_380),
    ("25", "Doubs",                    "Besançon",               "25056",  47.2378,  6.0241,    544_948),
    ("26", "Drôme",                    "Valence",                "26362",  44.9329,  4.8921,    519_029),
    ("27", "Eure",                     "Évreux",                 "27229",  49.0240,  1.1517,    608_769),
    ("28", "Eure-et-Loir",             "Chartres",               "28085",  48.4469,  1.4881,    432_564),
    ("29", "Finistère",                "Quimper",                "29232",  47.9960, -4.0976,    908_333),
    ("30", "Gard",                     "Nîmes",                  "30189",  43.8367,  4.3601,    750_668),
    ("31", "Haute-Garonne",            "Toulouse",               "31555",  43.6047,  1.4442,  1_432_476),
    ("32", "Gers",                     "Auch",                   "32013",  43.6459,  0.5858,    191_283),
    ("33", "Gironde",                  "Bordeaux",               "33063",  44.8378, -0.5792,  1_625_358),
    ("34", "Hérault",                  "Montpellier",            "34172",  43.6108,  3.8767,  1_175_623),
    ("35", "Ille-et-Vilaine",          "Rennes",                 "35238",  48.1173, -1.6778,  1_079_498),
    ("36", "Indre",                    "Châteauroux",            "36044",  46.8133,  1.6913,    218_335),
    ("37", "Indre-et-Loire",           "Tours",                  "37261",  47.3941,  0.6848,    614_093),
    ("38", "Isère",                    "Grenoble",               "38185",  45.1885,  5.7245,  1_271_166),
    ("39", "Jura",                     "Lons-le-Saunier",        "39300",  46.6737,  5.5538,    260_711),
    ("40", "Landes",                   "Mont-de-Marsan",         "40192",  43.8937, -0.4997,    416_219),
    ("41", "Loir-et-Cher",             "Blois",                  "41018",  47.5886,  1.3346,    334_054),
    ("42", "Loire",                    "Saint-Étienne",          "42218",  45.4347,  4.3903,    769_117),
    ("43", "Haute-Loire",              "Le Puy-en-Velay",        "43157",  45.0433,  3.8839,    229_862),
    ("44", "Loire-Atlantique",         "Nantes",                 "44109",  47.2184, -1.5536,  1_429_272),
    ("45", "Loiret",                   "Orléans",                "45234",  47.9029,  1.9039,    689_762),
    ("46", "Lot",                      "Cahors",                 "46042",  44.4480,  1.4413,    175_440),
    ("47", "Lot-et-Garonne",           "Agen",                   "47001",  44.2016,  0.6240,    331_652),
    ("48", "Lozère",                   "Mende",                  "48095",  44.5188,  3.4998,     74_825),
    ("49", "Maine-et-Loire",           "Angers",                 "49007",  47.4784, -0.5632,    815_963),
    ("50", "Manche",                   "Saint-Lô",               "50502",  49.1159, -1.0923,    494_023),
    ("51", "Marne",                    "Châlons-en-Champagne",   "51108",  48.9576,  4.3646,    567_002),
    ("52", "Haute-Marne",              "Chaumont",               "52121",  48.1117,  5.1393,    173_074),
    ("53", "Mayenne",                  "Laval",                  "53130",  48.0733, -0.7660,    307_228),
    ("54", "Meurthe-et-Moselle",       "Nancy",                  "54395",  48.6921,  6.1844,    733_481),
    ("55", "Meuse",                    "Bar-le-Duc",             "55029",  48.7735,  5.1633,    183_337),
    ("56", "Morbihan",                 "Vannes",                 "56260",  47.6587, -2.7607,    750_863),
    ("57", "Moselle",                  "Metz",                   "57463",  49.1193,  6.1757,  1_048_348),
    ("58", "Nièvre",                   "Nevers",                 "58194",  46.9897,  3.1572,    204_547),
    ("59", "Nord",                     "Lille",                  "59350",  50.6292,  3.0573,  2_608_346),
    ("60", "Oise",                     "Beauvais",               "60057",  49.4295,  2.0807,    838_840),
    ("61", "Orne",                     "Alençon",                "61001",  48.4306,  0.0921,    278_279),
    ("62", "Pas-de-Calais",            "Arras",                  "62041",  50.2917,  2.7801,  1_469_590),
    ("63", "Puy-de-Dôme",              "Clermont-Ferrand",       "63113",  45.7797,  3.0863,    654_625),
    ("64", "Pyrénées-Atlantiques",     "Pau",                    "64445",  43.2951, -0.3708,    678_198),
    ("65", "Hautes-Pyrénées",          "Tarbes",                 "65440",  43.2327,  0.0767,    228_530),
    ("66", "Pyrénées-Orientales",      "Perpignan",              "66136",  42.6887,  2.8948,    478_877),
    ("67", "Bas-Rhin",                 "Strasbourg",             "67482",  48.5734,  7.7521,  1_145_299),
    ("68", "Haut-Rhin",                "Colmar",                 "68066",  48.0794,  7.3580,    762_975),
    ("69", "Rhône",                    "Lyon",                   "69123",  45.7640,  4.8357,  1_866_637),
    ("70", "Haute-Saône",              "Vesoul",                 "70550",  47.6245,  6.1546,    234_473),
    ("71", "Saône-et-Loire",           "Mâcon",                  "71270",  46.3076,  4.8321,    553_570),
    ("72", "Sarthe",                   "Le Mans",                "72181",  48.0061,  0.1996,    564_057),
    ("73", "Savoie",                   "Chambéry",               "73065",  45.5646,  5.9178,    438_388),
    ("74", "Haute-Savoie",             "Annecy",                 "74010",  45.8992,  6.1294,    829_500),
    ("75", "Paris",                    "Paris",                  "75056",  48.8566,  2.3522,  2_145_906),
    ("76", "Seine-Maritime",           "Rouen",                  "76540",  49.4432,  1.0993,  1_254_136),
    ("77", "Seine-et-Marne",           "Melun",                  "77288",  48.5408,  2.6548,  1_432_092),
    ("78", "Yvelines",                 "Versailles",             "78646",  48.8014,  2.1301,  1_438_266),
    ("79", "Deux-Sèvres",              "Niort",                  "79191",  46.3239, -0.4620,    372_576),
    ("80", "Somme",                    "Amiens",                 "80021",  49.8941,  2.2958,    570_559),
    ("81", "Tarn",                     "Albi",                   "81004",  43.9298,  2.1481,    388_847),
    ("82", "Tarn-et-Garonne",          "Montauban",              "82121",  44.0197,  1.3534,    263_381),
    ("83", "Var",                      "Toulon",                 "83137",  43.1258,  5.9306,  1_079_757),
    ("84", "Vaucluse",                 "Avignon",                "84007",  43.9493,  4.8055,    565_095),
    ("85", "Vendée",                   "La Roche-sur-Yon",       "85191",  46.6706, -1.4264,    685_442),
    ("86", "Vienne",                   "Poitiers",               "86194",  46.5802,  0.3404,    437_085),
    ("87", "Haute-Vienne",             "Limoges",                "87085",  45.8336,  1.2611,    374_426),
    ("88", "Vosges",                   "Épinal",                 "88160",  48.1797,  6.4498,    364_499),
    ("89", "Yonne",                    "Auxerre",                "89025",  47.7985,  3.5736,    333_116),
    ("90", "Territoire de Belfort",    "Belfort",                "90010",  47.6382,  6.8628,    143_963),
    ("91", "Essonne",                  "Évry-Courcouronnes",     "91228",  48.6235,  2.4450,  1_310_053),
    ("92", "Hauts-de-Seine",           "Nanterre",               "92050",  48.8924,  2.2073,  1_617_369),
    ("93", "Seine-Saint-Denis",        "Bobigny",                "93008",  48.9069,  2.4400,  1_659_513),
    ("94", "Val-de-Marne",             "Créteil",                "94028",  48.7803,  2.4570,  1_394_272),
    ("95", "Val-d'Oise",               "Cergy",                  "95127",  49.0362,  2.0618,  1_244_963),
]

# Safety net: Paris may be absent from Wikidata despite the P279* traversal
PARIS_FALLBACK: dict = {
    "dept_code": "75", "dept_name": "Paris", "prefecture": "Paris",
    "pref_insee": "75056", "latitude": 48.8566, "longitude": 2.3522,
    "population": 2_145_906,
}


# ===========================================================================
# TypedDict
# ===========================================================================

class DepartmentRecord(TypedDict):
    dept_code:  str
    dept_name:  str
    prefecture: str
    pref_insee: str
    latitude:   float
    longitude:  float
    population: int


# ===========================================================================
# 3. Utility helpers
# ===========================================================================

def wmo_label(code) -> str:
    """Return a human-readable weather description from a WMO code."""
    if code is None:
        return None
    try:
        return WMO_CODES.get(int(code), f"Code {int(code)}")
    except (TypeError, ValueError):
        return None


def wind_direction_label(degrees) -> str | None:
    """Convert wind direction in degrees (0-360) to a 16-point compass label."""
    if degrees is None:
        return None
    try:
        points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return points[round(float(degrees) / 22.5) % 16]
    except (TypeError, ValueError):
        return None


# ===========================================================================
# 4. Department data retrieval (Wikidata + INSEE fallback)
# ===========================================================================

def deduplicate_departments_by_pref_population(
    raw_rows: list[dict],
) -> list[DepartmentRecord]:
    """
    Deduplicate raw SPARQL rows to one DepartmentRecord per department code,
    keeping the prefecture with the highest population.

    The SPARQL query is sorted DESC(?prefPop), so the first occurrence of each
    dept_code is already the most populated prefecture — deduplication is a
    simple "keep first seen per dept_code" pass.

    Also injects Paris (dept 75) if absent from Wikidata results.
    """
    seen: set[str]                      = set()
    departments: list[DepartmentRecord] = []

    for row in raw_rows:
        dept_code = row.get("deptCode", {}).get("value", "")
        if not dept_code or dept_code in seen:
            continue
        try:
            departments.append(DepartmentRecord(
                dept_code=  dept_code,
                dept_name=  row["deptLabel"]["value"],
                prefecture= row["prefLabel"]["value"],
                pref_insee= row["prefInsee"]["value"],
                latitude=   float(row["latitude"]["value"]),
                longitude=  float(row["longitude"]["value"]),
                population= int(float(row["pop"]["value"])),
            ))
            seen.add(dept_code)
        except (KeyError, ValueError):
            continue

    if "75" not in seen:
        print("[WARNING] Dept 75 (Paris) absent from Wikidata — injecting fallback.")
        departments.append(DepartmentRecord(**PARIS_FALLBACK))

    return sorted(departments, key=lambda d: d["dept_code"])


def get_departments_from_wikidata() -> list[DepartmentRecord]:
    """
    Fetch metropolitan French departments from Wikidata SPARQL.

    Uses wdt:P31/wdt:P279* to include Paris (Collectivité de Paris, Q1142326).
    Deduplicates to one prefecture per department by population.

    Returns:
        Sorted list of 94 DepartmentRecord.
    Raises:
        requests.RequestException if the endpoint is unreachable.
    """
    print("[Wikidata] Querying departments via SPARQL...")
    headers = {
        "User-Agent": "FranceWeatherBot/1.0 (educational project)",
        "Accept":     "application/sparql-results+json",
    }
    response = requests.get(
        WIKIDATA_SPARQL_URL,
        params={"query": WIKIDATA_DEPARTMENTS_QUERY, "format": "json"},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    raw_rows    = response.json()["results"]["bindings"]
    departments = deduplicate_departments_by_pref_population(raw_rows)
    print(f"[OK] Wikidata — {len(departments)} departments (after deduplication).")
    return departments


def get_departments(use_fallback: bool = False) -> list[DepartmentRecord]:
    """
    Return all 94 metropolitan French departments (excluding Corsica).

    Attempt 1: Wikidata SPARQL (with Paris + deduplication fixes).
    Attempt 2: Embedded INSEE 2021 fallback dataset.
    """
    if not use_fallback:
        try:
            return get_departments_from_wikidata()
        except Exception as exc:
            print(f"[WARNING] Wikidata unreachable ({exc}). Using fallback dataset.")

    departments = [
        DepartmentRecord(
            dept_code=code, dept_name=name, prefecture=pref,
            pref_insee=pref_insee, latitude=lat, longitude=lon,
            population=pop,
        )
        for code, name, pref, pref_insee, lat, lon, pop in FALLBACK_DEPARTMENTS
    ]
    print(f"[OK] Fallback dataset — {len(departments)} departments loaded.")
    return departments


# ===========================================================================
# 5. Last completed hour
# ===========================================================================

def last_completed_hour() -> datetime:
    """
    Return the most recently completed hour (minutes=0, seconds=0) strictly
    before the current wall-clock time.

    Examples:
        Current time 14:37 -> returns 14:00 of the same day
        Current time 09:00:00 exactly -> returns 08:00 (strictly before)
    """
    now          = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    if now == current_hour:
        return current_hour - timedelta(hours=1)
    return current_hour


# ===========================================================================
# 6. Single multi-location API call
# ===========================================================================

def fetch_weather_all_departments(
    departments: list[DepartmentRecord],
    past_days:   int,
    variables:   list[str] = HOURLY_VARIABLES,
) -> list[dict]:
    """
    Fetch all hourly weather variables for all departments in a single
    Open-Meteo multi-location API call (up to 1 000 coordinates).

    Args:
        departments : List of DepartmentRecord (94 entries).
        past_days   : Number of past days to retrieve (max 92).

    Returns:
        List of per-location result dicts (same order as departments).

    Raises:
        requests.RequestException on network or HTTP errors.
    """
    lats = ",".join(str(d["latitude"])  for d in departments)
    lons = ",".join(str(d["longitude"]) for d in departments)

    params = {
        "latitude":      lats,
        "longitude":     lons,
        "hourly":        ",".join(variables),
        "past_days":     past_days,
        "forecast_days": 1,             # include today's partial data
        "timezone":      "Europe/Paris",
        "wind_speed_unit":    "kmh",
        "precipitation_unit": "mm",
    }

    print(f"[Open-Meteo] Fetching {len(variables)} variable(s) for "
          f"{len(departments)} departments (past_days={past_days})...")
    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Normalise to list — single location returns a dict, multiple returns a list
    return data if isinstance(data, list) else [data]


# ===========================================================================
# 7. Build flat records
# ===========================================================================

def build_records(
    departments: list[DepartmentRecord],
    api_results: list[dict],
    cutoff:      datetime,
) -> list[dict]:
    """
    Build one flat dict per (department, completed hour) pair.

    Only completed-hour timestamps (minutes=0, seconds=0) at or before
    `cutoff` are included. Missing API values are left as None, which pandas
    converts to NaN — the standard representation for missing numeric data.

    Derived fields added:
      - "Weather"          : human-readable WMO label (None if weather_code absent)
      - "Wind Direction"   : compass label derived from wind_direction_10m degrees
      - "Sunshine (min/h)" : sunshine_duration converted from seconds to minutes

    Args:
        departments : Ordered list of DepartmentRecord matching api_results.
        api_results : Per-location dicts from the Open-Meteo response.
        cutoff      : Last completed hour; later timestamps are discarded.

    Returns:
        Flat list of dicts sorted by department code then timestamp.
    """
    records: list[dict] = []

    for dept, result in zip(departments, api_results):
        hourly  = result.get("hourly", {})
        times   = hourly.get("time", [])
        n_steps = len(times)

        def col(key: str) -> list:
            """Retrieve a variable column, padding with None if shorter than expected."""
            values = hourly.get(key, [])
            return list(values) + [None] * (n_steps - len(values))

        # Pre-fetch all columns once before the timestamp loop
        temperature    = col("temperature_2m")
        feels_like     = col("apparent_temperature")
        humidity       = col("relative_humidity_2m")
        dew_point      = col("dew_point_2m")
        wind_speed     = col("wind_speed_10m")
        wind_dir       = col("wind_direction_10m")
        wind_gusts     = col("wind_gusts_10m")
        precipitation  = col("precipitation")
        rain           = col("rain")
        snow           = col("snowfall")
        precip_prob    = col("precipitation_probability")
        weather_code   = col("weather_code")
        cloud_cover    = col("cloud_cover")
        sunshine_s     = col("sunshine_duration")
        uv_index       = col("uv_index")
        uv_clear_sky   = col("uv_index_clear_sky")
        pressure       = col("surface_pressure")
        visibility     = col("visibility")
        is_day         = col("is_day")
        evapotrans     = col("et0_fao_evapotranspiration")
        cape           = col("cape")
        freezing_lvl   = col("freezing_level_height")

        for i, ts_str in enumerate(times):
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            # Keep only completed hours at or before the cutoff
            if ts.minute != 0 or ts.second != 0:
                continue
            if ts > cutoff:
                continue

            sun_s   = sunshine_s[i]
            sun_min = round(sun_s / 60, 1) if sun_s is not None else None

            records.append({
                "Department Code":         dept["dept_code"],
                "Department Name":         dept["dept_name"],
                "Prefecture":              dept["prefecture"],
                "Prefecture INSEE Code":   dept["pref_insee"],
                "Population":              dept["population"],
                "Latitude":                round(dept["latitude"], 4),
                "Longitude":               round(dept["longitude"], 4),
                "Timestamp":               ts_str,
                "Weather":                 wmo_label(weather_code[i]),
                "Temperature (°C)":        temperature[i],
                "Feels Like (°C)":         feels_like[i],
                "Humidity (%)":            humidity[i],
                "Dew Point (°C)":          dew_point[i],
                "Wind Speed (km/h)":       wind_speed[i],
                "Wind Direction (°)":      wind_dir[i],
                "Wind Direction":          wind_direction_label(wind_dir[i]),
                "Wind Gusts (km/h)":       wind_gusts[i],
                "Precipitation (mm)":      precipitation[i],
                "Rain (mm)":               rain[i],
                "Snow (cm)":               snow[i],
                "Cloud Cover (%)":         cloud_cover[i],
                "Sunshine (min/h)":        sun_min,
                "Surface Pressure (hPa)":  pressure[i],
                "Visibility (m)":          visibility[i],
                "Day / Night":             ("Day" if is_day[i] == 1
                                            else "Night" if is_day[i] == 0
                                            else None),
                "Evapotranspiration (mm)": evapotrans[i],
                "Freezing Level (m)":      freezing_lvl[i],
                "Weather Code":            weather_code[i],
                "Precip. Probability (%)": precip_prob[i],
                "UV Index":                uv_index[i],
                "UV Index (clear sky)":    uv_clear_sky[i],
                "CAPE (J/kg)":             cape[i],
            })

    return records


# ===========================================================================
# 8. Main entry point
# ===========================================================================

def fetch_department_weather(
    days:         int             | None = None,
    use_fallback: bool                   = False,
    export_csv:   bool                   = False,
    weather_data: str | list[str]        = "temperature_2m",
) -> pd.DataFrame:
    """
    Fetch all hourly weather variables for all 94 metropolitan French
    departments (excl. Corsica) over the last N days, and export to one CSV.

    A single Open-Meteo /v1/forecast multi-location request covers all 94
    prefectures at once. Missing values (forecast-only variables for past
    timestamps) are represented as NaN in the output DataFrame and CSV.

    Can be called from Python or the command line. CLI arguments take priority.

    Args:
        days         : Number of past days to fetch (default: 35, max: 92).
        use_fallback : Skip Wikidata and use the embedded INSEE 2021 dataset.
        weather_data : Variable(s) to fetch. Accepts a single variable name (str)
                       or a list of variable names (list[str]).
                       Default: "temperature_2m". All valid Open-Meteo hourly
                       variable names are accepted (see HOURLY_VARIABLES for
                       the full list used in the original script).

    Returns:
        DataFrame with one row per (department, completed hour).
        Also writes the DataFrame to weather_data_departments_{YYYYMMDD_hhmmss}.csv.
    """
    # ------------------------------------------------------------------
    # CLI argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all hourly weather variables for all 94 metropolitan French "
            "departments over the last N days. Single API call, single CSV output."
        )
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help=f"Number of past days to fetch (default: 35, max: {MAX_PAST_DAYS})"
    )
    parser.add_argument(
        "--use-fallback", action="store_true", default=False,
        help="Skip Wikidata and use the embedded INSEE 2021 fallback dataset"
    )
    parser.add_argument(
        "--export-csv", action="store_true", default=False,
        help="If set, export the result to a timestamped CSV file"
    )
    cli_args, _ = parser.parse_known_args()

    days         = max(1, min(
        cli_args.days if cli_args.days is not None else (days or 35),
        MAX_PAST_DAYS,
    ))
    use_fallback = cli_args.use_fallback or use_fallback
    export_csv   = cli_args.export_csv or export_csv
    resolved_variables: list[str] = (
        [weather_data] if isinstance(weather_data, str) else list(weather_data)
    )

    run_ts   = datetime.now()
    cutoff   = last_completed_hour()
    ts_str   = run_ts.strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print(f"  France Department Weather — {len(resolved_variables)} variable(s) — last {days} day(s)")
    print(f"  Cutoff : {cutoff.strftime('%Y-%m-%dT%H:%M')} (last completed hour)")
    print(f"  Run    : {run_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1 — Department list
    # ------------------------------------------------------------------
    departments = get_departments(use_fallback=use_fallback)
    print(f"[OK] {len(departments)} departments loaded.")

    # ------------------------------------------------------------------
    # Step 2 — Single multi-location API call
    # ------------------------------------------------------------------
    try:
        api_results = fetch_weather_all_departments(departments, days, resolved_variables)
    except requests.RequestException as exc:
        print(f"[ERROR] Open-Meteo API call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] API response received ({len(api_results)} location(s)).")

    # ------------------------------------------------------------------
    # Step 3 — Build flat records, filter to completed hours ≤ cutoff
    # ------------------------------------------------------------------
    records = build_records(departments, api_results, cutoff)

    if not records:
        print("[WARNING] No records produced. Check past_days or timezone settings.")
        return pd.DataFrame()

    n_depts = len({r["Department Code"] for r in records})
    n_ts    = len({r["Timestamp"]       for r in records})
    print(f"[OK] {len(records):,} rows built ({n_depts} depts × {n_ts} timestamps).")

    # ------------------------------------------------------------------
    # Step 4 — Export to CSV if requested, otherwise return DataFrame only
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)

    if export_csv:
        out_path = Path(f"weather_data_departments_{ts_str}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n[Export] {out_path.resolve()}")
        print(f"         {len(df):,} rows | {df['Department Code'].nunique()} departments "
              f"| {df['Timestamp'].nunique()} timestamps "
              f"| {len(df.columns)} columns")
    else:
        print(f"\n[OK] DataFrame returned ({len(df):,} rows) — no CSV export.")

    return df


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    fetch_department_weather()
