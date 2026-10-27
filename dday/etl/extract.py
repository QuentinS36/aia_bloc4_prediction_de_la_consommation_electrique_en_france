"""Extraction : consommation RTE (puissance réalisée) et température Open-Meteo."""
from __future__ import annotations

import base64
import os
import datetime as dt

import pandas as pd
import requests

from dday import config  # noqa: F401  (charge le fichier .env)
from dday.etl.departments import DEPARTMENTS

RTE_TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
RTE_API_URL = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 60


def _rte_token():
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic " + os.getenv("RTE_SECRET_KEY")
    }
    try:
        response_token = requests.post(RTE_TOKEN_URL, headers=headers)
        if response_token.status_code == 200:
            data = response_token.json()
            oauth_token = data["access_token"]
            return oauth_token
        else:
            raise Exception(f"Token Error: {response_token.status_code}")
    except Exception as e:
        raise(e)


def fetch_rte_consumption(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Consommation réalisée au pas de 15 min entre start et end (UTC). Colonnes : start_date, end_date, value."""
    r = requests.get(
        RTE_API_URL, timeout=TIMEOUT,
        headers={"Authorization": "Bearer " + _rte_token()},
        params={"type": "REALISED", "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    r.raise_for_status()
    values = r.json()["short_term"][0]["values"]
    return pd.DataFrame(values)[["start_date", "end_date", "value"]]


def fetch_weather(past_days: int = 0, forecast_days: int = 1) -> pd.DataFrame:
    """Température à 2 m, toutes préfectures, horodatage UTC. Colonnes : dept, ds, temperature."""
    params = {
        "latitude": ",".join(str(d[1]) for d in DEPARTMENTS),
        "longitude": ",".join(str(d[2]) for d in DEPARTMENTS),
        "hourly": "temperature_2m",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    data = data if isinstance(data, list) else [data]
    frames = []
    for dept, loc in zip(DEPARTMENTS, data):
        h = loc["hourly"]
        frames.append(pd.DataFrame({"dept": dept[0], "ds": pd.to_datetime(h["time"]),
                                    "temperature": h["temperature_2m"]}))
    return pd.concat(frames, ignore_index=True)
