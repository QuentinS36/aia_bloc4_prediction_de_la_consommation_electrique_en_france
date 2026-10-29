"""Extraction : consommation RTE (puissance réalisée) et température Open-Meteo."""
from __future__ import annotations

import base64
import os
import time
import datetime as dt

import pandas as pd
import requests

from dday import config  # noqa: F401  (charge le fichier .env)
from dday.etl.departments import DEPARTMENTS

RTE_TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
RTE_API_URL = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 60


def _clean_rte_key(raw: str | None) -> str:
    """Normalise RTE_SECRET_KEY (guillemets, espaces, préfixe « Basic ») et vérifie le format base64(id:secret)."""
    if not raw or not raw.strip():
        raise RuntimeError("RTE_SECRET_KEY est absent ou vide : vérifiez le fichier .env à la racine du repo "
                           "(python -m dday.config)")
    key = raw.strip().strip("\"'").strip()
    if key.lower().startswith("basic "):
        key = key[6:].strip()
    try:
        decoded = base64.b64decode(key, validate=True).decode("utf-8")
    except Exception:
        raise ValueError("RTE_SECRET_KEY n'est pas du base64 valide : il faut le base64 de 'client_id:client_secret' "
                         "(sans espace ni retour à la ligne, sans le préfixe 'Basic')") from None
    if ":" not in decoded:
        raise ValueError("RTE_SECRET_KEY décodée ne contient pas 'client_id:client_secret' : "
                         "encodez bien les deux valeurs séparées par ':'")
    return key


def _rte_token() -> str:
    key = _clean_rte_key(os.getenv("RTE_SECRET_KEY"))
    r = requests.post(
        RTE_TOKEN_URL, timeout=TIMEOUT,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": "Basic " + key},
    )
    if r.status_code != 200:
        raise RuntimeError(f"RTE a refusé la demande de jeton (HTTP {r.status_code}) : {r.text[:300]}")
    return r.json()["access_token"]


def fetch_rte_consumption(start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Consommation réalisée au pas de 15 min entre start et end (UTC). Colonnes : start_date, end_date, value."""
    r = requests.get(
        RTE_API_URL, timeout=TIMEOUT,
        headers={"Authorization": "Bearer " + _rte_token()},
        params={"type": "REALISED", "start_date": start.isoformat(), "end_date": end.isoformat()},
    )
    if r.status_code != 200:
        hint = (" Le jeton a été obtenu mais RTE refuse l'accès aux données : vérifiez que l'application est "
                "abonnée à l'API Consumption et que l'abonnement est actif sur le portail."
                if r.status_code in (401, 403) else "")
        raise RuntimeError(f"RTE a refusé la requête de consommation (HTTP {r.status_code}) : {r.text[:300]}.{hint}")
    values = r.json()["short_term"][0]["values"]
    return pd.DataFrame(values)[["start_date", "end_date", "value"]]


RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAYS = (2, 5, 10)  # secondes entre deux tentatives


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """GET avec nouvelles tentatives sur limite de débit (429), erreurs serveur (5xx) et coupures réseau."""
    attempts = len(RETRY_DELAYS) + 1
    for i in range(attempts):
        last_try = i == attempts - 1
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
        except (requests.ConnectionError, requests.Timeout):
            if last_try:
                raise
            time.sleep(RETRY_DELAYS[i])
            continue
        if r.status_code in RETRY_STATUS and not last_try:
            try:  # respecte Retry-After s'il est fourni (plafonné)
                delay = min(float(r.headers.get("Retry-After", "")), 20.0)
            except ValueError:
                delay = RETRY_DELAYS[i]
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("inatteignable")


def fetch_weather(past_days: int = 0, forecast_days: int = 1, start_date: str | None = None) -> pd.DataFrame:
    """Température à 2 m, toutes préfectures, horodatage UTC. Colonnes : dept, ds, temperature.

    Avec start_date (AAAA-MM-JJ), la requête est limitée à ce seul jour : plus légère pour les limites de débit.
    """
    params = {
        "latitude": ",".join(str(d[1]) for d in DEPARTMENTS),
        "longitude": ",".join(str(d[2]) for d in DEPARTMENTS),
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    if start_date:
        params.update({"start_date": start_date, "end_date": start_date})
    else:
        params.update({"past_days": past_days, "forecast_days": forecast_days})
    data = _get_with_retry(OPEN_METEO_URL, params).json()
    data = data if isinstance(data, list) else [data]
    frames = []
    for dept, loc in zip(DEPARTMENTS, data):
        h = loc["hourly"]
        frames.append(pd.DataFrame({"dept": dept[0], "ds": pd.to_datetime(h["time"]),
                                    "temperature": h["temperature_2m"]}))
    return pd.concat(frames, ignore_index=True)


def _diagnose() -> None:
    """python -m dday.etl.extract : teste la clé, le jeton puis l'accès aux données (n'affiche aucun secret)."""
    raw = os.getenv("RTE_SECRET_KEY")
    print("1/3 Clé :", end=" ")
    try:
        key = _clean_rte_key(raw)
        ident, secret = base64.b64decode(key).decode().split(":", 1)
        print(f"format OK (identifiant : {len(ident)} caractères, secret : {len(secret)} caractères)")
    except Exception as e:
        print("PROBLÈME ->", e)
        return
    print("2/3 Jeton :", end=" ")
    r = requests.post(RTE_TOKEN_URL, timeout=TIMEOUT, headers={
        "Content-Type": "application/x-www-form-urlencoded", "Authorization": "Basic " + key})
    if r.status_code != 200:
        print(f"REFUSÉ (HTTP {r.status_code}) -> identifiants rejetés par RTE : {r.text[:200]}")
        return
    token = r.json()["access_token"]
    print("obtenu")
    print("3/3 Données :", end=" ")
    end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    r = requests.get(RTE_API_URL, timeout=TIMEOUT, headers={"Authorization": "Bearer " + token},
                     params={"type": "REALISED", "start_date": (end - dt.timedelta(hours=3)).isoformat(),
                             "end_date": end.isoformat()})
    print(f"HTTP {r.status_code}", "-> OK" if r.status_code == 200 else f"-> {r.text[:200]}")


if __name__ == "__main__":
    _diagnose()
