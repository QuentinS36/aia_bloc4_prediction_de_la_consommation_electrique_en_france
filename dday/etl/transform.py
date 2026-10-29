"""Transformation et contrôle qualité. Fonctions pures, testables sans réseau."""
from __future__ import annotations

import pandas as pd

TEMP_MIN, TEMP_MAX = -30.0, 50.0           # bornes physiques plausibles (°C, France)
CONSO_MIN, CONSO_MAX = 15_000.0, 110_000.0  # bornes plausibles de la consommation nationale (MW)


def to_hourly_consumption(raw: pd.DataFrame) -> pd.DataFrame:
    """Agrège les relevés 15 min en moyenne horaire. ds = fin de la période (UTC, naïf), comme dans les notebooks.

    Les heures incomplètes (moins de 4 relevés) sont écartées.
    """
    df = raw.copy()
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(None)
    df["ds"] = df["end_date"].dt.ceil("h")
    g = df.groupby("ds")["value"].agg(y="mean", n="count").reset_index()
    return g[g["n"] >= 4].drop(columns="n").reset_index(drop=True)


def aggregate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Statistiques de température sur l'ensemble des départements, par heure."""
    g = weather.groupby("ds")["temperature"]
    out = g.agg(["mean", "min", "max", "median", "std"]).reset_index()
    out["q1"] = g.quantile(0.25).values
    out["q3"] = g.quantile(0.75).values
    return out


def build_dataset(consumption_hourly: pd.DataFrame, weather_stats: pd.DataFrame) -> pd.DataFrame:
    """Jointure consommation x météo ; ne garde que les heures où les deux sont connues."""
    df = consumption_hourly.merge(weather_stats, on="ds", how="inner")
    return df.sort_values("ds").reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> dict:
    """Contrôles qualité. 'errors' bloque le chargement ; 'warnings' sont seulement journalisés."""
    errors, warnings = [], []
    if df.empty:
        return {"rows": 0, "errors": ["dataset vide"], "warnings": []}
    if df["ds"].duplicated().any():
        errors.append("horodatages dupliqués")
    if df[["y", "mean", "min", "max", "std", "q1", "q3"]].isna().any().any():
        errors.append("valeurs manquantes")
    if not df["y"].between(CONSO_MIN, CONSO_MAX).all():
        errors.append("consommation hors bornes plausibles")
    if not df["mean"].between(TEMP_MIN, TEMP_MAX).all():
        errors.append("température hors bornes plausibles")
    full = pd.date_range(df["ds"].min(), df["ds"].max(), freq="h")
    missing = len(full) - df["ds"].nunique()
    if missing:
        warnings.append(f"{missing} heure(s) manquante(s) dans la plage")
    return {"rows": len(df), "errors": errors, "warnings": warnings}


FORECAST_REGRESSORS = ["mean", "min", "max", "std", "q1", "q3"]


def forecast_rows(stats: pd.DataFrame, now) -> pd.DataFrame:
    """Statistiques de température à partir de minuit (UTC) du jour courant, lignes complètes seulement."""
    ts = pd.Timestamp(now)
    ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
    out = stats[stats["ds"] >= ts.normalize()].dropna(subset=FORECAST_REGRESSORS)
    return out.reset_index(drop=True)
