"""API de prédiction. Lancement : uvicorn dday.api.main:app --host 0.0.0.0 --port 8000

Documentation interactive (OpenAPI) : /docs
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time

import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from dday import config
from dday.etl import extract, transform

log = logging.getLogger("dday.api")

app = FastAPI(
    title="API de prédiction de la consommation électrique",
    description="Prévision horaire de la consommation française (MW) pour un jour donné, "
                "à partir de la prévision de température Open-Meteo et du modèle Prophet champion.",
    version="1.0.0",
)

_state: dict = {"model": None, "version": None}
_lock = threading.Lock()
_stats = {"requests": 0, "errors": 0, "latency_total": 0.0}


class PredictionPoint(BaseModel):
    ds: dt.datetime = Field(description="Heure (UTC)")
    yhat: float = Field(description="Consommation prévue (MW)")


class PredictionResponse(BaseModel):
    date: dt.date
    model_version: str
    unit: str = "MW"
    predictions: list[PredictionPoint]


class ModelInfo(BaseModel):
    model_name: str
    version: str | None
    loaded: bool


def load_model() -> None:
    """Charge (ou recharge) le modèle champion depuis le registry MLflow."""
    from dday.ml import registry
    with _lock:
        _state["model"], _state["version"] = registry.load_champion()
    log.info("Modèle champion v%s chargé", _state["version"])


def _get_model():
    if _state["model"] is None:
        try:
            load_model()
        except LookupError as e:
            raise HTTPException(503, f"Aucun modèle disponible : {e}")
    return _state["model"], _state["version"]


def _forecast_features(day: dt.date) -> pd.DataFrame:
    """Statistiques de température prévues pour `day` (24 heures UTC)."""
    horizon = (day - dt.datetime.now(dt.timezone.utc).date()).days + 1
    if horizon < 1 or horizon > 14:
        raise HTTPException(422, "La date doit être comprise entre aujourd'hui et J+13")
    stats = transform.aggregate_weather(extract.fetch_weather(past_days=0, forecast_days=horizon + 1))
    start = pd.Timestamp(day)
    out = stats[(stats["ds"] >= start) & (stats["ds"] < start + pd.Timedelta(days=1))]
    if len(out) < 24:
        raise HTTPException(502, "Prévision météo incomplète pour cette date")
    return out


def _persist(df: pd.DataFrame, version: str) -> None:
    if not os.getenv("DATABASE_URL"):
        return
    try:
        from dday.db import get_engine, init_schema, upsert_predictions
        engine = get_engine()
        init_schema(engine)
        upsert_predictions(engine, df, version)
    except Exception:  # la prévision reste servie même si l'historisation échoue
        log.exception("Historisation des prévisions impossible")


@app.get("/health", summary="État du service")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/model-info", response_model=ModelInfo, summary="Version du modèle servi")
def model_info():
    return ModelInfo(model_name=config.MODEL_NAME, version=_state["version"], loaded=_state["model"] is not None)


@app.get("/metrics", summary="Compteurs de requêtes et latence moyenne (monitoring)")
def metrics():
    n = _stats["requests"]
    return {"requests": n, "errors": _stats["errors"],
            "avg_latency_ms": round(1000 * _stats["latency_total"] / n, 1) if n else None}


@app.get("/predict", response_model=PredictionResponse, summary="Prévision horaire pour un jour donné")
def predict(date: dt.date | None = Query(
        None, description="Jour à prévoir (UTC, AAAA-MM-JJ). Par défaut : demain.")):
    t0 = time.perf_counter()
    _stats["requests"] += 1
    try:
        day = date or (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
        model, version = _get_model()
        feats = _forecast_features(day)
        pred = model.predict(feats[["ds"] + config.REGRESSORS])[["ds", "yhat"]]
        _persist(pred, version)
        return PredictionResponse(
            date=day, model_version=version,
            predictions=[PredictionPoint(ds=r.ds, yhat=round(float(r.yhat), 1)) for r in pred.itertuples()],
        )
    except HTTPException:
        _stats["errors"] += 1
        raise
    except requests.RequestException as e:  # prévision météo (Open-Meteo)
        _stats["errors"] += 1
        log.exception("Service météo indisponible")
        status = getattr(getattr(e, "response", None), "status_code", None)
        raise HTTPException(502, "Service météo (Open-Meteo) indisponible" + (f" : HTTP {status}" if status else f" : {type(e).__name__}"))
    except Exception as e:
        _stats["errors"] += 1
        log.exception("Erreur de prédiction")
        if type(e).__module__.startswith("mlflow"):  # registry MLflow / DagsHub
            raise HTTPException(503, f"Registre de modèles (MLflow/DagsHub) indisponible : {type(e).__name__}")
        raise HTTPException(500, f"Erreur interne lors de la prédiction ({type(e).__name__})")
    finally:
        _stats["latency_total"] += time.perf_counter() - t0


@app.post("/reload", summary="Recharge le modèle champion (après promotion ou rollback)")
def reload_model(x_api_key: str | None = Header(None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Clé API invalide")
    try:
        load_model()
    except LookupError as e:
        raise HTTPException(503, str(e))
    return {"model_version": _state["version"]}
