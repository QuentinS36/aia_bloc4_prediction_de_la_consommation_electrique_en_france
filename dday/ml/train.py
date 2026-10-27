"""Entraînement Prophet + suivi MLflow + promotion conditionnelle.

Usage : python -m dday.ml.train [--window-days 30] [--force-window-days 60 --grid] [--no-promote]
"""
from __future__ import annotations

import argparse
import itertools
import logging

import mlflow
import mlflow.prophet
import pandas as pd
from mlflow import MlflowClient
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from dday import config
from dday.db import get_engine, read_dataset
from dday.ml import registry

log = logging.getLogger("dday.ml")
EXPERIMENT = "dday-consommation"


def fit(train: pd.DataFrame, hyperparams: dict | None = None) -> Prophet:
    model = Prophet(**(hyperparams or config.HYPERPARAMS))
    for r in config.REGRESSORS:
        model.add_regressor(r)
    model.fit(train[["ds", "y"] + config.REGRESSORS])
    return model


def score(model: Prophet, test: pd.DataFrame) -> dict:
    pred = model.predict(test[["ds"] + config.REGRESSORS])
    return {"mape": float(mean_absolute_percentage_error(test["y"], pred["yhat"])),
            "mae": float(mean_absolute_error(test["y"], pred["yhat"]))}


def split_last_day(df: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test = dernières 24 h ; train = `window_days` jours qui précèdent (comme la simulation des notebooks)."""
    df = df.dropna(subset=["y"] + config.REGRESSORS).sort_values("ds")
    t_end = df["ds"].max()
    t_start = t_end - pd.Timedelta(hours=23)
    test = df[df["ds"] >= t_start]
    train = df[(df["ds"] < t_start) & (df["ds"] >= t_start - pd.Timedelta(days=window_days))]
    if len(train) < 24 * 7 or len(test) < 12:
        raise ValueError(f"Données insuffisantes (train={len(train)}, test={len(test)})")
    return train, test


def train_and_register(df: pd.DataFrame, window_days: int = config.TRAIN_WINDOW_DAYS,
                       grid: bool = False, promote: bool = True) -> dict:
    train, test = split_last_day(df, window_days)
    candidates = [config.HYPERPARAMS]
    if grid:
        space = {"changepoint_prior_scale": [0.3, 0.35, 0.4, 0.45], "seasonality_prior_scale": [0.5, 1.0]}
        candidates = [dict(zip(space, v)) for v in itertools.product(*space.values())]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"train-{test['ds'].max():%Y%m%d}") as run:
        best = None
        for hp in candidates:
            m = fit(train, hp)
            s = score(m, test)
            if best is None or s["mape"] < best[1]["mape"]:
                best = (hp, s)
        hp, metrics = best
        mlflow.log_params({**hp, "window_days": window_days, "regressors": ",".join(config.REGRESSORS),
                           "n_candidates": len(candidates), "test_end": str(test["ds"].max())})
        mlflow.log_metrics({"test_mape": metrics["mape"], "test_mae": metrics["mae"]})

        # Modèle déployé : réentraîné sur la fenêtre complète, incluant le jour de validation
        full = df.dropna(subset=["y"] + config.REGRESSORS)
        full = full[full["ds"] > full["ds"].max() - pd.Timedelta(days=window_days)]
        final = fit(full, hp)
        info = mlflow.prophet.log_model(final, name="model")
        version = registry.register(info.model_uri, run.info.run_id, metrics)

        result = {"run_id": run.info.run_id, "version": version, **{f"test_{k}": v for k, v in metrics.items()}}
        result["promoted"] = registry.promote_if_better(version, test, metrics["mape"]) if promote else False
        mlflow.set_tag("promoted", str(result["promoted"]))
        log.info("Entraînement terminé : %s", result)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=config.TRAIN_WINDOW_DAYS)
    ap.add_argument("--grid", action="store_true", help="Recherche d'hyperparamètres (réentraînement forcé)")
    ap.add_argument("--no-promote", action="store_true")
    a = ap.parse_args()
    data = read_dataset(get_engine(), since=pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=a.window_days + 3))
    train_and_register(data, a.window_days, a.grid, not a.no_promote)
