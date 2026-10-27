"""Model Registry MLflow : enregistrement, promotion conditionnelle, chargement et rollback.

Rollback : python -m dday.ml.registry rollback [--version N]
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.prophet
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_percentage_error

from dday import config

log = logging.getLogger("dday.registry")


def register(model_uri: str, run_id: str, metrics: dict) -> str:
    mv = mlflow.register_model(model_uri, config.MODEL_NAME)
    client = MlflowClient()
    client.set_model_version_tag(config.MODEL_NAME, mv.version, "test_mape", f"{metrics['mape']:.6f}")
    client.set_model_version_tag(config.MODEL_NAME, mv.version, "run_id", run_id)
    return str(mv.version)


def champion_version() -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS).version)
    except MlflowException:
        return None


def load_champion():
    """Retourne (modèle Prophet, version) du champion."""
    v = champion_version()
    if v is None:
        raise LookupError("Aucun modèle champion enregistré")
    return mlflow.prophet.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}"), v


def _set_champion(version: str) -> None:
    MlflowClient().set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, version)


def promote_if_better(version: str, test: pd.DataFrame, candidate_mape: float) -> bool:
    """Promeut `version` si son MAPE sur la fenêtre de test est <= à celui du champion actuel, évalué sur la même fenêtre."""
    current = champion_version()
    if current is None:
        _set_champion(version)
        log.info("Premier modèle : v%s promu champion", version)
        return True
    champ = mlflow.prophet.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    pred = champ.predict(test[["ds"] + config.REGRESSORS])
    champ_mape = float(mean_absolute_percentage_error(test["y"], pred["yhat"]))
    if candidate_mape <= champ_mape:
        _set_champion(version)
        log.info("v%s promu (MAPE %.4f <= champion v%s %.4f)", version, candidate_mape, current, champ_mape)
        return True
    log.warning("v%s non promu (MAPE %.4f > champion v%s %.4f) : le champion est conservé",
                version, candidate_mape, current, champ_mape)
    return False


def rollback(version: str | None = None) -> str:
    """Repointe l'alias champion vers `version`, ou vers la version précédente par défaut."""
    client = MlflowClient()
    current = champion_version()
    if version is None:
        versions = sorted(int(v.version) for v in client.search_model_versions(f"name='{config.MODEL_NAME}'"))
        older = [v for v in versions if current is not None and v < int(current)]
        if not older:
            raise LookupError("Aucune version antérieure vers laquelle revenir")
        version = str(older[-1])
    _set_champion(str(version))
    log.warning("Rollback : champion v%s -> v%s", current, version)
    return str(version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["rollback", "show"])
    ap.add_argument("--version")
    a = ap.parse_args()
    print(rollback(a.version) if a.action == "rollback" else champion_version())
