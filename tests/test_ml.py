import pandas as pd
import pytest

from dday.ml import registry, train
from tests.conftest import synthetic_dataset


def test_split_last_day():
    tr, te = train.split_last_day(synthetic_dataset(days=40), 30)
    assert len(te) == 24 and tr["ds"].max() < te["ds"].min()
    assert tr["ds"].min() >= te["ds"].min() - pd.Timedelta(days=30)


def test_split_rejects_small_data():
    with pytest.raises(ValueError):
        train.split_last_day(synthetic_dataset(days=3), 30)


def test_train_register_promote_and_rollback(mlflow_local, require_stan):
    df = synthetic_dataset(days=40)
    r1 = train.train_and_register(df, window_days=30)
    assert r1["promoted"] and registry.champion_version() == r1["version"]
    assert r1["test_mape"] < 0.05  # données synthétiques propres

    # Second modèle entraîné sur des données dégradées : ne doit pas détrôner le champion
    bad = df.copy()
    bad.loc[bad["ds"] < bad["ds"].max() - pd.Timedelta(hours=24), "y"] *= 1.5
    r2 = train.train_and_register(bad, window_days=30)
    assert not r2["promoted"] and registry.champion_version() == r1["version"]

    # Rollback explicite vers une version donnée puis retour arrière par défaut
    registry._set_champion(r2["version"])
    assert registry.rollback() == r1["version"] and registry.champion_version() == r1["version"]
    model, version = registry.load_champion()
    assert version == r1["version"] and hasattr(model, "predict")
