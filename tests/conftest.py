import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from dday.db import init_schema


def synthetic_dataset(days: int = 45, end: str = "2026-03-01 00:00", seed: int = 0) -> pd.DataFrame:
    """Consommation synthétique : saisonnalité jour/semaine + dépendance à la température."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range(end=end, periods=days * 24, freq="h")
    temp = 8 + 5 * np.sin(2 * np.pi * (ds.hour - 9) / 24) + rng.normal(0, 0.5, len(ds))
    y = (55_000 - 1_200 * (temp - 8) + 6_000 * np.sin(2 * np.pi * (ds.hour - 6) / 24)
         - 3_000 * (ds.dayofweek >= 5) + rng.normal(0, 400, len(ds)))
    df = pd.DataFrame({"ds": ds, "y": y, "mean": temp})
    df["min"], df["max"] = temp - 3, temp + 3
    df["median"], df["std"] = temp, 2.0 + 0.1 * rng.random(len(ds))
    df["q1"], df["q3"] = temp - 1.5, temp + 1.5
    return df


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Les tests ne doivent jamais toucher aux vrais services, même si un .env est présent."""
    for var in ("REPORT_S3_URI", "ALERT_WEBHOOK_URL", "API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def engine(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path/'test.db'}")
    init_schema(e)
    return e


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path/'mlflow.db'}")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path/'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    yield
