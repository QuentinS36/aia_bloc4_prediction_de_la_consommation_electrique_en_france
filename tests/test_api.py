import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dday.api import main
from tests.conftest import synthetic_dataset


class FakeModel:
    def predict(self, df):
        return pd.DataFrame({"ds": df["ds"], "yhat": 50_000.0})


@pytest.fixture
def client(monkeypatch):
    tomorrow = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))
    feats = synthetic_dataset(days=1, end=str(tomorrow + pd.Timedelta(hours=23)))
    monkeypatch.setattr(main, "_forecast_features", lambda day: feats)
    monkeypatch.setattr(main, "_persist", lambda df, v: None)
    monkeypatch.setitem(main._state, "model", FakeModel())
    monkeypatch.setitem(main._state, "version", "7")
    return TestClient(main.app)


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/model-info").json()["version"] == "7"


def test_predict_default_tomorrow(client):
    r = client.get("/predict")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "7" and body["unit"] == "MW" and len(body["predictions"]) == 24
    assert body["date"] == str(dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1))


def test_predict_invalid_date(client):
    assert client.get("/predict?date=pas-une-date").status_code == 422


def test_predict_without_model(monkeypatch):
    monkeypatch.setitem(main._state, "model", None)
    monkeypatch.setattr(main, "load_model", lambda: (_ for _ in ()).throw(LookupError("vide")))
    assert TestClient(main.app).get("/predict").status_code == 503


def test_reload_requires_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/reload").status_code == 401
