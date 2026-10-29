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


def test_weather_failure_is_explained_as_502(client, monkeypatch):
    import requests

    def boom(day):
        resp = requests.Response(); resp.status_code = 429
        raise requests.HTTPError("429", response=resp)
    monkeypatch.setattr(main, "_forecast_features", boom)
    r = client.get("/predict")
    assert r.status_code == 502 and "Open-Meteo" in r.json()["detail"] and "429" in r.json()["detail"]


def test_registry_failure_is_explained_as_503(client, monkeypatch):
    err = type("MlflowException", (Exception,), {"__module__": "mlflow.exceptions"})
    monkeypatch.setattr(main, "_get_model", lambda: (_ for _ in ()).throw(err("boom")))
    r = client.get("/predict")
    assert r.status_code == 503 and "MLflow" in r.json()["detail"]


def test_unknown_failure_names_the_exception_type(client, monkeypatch):
    class BadModel:
        def predict(self, df):
            raise ValueError("Found NaN in column")
    monkeypatch.setitem(main._state, "model", BadModel())
    r = client.get("/predict")
    assert r.status_code == 500 and "ValueError" in r.json()["detail"] and "NaN" not in r.json()["detail"]


def _fake_weather(day):
    ds = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
    return pd.concat([pd.DataFrame({"dept": d, "ds": ds, "temperature": t}) for d, t in [("01", 5.0), ("02", 9.0), ("03", 7.0)]])


def test_weather_is_cached_between_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(main.extract, "fetch_weather", lambda **k: calls.append(k) or _fake_weather(k["start_date"]))
    main._weather_cache.clear()
    day = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    a = main._forecast_features(day)
    b = main._forecast_features(day)
    assert len(calls) == 1 and len(a) == 24 and a is b
    assert calls[0]["start_date"] == day.isoformat()  # requête limitée au jour demandé


def test_stale_weather_is_used_when_open_meteo_fails(monkeypatch):
    import requests
    day = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    main._weather_cache.clear()
    monkeypatch.setattr(main.extract, "fetch_weather", lambda **k: _fake_weather(k["start_date"]))
    first = main._forecast_features(day)
    ts, df = main._weather_cache[day]
    main._weather_cache[day] = (ts - main.WEATHER_FRESH_SECONDS - 60, df)  # périmé mais utilisable

    def down(**k):
        raise requests.ConnectionError("hors service")
    monkeypatch.setattr(main.extract, "fetch_weather", down)
    assert main._forecast_features(day) is first          # dernière prévision connue
    main._weather_cache.clear()
    with __import__("pytest").raises(requests.ConnectionError):
        main._forecast_features(day)                      # rien en mémoire : l'erreur remonte (502 côté route)
