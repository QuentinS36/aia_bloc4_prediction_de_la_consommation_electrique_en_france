import datetime as dt

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from dday.api import main
from dday.db import read_weather_forecast, upsert_weather_forecast
from dday.etl import extract
from dday.etl import main as etl_main
from tests.conftest import synthetic_dataset


class Resp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


def _payload(hours=2):
    times = [f"2026-03-01T{h:02d}:00" for h in range(hours)]
    return [{"hourly": {"time": times, "temperature_2m": [5.0] * hours}} for _ in extract.DEPARTMENTS]


def _sequence(monkeypatch, responses):
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return responses[min(len(calls) - 1, len(responses) - 1)]
    monkeypatch.setattr(extract.requests, "get", fake_get)
    slept = []
    monkeypatch.setattr(extract.time, "sleep", lambda s: slept.append(s))
    return calls, slept


def test_retry_on_429_then_success(monkeypatch):
    calls, slept = _sequence(monkeypatch, [Resp(429), Resp(429), Resp(200, _payload())])
    df = extract.fetch_weather(past_days=0, forecast_days=1)
    assert len(df) == 2 * len(extract.DEPARTMENTS) and len(calls) == 3 and slept == [2, 5]


def test_gives_up_after_last_attempt(monkeypatch):
    calls, slept = _sequence(monkeypatch, [Resp(429)])
    with pytest.raises(requests.HTTPError):
        extract.fetch_weather()
    assert len(calls) == 4 and slept == [2, 5, 10]


def test_retry_after_header_is_respected(monkeypatch):
    _, slept = _sequence(monkeypatch, [Resp(429, headers={"Retry-After": "3"}), Resp(200, _payload())])
    extract.fetch_weather()
    assert slept == [3.0]


def test_client_error_is_not_retried(monkeypatch):
    calls, _ = _sequence(monkeypatch, [Resp(400)])
    with pytest.raises(requests.HTTPError):
        extract.fetch_weather()
    assert len(calls) == 1


def test_forecast_table_roundtrip(engine):
    df = synthetic_dataset(days=1, end="2026-03-01 23:00")
    upsert_weather_forecast(engine, df)
    upsert_weather_forecast(engine, df)  # idempotent
    out = read_weather_forecast(engine, pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02"))
    assert len(out) == 24 and out["fetched_at"].notna().all()


def test_etl_run_stores_history_and_forecast(engine, monkeypatch):
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)

    def fake_rte(start, end):
        idx = pd.date_range(start, end, freq="15min", tz="UTC")
        return pd.DataFrame({"start_date": idx.astype(str), "end_date": (idx + pd.Timedelta(minutes=15)).astype(str),
                             "value": 50_000.0})

    def fake_weather(past_days=0, forecast_days=1, start_date=None):
        ds = pd.date_range((now - pd.Timedelta(days=past_days)).replace(tzinfo=None),
                           (now + pd.Timedelta(days=forecast_days)).replace(tzinfo=None), freq="h")
        return pd.concat([pd.DataFrame({"dept": d, "ds": ds, "temperature": t}) for d, t in [("01", 4.0), ("02", 8.0)]])

    monkeypatch.setattr(extract, "fetch_rte_consumption", fake_rte)
    monkeypatch.setattr(extract, "fetch_weather", fake_weather)
    res = etl_main.run(days=2, engine=engine)
    assert res["loaded"] > 24 and res["forecast_loaded"] >= 24 * 6
    start = pd.Timestamp(now.date()) + pd.Timedelta(days=1)
    assert len(read_weather_forecast(engine, start, start + pd.Timedelta(days=1))) == 24


class FakeModel:
    def predict(self, df):
        return pd.DataFrame({"ds": df["ds"], "yhat": 50_000.0})


def _api(monkeypatch, engine_url):
    monkeypatch.setenv("DATABASE_URL", engine_url)
    monkeypatch.setattr(main, "_engines", {})
    monkeypatch.setattr(main, "_weather_cache", {})
    monkeypatch.setitem(main._state, "model", FakeModel())
    monkeypatch.setitem(main._state, "version", "3")
    monkeypatch.setattr(main, "_persist", lambda df, v: None)
    return TestClient(main.app)


def test_api_uses_database_forecast_without_calling_open_meteo(engine, monkeypatch, tmp_path):
    tomorrow = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    upsert_weather_forecast(engine, synthetic_dataset(days=1, end=str(pd.Timestamp(tomorrow) + pd.Timedelta(hours=23))))
    client = _api(monkeypatch, str(engine.url))

    def forbidden(*a, **k):
        raise AssertionError("Open-Meteo ne doit pas être appelé")
    monkeypatch.setattr(extract, "fetch_weather", forbidden)
    r = client.get("/predict")
    assert r.status_code == 200 and r.json()["weather_source"] == "base de données" and len(r.json()["predictions"]) == 24


def test_api_falls_back_to_open_meteo_then_uses_cache(engine, monkeypatch):
    tomorrow = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    client = _api(monkeypatch, str(engine.url))  # table vide
    ds = pd.date_range(pd.Timestamp(tomorrow), periods=24, freq="h")
    calls = []

    def live(**k):
        calls.append(k)
        return pd.concat([pd.DataFrame({"dept": d, "ds": ds, "temperature": t}) for d, t in [("01", 4.0), ("02", 8.0)]])
    monkeypatch.setattr(extract, "fetch_weather", live)
    r1, r2 = client.get("/predict"), client.get("/predict")
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["weather_source"] == "open-meteo" and len(calls) == 1  # 2e appel servi par le cache
    assert calls[0]["start_date"] == tomorrow.isoformat()
