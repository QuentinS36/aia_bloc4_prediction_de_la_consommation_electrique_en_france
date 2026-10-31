import json

import pandas as pd

from dday import config
from dday.db import upsert_dataset, upsert_predictions
from dday.monitoring import run as mon
from tests.conftest import synthetic_dataset

NOW = pd.Timestamp("2026-03-01 00:00")


def _seed(engine, shift_last_week=0.0, mape_factor=1.0, with_predictions=True):
    df = synthetic_dataset(days=40, end=str(NOW))
    last_week = df["ds"] >= NOW - pd.Timedelta(days=7)
    df.loc[last_week, ["mean", "min", "max", "median", "q1", "q3"]] += shift_last_week
    df.loc[last_week, "y"] -= 1_200 * shift_last_week
    upsert_dataset(engine, df)
    if with_predictions:
        recent = df[df["ds"] >= NOW - pd.Timedelta(days=2)]
        upsert_predictions(engine, pd.DataFrame({"ds": recent["ds"], "yhat": recent["y"] * mape_factor}), "1")
    return df


def test_decide_retrains_only_on_performance():
    assert mon.decide(0.1, 0.02) == (False, [])
    assert mon.decide(0.9, 0.02) == (False, [])      # dérive seule : pas de réentraînement forcé
    assert mon.decide(0.9, None) == (False, [])      # performance inconnue : idem
    retrain, reasons = mon.decide(0.1, 0.08)
    assert retrain and len(reasons) == 1 and "MAPE" in reasons[0]


def test_watch_warns_about_drift_and_missing_performance():
    assert mon.watch(0.1, 0.02) == []
    w = mon.watch(0.86, None)
    assert len(w) == 2 and "dérive" in w[0] and "performance non évaluée" in w[1]


def test_healthy_case_no_retrain(engine, tmp_path):
    _seed(engine)
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert not res["retrain_needed"] and not res["drift_detected"] and res["mape"] < 0.01
    assert (tmp_path / "20260301" / "data_drift.html").exists()
    assert (tmp_path / "20260301" / "performance.html").exists()


def test_drift_alone_warns_but_does_not_retrain(engine, tmp_path, monkeypatch):
    _seed(engine, shift_last_week=15.0)
    alerts = []
    monkeypatch.setattr(mon, "alert", lambda m: alerts.append(m))
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert res["drift_detected"] and res["drift_share"] >= 0.5
    assert res["retrain_needed"] is False and res["reasons"] == []
    assert any("dérive" in w for w in res["warnings"])
    assert alerts == []  # pas de webhook par défaut pour une simple dérive
    saved = json.loads((tmp_path / "20260301" / "summary.json").read_text(encoding="utf-8"))
    assert saved["drift_detected"] is True and saved["retrain_needed"] is False


def test_drift_alert_is_sent_when_enabled(engine, tmp_path, monkeypatch):
    _seed(engine, shift_last_week=15.0)
    alerts = []
    monkeypatch.setattr(mon, "alert", lambda m: alerts.append(m))
    monkeypatch.setattr(config, "ALERT_ON_DRIFT", True)
    mon.run(engine, str(tmp_path), now=NOW)
    assert len(alerts) == 1 and "dérive" in alerts[0]


def test_bad_mape_triggers_forced_retrain_and_alert(engine, tmp_path, monkeypatch):
    _seed(engine, shift_last_week=15.0, mape_factor=1.10)
    alerts = []
    monkeypatch.setattr(mon, "alert", lambda m: alerts.append(m))
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert res["retrain_needed"] is True and res["mape"] > 0.04
    assert len(res["reasons"]) == 1 and "MAPE" in res["reasons"][0]
    assert res["drift_detected"] is True
    assert len(alerts) == 1 and "réentraînement forcé" in alerts[0]


def test_missing_predictions_is_a_warning_not_a_retrain(engine, tmp_path):
    _seed(engine, with_predictions=False)
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert res["mape"] is None and res["retrain_needed"] is False
    assert any("performance non évaluée" in w for w in res["warnings"])
