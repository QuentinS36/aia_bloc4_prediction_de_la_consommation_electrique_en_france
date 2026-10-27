import pandas as pd

from dday.db import upsert_dataset, upsert_predictions
from dday.monitoring import run as mon
from tests.conftest import synthetic_dataset

NOW = pd.Timestamp("2026-03-01 00:00")


def _seed(engine, shift_last_week=0.0, mape_factor=1.0):
    df = synthetic_dataset(days=40, end=str(NOW))
    last_week = df["ds"] >= NOW - pd.Timedelta(days=7)
    df.loc[last_week, ["mean", "min", "max", "median", "q1", "q3"]] += shift_last_week
    df.loc[last_week, "y"] -= 1_200 * shift_last_week
    upsert_dataset(engine, df)
    recent = df[df["ds"] >= NOW - pd.Timedelta(days=2)]
    upsert_predictions(engine, pd.DataFrame({"ds": recent["ds"], "yhat": recent["y"] * mape_factor}), "1")
    return df


def test_decide_thresholds():
    assert mon.decide(0.1, 0.02) == (False, [])
    assert mon.decide(0.8, 0.02)[0] and mon.decide(0.1, 0.08)[0]


def test_healthy_case_no_retrain(engine, tmp_path):
    _seed(engine)
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert not res["retrain_needed"] and res["mape"] < 0.01
    assert (tmp_path / "20260301" / "data_drift.html").exists()
    assert (tmp_path / "20260301" / "performance.html").exists()


def test_drift_and_bad_mape_trigger_retrain(engine, tmp_path):
    _seed(engine, shift_last_week=15.0, mape_factor=1.10)
    res = mon.run(engine, str(tmp_path), now=NOW)
    assert res["retrain_needed"] and len(res["reasons"]) == 2
    assert res["drift_share"] >= 0.5 and res["mape"] > 0.04
