import pandas as pd

from dday.db import read_dataset, upsert_dataset
from dday.etl import transform
from tests.conftest import synthetic_dataset


def _raw_rte(hours=6):
    idx = pd.date_range("2026-03-01 00:00", periods=hours * 4, freq="15min", tz="UTC")
    return pd.DataFrame({"start_date": idx.astype(str), "end_date": (idx + pd.Timedelta(minutes=15)).astype(str),
                         "value": 50_000.0})


def test_hourly_aggregation_keeps_full_hours_only():
    raw = _raw_rte(hours=3).iloc[:-2]  # dernière heure incomplète (2 relevés sur 4)
    out = transform.to_hourly_consumption(raw)
    assert len(out) == 2 and (out["y"] == 50_000).all()
    assert out["ds"].iloc[0] == pd.Timestamp("2026-03-01 01:00")  # ds = fin de période


def test_weather_aggregation_and_merge():
    ds = pd.date_range("2026-03-01 01:00", periods=2, freq="h")
    w = pd.concat([pd.DataFrame({"dept": d, "ds": ds, "temperature": t}) for d, t in [("01", 5.0), ("02", 9.0)]])
    stats = transform.aggregate_weather(w)
    assert stats.loc[0, "mean"] == 7.0 and stats.loc[0, "min"] == 5.0 and stats.loc[0, "max"] == 9.0
    merged = transform.build_dataset(transform.to_hourly_consumption(_raw_rte(2)), stats)
    assert len(merged) == 2 and {"y", "q1", "q3", "std"} <= set(merged.columns)


def test_quality_report_flags_problems():
    ok = synthetic_dataset(days=2)
    assert transform.quality_report(ok)["errors"] == []
    bad = ok.copy()
    bad.loc[3, "y"] = None
    bad.loc[5, "mean"] = 99
    errs = transform.quality_report(bad)["errors"]
    assert "valeurs manquantes" in errs and "température hors bornes plausibles" in errs
    assert transform.quality_report(pd.concat([ok, ok.iloc[:1]]))["errors"] == ["horodatages dupliqués"]


def test_load_is_idempotent(engine):
    df = synthetic_dataset(days=2)
    upsert_dataset(engine, df)
    df2 = df.copy()
    df2["y"] += 1
    upsert_dataset(engine, df2)  # rejeu avec valeurs corrigées : pas de doublon, valeur mise à jour
    out = read_dataset(engine)
    assert len(out) == len(df)
    assert abs(out["y"].iloc[0] - df2["y"].iloc[0]) < 1e-6
