"""ETL quotidien : extract -> transform -> contrôle qualité -> load (Neon).

Usage : python -m dday.etl.main --days 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from dday.db import get_engine, init_schema, upsert_dataset
from dday.etl import extract, transform

log = logging.getLogger("dday.etl")


def run(days: int = 3, engine=None) -> dict:
    """Recharge les `days` derniers jours (recouvrement volontaire : upsert idempotent, rattrape les retards RTE)."""
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=days)

    raw = extract.fetch_rte_consumption(start - dt.timedelta(hours=1), end)
    weather = extract.fetch_weather(past_days=days + 1, forecast_days=1)

    df = transform.build_dataset(
        transform.to_hourly_consumption(raw), transform.aggregate_weather(weather)
    )
    report = transform.quality_report(df)
    for w in report["warnings"]:
        log.warning(w)
    if report["errors"]:
        raise ValueError(f"Contrôle qualité échoué : {report['errors']}")

    engine = engine or get_engine()
    init_schema(engine)
    n = upsert_dataset(engine, df)
    log.info("ETL terminé : %s lignes chargées (%s -> %s)", n, df["ds"].min(), df["ds"].max())
    return {**report, "loaded": n}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    run(ap.parse_args().days)
