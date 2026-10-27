"""Monitoring quotidien avec Evidently : dérive des données + performance réelle vs prévue.

Usage : python -m dday.monitoring.run
La dernière ligne affichée est RETRAIN_NEEDED=true|false (lue par Airflow).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset
from sklearn.metrics import mean_absolute_percentage_error

from dday import config
from dday.db import get_engine, read_dataset, read_predictions

log = logging.getLogger("dday.monitoring")

REFERENCE_DAYS = 30   # période de référence : les 30 jours qui précèdent la fenêtre courante
CURRENT_DAYS = 7      # fenêtre courante
PERF_DAYS = 3         # fenêtre d'évaluation de la performance


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> float:
    """Rapport de dérive Evidently ; retourne la part de colonnes en dérive."""
    cols = ["y"] + config.REGRESSORS
    dd = DataDefinition(numerical_columns=cols)
    snap = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current[cols], data_definition=dd),
        Dataset.from_pandas(reference[cols], data_definition=dd),
    )
    snap.save_html(str(out_html))
    for m in snap.dict().get("metrics", []):
        if "DriftedColumnsCount" in str(m.get("metric_name", "")):
            return float(m["value"]["share"])
    raise RuntimeError("Part de colonnes en dérive introuvable dans le rapport Evidently")


def performance_report(joined: pd.DataFrame, out_html: Path) -> float:
    """Rapport de régression Evidently sur (réel, prévu) ; retourne le MAPE."""
    data = joined.rename(columns={"y": "target", "yhat": "prediction"})[["target", "prediction"]]
    dd = DataDefinition(regression=[Regression(target="target", prediction="prediction")])
    ds = Dataset.from_pandas(data, data_definition=dd)
    Report([RegressionPreset()]).run(ds, None).save_html(str(out_html))
    return float(mean_absolute_percentage_error(data["target"], data["prediction"]))


def publish(out_dir: Path) -> None:
    """Copie les rapports sur S3 si REPORT_S3_URI est défini (ex. s3://bucket/monitoring)."""
    uri = os.getenv("REPORT_S3_URI")
    if not uri or "<" in uri:  # absent, ou valeur d'exemple de .env.example non remplie
        return
    import boto3
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    for f in out_dir.glob("*"):
        s3.upload_file(str(f), bucket, f"{prefix.rstrip('/')}/{out_dir.name}/{f.name}")


def alert(message: str) -> None:
    log.error("ALERTE : %s", message)
    url = os.getenv("ALERT_WEBHOOK_URL")  # webhook Slack entrant, par ex.
    if url and "<" not in url:
        try:
            requests.post(url, json={"text": f":warning: {message}"}, timeout=10)
        except requests.RequestException:
            log.exception("Envoi de l'alerte impossible")


def decide(drift_share: float | None, mape: float | None) -> tuple[bool, list[str]]:
    reasons = []
    if drift_share is not None and drift_share >= config.DRIFT_SHARE_ALERT:
        reasons.append(f"dérive des données : {drift_share:.0%} des colonnes (seuil {config.DRIFT_SHARE_ALERT:.0%})")
    if mape is not None and mape > config.MAPE_ALERT:
        reasons.append(f"MAPE réel {mape:.2%} > seuil {config.MAPE_ALERT:.0%}")
    return bool(reasons), reasons


def run(engine=None, out_root: str = "monitoring_reports", now: pd.Timestamp | None = None) -> dict:
    engine = engine or get_engine()
    now = now or pd.Timestamp.now("UTC").tz_localize(None)
    out_dir = Path(out_root) / f"{now:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_dataset(engine, since=now - pd.Timedelta(days=REFERENCE_DAYS + CURRENT_DAYS)).dropna(subset=["y"])
    cut = now - pd.Timedelta(days=CURRENT_DAYS)
    reference, current = data[data["ds"] < cut], data[data["ds"] >= cut]

    drift_share = mape = None
    if len(reference) >= 24 * 7 and len(current) >= 24:
        drift_share = drift_report(reference, current, out_dir / "data_drift.html")
    else:
        log.warning("Données insuffisantes pour le rapport de dérive")

    preds = read_predictions(engine, since=now - pd.Timedelta(days=PERF_DAYS))
    if not preds.empty:
        preds = preds.sort_values("created_at").drop_duplicates("ds", keep="last") if "created_at" in preds else preds
        joined = preds[["ds", "yhat"]].merge(data[["ds", "y"]], on="ds", how="inner")
        if len(joined) >= 12:
            mape = performance_report(joined, out_dir / "performance.html")
    if mape is None:
        log.warning("Aucune prévision comparable aux données réelles : performance non évaluée")

    retrain, reasons = decide(drift_share, mape)
    summary = {"date": f"{now:%Y-%m-%d}", "drift_share": drift_share, "mape": mape,
               "retrain_needed": retrain, "reasons": reasons}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    publish(out_dir)
    if retrain:
        alert("Monitoring DDay : " + " ; ".join(reasons))
    log.info("Monitoring : %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    res = run()
    print(f"RETRAIN_NEEDED={'true' if res['retrain_needed'] else 'false'}")
