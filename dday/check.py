"""Vérifie en une commande que le .env permet de joindre chaque service.

Usage : python -m dday.check [--bucket NOM_DU_BUCKET]
N'affiche jamais les valeurs des secrets (relisez tout de même les messages avant de les partager).
"""
from __future__ import annotations

import argparse
import os

from dday import config  # noqa: F401  (charge le fichier .env : indispensable pour boto3 et MLflow)


def _run(name: str, fn) -> bool:
    try:
        print(f"{name:8s} OK      {fn() or ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{name:8s} ÉCHEC   {type(e).__name__} : {str(e)[:250]}")
        return False


def check_rte() -> str:
    from dday.etl.extract import _rte_token
    _rte_token()
    return "jeton obtenu"


def check_db() -> str:
    from sqlalchemy import text
    from dday.db import get_engine
    with get_engine().connect() as c:
        c.execute(text("select 1"))
    return "base joignable"


def check_mlflow() -> str:
    import mlflow
    return f"{len(mlflow.search_experiments())} expérience(s) visible(s)"


def check_s3(bucket: str) -> str:
    import boto3
    r = boto3.client("s3").list_objects_v2(Bucket=bucket, MaxKeys=1)
    return f"bucket '{bucket}' accessible ({r['KeyCount']} objet listé)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", help="Nom du bucket S3 à tester (sinon déduit de REPORT_S3_URI)")
    a = ap.parse_args()
    bucket = a.bucket
    uri = os.getenv("REPORT_S3_URI", "")
    if not bucket and uri.startswith("s3://") and "<" not in uri:
        bucket = uri.removeprefix("s3://").split("/", 1)[0]

    results = [_run("RTE", check_rte), _run("Base", check_db), _run("MLflow", check_mlflow)]
    if bucket:
        results.append(_run("S3", lambda: check_s3(bucket)))
    else:
        print("S3       ignoré  (précisez --bucket NOM_DU_BUCKET)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
