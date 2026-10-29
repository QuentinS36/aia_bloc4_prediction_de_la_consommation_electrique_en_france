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


def _explain_mlflow_error(e: Exception) -> Exception:
    """Traduit l'erreur brute (souvent une page HTML entière) en un message exploitable."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    text = str(e)
    if "404" in text and ("<html" in text.lower() or "<!doctype" in text.lower()):
        return RuntimeError(
            f"L'adresse {uri!r} répond par une page web (404) et non par un serveur MLflow. Vérifiez : "
            "1) qu'elle se termine par .mlflow (https://dagshub.com/<utilisateur>/<depot>.mlflow), "
            "2) l'orthographe exacte de l'utilisateur et du dépôt, 3) l'absence d'espace ou de retour à la ligne, "
            "4) que MLFLOW_TRACKING_USERNAME/PASSWORD (jeton) donnent accès au dépôt s'il est privé.")
    if "401" in text or "403" in text:
        return RuntimeError(
            "Accès refusé. Sur DagsHub, un dépôt public se lit sans authentification : une lecture réussie ne prouve donc "
            "pas que vos identifiants sont bons. Vérifiez que MLFLOW_TRACKING_USERNAME est le nom d'utilisateur DagsHub "
            "(ni l'e-mail ni le nom GitHub), que MLFLOW_TRACKING_PASSWORD est un jeton valide créé avec ce même compte, "
            "et que ce compte a le droit d'écrire dans le dépôt.")
    return e


def check_mlflow() -> str:
    import mlflow
    from mlflow import MlflowClient
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if "dagshub.com" in uri and not uri.rstrip("/").endswith(".mlflow"):
        raise RuntimeError(f"L'adresse DagsHub doit se terminer par .mlflow : {uri!r}")
    try:
        n_exp = len(mlflow.search_experiments())
        n_models = len(MlflowClient().search_registered_models(max_results=5))
    except Exception as e:  # noqa: BLE001
        raise _explain_mlflow_error(e) from None
    return f"{n_exp} expérience(s), {n_models} modèle(s) enregistré(s) visibles ({uri})"


def check_mlflow_write() -> str:
    """Test d'écriture complet (suivi, artifact, registry, alias) avec nettoyage, avant le premier entraînement."""
    import time

    from mlflow import MlflowClient

    c = MlflowClient()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"dday-check-model-{stamp}"
    try:
        exp_id = c.create_experiment(f"dday-check-{stamp}")
    except Exception as e:  # noqa: BLE001
        raise _explain_mlflow_error(e) from None
    try:
        run = c.create_run(exp_id)
        rid = run.info.run_id
        c.log_param(rid, "check", "ok")
        c.log_metric(rid, "test_mape", 0.0)
        c.log_text(rid, "ok", "check.txt")
        c.create_registered_model(model_name)
        mv = c.create_model_version(model_name, source=f"{run.info.artifact_uri}/check.txt", run_id=rid)
        c.set_registered_model_alias(model_name, "champion", mv.version)
        if str(c.get_model_version_by_alias(model_name, "champion").version) != str(mv.version):
            raise RuntimeError("l'alias champion ne renvoie pas la bonne version")
        c.set_terminated(rid)
        return "écriture, artifact, registry et alias OK (éléments de test supprimés)"
    finally:
        for cleanup in (lambda: c.delete_registered_model(model_name), lambda: c.delete_experiment(exp_id)):
            try:
                cleanup()
            except Exception:  # noqa: BLE001
                pass


def check_s3(bucket: str) -> str:
    import boto3
    r = boto3.client("s3").list_objects_v2(Bucket=bucket, MaxKeys=1)
    return f"bucket '{bucket}' accessible ({r['KeyCount']} objet listé)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", help="Nom du bucket S3 à tester (optionnel ; sinon déduit de REPORT_S3_URI)")
    ap.add_argument("--mlflow-write", action="store_true", help="Test d'écriture MLflow (crée puis supprime des éléments de test)")
    a = ap.parse_args()
    bucket = a.bucket
    uri = os.getenv("REPORT_S3_URI", "")
    if not bucket and uri.startswith("s3://") and "<" not in uri:
        bucket = uri.removeprefix("s3://").split("/", 1)[0]

    results = [_run("RTE", check_rte), _run("Base", check_db), _run("MLflow", check_mlflow)]
    if a.mlflow_write:
        results.append(_run("MLflow+", check_mlflow_write))
    if bucket:
        results.append(_run("S3", lambda: check_s3(bucket)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
