"""DAG quotidien : EC2 à la demande -> ETL -> monitoring Evidently -> entraînement -> rechargement API -> prévision du lendemain -> arrêt EC2.

Pré-requis AWS (une fois) : AMI avec Docker et le fichier /home/ec2-user/.env, rôle IAM, groupe de sécurité SSH.
Variables Airflow : AMI_ID, SECURITY_GROUP_ID, KEY_NAME, DOCKERHUB_USER, API_URL, API_KEY.
Connexion Airflow : ec2_ssh (utilisateur ec2-user + clé privée).
"""
from __future__ import annotations

import base64
import socket
import time
from datetime import datetime, timedelta

import boto3
import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator

IMG_ETL = "{{ var.value.DOCKERHUB_USER }}/dday-etl:latest"
IMG_ML = "{{ var.value.DOCKERHUB_USER }}/dday-ml:latest"
RUN = "docker pull {img} && docker run --rm --env-file /home/ec2-user/.env {img} {cmd}"


def create_ec2(**_) -> dict:
    ec2 = boto3.client("ec2")
    inst = ec2.run_instances(
        ImageId=Variable.get("AMI_ID"), InstanceType="t3.large", MinCount=1, MaxCount=1,
        KeyName=Variable.get("KEY_NAME"), SecurityGroupIds=[Variable.get("SECURITY_GROUP_ID")],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "dday-batch"}]}],
    )["Instances"][0]
    iid = inst["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    for _ in range(30):  # attend que SSH réponde
        try:
            socket.create_connection((ip, 22), timeout=5).close()
            break
        except OSError:
            time.sleep(10)
    else:
        raise TimeoutError("SSH injoignable")
    return {"instance_id": iid, "ip": ip}


def terminate_ec2(ti, **_) -> None:
    info = ti.xcom_pull(task_ids="create_ec2")
    if info:
        boto3.client("ec2").terminate_instances(InstanceIds=[info["instance_id"]])


def _decode(out) -> str:
    try:  # selon la version du provider SSH, la sortie est encodée en base64
        return base64.b64decode(out).decode()
    except Exception:
        return str(out)


def choose_training(ti, **_) -> str:
    out = _decode(ti.xcom_pull(task_ids="monitoring"))
    return "train_forced" if "RETRAIN_NEEDED=true" in out else "train_daily"


def reload_api(**_) -> None:
    r = requests.post(Variable.get("API_URL") + "/reload", headers={"X-API-Key": Variable.get("API_KEY", "")}, timeout=60)
    r.raise_for_status()


def predict_tomorrow(**_) -> None:
    """Appelle /predict (demain par défaut) : la prévision est enregistrée dans Neon et alimente le monitoring de performance."""
    r = requests.get(Variable.get("API_URL") + "/predict", timeout=300)  # délai long : l'API peut être en veille
    r.raise_for_status()


def ssh(task_id: str, cmd: str, **kw) -> SSHOperator:
    return SSHOperator(task_id=task_id, ssh_conn_id="ec2_ssh", cmd_timeout=3600,
                       remote_host="{{ ti.xcom_pull(task_ids='create_ec2')['ip'] }}", command=cmd, **kw)


with DAG(
    "dday_daily", start_date=datetime(2026, 1, 1), schedule="0 5 * * *", catchup=False, max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["dday", "mlops"],
) as dag:
    create = PythonOperator(task_id="create_ec2", python_callable=create_ec2)
    etl = ssh("etl", RUN.format(img=IMG_ETL, cmd=""))
    monitoring = ssh("monitoring", RUN.format(img=IMG_ML, cmd="python -m dday.monitoring.run"), do_xcom_push=True)
    branch = BranchPythonOperator(task_id="choose_training", python_callable=choose_training)
    train_daily = ssh("train_daily", RUN.format(img=IMG_ML, cmd="python -m dday.ml.train"))
    # Réentraînement forcé (dérive ou MAPE > seuil) : fenêtre élargie + recherche d'hyperparamètres
    train_forced = ssh("train_forced", RUN.format(img=IMG_ML, cmd="python -m dday.ml.train --window-days 60 --grid"))
    reload = PythonOperator(task_id="reload_api", python_callable=reload_api, trigger_rule="none_failed_min_one_success")
    predict = PythonOperator(task_id="predict_tomorrow", python_callable=predict_tomorrow)
    # all_done : l'instance est toujours arrêtée, même en cas d'échec (maîtrise des coûts)
    terminate = PythonOperator(task_id="terminate_ec2", python_callable=terminate_ec2, trigger_rule="all_done")

    create >> etl >> monitoring >> branch >> [train_daily, train_forced] >> reload >> predict >> terminate
    [etl, monitoring, train_daily, train_forced, reload, predict] >> terminate
