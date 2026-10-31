"""Vérifie la configuration d'Airflow sans lancer d'instance EC2 ni modifier quoi que ce soit.

Avec Docker :  docker compose -f airflow/docker-compose.yml exec airflow python /opt/airflow/check_config.py
Sans Docker :  source airflow/env.sh && python airflow/check_config.py
N'affiche jamais les secrets (clé API, clés AWS).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

REQUIRED = ["AMI_ID", "SECURITY_GROUP_ID", "KEY_NAME", "DOCKERHUB_USER", "API_URL", "API_KEY"]
CONTAINER_KEY = Path("/opt/airflow/keys/ec2.pem")


def var(name: str) -> str:
    """Valeur vue par Airflow (AIRFLOW_VAR_X) ou, à défaut, la variable brute du fichier .env."""
    return (os.getenv(f"AIRFLOW_VAR_{name}") or os.getenv(name) or "").strip()


def _ec2():
    import boto3
    return boto3.client("ec2")


def check_variables() -> str:
    missing = [n for n in REQUIRED if not var(n)]
    if missing:
        raise RuntimeError("variables vides ou absentes : " + ", ".join(missing))
    return "les 6 variables du DAG sont renseignées"


def check_api() -> str:
    url = var("API_URL")
    if url.endswith("/") or "/predict" in url or not url.startswith("http"):
        raise RuntimeError("API_URL doit être l'adresse seule, sans / final ni /predict (ex. https://mon-api.onrender.com)")
    r = requests.get(url + "/health", timeout=120)  # délai long : l'API gratuite peut être en veille
    r.raise_for_status()
    return f"{url}/health répond ({r.json()})"


def check_credentials() -> str:
    import boto3
    return "identité AWS : " + boto3.client("sts").get_caller_identity()["Arn"]


def check_region() -> str:
    import boto3
    return "région : " + (boto3.session.Session().region_name or "(non définie)")


def check_ami() -> str:
    img = _ec2().describe_images(ImageIds=[var("AMI_ID")])["Images"][0]
    if img["State"] != "available":
        raise RuntimeError(f"l'image est dans l'état « {img['State']} » (attendre « available »)")
    return f"{img['ImageId']} disponible ({img.get('Name', 'sans nom')})"


def check_security_group() -> str:
    ec2 = _ec2()
    sg = ec2.describe_security_groups(GroupIds=[var("SECURITY_GROUP_ID")])["SecurityGroups"][0]
    default = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if default and sg["VpcId"] != default[0]["VpcId"]:
        raise RuntimeError(f"ce groupe appartient au VPC {sg['VpcId']}, pas au VPC par défaut {default[0]['VpcId']} "
                           "où le DAG lance l'instance : choisir un groupe du VPC par défaut")
    ssh = [r for r in sg["IpPermissions"] if r.get("FromPort") is not None and r["FromPort"] <= 22 <= r["ToPort"]]
    if not ssh:
        raise RuntimeError("aucune règle entrante n'autorise le port 22 (SSH)")
    sources = [x["CidrIp"] for r in ssh for x in r.get("IpRanges", [])]
    note = " (0.0.0.0/0 : ouvert à tous, à restreindre à votre adresse)" if "0.0.0.0/0" in sources else ""
    return f"{sg['GroupId']} : SSH autorisé depuis {', '.join(sources) or 'un autre groupe'}{note}"


def check_key_pair() -> str:
    kp = _ec2().describe_key_pairs(KeyNames=[var("KEY_NAME")])["KeyPairs"][0]
    return f"paire de clés « {kp['KeyName']} » trouvée"


def check_ssh_key_file() -> str:
    path = CONTAINER_KEY if CONTAINER_KEY.exists() else Path(os.getenv("EC2_SSH_KEY_FILE", ""))
    if not path or not path.is_file():
        raise RuntimeError("fichier de clé introuvable : copier la clé dans airflow/keys/ec2.pem (Docker) ou renseigner EC2_SSH_KEY_FILE")
    if not path.read_text(errors="ignore").lstrip().startswith("-----BEGIN"):
        raise RuntimeError("le fichier n'a pas le format d'une clé privée (.pem)")
    return f"clé privée présente ({path.stat().st_size} octets)"


def check_dockerhub_images() -> str:
    user, found = var("DOCKERHUB_USER"), []
    for image in ("dday-etl", "dday-ml"):
        r = requests.get(f"https://hub.docker.com/v2/repositories/{user}/{image}/tags/latest", timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{user}/{image}:latest introuvable (HTTP {r.status_code}) : image absente, dépôt privé "
                               "ou nom d'utilisateur Docker Hub incorrect (l'instance ne peut pas tirer une image privée)")
        found.append(f"{image}")
    return "images publiques trouvées : " + ", ".join(found)


def check_orphans() -> str:
    res = _ec2().describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["dday-batch"]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]
    n = sum(len(r["Instances"]) for r in res)
    if n:
        raise RuntimeError(f"{n} instance(s) dday-batch déjà en cours (elles seront arrêtées au prochain lancement du DAG)")
    return "aucune instance dday-batch en cours"


CHECKS = [
    ("Variables", check_variables), ("API", check_api), ("Identité AWS", check_credentials), ("Région", check_region),
    ("Image (AMI)", check_ami), ("Groupe de sécurité", check_security_group), ("Paire de clés", check_key_pair),
    ("Clé SSH locale", check_ssh_key_file), ("Images Docker Hub", check_dockerhub_images), ("Instances en cours", check_orphans),
]


def main() -> int:
    ok_all = True
    for label, fn in CHECKS:
        try:
            print(f"{label:20s} OK      {fn()}")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"{label:20s} ÉCHEC   {type(e).__name__} : {str(e)[:260]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
