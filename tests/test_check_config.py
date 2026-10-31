"""Tests de airflow/check_config.py avec un AWS simulé (moto) et un réseau simulé."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("moto")
import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "airflow" / "check_config.py"


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location("check_config", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_config"] = mod
    spec.loader.exec_module(mod)
    return mod


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _env(monkeypatch, tmp_path, **over):
    for k, v in {"AWS_ACCESS_KEY_ID": "t", "AWS_SECRET_ACCESS_KEY": "t", "AWS_DEFAULT_REGION": "eu-west-3"}.items():
        monkeypatch.setenv(k, v)
    ec2 = boto3.client("ec2")
    values = {
        "AMI_ID": ec2.describe_images()["Images"][0]["ImageId"],
        "SECURITY_GROUP_ID": ec2.describe_security_groups()["SecurityGroups"][0]["GroupId"],
        "KEY_NAME": ec2.create_key_pair(KeyName="dday-key")["KeyName"],
        "DOCKERHUB_USER": "moi", "API_URL": "https://api.exemple.test", "API_KEY": "secret",
    }
    values.update(over)
    for k, v in values.items():
        monkeypatch.setenv(f"AIRFLOW_VAR_{k}", v)
    pem = tmp_path / "ec2.pem"
    pem.write_text("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----\n")
    monkeypatch.setenv("EC2_SSH_KEY_FILE", str(pem))
    return values


@mock_aws
def test_everything_ok(cc, monkeypatch, tmp_path, capsys):
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(cc.requests, "get", lambda url, **k: Resp(200, {"status": "ok"}))
    # le groupe de sécurité par défaut de moto n'a pas de règle SSH : on l'ajoute
    ec2 = boto3.client("ec2")
    sg = ec2.describe_security_groups()["SecurityGroups"][0]["GroupId"]
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "203.0.113.7/32"}]}])
    assert cc.main() == 0
    out = capsys.readouterr().out
    assert out.count("OK") == len(cc.CHECKS) and "ÉCHEC" not in out
    assert "secret" not in out  # la clé API n'est jamais affichée


@mock_aws
def test_each_mistake_is_named(cc, monkeypatch, tmp_path, capsys):
    _env(monkeypatch, tmp_path, AMI_ID="ami-00000000000000000", KEY_NAME="mauvaise-cle",
         API_URL="https://api.exemple.test/", DOCKERHUB_USER="")
    monkeypatch.setattr(cc.requests, "get", lambda url, **k: Resp(404))
    assert cc.main() == 1
    out = capsys.readouterr().out
    assert "variables vides ou absentes : DOCKERHUB_USER" in out
    assert "sans / final ni /predict" in out
    assert "Image (AMI)" in out and "ÉCHEC" in out.split("Image (AMI)")[1].splitlines()[0]
    assert "Paire de clés" in out and "ÉCHEC" in out.split("Paire de clés")[1].splitlines()[0]
    assert "aucune règle entrante n'autorise le port 22" in out


@mock_aws
def test_private_or_missing_dockerhub_image_is_explained(cc, monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(cc.requests, "get", lambda url, **k: Resp(404))
    with pytest.raises(RuntimeError, match="introuvable"):
        cc.check_dockerhub_images()
