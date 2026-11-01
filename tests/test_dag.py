"""Tests du DAG Airflow : structure, logique de branchement et gestion des instances EC2 (EC2 simulé par moto).

Ces tests exigent apache-airflow et moto ; ils sont ignorés sinon (voir .github/workflows/dag.yml pour l'installation).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")
pytest.importorskip("moto")
import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

DAG_FILE = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "dday_daily.py"


@pytest.fixture(scope="module")
def dagmod():
    spec = importlib.util.spec_from_file_location("dday_daily", DAG_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dday_daily"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "eu-west-3"}.items():
        monkeypatch.setenv(k, v)


def test_dag_structure(dagmod):
    dag = dagmod.dag
    assert dag.schedule_interval == "0 5 * * *" and dag.max_active_runs == 1 and dag.catchup is False
    ids = {t.task_id for t in dag.tasks}
    assert ids == {"create_ec2", "etl", "monitoring", "choose_training", "train_daily", "train_forced",
                   "reload_api", "predict_tomorrow", "terminate_ec2"}
    assert dag.get_task("terminate_ec2").trigger_rule == "all_done"          # toujours exécuté
    assert dag.get_task("reload_api").trigger_rule == "none_failed_min_one_success"
    assert dag.get_task("choose_training").downstream_task_ids == {"train_daily", "train_forced"}
    # terminate_ec2 est en aval de chaque tâche qui utilise l'instance
    for t in ("etl", "monitoring", "train_daily", "train_forced", "reload_api", "predict_tomorrow"):
        assert "terminate_ec2" in dag.get_task(t).downstream_task_ids


def test_commands_use_the_right_images_and_arguments(dagmod):
    dag = dagmod.dag
    assert "dday-etl" in dag.get_task("etl").command
    assert "python -m dday.monitoring.run" in dag.get_task("monitoring").command
    assert dag.get_task("train_daily").command.rstrip().endswith("python -m dday.ml.train")
    assert "--window-days 60 --grid" in dag.get_task("train_forced").command


class FakeTI:
    def __init__(self, value):
        self.value = value

    def xcom_pull(self, task_ids=None):
        return self.value


@pytest.mark.parametrize("output, expected", [
    ("DRIFT_DETECTED=true\nRETRAIN_NEEDED=true", "train_forced"),
    ("DRIFT_DETECTED=true\nRETRAIN_NEEDED=false", "train_daily"),   # dérive seule : entraînement normal
    ("RETRAIN_NEEDED=false", "train_daily"),
    ("", "train_daily"),
])
def test_branching_follows_retrain_flag(dagmod, output, expected):
    import base64
    assert dagmod.choose_training(FakeTI(output)) == expected
    assert dagmod.choose_training(FakeTI(base64.b64encode(output.encode()).decode())) == expected  # sortie encodée


def _setup_aws(monkeypatch, dagmod):
    ec2 = boto3.client("ec2")
    ami = ec2.describe_images()["Images"][0]["ImageId"]
    sg = ec2.describe_security_groups()["SecurityGroups"][0]["GroupId"]
    ec2.create_key_pair(KeyName="dday-key")
    values = {"AMI_ID": ami, "SECURITY_GROUP_ID": sg, "KEY_NAME": "dday-key"}
    monkeypatch.setattr(dagmod.Variable, "get", staticmethod(lambda key, default_var=None, **k: values.get(key, default_var)))
    return ec2


def _states(ec2):
    return sorted(i["State"]["Name"] for r in ec2.describe_instances()["Reservations"] for i in r["Instances"])


@mock_aws
def test_create_then_terminate_instance(dagmod, monkeypatch):
    ec2 = _setup_aws(monkeypatch, dagmod)
    monkeypatch.setattr(dagmod.socket, "create_connection", lambda *a, **k: type("S", (), {"close": lambda self: None})())
    info = dagmod.create_ec2()
    assert info["instance_id"].startswith("i-") and info["ip"]
    assert _states(ec2) == ["running"]
    dagmod.terminate_ec2(FakeTI(info))
    assert _states(ec2) == ["terminated"]


@mock_aws
def test_instance_is_terminated_if_ssh_never_answers(dagmod, monkeypatch):
    ec2 = _setup_aws(monkeypatch, dagmod)

    def refuse(*a, **k):
        raise OSError("refusé")
    monkeypatch.setattr(dagmod.socket, "create_connection", refuse)
    monkeypatch.setattr(dagmod.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        dagmod.create_ec2()
    assert _states(ec2) == ["terminated"]  # aucune instance orpheline malgré l'échec


@mock_aws
def test_orphans_from_a_previous_run_are_terminated(dagmod, monkeypatch):
    ec2 = _setup_aws(monkeypatch, dagmod)
    ami = ec2.describe_images()["Images"][0]["ImageId"]
    ec2.run_instances(ImageId=ami, MinCount=1, MaxCount=1,
                      TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "dday-batch"}]}])
    other = ec2.run_instances(ImageId=ami, MinCount=1, MaxCount=1,
                              TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "autre-projet"}]}])
    monkeypatch.setattr(dagmod.socket, "create_connection", lambda *a, **k: type("S", (), {"close": lambda self: None})())
    dagmod.create_ec2()
    assert sorted(_states(ec2)) == ["running", "running", "terminated"]
    # l'instance d'un autre projet n'est jamais touchée
    assert ec2.describe_instances(InstanceIds=[other["Instances"][0]["InstanceId"]])["Reservations"][0]["Instances"][0]["State"]["Name"] == "running"


# ---- Connexion SSH : régression « Missing required param: remote_host » ----
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

import yaml  # noqa: E402

AIRFLOW_DIR = Path(__file__).resolve().parents[1] / "airflow"
SSH_TASKS = ("etl", "monitoring", "train_daily", "train_forced")


def _conn_from_env_sh(tmp_path):
    """JSON de connexion produit réellement par airflow/env.sh (exécuté dans bash)."""
    (tmp_path / "airflow").mkdir()
    (tmp_path / "airflow" / "env.sh").write_text((AIRFLOW_DIR / "env.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "airflow" / ".env").write_text("EC2_SSH_KEY_FILE=/tmp/ec2.pem\n", encoding="utf-8")
    out = subprocess.run(["bash", "-c", 'source "$1"; printf %s "$AIRFLOW_CONN_EC2_SSH"', "_", str(tmp_path / "airflow" / "env.sh")],
                         capture_output=True, text=True, check=True)
    return out.stdout


def _conn_from_compose():
    raw = yaml.safe_load((AIRFLOW_DIR / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["airflow"]["environment"]["AIRFLOW_CONN_EC2_SSH"]
    return re.sub(r"\$\{EC2_SSH_USER:-([^}]*)\}", r"\1", raw)


@pytest.mark.parametrize("source", ["env.sh", "docker-compose"])
@pytest.mark.parametrize("task_id", SSH_TASKS)
def test_ssh_tasks_can_open_their_connection(dagmod, monkeypatch, tmp_path, source, task_id):
    conn = _conn_from_env_sh(tmp_path) if source == "env.sh" else _conn_from_compose()
    json.loads(conn)  # JSON valide
    monkeypatch.setenv("AIRFLOW_CONN_EC2_SSH", conn)
    op = dagmod.dag.get_task(task_id)
    op.remote_host = "13.48.1.2"          # valeur obtenue après rendu du modèle Jinja
    op.__dict__.pop("ssh_hook", None)     # vide le cache de la propriété
    hook = op.ssh_hook                    # lève « Missing required param: remote_host » si la connexion n'a pas de host
    assert hook.remote_host == "13.48.1.2" and hook.username == "ec2-user"


def test_templates_render_with_the_instance_address_and_image(dagmod):
    op = dagmod.dag.get_task("etl")
    context = {"ti": FakeTI({"ip": "13.48.1.2", "instance_id": "i-1"}), "var": {"value": {"DOCKERHUB_USER": "moi"}}}
    assert op.render_template(op.remote_host, context) == "13.48.1.2"
    assert "moi/dday-etl:latest" in op.render_template(op.command, context)
