from dday import check


def test_check_db_and_failure_reporting(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    assert check._run("Base", check.check_db) is True
    assert check._run("X", lambda: (_ for _ in ()).throw(ValueError("boom"))) is False
    out = capsys.readouterr().out
    assert "OK" in out and "ÉCHEC" in out and "boom" in out


def test_postgres_urls_use_psycopg2_driver():
    from dday.config import to_sqlalchemy_url
    assert to_sqlalchemy_url("postgres://u:p@h/db?sslmode=require") == "postgresql+psycopg2://u:p@h/db?sslmode=require"
    assert to_sqlalchemy_url("postgresql://u:p@h/db") == "postgresql+psycopg2://u:p@h/db"
    assert to_sqlalchemy_url("postgresql+psycopg2://u:p@h/db") == "postgresql+psycopg2://u:p@h/db"
    assert to_sqlalchemy_url("sqlite:///x.db") == "sqlite:///x.db"


def test_postgres_engine_builds_without_psycopg_v3():
    """Régression : avec SQLAlchemy 2.1, postgresql:// exigeait psycopg v3 (non installé)."""
    import pytest
    pytest.importorskip("psycopg2")
    from dday.db import get_engine
    assert get_engine("postgresql://u:p@localhost/db").dialect.driver == "psycopg2"


def test_mlflow_write_check_roundtrip_and_cleanup(mlflow_local):
    from mlflow import MlflowClient
    from dday import check
    assert "alias OK" in check.check_mlflow_write()
    c = MlflowClient()
    assert [e.name for e in c.search_experiments()] == ["Default"]  # expérience de test supprimée
    assert c.search_registered_models() == []                         # modèle de test supprimé
    assert "expérience" in check.check_mlflow()


def test_stan_policy_block_detection(monkeypatch):
    import subprocess
    from tests import conftest

    def blocked(*a, **k):
        e = OSError("bloqué"); e.winerror = 4551
        raise e
    monkeypatch.setattr(subprocess, "run", blocked)
    assert conftest.stan_blocked_by_policy() is True

    def other_error(*a, **k):
        raise OSError("autre erreur")
    monkeypatch.setattr(subprocess, "run", other_error)
    assert conftest.stan_blocked_by_policy() is False


def test_mlflow_404_html_is_explained(monkeypatch):
    import mlflow
    from mlflow.exceptions import MlflowException
    from dday import check
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/depot.mlflow")

    def boom(*a, **k):
        raise MlflowException("API request to endpoint /x failed with error code 404 != 200. Response body: '<!DOCTYPE html><html>'")
    monkeypatch.setattr(mlflow, "search_experiments", boom)
    import pytest
    with pytest.raises(RuntimeError, match="page web"):
        check.check_mlflow()


def test_dagshub_uri_without_suffix_is_rejected(monkeypatch):
    import pytest
    from dday import check
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/depot")
    with pytest.raises(RuntimeError, match=r"\.mlflow"):
        check.check_mlflow()


def test_config_strips_whitespace_from_secrets(monkeypatch):
    import importlib
    from dday import config
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/depot.mlflow \n")
    importlib.reload(config)
    import os
    assert os.environ["MLFLOW_TRACKING_URI"] == "https://dagshub.com/u/depot.mlflow"


def test_write_refused_is_explained():
    from mlflow.exceptions import MlflowException
    from dday import check
    e = MlflowException("API request to endpoint /api/2.0/mlflow/experiments/create failed with error code 403 != 200. Response body: ''")
    out = check._explain_mlflow_error(e)
    assert isinstance(out, RuntimeError) and "nom d'utilisateur DagsHub" in str(out)
