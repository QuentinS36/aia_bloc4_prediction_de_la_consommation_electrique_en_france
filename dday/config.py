"""Configuration centralisée, lue dans les variables d'environnement (.env en local)."""
import os

from dotenv import load_dotenv

load_dotenv()

# Régresseurs météo utilisés par Prophet (retenus lors du fine-tuning, notebooks 04 et 05)
REGRESSORS = ["mean", "min", "max", "std", "q1", "q3"]
HYPERPARAMS = {"changepoint_prior_scale": 0.35, "seasonality_prior_scale": 0.5}
TRAIN_WINDOW_DAYS = 30
MODEL_NAME = os.getenv("MODEL_NAME", "dday-prophet-conso")
CHAMPION_ALIAS = "champion"

# Seuils de monitoring (cf. cahier des charges)
MAPE_TARGET = float(os.getenv("MAPE_TARGET", "0.03"))      # objectif métier : 3 %
MAPE_ALERT = float(os.getenv("MAPE_ALERT", "0.04"))        # alerte + réentraînement forcé : 4 %
DRIFT_SHARE_ALERT = float(os.getenv("DRIFT_SHARE_ALERT", "0.5"))  # part de colonnes en dérive


def to_sqlalchemy_url(url: str) -> str:
    """Impose le pilote psycopg2 pour Postgres.

    SQLAlchemy 2.1 utilise psycopg (v3) par défaut pour postgresql://, alors que seul psycopg2 est installé.
    Les autres bases (sqlite pour les tests) sont laissées telles quelles.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL n'est pas défini")
    return to_sqlalchemy_url(url)

if __name__ == "__main__":  # diagnostic : python -m dday.config  (n'affiche jamais les valeurs)
    from dotenv import find_dotenv

    print(".env trouvé :", find_dotenv() or "AUCUN")
    for var in ("RTE_SECRET_KEY", "DATABASE_URL", "MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                "MLFLOW_TRACKING_PASSWORD", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION", "MODEL_NAME"):
        print(f"{var:28s}", "défini" if os.getenv(var) else "ABSENT ou VIDE")
