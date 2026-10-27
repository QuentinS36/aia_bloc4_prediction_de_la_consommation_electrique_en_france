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


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL n'est pas défini")
    # SQLAlchemy attend le préfixe postgresql://
    return url.replace("postgres://", "postgresql://", 1)
