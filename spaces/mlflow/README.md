# Serveur MLflow (déployé sur Render)

Serveur MLflow avec authentification de base, base de métadonnées Neon (Postgres) et artifacts sur S3.

Déploiement : **Web Service** Render en mode **Docker**, relié au dépôt GitHub
(Dockerfile : `spaces/mlflow/Dockerfile`, contexte : `spaces/mlflow`). Chaque push sur la branche suivie redéploie le service.

## Variables d'environnement à définir dans Render

| Variable | Rôle |
|---|---|
| `MLFLOW_PUBLIC_HOST` | Nom d'hôte public sans `https://` (ex. `dday-mlflow.onrender.com`). Repli : `RENDER_EXTERNAL_HOSTNAME` |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | Compte administrateur |
| `NEON_USERNAME`, `NEON_PASSWORD`, `NEON_HOST` | Base de métadonnées (base `neondb`) |
| `ARTIFACT_STORE_URI` | Racine des artifacts, ex. `s3://bucket/mlflow` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Accès S3 du serveur (navigation des artifacts dans l'interface) |
| `USERNAME_CR`/`PASSWORD_CR`, `USERNAME_NB`/`PASSWORD_NB`, `USERNAME_YR`/`PASSWORD_YR` | Comptes des membres de l'équipe |
| `DEFAULT_PERMISSION` (optionnel) | `READ` par défaut |
| `AUTH_DB_URI` (optionnel) | Base persistante pour les comptes et permissions (sinon SQLite local éphémère) |

Ne pas définir `PORT` : Render le fournit.
