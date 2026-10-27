#!/bin/bash
set -e

# Nom d'hote public du service (sans https://), ex. dday-mlflow.onrender.com.
# Render fournit RENDER_EXTERNAL_HOSTNAME ; MLFLOW_PUBLIC_HOST permet de le fixer explicitement.
PUBLIC_HOST="${MLFLOW_PUBLIC_HOST:-$RENDER_EXTERNAL_HOSTNAME}"
if [ -z "$PUBLIC_HOST" ]; then
  echo "ERREUR : definissez MLFLOW_PUBLIC_HOST (ex. dday-mlflow.onrender.com)" >&2
  exit 1
fi

echo ">>> Creating basic_auth.ini ..."
# Genere sans sed : les mots de passe contenant / ou & ne cassent plus la configuration.
# AUTH_DB_URI (optionnel) : base persistante pour les comptes et permissions. Par defaut, SQLite local,
# efface a chaque redeploiement sur Render (les comptes sont recrees plus bas, mais pas les permissions).
cat > /app/basic_auth.ini <<CONF
[mlflow]
default_permission = ${DEFAULT_PERMISSION:-READ}
database_uri = ${AUTH_DB_URI:-sqlite:////app/auth.db}
admin_username = ${ADMIN_USERNAME}
admin_password = ${ADMIN_PASSWORD}
CONF

echo ">>> Launching MLflow..."
export MLFLOW_AUTH_CONFIG_PATH=/app/basic_auth.ini
mlflow server \
  --app-name basic-auth \
  --host 0.0.0.0 \
  --port ${PORT} \
  --workers 1 \
  --allowed-hosts ${PUBLIC_HOST} \
  --cors-allowed-origins "https://${PUBLIC_HOST}" \
  --backend-store-uri "postgresql://${NEON_USERNAME}:${NEON_PASSWORD}@${NEON_HOST}/neondb?sslmode=require" \
  --default-artifact-root "${ARTIFACT_STORE_URI}" &

MLFLOW_PID=$!

echo ">>> Waiting for MLflow to start ..."
until curl -s -f \
  -H "Host: ${PUBLIC_HOST}" \
  -H "Content-Type: application/json" \
  -u "${ADMIN_USERNAME}:${ADMIN_PASSWORD}" \
  http://localhost:${PORT}/api/2.0/mlflow/experiments/search \
  -d '{"max_results": 1}' > /dev/null 2>&1; do
  echo "  Waiting..."
  sleep 3
done || true
echo ">>> MLflow is ready."

echo ">>> Initialisation completed."
wait $MLFLOW_PID