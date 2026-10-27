#!/bin/bash
set -e

echo ">>> Creating basic_auth.ini ..."
sed \
  -e "s/PLACEHOLDER_ADMIN_USERNAME/${ADMIN_USERNAME}/g" \
  -e "s/PLACEHOLDER_ADMIN_PASSWORD/${ADMIN_PASSWORD}/g" \
  /app/basic_auth.ini.template > /app/basic_auth.ini

echo ">>> Launching MLflow..."
export MLFLOW_AUTH_CONFIG_PATH=/app/basic_auth.ini
mlflow server \
  --app-name basic-auth \
  --host 0.0.0.0 \
  --port ${PORT} \
  --workers 1 \
  --allowed-hosts ${HF_DEMODAY_SPACE_URL} \
  --cors-allowed-origins "https://${HF_DEMODAY_SPACE_URL}" \
  --backend-store-uri "postgresql://${NEON_USERNAME}:${NEON_PASSWORD}@${NEON_HOST}/neondb?sslmode=require" \
  --default-artifact-root "${ARTIFACT_STORE_URI}" &

MLFLOW_PID=$!

echo ">>> Waiting for MLflow to start ..."
until curl -s -f \
  -H "Host: ${HF_DEMODAY_SPACE_URL}" \
  -H "Content-Type: application/json" \
  -u "${ADMIN_USERNAME}:${ADMIN_PASSWORD}" \
  http://localhost:${PORT}/api/2.0/mlflow/experiments/search \
  -d '{"max_results": 1}' > /dev/null 2>&1; do
  echo "  Waiting..."
  sleep 3
done || true
echo ">>> MLflow is ready."

echo ">>> Creating user accounts ..."

create_user() {
  local USERNAME=$1
  local PASSWORD=$2
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:${PORT}/api/2.0/mlflow/users/create \
    -H "Host: ${HF_DEMODAY_SPACE_URL}" \
    -u "${ADMIN_USERNAME}:${ADMIN_PASSWORD}" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${USERNAME}\", \"password\": \"${PASSWORD}\"}")

  if [ "$RESPONSE" = "200" ]; then
    echo "  Account '${USERNAME}' created."
  elif [ "$RESPONSE" = "400" ]; then
    echo "  Account '${USERNAME}' already exists, ignored."
  else
    echo "  Error for account '${USERNAME}' (HTTP ${RESPONSE})."
  fi
}

create_user "${USERNAME_CR}" "${PASSWORD_CR}"
create_user "${USERNAME_NB}" "${PASSWORD_NB}"
create_user "${USERNAME_YR}" "${PASSWORD_YR}"

echo ">>> Initialisation completed."
wait $MLFLOW_PID