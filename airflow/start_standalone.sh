#!/usr/bin/env bash
# Démarre Airflow en local (Linux, WSL2, GitHub Codespaces). Usage : bash airflow/start_standalone.sh
# Interface : http://localhost:8080 (utilisateur admin, mot de passe dans $AIRFLOW_HOME/standalone_admin_password.txt)
set -euo pipefail
AIRFLOW_VERSION="2.10.5"

# shellcheck source=airflow/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ ! -x "$AIRFLOW_VENV/bin/airflow" ]; then
  PYV="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo ">>> Installation d'Airflow $AIRFLOW_VERSION (Python $PYV) dans $AIRFLOW_VENV : quelques minutes..."
  python3 -m venv "$AIRFLOW_VENV"
  "$AIRFLOW_VENV/bin/pip" install --upgrade pip
  "$AIRFLOW_VENV/bin/pip" install "apache-airflow==$AIRFLOW_VERSION" apache-airflow-providers-ssh apache-airflow-providers-amazon \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-$AIRFLOW_VERSION/constraints-$PYV.txt"
  export PATH="$AIRFLOW_VENV/bin:$PATH"
fi

echo ">>> Airflow démarre (dossier de travail : $AIRFLOW_HOME)"
echo ">>> Mot de passe administrateur : $AIRFLOW_HOME/standalone_admin_password.txt (créé au premier démarrage)"
exec airflow standalone
