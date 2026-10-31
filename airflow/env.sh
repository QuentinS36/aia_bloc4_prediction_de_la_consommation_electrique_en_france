# À charger avec : source airflow/env.sh
# Lit airflow/.env et configure Airflow (dossier, variables du DAG, connexion SSH) pour le shell courant.
# Sert aussi à tester une tâche seule : source airflow/env.sh && airflow tasks test dday_daily predict_tomorrow 2026-01-01

_AF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$_AF_ROOT/airflow/.env" ]; then
  echo "airflow/.env introuvable : copiez airflow/.env.example en airflow/.env et remplissez-le." >&2
  return 1
fi
set -a
. "$_AF_ROOT/airflow/.env"
set +a

export AIRFLOW_HOME="${AIRFLOW_HOME:-$_AF_ROOT/.airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$_AF_ROOT/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW_VENV="${AIRFLOW_VENV:-$_AF_ROOT/.venv-airflow}"
[ -d "$AIRFLOW_VENV/bin" ] && export PATH="$AIRFLOW_VENV/bin:$PATH"

# Variables lues par le DAG (Variable.get et {{ var.value.X }}) : Airflow lit les variables AIRFLOW_VAR_*
for _v in AMI_ID SECURITY_GROUP_ID KEY_NAME DOCKERHUB_USER API_URL API_KEY EC2_INSTANCE_TYPE; do
  if [ -n "${!_v:-}" ]; then export "AIRFLOW_VAR_${_v}=${!_v}"; fi
done

# Connexion SSH vers les instances (l'hôte change à chaque exécution : pas de vérification de clé d'hôte)
if [ -n "${EC2_SSH_KEY_FILE:-}" ]; then
  export AIRFLOW_CONN_EC2_SSH="{\"conn_type\": \"ssh\", \"login\": \"${EC2_SSH_USER:-ec2-user}\", \"extra\": {\"key_file\": \"${EC2_SSH_KEY_FILE}\", \"no_host_key_check\": true}}"
else
  echo "Attention : EC2_SSH_KEY_FILE n'est pas défini, la connexion ec2_ssh n'est pas créée." >&2
fi
