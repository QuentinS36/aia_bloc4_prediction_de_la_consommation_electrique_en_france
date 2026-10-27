# Prédiction de la consommation électrique en France – Pipeline MLOps

Projet de certification Jedha (Bloc 4 RNCP38777 – *Construire, déployer et piloter des solutions d'IA*).
Équipe : Nicolas Billard, Clément Redondo, Quentin Seng.

## 1. Objectif

Prévoir la consommation électrique française au pas horaire à J+1 pour anticiper les déséquilibres du réseau.
Le projet automatise tout le cycle de vie du modèle : collecte, entraînement, déploiement, monitoring, réentraînement et retour arrière.

| Exigence | Cible |
|---|---|
| Qualité de prévision | MAPE < 3 % (2,71 % en simulation sur 100 jours) |
| Fraîcheur des données | Mise à jour quotidienne |
| Réentraînement | Quotidien, 30 jours glissants, et forcé si dérive |
| Disponibilité | API REST documentée (OpenAPI) |
| Traçabilité | Runs, modèles et rapports versionnés, rollback possible |
| Accessibilité | Dashboard conforme aux bonnes pratiques RGAA/WCAG (voir §9) |

## 2. Architecture

```
RTE Eco2mix + Open-Meteo
          │
          ▼
   ETL (Docker, EC2) ──► Neon DB (Postgres)
                              │
          ┌───────────────────┤
          ▼                   ▼
 ML : Prophet + MLflow     Dashboard Streamlit (Hugging Face)
 (Docker, EC2)                ▲
          │                   │
          ▼                   │
 Registry MLflow ──► API FastAPI /predict ─┘
 (Neon + S3)                  │
                              ▼
                   Monitoring Evidently ──► alerte ──► réentraînement
 Orchestration : Apache Airflow     CI/CD : GitHub Actions
```

| Brique | Technologie |
|---|---|
| Sources | RTE Eco2mix API, Open-Meteo API |
| Stockage données | Neon DB (Postgres) |
| Modèle | Prophet (Meta) avec régresseurs météo |
| Suivi d'expériences et registry | MLflow (Hugging Face Space, backend Neon, artifacts S3) |
| API | FastAPI |
| Conteneurisation | Docker (un conteneur par service) |
| Orchestration | Apache Airflow |
| CI/CD | GitHub Actions, DockerHub, Hugging Face |
| Monitoring | Evidently |
| Dashboard | Streamlit |

## 3. Structure du dépôt

```
.
├── dday/                    # Package Python du pipeline
│   ├── config.py            # Configuration, régresseurs, hyperparamètres, seuils
│   ├── db.py                # Schéma Postgres et écritures idempotentes (upsert)
│   ├── etl/                 # extract.py, transform.py (contrôle qualité), main.py
│   ├── ml/                  # train.py (Prophet + MLflow), registry.py (promotion, rollback)
│   ├── api/main.py          # API FastAPI
│   └── monitoring/run.py    # Rapports Evidently, décision de réentraînement, alerte
├── airflow/dags/dday_daily.py   # DAG quotidien
├── docker/                  # Dockerfile.etl, Dockerfile.ml, Dockerfile.api
├── requirements/            # Dépendances par service (+ test.txt)
├── tests/                   # Tests pytest (ETL, ML, API, monitoring)
├── .github/workflows/       # etl.yml, ml.yml, api.yml + synchronisation Hugging Face
├── spaces/mlflow/           # Serveur MLflow (Hugging Face)
├── spaces/streamlit/        # Dashboard (Hugging Face)
├── eda-data/, ML/prophet/   # Exploration et notebooks de modélisation (historique du projet)
├── .env.example
├── environment.yml
└── README.md
```

## 4. Installation locale

```bash
conda env create -f environment.yml
conda activate dday
conda env update -f environment.yml --prune   # après modification des dépendances
```

Copier `.env.example` en `.env` et renseigner les valeurs (ne jamais versionner `.env`).

## 5. Pipeline de données (ETL)

1. **Extract** : consommation RTE (`short_term`, type REALISED) et température Open-Meteo par département.
2. **Transform** : agrégation horaire, statistiques de température (moyenne, min, max, médiane, écart-type, quartiles), contrôle qualité (manquants, doublons, outliers).
3. **Load** : insertion idempotente dans Neon DB (clé unique sur l'horodatage).

```bash
python -m dday.etl.main --days 3                 # en local
docker build -f docker/Dockerfile.etl -t dday-etl .
docker run --rm --env-file .env dday-etl
```

## 6. Entraînement, versioning et rollback

- Fenêtre d'entraînement glissante de 30 jours, régresseurs météo, prévision J+1.
- Chaque run est tracé dans MLflow : hyperparamètres, MAPE/MAE, artefacts.
- Le modèle est enregistré dans le **Model Registry** ; l'alias `champion` désigne la version servie.
- Un modèle n'est promu que si son MAPE, mesuré sur la même journée de test, est inférieur ou égal à celui du champion actuel. Le modèle enregistré est ensuite réentraîné sur la fenêtre complète (jour de test inclus).
- Réentraînement forcé (alerte monitoring) : fenêtre de 60 jours et recherche d'hyperparamètres (`--window-days 60 --grid`).
- **Rollback** : repointer l'alias `champion` vers la version précédente, puis recharger l'API (`POST /reload`).

```bash
python -m dday.ml.train                 # entraînement quotidien
python -m dday.ml.registry rollback     # retour à la version précédente
docker build -f docker/Dockerfile.ml -t dday-ml .
docker run --rm --env-file .env dday-ml
```

## 7. API

Documentation interactive : `http://<hôte>:8000/docs` (OpenAPI).

| Méthode | Route | Description |
|---|---|---|
| GET | `/predict?date=AAAA-MM-JJ` | Prévision horaire (24 valeurs, UTC) ; par défaut demain |
| GET | `/health` | État du service |
| GET | `/model-info` | Version du modèle servi |
| GET | `/metrics` | Nombre de requêtes, erreurs, latence moyenne |
| POST | `/reload` | Recharge le modèle champion (en-tête `X-API-Key` si `API_KEY` défini) |

Lancement local : `uvicorn dday.api.main:app --port 8000` (nécessite un modèle champion dans MLflow).

Exemple :

```bash
curl http://localhost:8000/predict
```

```json
{"model_version": "12", "predictions": [{"ds": "2026-06-05T00:00:00", "yhat": 46210.4}]}
```

*(exemple de format ; remplacer par la sortie réelle de votre déploiement)*

## 8. CI/CD, orchestration et monitoring

**GitHub Actions** (à chaque push) :
- ETL, ML, API : tests `pytest` → build Docker → push DockerHub.
- Dashboard et MLflow : tests → synchronisation du Space Hugging Face.
- Secrets stockés dans GitHub Secrets.

**Airflow** – DAG `dday_daily` (5 h chaque jour) : création de l'instance EC2 → ETL → monitoring Evidently → entraînement (normal ou forcé selon le monitoring) → rechargement de l'API → arrêt de l'instance (toujours exécuté, même en cas d'échec). Variables Airflow : `AMI_ID`, `SECURITY_GROUP_ID`, `KEY_NAME`, `DOCKERHUB_USER`, `API_URL`, `API_KEY` ; connexion `ec2_ssh`.

**Evidently** – chaque jour :
- dérive des données (température, consommation) par rapport à la référence ;
- performance réelle (MAPE sur la veille) ;
- rapport HTML stocké sur S3 ;
- seuil d'alerte : MAPE > 4 % ou dérive détectée → notification et réentraînement forcé.

## 9. Accessibilité

- Contrastes de couleurs conformes WCAG AA, palette compatible daltonisme.
- Description textuelle (titre + résumé chiffré) associée à chaque graphique.
- Contrôles Streamlit utilisables au clavier, libellés explicites.
- API : messages d'erreur explicites, documentation lisible.

## 10. Tests

```bash
pip install -r requirements/ml.txt -r requirements/api.txt -r requirements/test.txt
python -m pytest -q tests
```

Les tests utilisent des données synthétiques, SQLite et un MLflow local : ils ne nécessitent ni secret ni accès réseau.

## 11. Déploiement

| Composant | Démo | Production cible |
|---|---|---|
| Base de données | Neon DB | AWS Redshift |
| Application web | Hugging Face Space | Serveur dédié ou VPS |
| Airflow | Local | Serveur dédié |
| ETL / ML | EC2 à la demande | EC2 / ECS |

## 12. Perspectives

Sécurité (rôles IAM minimaux, rotation des secrets), monitoring continu avec alerting, dashboard enrichi, mise à l'échelle (ECS/Kubernetes).
