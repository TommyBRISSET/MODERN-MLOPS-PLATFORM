# Modern MLOps Platform — End-to-End Production Stack

Plateforme MLOps complète, locale, sécurisée et reproductible, bâtie sur les standards industriels open-source pour la prédiction de prix immobiliers (dataset Ames Housing).

## 🚀 Technologies & Composants

- **Traitement de données & Feature Store** : PySpark 3.5, Feast 0.47
- **Suivi d'expériences & Registre de modèles** : MLflow 2.22
- **Service d'inférence en ligne** : FastAPI, Uvicorn, Pydantic V2
- **Orchestration des pipelines** : Dagster 1.8 (Software-Defined Assets)
- **Stockage relationnel & Feedback** : PostgreSQL 16
- **Monitoring & Observabilité** : Evidently AI, Prometheus, Grafana
- **Qualité, Tests & CI/CD** : uv, Ruff, Mypy, Pytest, GitHub Actions, Dependabot

---

## 📋 Démarrage rapide

### 1. Configuration de l'environnement
```bash
cp .env.example .env
# Personnalisez les mots de passe et la clé d'API dans .env
```

### 2. Démarrage de la stack
```bash
docker compose up --build -d
docker compose ps
```

### 3. Entraînement initial du modèle
```bash
docker compose exec dagster-daemon dagster job execute -m mlops.orchestration -j training_job
```

### 4. Promotion du champion & Redémarrage de l'API
```bash
docker compose exec api mlops-model status
docker compose exec api mlops-model promote
docker compose restart api
```

### 5. Vérification & Smoke Test
```bash
uv run python scripts/smoke_test.py --api-key "<VOTRE_CLÉ_API>"
```

### 6. Envoi de trafic & Rapport de monitoring
```bash
uv run python scripts/replay_holdout.py --api-key "<VOTRE_CLÉ_API>" --rows 200
docker compose exec dagster-daemon dagster job execute -m mlops.orchestration -j monitoring_job
```

---

## 📚 Documentation complète

- [Operations Runbook](docs/RUNBOOK.md) : Guide d'exploitation complet, architecture Mermaid, procédures d'incident (503, dérive, pannes), matrice des ports et accès.

---

## 🧪 Tests et Contrôles Qualité

```bash
# Vérification du formatage et des linters
uv run ruff check .
uv run ruff format --check .

# Typage statique
uv run mypy src/

# Tests unitaires et couverture
uv run pytest --cov=src/mlops tests/

# Audit de sécurité
uv run pip-audit --desc
```
