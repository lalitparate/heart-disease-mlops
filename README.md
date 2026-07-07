# MLOps Assignment 01 — Heart Disease Prediction Pipeline

**Course:** AIMLCZG523 — Machine Learning Operations  
**Dataset:** Heart Disease UCI Dataset  
**Problem:** Binary classification — predict presence/absence of heart disease

---

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py          # Script to download UCI dataset
│   └── heart.csv                 # Dataset (downloaded by script)
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Preprocessing pipeline
│   ├── train.py                  # Model training + MLflow logging
│   ├── evaluate.py               # Evaluation utilities
│   └── predict.py                # Inference utilities
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   └── schemas.py                # Pydantic request/response schemas
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py        # Unit tests for preprocessing
│   ├── test_model.py             # Unit tests for model
│   └── test_api.py               # Unit tests for API
├── k8s/
│   ├── deployment.yaml           # Kubernetes Deployment manifest
│   ├── service.yaml              # Kubernetes Service manifest
│   └── ingress.yaml              # Kubernetes Ingress manifest
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD pipeline
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana_dashboard.json    # Grafana dashboard definition
├── models/                       # Saved models (gitignored except .gitkeep)
│   └── .gitkeep
├── mlruns/                       # MLflow tracking directory
├── screenshots/                  # Deployment/CI screenshots folder
│   └── .gitkeep
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Local multi-service compose
├── requirements.txt              # Python dependencies
├── .gitignore
└── report.md                     # Final written report
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/download_data.py
```

### 3. Run EDA Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

### 4. Train Models

```bash
python src/train.py
```

### 5. View MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 6. Run API Locally

```bash
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs  (Swagger UI)
```

### 7. Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
  }'
```

### 8. Run Tests

```bash
pytest tests/ -v --tb=short
```

### 9. Docker Build & Run

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

### 10. Docker Compose (API + MLflow + Prometheus + Grafana)

```bash
docker-compose up --build
```

---

## Kubernetes Deployment

```bash
# Minikube
minikube start
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl get pods
kubectl get svc
minikube service heart-disease-api-service --url
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

---

## Models Trained

1. **Logistic Regression** — baseline linear classifier
2. **Random Forest** — ensemble tree-based classifier
3. **XGBoost** — gradient boosting (optional tuned model)

Best model is automatically selected by ROC-AUC and saved.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (primary selection metric)
- Cross-validation (5-fold stratified)
- Confusion matrix

---

## CI/CD Pipeline (GitHub Actions)

Triggers on every push/PR to `main`:

1. **Lint** — flake8 code quality check  
2. **Unit Tests** — pytest with coverage report  
3. **Train & Validate** — model training smoke test  
4. **Docker Build** — verify container builds successfully  
