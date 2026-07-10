"""
Unit tests for the FastAPI application.
Run: pytest tests/test_api.py -v
"""

from src.preprocess import ALL_FEATURES, build_preprocessor
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os
import sys
import pytest
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Patch model/preprocessor so tests run without trained files ───────────────

# Build a tiny fitted mock
_np = __import__("numpy")
_rng = _np.random.default_rng(42)
_n = 60
_mock_X = pd.DataFrame({c: _rng.random(_n) for c in ALL_FEATURES})
_mock_y = _rng.integers(0, 2, _n)
_mock_pre = build_preprocessor()
_mock_pre.fit(_mock_X)
_mock_X_t = _mock_pre.transform(_mock_X)
_mock_model = LogisticRegression(max_iter=200).fit(_mock_X_t, _mock_y)


def _patched_load_model(*args, **kwargs):
    return _mock_model


def _patched_load_preprocessor(*args, **kwargs):
    return _mock_pre


@pytest.fixture(autouse=True)
def patch_io(monkeypatch):
    monkeypatch.setattr("src.predict.load_model",         _patched_load_model)
    monkeypatch.setattr("src.predict.load_preprocessor",  _patched_load_preprocessor)
    monkeypatch.setattr("api.main.load_model",            _patched_load_model)


@pytest.fixture
def client(patch_io):
    from api.main import app
    app.state.model_loaded = True
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
    "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


# ── Health endpoints ──────────────────────────────────────────────────────────

def test_root_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_type_returns_422(client):
    payload = {**VALID_PAYLOAD, "age": "not_a_number"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_label_is_string(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert isinstance(r.json()["label"], str)


# ── /predict/batch ────────────────────────────────────────────────────────────

def test_batch_predict_two_records(client):
    r = client.post("/predict/batch", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_batch_predict_empty_returns_422(client):
    r = client.post("/predict/batch", json={"records": []})
    assert r.status_code == 422


def test_batch_predict_structure(client):
    r = client.post("/predict/batch", json={"records": [VALID_PAYLOAD]})
    pred = r.json()["predictions"][0]
    assert "prediction" in pred
    assert "probability" in pred
    assert "label" in pred


# ── 404 ───────────────────────────────────────────────────────────────────────

def test_unknown_endpoint_returns_404(client):
    r = client.get("/nonexistent")
    assert r.status_code == 404
