"""
Unit tests for model training and inference.
Run: pytest tests/ -v
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.evaluate import compute_metrics
from src.preprocess import build_preprocessor, ALL_FEATURES
import os
import sys
import pytest
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    np.random.seed(0)
    n = 100
    df = pd.DataFrame({
        col: np.random.rand(n) for col in ALL_FEATURES
    })
    y = np.random.randint(0, 2, n)
    return df, y


@pytest.fixture
def fitted_preprocessor_and_data(synthetic_data):
    X, y = synthetic_data
    pre  = build_preprocessor()
    X_t  = pre.fit_transform(X)
    return pre, X_t, y


@pytest.fixture
def trained_lr(fitted_preprocessor_and_data):
    _, X_t, y = fitted_preprocessor_and_data
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_t, y)
    return model


@pytest.fixture
def trained_rf(fitted_preprocessor_and_data):
    _, X_t, y = fitted_preprocessor_and_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_t, y)
    return model


# ── Model basics ──────────────────────────────────────────────────────────────

def test_lr_predict_shape(trained_lr, fitted_preprocessor_and_data):
    _, X_t, _ = fitted_preprocessor_and_data
    preds = trained_lr.predict(X_t)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1})


def test_rf_predict_shape(trained_rf, fitted_preprocessor_and_data):
    _, X_t, _ = fitted_preprocessor_and_data
    preds = trained_rf.predict(X_t)
    assert preds.shape == (100,)
    assert set(preds).issubset({0, 1})


def test_lr_predict_proba_shape(trained_lr, fitted_preprocessor_and_data):
    _, X_t, _ = fitted_preprocessor_and_data
    probas = trained_lr.predict_proba(X_t)
    assert probas.shape == (100, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_rf_predict_proba_shape(trained_rf, fitted_preprocessor_and_data):
    _, X_t, _ = fitted_preprocessor_and_data
    probas = trained_rf.predict_proba(X_t)
    assert probas.shape == (100, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_compute_metrics_keys(trained_lr, fitted_preprocessor_and_data):
    _, X_t, y = fitted_preprocessor_and_data
    y_pred  = trained_lr.predict(X_t)
    y_proba = trained_lr.predict_proba(X_t)[:, 1]
    metrics = compute_metrics(y, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics


def test_metrics_range(trained_rf, fitted_preprocessor_and_data):
    _, X_t, y = fitted_preprocessor_and_data
    y_pred  = trained_rf.predict(X_t)
    y_proba = trained_rf.predict_proba(X_t)[:, 1]
    metrics = compute_metrics(y, y_pred, y_proba)
    for val in metrics.values():
        assert 0.0 <= val <= 1.0


# ── Model serialisation ───────────────────────────────────────────────────────

def test_model_save_load(tmp_path, trained_rf):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_rf, path)
    loaded = joblib.load(path)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "predict_proba")


def test_saved_model_predicts_same(tmp_path, trained_rf, fitted_preprocessor_and_data):
    _, X_t, _ = fitted_preprocessor_and_data
    path = tmp_path / "model.joblib"
    joblib.dump(trained_rf, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(
        trained_rf.predict(X_t),
        loaded.predict(X_t)
    )


# ── Single-row inference ──────────────────────────────────────────────────────

def test_single_row_inference(trained_lr, fitted_preprocessor_and_data):
    pre, X_t, _ = fitted_preprocessor_and_data
    single = X_t[[0]]
    pred   = trained_lr.predict(single)
    proba  = trained_lr.predict_proba(single)
    assert pred.shape  == (1,)
    assert proba.shape == (1, 2)
    assert pred[0] in {0, 1}
