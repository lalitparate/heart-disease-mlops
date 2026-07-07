"""
Inference utilities — load model + preprocessor and run predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.preprocess import load_preprocessor, ALL_FEATURES

BEST_MODEL_PATH = os.path.join("models", "best_model.joblib")


def load_model(path: str = BEST_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run 'python src/train.py' first."
        )
    return joblib.load(path)


def predict_single(data: dict) -> dict:
    """
    data : dict with keys matching ALL_FEATURES
    Returns {"prediction": int, "probability": float, "label": str}
    """
    pre   = load_preprocessor()
    model = load_model()

    df = pd.DataFrame([data])[ALL_FEATURES]
    X  = pre.transform(df)

    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    return {"prediction": pred, "probability": round(proba, 4), "label": label}


def predict_batch(records: list[dict]) -> list[dict]:
    """
    records : list of feature dicts
    Returns list of prediction dicts
    """
    pre   = load_preprocessor()
    model = load_model()

    df = pd.DataFrame(records)[ALL_FEATURES]
    X  = pre.transform(df)

    preds  = model.predict(X).tolist()
    probas = model.predict_proba(X)[:, 1].tolist()

    return [
        {
            "prediction":  int(p),
            "probability": round(float(pr), 4),
            "label":       "Heart Disease Detected" if p == 1 else "No Heart Disease",
        }
        for p, pr in zip(preds, probas)
    ]
