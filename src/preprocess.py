"""
Preprocessing pipeline — handles missing values, encoding, and scaling.
Saves a fitted sklearn Pipeline for reuse at inference time.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ── Column groups ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "preprocessor.joblib"
)


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,  NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    df = df[ALL_FEATURES + [TARGET_COL]].copy()
    # Replace '?' artifacts that may survive as strings
    df.replace("?", np.nan, inplace=True)
    for col in ALL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) as DataFrame and Series."""
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]
    return X, y


def fit_and_save_preprocessor(X: pd.DataFrame, path: str = PIPELINE_PATH):
    """Fit preprocessor on training data and persist it."""
    pre = build_preprocessor()
    pre.fit(X)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pre, path)
    print(f"Preprocessor saved → {path}")
    return pre


def load_preprocessor(path: str = PIPELINE_PATH):
    """Load a previously fitted preprocessor."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run training first."
        )
    return joblib.load(path)


def preprocess_input(data: dict) -> np.ndarray:
    """
    Accept a single-record dict (from API) and return a transformed
    numpy array ready for model.predict().
    """
    pre = load_preprocessor()
    df = pd.DataFrame([data])[ALL_FEATURES]
    return pre.transform(df)
