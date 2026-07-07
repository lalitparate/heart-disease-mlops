"""
Download the Heart Disease UCI Dataset from UCI ML Repository.
Saves to data/heart.csv in the project root.
"""

import os
import urllib.request
import pandas as pd

# UCI Heart Disease dataset (Cleveland) — direct URL
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")


def download():
    print(f"Downloading Heart Disease UCI dataset from:\n  {UCI_URL}\n")
    try:
        urllib.request.urlretrieve(UCI_URL, "raw_cleveland.data")
        df = pd.read_csv(
            "raw_cleveland.data",
            header=None,
            names=COLUMN_NAMES,
            na_values="?"
        )
        # Binarise target: 0 = no disease, 1 = disease (values 1-4 → 1)
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(OUTPUT_PATH, index=False)
        os.remove("raw_cleveland.data")
        print(f"Dataset saved to: {OUTPUT_PATH}")
        print(f"Shape: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts().to_string()}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Falling back to local synthetic sample for CI/testing...")
        _generate_sample()


def _generate_sample():
    """Generate a small synthetic dataset for CI smoke tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 303
    df = pd.DataFrame({
        "age":      rng.integers(29, 77, n).astype(float),
        "sex":      rng.integers(0, 2, n).astype(float),
        "cp":       rng.integers(0, 4, n).astype(float),
        "trestbps": rng.integers(94, 200, n).astype(float),
        "chol":     rng.integers(126, 564, n).astype(float),
        "fbs":      rng.integers(0, 2, n).astype(float),
        "restecg":  rng.integers(0, 3, n).astype(float),
        "thalach":  rng.integers(71, 202, n).astype(float),
        "exang":    rng.integers(0, 2, n).astype(float),
        "oldpeak":  rng.uniform(0, 6.2, n).round(1),
        "slope":    rng.integers(0, 3, n).astype(float),
        "ca":       rng.integers(0, 4, n).astype(float),
        "thal":     rng.choice([1.0, 2.0, 3.0], n),
        "target":   rng.integers(0, 2, n),
    })
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic sample saved to: {OUTPUT_PATH}  (shape: {df.shape})")


if __name__ == "__main__":
    download()
