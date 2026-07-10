"""
Unit tests for preprocessing pipeline.
Run: pytest tests/ -v
"""

from src.preprocess import (
    build_preprocessor,
    load_data,
    split_features_target,
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Small synthetic DataFrame mimicking the heart disease dataset."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "age": np.random.randint(30, 75, n).astype(float),
            "sex": np.random.randint(0, 2, n).astype(float),
            "cp": np.random.randint(0, 4, n).astype(float),
            "trestbps": np.random.randint(90, 200, n).astype(float),
            "chol": np.random.randint(150, 400, n).astype(float),
            "fbs": np.random.randint(0, 2, n).astype(float),
            "restecg": np.random.randint(0, 3, n).astype(float),
            "thalach": np.random.randint(80, 200, n).astype(float),
            "exang": np.random.randint(0, 2, n).astype(float),
            "oldpeak": np.random.uniform(0, 5, n).round(1),
            "slope": np.random.randint(0, 3, n).astype(float),
            "ca": np.random.randint(0, 4, n).astype(float),
            "thal": np.random.choice([1.0, 2.0, 3.0], n),
            "target": np.random.randint(0, 2, n),
        }
    )


@pytest.fixture
def sample_df_with_missing(sample_df):
    """DataFrame with deliberate NaN values in several columns."""
    df = sample_df.copy()
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df.loc[2, "trestbps"] = np.nan
    return df


# ── Column tests ──────────────────────────────────────────────────────────────


def test_all_features_defined():
    assert len(ALL_FEATURES) == 13
    assert TARGET_COL == "target"


def test_numeric_and_categorical_disjoint():
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))


def test_features_cover_all():
    assert set(ALL_FEATURES) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


# ── split_features_target ─────────────────────────────────────────────────────


def test_split_features_target(sample_df):
    X, y = split_features_target(sample_df)
    assert list(X.columns) == ALL_FEATURES
    assert y.name == TARGET_COL
    assert len(X) == len(y) == 50


# ── Preprocessor output shape ─────────────────────────────────────────────────


def test_preprocessor_output_shape(sample_df):
    X, _ = split_features_target(sample_df)
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape == (50, len(ALL_FEATURES))


def test_preprocessor_no_nans_after_fit(sample_df_with_missing):
    X, _ = split_features_target(sample_df_with_missing)
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert not np.isnan(out).any(), "Preprocessor should impute all NaN values"


def test_preprocessor_transform_single_row(sample_df):
    X, _ = split_features_target(sample_df)
    pre = build_preprocessor()
    pre.fit(X)
    single = X.iloc[[0]]
    out = pre.transform(single)
    assert out.shape == (1, len(ALL_FEATURES))
    assert not np.isnan(out).any()


# ── Data loading ──────────────────────────────────────────────────────────────


def test_load_data_columns(tmp_path, sample_df):
    path = tmp_path / "heart.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert TARGET_COL in df.columns
    for col in ALL_FEATURES:
        assert col in df.columns


def test_load_data_numeric_types(tmp_path, sample_df):
    path = tmp_path / "heart.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    for col in ALL_FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"


def test_load_data_target_binary(tmp_path, sample_df):
    path = tmp_path / "heart.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    assert set(df[TARGET_COL].dropna().unique()).issubset({0, 1})
