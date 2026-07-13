"""Shared fixtures: synthetic market data so every test runs offline."""

import os
import sys

# Make the project root importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import DataPreprocessor


def make_ohlcv(n=420, seed=42, drift=0.0005):
    """Deterministic synthetic OHLCV random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    log_ret = rng.normal(drift, 0.02, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    high = close * (1 + rng.uniform(0.0, 0.02, n))
    low = close * (1 - rng.uniform(0.0, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture(scope="session")
def ohlcv():
    return make_ohlcv()


@pytest.fixture(scope="session")
def feature_data(ohlcv):
    """(processor, feature_df, X, y) for classification."""
    proc = DataPreprocessor(ohlcv)
    feature_df = proc.add_technical_indicators()
    X, y = proc.prepare_data_for_training()
    return proc, feature_df, X, y


@pytest.fixture(scope="session")
def regression_data(ohlcv):
    """(X, y) for regression."""
    proc = DataPreprocessor(ohlcv)
    proc.add_technical_indicators()
    return proc.prepare_data_for_regression()


def redirect_paths(model, folder, name):
    """Point a model's persistence path at a temp folder so tests never
    touch the real artifacts in models/."""
    model.path = os.path.join(str(folder), f"{name}.pkl")
    if hasattr(model, "config_path"):
        model.config_path = os.path.join(str(folder), f"{name}_config.ini")
    return model
