import numpy as np
import pytest
import tempfile, os
from src.model import train_model, predict_proba, evaluate_model, save_model, load_model

N_FEATURES = 6


def _dummy_data(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(n, N_FEATURES)
    y = (X[:, 0] + np.random.randn(n) * 0.5 > 0).astype(int)
    return X, y


def test_train_returns_calibrated_model():
    X, y = _dummy_data()
    model = train_model(X, y)
    assert hasattr(model, "predict_proba")


def test_predict_proba_range():
    X, y = _dummy_data()
    model = train_model(X, y)
    probs = predict_proba(model, X[:10])
    assert probs.shape == (10,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_proba_not_overconfident():
    X, y = _dummy_data()
    model = train_model(X, y)
    probs = predict_proba(model, X)
    assert (probs > 0.95).sum() < len(probs) * 0.1


def test_evaluate_returns_expected_keys():
    X, y = _dummy_data()
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)
    assert "cv_accuracy" in metrics
    assert "cv_log_loss" in metrics


def test_save_load_roundtrip():
    X, y = _dummy_data()
    model = train_model(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        save_model(model, path)
        loaded = load_model(path)
    np.testing.assert_array_almost_equal(
        predict_proba(model, X[:5]),
        predict_proba(loaded, X[:5]),
    )
