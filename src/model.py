import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODELS_DIR = Path("models")


def train_model(X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    """Train XGBoost with Platt (sigmoid) calibration via 5-fold CV."""
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(xgb, method="sigmoid", cv=5)
    calibrated.fit(X, y)
    return calibrated


def evaluate_model(
    model: CalibratedClassifierCV, X: np.ndarray, y: np.ndarray
) -> dict:
    """Return mean 5-fold CV accuracy and log-loss."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    ll = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss").mean()
    return {"cv_accuracy": round(float(acc), 3), "cv_log_loss": round(float(-ll), 3)}


def predict_proba(model: CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    """Return P(home_win) for each row in X."""
    return model.predict_proba(X)[:, 1]


def save_model(model: CalibratedClassifierCV, path: str | None = None) -> str:
    path = path or str(MODELS_DIR / "xgb_calibrated.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | None = None) -> CalibratedClassifierCV:
    path = path or str(MODELS_DIR / "xgb_calibrated.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
