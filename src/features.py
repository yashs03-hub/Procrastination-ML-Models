import numpy as np
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

FEATURE_NAMES = [
    "delta_pts_scored_pg",
    "delta_pts_conceded_pg",
    "delta_try_bonus_rate",
    "delta_home_away_win_pct",
    "delta_last5_form",
    "is_neutral",
]


def compute_delta_features(home: dict, away: dict) -> dict:
    """Compute home-minus-away deltas for each stat."""
    return {
        "delta_pts_scored_pg": home["pts_scored_pg"] - away["pts_scored_pg"],
        "delta_pts_conceded_pg": home["pts_conceded_pg"] - away["pts_conceded_pg"],
        "delta_try_bonus_rate": home["try_bonus_rate"] - away["try_bonus_rate"],
        "delta_home_away_win_pct": home["home_win_pct"] - away["away_win_pct"],
        "delta_last5_form": home["last5_form"] - away["last5_form"],
    }


def build_match_features(home: dict, away: dict, is_neutral: bool = False) -> np.ndarray:
    """Return a 6-element feature vector for one matchup."""
    d = compute_delta_features(home, away)
    return np.array([
        d["delta_pts_scored_pg"],
        d["delta_pts_conceded_pg"],
        d["delta_try_bonus_rate"],
        d["delta_home_away_win_pct"],
        d["delta_last5_form"],
        float(is_neutral),
    ])


def build_training_matrix(
    matches: pd.DataFrame,
    team_stats: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for XGBoost from a matches DataFrame.
    Rows where either team has no entry in team_stats are silently dropped.
    """
    X_rows, y_rows = [], []
    for _, row in matches.iterrows():
        home_stats = team_stats.get(row["home_team"])
        away_stats = team_stats.get(row["away_team"])
        if home_stats is None or away_stats is None:
            continue
        X_rows.append(build_match_features(home_stats, away_stats, is_neutral=False))
        y_rows.append(int(row["home_win"]))
    return np.array(X_rows), np.array(y_rows)


def save_training_matrix(X: np.ndarray, y: np.ndarray) -> None:
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["home_win"] = y
    df.to_csv(DATA_PROCESSED / "features_train.csv", index=False)
    print(f"Saved training matrix: {X.shape[0]} rows x {X.shape[1]} features")
