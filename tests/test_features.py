import numpy as np
import pandas as pd
import pytest
from src.features import (
    compute_delta_features,
    build_match_features,
    build_training_matrix,
    FEATURE_NAMES,
)


def _stats(pts_scored=30.0, pts_conceded=20.0, try_bonus=0.4,
           home_win_pct=0.7, away_win_pct=0.4, last5_form=8.0):
    return {
        "pts_scored_pg": pts_scored,
        "pts_conceded_pg": pts_conceded,
        "try_bonus_rate": try_bonus,
        "home_win_pct": home_win_pct,
        "away_win_pct": away_win_pct,
        "last5_form": last5_form,
    }


def test_delta_sign():
    deltas = compute_delta_features(_stats(pts_scored=35.0), _stats(pts_scored=25.0))
    assert deltas["delta_pts_scored_pg"] == pytest.approx(10.0)


def test_delta_keys():
    deltas = compute_delta_features(_stats(), _stats())
    assert set(deltas.keys()) == {
        "delta_pts_scored_pg", "delta_pts_conceded_pg", "delta_try_bonus_rate",
        "delta_home_away_win_pct", "delta_last5_form",
    }


def test_feature_vector_shape():
    arr = build_match_features(_stats(), _stats(), is_neutral=False)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(FEATURE_NAMES),)


def test_feature_vector_neutral_flag():
    arr_neutral = build_match_features(_stats(), _stats(), is_neutral=True)
    arr_home = build_match_features(_stats(), _stats(), is_neutral=False)
    assert arr_neutral[-1] == 1.0
    assert arr_home[-1] == 0.0


def test_training_matrix_shape():
    matches = pd.DataFrame({
        "home_team": ["Bath Rugby", "Exeter Chiefs"],
        "away_team": ["Exeter Chiefs", "Bath Rugby"],
        "home_win": [1, 0],
    })
    stats = {
        "Bath Rugby": _stats(),
        "Exeter Chiefs": _stats(pts_scored=28.0),
    }
    X, y = build_training_matrix(matches, stats)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert y.tolist() == [1, 0]


def test_training_matrix_drops_unknown_teams():
    matches = pd.DataFrame({
        "home_team": ["Bath Rugby", "Unknown FC"],
        "away_team": ["Exeter Chiefs", "Bath Rugby"],
        "home_win": [1, 0],
    })
    stats = {"Bath Rugby": _stats(), "Exeter Chiefs": _stats()}
    X, y = build_training_matrix(matches, stats)
    assert X.shape[0] == 1
