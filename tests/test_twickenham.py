import pytest
import numpy as np
from src.twickenham import (
    aggregate_twickenham_stats,
    build_twickenham_features,
    substitute_twickenham_into_vector,
)

MOCK_PLAYERS_A = [
    {"name": "Russell",  "twk_appearances": 12, "twk_wins": 8,  "twk_avg_rating": 7.5},
    {"name": "Spencer",  "twk_appearances": 6,  "twk_wins": 3,  "twk_avg_rating": 7.0},
]
MOCK_PLAYERS_B = [
    {"name": "Curry",    "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.2},
]


def test_aggregate_keys():
    stats = aggregate_twickenham_stats(MOCK_PLAYERS_A)
    assert set(stats.keys()) == {"twk_player_avg_rating", "twk_experience", "twk_win_pct"}


def test_aggregate_values():
    stats = aggregate_twickenham_stats(MOCK_PLAYERS_A)
    assert stats["twk_player_avg_rating"] == pytest.approx(7.25)
    assert stats["twk_experience"] == pytest.approx(9.0)
    assert stats["twk_win_pct"] == pytest.approx(11 / 18)


def test_aggregate_empty_returns_defaults():
    stats = aggregate_twickenham_stats([])
    assert stats["twk_win_pct"] == 0.5
    assert stats["twk_experience"] == 0.0


def test_build_twickenham_features_delta():
    feat = build_twickenham_features(MOCK_PLAYERS_A, MOCK_PLAYERS_B)
    assert "delta_twk_player_avg_rating" in feat
    assert "delta_twk_experience" in feat
    assert "delta_twk_win_pct" in feat
    assert feat["is_neutral"] == 1.0


def test_substitute_replaces_index_3_and_5():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    result = substitute_twickenham_into_vector(MOCK_PLAYERS_A, MOCK_PLAYERS_B, base)
    assert result[5] == 1.0          # is_neutral replaced
    assert result[3] != 4.0          # delta_home_away_win_pct replaced
    assert result[0] == 1.0          # unchanged
