import numpy as np
import pytest
from src.simulate import (
    simulate_match,
    simulate_scoreline,
    run_tournament,
    TournamentTeams,
)

TEAMS = TournamentTeams(
    semi1_home="Bath Rugby",
    semi1_away="Exeter Chiefs",
    semi2_home="Leicester Tigers",
    semi2_away="Northampton Saints",
)

EVEN_PROBS = {
    ("Bath Rugby",       "Exeter Chiefs"):        0.5,
    ("Leicester Tigers", "Northampton Saints"):   0.5,
    ("Bath Rugby",       "Leicester Tigers"):     0.5,
    ("Bath Rugby",       "Northampton Saints"):   0.5,
    ("Exeter Chiefs",    "Leicester Tigers"):     0.5,
    ("Exeter Chiefs",    "Northampton Saints"):   0.5,
}

BATH_DOMINANT = {
    ("Bath Rugby",       "Exeter Chiefs"):        0.95,
    ("Leicester Tigers", "Northampton Saints"):   0.95,
    ("Bath Rugby",       "Leicester Tigers"):     0.95,
    ("Bath Rugby",       "Northampton Saints"):   0.95,
    ("Exeter Chiefs",    "Leicester Tigers"):     0.5,
    ("Exeter Chiefs",    "Northampton Saints"):   0.5,
}


def test_simulate_match_respects_probability():
    np.random.seed(0)
    wins = sum(simulate_match(0.9) == "home" for _ in range(2000))
    assert 1700 < wins < 1950


def test_simulate_match_valid_output():
    assert simulate_match(0.0) == "away"
    assert simulate_match(1.0) == "home"


def test_simulate_scoreline_consistent_with_outcome():
    np.random.seed(0)
    home_pts, away_pts = simulate_scoreline(30.0, 20.0, "home")
    assert home_pts > away_pts

    home_pts, away_pts = simulate_scoreline(20.0, 30.0, "away")
    assert away_pts > home_pts


def test_run_tournament_sums_to_one():
    result = run_tournament(TEAMS, EVEN_PROBS, n=2000)
    assert abs(sum(result.values()) - 1.0) < 0.01


def test_run_tournament_all_teams_present():
    result = run_tournament(TEAMS, EVEN_PROBS, n=500)
    assert set(result.keys()) == {
        "Bath Rugby", "Exeter Chiefs", "Leicester Tigers", "Northampton Saints"
    }


def test_run_tournament_dominant_team_wins_more():
    result = run_tournament(TEAMS, BATH_DOMINANT, n=3000)
    assert result["Bath Rugby"] > 0.8
