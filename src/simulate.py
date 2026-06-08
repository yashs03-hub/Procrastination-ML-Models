import numpy as np
from dataclasses import dataclass


@dataclass
class TournamentTeams:
    semi1_home: str
    semi1_away: str
    semi2_home: str
    semi2_away: str


def simulate_match(p_home: float) -> str:
    """Return 'home' or 'away' based on P(home wins)."""
    return "home" if np.random.random() < p_home else "away"


def simulate_scoreline(
    home_pts_mean: float,
    away_pts_mean: float,
    outcome: str,
    std: float = 8.0,
) -> tuple[int, int]:
    """
    Sample a scoreline conditioned on the declared outcome.
    Resamples until the scoreline is consistent (winner has more points).
    """
    while True:
        home_pts = max(0, int(np.random.normal(home_pts_mean, std)))
        away_pts = max(0, int(np.random.normal(away_pts_mean, std)))
        if outcome == "home" and home_pts > away_pts:
            return home_pts, away_pts
        if outcome == "away" and away_pts > home_pts:
            return home_pts, away_pts


def run_tournament(
    teams: TournamentTeams,
    match_probs: dict[tuple[str, str], float],
    n: int = 10_000,
) -> dict[str, float]:
    """
    Run n Monte Carlo simulations of both semis + all possible finals.

    match_probs keys are (home_or_first_team, away_or_second_team).
    P(first team wins) is the value. For the neutral final, the first-listed
    team is used as nominal 'home' when looking up the probability.

    Returns championship win fraction per team (sums to 1.0).
    """
    wins: dict[str, int] = {
        teams.semi1_home: 0,
        teams.semi1_away: 0,
        teams.semi2_home: 0,
        teams.semi2_away: 0,
    }

    for _ in range(n):
        # Semi 1
        p1 = match_probs[(teams.semi1_home, teams.semi1_away)]
        s1_winner = teams.semi1_home if simulate_match(p1) == "home" else teams.semi1_away

        # Semi 2
        p2 = match_probs[(teams.semi2_home, teams.semi2_away)]
        s2_winner = teams.semi2_home if simulate_match(p2) == "home" else teams.semi2_away

        # Final — always look up (s1_winner, s2_winner) first
        final_key = (s1_winner, s2_winner)
        if final_key in match_probs:
            p_final = match_probs[final_key]
            champion = s1_winner if simulate_match(p_final) == "home" else s2_winner
        else:
            rev_key = (s2_winner, s1_winner)
            p_final = match_probs.get(rev_key, 0.5)
            champion = s2_winner if simulate_match(p_final) == "home" else s1_winner

        wins[champion] += 1

    return {team: count / n for team, count in wins.items()}
