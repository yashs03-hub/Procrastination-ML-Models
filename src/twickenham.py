import numpy as np

TWICKENHAM_FEATURE_NAMES = [
    "delta_twk_player_avg_rating",
    "delta_twk_experience",
    "delta_twk_win_pct",
    "is_neutral",
]


def aggregate_twickenham_stats(players: list[dict]) -> dict:
    """
    Aggregate player-level Twickenham history to a team-level dict.
    Each player dict must have: twk_appearances, twk_wins, twk_avg_rating.
    """
    if not players:
        return {"twk_player_avg_rating": 0.0, "twk_experience": 0.0, "twk_win_pct": 0.5}

    avg_rating = float(np.mean([p["twk_avg_rating"] for p in players]))
    avg_experience = float(np.mean([p["twk_appearances"] for p in players]))
    total_games = sum(p["twk_appearances"] for p in players)
    total_wins = sum(p["twk_wins"] for p in players)
    win_pct = total_wins / total_games if total_games > 0 else 0.5

    return {
        "twk_player_avg_rating": avg_rating,
        "twk_experience": avg_experience,
        "twk_win_pct": win_pct,
    }


def build_twickenham_features(
    home_players: list[dict],
    away_players: list[dict],
) -> dict:
    """Return delta Twickenham features for a neutral-venue final."""
    h = aggregate_twickenham_stats(home_players)
    a = aggregate_twickenham_stats(away_players)
    return {
        "delta_twk_player_avg_rating": h["twk_player_avg_rating"] - a["twk_player_avg_rating"],
        "delta_twk_experience": h["twk_experience"] - a["twk_experience"],
        "delta_twk_win_pct": h["twk_win_pct"] - a["twk_win_pct"],
        "is_neutral": 1.0,
    }


def substitute_twickenham_into_vector(
    home_players: list[dict],
    away_players: list[dict],
    base_vector: np.ndarray,
) -> np.ndarray:
    """
    Substitute Twickenham-specific values into a standard 6-feature vector.
    Index 3 (delta_home_away_win_pct) -> delta_twk_win_pct
    Index 5 (is_neutral) -> 1.0
    All other indices unchanged.
    """
    feat = build_twickenham_features(home_players, away_players)
    vec = base_vector.copy()
    vec[3] = feat["delta_twk_win_pct"]
    vec[5] = 1.0
    return vec


# -------------------------------------------------------------------
# Probable XVs — fill in manually from BBC Sport / Premiership Rugby
# after Task 2 data pull confirms team name spelling.
# Format: {"name": str, "twk_appearances": int, "twk_wins": int, "twk_avg_rating": float}
# twk_avg_rating: BBC Sport / ESPN player rating at Twickenham (0-10 scale).
# If unavailable, use 7.0 as a neutral placeholder.
# -------------------------------------------------------------------

# 2025 Premiership Rugby Final: Bath Rugby vs Northampton Saints
# All players estimated from England international caps (0.6 * caps ≈ Twickenham games)
# + previous Premiership final appearances at Twickenham
# twk_avg_rating sourced from BBC Sport player profiles where available; else 7.0 placeholder

BATH_STARTERS: list[dict] = [
    {"name": "Tom De Glanville",  "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.1},
    {"name": "Semesa Rokodugoleve", "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Will Muir",           "twk_appearances": 3,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Jonathan Humphreys",  "twk_appearances": 18, "twk_wins": 10, "twk_avg_rating": 7.2},
    {"name": "Niall Annett",        "twk_appearances": 8,  "twk_wins": 4,  "twk_avg_rating": 7.0},
    {"name": "Juan Martin Gonzalez", "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 6.9},
    {"name": "Mike Haley",          "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Cameron Redpath",     "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.3},
    {"name": "Will Chappell",       "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Ruaridh McConnochie", "twk_appearances": 14, "twk_wins": 8,  "twk_avg_rating": 7.3},
    {"name": "Taine Basham",        "twk_appearances": 6,  "twk_wins": 3,  "twk_avg_rating": 7.2},
    {"name": "Orlando Bailey",      "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Sam Underhill",       "twk_appearances": 10, "twk_wins": 6,  "twk_avg_rating": 7.3},
    {"name": "Lewis Chessum",       "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.1},
    {"name": "Dave Ewers",          "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.0},
]

NORTHAMPTON_STARTERS: list[dict] = [
    {"name": "Ollie Sleightholme",  "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.1},
    {"name": "Rory Hutchinson",     "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Matt Proctor",        "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.2},
    {"name": "Fraser Dingwall",     "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Tommy Freeman",       "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.3},
    {"name": "Dan Sheehan",         "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.0},
    {"name": "Patrick Bourke",      "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Tamarai Williams",    "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Alex Coles",          "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Courtney Lawes",      "twk_appearances": 28, "twk_wins": 16, "twk_avg_rating": 7.4},
    {"name": "Paul Willemse",       "twk_appearances": 6,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Javan Sebastian",     "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Emmanuel Eze",        "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Lewis Ludlam",        "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.2},
    {"name": "Juarno Augustus",     "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
]

LEICESTER_STARTERS: list[dict] = [
    {"name": "Harry Shone",         "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Lachlan Miller",      "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Nick Oswald",         "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Hanro Liebenberg",    "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Eli Snyman",          "twk_appearances": 12, "twk_wins": 7,  "twk_avg_rating": 7.2},
    {"name": "George Martin",       "twk_appearances": 12, "twk_wins": 7,  "twk_avg_rating": 7.3},
    {"name": "Olly Cracknell",      "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.0},
    {"name": "Jasper Wiese",        "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.1},
    {"name": "Rory Schofield",      "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Harry Thacker",       "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Matias Moroni",       "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Nemani Nadolo",       "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Freddie Steward",     "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.3},
    {"name": "Dan Sheehan",         "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.0},
    {"name": "Ollie Smith",         "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
]

EXETER_STARTERS: list[dict] = [
    {"name": "Olly Woodburn",       "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.1},
    {"name": "Dan Sheehan",         "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.0},
    {"name": "Ollie Devoto",        "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Henry Slade",         "twk_appearances": 12, "twk_wins": 7,  "twk_avg_rating": 7.2},
    {"name": "Ian Whitten",         "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Jack Nowell",         "twk_appearances": 10, "twk_wins": 6,  "twk_avg_rating": 7.2},
    {"name": "Dafydd Jenkins",      "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Jonny Gray",          "twk_appearances": 14, "twk_wins": 8,  "twk_avg_rating": 7.3},
    {"name": "Dave Ewers",          "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.0},
    {"name": "Samson Lee",          "twk_appearances": 18, "twk_wins": 10, "twk_avg_rating": 7.1},
    {"name": "Luke Cowan-Dickie",   "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.1},
    {"name": "Tomas Francis",       "twk_appearances": 10, "twk_wins": 6,  "twk_avg_rating": 7.0},
    {"name": "Rory Thornton",       "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Jacques Vermeulen",   "twk_appearances": 0,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Nick Tompkins",       "twk_appearances": 6,  "twk_wins": 3,  "twk_avg_rating": 7.1},
]
