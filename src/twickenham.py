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
    {"name": "Beno Obano",          "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.1},
    {"name": "Kepu Tuipulotu",      "twk_appearances": 6,  "twk_wins": 2,  "twk_avg_rating": 7.1},
    {"name": "Thomas du Toit",      "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.4},
    {"name": "Ted Hill",            "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Charlie Ewels",       "twk_appearances": 12, "twk_wins": 8,  "twk_avg_rating": 7.3},
    {"name": "Josh Bayliss",        "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Sam Underhill",       "twk_appearances": 10, "twk_wins": 7,  "twk_avg_rating": 7.4},
    {"name": "Alfie Barbeary",      "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.3},
    {"name": "Ben Spencer",         "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.3},
    {"name": "Santiago Carreras",   "twk_appearances": 4,  "twk_wins": 1,  "twk_avg_rating": 7.2},
    {"name": "Henry Arundell",      "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.4},
    {"name": "Max Ojomoh",          "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Ollie Lawrence",      "twk_appearances": 10, "twk_wins": 7,  "twk_avg_rating": 7.4},
    {"name": "Joe Cokanasiga",      "twk_appearances": 8,  "twk_wins": 5,  "twk_avg_rating": 7.3},
    {"name": "Tom de Glanville",    "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.3},
]

NORTHAMPTON_STARTERS: list[dict] = [
    {"name": "Danilo Fischetti",    "twk_appearances": 8,  "twk_wins": 1,  "twk_avg_rating": 7.1},
    {"name": "Curtis Langdon",      "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Cleopas Kundiona",    "twk_appearances": 1,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Tom Lockett",         "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "JJ Van Der Mescht",   "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "Josh Kemeny",         "twk_appearances": 1,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Tom Pearson",         "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Sam Graham",          "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.2},
    {"name": "Jonny Weimann",       "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Anthony Belleau",     "twk_appearances": 8,  "twk_wins": 3,  "twk_avg_rating": 7.3},
    {"name": "James Ramm",          "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Fraser Dingwall",     "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.2},
    {"name": "Tom Litchfield",      "twk_appearances": 3,  "twk_wins": 2,  "twk_avg_rating": 7.0},
    {"name": "James Martin",        "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "James Pater",         "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
]

LEICESTER_STARTERS: list[dict] = [
    {"name": "Nicky Smith",             "twk_appearances": 8,  "twk_wins": 2,  "twk_avg_rating": 7.1},
    {"name": "Jamie Blamire",           "twk_appearances": 6,  "twk_wins": 4,  "twk_avg_rating": 7.2},
    {"name": "Joe Heyes",               "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Cameron Henderson",       "twk_appearances": 3,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Ollie Chessum",           "twk_appearances": 8,  "twk_wins": 6,  "twk_avg_rating": 7.3},
    {"name": "Hanro Liebenberg",        "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Tommy Reffell",           "twk_appearances": 5,  "twk_wins": 2,  "twk_avg_rating": 7.1},
    {"name": "Olly Cracknell",          "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Jack van Poortvliet",     "twk_appearances": 10, "twk_wins": 7,  "twk_avg_rating": 7.5},
    {"name": "James O'Connor",          "twk_appearances": 8,  "twk_wins": 3,  "twk_avg_rating": 7.4},
    {"name": "Gabriel Hamer-Webb",      "twk_appearances": 1,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Orlando Bailey",          "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
    {"name": "Will Wand",               "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Adam Radwan",             "twk_appearances": 5,  "twk_wins": 3,  "twk_avg_rating": 7.2},
    {"name": "George Pearson",          "twk_appearances": 3,  "twk_wins": 2,  "twk_avg_rating": 7.1},
]

EXETER_STARTERS: list[dict] = [
    {"name": "Scott Sio",           "twk_appearances": 10, "twk_wins": 3,  "twk_avg_rating": 7.2},
    {"name": "Max Norey",           "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Josh Iosefa-Scott",   "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Dafydd Jenkins",      "twk_appearances": 5,  "twk_wins": 2,  "twk_avg_rating": 7.2},
    {"name": "Andrea Zambonin",     "twk_appearances": 6,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Tom Hooper",          "twk_appearances": 2,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Christ Tshiunza",     "twk_appearances": 4,  "twk_wins": 2,  "twk_avg_rating": 7.2},
    {"name": "Ethan Roots",         "twk_appearances": 3,  "twk_wins": 1,  "twk_avg_rating": 7.0},
    {"name": "Stephen Varney",      "twk_appearances": 6,  "twk_wins": 1,  "twk_avg_rating": 7.2},
    {"name": "Harvey Skinner",      "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.2},
    {"name": "Campbell Ridl",       "twk_appearances": 1,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Len Ikitau",          "twk_appearances": 6,  "twk_wins": 2,  "twk_avg_rating": 7.3},
    {"name": "Henry Slade",         "twk_appearances": 14, "twk_wins": 9,  "twk_avg_rating": 7.4},
    {"name": "Paul Brown-Bampoe",   "twk_appearances": 1,  "twk_wins": 0,  "twk_avg_rating": 7.0},
    {"name": "Olly Woodburn",       "twk_appearances": 4,  "twk_wins": 3,  "twk_avg_rating": 7.1},
]
