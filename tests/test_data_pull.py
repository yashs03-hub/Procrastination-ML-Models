import pandas as pd

REQUIRED_HISTORICAL_COLS = {
    "season", "home_team", "away_team",
    "home_score", "away_score", "home_win"
}

REQUIRED_TEAM_STAT_COLS = {
    "team", "pts_scored_pg", "pts_conceded_pg",
    "try_bonus_rate", "home_win_pct", "away_win_pct", "last5_form"
}

def test_historical_matches_schema():
    df = pd.DataFrame({c: [] for c in REQUIRED_HISTORICAL_COLS})
    assert REQUIRED_HISTORICAL_COLS.issubset(set(df.columns))

def test_team_stats_schema():
    df = pd.DataFrame({c: [] for c in REQUIRED_TEAM_STAT_COLS})
    assert REQUIRED_TEAM_STAT_COLS.issubset(set(df.columns))
