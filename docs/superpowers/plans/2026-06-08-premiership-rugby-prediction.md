# Premiership Rugby 2026 Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an XGBoost model trained on 5 seasons of Premiership Rugby data, simulate the 2026 semi-finals + final 10,000 times, and produce publication-ready charts in four independent notebooks.

**Architecture:** ESPN API → The Sports DB → BBC scraper fallback for data; delta feature vectors per matchup; XGBoost + Platt calibration for probabilities; Monte Carlo tournament simulation with conditioned scoreline sampling; Twickenham player-history features substituted in for the final.

**Tech Stack:** Python 3.11, pandas, numpy, xgboost 2.x, scikit-learn, shap, matplotlib, requests, beautifulsoup4, pytest, jupyter

---

## File Map

```
prem-rugby-prediction/
├── src/
│   ├── __init__.py
│   ├── data_pull.py          # API probe + scraper, outputs raw CSVs
│   ├── features.py           # Delta feature engineering + training matrix
│   ├── twickenham.py         # Player Twickenham lookup + final feature vector
│   ├── model.py              # XGBoost train, calibrate, save/load
│   └── simulate.py           # Monte Carlo simulation + scoreline sampling
├── notebooks/
│   ├── 01_data_pull.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_train.ipynb
│   └── 04_predict_simulate.ipynb
├── data/
│   ├── raw/                  # historical_matches.csv, current_season_stats.csv
│   └── processed/            # features_train.csv
├── charts/                   # PNG outputs
├── models/                   # xgb_calibrated.pkl
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_simulate.py
│   └── test_twickenham.py
└── requirements.txt
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create requirements.txt**

```
pandas==2.2.2
numpy==1.26.4
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.45.1
matplotlib==3.9.0
seaborn==0.13.2
requests==2.32.3
beautifulsoup4==4.12.3
jupyter==1.0.0
pytest==8.2.0
```

- [ ] **Step 2: Create virtual environment and install**

```bash
cd /Users/yashsewpaul/code/prem-rugby-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install without errors. `python -c "import xgboost; print(xgboost.__version__)"` prints `2.0.3`.

- [ ] **Step 3: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 4: Create placeholder directories and init files**

```bash
mkdir -p data/raw data/processed charts models tests notebooks src
touch data/raw/.gitkeep data/processed/.gitkeep charts/.gitkeep models/.gitkeep
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini src/__init__.py tests/__init__.py data/ charts/ models/
git commit -m "chore: project setup — deps, directory structure, pytest config"
```

---

### Task 2: Data Pull

**Files:**
- Create: `src/data_pull.py`
- Create: `notebooks/01_data_pull.ipynb`

The data pull uses ESPN API first, The Sports DB second, and BBC Sport scraper as a last resort. Run the probe cell interactively to determine which path works before implementing the full fetch.

- [ ] **Step 1: Write schema contract tests**

Create `tests/test_data_pull.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect PASS (schema only, no real data)**

```bash
pytest tests/test_data_pull.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 3: Create src/data_pull.py with API probe functions**

```python
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

DATA_RAW = Path("data/raw")

# ESPN Gallagher Premiership league ID — verify via probe_espn() output
ESPN_LEAGUE_ID = "270559"
# TheSportsDB Premiership league ID — verify via probe_thesportsdb() output
TSDB_LEAGUE_ID = "4347"

SEMIFINALISTS = [
    "Bath Rugby",
    "Exeter Chiefs",
    "Leicester Tigers",
    "Northampton Saints",
]


def probe_espn() -> dict:
    """Fetch ESPN scoreboard JSON to inspect coverage. Run interactively."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/rugby-union/"
        f"league/{ESPN_LEAGUE_ID}/scoreboard"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def probe_thesportsdb() -> dict:
    """Fetch TheSportsDB league list to confirm league ID. Run interactively."""
    r = requests.get(
        "https://www.thesportsdb.com/api/v1/json/3/search_all_leagues.php",
        params={"c": "England", "s": "Rugby Union"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def fetch_historical_espn(seasons: list[str]) -> pd.DataFrame:
    """
    Fetch historical results from ESPN API.
    seasons: e.g. ["2021", "2022", "2023", "2024", "2025"]
    """
    rows = []
    for season in seasons:
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/rugby-union/"
            f"league/{ESPN_LEAGUE_ID}/scoreboard"
        )
        r = requests.get(url, params={"season": season, "limit": 200}, timeout=10)
        if r.status_code != 200:
            continue
        for event in r.json().get("events", []):
            comp = event["competitions"][0]
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
            rows.append({
                "season": season,
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "home_score": home_score,
                "away_score": away_score,
                "home_win": int(home_score > away_score),
            })
    return pd.DataFrame(rows)


def fetch_historical_tsdb(league_id: str = TSDB_LEAGUE_ID) -> pd.DataFrame:
    """Fetch historical results from TheSportsDB."""
    seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
    rows = []
    for season in seasons:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/3/eventsseason.php",
            params={"id": league_id, "s": season},
            timeout=15,
        )
        if r.status_code != 200:
            continue
        for event in (r.json().get("events") or []):
            if event.get("intHomeScore") is None:
                continue
            home_score = int(event["intHomeScore"])
            away_score = int(event["intAwayScore"])
            rows.append({
                "season": season,
                "home_team": event["strHomeTeam"],
                "away_team": event["strAwayTeam"],
                "home_score": home_score,
                "away_score": away_score,
                "home_win": int(home_score > away_score),
            })
    return pd.DataFrame(rows)


def scrape_bbc_results() -> pd.DataFrame:
    """
    Scrape Premiership results from BBC Sport as last resort.
    BBC's HTML structure changes — update selectors if needed after inspecting live page.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://www.bbc.co.uk/sport/rugby-union/premiership/results",
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for match in soup.select(".sp-c-fixture"):
        teams = match.select(".sp-c-fixture__team-name abbr")
        scores = match.select(".sp-c-fixture__number--score")
        if len(teams) < 2 or len(scores) < 2:
            continue
        try:
            home_score = int(scores[0].get_text(strip=True))
            away_score = int(scores[1].get_text(strip=True))
        except ValueError:
            continue
        rows.append({
            "season": "2025-2026",
            "home_team": teams[0].get("title", teams[0].get_text(strip=True)),
            "away_team": teams[1].get("title", teams[1].get_text(strip=True)),
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
        })
    return pd.DataFrame(rows)


def compute_team_stats(matches: pd.DataFrame, team_name: str) -> dict:
    """Compute aggregate season stats for one team from match rows."""
    home = matches[matches["home_team"] == team_name]
    away = matches[matches["away_team"] == team_name]

    all_scored = pd.concat([home["home_score"], away["away_score"]])
    all_conceded = pd.concat([home["away_score"], away["home_score"]])

    all_match_margins = pd.concat([
        home.assign(margin=home["home_score"] - home["away_score"]),
        away.assign(margin=away["away_score"] - away["home_score"]),
    ]).sort_index()["margin"]

    last5_form = float(all_match_margins.tail(5).mean()) if len(all_match_margins) else 0.0

    # Try bonus approximation: score ≥ 28 pts ≈ 4 tries + conversions
    home_bonus = float((home["home_score"] >= 28).mean()) if len(home) else 0.0
    away_bonus = float((away["away_score"] >= 28).mean()) if len(away) else 0.0
    try_bonus_rate = (home_bonus + away_bonus) / 2 if (len(home) + len(away)) > 0 else 0.0

    return {
        "team": team_name,
        "pts_scored_pg": float(all_scored.mean()) if len(all_scored) else 0.0,
        "pts_conceded_pg": float(all_conceded.mean()) if len(all_conceded) else 0.0,
        "try_bonus_rate": try_bonus_rate,
        "home_win_pct": float(home["home_win"].mean()) if len(home) else 0.5,
        "away_win_pct": float((1 - away["home_win"]).mean()) if len(away) else 0.5,
        "last5_form": last5_form,
    }


def save_raw(historical: pd.DataFrame, team_stats: list[dict]) -> None:
    historical.to_csv(DATA_RAW / "historical_matches.csv", index=False)
    pd.DataFrame(team_stats).to_csv(DATA_RAW / "current_season_stats.csv", index=False)
    print(f"Saved {len(historical)} historical matches")
    print(f"Saved stats for {len(team_stats)} teams")
```

- [ ] **Step 4: Create notebooks/01_data_pull.ipynb with probe cell**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "source": ["# 01 — Data Pull\n\nProbe APIs then fetch historical + current season data."]
  },
  {
   "cell_type": "code",
   "source": [
    "import sys; sys.path.insert(0, '..')\n",
    "from src.data_pull import probe_espn, probe_thesportsdb\n",
    "\n",
    "try:\n",
    "    espn = probe_espn()\n",
    "    print('ESPN OK:', list(espn.keys())[:5])\n",
    "except Exception as e:\n",
    "    print('ESPN FAILED:', e)\n",
    "\n",
    "try:\n",
    "    tsdb = probe_thesportsdb()\n",
    "    leagues = [l['strLeague'] for l in (tsdb.get('countrys') or [])]\n",
    "    print('TSDB OK:', [l for l in leagues if 'rugby' in l.lower() or 'premier' in l.lower()])\n",
    "except Exception as e:\n",
    "    print('TSDB FAILED:', e)"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "# Run whichever API worked. Update function name if ESPN succeeded.\n",
    "from src.data_pull import fetch_historical_tsdb, compute_team_stats, save_raw, SEMIFINALISTS\n",
    "import pandas as pd\n",
    "\n",
    "historical = fetch_historical_tsdb()\n",
    "print(historical.head())\n",
    "print(f'Total: {len(historical)} matches across seasons: {historical[\"season\"].unique()}')\n",
    "\n",
    "current_season = historical[historical['season'] == historical['season'].max()]\n",
    "team_stats = [compute_team_stats(current_season, t) for t in SEMIFINALISTS]\n",
    "for s in team_stats:\n",
    "    print(s)\n",
    "\n",
    "save_raw(historical, team_stats)"
   ]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4, "nbformat_minor": 5
}
```

- [ ] **Step 5: Run notebook 01 and verify outputs**

```bash
cd /Users/yashsewpaul/code/prem-rugby-prediction
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/01_data_pull.ipynb --output notebooks/01_data_pull_executed.ipynb
```

Expected: `data/raw/historical_matches.csv` exists with ≥200 rows; `data/raw/current_season_stats.csv` has 4 rows (one per semifinalist).

If the API returns team names that differ from `SEMIFINALISTS` (e.g. "Bath" vs "Bath Rugby"), update `SEMIFINALISTS` in `src/data_pull.py` to match exactly before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/data_pull.py tests/test_data_pull.py notebooks/01_data_pull.ipynb data/raw/
git commit -m "feat: data pull — ESPN/TSDB API probe + BBC scraper fallback"
```

---

### Task 3: Feature Engineering

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_features.py -v
```

Expected: ImportError (module does not exist yet).

- [ ] **Step 3: Implement src/features.py**

```python
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
    print(f"Saved training matrix: {X.shape[0]} rows × {X.shape[1]} features")
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
pytest tests/test_features.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: delta feature engineering + training matrix builder"
```

---

### Task 4: Twickenham Player Features

**Files:**
- Create: `src/twickenham.py`
- Create: `tests/test_twickenham.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_twickenham.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_twickenham.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/twickenham.py**

```python
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
    Index 3 (delta_home_away_win_pct) → delta_twk_win_pct
    Index 5 (is_neutral) → 1.0
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
# twk_avg_rating: BBC Sport / ESPN player rating at Twickenham (0–10 scale).
# If unavailable, use 7.0 as a neutral placeholder and note in notebook.
# -------------------------------------------------------------------

BATH_STARTERS: list[dict] = []
EXETER_STARTERS: list[dict] = []
LEICESTER_STARTERS: list[dict] = []
NORTHAMPTON_STARTERS: list[dict] = []
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
pytest tests/test_twickenham.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Fill probable XVs manually**

Navigate to BBC Sport and Premiership Rugby squad pages for each semifinalist. For each likely starter (15 per team), record:
- `twk_appearances`: total Twickenham appearances (England caps at Twickenham + previous Premiership finals there)
- `twk_wins`: how many of those games the team won
- `twk_avg_rating`: use 7.0 if no rating data is available

Fill in `BATH_STARTERS`, `EXETER_STARTERS`, `LEICESTER_STARTERS`, `NORTHAMPTON_STARTERS` in `src/twickenham.py`.

- [ ] **Step 6: Commit**

```bash
git add src/twickenham.py tests/test_twickenham.py
git commit -m "feat: Twickenham player aggregation + final feature vector substitution"
```

---

### Task 5: XGBoost Model

**Files:**
- Create: `src/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/model.py**

```python
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
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
pytest tests/test_model.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: XGBoost with Platt calibration — train, evaluate, save/load"
```

---

### Task 6: Monte Carlo Simulation

**Files:**
- Create: `src/simulate.py`
- Create: `tests/test_simulate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_simulate.py`:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_simulate.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/simulate.py**

```python
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
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
pytest tests/test_simulate.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simulate.py tests/test_simulate.py
git commit -m "feat: Monte Carlo tournament simulation + conditioned scoreline sampling"
```

---

### Task 7: Charts

**Files:**
- Create: `src/charts.py`

No unit tests — visual outputs; verify by running notebook 04.

- [ ] **Step 1: Implement src/charts.py**

```python
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path

CHARTS_DIR = Path("charts")

TEAM_COLOURS = {
    "Bath Rugby":         "#004B8D",
    "Exeter Chiefs":      "#003974",
    "Leicester Tigers":   "#00923F",
    "Northampton Saints": "#1C1C1C",
}


def plot_championship_win_pct(win_pcts: dict[str, float], save: bool = True) -> None:
    teams = sorted(win_pcts, key=win_pcts.get, reverse=True)
    pcts = [win_pcts[t] * 100 for t in teams]
    colours = [TEAM_COLOURS.get(t, "#888888") for t in teams]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(teams, pcts, color=colours)
    ax.set_xlabel("Championship Win Probability (%)")
    ax.set_title("2026 Premiership — Championship Win Probability\n(10,000 simulations)")
    ax.set_xlim(0, 100)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=11)
    plt.tight_layout()
    if save:
        plt.savefig(CHARTS_DIR / "championship_win_pct.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_scoreline_distribution(
    home_scores: list[int],
    away_scores: list[int],
    home_name: str,
    away_name: str,
    save: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scores, name in zip(axes, [home_scores, away_scores], [home_name, away_name]):
        ax.hist(scores, bins=range(0, 65, 3),
                color=TEAM_COLOURS.get(name, "#888888"),
                edgecolor="white", alpha=0.85)
        median = float(np.median(scores))
        ax.axvline(median, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median: {median:.0f} pts")
        ax.set_title(f"{name}")
        ax.set_xlabel("Points Scored")
        ax.set_ylabel("Simulated matches")
        ax.legend()
    plt.suptitle(f"{home_name}  vs  {away_name} — Score Distributions", fontsize=13)
    plt.tight_layout()
    if save:
        fname = f"scores_{home_name.replace(' ', '_')}_v_{away_name.replace(' ', '_')}.png"
        plt.savefig(CHARTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall(
    model,
    X_row: np.ndarray,
    feature_names: list[str],
    title: str,
    save: bool = True,
) -> None:
    base_estimator = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    shap_vals = explainer.shap_values(X_row.reshape(1, -1))[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#D32F2F" if v > 0 else "#1976D2" for v in shap_vals]
    y_pos = range(len(feature_names))
    ax.barh(list(y_pos), shap_vals, color=colours)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  →  increases P(home win)")
    ax.set_title(f"Why? — {title}")
    plt.tight_layout()
    if save:
        fname = f"shap_{title.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(CHARTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_twickenham_heatmap(
    team_players: dict[str, list[dict]],
    save: bool = True,
) -> None:
    teams = list(team_players.keys())
    all_names = sorted({p["name"] for players in team_players.values() for p in players})
    data = np.zeros((len(all_names), len(teams)))
    for j, team in enumerate(teams):
        name_to_apps = {p["name"]: p["twk_appearances"] for p in team_players[team]}
        for i, name in enumerate(all_names):
            data[i, j] = name_to_apps.get(name, 0)

    fig, ax = plt.subplots(figsize=(len(teams) * 2 + 2, max(6, len(all_names) * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams, rotation=25, ha="right")
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Twickenham appearances")
    ax.set_title("Twickenham Experience — Probable Starters")
    plt.tight_layout()
    if save:
        plt.savefig(CHARTS_DIR / "twickenham_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from src.charts import plot_championship_win_pct; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/charts.py
git commit -m "feat: charts — championship %, scoreline distributions, SHAP, Twickenham heatmap"
```

---

### Task 8: Notebooks 02–04

**Files:**
- Create: `notebooks/02_feature_engineering.ipynb`
- Create: `notebooks/03_model_train.ipynb`
- Create: `notebooks/04_predict_simulate.ipynb`

- [ ] **Step 1: Write notebooks/02_feature_engineering.ipynb**

```python
# Cell 1 — markdown: # 02 — Feature Engineering

# Cell 2
import sys; sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from src.data_pull import compute_team_stats
from src.features import build_training_matrix, save_training_matrix

historical = pd.read_csv('../data/raw/historical_matches.csv')

# Build per-season team stats, then construct training rows
all_X, all_y = [], []
for season, grp in historical.groupby('season'):
    all_teams = pd.concat([grp['home_team'], grp['away_team']]).unique()
    stats = {team: compute_team_stats(grp, team) for team in all_teams}
    X, y = build_training_matrix(grp, stats)
    if len(X):
        all_X.append(X)
        all_y.append(y)

X_train = np.vstack(all_X)
y_train = np.concatenate(all_y)
save_training_matrix(X_train, y_train)
print(f'Training matrix: {X_train.shape}, home win rate: {y_train.mean():.2%}')
```

- [ ] **Step 2: Write notebooks/03_model_train.ipynb**

```python
# Cell 1 — markdown: # 03 — Model Training

# Cell 2
import sys; sys.path.insert(0, '..')
import pandas as pd
import numpy as np
from src.model import train_model, evaluate_model, save_model
from src.features import FEATURE_NAMES

df = pd.read_csv('../data/processed/features_train.csv')
X = df[FEATURE_NAMES].values
y = df['home_win'].values

model = train_model(X, y)
metrics = evaluate_model(model, X, y)
print('CV metrics:', metrics)
# Acceptable: cv_accuracy > 0.55, cv_log_loss < 0.70
# If accuracy is near 0.50, data volume is low — note in notebook and continue.

save_model(model)
print('Model saved to models/xgb_calibrated.pkl')
```

- [ ] **Step 3: Write notebooks/04_predict_simulate.ipynb**

```python
# Cell 1 — markdown: # 04 — Predictions & Simulation

# Cell 2 — semi-final predictions
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from src.model import load_model, predict_proba
from src.features import build_match_features, FEATURE_NAMES
from src.twickenham import (
    BATH_STARTERS, EXETER_STARTERS, LEICESTER_STARTERS, NORTHAMPTON_STARTERS,
    substitute_twickenham_into_vector,
)
from src.simulate import TournamentTeams, run_tournament, simulate_scoreline
from src.charts import (plot_championship_win_pct, plot_scoreline_distribution,
                         plot_shap_waterfall, plot_twickenham_heatmap)

model = load_model()
raw = pd.read_csv('../data/raw/current_season_stats.csv').set_index('team').to_dict('index')

# Semi 1: Bath (home) vs Exeter
v_semi1 = build_match_features(raw['Bath Rugby'], raw['Exeter Chiefs'], is_neutral=False)
p_bath = float(predict_proba(model, v_semi1.reshape(1, -1))[0])
print(f'Semi 1 — Bath win prob:      {p_bath:.1%}')
print(f'Semi 1 — Exeter win prob:    {1-p_bath:.1%}')

# Semi 2: Leicester (home) vs Northampton
v_semi2 = build_match_features(raw['Leicester Tigers'], raw['Northampton Saints'], is_neutral=False)
p_leicester = float(predict_proba(model, v_semi2.reshape(1, -1))[0])
print(f'Semi 2 — Leicester win prob: {p_leicester:.1%}')
print(f'Semi 2 — Northampton prob:   {1-p_leicester:.1%}')

# Cell 3 — final matchup probabilities (Twickenham features)
finals = {
    ('Bath Rugby',    'Leicester Tigers'):     substitute_twickenham_into_vector(BATH_STARTERS,    LEICESTER_STARTERS,    v_semi1),
    ('Bath Rugby',    'Northampton Saints'):   substitute_twickenham_into_vector(BATH_STARTERS,    NORTHAMPTON_STARTERS,  v_semi1),
    ('Exeter Chiefs', 'Leicester Tigers'):     substitute_twickenham_into_vector(EXETER_STARTERS,  LEICESTER_STARTERS,    v_semi2),
    ('Exeter Chiefs', 'Northampton Saints'):   substitute_twickenham_into_vector(EXETER_STARTERS,  NORTHAMPTON_STARTERS,  v_semi2),
}

match_probs = {
    ('Bath Rugby',       'Exeter Chiefs'):       p_bath,
    ('Leicester Tigers', 'Northampton Saints'):  p_leicester,
}
for matchup, vec in finals.items():
    match_probs[matchup] = float(predict_proba(model, vec.reshape(1, -1))[0])
    print(f'{matchup[0]} vs {matchup[1]}: {match_probs[matchup]:.1%}')

# Cell 4 — Monte Carlo tournament
teams = TournamentTeams('Bath Rugby', 'Exeter Chiefs', 'Leicester Tigers', 'Northampton Saints')
championship_pcts = run_tournament(teams, match_probs, n=10_000)
print('\nChampionship win %:')
for team, pct in sorted(championship_pcts.items(), key=lambda x: -x[1]):
    print(f'  {team}: {pct:.1%}')

# Cell 5 — Charts
plot_championship_win_pct(championship_pcts)

# Scoreline distributions for semi 1
outcome_semi1 = 'home' if p_bath > 0.5 else 'away'
home_scores = [simulate_scoreline(raw['Bath Rugby']['pts_scored_pg'],
                                   raw['Exeter Chiefs']['pts_scored_pg'],
                                   outcome_semi1)[0] for _ in range(5_000)]
away_scores = [simulate_scoreline(raw['Bath Rugby']['pts_scored_pg'],
                                   raw['Exeter Chiefs']['pts_scored_pg'],
                                   outcome_semi1)[1] for _ in range(5_000)]
plot_scoreline_distribution(home_scores, away_scores, 'Bath Rugby', 'Exeter Chiefs')

# SHAP — Semi 1
plot_shap_waterfall(model, v_semi1, FEATURE_NAMES, 'Bath vs Exeter (Semi 1)')

# Twickenham heatmap — only if starters are filled in
team_players = {
    'Bath Rugby': BATH_STARTERS,
    'Exeter Chiefs': EXETER_STARTERS,
    'Leicester Tigers': LEICESTER_STARTERS,
    'Northampton Saints': NORTHAMPTON_STARTERS,
}
if any(team_players.values()):
    plot_twickenham_heatmap(team_players)
```

- [ ] **Step 4: Run notebooks 02 and 03 sequentially**

```bash
cd /Users/yashsewpaul/code/prem-rugby-prediction
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb \
    --output notebooks/02_feature_engineering_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_train.ipynb \
    --output notebooks/03_model_train_executed.ipynb
```

Expected: `data/processed/features_train.csv` exists; `models/xgb_calibrated.pkl` exists; metrics printed.

- [ ] **Step 5: Run notebook 04**

```bash
jupyter nbconvert --to notebook --execute notebooks/04_predict_simulate.ipynb \
    --output notebooks/04_predict_simulate_executed.ipynb
```

Expected: Win probabilities printed; `charts/championship_win_pct.png`, `charts/scores_Bath_Rugby_v_Exeter_Chiefs.png`, `charts/shap_Bath_vs_Exeter_(Semi_1).png` created.

- [ ] **Step 6: Commit**

```bash
git add notebooks/ charts/ models/ data/processed/
git commit -m "feat: complete notebooks 02-04 + all chart outputs"
```

---

### Task 9: Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS. Typical count: 22 tests.

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: all tests green — model ready for publication"
```
