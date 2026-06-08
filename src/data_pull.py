import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

DATA_RAW = Path("data/raw")

# ESPN Gallagher Premiership league ID — verify via probe_espn() output
ESPN_LEAGUE_ID = "270559"
# TheSportsDB English Premiership Rugby league ID — confirmed via searchevents probe
# ID 4414 = "English Premiership Rugby" (free tier: ~15 results/season, no pagination)
TSDB_LEAGUE_ID = "4414"

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
    """Fetch historical results from TheSportsDB.

    Free tier returns up to ~15 unique events per season via eventsseason.
    Deduplicates by event ID to guard against duplicate pagination responses.
    """
    import time

    seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
    seen_ids: set = set()
    rows = []
    for season in seasons:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/3/eventsseason.php",
            params={"id": league_id, "s": season},
            timeout=15,
        )
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(
                "https://www.thesportsdb.com/api/v1/json/3/eventsseason.php",
                params={"id": league_id, "s": season},
                timeout=15,
            )
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        for event in (data.get("events") or []):
            event_id = event.get("idEvent")
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)
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
        time.sleep(2)  # respect free-tier rate limit between season requests
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

    # Try bonus approximation: score >= 28 pts ~ 4 tries + conversions
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
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    historical.to_csv(DATA_RAW / "historical_matches.csv", index=False)
    pd.DataFrame(team_stats).to_csv(DATA_RAW / "current_season_stats.csv", index=False)
    print(f"Saved {len(historical)} historical matches")
    print(f"Saved stats for {len(team_stats)} teams")
