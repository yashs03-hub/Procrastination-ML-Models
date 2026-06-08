# Premiership Rugby 2026 Semi-Finals Prediction — Design Spec

**Date:** 2026-06-08
**Matches:** Bath vs Exeter, Leicester vs Northampton
**Author:** Yash Sewpaul

---

## 1. Objective

Build an XGBoost ML model trained on 3–5 seasons of Premiership Rugby data to predict:
- Win probabilities for both semi-finals
- Probable scorelines for both semi-finals
- Championship win % for all four teams via 10,000 Monte Carlo simulations of the final
- A LinkedIn post + shareable notebook documenting the methodology and findings

---

## 2. Architecture

```
ESPN API / Premiership scraper (ESPN first, scraper fallback)
        ↓
Historical match data (3–5 seasons) + Current 2025-26 team stats
        ↓
Feature engineering (per-matchup delta vector)
        ↓
XGBoost binary classifier trained on historical matches
        ↓
Platt scaling calibration
        ↓
Two inference paths:
  ├── Semis: home-ground feature vector
  └── Final: Twickenham-specific feature vector (player history)
        ↓
Monte Carlo simulation (10,000 runs)
        ↓
Output: probability charts, scoreline distributions, SHAP plots, LinkedIn post
```

---

## 3. Data Sources

**Priority order:**
1. ESPN hidden API (check for 2025-26 Premiership coverage)
2. The Sports DB API
3. Scraper fallback: Premiership Rugby website / BBC Sport / ESPN web

**Historical data required:** 3–5 seasons of Premiership regular season + playoff results. Each row = one match.

**Current season data required:** 2025-26 stats for Bath, Exeter, Leicester, Northampton.

**Player data (final simulation only):** Probable XVs for each potential finalist, with Twickenham appearance history (Premiership finals + England internationals).

---

## 4. Feature Engineering

All features computed as **deltas** (home team minus away team) to capture relative dominance and halve the feature space.

### Match-level features (semis + final)

| Feature | Description |
|---|---|
| `Δ pts_scored_pg` | Avg points scored per game this season |
| `Δ pts_conceded_pg` | Avg points conceded per game |
| `Δ try_bonus_rate` | Try bonus point frequency (attacking intent proxy) |
| `Δ home_away_win_pct` | Home team's home win %, away team's away win % |
| `Δ last5_form` | Avg points margin over last 5 games |
| `Δ lineout_win_pct` | Lineout dominance |
| `Δ scrum_win_pct` | Scrum dominance |

### Twickenham-specific features (final simulation only)

| Feature | Description |
|---|---|
| `Δ twk_player_avg_rating` | Mean player rating at Twickenham across likely starters |
| `Δ twk_experience` | Avg number of Twickenham appearances per starting XV |
| `Δ twk_win_pct` | Team's historical win rate at Twickenham |
| `is_neutral` | Always 1 for final (no home advantage) |

For the final, Twickenham features replace `home_away_win_pct`. Player lookup scoped to full starting XV (15 players per team).

---

## 5. Model

**Type:** XGBoost binary classifier. Target = `home_win` (1/0). No draw class — knockout rugby has no draws.

**Training pipeline:**
```
historical matches (3–5 seasons)
    → delta feature computation
    → train/test split stratified by season (prevents data leakage)
    → XGBoost with 5-fold cross-validation
    → Platt scaling calibration
    → save calibrated model
```

**Inference:**
- Same trained model for both semis and the final
- Semis: standard delta feature vector (home ground applies)
- Final: Twickenham feature vector (player history replaces home/away)

**Explainability:**
- SHAP values per prediction (feature contribution to this specific match)
- Global feature importance bar chart (across all historical matches)

---

## 6. Monte Carlo Simulation

10,000 simulation runs:

```
for i in range(10_000):
    p_bath    = model.predict_proba(semi1_features)[1]
    semi1_winner = "Bath"      if random() < p_bath    else "Exeter"

    p_leicester = model.predict_proba(semi2_features)[1]
    semi2_winner = "Leicester" if random() < p_leicester else "Northampton"

    final_features = get_twickenham_features(semi1_winner, semi2_winner)
    p_final = model.predict_proba(final_features)[1]
    champion = semi1_winner if random() < p_final else semi2_winner
```

**Scoreline estimation:** Sample from a team-specific points distribution fitted to season data, conditioned on win/loss outcome. Prevents the artefact of a "winner" scoring fewer points than the "loser."

**Simulation outputs:**
- Championship win % for all 4 teams
- Win probability per match (semi 1, semi 2, each possible final)
- Scoreline distribution per match
- Most likely final matchup

---

## 7. Notebook Structure

Four notebooks, independently re-runnable:

| Notebook | Purpose |
|---|---|
| `01_data_pull.ipynb` | API calls + scraper fallback, save raw data to CSV |
| `02_feature_engineering.ipynb` | Delta features, Twickenham player lookup |
| `03_model_train.ipynb` | XGBoost training, calibration, CV metrics |
| `04_predict_simulate.ipynb` | Semi + final simulation, all charts |

**Charts produced:**
- Championship win % bar chart (hero visual)
- Ridgeline scoreline distribution plots per match
- SHAP waterfall chart per match
- Twickenham experience heatmap (starters × appearances)

---

## 8. LinkedIn Post

**Format:** Long-form LinkedIn article (not status update — better reach + indexed).

**Structure:**
```
Hook
"Bath are favourites. But by how much? I built an ML model to find out —
here's what 10,000 simulations of the Premiership semis say."

The method (3–4 lines, no jargon)
- 5 seasons of Premiership data
- XGBoost trained on set-piece, form, home advantage
- Player-level Twickenham experience for the final

The findings (bullet points + hero chart)
- Championship win % for all 4 teams
- Most likely final matchup
- The one stat that swung it (SHAP insight)

Closing hook
"The model is wrong about something. It always is.
But here's what it can't account for..."
```

**Visuals attached:** Championship win % bar chart (hero) + one SHAP waterfall. Two images maximum.

**Notebook link:** Posted as first comment (not in body) to avoid LinkedIn algorithm suppression.

---

## 9. Constraints & Risks

| Risk | Mitigation |
|---|---|
| ESPN API doesn't cover 2025-26 Premiership | Fall back to scraper (BBC Sport / ESPN web) |
| Set-piece stats not available via API | Use points scored/conceded only; note limitation in post |
| Twickenham player history sparse for some players | Use team-level Twickenham win % as fallback |
| Too few historical seasons → underfitting | Use 5 seasons minimum; if unavailable, note in post |
| Model accuracy on held-out test set is low | Report accuracy honestly — uncertainty is part of the narrative |
