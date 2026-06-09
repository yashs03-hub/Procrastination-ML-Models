import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    ax.set_xlabel("SHAP value  ->  increases P(home win)")
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
