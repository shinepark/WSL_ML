"""
Extra visuals: shot maps, xG trendlines, and radar charts comparing WSL's top 4
teams against league average, per era.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplsoccer import Pitch
from math import pi

from build_features import load_events, load_matches
from analyze import FEATURE_COLS, ERA_ORDER

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def compute_standings(matches: pd.DataFrame) -> pd.Series:
    """League points table from match results."""
    records = []
    for _, m in matches.iterrows():
        home, away, hg, ag = m["home_team"], m["away_team"], m["home_score"], m["away_score"]
        if hg > ag:
            records += [(home, 3), (away, 0)]
        elif hg < ag:
            records += [(home, 0), (away, 3)]
        else:
            records += [(home, 1), (away, 1)]
    pts = pd.DataFrame(records, columns=["team", "points"])
    return pts.groupby("team")["points"].sum().sort_values(ascending=False)


def top_n_teams(era: str, n: int = 4) -> list:
    matches = load_matches(era)
    return compute_standings(matches).head(n).index.tolist()


def plot_team_shotmap(ev: pd.DataFrame, team: str, ax):
    shots = ev[(ev["team"] == team) & (ev["type"] == "Shot")].dropna(subset=["location", "shot_statsbomb_xg"])
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc")
    pitch.draw(ax=ax)
    xs = shots["location"].apply(lambda l: l[0])
    ys = shots["location"].apply(lambda l: l[1])
    is_goal = shots["shot_outcome"] == "Goal"
    pitch.scatter(xs[~is_goal], ys[~is_goal], s=shots.loc[~is_goal, "shot_statsbomb_xg"] * 900,
                  ax=ax, color="#e0e0e0", edgecolors="black", alpha=0.7)
    pitch.scatter(xs[is_goal], ys[is_goal], s=shots.loc[is_goal, "shot_statsbomb_xg"] * 900,
                  ax=ax, color="#ff5c5c", edgecolors="black")
    ax.set_title(team, fontsize=11)


def plot_top4_shotmaps(ev: pd.DataFrame, teams: list, era_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, team in zip(axes.flat, teams):
        plot_team_shotmap(ev, team, ax)
    fig.suptitle(f"Top 4 WSL Shot Maps - {era_label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top4_shotmaps.png", dpi=150)


def team_xg_by_match(ev: pd.DataFrame, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    shots = ev[(ev["team"] == team) & (ev["type"] == "Shot")]
    per_match = shots.groupby("match_id")["shot_statsbomb_xg"].sum().reset_index()

    # one row per match_id, and a real datetime dtype (was being sorted as a
    # string before, which is why the trendline zigzagged out of order)
    m = matches.drop_duplicates("match_id")[["match_id", "match_date"]].copy()
    m["match_date"] = pd.to_datetime(m["match_date"])

    per_match = per_match.merge(m, on="match_id").sort_values("match_date").reset_index(drop=True)
    per_match["rolling_xg"] = per_match["shot_statsbomb_xg"].rolling(5, min_periods=1).mean()
    return per_match


def plot_top4_xg_trend(ev: pd.DataFrame, matches: pd.DataFrame, teams: list, era_label: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for team in teams:
        trend = team_xg_by_match(ev, matches, team)
        ax.plot(trend["match_date"], trend["rolling_xg"], label=team, linewidth=2)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    ax.set_title(f"Top 4 WSL — Rolling 5-Match xG ({era_label})")
    ax.set_ylabel("xG per match (5-match rolling avg)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top4_xg_trend.png", dpi=150)


def radar_league_vs_top4(df: pd.DataFrame, feature_cols: list, teams: list, era_key: str, ax):
    # filter on the raw era key ("post_covid"), not the display season label
    # ("2023/24") — df["season"] holds the latter, so filtering on it here
    # silently returned an empty frame and every plotted value came out NaN
    subset = df[df["era"] == era_key].reset_index(drop=True)
    norm = (subset[feature_cols] - subset[feature_cols].min()) / (subset[feature_cols].max() - subset[feature_cols].min())
    norm["team"] = subset["team"]

    angles = [n / len(feature_cols) * 2 * pi for n in range(len(feature_cols))]
    angles += angles[:1]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=8)

    league_vals = norm[feature_cols].mean().tolist()
    league_vals += league_vals[:1]
    ax.plot(angles, league_vals, color="grey", linestyle="--", linewidth=2, label="League avg")
    ax.fill(angles, league_vals, color="grey", alpha=0.05)

    for team in teams:
        vals = norm.loc[norm["team"] == team, feature_cols].mean().tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=team)

    ax.set_title(f"Top 4 vs League Average — {era_key}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))


def main():
    era = "post_covid"
    matches = load_matches(era)
    ev = load_events(era)
    df = pd.read_csv(DATA_DIR / "team_season_features.csv")

    teams = top_n_teams(era)
    plot_top4_shotmaps(ev, teams, era)
    plot_top4_xg_trend(ev, matches, teams, era)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    radar_league_vs_top4(df, FEATURE_COLS[:6], teams, era, ax)
    fig.savefig(OUT_DIR / "top4_radar.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()