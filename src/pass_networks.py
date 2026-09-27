"""
Passing-network analysis: builds per-match passing networks, computes
centrality/structure metrics, and rolls them up into team-era features that
plug into the existing ANOVA/RF pipeline in analyze.py.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from build_features import load_events, load_matches, TEAM_NAME_FIXES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

ERAS = ["pre_covid", "covid", "post_covid"]
SEASON_LABELS = {"pre_covid": "2018/19", "covid": "2020/21", "post_covid": "2023/24"}


def build_match_pass_network(ev: pd.DataFrame, team: str, match_id: int) -> nx.Graph:
    m = ev[(ev["match_id"] == match_id) & (ev["team"] == team)]
    passes = m[(m["type"] == "Pass") & (m["pass_outcome"].isna())]
    passes = passes.dropna(subset=["player", "pass_recipient", "location", "pass_end_location"])

    if passes.empty:
        return nx.Graph()

    # position a player by every touch they're involved in: where they were
    # when passing, and where they were when receiving — otherwise a player
    # who only ever receives (never passes) in this match has no position at all
    passer_pos = passes[["player", "location"]].rename(columns={"player": "name", "location": "loc"})
    recipient_pos = passes[["pass_recipient", "pass_end_location"]].rename(columns={"pass_recipient": "name", "pass_end_location": "loc"})
    all_pos = pd.concat([passer_pos, recipient_pos], ignore_index=True)
    all_pos["x"] = all_pos["loc"].apply(lambda l: l[0])
    all_pos["y"] = all_pos["loc"].apply(lambda l: l[1])
    avg_pos = all_pos.groupby("name")[["x", "y"]].mean()

    G = nx.Graph()
    for player, row in avg_pos.iterrows():
        G.add_node(player, x=row["x"], y=row["y"])

    edges = passes.groupby(["player", "pass_recipient"]).size().reset_index(name="weight")
    for _, r in edges.iterrows():
        a, b, w = r["player"], r["pass_recipient"], r["weight"]
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)

    return G

def network_summary_stats(G: nx.Graph, min_passes: int = 3) -> dict:
    """
    Structural summary of one network: how hub-dependent (centralized) vs.
    distributed the passing was, plus density and clustering.
    """
    # drop fringe nodes (subs who touched the ball once or twice) so
    # centrality isn't skewed by noise
    keep = [n for n, d in G.degree(weight="weight") if d >= min_passes]
    H = G.subgraph(keep).copy()

    if H.number_of_nodes() < 3:
        return {"betweenness_mean": np.nan, "centralization": np.nan,
                "density": np.nan, "clustering": np.nan}

    betw = nx.betweenness_centrality(H, weight="weight", normalized=True)
    betw_vals = list(betw.values())

    # centralization: how far the network is from "everyone equally central"
    # (a single-hub star network -> high; an evenly distributed network -> low)
    max_betw = max(betw_vals)
    centralization = sum(max_betw - v for v in betw_vals) / (len(betw_vals) - 1)

    return {
        "betweenness_mean": np.mean(betw_vals),
        "centralization": centralization,
        "density": nx.density(H),
        "clustering": nx.average_clustering(H, weight="weight"),
    }


def team_era_network_features(era: str) -> pd.DataFrame:
    """One row per team for this era: network stats averaged across matches."""
    ev = load_events_raw(era)
    matches = load_matches(era)

    rows = []
    for team in ev["team"].dropna().unique():
        team_match_ids = ev[ev["team"] == team]["match_id"].unique()
        match_stats = []
        for mid in team_match_ids:
            G = build_match_pass_network(ev, team, mid)
            match_stats.append(network_summary_stats(G))

        stats_df = pd.DataFrame(match_stats)
        row = {"team": team, "era": era, "season": SEASON_LABELS[era]}
        row.update(stats_df.mean(numeric_only=True).to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def load_events_raw(era: str) -> pd.DataFrame:
    """Same team-name cleanup as build_features.load_events, kept local
    here so this module doesn't fight over which team-name fix list wins."""
    ev = pd.read_parquet(DATA_DIR / f"events_{era}.parquet")
    ev["team"] = ev["team"].replace(TEAM_NAME_FIXES)
    return ev


def plot_pass_network(G: nx.Graph, team: str, ax, min_passes: int = 3):
    keep = [n for n, d in G.degree(weight="weight") if d >= min_passes]
    H = G.subgraph(keep)

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#22312b", line_color="#c7d5cc")
    pitch.draw(ax=ax)

    pos = {n: (H.nodes[n]["x"], H.nodes[n]["y"]) for n in H.nodes}
    degrees = dict(H.degree(weight="wewight"))

    for a, b, d in H.edges(data=True):
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, color="white", alpha=0.35, linewidth=d["weight"] / 8, zorder=1)

    xs = [pos[n][0] for n in H.nodes]
    ys = [pos[n][1] for n in H.nodes]
    sizes = [degrees[n] * 12 for n in H.nodes]
    ax.scatter(xs, ys, s=sizes, color="#ff5c5c", edgecolors="black", zorder=2)

    for n in H.nodes:
        ax.annotate(n.split()[-1], pos[n], fontsize=7, color="white",
                    ha="center", va="center", zorder=3)

    ax.set_title(team, fontsize=11)

def showcase_match(ev: pd.DataFrame, team: str) -> int:
    """Pick the team's highest-pass-volume match in this era's data as the
    match to visualize"""
    counts = ev[(ev["team"] == team) & (ev["type"] == "Pass")].groupby("match_id").size()
    return counts.idxmax()

def main():
    all_features = []
    for era in ERAS:
        print(f"Building network features for {era}...")
        all_features.append(team_era_network_features(era))
    result = pd.concat(all_features, ignore_index=True)
    result.to_csv(DATA_DIR / "network_features.csv", index=False)
    print(result.describe())

    # showcase passing networks for the top-4 teams, post-covid era
    from plot_top4 import top_n_teams  # reuse the standings logic 
    era = "post_covid"
    ev = load_events_raw(era)
    teams = top_n_teams(era)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, team in zip(axes.flat, teams):
        mid = showcase_match(ev, team)
        G = build_match_pass_network(ev, team, mid)
        plot_pass_network(G, team, ax)
    fig.suptitle(f"Top 4 WSL Passing Networks — {era} (highest-volume match each)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top4_pass_networks.png", dpi=150)


if __name__ == "__main__":
    main()