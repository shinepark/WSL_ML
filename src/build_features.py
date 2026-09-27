"""
Transforms raw StatsBomb event data into team season feature table that captures playing style, quality,
and physical intensity.

Features (per team, per era):
    - possession_pct            : share of team possessions vs total match possessions
    - pass_completion_pct       : successful passes / attempted passes
    - passes_per_90             : passing volume, tempo proxy
    - progressive_passes_per_90 : passes that move the ball >=10m toward goal
    - avg_pass_length           : directness of play
    - long_ball_pct             : % of passes >30m (long-ball reliance)
    - pressures_per_90          : defensive intensity / press volume
    - ppda                      : passes allowed per defensive action (lower = higher press)
    - shots_per_90
    - xg_per_90                 : StatsBomb expected goals per 90
    - xg_per_shot               : shot quality
    - dribbles_per_90
    - carries_per_90
    - avg_carry_length          : ball-progression-by-carry proxy
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ERAS = ["pre_covid", "covid", "post_covid"]
SEASON_LABELS = {"pre_covid": "2018/19", "covid": "2020/21", "post_covid": "2023/24"}

# fixing inconsisten team name strings
TEAM_NAME_FIXES = {
    "Aston Villa": "Aston Villa W",
    "Manchester United": "Manchester United W",
}

def load_events(era: str) -> pd.DataFrame:
    ev = pd.read_parquet(DATA_DIR / f"events_{era}.parquet")
    ev["team"] = ev["team"].replace(TEAM_NAME_FIXES)
    ev["possession_team"] = ev["possession_team"].replace(TEAM_NAME_FIXES)
    return ev

def load_matches(era: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"matches_{era}.parquet")

def minutes_played_per_team(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Rough team minutes: 90 per match played (disregards extra time)
    """
    home = matches[["home_team"]].rename(columns={"home_team": "team"})
    away = matches[["away_team"]].rename(columns={"away_team": "team"})
    teams = pd.concat([home, away])
    minutes = teams.groupby("team").size().reset_index(name="matches_played")
    minutes["minutes_played"] = minutes["matches_played"] * 90
    return minutes

def pass_progression(row) -> bool:
    """
    a pass counts as progressive if it moves the ball at least 10 meters closer to opponent's goal
    """
    try:
        x0, _ = row["location"]
        x1, _ = row["pass_end_location"]
        return (x1 - x0) >= 10
    except Exception:
        return False

def build_team_features(era: str) -> pd.DataFrame:
    ev = load_events(era)
    matches = load_matches(era)
    minutes = minutes_played_per_team(matches)

    rows = []
    for team, g in ev.groupby("team"):
        if pd.isna(team):
            continue

        team_minutes_row = minutes[minutes["team"] == team]
        if team_minutes_row.empty:
            continue
        team_minutes = team_minutes_row["minutes_played"].values[0]
        matches_played = team_minutes_row["matches_played"].values[0]
        norm90 = 90.0 / team_minutes # mult per team totals by this to get per 90

        passes = g[g["type"] == "Pass"]
        n_passes = len(passes)
        completed_passes = passes["pass_outcome"].isna().sum() # NaN outcome = complete
        pass_completion_pct = 100 * completed_passes / n_passes if n_passes else np.nan

        pass_lengths = passes["pass_length"].dropna()
        avg_pass_length = pass_lengths.mean() if len(pass_lengths) else np.nan
        long_ball_pct = 100 * (pass_lengths > 30).sum() / len(pass_lengths) if len(pass_lengths) else np.nan

        has_loc = passes.dropna(subset=["location", "pass_end_location"])
        progressive = has_loc.apply(pass_progression, axis = 1).sum() if len(has_loc) else 0
        progressive_passes_per_90 = progressive * norm90

        pressures = (g["type"] == "Pressure").sum()
        pressures_per_90 = pressures * norm90

        shots = g[g["type"] == "Shot"]
        n_shots = len(shots)
        shots_per_90 = n_shots * norm90
        xg_total = shots["shot_statsbomb_xg"].sum()
        xg_per_90 = xg_total * norm90
        xg_per_shot = xg_total / n_shots if n_shots else np.nan

        dribbles = (g["type"] == "Dribble").sum()
        dribbles_per_90 = dribbles * norm90

        carries = g[g["type"] == "Carry"]
        n_carries = len(carries)
        carries_per_90 = n_carries * norm90
        carry_loc = carries.dropna(subset=["location", "carry_end_location"])
        if len(carry_loc):
            carry_dist = carry_loc.apply(
                lambda r: np.hypot(
                    r["carry_end_location"][0] - r["location"][0],
                    r["carry_end_location"][1] - r["location"][1],
                ),
                axis=1,
            )
            avg_carry_length = carry_dist.mean()
        else:
            avg_carry_length = np.nan

        # possession share: fraction of possession IDs in match where team was possessing team across all matches
        team_match_ids = g["match_id"].unique()
        poss_share_list = []
        for mid in team_match_ids:
            match_ev = ev[ev["match_id"] == mid]
            total_poss = match_ev["possession"].nunique()
            team_poss = match_ev[match_ev["possession_team"] == team]["possession"].nunique()
            if total_poss:
                poss_share_list.append(100 * team_poss / total_poss)
        possession_pct = np.mean(poss_share_list) if poss_share_list else np.nan

        rows.append(
            {
                "team": team,
                "era": era,
                "season": SEASON_LABELS[era],
                "matches_played": matches_played,
                "possession_pct": possession_pct,
                "pass_completion_pct": pass_completion_pct,
                "passes_per_90": n_passes * norm90,
                "progressive_passes_per_90": progressive_passes_per_90,
                "avg_pass_length": avg_pass_length,
                "long_ball_pct": long_ball_pct,
                "pressures_per_90": pressures_per_90,
                "shots_per_90": shots_per_90,
                "xg_per_90": xg_per_90,
                "xg_per_shot": xg_per_shot,
                "dribbles_per_90": dribbles_per_90,
                "carries_per_90": carries_per_90,
                "avg_carry_length": avg_carry_length,
            }
        )

    return pd.DataFrame(rows)

def main():
    all_features = []

    for era in ERAS:
        feats = build_team_features(era)
        all_features.append(feats)
        print(f"    {len(feats)} teams")

    result = pd.concat(all_features, ignore_index=True)
    out_path = DATA_DIR / "team_season_features.csv"
    result.to_csv(out_path, index=False)
    print(result.describe())

if __name__ == "__main__":
    main()