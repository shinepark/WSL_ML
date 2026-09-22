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