"""
Pulls WSL math and event data from StatsBomb's open data via statsbompy for 3 eras:

    Pre-COVID   :   2018/19 (season_id=4)
    COVID       :   2020/21 (season_id=90) played behind closed doors
    Post-COVID  :   2023/23 (season_id=281) post-Euro 2022 boom

Saves raw match lists and event data to /data

Useage:
    python src/collect_data.py
"""

import time
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COMPETITION_ID = 37 # WSL

ERAS = {
    "pre_covid": {"season_id": 4, "season_name": "2018/19"},
    "covid": {"season_id": 90, "season_name": "2020/21"},
    "post_covid": {"season_id": 281, "season_name": "2023/24"},
}

def fetch_matches(season_id: int) -> pd.DataFrame:
    return sb.matches(competition_id=COMPETITION_ID, season_id=season_id)

def fetch_events_for_season(season_id: int, era_label:str) ->  pd.DataFrame:
    """
    pull event data for every match in a season, tagging rows w/ era
    """
    matches = fetch_matches(season_id)
    all_events = []

    for i, match_id in enumerate(matches["match_id"], start=1):
        try:
            ev = sb.events(match_id=match_id)
            ev["match_id"] = match_id
            ev["era"] = era_label
            all_events.append(ev)
        except Exception as e:
            print(f" [warn] match {match_id} failed: {e}")
        if i % 20 == 0:
            print(f" ...{i}/{len(matches)} matches pulled")
        time.sleep(0.05)

    return pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

def main():
    match_frames = []

    for era_label, info in ERAS.items():
        season_id = info["season_id"]
        season_name = info["season_name"]
        print(f"\n {era_label} ({season_name})")

        matches = fetch_matches(season_id)
        matches["era"] = era_label
        match_frames.append(matches)
        matches.to_parquet(DATA_DIR / f"matches_{era_label}.parquet")
        print(f"Matches: {len(matches)}")

        events = fetch_events_for_season(season_id, era_label)
        events.to_parquet(DATA_DIR / f"events_{era_label}.parquet")
        print(f"Events collected: {len(events)}")

    all_matches = pd.concat(match_frames, ignore_index=True)
    all_matches.to_parquet(DATA_DIR / "matches_all.parquet")

if __name__ == "__main__":
    main()