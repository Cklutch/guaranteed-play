from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import (
    CACHE_DIR,
    csv_assets_for_years,
    fetch_release_assets,
    pfr_to_gsis_crosswalk,
    read_remote_csv,
)
from validation_utils import VALIDATION_DIR, clean_name

SNAP_COUNTS_TAG = "snap_counts"
# nflverse snap count tracking starts in 2012 -- earlier seasons simply have
# no data, not a bug in this script.
SEASONS = tuple(range(2012, 2026))


def _fetch_snap_counts_season(season: int) -> pd.DataFrame:
    assets = fetch_release_assets(SNAP_COUNTS_TAG)
    matches = [
        asset for asset in assets
        if str(asset.get("name", "")) == f"snap_counts_{season}.csv"
    ]
    if not matches:
        return pd.DataFrame()
    return read_remote_csv(matches[0])


def _pull_all_seasons(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "raw_snap_counts_2012_2025.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    frames = []
    for season in SEASONS:
        df = _fetch_snap_counts_season(season)
        if df.empty:
            continue
        frames.append(df)
        print(f"snap_counts {season}: {len(df)} player-game rows")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    return combined


def build_snap_share_dataset() -> pd.DataFrame:
    raw = _pull_all_seasons()
    if raw.empty:
        return pd.DataFrame(columns=["season", "player_key", "player_name", "position", "team", "prior_snap_share"])

    df = raw[raw["game_type"].astype(str).eq("REG")].copy()
    df["offense_snaps"] = pd.to_numeric(df["offense_snaps"], errors="coerce").fillna(0.0)

    # Team offensive snaps for a given game = the max player offense_snaps in
    # that team-game (the player(s) on the field for every offensive snap).
    team_game_snaps = (
        df.groupby(["season", "team", "game_id"])["offense_snaps"]
        .max()
        .reset_index()
        .rename(columns={"offense_snaps": "team_game_offense_snaps"})
    )
    team_season_snaps = (
        team_game_snaps.groupby(["season", "team"])["team_game_offense_snaps"]
        .sum()
        .reset_index()
        .rename(columns={"team_game_offense_snaps": "team_season_offense_snaps"})
    )

    # A traded player can appear under multiple teams in one season -- sum
    # their snaps per team, then keep the team they played the most for as
    # their season-of-record team (simple, matches how the rest of this
    # project already treats mid-season trades elsewhere).
    # Group by pfr_player_id (a real stable ID) as well as name, so the
    # crosswalk to gsis_id below has a key to join on and two players with
    # the same abbreviated name never merge into one row.
    player_team_season = (
        df.groupby(["season", "pfr_player_id", "player", "position", "team"])["offense_snaps"]
        .sum()
        .reset_index()
        .rename(columns={"offense_snaps": "player_team_season_snaps"})
    )
    player_team_season = player_team_season.sort_values(
        ["season", "pfr_player_id", "player_team_season_snaps"], ascending=[True, True, False]
    )
    primary_team = player_team_season.drop_duplicates(["season", "pfr_player_id"], keep="first")

    merged = primary_team.merge(team_season_snaps, on=["season", "team"], how="left")
    merged["snap_share"] = (
        merged["player_team_season_snaps"] / merged["team_season_offense_snaps"].replace(0, np.nan)
    )

    merged["player_key"] = merged["player"].apply(clean_name)
    merged["prior_source_season"] = merged["season"] + 1

    # Attach the real gsis_id via the pfr_id crosswalk so downstream joins
    # can key on a stable player ID instead of a name. Name-based joining
    # collides distinct players (13.8% of rows measured), e.g. 2017's
    # D.Johnson (ARI) and D.Johnson Jr. (CLE) both resolving to "d johnson".
    crosswalk = pfr_to_gsis_crosswalk(SEASONS)
    if not crosswalk.empty:
        merged = merged.merge(
            crosswalk.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"}),
            on="pfr_player_id",
            how="left",
        )
    else:
        merged["player_id"] = pd.NA

    out = merged.rename(columns={
        "player": "player_name",
        "prior_source_season": "season_out",
        "snap_share": "prior_snap_share",
    })
    out = out[["season_out", "player_id", "player_key", "player_name", "position", "team", "prior_snap_share"]]
    out = out.rename(columns={"season_out": "season"})
    return out.reset_index(drop=True)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_snap_share_dataset()
    output_path = VALIDATION_DIR / "data" / "snap_share_player_seasons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Snap share dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Seasons covered (as prior-season input): {sorted(dataset['season'].dropna().unique().tolist())}")
    print(f"Non-null prior_snap_share: {int(dataset['prior_snap_share'].notna().sum())}")


if __name__ == "__main__":
    main()
