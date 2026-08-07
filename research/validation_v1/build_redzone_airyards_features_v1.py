from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets
from validation_utils import VALIDATION_DIR, clean_name, initial_last_key

PBP_TAG = "pbp"
SEASONS = tuple(range(1999, 2026))
RED_ZONE_YARDLINE_100 = 20  # inside the opponent's 20-yard line

PBP_USECOLS = [
    "game_id", "season", "season_type", "week", "posteam",
    "yardline_100", "play_type",
    "pass_attempt", "rush_attempt", "complete_pass",
    "air_yards", "pass_touchdown", "rush_touchdown",
    "yards_gained", "rushing_yards", "receiving_yards",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
]

# "Explosive play" thresholds. 15+ rushing yards and 20+ receiving yards are
# the conventional public-analytics cutoffs; using the standard lines keeps
# the resulting archetypes comparable to outside work rather than tuned to
# this dataset (which would risk fitting noise).
EXPLOSIVE_RUSH_YARDS = 15
EXPLOSIVE_REC_YARDS = 20


def _season_asset_url(season: int, assets: list) -> str | None:
    matches = [a for a in assets if str(a.get("name", "")) == f"play_by_play_{season}.csv.gz"]
    if not matches:
        return None
    return matches[0].get("browser_download_url")


def _aggregate_one_season(season: int, assets: list) -> pd.DataFrame:
    """
    Fetch one season's play-by-play, filter to usable columns immediately
    (372 raw columns x ~45k plays/season is too heavy to concat across 27
    seasons), and return season-level receiver/rusher aggregates.

    `assets` is the pbp release's asset listing, fetched ONCE by the caller
    -- fetching it fresh per season burns the GitHub API's unauthenticated
    60-requests/hour limit in well under one full loop (hit this exact bug
    building this script: 7 seasons in, 403 rate limit exceeded).
    """
    url = _season_asset_url(season, assets)
    if url is None:
        return pd.DataFrame()

    df = pd.read_csv(url, usecols=lambda c: c in PBP_USECOLS, low_memory=False)
    df = df[df["season_type"].astype(str).eq("REG")].copy()
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce")
    df["is_red_zone"] = df["yardline_100"] <= RED_ZONE_YARDLINE_100

    # Receiving side: targets, red zone targets, air yards, explosive catches.
    targets = df[df["pass_attempt"].fillna(0).astype(int).eq(1) & df["receiver_player_id"].notna()].copy()
    targets["rec_yards_play"] = pd.to_numeric(targets.get("receiving_yards"), errors="coerce")
    if targets["rec_yards_play"].isna().all():
        targets["rec_yards_play"] = pd.to_numeric(targets.get("yards_gained"), errors="coerce")
    targets["is_explosive_rec"] = targets["rec_yards_play"] >= EXPLOSIVE_REC_YARDS
    receiver_agg = targets.groupby(["receiver_player_id", "receiver_player_name", "posteam"]).agg(
        targets=("receiver_player_id", "size"),
        redzone_targets=("is_red_zone", "sum"),
        air_yards_total=("air_yards", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
        explosive_receptions=("is_explosive_rec", "sum"),
    ).reset_index().rename(columns={
        "receiver_player_id": "player_id",
        "receiver_player_name": "player_name",
        "posteam": "team",
    })

    # Rushing side: carries, red zone carries, explosive runs.
    carries = df[df["rush_attempt"].fillna(0).astype(int).eq(1) & df["rusher_player_id"].notna()].copy()
    carries["rush_yards_play"] = pd.to_numeric(carries.get("rushing_yards"), errors="coerce")
    if carries["rush_yards_play"].isna().all():
        carries["rush_yards_play"] = pd.to_numeric(carries.get("yards_gained"), errors="coerce")
    carries["is_explosive_run"] = carries["rush_yards_play"] >= EXPLOSIVE_RUSH_YARDS
    rusher_agg = carries.groupby(["rusher_player_id", "rusher_player_name", "posteam"]).agg(
        carries=("rusher_player_id", "size"),
        redzone_carries=("is_red_zone", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
        explosive_runs=("is_explosive_run", "sum"),
        rush_yards_total=("rush_yards_play", "sum"),
    ).reset_index().rename(columns={
        "rusher_player_id": "player_id",
        "rusher_player_name": "player_name",
        "posteam": "team",
    })

    player_agg = receiver_agg.merge(rusher_agg, on=["player_id", "player_name", "team"], how="outer")

    # Team-season red zone play totals (for share denominators).
    team_redzone_targets = targets[targets["is_red_zone"]].groupby("posteam").size().rename("team_redzone_targets")
    team_redzone_carries = carries[carries["is_red_zone"]].groupby("posteam").size().rename("team_redzone_carries")
    team_air_yards = targets.groupby("posteam")["air_yards"].sum().rename("team_air_yards_total")
    team_agg = pd.concat([team_redzone_targets, team_redzone_carries, team_air_yards], axis=1).reset_index().rename(columns={"posteam": "team"})

    player_agg = player_agg.merge(team_agg, on="team", how="left")
    player_agg["season"] = season
    return player_agg


def build_redzone_airyards_dataset(force_refresh: bool = False) -> pd.DataFrame:
    # Per-season cache files (not one combined cache) so a rate-limit hit or
    # any other mid-run failure doesn't lose already-fetched seasons -- a
    # re-run only needs to fetch whatever's still missing.
    season_cache_dir = CACHE_DIR / "redzone_airyards_by_season"
    season_cache_dir.mkdir(parents=True, exist_ok=True)

    missing_seasons = [
        season for season in SEASONS
        if force_refresh or not (season_cache_dir / f"{season}.csv").exists()
    ]
    assets = fetch_release_assets(PBP_TAG) if missing_seasons else []

    for season in missing_seasons:
        agg = _aggregate_one_season(season, assets)
        agg.to_csv(season_cache_dir / f"{season}.csv", index=False)
        print(f"pbp {season}: {len(agg)} player rows aggregated")

    frames = []
    for season in SEASONS:
        season_path = season_cache_dir / f"{season}.csv"
        if season_path.exists():
            season_df = pd.read_csv(season_path)
            if not season_df.empty:
                frames.append(season_df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if raw.empty:
        return pd.DataFrame(columns=[
            "season", "player_key", "player_name", "team",
            "prior_redzone_target_share", "prior_redzone_carry_share",
            "prior_air_yards", "prior_air_yards_share", "prior_targets", "prior_carries_pbp",
        ])

    df = raw.copy()
    for col in [
        "targets", "redzone_targets", "air_yards_total", "receiving_tds",
        "carries", "redzone_carries", "rushing_tds",
        "explosive_receptions", "explosive_runs", "rush_yards_total",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["redzone_target_share"] = df["redzone_targets"] / df["team_redzone_targets"].replace(0, np.nan)
    df["redzone_carry_share"] = df["redzone_carries"] / df["team_redzone_carries"].replace(0, np.nan)
    df["air_yards_share"] = df["air_yards_total"] / df["team_air_yards_total"].replace(0, np.nan)

    # aDOT (average depth of target) -- the standard separator between deep
    # threats and possession/slot receivers.
    df["adot"] = df["air_yards_total"] / df["targets"].replace(0, np.nan)
    # Explosive-play rates: the "long run" / big-play dimension.
    df["explosive_run_rate"] = df["explosive_runs"] / df["carries"].replace(0, np.nan)
    df["explosive_rec_rate"] = df["explosive_receptions"] / df["targets"].replace(0, np.nan)
    df["yards_per_carry"] = df["rush_yards_total"] / df["carries"].replace(0, np.nan)

    df["player_key"] = df["player_name"].apply(clean_name)
    # pbp's receiver/rusher name fields are PFR-style abbreviated
    # ("J.Jefferson"), not full names ("Justin Jefferson") like every other
    # source in this chain -- clean_name() alone would silently fail to
    # match almost anyone against the base dataset. initial_last_key()
    # (already used elsewhere in this project for exactly this kind of
    # fuzzy match, e.g. build_predraft_dataset.py's ADP fallback) collapses
    # both formats to the same "j jefferson" key.
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    # Shift +1 season so this becomes a prior-season (pre-draft-safe) input,
    # same convention as build_snap_share_features_v1.py.
    df["season"] = df["season"] + 1

    out = df.rename(columns={
        "redzone_target_share": "prior_redzone_target_share",
        "redzone_carry_share": "prior_redzone_carry_share",
        "air_yards_total": "prior_air_yards",
        "air_yards_share": "prior_air_yards_share",
        "targets": "prior_targets_pbp",
        "carries": "prior_carries_pbp",
        "adot": "prior_adot",
        "explosive_run_rate": "prior_explosive_run_rate",
        "explosive_rec_rate": "prior_explosive_rec_rate",
        "yards_per_carry": "prior_yards_per_carry",
        "player_id": "player_id_gsis_lookalike",
    })
    keep_cols = [
        "season", "player_key", "initial_last_key", "player_name", "team",
        "prior_redzone_target_share", "prior_redzone_carry_share",
        "prior_air_yards", "prior_air_yards_share",
        "prior_targets_pbp", "prior_carries_pbp",
        "prior_adot", "prior_explosive_run_rate", "prior_explosive_rec_rate",
        "prior_yards_per_carry",
    ]
    out = out[keep_cols].reset_index(drop=True)
    return out


def main() -> None:
    dataset = build_redzone_airyards_dataset()
    output_path = VALIDATION_DIR / "data" / "redzone_airyards_player_seasons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Red zone / air yards dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Non-null prior_redzone_target_share: {int(dataset['prior_redzone_target_share'].notna().sum())}")
    print(f"Non-null prior_air_yards_share: {int(dataset['prior_air_yards_share'].notna().sum())}")


if __name__ == "__main__":
    main()
