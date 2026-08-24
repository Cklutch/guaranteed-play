"""
RB archetype spec: third_down_snap_share.

Same real join this repo already relies on for route participation
(build_route_participation_features_v1.py) -- nflverse `pbp_participation`'s
`offense_players` column (real gsis_ids, one row per play) joined to real
play-by-play -- but filtered to `down == 3` across BOTH pass and rush plays,
not pass-only. Route participation only cares about dropbacks; third-down
snap share is about who's on the field for the money down regardless of
play call, which is exactly the signal the RB archetype spec's
receiving_back/committee_back checks need (a true receiving-down back
should show up here even on the rare third-down run).

Confirmed real, not proxied: `down` is a standard nflverse pbp column, not
currently in build_redzone_airyards_features_v1.py's PBP_USECOLS allowlist,
but trivially addable -- same PBP release, same fetch mechanism, no new
data source. participation coverage starts 2016 (same limit route
participation already has), so seasons before 2017 (as a prior-season
input) fall back to NaN, same as route participation's own floor.

Leakage: shifted +1 season, same convention as every other prior_* input
in this chain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets
from validation_utils import VALIDATION_DIR

PARTICIPATION_TAG = "pbp_participation"
SEASONS = tuple(range(2016, 2026))

# A real snap-share denominator needs a real sample. Team third-down play
# counts run ~150-220/season; a player under this many THIRD-DOWN snaps
# specifically (not total snaps) is too thin a sample for a stable rate.
MIN_THIRD_DOWN_SNAPS_FOR_RATE = 20


def _asset_url(assets: list, name: str) -> str | None:
    for asset in assets:
        if str(asset.get("name", "")) == name:
            return asset.get("browser_download_url")
    return None


def _aggregate_one_season(season: int, assets: list) -> pd.DataFrame:
    url = _asset_url(assets, f"pbp_participation_{season}.csv")
    if url is None:
        return pd.DataFrame()

    part = pd.read_csv(
        url, usecols=["nflverse_game_id", "play_id", "offense_players"], low_memory=False
    )
    pbp = pd.read_csv(
        f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz",
        usecols=lambda c: c in {
            "game_id", "play_id", "season_type", "posteam",
            "down", "pass_attempt", "rush_attempt",
        },
        low_memory=False, compression="gzip",
    )
    pbp = pbp[pbp["season_type"].astype(str).eq("REG")].copy()
    pbp["down"] = pd.to_numeric(pbp["down"], errors="coerce")
    # Real offensive plays only -- excludes timeouts/penalties/no-plays
    # that still carry a `down` value but never actually snapped the ball.
    is_real_play = pbp["pass_attempt"].fillna(0).astype(int).eq(1) | pbp["rush_attempt"].fillna(0).astype(int).eq(1)
    third_down = pbp[(pbp["down"] == 3) & is_real_play]

    merged = part.merge(
        third_down, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame()

    # One row per (player on field, third-down play) = one third-down snap.
    exploded = merged.assign(
        player_id=merged["offense_players"].astype(str).str.split(";")
    ).explode("player_id")
    exploded = exploded[exploded["player_id"].str.startswith("00-", na=False)]

    snaps = exploded.groupby("player_id").size().rename("third_down_snaps")

    # Primary team = whoever the player took the most third-down snaps for,
    # same tie-break convention as build_route_participation_features_v1.py.
    primary_team = (
        exploded.groupby(["player_id", "posteam"]).size().reset_index(name="n")
        .sort_values("n").drop_duplicates("player_id", keep="last")
        .set_index("player_id")["posteam"]
    )
    team_third_down_plays = merged.groupby("posteam").size()

    out = snaps.to_frame().join(primary_team.rename("team"))
    out["team_third_down_plays"] = out["team"].map(team_third_down_plays)
    out["season"] = season
    return out.reset_index()


def build_third_down_dataset(force_refresh: bool = False) -> pd.DataFrame:
    cache_dir = CACHE_DIR / "third_down_participation_by_season"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SEASONS if force_refresh or not (cache_dir / f"{s}.csv").exists()]
    assets = fetch_release_assets(PARTICIPATION_TAG) if missing else []

    for season in missing:
        agg = _aggregate_one_season(season, assets)
        agg.to_csv(cache_dir / f"{season}.csv", index=False)
        print(f"third-down participation {season}: {len(agg)} player rows", flush=True)

    frames = []
    for season in SEASONS:
        path = cache_dir / f"{season}.csv"
        if path.exists():
            got = pd.read_csv(path)
            if not got.empty:
                frames.append(got)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    for col in ["third_down_snaps", "team_third_down_plays"]:
        raw[col] = pd.to_numeric(raw.get(col), errors="coerce")

    enough = raw["third_down_snaps"] >= MIN_THIRD_DOWN_SNAPS_FOR_RATE
    rate = raw["third_down_snaps"] / raw["team_third_down_plays"].replace(0, np.nan)

    out = pd.DataFrame({
        # +1 shift: a 2024 aggregate is a 2025 prior-season input.
        "season": raw["season"] + 1,
        "player_id": raw["player_id"].astype(str),
        "team": raw["team"],
        "prior_third_down_snaps": raw["third_down_snaps"],
        "prior_third_down_snap_share": rate.where(enough),
    })
    return out.drop_duplicates(["season", "player_id"], keep="first").reset_index(drop=True)


def main() -> None:
    dataset = build_third_down_dataset()
    path = VALIDATION_DIR / "data" / "third_down_participation_player_seasons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    print(f"\nThird-down participation dataset written: {path}")
    print(f"Rows: {len(dataset)}")
    if not dataset.empty:
        print(f"Seasons (as prior-season input): {int(dataset.season.min())}-{int(dataset.season.max())}")
        print(f"  prior_third_down_snap_share: {int(dataset['prior_third_down_snap_share'].notna().sum())} non-null")


if __name__ == "__main__":
    main()
