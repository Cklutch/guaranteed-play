"""
Step 11: garbage-time-ADJUSTED prior production.

This is deliberately NOT a garbage-time predictor. Persistence was tested
first and the predictive version is dead: year-over-year correlation of
garbage-time share is r=+0.166 at player level and +0.287 at team level
(2018-2025), against r~0.6-0.7 for genuinely sticky metrics like target
share. Knowing a player saw garbage time last year tells you almost
nothing about next year, so "he'll get garbage time again" is not a
feature worth building.

What survives is a different claim: garbage time inflates the PRIOR-SEASON
production that already feeds the model as a talent/role signal. Trey
McBride took 35.9% of his 170 targets with win probability outside the
10-90% band -- 3rd highest of 79 receivers with 70+ targets, on an Arizona
offense that ran 33.8% of its pass plays in garbage time (3rd worst in the
league). His raw line overstates his real role. Cleaning that is a
MEASUREMENT CORRECTION on a historical input, and it holds whether or not
2026 repeats the game script.

So the single question this script exists to answer:

    Does garbage-ADJUSTED prior production predict next-season outcomes
    better than RAW prior production?

Tested head-to-head as `adp_prior_production_ADJUSTED` against the
otherwise-identical `adp_prior_production_baseline`. Pass or fail, nothing
else is claimed.

Leakage: every output is shifted +1 season, so a 2024 aggregate becomes a
2025 prior_* input. Same convention as the rest of the chain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets
from validation_utils import VALIDATION_DIR, clean_name, initial_last_key

PBP_TAG = "pbp"
# Start at 2010 -- the harness only has real ADP from 2010 on, so earlier
# seasons cannot be evaluated against the market baseline anyway.
SEASONS = tuple(range(2010, 2026))

# Conventional public definition of garbage time: win probability outside
# the 10-90% band. Not tuned on this dataset -- same discipline as the
# 15/20-yard explosive-play cutoffs in the redzone script.
GARBAGE_WP_LOW = 0.10
GARBAGE_WP_HIGH = 0.90

PBP_USECOLS = [
    "season", "season_type", "posteam", "wp",
    "pass_attempt", "rush_attempt",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
    "receiving_yards", "rushing_yards", "yards_gained",
    "pass_touchdown", "rush_touchdown",
]


def _season_url(season: int, assets: list) -> str | None:
    for asset in assets:
        if str(asset.get("name", "")) == f"play_by_play_{season}.csv.gz":
            return asset.get("browser_download_url")
    return None


def _aggregate_one_season(season: int, assets: list) -> pd.DataFrame:
    url = _season_url(season, assets)
    if url is None:
        return pd.DataFrame()

    df = pd.read_csv(url, usecols=lambda c: c in PBP_USECOLS, low_memory=False)
    df = df[df["season_type"].astype(str).eq("REG")].copy()

    wp = pd.to_numeric(df.get("wp"), errors="coerce")
    df["garbage"] = (wp < GARBAGE_WP_LOW) | (wp > GARBAGE_WP_HIGH)
    # Plays with no win-probability value can't be classified; treating them
    # as competitive (rather than dropping) keeps the denominator honest.
    df["garbage"] = df["garbage"].fillna(False)
    live = df[~df["garbage"]]

    frames = []

    # --- Receiving ---
    def _rec(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        t = frame[frame["pass_attempt"].fillna(0).astype(int).eq(1) & frame["receiver_player_id"].notna()].copy()
        if t.empty:
            return pd.DataFrame()
        yards = pd.to_numeric(t.get("receiving_yards"), errors="coerce")
        if yards.isna().all():
            yards = pd.to_numeric(t.get("yards_gained"), errors="coerce")
        t["_yds"] = yards
        return t.groupby(["receiver_player_id", "receiver_player_name"]).agg(**{
            f"targets{suffix}": ("receiver_player_id", "size"),
            f"receiving_yards{suffix}": ("_yds", "sum"),
            f"receiving_tds{suffix}": ("pass_touchdown", "sum"),
        }).reset_index().rename(columns={
            "receiver_player_id": "player_id", "receiver_player_name": "player_name",
        })

    # --- Rushing ---
    def _rush(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        r = frame[frame["rush_attempt"].fillna(0).astype(int).eq(1) & frame["rusher_player_id"].notna()].copy()
        if r.empty:
            return pd.DataFrame()
        yards = pd.to_numeric(r.get("rushing_yards"), errors="coerce")
        if yards.isna().all():
            yards = pd.to_numeric(r.get("yards_gained"), errors="coerce")
        r["_yds"] = yards
        return r.groupby(["rusher_player_id", "rusher_player_name"]).agg(**{
            f"carries{suffix}": ("rusher_player_id", "size"),
            f"rushing_yards{suffix}": ("_yds", "sum"),
            f"rushing_tds{suffix}": ("rush_touchdown", "sum"),
        }).reset_index().rename(columns={
            "rusher_player_id": "player_id", "rusher_player_name": "player_name",
        })

    for frame, suffix in [(df, "_all"), (live, "_live")]:
        for fn in (_rec, _rush):
            got = fn(frame, suffix)
            if not got.empty:
                frames.append(got.set_index(["player_id", "player_name"]))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).reset_index()
    out["season"] = season
    return out


def build_garbage_time_dataset(force_refresh: bool = False) -> pd.DataFrame:
    season_cache_dir = CACHE_DIR / "garbage_time_by_season"
    season_cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SEASONS if force_refresh or not (season_cache_dir / f"{s}.csv").exists()]
    # Fetch the asset listing ONCE -- re-listing per season is what blew the
    # GitHub rate limit while building the redzone script.
    assets = fetch_release_assets(PBP_TAG) if missing else []

    for season in missing:
        agg = _aggregate_one_season(season, assets)
        agg.to_csv(season_cache_dir / f"{season}.csv", index=False)
        print(f"pbp {season}: {len(agg)} player rows", flush=True)

    frames = []
    for season in SEASONS:
        path = season_cache_dir / f"{season}.csv"
        if path.exists():
            got = pd.read_csv(path)
            if not got.empty:
                frames.append(got)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    counting = [
        "targets_all", "receiving_yards_all", "receiving_tds_all",
        "carries_all", "rushing_yards_all", "rushing_tds_all",
        "targets_live", "receiving_yards_live", "receiving_tds_live",
        "carries_live", "rushing_yards_live", "rushing_tds_live",
    ]
    for col in counting:
        if col not in raw.columns:
            raw[col] = 0.0
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        # +1 shift: a 2024 aggregate is a 2025 prior-season input.
        "season": raw["season"] + 1,
        "player_id": raw["player_id"],
        "player_name": raw["player_name"],
    })
    out["player_key"] = raw["player_name"].apply(clean_name)
    out["initial_last_key"] = raw["player_name"].apply(initial_last_key)

    # Garbage-adjusted replacements for the raw prior-production columns.
    out["prior_non_garbage_targets"] = raw["targets_live"]
    out["prior_non_garbage_receiving_yards"] = raw["receiving_yards_live"]
    out["prior_non_garbage_receiving_tds"] = raw["receiving_tds_live"]
    out["prior_non_garbage_carries"] = raw["carries_live"]
    out["prior_non_garbage_rushing_yards"] = raw["rushing_yards_live"]
    out["prior_non_garbage_rushing_tds"] = raw["rushing_tds_live"]

    # Diagnostic only -- the persistence test ruled this out as a predictor,
    # so it is emitted for inspection and NOT added to any feature group.
    total_touches = raw["targets_all"] + raw["carries_all"]
    live_touches = raw["targets_live"] + raw["carries_live"]
    out["prior_garbage_time_share"] = 1 - (live_touches / total_touches.replace(0, np.nan))

    return out.reset_index(drop=True)


def main() -> None:
    dataset = build_garbage_time_dataset()
    path = VALIDATION_DIR / "data" / "garbage_time_player_seasons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    print(f"\nGarbage-time dataset written: {path}")
    print(f"Rows: {len(dataset)}")
    if not dataset.empty:
        print(f"Seasons (as prior-season input): {int(dataset.season.min())}-{int(dataset.season.max())}")
        print(f"Median garbage-time share: {dataset['prior_garbage_time_share'].median():.1%}")


if __name__ == "__main__":
    main()
