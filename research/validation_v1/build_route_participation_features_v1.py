"""
Step 12: route participation and targets-per-route-run (TPRR).

The design doc flagged route participation as the highest-value missing
input, and unlike most of what has been tried here it clears its gate
before any modeling. Year-over-year persistence of route participation
rate, measured 2019-2024 on players with 100+ routes:

    2019->2020 r=+0.503   2022->2023 r=+0.523
    2020->2021 r=+0.463   2023->2024 r=+0.582
    2021->2022 r=+0.487   MEAN      r=+0.511

That is ~3x garbage-time share (+0.166, rejected on exactly this test) and
approaching the +0.6-0.7 of genuinely sticky metrics -- which makes sense,
since route participation reflects how a coach deploys a player rather
than game script.

Why it can carry information ADP does not already price: TPRR separates
"efficient on limited routes" from "runs every route and still draws few
targets." A stat line shows targets and yards; it does not show the
denominator. That is the same property that made garbage time worth
testing, except route rate actually persists.

Source: nflverse `pbp_participation` (2016-2025), whose `offense_players`
column is a semicolon-delimited list of **gsis_ids** on every play. That is
a native ID join -- no crosswalk and no ambiguity guard, unlike snap share
(13.8% name collisions) or the pbp aggregates (PFR-abbreviated names).

Leakage: outputs are shifted +1 season, so a 2024 aggregate becomes a 2025
prior_* input. Same convention as the rest of the chain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets
from validation_utils import VALIDATION_DIR

PARTICIPATION_TAG = "pbp_participation"
# Participation data starts in 2016, so prior-season inputs start 2017.
SEASONS = tuple(range(2016, 2026))

# A rate needs a real denominator behind it. 100 routes is roughly a
# sixth of a full-time season -- below that, TPRR is noise. Same discipline
# as MIN_TARGETS_FOR_RATE / MIN_CARRIES_FOR_RATE in the archetype engine,
# which had to be raised once after backups polluted long_run_rb.
MIN_ROUTES_FOR_RATE = 100


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
            "pass_attempt", "receiver_player_id", "receiving_yards", "yards_gained",
        },
        low_memory=False, compression="gzip",
    )
    pbp = pbp[pbp["season_type"].astype(str).eq("REG")].copy()
    passes = pbp[pbp["pass_attempt"].fillna(0).astype(int).eq(1)]

    merged = part.merge(
        passes, left_on=["nflverse_game_id", "play_id"], right_on=["game_id", "play_id"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame()

    # One row per (player on field, pass play) = one route opportunity.
    exploded = merged.assign(
        player_id=merged["offense_players"].astype(str).str.split(";")
    ).explode("player_id")
    exploded = exploded[exploded["player_id"].str.startswith("00-", na=False)]

    routes = exploded.groupby("player_id").size().rename("routes_run")

    # Primary team = whoever the player ran the most routes for, so a
    # midseason trade resolves to one team rather than splitting the row.
    primary_team = (
        exploded.groupby(["player_id", "posteam"]).size().reset_index(name="n")
        .sort_values("n").drop_duplicates("player_id", keep="last")
        .set_index("player_id")["posteam"]
    )
    team_pass_plays = merged.groupby("posteam").size()

    # Targets/yards come from the pass plays themselves, not the
    # participation list -- same underlying rows, so the ratio is coherent.
    tgt = passes[passes["receiver_player_id"].notna()].copy()
    yards = pd.to_numeric(tgt.get("receiving_yards"), errors="coerce")
    if yards.isna().all():
        yards = pd.to_numeric(tgt.get("yards_gained"), errors="coerce")
    tgt["_yds"] = yards
    rec = tgt.groupby("receiver_player_id").agg(
        targets=("receiver_player_id", "size"), receiving_yards=("_yds", "sum")
    )

    out = routes.to_frame().join(primary_team.rename("team")).join(rec)
    out["team_pass_plays"] = out["team"].map(team_pass_plays)
    out["season"] = season
    return out.reset_index()


def build_route_dataset(force_refresh: bool = False) -> pd.DataFrame:
    cache_dir = CACHE_DIR / "route_participation_by_season"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SEASONS if force_refresh or not (cache_dir / f"{s}.csv").exists()]
    # Asset listing fetched ONCE -- re-listing per season is what tripped the
    # GitHub rate limit while building the redzone script.
    assets = fetch_release_assets(PARTICIPATION_TAG) if missing else []

    for season in missing:
        agg = _aggregate_one_season(season, assets)
        agg.to_csv(cache_dir / f"{season}.csv", index=False)
        print(f"participation {season}: {len(agg)} player rows", flush=True)

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
    for col in ["routes_run", "targets", "receiving_yards", "team_pass_plays"]:
        raw[col] = pd.to_numeric(raw.get(col), errors="coerce")
    raw["targets"] = raw["targets"].fillna(0.0)
    raw["receiving_yards"] = raw["receiving_yards"].fillna(0.0)

    enough = raw["routes_run"] >= MIN_ROUTES_FOR_RATE
    routes = raw["routes_run"].replace(0, np.nan)

    out = pd.DataFrame({
        # +1 shift: a 2024 aggregate is a 2025 prior-season input.
        "season": raw["season"] + 1,
        "player_id": raw["player_id"].astype(str),
        "team": raw["team"],
        "prior_routes_run": raw["routes_run"],
        "prior_route_participation_rate": raw["routes_run"] / raw["team_pass_plays"].replace(0, np.nan),
    })
    # Rates only where the denominator supports them; below the gate they
    # stay NaN rather than becoming confident-looking noise.
    out["prior_targets_per_route_run"] = (raw["targets"] / routes).where(enough)
    out["prior_yards_per_route_run"] = (raw["receiving_yards"] / routes).where(enough)

    return out.drop_duplicates(["season", "player_id"], keep="first").reset_index(drop=True)


def main() -> None:
    dataset = build_route_dataset()
    path = VALIDATION_DIR / "data" / "route_participation_player_seasons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    print(f"\nRoute participation dataset written: {path}")
    print(f"Rows: {len(dataset)}")
    if not dataset.empty:
        print(f"Seasons (as prior-season input): {int(dataset.season.min())}-{int(dataset.season.max())}")
        for col in ["prior_route_participation_rate", "prior_targets_per_route_run", "prior_yards_per_route_run"]:
            print(f"  {col}: {int(dataset[col].notna().sum())} non-null")


if __name__ == "__main__":
    main()
