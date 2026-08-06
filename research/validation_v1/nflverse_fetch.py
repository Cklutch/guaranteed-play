"""
Shared nflverse GitHub Releases fetch helpers for research/validation_v1
feature-ingestion scripts.

Re-exports the real, working fetch/pagination logic already built in
case_studies/rb_elite_age_analysis.py (_fetch_release_assets,
_csv_assets_for_years, _read_remote_csv) rather than duplicating it --
every build_*_features_v1.py script in this directory should import from
here instead of re-implementing GitHub Releases pagination.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from case_studies.rb_elite_age_analysis import (  # noqa: E402
    _csv_assets_for_years as csv_assets_for_years,
    _fetch_release_assets as fetch_release_assets,
    _read_remote_csv as read_remote_csv,
)

CACHE_DIR = Path(__file__).resolve().parent / "data"


def pfr_to_gsis_crosswalk(seasons, force_refresh: bool = False):
    """
    Map nflverse `pfr_id` -> `gsis_id` using the rosters release tag.

    Needed because snap_counts is keyed by pfr_player_id while every other
    dataset in this project is keyed by gsis_id. Joining those two on player
    NAME instead collides real players (measured: 13.8% of snap-share rows
    hit an initial+lastname collision, e.g. 2017 D.Johnson (ARI) vs
    D.Johnson Jr. (CLE) silently sharing one player's usage).
    """
    import pandas as pd

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "pfr_gsis_crosswalk.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    assets = fetch_release_assets("rosters")
    frames = []
    for season in seasons:
        matches = [a for a in assets if str(a.get("name", "")) == f"roster_{season}.csv"]
        if not matches:
            continue
        df = read_remote_csv(matches[0])
        if df.empty or "pfr_id" not in df.columns or "gsis_id" not in df.columns:
            continue
        frames.append(df[["pfr_id", "gsis_id"]].dropna())

    if not frames:
        return pd.DataFrame(columns=["pfr_id", "gsis_id"])

    crosswalk = pd.concat(frames, ignore_index=True).drop_duplicates("pfr_id", keep="first")
    crosswalk.to_csv(cache_path, index=False)
    return crosswalk


def cached_or_fetch(cache_name: str, build_fn, force_refresh: bool = False):
    """
    Same cache-if-exists-else-fetch-then-write pattern already used by
    case_studies' pull_historical_seasons_df()/pull_wr_opportunity_df().
    """
    import pandas as pd

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)
    df = build_fn()
    df.to_csv(cache_path, index=False)
    return df
