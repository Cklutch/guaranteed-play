from __future__ import annotations

import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets, read_remote_csv
from validation_utils import VALIDATION_DIR, clean_name

DRAFT_PICKS_TAG = "draft_picks"

# Unlike snap share / red zone / vacated volume (which vary by season and
# get shifted +1 as a "prior season" feature), draft capital is a single,
# fixed, career-long attribute -- a player's draft round/pick never changes
# season to season. No prior-season shift is applied here; the join step
# in build_predraft_dataset.py-style chains should attach these columns to
# every season-row for that player unchanged, and separately compute
# years_since_drafted = season - draft_season per row.


def _fetch_draft_picks(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "raw_draft_picks.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    assets = fetch_release_assets(DRAFT_PICKS_TAG)
    matches = [a for a in assets if str(a.get("name", "")) == "draft_picks.csv"]
    if not matches:
        return pd.DataFrame()
    df = read_remote_csv(matches[0])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def build_draft_capital_dataset() -> pd.DataFrame:
    raw = _fetch_draft_picks()
    if raw.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_key", "player_name", "draft_season", "draft_round", "draft_pick_overall",
        ])

    df = raw[raw["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df["player_id"] = df["gsis_id"]
    df["player_key"] = df["pfr_player_name"].apply(clean_name)
    out = df.rename(columns={
        "season": "draft_season",
        "round": "draft_round",
        "pick": "draft_pick_overall",
        "pfr_player_name": "player_name",
    })
    out = out[["player_id", "player_key", "player_name", "draft_season", "draft_round", "draft_pick_overall"]]
    # A handful of players are drafted more than once across different
    # source rows in edge cases (rare data quirks); keep the first (should
    # be unique in practice since gsis_id is a stable per-player ID).
    out = out.drop_duplicates(["player_id"], keep="first")
    return out.reset_index(drop=True)


def main() -> None:
    dataset = build_draft_capital_dataset()
    output_path = VALIDATION_DIR / "data" / "draft_capital_player_seasons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Draft capital dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Rows with player_id (gsis_id): {int(dataset['player_id'].notna().sum())}")
    print(f"Draft seasons covered: {sorted(dataset['draft_season'].dropna().unique().tolist())[:5]}...{sorted(dataset['draft_season'].dropna().unique().tolist())[-5:]}")


if __name__ == "__main__":
    main()
