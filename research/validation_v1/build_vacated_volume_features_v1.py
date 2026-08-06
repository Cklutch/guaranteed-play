from __future__ import annotations

import numpy as np
import pandas as pd

from nflverse_fetch import CACHE_DIR, fetch_release_assets, read_remote_csv
from validation_utils import VALIDATION_DIR, clean_name

STATS_PLAYER_TAG = "stats_player"
SEASONS = tuple(range(1999, 2026))

# This is intentionally a TEAM-level context feature, not "how many of the
# vacated targets did this specific player absorb." The latter would use
# the player's own season-S+1 realized target share, which is a same-season
# actual -- exactly the leakage the rest of this project's prior_* columns
# are careful to avoid. What's genuinely pre-draft-safe is "how much of this
# team's target/touch volume became vacant" -- real, knowable from the
# roster/depth chart before the season, same as vacated_targets/
# vacated_touches were already stubbed for in build_predraft_dataset.py.


ROSTERS_TAG = "rosters"
_INACTIVE_ROSTER_STATUSES = {"RET", "RES", "CUT"}


def _next_season_roster_from_rosters_tag(season: int) -> set:
    cache_path = CACHE_DIR / f"roster_{season}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        assets = fetch_release_assets(ROSTERS_TAG)
        matches = [a for a in assets if str(a.get("name", "")) == f"roster_{season}.csv"]
        if not matches:
            return set()
        df = read_remote_csv(matches[0])
        df.to_csv(cache_path, index=False)
    if df.empty or "gsis_id" not in df.columns or "team" not in df.columns:
        return set()
    active = df[~df.get("status", "").astype(str).isin(_INACTIVE_ROSTER_STATUSES)]
    return set(zip(active["team"], active["gsis_id"]))


def _fetch_stats_player_reg_season(season: int) -> pd.DataFrame:
    assets = fetch_release_assets(STATS_PLAYER_TAG)
    matches = [a for a in assets if str(a.get("name", "")) == f"stats_player_reg_{season}.csv"]
    if not matches:
        return pd.DataFrame()
    return read_remote_csv(matches[0])


def _pull_all_seasons(force_refresh: bool = False) -> pd.DataFrame:
    season_cache_dir = CACHE_DIR / "stats_player_reg_by_season"
    season_cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in SEASONS:
        season_path = season_cache_dir / f"{season}.csv"
        if season_path.exists() and not force_refresh:
            df = pd.read_csv(season_path)
        else:
            df = _fetch_stats_player_reg_season(season)
            df.to_csv(season_path, index=False)
            print(f"stats_player_reg {season}: {len(df)} player rows")
        if not df.empty:
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_vacated_volume_dataset() -> pd.DataFrame:
    raw = _pull_all_seasons()
    if raw.empty:
        return pd.DataFrame(columns=[
            "season", "team", "prior_team_vacated_target_share", "prior_team_vacated_carry_share",
        ])

    df = raw[raw["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df["targets"] = pd.to_numeric(df.get("targets"), errors="coerce").fillna(0.0)
    df["carries"] = pd.to_numeric(df.get("carries"), errors="coerce").fillna(0.0)
    df["team"] = df["recent_team"]

    team_totals = df.groupby(["season", "team"]).agg(
        team_targets=("targets", "sum"),
        team_carries=("carries", "sum"),
    ).reset_index()

    # Roster presence per (season, team, player) to detect departures.
    roster = df[["season", "team", "player_id"]].drop_duplicates()
    all_seasons = sorted(df["season"].dropna().unique())
    max_stats_season = max(all_seasons)

    rows = []
    for season in all_seasons:
        next_season = season + 1
        this_year = df[df["season"] == season]

        if next_season > max_stats_season:
            # No stats_player_reg file exists yet for next_season (it hasn't
            # been played) -- every player would look "departed" by default,
            # which is an artifact, not signal. Use the rosters tag's real
            # current team assignments for this boundary season instead.
            next_roster = _next_season_roster_from_rosters_tag(next_season)
        else:
            next_roster = set(
                zip(
                    roster.loc[roster["season"] == next_season, "team"],
                    roster.loc[roster["season"] == next_season, "player_id"],
                )
            )

        for team, team_df in this_year.groupby("team"):
            departed = team_df[~team_df.apply(lambda r: (team, r["player_id"]) in next_roster, axis=1)]
            vacated_targets = departed["targets"].sum()
            vacated_carries = departed["carries"].sum()
            rows.append({
                "season": next_season,
                "team": team,
                "team_vacated_targets": vacated_targets,
                "team_vacated_carries": vacated_carries,
            })

    vacated = pd.DataFrame(rows)

    # vacated["season"] is already season S+1 (the draft-relevant season),
    # so match team_totals at season S = vacated["season"] - 1 to get the
    # share of that prior season's team volume that departed.
    prior_totals = team_totals.rename(columns={"season": "season_minus_1"})
    vacated["season_minus_1"] = vacated["season"] - 1
    vacated = vacated.merge(prior_totals, on=["season_minus_1", "team"], how="left")

    vacated["prior_team_vacated_target_share"] = (
        vacated["team_vacated_targets"] / vacated["team_targets"].replace(0, np.nan)
    )
    vacated["prior_team_vacated_carry_share"] = (
        vacated["team_vacated_carries"] / vacated["team_carries"].replace(0, np.nan)
    )

    out = vacated[["season", "team", "prior_team_vacated_target_share", "prior_team_vacated_carry_share"]].copy()
    return out.reset_index(drop=True)


def main() -> None:
    dataset = build_vacated_volume_dataset()
    output_path = VALIDATION_DIR / "data" / "vacated_volume_team_seasons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Vacated volume dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Non-null prior_team_vacated_target_share: {int(dataset['prior_team_vacated_target_share'].notna().sum())}")


if __name__ == "__main__":
    main()
