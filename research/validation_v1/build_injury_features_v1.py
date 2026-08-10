"""
Step 3b Item 1: missed-games risk / durability feature.

Deliberately "missed games risk," not narrowly "injury": suspensions,
personal-reason absences, and healthy benchings all count the same as
injuries here, since all of them represent a real (and somewhat recurring)
risk that a player isn't on the field -- exactly what this feature is for.
No attempt is made to distinguish cause.

Two nflverse sources:
- `snap_counts` (already cached by build_snap_share_features_v1.py as
  raw_snap_counts_2012_2025.csv): weekly per-player snap data, used to
  compute games_missed_pct (all-cause) and each player's typical role.
- `injuries` release tag (weekly injury report, gsis_id-native): used for
  injury_report_rate, a secondary "nagging issue" signal that doesn't
  require the player to have actually missed a game.

Role gate: a deep backup who's inactive most weeks never had a role to
begin with -- he isn't "missing games," and scoring his absence would read
a non-signal as either fragile or durable at random. So games_missed_pct
only counts for player-seasons where the player's average offense snap %
IN THE GAMES HE DID PLAY clears MIN_SNAP_PCT_FOR_ROLE. Below it, the season
is excluded (composite = NaN) rather than scored -- a low-usage player, not
a durable or fragile one.

Leakage: every input is a prior_* aggregate. The lookback is a weighted
average of the three seasons before the season being predicted (recency
weights 3/2/1), so no outcome data for the target season is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_snap_share_features_v1 import _pull_all_seasons as pull_snap_counts
from nflverse_fetch import CACHE_DIR, fetch_release_assets, pfr_to_gsis_crosswalk, read_remote_csv
from validation_utils import VALIDATION_DIR

INJURIES_TAG = "injuries"
SEASONS = tuple(range(2012, 2026))  # matches snap_counts' start (2012)

# NFL regular-season schedule length by season -- used as the fixed
# denominator for both games_missed_pct and injury_report_rate rather than
# deriving "team games played" from data presence (simpler, and avoids
# treating a team-week with no snap-count rows as a phantom bye week).
TEAM_GAMES_PER_SEASON = {season: (17 if season >= 2021 else 16) for season in SEASONS}

# See module docstring -- flagged as an inference, not a verified cutoff,
# same as the archetype engine's volume gates before they were tightened
# (MIN_CARRIES_FOR_RATE started at 20, had to be raised to 75 after
# inspection). Revisit if spot-checks show it's letting fringe players
# through or excluding real rotational players.
MIN_SNAP_PCT_FOR_ROLE = 0.40

RECENCY_WEIGHTS = {1: 3, 2: 2, 3: 1}  # season offset from target -> weight


def _fetch_injuries_season(season: int, assets: list) -> pd.DataFrame:
    matches = [a for a in assets if str(a.get("name", "")) == f"injuries_{season}.csv"]
    if not matches:
        return pd.DataFrame()
    return read_remote_csv(matches[0])


def _pull_injuries(force_refresh: bool = False) -> pd.DataFrame:
    # Per-season cache files, not one combined cache -- same rate-limit fix
    # as build_redzone_airyards_features_v1.py: fetch the release's asset
    # listing exactly once, then reuse it across the per-season loop.
    season_cache_dir = CACHE_DIR / "injuries_by_season"
    season_cache_dir.mkdir(parents=True, exist_ok=True)

    missing_seasons = [
        season for season in SEASONS
        if force_refresh or not (season_cache_dir / f"{season}.csv").exists()
    ]
    assets = fetch_release_assets(INJURIES_TAG) if missing_seasons else []

    for season in missing_seasons:
        df = _fetch_injuries_season(season, assets)
        df.to_csv(season_cache_dir / f"{season}.csv", index=False)
        print(f"injuries {season}: {len(df)} report rows")

    frames = []
    for season in SEASONS:
        path = season_cache_dir / f"{season}.csv"
        if path.exists():
            season_df = pd.read_csv(path)
            if not season_df.empty:
                frames.append(season_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _role_and_missed_games(snap_raw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, pfr_player_id): games_missed_pct and the role gate."""
    df = snap_raw[snap_raw["game_type"].astype(str).eq("REG")].copy()
    for col in ["offense_snaps", "offense_pct", "defense_snaps", "st_snaps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["appeared"] = (df["offense_snaps"] > 0) | (df["defense_snaps"] > 0) | (df["st_snaps"] > 0)
    active = df[df["appeared"]].copy()

    games_appeared = (
        active.groupby(["season", "pfr_player_id", "player", "position", "team"])["game_id"]
        .nunique().reset_index().rename(columns={"game_id": "games_appeared"})
    )
    avg_snap_pct = (
        active.groupby(["season", "pfr_player_id"])["offense_pct"]
        .mean().reset_index().rename(columns={"offense_pct": "avg_snap_pct_when_active"})
    )
    out = games_appeared.merge(avg_snap_pct, on=["season", "pfr_player_id"], how="left")

    out["team_games_played"] = out["season"].map(TEAM_GAMES_PER_SEASON)
    out["games_missed_pct"] = 1 - (out["games_appeared"] / out["team_games_played"])
    out["role_qualified"] = out["avg_snap_pct_when_active"] >= MIN_SNAP_PCT_FOR_ROLE
    return out


def _injury_report_rate(injuries_raw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, gsis_id): share of team games with any injury-report entry."""
    if injuries_raw.empty:
        return pd.DataFrame(columns=["season", "gsis_id", "injury_report_rate"])
    df = injuries_raw[injuries_raw["game_type"].astype(str).eq("REG")].copy()
    df["on_report"] = df["report_status"].notna() & (df["report_status"].astype(str).str.strip() != "")

    report_weeks = (
        df[df["on_report"]].groupby(["season", "gsis_id"])["week"]
        .nunique().reset_index().rename(columns={"week": "report_weeks"})
    )
    report_weeks["team_games_played"] = report_weeks["season"].map(TEAM_GAMES_PER_SEASON)
    report_weeks["injury_report_rate"] = report_weeks["report_weeks"] / report_weeks["team_games_played"]
    return report_weeks[["season", "gsis_id", "injury_report_rate"]]


def _weighted_lookback(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, build every target season (= an observed season + 1)
    and weight-average the composite from the 3 seasons before it
    (recency weights 3/2/1). Missing/non-qualifying seasons are skipped and
    the remaining weights renormalized -- same pattern as _squash() in
    build_archetypes_v1.py. A target season with zero qualifying seasons in
    its 3-year window gets NaN (rookie, gap year, or career low-usage
    player), not a fabricated average.
    """
    base = player_seasons[["player_id", "season", "composite", "position", "team", "player_name"]].copy()
    base = base.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season"])

    targets = base[["player_id", "season"]].copy()
    targets["target_season"] = targets["season"] + 1
    targets = targets[["player_id", "target_season"]].drop_duplicates().reset_index(drop=True)

    offset_cols = []
    for offset, weight in RECENCY_WEIGHTS.items():
        col = f"c_offset{offset}"
        offset_cols.append(col)
        targets[f"lookup_season_{offset}"] = targets["target_season"] - offset
        lag = base[["player_id", "season", "composite"]].rename(
            columns={"season": f"lookup_season_{offset}", "composite": col}
        )
        targets = targets.merge(lag, on=["player_id", f"lookup_season_{offset}"], how="left")

    weight_arr = np.array([RECENCY_WEIGHTS[1], RECENCY_WEIGHTS[2], RECENCY_WEIGHTS[3]])
    vals = targets[offset_cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    weighted_sum = np.nansum(vals * weight_arr, axis=1)
    weight_total = (mask * weight_arr).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        targets["prior_injury_composite"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)

    # Carry player_name/position/team from the most recent season in the
    # window (offset 1) -- most reliable identity/position for the target.
    identity = base[["player_id", "season", "player_name", "position", "team"]].rename(
        columns={"season": "lookup_season_1"}
    )
    targets = targets.merge(identity, on=["player_id", "lookup_season_1"], how="left")

    return targets[["player_id", "target_season", "player_name", "position", "team", "prior_injury_composite"]]


def build_injury_dataset() -> pd.DataFrame:
    snap_raw = pull_snap_counts()
    role_and_missed = _role_and_missed_games(snap_raw)

    crosswalk = pfr_to_gsis_crosswalk(SEASONS)
    if not crosswalk.empty:
        role_and_missed = role_and_missed.merge(
            crosswalk.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"}),
            on="pfr_player_id", how="left",
        )
    else:
        role_and_missed["player_id"] = pd.NA

    injuries_raw = _pull_injuries()
    report_rate = _injury_report_rate(injuries_raw)
    report_rate = report_rate.rename(columns={"gsis_id": "player_id"})
    report_rate["player_id"] = report_rate["player_id"].astype(str)

    role_and_missed["player_id"] = role_and_missed["player_id"].astype(str)
    merged = role_and_missed.merge(report_rate, on=["season", "player_id"], how="left")
    # A role-qualified player with no injury-report rows that season really
    # was never on the report -- 0, not missing.
    merged.loc[merged["role_qualified"], "injury_report_rate"] = merged.loc[
        merged["role_qualified"], "injury_report_rate"
    ].fillna(0.0)

    merged["composite"] = np.where(
        merged["role_qualified"],
        0.6 * merged["games_missed_pct"] + 0.4 * merged["injury_report_rate"],
        np.nan,
    )
    merged = merged.rename(columns={"player": "player_name"})

    lookback = _weighted_lookback(merged)

    # Z-score within (target_season, position), then invert sign so higher
    # = more durable / lower risk (the raw composite is a risk score).
    grouped = lookback.groupby(["target_season", "position"])["prior_injury_composite"]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (lookback["prior_injury_composite"] - mean) / std.replace(0, np.nan)
    lookback["prior_durability_score"] = -1 * z

    out = lookback.rename(columns={"target_season": "season"})
    out = out[["season", "player_id", "player_name", "position", "team", "prior_durability_score"]]
    return out.reset_index(drop=True)


def main() -> None:
    dataset = build_injury_dataset()
    output_path = VALIDATION_DIR / "data" / "injury_durability_player_seasons.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Injury/durability dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Non-null prior_durability_score: {int(dataset['prior_durability_score'].notna().sum())}")
    print(f"Seasons covered: {sorted(dataset['season'].dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
