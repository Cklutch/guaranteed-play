from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from build_predraft_dataset import build_dataset
from validation_utils import VALIDATION_DIR, clean_name, initial_last_key

DATA_DIR = VALIDATION_DIR / "data"

NEW_WORKLOAD_COLS = [
    "prior_snap_share",
    "draft_round", "draft_pick_overall", "years_since_drafted",
    "prior_redzone_target_share", "prior_redzone_carry_share",
    "prior_air_yards", "prior_air_yards_share",
    "prior_team_vacated_target_share", "prior_team_vacated_carry_share",
    "prior_wopr",
    "prior_durability_score",
]


def _load_snap_share() -> pd.DataFrame:
    """
    Snap share, split into an ID-keyed frame (preferred) and a name-keyed
    fallback frame for rows the pfr_id -> gsis_id crosswalk doesn't cover.

    Name-only joining was measurably wrong here: 13.8% of snap-share rows
    hit an initial+lastname collision (e.g. 2017 D.Johnson (ARI) and
    D.Johnson Jr. (CLE) sharing one player's usage), so the ID path is used
    wherever a gsis_id exists and names are only a last resort.
    """
    path = DATA_DIR / "snap_share_player_seasons.csv"
    empty = pd.DataFrame(columns=["season", "player_id", "prior_snap_share", "team"])
    if not path.exists():
        return empty, empty
    df = pd.read_csv(path)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)

    by_id = df[df["player_id"].notna()].copy()
    by_id["player_id"] = by_id["player_id"].astype(str)
    by_id = by_id.sort_values("prior_snap_share", na_position="first")
    by_id = by_id.drop_duplicates(["season", "player_id"], keep="last")
    by_id = by_id[["season", "player_id", "prior_snap_share", "team"]]

    by_name = df[df["player_id"].isna()].copy()
    by_name = by_name.sort_values("prior_snap_share", na_position="first")
    by_name = by_name.drop_duplicates(["season", "initial_last_key"], keep="last")
    by_name = by_name[["season", "initial_last_key", "prior_snap_share", "team"]]

    return by_id, by_name


def _load_draft_capital() -> pd.DataFrame:
    path = DATA_DIR / "draft_capital_player_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "player_key", "draft_season", "draft_round", "draft_pick_overall"])
    df = pd.read_csv(path)
    return df[["player_id", "player_key", "draft_season", "draft_round", "draft_pick_overall"]]


def _load_redzone_airyards() -> pd.DataFrame:
    path = DATA_DIR / "redzone_airyards_player_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "season", "initial_last_key", "team",
            "prior_redzone_target_share", "prior_redzone_carry_share",
            "prior_air_yards", "prior_air_yards_share",
        ])
    df = pd.read_csv(path)
    # Joined on (season, initial_last_key, team), NOT player_key -- pbp's
    # receiver/rusher names are PFR-abbreviated ("J.Jefferson"), which
    # clean_name() cannot match against the base dataset's full names
    # ("Justin Jefferson"). See build_redzone_airyards_features_v1.py.
    # prior_targets_pbp / prior_carries_pbp are the real per-player volume
    # counts from play-by-play. Carried through because the base dataset's
    # own prior_receptions/prior_carries columns are never populated (0 rows)
    # and prior_targets only covers the 1.5k WR-opportunity rows -- these pbp
    # counts are the only broad-coverage volume denominators available for
    # the Step 4b efficiency ratios (yards per target, yards per carry).
    cols = [
        "season", "initial_last_key", "team",
        "prior_redzone_target_share", "prior_redzone_carry_share",
        "prior_air_yards", "prior_air_yards_share",
        "prior_targets_pbp", "prior_carries_pbp",
        # Play-level rates used by the archetype engine: aDOT separates deep
        # threats from possession receivers, explosive rates are the
        # big-play / "long run" dimension.
        "prior_adot", "prior_explosive_run_rate", "prior_explosive_rec_rate",
        "prior_yards_per_carry",
    ]
    df = df[cols].copy()
    # A handful of (season, initial_last_key, team) combos can collide
    # (e.g. two same-initial teammates) -- keep the row with more usable
    # signal (non-null air yards share) to reduce which row "wins" being
    # arbitrary.
    df = df.sort_values("prior_air_yards_share", na_position="last")
    df = df.drop_duplicates(["season", "initial_last_key", "team"], keep="last")
    return df


def _load_vacated_volume() -> pd.DataFrame:
    path = DATA_DIR / "vacated_volume_team_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "team", "prior_team_vacated_target_share", "prior_team_vacated_carry_share"])
    return pd.read_csv(path)


def _load_garbage_time() -> pd.DataFrame:
    """
    Garbage-time-adjusted production, keyed on gsis_id.

    pbp's receiver/rusher player_id IS a gsis_id, so this joins on a real
    stable key rather than the abbreviated names the redzone aggregation is
    stuck with -- no crosswalk, no collision guard needed.
    """
    path = DATA_DIR / "garbage_time_player_seasons.csv"
    cols = [
        "prior_non_garbage_targets", "prior_non_garbage_receiving_yards",
        "prior_non_garbage_receiving_tds", "prior_non_garbage_carries",
        "prior_non_garbage_rushing_yards", "prior_non_garbage_rushing_tds",
        "prior_garbage_time_share",
    ]
    if not path.exists():
        return pd.DataFrame(columns=["season", "player_id"] + cols)
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)
    keep = ["season", "player_id"] + [c for c in cols if c in df.columns]
    # A traded player can appear twice in a season; keep the higher-volume row.
    df = df.sort_values("prior_non_garbage_targets", na_position="first")
    return df[keep].drop_duplicates(["season", "player_id"], keep="last")


def _load_route_participation() -> pd.DataFrame:
    """
    Route participation and targets/yards per route run, keyed on gsis_id.

    `pbp_participation.offense_players` is a list of gsis_ids, so this joins
    on a real stable key -- no crosswalk and no ambiguity guard, unlike snap
    share (13.8% name collisions) or the pbp aggregates (abbreviated names).

    Note the source rows include offensive linemen (everyone on the field
    for a pass play). They simply don't match a QB/RB/WR/TE in the base
    dataset and drop out of the join, so no explicit filter is needed.
    """
    path = DATA_DIR / "route_participation_player_seasons.csv"
    cols = [
        "prior_routes_run", "prior_route_participation_rate",
        "prior_targets_per_route_run", "prior_yards_per_route_run",
    ]
    if not path.exists():
        return pd.DataFrame(columns=["season", "player_id"] + cols)
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)
    keep = ["season", "player_id"] + [c for c in cols if c in df.columns]
    df = df.sort_values("prior_routes_run", na_position="first")
    return df[keep].drop_duplicates(["season", "player_id"], keep="last")


def _load_offense_environment() -> pd.DataFrame:
    """
    Team offensive environment: prior-season LEVELS plus deviation from the
    team's own trailing 3-year baseline. Joined on (season, team), the same
    team-level treatment as prior_team_vacated_*.
    """
    path = DATA_DIR / "offense_environment_team_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "team"])
    df = pd.read_csv(path)
    df["team"] = df["team"].astype(str)
    return df.drop_duplicates(["season", "team"], keep="first")


def _load_injury_durability() -> pd.DataFrame:
    path = DATA_DIR / "injury_durability_player_seasons.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "player_id", "prior_durability_score"])
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)
    return df[["season", "player_id", "prior_durability_score"]]


def build_workload_dataset(base: pd.DataFrame | None = None) -> pd.DataFrame:
    if base is None:
        base, _metadata = build_dataset()
    dataset = base.copy()
    if "player_key" not in dataset.columns:
        dataset["player_key"] = dataset["player_name"].apply(clean_name)
    # The base dataset uses PFR-style abbreviated names ("M.Faulk") for
    # player_name, same as pbp/stats_player -- initial_last_key is the join
    # key that actually works against every new source in this chain
    # (clean_name()/player_key only matches sources that already have full
    # names, which none of these new nflverse-derived ones do).
    dataset["initial_last_key"] = dataset["player_name"].apply(initial_last_key)
    # The base dataset's own "team" column (from IDENTITY_COLS) is always
    # empty -- the underlying outcome source CSV has no team field at all.
    # Same story for "prior_air_yards"/"prior_snap_share": they're already
    # stubbed as always-empty placeholder columns in build_predraft_dataset.
    # py's PRIOR_COLS. Drop all three so the real data merged in below
    # populates them cleanly instead of colliding into _x/_y suffixes.
    dataset = dataset.drop(columns=["team", "prior_air_yards", "prior_snap_share"], errors="ignore")

    # --- Snap share; also the first source of real per-player-season team
    # assignment in this chain. Joined by gsis_id where available, falling
    # back to the name key only for rows the crosswalk misses. ---
    snap_by_id, snap_by_name = _load_snap_share()
    dataset["player_id"] = dataset["player_id"].astype(str)
    dataset = dataset.merge(snap_by_id, on=["season", "player_id"], how="left")

    if not snap_by_name.empty:
        fallback = dataset["prior_snap_share"].isna()
        filled = dataset.loc[fallback, ["season", "initial_last_key"]].merge(
            snap_by_name, on=["season", "initial_last_key"], how="left"
        )
        dataset.loc[fallback, "prior_snap_share"] = filled["prior_snap_share"].to_numpy()
        dataset.loc[fallback, "team"] = filled["team"].to_numpy()

    # --- Draft capital (player_id join -- gsis_id is a stable, reliable
    # key shared between the outcome dataset and nflverse's draft_picks
    # tag, unlike the name-based joins everything else here needs). Draft
    # capital is a fixed career attribute, so it's joined WITHOUT a season
    # match, then years_since_drafted is computed per row. ---
    draft_capital = _load_draft_capital().drop(columns=["player_key"])
    dataset["player_id"] = dataset["player_id"].astype(str)
    draft_capital["player_id"] = draft_capital["player_id"].astype(str)
    dataset = dataset.merge(draft_capital, on="player_id", how="left")
    dataset["years_since_drafted"] = dataset["season"] - dataset["draft_season"]

    # --- Red zone usage / air yards (season + initial_last_key + team join
    # -- see _load_redzone_airyards() for why player_key doesn't work here). ---
    redzone = _load_redzone_airyards()
    if "team" in dataset.columns:
        dataset["team"] = dataset["team"].astype(str)
        redzone = redzone.copy()
        redzone["team"] = redzone["team"].astype(str)
        dataset = dataset.merge(redzone, on=["season", "initial_last_key", "team"], how="left")
    else:
        for col in ["prior_redzone_target_share", "prior_redzone_carry_share", "prior_air_yards", "prior_air_yards_share"]:
            dataset[col] = np.nan

    # --- Vacated team volume (team + season join -- a team-level context
    # feature shared by every player on that team, not a player-specific
    # realized share; see build_vacated_volume_features_v1.py for why).
    # Uses the `team` column carried in from snap share above, since the
    # base dataset itself has no team column. Rows with no snap-share match
    # (pre-2012 seasons, zero-snap players) simply won't get this feature. ---
    vacated = _load_vacated_volume()
    if "team" in dataset.columns and not vacated.empty:
        dataset["team"] = dataset["team"].astype(str)
        vacated = vacated.copy()
        vacated["team"] = vacated["team"].astype(str)
        dataset = dataset.merge(vacated, on=["season", "team"], how="left")
    else:
        dataset["prior_team_vacated_target_share"] = np.nan
        dataset["prior_team_vacated_carry_share"] = np.nan

    # --- WOPR (Weighted Opportunity Rating), derived, no new ingestion ---
    # Standard formula: 1.5 * target_share + 0.7 * air_yards_share. Uses
    # prior_target_share (already in the base dataset for WR) and the new
    # prior_air_yards_share. Most meaningful for WR/TE; left as NaN for
    # positions/rows missing either input rather than assuming 0.
    target_share = pd.to_numeric(dataset.get("prior_target_share"), errors="coerce")
    air_yards_share = pd.to_numeric(dataset.get("prior_air_yards_share"), errors="coerce")
    dataset["prior_wopr"] = 1.5 * target_share + 0.7 * air_yards_share
    dataset.loc[target_share.isna() & air_yards_share.isna(), "prior_wopr"] = np.nan

    # --- Injury / missed-games risk (season + player_id join). All-cause
    # missed-games risk, not narrowly "injury" -- see
    # build_injury_features_v1.py for why. ---
    injury = _load_injury_durability()
    dataset["player_id"] = dataset["player_id"].astype(str)
    dataset = dataset.merge(injury, on=["season", "player_id"], how="left")

    # --- Garbage-time-adjusted production (season + gsis_id join). See
    # build_garbage_time_features_v1.py: this is a measurement correction on
    # prior-season volume, not a prediction that garbage time recurs. ---
    dataset = dataset.merge(_load_garbage_time(), on=["season", "player_id"], how="left")

    # --- Route participation / TPRR (season + gsis_id join). Cleared its
    # persistence gate at r=+0.511 before being built; see
    # build_route_participation_features_v1.py. ---
    dataset = dataset.merge(_load_route_participation(), on=["season", "player_id"], how="left")

    # --- Team offensive environment (season + team join). Levels are the
    # control; the *_vs_baseline columns are the mean-reversion hypothesis.
    # See build_offense_environment_features_v1.py. ---
    offense = _load_offense_environment()
    if "team" in dataset.columns and not offense.empty:
        dataset["team"] = dataset["team"].astype(str)
        dataset = dataset.merge(offense, on=["season", "team"], how="left")

    return dataset


def _append_feature_inventory_rows() -> None:
    inventory_path = VALIDATION_DIR / "feature_expansion_inventory.csv"
    new_rows = pd.DataFrame([
        {"feature_name": "prior_snap_share", "position_model": "QB/RB/WR/TE", "what_it_represents": "Prior-season offensive snap share", "source_or_calculation": "nflverse snap_counts, player offense_snaps / team offense_snaps", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium (starts 2012)", "trust_status": "trusted"},
        {"feature_name": "draft_round", "position_model": "QB/RB/WR/TE", "what_it_represents": "NFL draft round (fixed, career-long)", "source_or_calculation": "nflverse draft_picks, gsis_id join", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium (undrafted players)", "trust_status": "trusted"},
        {"feature_name": "draft_pick_overall", "position_model": "QB/RB/WR/TE", "what_it_represents": "NFL draft overall pick (fixed, career-long)", "source_or_calculation": "nflverse draft_picks, gsis_id join", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium (undrafted players)", "trust_status": "trusted"},
        {"feature_name": "years_since_drafted", "position_model": "QB/RB/WR/TE", "what_it_represents": "Seasons of NFL experience", "source_or_calculation": "season - draft_season", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium", "trust_status": "trusted"},
        {"feature_name": "prior_redzone_target_share", "position_model": "WR/TE/RB", "what_it_represents": "Prior-season share of team red-zone targets", "source_or_calculation": "nflverse pbp, yardline_100<=20 targets / team red-zone targets", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium", "trust_status": "trusted"},
        {"feature_name": "prior_redzone_carry_share", "position_model": "RB", "what_it_represents": "Prior-season share of team red-zone carries", "source_or_calculation": "nflverse pbp, yardline_100<=20 carries / team red-zone carries", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium", "trust_status": "trusted"},
        {"feature_name": "prior_air_yards_share", "position_model": "WR/TE", "what_it_represents": "Prior-season share of team air yards", "source_or_calculation": "nflverse pbp, player air_yards / team air_yards", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium", "trust_status": "trusted"},
        {"feature_name": "prior_team_vacated_target_share", "position_model": "QB/RB/WR/TE", "what_it_represents": "Share of team's prior-season target volume that departed via free agency/retirement/trade", "source_or_calculation": "nflverse stats_player_reg diffed against next season's roster (rosters tag for the current boundary season)", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low (team-level context, not player-specific realized share)", "missing_value_risk": "medium", "trust_status": "trusted"},
        {"feature_name": "prior_wopr", "position_model": "WR/TE", "what_it_represents": "Weighted Opportunity Rating (1.5*target_share + 0.7*air_yards_share)", "source_or_calculation": "derived from prior_target_share and prior_air_yards_share", "objective_or_subjective": "objective", "pre_draft_available": "yes", "leakage_risk": "low", "missing_value_risk": "medium", "trust_status": "trusted"},
    ])
    if inventory_path.exists():
        existing = pd.read_csv(inventory_path)
        existing = existing[~existing["feature_name"].isin(new_rows["feature_name"])]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(inventory_path, index=False)


def main() -> None:
    dataset = build_workload_dataset()
    output_path = VALIDATION_DIR / "predraft_validation_dataset_workload_v1.csv"
    dataset.to_csv(output_path, index=False)
    _append_feature_inventory_rows()

    print(f"Workload dataset written: {output_path}")
    print(f"Rows: {len(dataset)}")
    for col in NEW_WORKLOAD_COLS:
        if col in dataset.columns:
            print(f"Non-null {col}: {int(dataset[col].notna().sum())}")
    print("Feature expansion inventory updated: research/validation_v1/feature_expansion_inventory.csv")


if __name__ == "__main__":
    main()
