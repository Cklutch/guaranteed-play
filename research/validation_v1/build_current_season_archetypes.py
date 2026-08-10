"""
Assign archetypes to the CURRENT draft pool and write them where the app
can read them.

The historical archetype file (predraft_validation_dataset_archetypes_v1.csv)
stops at 2025 because it's built from the outcome dataset, which by
definition needs results that have already happened. For the live draft we
need the opposite: archetypes for the upcoming season, computed purely from
last season's usage, with no outcome involved. Same assignment logic, run
against the current player pool.

Output: research/validation_v1/data/current_season_archetypes.csv, joined
into data/processed/master_players.csv by the data pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_archetypes_v1 import build_archetypes  # noqa: E402
from validation_utils import VALIDATION_DIR, clean_name, initial_last_key  # noqa: E402

DATA_DIR = VALIDATION_DIR / "data"
MASTER_PLAYERS = PROJECT_ROOT / "data" / "processed" / "master_players.csv"
OUTPUT_PATH = DATA_DIR / "current_season_archetypes.csv"
DEFAULT_SEASON = 2026


def _load(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def _ambiguous_keys(frame: pd.DataFrame, key_col: str) -> set:
    """Keys that map to more than one distinct player within `frame`."""
    if key_col not in frame.columns or "player_name" not in frame.columns:
        return set()
    counts = frame.groupby(key_col)["player_name"].nunique()
    return set(counts[counts > 1].index)


def _safe_merge(
    pool: pd.DataFrame,
    right: pd.DataFrame,
    key_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Join on `key_col`, but only for keys that identify exactly one player on
    BOTH sides -- and scope that ambiguity test to fantasy positions.

    The pool is ~3,900 players deep including every kicker and defender, so
    abbreviated keys collide badly: "r white" alone covers Ricky White,
    Reggie White and Roddy White. Left unhandled, each inherits a single
    player's usage -- the first run of this script labeled two retired
    players as possession receivers and a cornerback as a receiving tight
    end.

    But testing ambiguity against the *whole* pool over-blocks just as
    badly. "j allen" collides only because a defensive lineman shares the
    key with Josh Allen; "l jackson" only because of a cornerback. Every
    source joined here carries offensive skill-position usage, which a
    defender cannot own, so restricting the test to QB/RB/WR/TE resolves
    both keys without loosening the guard where it matters. Keys still
    ambiguous among skill players stay blocked, and those players stay
    unclassified -- the honest outcome.
    """
    value_cols = [c for c in value_cols if c in right.columns]
    if not value_cols or key_col not in right.columns:
        return pool

    skill_pool = pool[pool["position"].astype(str).str.upper().isin(FANTASY_POSITIONS)]

    if "position" in right.columns:
        # Position is available on both sides, so qualify the key with it.
        # Most remaining collisions are cross-position -- "j allen" is Josh
        # Allen (QB) and Javorius Allen (RB); "l jackson" is Lamar Jackson
        # (QB) and Lucky Jackson (WR). On a name-only key both of the
        # league's signature dual-threat quarterbacks were blocked and left
        # unclassified. Adding position separates them exactly.
        right = right[right["position"].astype(str).str.upper().isin(FANTASY_POSITIONS)].copy()
        pool = pool.copy()
        qualified = "_join_key_pos"
        pool[qualified] = pool["position"].astype(str).str.upper() + "|" + pool[key_col].astype(str)
        right[qualified] = right["position"].astype(str).str.upper() + "|" + right[key_col].astype(str)

        blocked = _ambiguous_keys(
            pool[pool["position"].astype(str).str.upper().isin(FANTASY_POSITIONS)], qualified
        ) | _ambiguous_keys(right, qualified)

        right = right[~right[qualified].isin(blocked)].drop_duplicates(qualified, keep="first")
        merged = pool.merge(right[[qualified] + value_cols], on=qualified, how="left")
        merged = merged.drop(columns=[qualified])
        non_skill = ~merged["position"].astype(str).str.upper().isin(FANTASY_POSITIONS)
        merged.loc[non_skill, value_cols] = np.nan
        return merged

    # No position on the right (play-by-play aggregates). Those rows are
    # receiving/rushing usage, so they're skill-position by construction,
    # but there's no position to qualify the key with -- fall back to
    # blocking any key ambiguous among skill players.
    blocked = _ambiguous_keys(skill_pool, key_col) | _ambiguous_keys(right, key_col)
    right = right[~right[key_col].isin(blocked)].drop_duplicates(key_col, keep="first")
    merged = pool.merge(right[[key_col] + value_cols], on=key_col, how="left")

    # Non-skill players must never inherit skill usage through a shared key.
    non_skill = ~merged["position"].astype(str).str.upper().isin(FANTASY_POSITIONS)
    merged.loc[non_skill, value_cols] = np.nan
    return merged


def build_current_pool(season: int) -> pd.DataFrame:
    """
    Assemble a season-row frame shaped like the historical dataset, so the
    same build_archetypes() logic applies unchanged.
    """
    master = pd.read_csv(MASTER_PLAYERS, low_memory=False)
    pool = master[["player_name", "position", "team", "age", "adp"]].copy()
    pool["season"] = season
    pool["overall_adp"] = pd.to_numeric(pool["adp"], errors="coerce")
    pool["initial_last_key"] = pool["player_name"].apply(initial_last_key)
    pool["player_key"] = pool["player_name"].apply(clean_name)

    # Snap share carries FULL player names (from nflverse snap_counts), same
    # as master_players -- so join on the full normalized name, which is far
    # less collision-prone than the initial+lastname key.
    snap = _load("snap_share_player_seasons.csv")
    if not snap.empty:
        snap = snap[snap["season"] == season].copy()
        snap["player_key"] = snap["player_name"].apply(clean_name)
        snap = snap.sort_values("prior_snap_share", na_position="first")
        pool = _safe_merge(pool, snap, "player_key", ["prior_snap_share"])

    # Play-by-play names are PFR-abbreviated ("J.Jefferson"), so full-name
    # matching is impossible here and the ambiguous-key guard does the work.
    rz = _load("redzone_airyards_player_seasons.csv")
    if not rz.empty:
        rz = rz[rz["season"] == season].copy()
        rz_cols = [
            "prior_redzone_target_share", "prior_redzone_carry_share",
            "prior_air_yards", "prior_air_yards_share",
            "prior_targets_pbp", "prior_carries_pbp",
            "prior_adot", "prior_explosive_run_rate", "prior_explosive_rec_rate",
            "prior_yards_per_carry",
        ]
        rz = rz.sort_values("prior_targets_pbp", na_position="first")
        pool = _safe_merge(pool, rz, "initial_last_key", rz_cols)

    # Prior-season counting stats from the outcome cache. For a {season}
    # draft, the prior season is {season - 1}, and those are completed
    # actuals -- not a forecast, so still pre-draft safe. This is what makes
    # the QB archetypes work: dual-threat vs pocket needs passing and
    # rushing yardage, which the play-by-play aggregation doesn't capture
    # for passers.
    outcome_path = PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_ppr.csv"
    if outcome_path.exists():
        prior = pd.read_csv(outcome_path, low_memory=False)
        prior = prior[prior["season"] == season - 1].copy()
        prior["initial_last_key"] = prior["player_name"].apply(initial_last_key)
        prior = prior.sort_values("fantasy_points", ascending=False, na_position="last")
        prior = prior.rename(columns={
            "passing_yards": "prior_passing_yards",
            "rushing_yards": "prior_rushing_yards",
            "receiving_yards": "prior_receiving_yards",
        })
        pool = _safe_merge(
            pool, prior, "initial_last_key",
            ["prior_passing_yards", "prior_rushing_yards", "prior_receiving_yards"],
        )

    # Columns build_archetypes() reads that this current-season path has no
    # source for. Created empty so the percentile helpers return NaN rather
    # than raising -- archetypes needing them simply won't fire. Both come
    # from the WR-opportunity cache, which is historical-only.
    for col in ["prior_target_share", "prior_wopr", "prior_rushing_yards", "prior_passing_yards"]:
        if col not in pool.columns:
            pool[col] = np.nan

    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign archetypes to the current draft pool.")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    args = parser.parse_args()

    pool = build_current_pool(args.season)
    scored = build_archetypes(pool)

    keep = ["player_name", "position", "team", "archetype_primary", "archetype_confidence"]
    keep += [c for c in scored.columns if c.startswith("arch_")]
    out = scored[keep].copy()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    classified = out[out["archetype_primary"] != "unclassified"]
    print(f"Current-season archetypes written: {OUTPUT_PATH}")
    print(f"Players in pool: {len(out)} | classified: {len(classified)}")
    print("\nDistribution:")
    print(classified["archetype_primary"].value_counts().to_string())


if __name__ == "__main__":
    main()
