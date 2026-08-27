"""Breakout Score V1 -- score the actual 2026 WR pool (2026-08-27, updated 2026-08-27).

ADP SOURCE: real Sleeper Half-PPR ADP, from a multi-source ADP sheet the
user supplied directly (JuiceBoxOne's 2026 rankings, aggregating FantasyPros/
ESPN/Sleeper/Yahoo ADP per player), copied into
research/validation_v1/data/juicebox_2026_adp_sources.csv. Uses its
"Sleeper Half" column specifically -- real community draft-position data,
not a proxy. (An earlier pass tried Sleeper's public API `search_rank` field
as a stand-in, since Sleeper has no public ADP endpoint; that's been
replaced now that real Sleeper ADP is available. Also caught and fixed on
that pass: Sleeper's `active` flag alone let retired players like Henry
Ruggs and William Fuller through -- not a concern with this file, since it
only lists players actually being drafted.)

79 WR rows in the sheet, 63 with a Sleeper Half value populated (some WRs
are ADP'd elsewhere but not commonly drafted in Sleeper leagues specifically
-- those are excluded rather than backfilled from another platform, since
mixing ADP sources within one feature would reintroduce exactly the
scale-inconsistency problem checked for during the search_rank pass).

Not a live UI wire-in (see research/MODEL_REGISTRY.md -- still RESEARCH_ONLY).
This locks in real predictions against the CURRENT draft pool, timestamped,
before the 2026 season happens. That's the only thing that actually resolves
the open question on this model: every check run so far (nested comparison,
permutation test, leave-one-season-out, coefficient stability, threshold
sensitivity) was computed on seasons that already existed and whose results
could -- even unintentionally -- shape which spec got kept (this file's own
FEATURES spec was chosen AFTER seeing that dropping div_opportunity_minus_
efficiency improved the backtest). A prediction written today, before Week 1,
is not subject to that risk. When 2026 outcomes are final, compare this
file's predictions against Beat_ADP_By_12 for real -- that comparison, not
another backtest pass, is what should move this model's registry status.

Feature source: same pattern as build_current_season_archetypes.py --
prior-season (2025) usage caches under research/validation_v1/data/, joined
to data/processed/master_players.csv by cleaned player name. Route
participation carries only a GSIS player_id (no name), so it's crosswalked
through snap_share_player_seasons.csv's id-to-name mapping first.

Run: python research/validation_v1/score_current_wr_pool_v1.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[1]
sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd

from validation_utils import clean_name, initial_last_key  # noqa: E402
from breakout_score_v1_scorer import (  # noqa: E402
    FEATURES, TARGET_DEFINITION, fit_final_model, load, score_players,
)

DATA_DIR = VALIDATION_DIR / "data"
ADP_SOURCE_PATH = DATA_DIR / "juicebox_2026_adp_sources.csv"
MASTER_PLAYERS = PROJECT_ROOT / "data" / "processed" / "master_players.csv"
SEASON = 2026
ADP_MAX = 150
TAG_MIN, TAG_MAX = 15, 20
OUTPUT_PATH = VALIDATION_DIR / "data" / "breakout_v1" / f"current_wr_predictions_{SEASON}.csv"
APP_TAG_PATH = PROJECT_ROOT / "data" / "processed" / f"breakout_tags_{SEASON}.csv"


def build_current_wr_pool() -> pd.DataFrame:
    adp_source = pd.read_csv(ADP_SOURCE_PATH)
    pool = adp_source[(adp_source["Pos"] == "WR") & adp_source["Sleeper Half"].notna()].copy()
    pool = pool[pool["Sleeper Half"] <= ADP_MAX].copy()
    pool = pool.rename(columns={"Name": "player_name", "Team": "team"})
    pool["overall_adp"] = pd.to_numeric(pool["Sleeper Half"], errors="coerce")
    pool["_key"] = pool["player_name"].apply(clean_name)

    master = pd.read_csv(MASTER_PLAYERS, low_memory=False)
    master_wr = master[master["position"] == "WR"][["player_name", "age"]].copy()
    master_wr["_key"] = master_wr["player_name"].apply(clean_name)
    master_wr = master_wr.dropna(subset=["_key"])
    master_wr = master_wr[~master_wr["_key"].duplicated(keep=False)]
    pool = pool.merge(master_wr[["_key", "age"]], on="_key", how="left")
    pool["_ilkey"] = pool["player_name"].apply(initial_last_key)
    dup = pool["_key"].duplicated(keep=False)
    if dup.any():
        pool = pool[~dup].copy()  # name-collision guard: drop rather than misattribute

    def _load(name: str) -> pd.DataFrame:
        p = DATA_DIR / name
        return pd.read_csv(p, low_memory=False) if p.exists() else pd.DataFrame()

    snap = _load("snap_share_player_seasons.csv")
    snap = snap[snap["season"] == SEASON].copy()
    snap["_key"] = snap["player_name"].apply(clean_name)
    id_to_name = snap.drop_duplicates("player_id").set_index("player_id")["player_name"]

    dur = _load("injury_durability_player_seasons.csv")
    dur = dur[dur["season"] == SEASON].copy()
    dur["_key"] = dur["player_name"].apply(clean_name)

    gt = _load("garbage_time_player_seasons.csv")
    gt = gt[gt["season"] == SEASON].copy()
    # PFR-abbreviated names ("A.Rodgers") -- full-name matching is impossible
    # here, same issue build_current_season_archetypes.py documents for
    # redzone_airyards_player_seasons.csv. Use initial_last_key instead.
    gt["_ilkey"] = gt["player_name"].apply(initial_last_key)

    rp = _load("route_participation_player_seasons.csv")
    rp = rp[rp["season"] == SEASON].copy()
    rp["player_name"] = rp["player_id"].map(id_to_name)
    rp = rp[rp["player_name"].notna()].copy()
    rp["_key"] = rp["player_name"].apply(clean_name)

    def _merge_one(pool, src, cols, key="_key"):
        cols = [c for c in cols if c in src.columns]
        src = src.dropna(subset=[key])
        src = src[~src[key].duplicated(keep=False)]  # collision guard on the source side too
        return pool.merge(src[[key] + cols], on=key, how="left")

    pool = _merge_one(pool, snap, ["prior_snap_share"])
    pool = _merge_one(pool, dur, ["prior_durability_score"])
    pool = _merge_one(pool, gt, ["prior_garbage_time_share"], key="_ilkey")
    pool = _merge_one(pool, rp, ["prior_route_participation_rate", "prior_targets_per_route_run"])

    return pool.drop(columns=["_key", "_ilkey"])


def select_natural_cutoff(probs: pd.Series, min_n=TAG_MIN, max_n=TAG_MAX) -> int:
    """How many of the top-ranked players to tag, between min_n and max_n --
    same idea as Home.py's _assign_automatic_tiers() (largest score gap
    decides the break, not a fixed count). Looks at the gaps between
    consecutive sorted probabilities strictly between min_n and max_n and
    cuts at the largest one; falls back to max_n if the pool is too small
    to have a gap in that window."""
    vals = probs.sort_values(ascending=False).to_numpy()
    if len(vals) <= min_n:
        return len(vals)
    window_hi = min(max_n, len(vals) - 1)
    if window_hi <= min_n:
        return window_hi
    # gaps[j] = vals[rank (min_n+j)] - vals[rank (min_n+j+1)] -- the drop if
    # the cut is placed right after rank (min_n+j), keeping min_n+j players.
    gaps = vals[min_n - 1:window_hi] - vals[min_n:window_hi + 1]
    best_offset = int(np.argmax(gaps))
    return min_n + best_offset


def main() -> int:
    pool = build_current_wr_pool()
    coverage = {f: round(float(pool[f].notna().mean() * 100), 1) for f in FEATURES if f in pool.columns}
    print(f"Current WR pool (Sleeper Half ADP<={ADP_MAX}): {len(pool)} players")
    print("Feature coverage (% non-missing):")
    for f, pct in coverage.items():
        print(f"  {f}: {pct}%")
    missing_any = [f for f in FEATURES if f not in pool.columns]
    if missing_any:
        print(f"MISSING ENTIRELY: {missing_any}")
        return 1

    d = load("WR")
    model = fit_final_model(d)
    scored = score_players(model, pool)
    scored = scored.sort_values("breakout_probability_v1", ascending=False)

    out_cols = ["player_name", "team", "overall_adp", "breakout_probability_v1", "breakout_decile_v1"] + \
               [f for f in FEATURES if f != "overall_adp"]
    out = scored[out_cols].copy()
    out.insert(0, "scored_at_utc", datetime.now(timezone.utc).isoformat())
    out.insert(1, "target_definition", TARGET_DEFINITION)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    n_tag = select_natural_cutoff(scored["breakout_probability_v1"])
    tagged = scored.head(n_tag).copy()
    tagged.insert(0, "breakout_rank", range(1, len(tagged) + 1))
    tag_out = tagged[["breakout_rank", "player_name", "overall_adp", "breakout_probability_v1"]].copy()
    tag_out.insert(0, "scored_at_utc", datetime.now(timezone.utc).isoformat())
    APP_TAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tag_out.to_csv(APP_TAG_PATH, index=False)

    print(f"\nTagged {n_tag} players (natural gap between {TAG_MIN}-{TAG_MAX}):")
    print(tag_out[["breakout_rank", "player_name", "overall_adp", "breakout_probability_v1"]].to_string(index=False))
    print(f"\nFull pool written: {OUTPUT_PATH}")
    print(f"App tag file written: {APP_TAG_PATH}")
    print("Prediction timestamp locked. Check against real Beat_ADP_By_12 outcomes after the 2026 season.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
