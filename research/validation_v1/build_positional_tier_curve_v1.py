"""
Fit the within-position value cliff from realized outcomes, walk-forward.

Projections are smooth; reality is cliffed. FantasyPros has TE4-6 projecting
close to TE1-3, but realized value says TE1-3 returned far more than TE4-6 or
TE7-9. QB behaves the same way. RB barely declines across tiers 1-12, and WR
decays gradually with no cliff. A model that only reads projections cannot
see any of that.

What this fits is deliberately a WITHIN-POSITION SHAPE correction. It is NOT
a cross-position multiplier. position_value_score already handles
cross-position scarcity through per-position replacement baselines -- scaling
it by a factor derived from cross-position VOR levels would double-count
scarcity, which is exactly the mistake the old TE 0.68 multiplier made. This
factor only answers "how fast does value fall off *inside* this position,"
which is the part projections genuinely miss.

Tiering is by POSITIONAL ADP RANK (TE1, TE2, ...), because that is what the
underlying measurement used and because ADP is the signal this project has
repeatedly failed to beat.

Walk-forward: factors for a test season are fit on prior seasons only, so
the reported stability is honest and the same discipline as Step 5c.

REPLACEMENT RANK IS NOW A GRID, NOT A CONSTANT (2026-08-21)
-----------------------------------------------------------
This script previously hardcoded REPLACEMENT_RANK = {"QB": 12, "RB": 29,
"WR": 29, "TE": 14} and emitted a single fit. Those ranks describe a roster
format that is not the one being scored: under a 12-team 1QB/2RB/3WR/1TE/
2FLEX lineup the real replacement ranks are QB 12, RB 34, WR 46, TE 17.
Replacement rank sets the baseline that VOR is measured against, so getting
it wrong does not merely shift the curve -- it changes its SIGN. Measured
directly on the real 2015-2025 table, TE mid-tiers move from negative to
roughly flat purely by correcting the rank:

    TE tier      rep=14 (old)    rep=17 (real)
    1-3               35.7            49.9
    4-6               -4.3            +1.1
    7-9              -15.1            -2.6
    10-12              0.0           +11.2
    13-18            -12.9            -3.4
    19-30            -28.3           -13.1

Those negatives were the entire basis for the downstream 0.05 floor that
collapsed every non-elite TE's score. They are an artifact of the wrong
roster format, not a property of tight ends. WR was mis-specified further
still (29 vs 46), which matters because the question that started this was a
cross-position TE-vs-WR comparison -- fitting one position against the wrong
baseline makes that comparison meaningless.

So the output is now keyed on replacement_rank across a grid covering
realistic league formats, and draft_analysis.load_realized_tier_vor() selects
the rows matching the ACTIVE league configuration at scoring time. Nothing
here bakes in 17/34/46: change the roster settings and a different fit is
selected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "data" / "positional_tier_curve.csv"

POSITIONS = ["QB", "RB", "WR", "TE"]

# Covers realistic replacement ranks across league sizes (8-16 teams) and
# formats (including superflex, TE-premium, and 3WR/2FLEX lineups). Emitting
# a grid keeps the fit itself offline while letting the live scorer pick the
# row matching whatever roster is actually configured.
REPLACEMENT_RANK_GRID = list(range(4, 81))

# Ranks used only for the human-readable report below -- a 12-team
# 1QB/2RB/3WR/1TE/2FLEX lineup. NOT used to filter the emitted grid.
REPORT_RANKS = {"QB": 12, "RB": 34, "WR": 46, "TE": 17}

# Tier edges by positional ADP rank. Kept coarse on purpose: at ~3 players
# per position per season, narrower tiers would be mostly noise.
TIERS = [("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9),
         ("10-12", 10, 12), ("13-18", 13, 18), ("19-30", 19, 30)]

# Modern-era only. The user's premise -- premium QB/TE hold ADP while the
# rest slip -- is a statement about how drafts behave now, and pre-2015
# seasons carry different league dynamics (lower passing volume, different
# TE usage). Costs sample size (n~33/cell vs ~80) and that is the trade.
MIN_SEASON = 2015

# Below this many player-seasons a tier estimate is too thin to trust; fall
# back to the neighbouring tier's factor instead of fitting noise.
MIN_TIER_N = 12


def load() -> pd.DataFrame:
    """
    Every player-season with a real final score.

    Baselines are computed over this FULL pool, including players who
    carried no preseason ADP -- an undrafted breakout is genuinely part of
    what replacement level returned that year, and excluding them lowers
    the Nth-best cutoff and inflates every tier's VOR. Measured: restricting
    the baseline pool to ADP'd players alone flips TE 4-6 from -4.3 to
    +24.2, which would have been a fake result. Only AFTER baselines are
    fixed is the frame restricted to players with a real ADP (the tiering
    key).
    """
    d = pd.read_csv(DATASET_PATH, low_memory=False)
    d = d[d["final_fantasy_points"].notna() & d["position"].isin(POSITIONS)].copy()
    d["final_fantasy_points"] = pd.to_numeric(d["final_fantasy_points"], errors="coerce")
    d["positional_adp"] = pd.to_numeric(d["positional_adp"], errors="coerce")
    return d


def _season_baselines(pool: pd.DataFrame, position: str, rank: int) -> dict:
    """Nth-best real finish per season for one position at one replacement rank."""
    baselines = {}
    for season, grp in pool[pool["position"] == position].groupby("season"):
        pts = grp["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        baselines[season] = float(pts[rank - 1]) if len(pts) >= rank else float(pts[-1])
    return baselines


def _tier_of(rank: float) -> str | None:
    for label, lo, hi in TIERS:
        if lo <= rank <= hi:
            return label
    return None


def fit_position(pool: pd.DataFrame, position: str, rank: int) -> pd.DataFrame:
    """
    Realized VOR per tier for one position at one replacement rank,
    weighted by PLAYER rather than by player-season.

    Season-weighting let a single career dominate a cell. Measured: Travis
    Kelce is 8 of the 33 TE1-3 seasons (24% of the cell) at +128.1 mean VOR,
    and 4 of the top 5 outcomes in the Bowers/McBride draft zone -- which
    made premium TE look far more valuable than it is, and Kelce isn't even
    in the 2026 premium pool (ADP 121). Concentration is systemic, not just
    a Kelce problem: McCaffrey is 22% of RB1-3, Mahomes 18% of QB1-3, and
    every cell has only ~13-17 distinct players across 11 seasons.

    So: collapse each player to his own median within a cell first, then
    take the median across players. No single career can carry a cell, and
    the median also absorbs the bust-season skew that drags the QB/RB/WR
    means down.
    """
    baselines = _season_baselines(pool, position, rank)

    sub = pool[
        (pool["position"] == position)
        & pool["positional_adp"].notna()
        & (pool["season"] >= MIN_SEASON)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["vor"] = sub["final_fantasy_points"] - sub["season"].map(baselines)
    sub["tier"] = sub["positional_adp"].apply(_tier_of)
    sub = sub[sub["tier"].notna()]
    if sub.empty:
        return pd.DataFrame()

    per_player = sub.groupby(["tier", "player_name"])["vor"].median().reset_index()
    stats = per_player.groupby("tier")["vor"].agg(["median", "size"])
    seasons_n = sub.groupby("tier").size()

    rows = []
    for label, _, _ in TIERS:
        if label not in stats.index:
            continue
        season_rows = int(seasons_n.get(label, 0))
        if season_rows < MIN_TIER_N:
            continue  # too thin to trust; downstream falls back to 1.0
        rows.append({
            "position": position,
            "replacement_rank": rank,
            "tier": label,
            "n": season_rows,
            "distinct_players": int(stats.loc[label, "size"]),
            "tier_vor": round(float(stats.loc[label, "median"]), 2),
        })
    return pd.DataFrame(rows)


def fit_grid(pool: pd.DataFrame) -> pd.DataFrame:
    frames = [
        fit_position(pool, position, rank)
        for position in POSITIONS
        for rank in REPLACEMENT_RANK_GRID
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def walk_forward_stability(pool: pd.DataFrame, ranks: dict) -> pd.DataFrame:
    """
    Fit on seasons < test, compare to what the test season delivered.
    Reports correlation between predicted and realized tier ordering, which
    is the question that matters: is the SHAPE stable enough to rely on?
    """
    seasons = sorted(s for s in pool["season"].unique() if s >= MIN_SEASON)
    rows = []
    for test_season in seasons:
        prior = pool[(pool["season"] < test_season) & (pool["season"] >= MIN_SEASON)]
        test = pool[pool["season"] == test_season]
        if prior["season"].nunique() < 3 or test.empty:
            continue

        pred = pd.concat(
            [fit_position(prior, p, ranks[p]) for p in POSITIONS], ignore_index=True
        )
        real = pd.concat(
            [fit_position(test, p, ranks[p]) for p in POSITIONS], ignore_index=True
        )
        if pred.empty or real.empty:
            continue

        merged = pred.merge(real, on=["position", "tier"], suffixes=("_pred", "_real"))
        if len(merged) < 4:
            continue
        rows.append({
            "test_season": int(test_season),
            "tier_cells": len(merged),
            "spearman_pred_vs_real": float(
                merged["tier_vor_pred"].corr(merged["tier_vor_real"], method="spearman")
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    pool = load()
    modern = pool[pool["season"] >= MIN_SEASON]
    print(f"Fitting on {int(modern['season'].min())}-{int(modern['season'].max())} "
          f"({modern['season'].nunique()} seasons, {len(modern)} player-seasons)")
    print(f"Replacement-rank grid: {REPLACEMENT_RANK_GRID[0]}-{REPLACEMENT_RANK_GRID[-1]}")

    grid = fit_grid(pool)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_PATH, index=False)
    print(f"Emitted {len(grid)} (position, replacement_rank, tier) rows")

    report = grid[
        [rank == REPORT_RANKS[pos] for pos, rank in zip(grid["position"], grid["replacement_rank"])]
    ]
    print(f"\n=== Player-weighted median realized VOR at {REPORT_RANKS} ===")
    piv = report.pivot_table(index="tier", columns="position", values="tier_vor")
    print(piv.reindex([label for label, _, _ in TIERS]).round(1).to_string())

    print("\n=== Distinct players per cell (concentration check) ===")
    npiv = report.pivot_table(index="tier", columns="position", values="distinct_players")
    print(npiv.reindex([label for label, _, _ in TIERS]).to_string())

    wf = walk_forward_stability(pool, REPORT_RANKS)
    if not wf.empty:
        print("\n=== Walk-forward shape stability (fit on prior seasons only) ===")
        print(wf.round(3).to_string(index=False))
        print(f"\nMean Spearman(predicted tier shape, realized tier shape): "
              f"{wf['spearman_pred_vs_real'].mean():.3f}")
        print("Positive and consistent => the cliff shape is stable enough to use.")

    print(f"\nWritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
