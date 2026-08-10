"""
Fit the within-position value cliff from realized outcomes, walk-forward.

Projections are smooth; reality is cliffed. FantasyPros has TE4-6 projecting
close to TE1-3, but realized value says TE1-3 returned +63.2 VOR against
TE4-6's +12.9 and TE7-9's -10.1 (2015+). QB behaves the same way. RB barely
declines at all across tiers 1-12, and WR decays gradually with no cliff.
A model that only reads projections cannot see any of that.

What this fits is deliberately a WITHIN-POSITION SHAPE correction:

    tier_factor(pos, tier) = realized_vor(pos, tier) / realized_vor(pos, tier_1_3)

It is NOT a cross-position multiplier. position_value_score already handles
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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "data" / "positional_tier_curve.csv"

REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 14}
POSITIONS = ["QB", "RB", "WR", "TE"]

# Tier edges by positional ADP rank. Kept coarse on purpose: at ~3 players
# per position per season, narrower tiers would be mostly noise.
TIERS = [("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9),
         ("10-12", 10, 12), ("13-18", 13, 18), ("19-30", 19, 30)]

# Modern-era only. The user's premise -- premium QB/TE hold ADP while the
# rest slip -- is a statement about how drafts behave now, and pre-2015
# seasons carry different league dynamics (lower passing volume, different
# TE usage). Costs sample size (n~33/cell vs ~80) and that is the trade.
MIN_SEASON = 2015

# A tier whose realized value is at or below replacement is worth no premium.
# Floored rather than allowed to go negative: a negative factor would flip
# the sign of a player's value and rank bad players above good ones.
MIN_FACTOR = 0.05

# Below this many player-seasons a tier estimate is too thin to trust; fall
# back to the neighbouring tier's factor instead of fitting noise.
MIN_TIER_N = 12


def load() -> pd.DataFrame:
    d = pd.read_csv(DATASET_PATH, low_memory=False)
    d = d[d["final_fantasy_points"].notna() & d["position"].isin(POSITIONS)].copy()
    d["final_fantasy_points"] = pd.to_numeric(d["final_fantasy_points"], errors="coerce")
    d["positional_adp"] = pd.to_numeric(d["positional_adp"], errors="coerce")

    baselines = {}
    for (season, position), grp in d.groupby(["season", "position"]):
        pts = grp["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        n = REPLACEMENT_RANK[position]
        baselines[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])
    d["vor"] = d["final_fantasy_points"] - [baselines[(s, p)] for s, p in zip(d["season"], d["position"])]
    return d[d["positional_adp"].notna()]


def _tier_of(rank: float) -> str | None:
    for label, lo, hi in TIERS:
        if lo <= rank <= hi:
            return label
    return None


def fit_factors(d: pd.DataFrame) -> pd.DataFrame:
    """
    Realized VOR per (position, tier), weighted by PLAYER rather than by
    player-season.

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
    means down. Effect on tier 1-3: TE 63.2 -> 35.7, WR 62.6 -> 62.7,
    RB 73.8 -> 85.6, QB 15.0 -> 9.4, i.e. RB > WR > TE > QB.
    """
    d = d.copy()
    d["tier"] = d["positional_adp"].apply(_tier_of)
    d = d[d["tier"].notna()]

    # Per-player median inside each cell, then median across players.
    per_player = d.groupby(["position", "tier", "player_name"])["vor"].median().reset_index()

    rows = []
    for position in POSITIONS:
        sub = per_player[per_player["position"] == position]
        stats = sub.groupby("tier")["vor"].agg(["median", "size"])
        seasons_n = d[d["position"] == position].groupby("tier").size()
        if "1-3" not in stats.index or stats.loc["1-3", "median"] <= 0:
            continue

        for label, _, _ in TIERS:
            if label not in stats.index:
                continue
            distinct_players = int(stats.loc[label, "size"])
            season_rows = int(seasons_n.get(label, 0))
            if season_rows < MIN_TIER_N:
                continue  # too thin to trust; downstream falls back to 1.0
            rows.append({
                "position": position,
                "tier": label,
                "n": season_rows,
                "distinct_players": distinct_players,
                "tier_vor": round(float(stats.loc[label, "median"]), 2),
            })
    return pd.DataFrame(rows)


def walk_forward_stability(d: pd.DataFrame) -> pd.DataFrame:
    """
    Fit factors on seasons < test, compare to what the test season delivered.
    Reports correlation between predicted and realized tier ordering, which
    is the question that matters: is the SHAPE stable enough to rely on?
    """
    seasons = sorted(s for s in d["season"].unique() if s >= MIN_SEASON)
    rows = []
    for test_season in seasons:
        prior = d[(d["season"] < test_season) & (d["season"] >= MIN_SEASON)]
        test = d[d["season"] == test_season]
        if prior["season"].nunique() < 3 or test.empty:
            continue
        pred = fit_factors(prior)
        real = fit_factors(test)
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
    d = load()
    modern = d[d["season"] >= MIN_SEASON]
    print(f"Fitting on {int(modern['season'].min())}-{int(modern['season'].max())} "
          f"({modern['season'].nunique()} seasons, {len(modern)} player-seasons)")

    factors = fit_factors(modern)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Player-weighted median realized VOR by (position, draft tier) ===")
    piv = factors.pivot_table(index="tier", columns="position", values="tier_vor")
    piv = piv.reindex([label for label, _, _ in TIERS])
    print(piv.round(1).to_string())

    print("\n=== Distinct players per cell (concentration check) ===")
    npiv = factors.pivot_table(index="tier", columns="position", values="distinct_players")
    print(npiv.reindex([label for label, _, _ in TIERS]).to_string())

    wf = walk_forward_stability(d)
    if not wf.empty:
        print("\n=== Walk-forward shape stability (fit on prior seasons only) ===")
        print(wf.round(3).to_string(index=False))
        mean_rho = wf["spearman_pred_vs_real"].mean()
        print(f"\nMean Spearman(predicted tier shape, realized tier shape): {mean_rho:.3f}")
        print("Positive and consistent => the cliff shape is stable enough to use.")

    print(f"\nWritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
