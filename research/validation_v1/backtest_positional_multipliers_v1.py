"""
Step 9 Part B: are the app's hand-set positional multipliers justified?

`draftkit/draft_analysis.py` applies POSITION_VALUE_MULTIPLIERS
{QB:1.00, RB:1.00, WR:1.00, TE:0.68} and POSITION_URGENCY_MULTIPLIERS
{TE:0.70} on top of a value-over-replacement calculation that ALREADY
accounts for positional scarcity. Part A measured the consequence: the
shipped board's top 50 contains zero tight ends, where ADP's top 50
contains three. Brock Bowers goes from ADP 19.7 to board rank 147.

So the multipliers make a testable empirical claim: *at the same overall
draft slot, a TE returns materially less value than an RB or WR.* This
script checks that against 16 seasons of real outcomes.

Method: within each (season, position), replacement level is the Nth-best
finisher at that position by actual fantasy points, where N is roughly the
number of weekly starters in a 12-team league. A player's realized
value-over-replacement (VOR) is his actual points minus that baseline.
Bucketing by the overall ADP a player was drafted at, the empirical
positional multiplier is simply mean-VOR-by-position divided by the
best position's mean VOR in the same bucket -- exactly the quantity the
hand-set constants are asserting.

Note on FLEX: replacement counts below model 12 starting QB/TE and 24
starting RB/WR and deliberately ignore the FLEX slot, which would push
RB/WR replacement deeper (~30) and *raise* their VOR relative to TE. That
biases this test slightly AGAINST tight ends, so a pro-TE finding here is
conservative rather than flattering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "positional_multiplier_backtest.csv"

# Weekly starters in a standard 12-team league; the point where the next
# player at that position is freely available on waivers.
REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}

# What the app currently ships (draft_analysis.py lines ~50-63).
SHIPPED_VALUE_MULTIPLIERS = {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 0.68}

ADP_BUCKETS = [
    ("1-12", 1, 12),
    ("13-24", 13, 24),
    ("25-48", 25, 48),
    ("49-84", 49, 84),
    ("85-120", 85, 120),
    ("121-180", 121, 180),
]


def _bucket(adp: float) -> str | None:
    for label, lo, hi in ADP_BUCKETS:
        if lo <= adp <= hi:
            return label
    return None


def build_vor(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["overall_adp"].notna() & df["final_fantasy_points"].notna()].copy()
    d["overall_adp"] = pd.to_numeric(d["overall_adp"], errors="coerce")
    d["final_fantasy_points"] = pd.to_numeric(d["final_fantasy_points"], errors="coerce")
    d = d[d["position"].isin(REPLACEMENT_RANK)].copy()

    # Replacement baseline per (season, position) from realized finishes.
    baselines = {}
    for (season, position), grp in d.groupby(["season", "position"]):
        n = REPLACEMENT_RANK[position]
        pts = grp["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        # If a season has fewer than N scored players at the position, fall
        # back to the worst available rather than inventing a baseline.
        baselines[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])

    d["replacement_points"] = [baselines[(s, p)] for s, p in zip(d["season"], d["position"])]
    d["vor"] = d["final_fantasy_points"] - d["replacement_points"]
    d["adp_bucket"] = d["overall_adp"].apply(_bucket)
    return d[d["adp_bucket"].notna()].copy()


def empirical_multipliers(d: pd.DataFrame) -> pd.DataFrame:
    """Mean realized VOR by (adp_bucket, position), normalized within bucket."""
    agg = d.groupby(["adp_bucket", "position"]).agg(
        n=("vor", "size"),
        mean_vor=("vor", "mean"),
        median_vor=("vor", "median"),
    ).reset_index()
    # Normalize so the strongest position in each bucket = 1.00, which is the
    # same convention the shipped POSITION_VALUE_MULTIPLIERS use.
    agg["empirical_multiplier"] = agg.groupby("adp_bucket")["mean_vor"].transform(
        lambda s: s / s.max() if s.max() > 0 else np.nan
    )
    agg["shipped_multiplier"] = agg["position"].map(SHIPPED_VALUE_MULTIPLIERS)
    order = {label: i for i, (label, _, _) in enumerate(ADP_BUCKETS)}
    return agg.sort_values(["adp_bucket", "position"], key=lambda s: s.map(order) if s.name == "adp_bucket" else s)


def walk_forward_te_check(d: pd.DataFrame) -> pd.DataFrame:
    """
    Stability check with no hindsight: for each test season, derive the
    TE-vs-best multiplier from PRIOR seasons only, then report what the
    test season actually delivered. Same discipline as Step 5c.
    """
    seasons = sorted(d["season"].dropna().unique().tolist())
    rows = []
    for test_season in seasons:
        prior = d[d["season"] < test_season]
        test = d[d["season"] == test_season]
        if prior.empty or test.empty:
            continue
        # Early-round only -- this is where the multiplier actually decides picks.
        prior_early = prior[prior["adp_bucket"].isin(["1-12", "13-24", "25-48"])]
        test_early = test[test["adp_bucket"].isin(["1-12", "13-24", "25-48"])]
        if prior_early.empty or test_early.empty:
            continue

        pm = prior_early.groupby("position")["vor"].mean()
        tm = test_early.groupby("position")["vor"].mean()
        if "TE" not in pm.index or "TE" not in tm.index or pm.max() <= 0 or tm.max() <= 0:
            continue
        rows.append({
            "test_season": int(test_season),
            "predicted_te_multiplier": float(pm["TE"] / pm.max()),
            "realized_te_multiplier": float(tm["TE"] / tm.max()),
            "te_n_test": int((test_early["position"] == "TE").sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    d = build_vor(df)
    print(f"Player-seasons with ADP + realized outcome: {len(d)} "
          f"({int(d['season'].min())}-{int(d['season'].max())})")

    agg = empirical_multipliers(d)
    agg.to_csv(OUTPUT_PATH, index=False)
    print(f"Written: {OUTPUT_PATH}")

    print("\n=== Realized value-over-replacement by draft slot and position ===")
    for label, _, _ in ADP_BUCKETS:
        sub = agg[agg["adp_bucket"] == label]
        if sub.empty:
            continue
        print(f"\n  ADP {label}:")
        print(sub[["position", "n", "mean_vor", "median_vor", "empirical_multiplier", "shipped_multiplier"]]
              .round(3).to_string(index=False))

    print("\n=== THE TE CLAIM ===")
    early = d[d["adp_bucket"].isin(["1-12", "13-24", "25-48"])]
    by_pos = early.groupby("position")["vor"].agg(["size", "mean", "median"]).round(2)
    print("Early rounds (ADP 1-48), realized VOR by position:")
    print(by_pos.to_string())
    if "TE" in by_pos.index:
        te_mult = by_pos.loc["TE", "mean"] / by_pos["mean"].max()
        print(f"\nEmpirical early-round TE multiplier: {te_mult:.3f}")
        print(f"Shipped TE multiplier:               {SHIPPED_VALUE_MULTIPLIERS['TE']:.3f}")
        verdict = "TOO HARSH -- the app penalizes TE more than history justifies" if te_mult > SHIPPED_VALUE_MULTIPLIERS["TE"] else "roughly justified"
        print(f"-> {verdict}")

    wf = walk_forward_te_check(d)
    if not wf.empty:
        print("\n=== Walk-forward stability (no hindsight) ===")
        print(wf.round(3).to_string(index=False))
        print(f"\nMean predicted TE multiplier: {wf['predicted_te_multiplier'].mean():.3f} | "
              f"mean realized: {wf['realized_te_multiplier'].mean():.3f}")


if __name__ == "__main__":
    main()
