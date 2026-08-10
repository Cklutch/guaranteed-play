"""
Where is the value cliff inside each position?

A single replacement baseline per position treats QB1 and QB8 as points on
one smooth line. Real drafts don't behave that way: premium QBs and TEs go
at or near their ADP because the elite tier is genuinely scarce, and then
the field slips to later rounds once the cliff is passed. Valuing those
positions correctly means finding where the cliff actually is, not
assuming a linear slope down to replacement.

Two views, because they answer different questions:

1. REALIZED CURVE -- mean fantasy points by finishing positional rank
   (QB1..QB24 etc.), season-normalized. This is the structural shape: how
   much scarcer is QB1 than QB12, really. It's hindsight, so it describes
   the position, not a draft strategy.

2. DRAFT-COST TIERS -- group players by the positional ADP they were
   DRAFTED at (QB1-3, QB4-6, ...) and measure what each tier actually
   returned. This is the actionable one: it answers "is paying up for a
   premium QB/TE worth it, and at which tier does it stop being worth it?"
   Unlike view 1 it uses only information available on draft day.

Scoring inflation across 2010-2025 is handled by expressing everything as
value over that season's own positional replacement level, so a 2012 point
and a 2024 point are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "positional_tier_cliffs.csv"

# Weekly starters in a 12-team league, FLEX split 40/40/20 across RB/WR/TE.
REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 14}

# Draft-cost tiers, by positional ADP rank.
ADP_TIERS = [
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    ("10-12", 10, 12),
    ("13-18", 13, 18),
    ("19-30", 19, 30),
]


def load() -> pd.DataFrame:
    d = pd.read_csv(DATASET_PATH, low_memory=False)
    d = d[d["final_fantasy_points"].notna() & d["position"].isin(REPLACEMENT_RANK)].copy()
    d["final_fantasy_points"] = pd.to_numeric(d["final_fantasy_points"], errors="coerce")
    d["positional_adp"] = pd.to_numeric(d["positional_adp"], errors="coerce")

    # Realized finish rank within (season, position), computed here rather
    # than trusting final_positional_finish, so the ranking basis matches
    # the points column being used.
    d["finish_rank"] = d.groupby(["season", "position"])["final_fantasy_points"].rank(
        ascending=False, method="first"
    )

    baselines = {}
    for (season, position), grp in d.groupby(["season", "position"]):
        pts = grp["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        n = REPLACEMENT_RANK[position]
        baselines[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])
    d["replacement"] = [baselines[(s, p)] for s, p in zip(d["season"], d["position"])]
    d["vor"] = d["final_fantasy_points"] - d["replacement"]
    return d


def realized_curve(d: pd.DataFrame, position: str, max_rank: int = 20) -> pd.DataFrame:
    """Mean VOR at each finishing rank -- the structural scarcity shape."""
    sub = d[(d["position"] == position) & (d["finish_rank"] <= max_rank)]
    curve = sub.groupby("finish_rank")["vor"].agg(["mean", "size"]).reset_index()
    curve = curve.rename(columns={"mean": "mean_vor", "size": "n"})
    curve["drop_from_prev"] = -curve["mean_vor"].diff()
    return curve


def cost_tiers(d: pd.DataFrame, position: str) -> pd.DataFrame:
    """What each DRAFT-COST tier actually returned. Draft-day information only."""
    sub = d[(d["position"] == position) & d["positional_adp"].notna()].copy()
    rows = []
    for label, lo, hi in ADP_TIERS:
        tier = sub[(sub["positional_adp"] >= lo) & (sub["positional_adp"] <= hi)]
        if tier.empty:
            continue
        rows.append({
            "position": position,
            "drafted_as": label,
            "n": len(tier),
            "mean_vor": tier["vor"].mean(),
            "median_vor": tier["vor"].median(),
            # Did the pick return a top-tier finish at its position?
            "hit_top3_rate": float((tier["finish_rank"] <= 3).mean()),
            "hit_top6_rate": float((tier["finish_rank"] <= 6).mean()),
            "bust_below_replacement": float((tier["vor"] < 0).mean()),
            "mean_finish": tier["finish_rank"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    print(f"Player-seasons: {len(d)} ({int(d['season'].min())}-{int(d['season'].max())})")

    all_tiers = []
    for position in ["QB", "TE", "RB", "WR"]:
        print(f"\n{'=' * 66}")
        print(f"  {position}")
        print("=" * 66)

        curve = realized_curve(d, position)
        print("\n  Realized scarcity curve (mean VOR by finishing rank):")
        show = curve.head(14).copy()
        show.columns = ["Finish", "Mean VOR", "n", "Drop vs prev"]
        print(show.round(1).to_string(index=False))
        # Largest single-step drop inside the startable range = the cliff.
        inner = curve[(curve["finish_rank"] >= 2) & (curve["finish_rank"] <= 14)]
        if not inner.empty and inner["drop_from_prev"].notna().any():
            cliff = inner.loc[inner["drop_from_prev"].idxmax()]
            print(f"  -> steepest drop: {int(cliff['finish_rank']) - 1} to "
                  f"{int(cliff['finish_rank'])} ({cliff['drop_from_prev']:.1f} pts)")

        tiers = cost_tiers(d, position)
        if not tiers.empty:
            print("\n  What each DRAFT-COST tier actually returned:")
            t = tiers.copy()
            t = t[["drafted_as", "n", "mean_vor", "median_vor", "hit_top3_rate",
                   "hit_top6_rate", "bust_below_replacement", "mean_finish"]]
            t.columns = ["Drafted as", "n", "Mean VOR", "Med VOR", "Top3%", "Top6%", "Bust%", "Avg finish"]
            print(t.round(2).to_string(index=False))
            all_tiers.append(tiers)

    if all_tiers:
        combined = pd.concat(all_tiers, ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"\nWritten: {OUTPUT_PATH}")

        print("\n" + "=" * 66)
        print("  PREMIUM vs FIELD -- mean VOR by draft-cost tier")
        print("=" * 66)
        piv = combined.pivot_table(index="drafted_as", columns="position", values="mean_vor")
        piv = piv.reindex([label for label, _, _ in ADP_TIERS])
        print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
