"""
Step 15: P(bust) -- a distributional target, not another feature.

Fourteen prior attempts all used the SAME objective: a binary "does he
finish top-N" label scored by top-decile hit rate. That framing can only
answer "does this feature beat ADP on average," and it is structurally
blind to anything happening outside the top ~6 players at a position.

This changes the objective instead of adding features.

The reason to believe it is that the project's own strongest result is
already distributional and was mislabeled. The same feature, same data,
opposite tails, produced very different reliability:

    fade (bottom decile):  -19.9 resid, 90% of seasons, clears every gate
    gem  (top decile):     +15.4 resid, 70% of seasons, misses

Decline is predictable; breakout is not. That is a statement about the
shape of the outcome distribution, and no average-lift test can express it.

DEFINITIONS
-----------
bust = finishes below that season's positional replacement level.
       Replacement uses weekly starters in a 12-team league with the FLEX
       split 40/40/20 across RB/WR/TE: QB12, RB29, WR29, TE14.

The comparison is always **within an ADP band**, never across the whole
pool. "This player busts" is trivially true of late picks and trivially
false of early ones; the only interesting question is whether he busts
more often than others drafted at the same price. This is the same control
that made the residual test work when the tilt tests failed.

METRIC
------
NOT top-decile hit rate -- it cannot see mid-round effects, which is where
the surviving fade signal lives (flagged players sit at ADP 90-140 and the
metric only inspects the top ~6). Instead:

  * AUC for separating busts from non-busts, computed WITHIN each ADP band
    so ADP itself cannot supply the separation.
  * Bust-rate lift: bust rate among flagged players minus bust rate among
    same-band peers.

Both are compared against an ADP-only baseline fit the same way, so any
credit ADP deserves is already subtracted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, auc_or_nan, fit_predict

DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT = VALIDATION_DIR / "bust_model_results.csv"

REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 14}

# Draft-cost bands. Wide enough to hold enough players per season to fit,
# narrow enough that "same price" is meaningful.
ADP_BANDS = [("1-36", 1, 36), ("37-84", 37, 84), ("85-150", 85, 150), ("151+", 151, 10**6)]

# Features already built and available. Deliberately reusing what exists --
# the hypothesis under test is that the OBJECTIVE was wrong, not that the
# features were missing.
BUST_FEATURES = [
    "div_opportunity_minus_efficiency",   # unsustainable efficiency -> the surviving fade signal
    "div_redzone_minus_overall_share",
    "prior_snap_share",
    "prior_durability_score",             # availability
    "prior_garbage_time_share",           # inflated prior-season volume
    "prior_targets_per_route_run",
    "prior_route_participation_rate",
    "prior_team_epa_pg_vs_baseline",      # team context due to regress
    "age",
]
ADP_ONLY = ["overall_adp", "positional_adp"]


def add_bust_label(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["final_fantasy_points"].notna() & d["position"].isin(REPLACEMENT_RANK)].copy()
    baselines = {}
    for (season, position), g in d.groupby(["season", "position"]):
        pts = g["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        n = REPLACEMENT_RANK[position]
        baselines[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])
    repl = [baselines[(s, p)] for s, p in zip(d["season"], d["position"])]
    d["is_bust"] = (d["final_fantasy_points"] < repl).astype(int)
    return d


def band_of(adp: float) -> str | None:
    for label, lo, hi in ADP_BANDS:
        if lo <= adp <= hi:
            return label
    return None


def main() -> None:
    d = pd.read_csv(DATASET, low_memory=False)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d = d[d["overall_adp"].notna()].copy()
    d = add_bust_label(d)
    d["band"] = d["overall_adp"].apply(band_of)
    d = d[d["band"].notna()]

    print("Bust rate by ADP band (sanity -- must rise with ADP or the label is wrong):")
    print(d.pivot_table(index="band", columns="position", values="is_bust", aggfunc="mean")
          .reindex([b for b, _, _ in ADP_BANDS]).round(2).to_string())
    print()

    feats = [f for f in BUST_FEATURES if f in d.columns]
    print(f"Features available: {len(feats)}/{len(BUST_FEATURES)}")
    print()

    rows = []
    rng = np.random.default_rng(0)
    for position in ["QB", "RB", "WR", "TE"]:
        pos_df = d[d["position"] == position]
        for band, _, _ in ADP_BANDS:
            cell = pos_df[pos_df["band"] == band]
            if len(cell) < 120 or cell["is_bust"].nunique() < 2:
                continue

            model_aucs, adp_aucs = [], []
            for test_season in sorted(cell["season"].dropna().astype(int).unique()):
                train = cell[cell["season"] < test_season]
                test = cell[cell["season"] == test_season]
                if len(train) < 60 or len(test) < 8 or test["is_bust"].nunique() < 2:
                    continue
                m_scores, m_status = fit_predict(train, test, "is_bust", feats, "regularized_logistic")
                a_scores, a_status = fit_predict(train, test, "is_bust", ADP_ONLY, "regularized_logistic")
                if m_status == "fit":
                    v = auc_or_nan(test["is_bust"], m_scores)
                    if pd.notna(v):
                        model_aucs.append(v)
                if a_status == "fit":
                    v = auc_or_nan(test["is_bust"], a_scores)
                    if pd.notna(v):
                        adp_aucs.append(v)

            if len(model_aucs) >= 6 and len(adp_aucs) >= 6:
                n = min(len(model_aucs), len(adp_aucs))
                diffs = np.array(model_aucs[:n]) - np.array(adp_aucs[:n])
                boots = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(10000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({
                    "position": position, "band": band, "n_players": len(cell), "seasons": n,
                    "bust_rate": round(float(cell["is_bust"].mean()), 3),
                    "model_auc": round(float(np.mean(model_aucs[:n])), 4),
                    "adp_auc": round(float(np.mean(adp_aucs[:n])), 4),
                    "auc_gain": round(float(diffs.mean()), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "beats": f"{(diffs > 0).mean():.0%}",
                })

    if not rows:
        print("No cells had enough data to fit.")
        return
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT, index=False)
    print("Can we predict WHO busts, among players drafted at the same price?")
    print("(AUC vs an ADP-only model fit identically -- gain is what ADP does NOT already know)\n")
    print(res.to_string(index=False))
    print(f"\nWritten: {OUTPUT}")
    sig = res[res["ci_lo"] > 0]
    print(f"\nCells where the CI excludes zero: {len(sig)} of {len(res)}")
    if not sig.empty:
        print(sig[["position", "band", "auc_gain", "ci_lo", "ci_hi", "beats"]].to_string(index=False))


if __name__ == "__main__":
    main()
