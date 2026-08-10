"""
Focused Step 12 test: does route participation / TPRR add lift over ADP?

The full evaluate_{pos}_models.py sweep fits ~14 feature groups x 3
estimators x 5 targets x 27 seasons and takes 10-20 minutes per position.
Three separate attempts to run it in the background were killed by process
teardown before writing results. The Step 12 question needs exactly two
groups on one target per position, so this runs the narrow comparison
directly -- seconds instead of tens of minutes, and no background process
to lose.

Same harness primitives (`fit_predict`, `hit_rate`,
`top_decile_threshold_count`) and same walk-forward discipline as the full
evaluators, so the numbers are comparable to
`workload_v1_step4_summary.csv`. QB is excluded: route features are empty
there by design (a QB is on the field for every pass play, so his route
rate is ~1.0 and TPRR is exactly 0.000).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import (
    VALIDATION_DIR,
    adp_baseline_scores,
    fit_predict,
    hit_rate,
    top_decile_threshold_count,
)
from workload_feature_groups import ADP_BASELINE, ROUTE

DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
PRIMARY_TARGET = {"RB": "RB_Top24", "WR": "WR_Top24", "TE": "TE_Top12"}
KINDS = ("logistic", "regularized_logistic", "random_forest")

GROUPS = {
    "adp_only_baseline": ADP_BASELINE,
    "adp_route_v1": ADP_BASELINE + ROUTE,
    "route_v1_only": ROUTE,
}


def season_lift(df: pd.DataFrame, target: str, features: list[str], kind: str) -> dict[int, float]:
    """Walk-forward top-decile lift over the ADP baseline, per test season."""
    out: dict[int, float] = {}
    seasons = sorted(df["season"].dropna().astype(int).unique())
    for test_season in seasons:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if train.empty or len(test) < 10:
            continue
        baseline = adp_baseline_scores(test)
        if baseline.notna().sum() == 0:
            continue

        top_n = top_decile_threshold_count(len(test))
        if not top_n:
            continue
        adp_top = test.assign(_b=baseline).sort_values("_b", ascending=False).head(top_n)
        base_hit = hit_rate(adp_top, target)
        if pd.isna(base_hit):
            continue

        scores, status = fit_predict(train, test, target, features, kind)
        if status != "fit" or scores.notna().sum() == 0:
            continue
        model_top = test.assign(_s=scores).sort_values("_s", ascending=False).head(top_n)
        model_hit = hit_rate(model_top, target)
        if pd.isna(model_hit):
            continue
        out[test_season] = float(model_hit - base_hit)
    return out


def main() -> None:
    data = pd.read_csv(DATASET, low_memory=False)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    rng = np.random.default_rng(0)

    print(f"Route feature coverage: "
          f"rate={int(data['prior_route_participation_rate'].notna().sum())} "
          f"tprr={int(data['prior_targets_per_route_run'].notna().sum())} of {len(data)} rows\n")

    for position, target in PRIMARY_TARGET.items():
        df = data[data["position"].eq(position)].copy()
        if target not in df.columns:
            continue
        print(f"=== {position} ({target}) ===")

        # Average across estimators, mirroring how summarize_workload_validation
        # groups model_name -> feature group.
        per_group: dict[str, dict[int, float]] = {}
        for name, feats in GROUPS.items():
            feats = [f for f in feats if f in df.columns]
            if not feats:
                continue
            acc: dict[int, list[float]] = {}
            for kind in KINDS:
                for season, lift in season_lift(df, target, feats, kind).items():
                    acc.setdefault(season, []).append(lift)
            per_group[name] = {s: float(np.mean(v)) for s, v in acc.items()}

        for name, lifts in per_group.items():
            if lifts:
                print(f"  {name:22s} mean lift vs ADP = {np.mean(list(lifts.values())):+.4f} "
                      f"({len(lifts)} seasons)")

        # Head-to-head on the seasons both groups actually ran.
        a, b = "adp_only_baseline", "adp_route_v1"
        if a in per_group and b in per_group:
            shared = sorted(set(per_group[a]) & set(per_group[b]))
            if shared:
                diffs = np.array([per_group[b][s] - per_group[a][s] for s in shared])
                boots = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(20000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                verdict = "ROUTE HELPS" if (diffs.mean() >= 0.01 and (diffs > 0).mean() > 0.5 and lo > 0) \
                    else "no evidence"
                print(f"  -> paired delta = {diffs.mean():+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
                      f"wins {(diffs > 0).mean():.0%} of {len(diffs)} matched seasons  -> {verdict}")
        print()


if __name__ == "__main__":
    main()
