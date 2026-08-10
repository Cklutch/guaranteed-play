"""
Focused Step 13 test: does team offensive environment add lift over ADP?

Two claims tested separately so they cannot be confused:

  LEVEL     -- prior-season team pace/pass-rate/scoring/efficiency.
               The control. Expected to fail, because ADP plainly knows
               which offenses were good.
  REVERSION -- each metric minus the team's own trailing 3-year baseline.
               The hypothesis: team context is only ~40% persistent
               year-over-year, so if the market extrapolates last season
               harder than that, players on breakout offenses are
               overpriced and players on down offenses are underpriced.

Uses the focused-test shape rather than the full evaluator sweep. The full
sweep fits ~16 groups x 3 estimators x 5 targets x 27 seasons at 10-20 min
per position, and was killed by process teardown three times while running
in the background. Same harness primitives and the same walk-forward
discipline, so the numbers stay comparable.
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
from workload_feature_groups import ADP_BASELINE, OFFENSE_LEVEL, OFFENSE_REVERSION

DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
PRIMARY_TARGET = {"QB": "QB_Top12", "RB": "RB_Top24", "WR": "WR_Top24", "TE": "TE_Top12"}
KINDS = ("logistic", "regularized_logistic", "random_forest")

GROUPS = {
    "adp_only_baseline": ADP_BASELINE,
    "adp_offense_level_v1": ADP_BASELINE + OFFENSE_LEVEL,
    "adp_offense_reversion_v1": ADP_BASELINE + OFFENSE_REVERSION,
}


def season_lift(df: pd.DataFrame, target: str, features: list[str], kind: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for test_season in sorted(df["season"].dropna().astype(int).unique()):
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
        base_hit = hit_rate(test.assign(_b=baseline).nlargest(top_n, "_b"), target)
        if pd.isna(base_hit):
            continue
        scores, status = fit_predict(train, test, target, features, kind)
        if status != "fit" or scores.notna().sum() == 0:
            continue
        model_hit = hit_rate(test.assign(_s=scores).nlargest(top_n, "_s"), target)
        if pd.isna(model_hit):
            continue
        out[test_season] = float(model_hit - base_hit)
    return out


def main() -> None:
    data = pd.read_csv(DATASET, low_memory=False)
    data["season"] = pd.to_numeric(data["season"], errors="coerce")
    rng = np.random.default_rng(0)

    cov = {c: int(data[c].notna().sum()) for c in OFFENSE_REVERSION if c in data.columns}
    print(f"Reversion feature coverage: {cov}\n")

    for position, target in PRIMARY_TARGET.items():
        df = data[data["position"].eq(position)].copy()
        if target not in df.columns:
            continue
        print(f"=== {position} ({target}) ===")

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
            if per_group[name]:
                print(f"  {name:26s} mean lift vs ADP = "
                      f"{np.mean(list(per_group[name].values())):+.4f} "
                      f"({len(per_group[name])} seasons)")

        base = "adp_only_baseline"
        for cand in ("adp_offense_level_v1", "adp_offense_reversion_v1"):
            if base not in per_group or cand not in per_group:
                continue
            shared = sorted(set(per_group[base]) & set(per_group[cand]))
            if not shared:
                continue
            diffs = np.array([per_group[cand][s] - per_group[base][s] for s in shared])
            boots = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(20000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            wins = float((diffs > 0).mean())
            # All three gates, same bar every other feature was held to.
            passed = diffs.mean() >= 0.01 and wins > 0.5 and lo > 0
            print(f"    {cand:26s} delta={diffs.mean():+.4f} CI=[{lo:+.4f},{hi:+.4f}] "
                  f"wins {wins:.0%} of {len(diffs)} -> {'HELPS' if passed else 'no evidence'}")
        print()


if __name__ == "__main__":
    main()
