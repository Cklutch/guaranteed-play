"""Breakout Score V1 -- robustness checks on the one surviving finding (2026-08-27).

build_breakout_validation_v1.py found exactly one result that cleared a
season-blocked bootstrap CI: WR / FULL spec, top-decile hit rate delta
+11.9% [+7.1%,+16.7%], 71% of seasons improved. Flagged caveats at the time:
only 7 test seasons, and it's 1 of 6 tested position x spec combinations
with no multiple-comparison correction applied.

This script runs four standard, independent robustness checks against that
specific finding -- not a fresh feature search, a stress test of the one
result worth stress-testing:

1. PERMUTATION TEST -- the most rigorous check available and the one that
   implicitly answers the multiple-comparison concern. Shuffles actual
   breakout outcomes WITHIN each season (fixing the real per-season base
   rate, fixing the model's real predictions, breaking only the true
   pairing between them), recomputes the identical statistic thousands of
   times, and reports where the real observed effect falls against that
   null. A CI excluding zero says "not exactly zero"; a permutation p-value
   says "how often would a real-looking effect this large happen from
   chance alone, given this exact procedure and these exact sample sizes."

2. LEAVE-ONE-SEASON-OUT -- is the whole effect carried by one unusually
   good season, or does it survive with any single season dropped?

3. FEATURE COEFFICIENT SIGN CONSISTENCY -- the same robustness check
   WR_BUST_DECISION_MEMO.md (section 8, item 6) uses: a real effect should
   rely on features with a stable direction of association across training
   folds, not ones that flip sign fold to fold.

4. THRESHOLD SENSITIVITY -- does the effect hold at top 5%/15%/20%, or is
   it a fragile artifact of exactly the 10% cutoff?

None of this is a new feature search. If any of these come back weak, that
is evidence AGAINST the WR FULL finding, reported as such, not reason to
go looking for a different cut that works.

Run: python research/validation_v1/build_breakout_robustness_checks_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from build_breakout_validation_v1 import (  # noqa: E402
    ADP_CONTINUOUS, FEATURES_FULL, SEED, assert_fold_contract,
    assert_no_train_test_overlap, folds, load, make_pipe,
)

POSITION = "WR"
SPEC_LABEL = "FULL"
START_SEASON = 2017
FEATURES = FEATURES_FULL
EXPANDED = ADP_CONTINUOUS + FEATURES
N_PERMUTATIONS = 5000


def collect_fold_predictions():
    """Refit the exact WR/FULL pipeline per season, same as the main study --
    self-contained here rather than re-importing run state, so this script
    can be run independently and still reproduce the same real result."""
    d = load(POSITION)
    schema_ref = []
    fold_data = []
    fold_models = []
    for ts, tr, te in folds(d, START_SEASON):
        assert_no_train_test_overlap(tr, te)
        assert_fold_contract(tr, EXPANDED, ts, schema_ref)
        base = make_pipe().fit(tr[ADP_CONTINUOUS], tr["is_breakout"])
        exp = make_pipe().fit(tr[EXPANDED], tr["is_breakout"])
        pb = base.predict_proba(te[ADP_CONTINUOUS])[:, 1]
        pe = exp.predict_proba(te[EXPANDED])[:, 1]
        y = te["is_breakout"].to_numpy()
        fold_data.append({"season": ts, "y": y, "pb": pb, "pe": pe, "n": len(te)})
        fold_models.append({"season": ts, "model": exp})
    return fold_data, fold_models


def topdecile_delta_for_labels(fold_data, label_key="y", top_pct=0.10):
    """Per-season top-decile delta (expanded minus base), given whichever
    label array is under label_key -- real y for the observed statistic,
    a shuffled y for a permutation replicate."""
    deltas = []
    for f in fold_data:
        y = f[label_key]
        n = f["n"]
        top_n = max(3, int(round(n * top_pct)))
        top_base = np.argsort(-f["pb"])[:top_n]
        top_exp = np.argsort(-f["pe"])[:top_n]
        deltas.append(float(y[top_exp].mean()) - float(y[top_base].mean()))
    return float(np.mean(deltas)), deltas


def check_1_permutation_test(fold_data, rng) -> dict:
    observed_mean, _ = topdecile_delta_for_labels(fold_data, "y")
    null_means = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm_fold_data = []
        for f in fold_data:
            y_perm = rng.permutation(f["y"])
            perm_fold_data.append({**f, "y_perm": y_perm})
        mean_i, _ = topdecile_delta_for_labels(perm_fold_data, "y_perm")
        null_means[i] = mean_i
    p_value_one_sided = float((null_means >= observed_mean).mean())
    return {
        "observed_delta": round(observed_mean, 4),
        "null_mean": round(float(null_means.mean()), 4),
        "null_std": round(float(null_means.std()), 4),
        "null_p95": round(float(np.percentile(null_means, 95)), 4),
        "null_p99": round(float(np.percentile(null_means, 99)), 4),
        "p_value_one_sided": p_value_one_sided,
        "n_permutations": N_PERMUTATIONS,
    }


def check_2_leave_one_season_out(fold_data) -> pd.DataFrame:
    _, per_season_deltas = topdecile_delta_for_labels(fold_data, "y")
    seasons = [f["season"] for f in fold_data]
    rows = []
    for i, held_out in enumerate(seasons):
        remaining = [d for j, d in enumerate(per_season_deltas) if j != i]
        rows.append({
            "excluded_season": held_out,
            "mean_delta_without_this_season": round(float(np.mean(remaining)), 4),
        })
    full_mean = round(float(np.mean(per_season_deltas)), 4)
    df = pd.DataFrame(rows)
    df["shift_from_full_mean"] = (df["mean_delta_without_this_season"] - full_mean).round(4)
    return df, full_mean


def check_3_coefficient_sign_consistency(fold_models) -> pd.DataFrame:
    rows = []
    for fm in fold_models:
        pipe = fm["model"]
        lr: LogisticRegression = pipe.named_steps["model"]
        for feat, coef in zip(EXPANDED, lr.coef_[0]):
            rows.append({"season": fm["season"], "feature": feat, "coef": float(coef)})
    df = pd.DataFrame(rows)
    summary = df.groupby("feature")["coef"].agg(
        mean_coef=lambda s: round(float(s.mean()), 4),
        pct_positive=lambda s: round(float((s > 0).mean() * 100), 1),
        pct_negative=lambda s: round(float((s < 0).mean() * 100), 1),
    ).reset_index()
    summary["sign_consistency_pct"] = summary[["pct_positive", "pct_negative"]].max(axis=1)
    return summary.sort_values("sign_consistency_pct", ascending=False)


def check_4_threshold_sensitivity(fold_data, rng) -> pd.DataFrame:
    from build_breakout_validation_v1 import season_bootstrap
    rows = []
    for pct in (0.05, 0.10, 0.15, 0.20):
        mean_delta, per_season = topdecile_delta_for_labels(fold_data, "y", top_pct=pct)
        lo, hi = season_bootstrap(pd.Series(per_season), rng)
        wins = float((np.array(per_season) > 0).mean())
        rows.append({
            "top_pct": pct, "mean_delta": round(mean_delta, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "seasons_improved": round(wins, 3),
        })
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(f"Refitting {POSITION}/{SPEC_LABEL} ({START_SEASON}+)...")
    fold_data, fold_models = collect_fold_predictions()
    print(f"{len(fold_data)} test seasons: {[f['season'] for f in fold_data]}\n")

    print("=" * 74)
    print("1. PERMUTATION TEST")
    print("=" * 74)
    perm = check_1_permutation_test(fold_data, rng)
    print(f"Observed delta: {perm['observed_delta']:+.4f}")
    print(f"Null distribution ({perm['n_permutations']} reps): mean={perm['null_mean']:+.4f}, "
          f"std={perm['null_std']:.4f}, 95th pct={perm['null_p95']:+.4f}, 99th pct={perm['null_p99']:+.4f}")
    print(f"P(null >= observed), one-sided: {perm['p_value_one_sided']:.4f}")
    verdict_1 = "SURVIVES" if perm["p_value_one_sided"] < 0.05 else "DOES NOT SURVIVE"
    print(f"-> {verdict_1} at p<0.05")

    print(f"\n{'=' * 74}\n2. LEAVE-ONE-SEASON-OUT\n{'=' * 74}")
    loo_df, full_mean = check_2_leave_one_season_out(fold_data)
    print(f"Full mean (all seasons): {full_mean:+.4f}")
    print(loo_df.to_string(index=False))
    max_shift = loo_df["shift_from_full_mean"].abs().max()
    verdict_2 = "ROBUST" if max_shift < abs(full_mean) * 0.5 else "FRAGILE -- one season drives most of the effect"
    print(f"-> {verdict_2} (max shift from dropping any one season: {max_shift:+.4f})")

    print(f"\n{'=' * 74}\n3. FEATURE COEFFICIENT SIGN CONSISTENCY\n{'=' * 74}")
    sign_df = check_3_coefficient_sign_consistency(fold_models)
    print(sign_df.to_string(index=False))

    print(f"\n{'=' * 74}\n4. THRESHOLD SENSITIVITY\n{'=' * 74}")
    thresh_df = check_4_threshold_sensitivity(fold_data, rng)
    print(thresh_df.to_string(index=False))
    n_hold = int(thresh_df["excludes_zero"].sum())
    print(f"-> excludes zero at {n_hold} of {len(thresh_df)} tested thresholds")

    report_path = VALIDATION_DIR / "breakout_v1_robustness_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Breakout Score V1 -- Robustness Checks (WR / FULL top-decile finding)\n\n")
        f.write(
            "Four independent checks against the one result from breakout_v1_validation_report.md "
            "that cleared a season-blocked bootstrap CI. See this script's module docstring for what "
            "each checks and why.\n\n"
        )
        f.write("## 1. Permutation test\n\n")
        f.write(pd.DataFrame([perm]).to_markdown(index=False) + f"\n\n**Verdict: {verdict_1} at p<0.05.**\n\n")
        f.write("## 2. Leave-one-season-out\n\n")
        f.write(f"Full mean (all seasons): {full_mean:+.4f}\n\n")
        f.write(loo_df.to_markdown(index=False) + f"\n\n**Verdict: {verdict_2}.**\n\n")
        f.write("## 3. Feature coefficient sign consistency\n\n")
        f.write(sign_df.to_markdown(index=False) + "\n\n")
        f.write("## 4. Threshold sensitivity\n\n")
        f.write(thresh_df.to_markdown(index=False) + f"\n\n**Excludes zero at {n_hold} of {len(thresh_df)} thresholds tested.**\n")

    print(f"\n\nReport: {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
