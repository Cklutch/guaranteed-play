"""Breakout Score V1 -- trimmed-feature re-validation (2026-08-27).

build_breakout_robustness_checks_v1.py found div_opportunity_minus_efficiency's
coefficient sign only 57% consistent across the 7 WR/FULL walk-forward folds
(vs. 100% for the other six features) -- closer to noise than signal at this
sample size. This re-runs the full battery (nested comparison, permutation
test, leave-one-season-out, threshold sensitivity) on a 6-feature spec with
that one feature dropped, to check the cleanest of the three options laid
out for it: does removing it barely move performance (expected, if the
other six carry the signal) or hurt it (would argue for keeping it despite
the instability)?

This does NOT overwrite FEATURES_FULL or the original validation report --
that result stays as the frozen historical record. This is a genuinely new,
separately reported comparison.

Run: python research/validation_v1/build_breakout_trimmed_feature_check_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import pandas as pd

from build_breakout_validation_v1 import (  # noqa: E402
    ADP_CONTINUOUS, FEATURES_FULL, SEED, load, nested_comparison,
    season_bootstrap,
)
from build_breakout_robustness_checks_v1 import (  # noqa: E402
    N_PERMUTATIONS, check_2_leave_one_season_out,
    check_4_threshold_sensitivity, topdecile_delta_for_labels,
)
from build_breakout_validation_v1 import folds, make_pipe, assert_fold_contract, assert_no_train_test_overlap  # noqa: E402

POSITION = "WR"
START_SEASON = 2017
DROPPED_FEATURE = "div_opportunity_minus_efficiency"
FEATURES_TRIMMED = [f for f in FEATURES_FULL if f != DROPPED_FEATURE]


def collect_fold_predictions_trimmed():
    d = load(POSITION)
    expanded = ADP_CONTINUOUS + FEATURES_TRIMMED
    schema_ref = []
    fold_data = []
    for ts, tr, te in folds(d, START_SEASON):
        assert_no_train_test_overlap(tr, te)
        assert_fold_contract(tr, expanded, ts, schema_ref)
        base = make_pipe().fit(tr[ADP_CONTINUOUS], tr["is_breakout"])
        exp = make_pipe().fit(tr[expanded], tr["is_breakout"])
        pb = base.predict_proba(te[ADP_CONTINUOUS])[:, 1]
        pe = exp.predict_proba(te[expanded])[:, 1]
        y = te["is_breakout"].to_numpy()
        fold_data.append({"season": ts, "y": y, "pb": pb, "pe": pe, "n": len(te)})
    return fold_data, d


def main() -> int:
    rng = np.random.default_rng(SEED)
    fold_data, d = collect_fold_predictions_trimmed()
    print(f"WR / FULL_TRIMMED (dropped {DROPPED_FEATURE}), {len(fold_data)} test seasons\n")

    result, _, _, _ = nested_comparison(d, FEATURES_TRIMMED, START_SEASON, "FULL_TRIMMED", POSITION, rng)
    summary = result["summary"]
    print("=== Nested comparison vs. original 7-feature FULL ===")
    print(f"{'metric':<20}{'FULL (7 feat, frozen)':<26}{'FULL_TRIMMED (6 feat)':<26}")
    print(f"{'AUC delta':<20}{'+0.0350 [-0.0068,+0.0764]':<26}"
          f"{summary['auc_delta']['mean']:+.4f} [{summary['auc_delta']['ci_lo']:+.4f},{summary['auc_delta']['ci_hi']:+.4f}]")
    print(f"{'Top-decile delta':<20}{'+11.9% [+7.1%,+16.7%]':<26}"
          f"{summary['topdecile_delta']['mean']:+.1%} [{summary['topdecile_delta']['ci_lo']:+.1%},{summary['topdecile_delta']['ci_hi']:+.1%}]")

    observed_mean, _ = topdecile_delta_for_labels(fold_data, "y")
    null_means = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm_fold_data = [{**f, "y_perm": rng.permutation(f["y"])} for f in fold_data]
        m, _ = topdecile_delta_for_labels(perm_fold_data, "y_perm")
        null_means[i] = m
    p_value = float((null_means >= observed_mean).mean())
    print(f"\n=== Permutation test (trimmed) ===")
    print(f"Observed delta: {observed_mean:+.4f}  P(null >= observed): {p_value:.4f}")
    bonf_alpha = 0.05 / 6
    print(f"Bonferroni family-wise threshold (6 tests): {bonf_alpha:.4f}  "
          f"{'CLEARS' if p_value < bonf_alpha else 'does not clear'} it")

    loo_df, full_mean = check_2_leave_one_season_out(fold_data)
    print(f"\n=== Leave-one-season-out (trimmed) ===")
    print(f"Full mean: {full_mean:+.4f}")
    print(loo_df.to_string(index=False))

    thresh_df = check_4_threshold_sensitivity(fold_data, rng)
    print(f"\n=== Threshold sensitivity (trimmed) ===")
    print(thresh_df.to_string(index=False))

    report_path = VALIDATION_DIR / "breakout_v1_trimmed_feature_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Breakout Score V1 -- Trimmed-Feature Re-validation\n\n")
        f.write(f"Dropped `{DROPPED_FEATURE}` (57% coefficient-sign consistency in the original "
                "robustness check) and re-ran the full battery on the remaining 6 features + ADP. "
                "The original 7-feature FULL result stays the frozen historical record in "
                "breakout_v1_validation_report.md; this is a separate comparison.\n\n")
        f.write("## Nested comparison\n\n")
        f.write("| metric | FULL (7 feat, frozen) | FULL_TRIMMED (6 feat) |\n|---|---|---|\n")
        f.write(f"| AUC delta | +0.0350 [-0.0068,+0.0764] | "
                f"{summary['auc_delta']['mean']:+.4f} [{summary['auc_delta']['ci_lo']:+.4f},{summary['auc_delta']['ci_hi']:+.4f}] |\n")
        f.write(f"| Top-decile delta | +11.9% [+7.1%,+16.7%] | "
                f"{summary['topdecile_delta']['mean']:+.1%} [{summary['topdecile_delta']['ci_lo']:+.1%},{summary['topdecile_delta']['ci_hi']:+.1%}] |\n\n")
        f.write(f"## Permutation test\n\nObserved delta: {observed_mean:+.4f}. "
                f"P(null >= observed), one-sided: {p_value:.4f}. "
                f"Bonferroni family-wise threshold across the 6 originally tested combinations: {bonf_alpha:.4f} -- "
                f"{'clears' if p_value < bonf_alpha else 'does not clear'} it.\n\n")
        f.write("## Leave-one-season-out\n\n")
        f.write(f"Full mean: {full_mean:+.4f}\n\n" + loo_df.to_markdown(index=False) + "\n\n")
        f.write("## Threshold sensitivity\n\n")
        f.write(thresh_df.to_markdown(index=False) + "\n")

    print(f"\nReport: {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
