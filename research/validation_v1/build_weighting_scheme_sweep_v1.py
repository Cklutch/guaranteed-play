"""projection_model_iteration_plan.pdf, Iteration 4: pick the best
season-weighting scheme.

Nothing structurally new versus Iteration 3 (build_baseline_projection_v1.py)
-- same features, same Ridge model. Only the 3-season recency weights
change, swept across four schemes:

  equal            [0.333, 0.333, 0.333]  -- no recency preference at all
  moderate_recency [0.5,   0.3,   0.2  ]  -- Iteration 3's original default
  strong_recency   [0.7,   0.2,   0.1  ]  -- heavy weight on most recent season
  smooth_decline   [0.571, 0.286, 0.143]  -- geometric decay, ratio 0.5 between
                                             each step (1 : 0.5 : 0.25,
                                             normalized) -- structurally
                                             different from the other two
                                             hand-picked schemes: a constant
                                             ratio rather than an arbitrary jump

Multi-season, not single-season: the first pass of this sweep used only the
2025 holdout and found QB/TE margins between schemes of 0.7-1.1% of the
best score -- thin enough to plausibly be noise. A second pass extended to
3 seasons (2023-2025) and DID flip both the QB and TE picks relative to the
single-season result -- direct proof the single-season picks weren't
reliable. This version extends the window again, to the same 5-season
window (2021-2025) build_baseline_projection_v1.py's own baseline-vs-ADP
comparison now uses, after that same 3-season window was shown to still
mislabel RB's ADP comparison as "essentially tied" when a real, majority-
confirmed loss was sitting one 2-season extension away. The 3-season
weighting locks were never re-checked against that same standard --
re-running them here closes that gap rather than assuming they still hold.

Winner selection is WIN-COUNT based (locked-in success criterion, matching
build_baseline_projection_v1.py's compare_model_versions()): the scheme
that wins a plurality of individual seasons is picked, with lowest average
MAE as a tiebreak only when win-counts are equal -- not lowest average MAE
outright, which is exactly the standard that let RB's ADP verdict hide a
real 4-of-5 loss behind a near-zero 3-season average.

Usage:
    python build_weighting_scheme_sweep_v1.py
"""

from __future__ import annotations

import pandas as pd

from build_baseline_projection_v1 import build_features, run_position
from validation_utils import VALIDATION_DIR

REPORT_PATH = VALIDATION_DIR / "weighting_scheme_sweep_v1_report.csv"

WEIGHTING_SCHEMES = {
    "equal": [1 / 3, 1 / 3, 1 / 3],
    "moderate_recency": [0.5, 0.3, 0.2],
    "strong_recency": [0.7, 0.2, 0.1],
    "smooth_decline": [4 / 7, 2 / 7, 1 / 7],
}
POSITIONS = ["QB", "RB", "WR", "TE"]
TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def main() -> None:
    rows = []
    for scheme_name, weights in WEIGHTING_SCHEMES.items():
        # Features (including the recency-weighted baseline for this scheme)
        # are season-agnostic -- computed once for the full 1999-2025 range,
        # then sliced per test season by run_position(). No need to rebuild
        # per test season.
        df = build_features(weights=weights)
        for position in POSITIONS:
            for test_season in TEST_SEASONS:
                _, report = run_position(df, position, test_season=test_season)
                if report["status"] != "fit":
                    rows.append({
                        "scheme": scheme_name, "position": position,
                        "test_season": test_season, "status": report["status"],
                    })
                    continue
                rows.append({
                    "scheme": scheme_name,
                    "position": position,
                    "test_season": test_season,
                    "status": "fit",
                    "test_rows": report["test_rows"],
                    "model_mae": report["model_mae_all_test_rows"],
                })

    result = pd.DataFrame(rows)
    result.to_csv(REPORT_PATH, index=False)
    fit = result[result["status"] == "fit"].copy()

    print(f"Iteration 4 -- weighting scheme sweep, {len(TEST_SEASONS)} independent holdout seasons\n")
    for position in POSITIONS:
        pos = fit[fit["position"] == position]
        if pos.empty:
            print(f"{position}: insufficient_data across all test seasons\n")
            continue
        pivot = pos.pivot(index="test_season", columns="scheme", values="model_mae")
        pivot = pivot[list(WEIGHTING_SCHEMES.keys())]
        print(f"{position} (MAE per test season):")
        print(pivot.to_string())

        avg = pivot.mean(axis=0)
        n_seasons_with_data = len(pivot)
        wins_per_season = pivot.idxmin(axis=1)
        win_counts = wins_per_season.value_counts().reindex(WEIGHTING_SCHEMES.keys(), fill_value=0)
        max_wins = int(win_counts.max())
        contenders = win_counts[win_counts == max_wins].index.tolist()
        # Win-count is primary; average MAE only breaks ties between
        # schemes with an equal win count -- never overrides a win-count
        # difference, which is the exact bug being fixed here.
        best_scheme = min(contenders, key=lambda s: avg[s]) if len(contenders) > 1 else contenders[0]
        spread = avg.max() - avg.min()
        spread_pct = spread / avg.min() * 100.0
        tie_note = f" (tiebroken vs {[c for c in contenders if c != best_scheme]} by lower avg MAE)" if len(contenders) > 1 else ""
        print(f"  avg MAE across {n_seasons_with_data} seasons: " + ", ".join(f"{k}={v:.3f}" for k, v in avg.items()))
        print(f"  per-season winner by year: " + ", ".join(f"{yr}={sch}" for yr, sch in wins_per_season.items()))
        print(
            f"  -> best by win-count: {best_scheme} "
            f"(won {max_wins}/{n_seasons_with_data} individual seasons{tie_note}, "
            f"avg-MAE spread across schemes = {spread_pct:.1f}% of best)"
        )
        print()

    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
