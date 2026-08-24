"""projection_model_iteration_plan.pdf, Iteration 6: distributional layer
(a "typical range" per player, not a floor/ceiling).

Scope note, checked directly before building: the plan calls for a range
"built from resampling real historical game logs." Re-confirmed (again)
that no weekly/game-level production file exists anywhere in this repo --
searched directly for game_log/boxscore/box_score/by_game files, found
none; the only real game-level data of any kind is snap counts (already
used in Iteration 5). Resampling real game logs is therefore not possible
with data available here.

Adapted instead to resample this model's own real, out-of-sample prediction
ERRORS -- a defensible substitution, not a downgrade in rigor: what a
draft-day range needs to communicate is "how much should I trust this
specific point estimate," which is a property of the MODEL's real
historical accuracy, not of a player's raw game-to-game production
variance (which the model already partly explains away via features like
target_share/team context). Built from real residuals
(actual final_fantasy_ppg - Iteration 5's current_best_features()
prediction), collected across all 5 validation seasons using each
position's actual locked-in model (current_best_features() from
build_trend_layer_v1.py) -- not a separate, untested model.

Naming, fixed per review: this interval is the 25th-to-75th percentile of
real residuals -- an empirical interquartile range, matching the plan's own
explicit validation target ("actual outcome inside the range ~50% of the
time"). "Floor" and "ceiling" colloquially imply a rare-miss bound (broken
occasionally), but a 50%-coverage interval is BY DESIGN violated below the
low end ~25% of the time and above the high end ~25% of the time -- one
miss in four on each side, not rare. Calling that a "floor/ceiling" would
mislead anyone reading the output into trusting tighter bounds than the
real statistics support. Renamed to typical_low/typical_high ("typical
range") throughout. A genuine floor/ceiling -- a rare-miss band, e.g. the
10th/90th percentile -- would be a legitimate, separate addition later if
wanted; not built here since the plan's own target was specifically the
50% interval, and widening it silently would change what got validated.

Known simplification, not fixed here: the range is built from each
position's POOLED residual distribution, not conditioned on individual
player volatility (e.g. a rushing QB likely has a different real error
distribution than a pocket passer). Reasonable for a v1; worth knowing if
a specific player's range looks miscalibrated in practice.

Typical_low = prediction + 25th-percentile residual
Typical_high = prediction + 75th-percentile residual

Validation: leave-one-season-out, not a single split. For each of the 5
test seasons, the percentiles are computed from the OTHER 4 seasons' real
residuals only (never the season being evaluated), then checked against
that season's real outcomes -- avoids using a season's own errors to build
the range that's then validated against that same season. Pooled coverage
rate (fraction of real outcomes landing inside [typical_low, typical_high])
reported per position against the plan's own ~50% target.

Usage:
    python build_floor_ceiling_v1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_baseline_projection_v1 import VALIDATION_TEST_SEASONS, run_position
from build_trend_layer_v1 import build_features_with_trend, current_best_features
from validation_utils import VALIDATION_DIR

REPORT_PATH = VALIDATION_DIR / "floor_ceiling_v1_report.csv"
PREDICTIONS_PATH = VALIDATION_DIR / "floor_ceiling_v1_predictions.csv"

POSITIONS = ["QB", "RB", "WR", "TE"]
LOWER_PCT = 25.0
UPPER_PCT = 75.0


def _collect_predictions(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Real, out-of-sample (test, model_prediction, final_fantasy_ppg)
    rows for this position across all 5 validation seasons, using the
    ACTUAL locked-in model (current_best_features()) -- the same model
    Iteration 5 validated and current_best_model_report.csv already
    reflects, not a separate distributional model."""
    parts = []
    for test_season in VALIDATION_TEST_SEASONS:
        test, report = run_position(df, position, test_season=test_season, extra_features=current_best_features(position))
        if report["status"] != "fit":
            continue
        parts.append(test[["season", "player_id", "player_name", "final_fantasy_ppg", "model_prediction"]].copy())
    out = pd.concat(parts, ignore_index=True)
    out["residual"] = out["final_fantasy_ppg"] - out["model_prediction"]
    return out


def _leave_one_season_out_ranges(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_out_season in VALIDATION_TEST_SEASONS:
        train_residuals = predictions.loc[predictions["season"] != held_out_season, "residual"]
        if len(train_residuals) < 20:
            continue
        lower = np.percentile(train_residuals, LOWER_PCT)
        upper = np.percentile(train_residuals, UPPER_PCT)
        test_rows = predictions[predictions["season"] == held_out_season].copy()
        test_rows["typical_low"] = test_rows["model_prediction"] + lower
        test_rows["typical_high"] = test_rows["model_prediction"] + upper
        test_rows["inside_range"] = test_rows["final_fantasy_ppg"].between(test_rows["typical_low"], test_rows["typical_high"])
        test_rows["range_lower_residual"] = lower
        test_rows["range_upper_residual"] = upper
        rows.append(test_rows)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = build_features_with_trend()
    all_scored = []
    report_rows = []

    print("Iteration 6 -- typical range (25th-75th percentile of real residuals), leave-one-season-out, target coverage ~50%\n")
    for position in POSITIONS:
        predictions = _collect_predictions(df, position)
        scored = _leave_one_season_out_ranges(predictions)
        all_scored.append(scored)

        coverage = scored["inside_range"].mean() * 100.0
        avg_width = (scored["typical_high"] - scored["typical_low"]).mean()
        n = len(scored)
        if 45.0 <= coverage <= 55.0:
            verdict = "well-calibrated"
        elif coverage > 55.0:
            verdict = "too wide"
        else:
            verdict = "too narrow"
        report_rows.append({
            "position": position, "n_scored": n, "coverage_pct": round(coverage, 1),
            "avg_range_width_ppg": round(avg_width, 2), "verdict": verdict,
        })
        print(
            f"{position}: actual outcomes fell inside the typical range "
            f"{coverage:.1f}% of the time (target ~50%, n={n}) -- {verdict}. "
            f"Avg range width: {avg_width:.2f} ppg."
        )

    pd.concat(all_scored, ignore_index=True).to_csv(PREDICTIONS_PATH, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)
    print(f"\nPredictions written: {PREDICTIONS_PATH}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
