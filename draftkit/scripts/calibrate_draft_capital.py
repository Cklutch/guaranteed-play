"""Real draft-capital curve calibration (rookie_data_backfill_plan.pdf's
Task B, Step 1 -- built fresh; the PDF referenced calibrate_decay_constants()/
draft_capital_positional.py/CalibrationPoint, none of which existed anywhere
in this repo, confirmed via a full-text grep before writing this).

draft_capital_score()'s decay shape (100 * (1 / (1 + pick/k))) has always
used a single k=15 constant shared across every position -- an asserted
number, not fit against anything real. This script fits a real, separate k
per position from real historical draft data.

Real empirical target: for each position, and each real pick number P a
player at that position was actually drafted at (last 10 completed classes,
2016-2025, from raw_draft_picks.csv), the real "percentile of lateness" --
what share of that position's own real draftees over the same span were
picked AFTER P. This is a real, data-grounded, monotonically decreasing
0-100 curve, directly encoding each position's own real draft-capital decay
(QBs dry up fast after round 1; RB/WR/TE decay more gradually) -- not
invented. The existing decay shape is then fit to this real curve via
least-squares (scipy.optimize.curve_fit) per position, extracting a real k.

Usage:
    python -m draftkit.scripts.calibrate_draft_capital
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DRAFT_PICKS_CSV = REPO_ROOT / "research" / "validation_v1" / "data" / "raw_draft_picks.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "draft_capital_calibration.csv"

CALIBRATION_SEASONS = range(2016, 2026)  # last 10 real completed classes
POSITIONS = ("RB", "WR", "QB", "TE")
ORIGINAL_SHARED_K = 15.0  # the single, unfit constant this replaces


def decay_curve(pick, k):
    return 100 * (1 / (1 + pick / k))


def empirical_lateness_percentile(picks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each real observed pick P (sorted, deduped), what % of this
    position's real picks over the calibration window were picked AFTER P.
    Returns (unique real picks, real percentile-of-lateness at each)."""
    picks = np.sort(picks)
    total = len(picks)
    unique_picks = np.unique(picks)
    percentiles = np.array([100 * np.sum(picks > p) / total for p in unique_picks])
    return unique_picks, percentiles


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return round(1 - ss_res / ss_tot, 4) if ss_tot else float("nan")


def calibrate_draft_capital() -> pd.DataFrame:
    draft = pd.read_csv(DRAFT_PICKS_CSV)
    draft = draft[draft["season"].isin(CALIBRATION_SEASONS)]

    rows = []
    for position in POSITIONS:
        picks = draft.loc[draft["position"] == position, "pick"].dropna().to_numpy()
        x, y_empirical = empirical_lateness_percentile(picks)

        (fitted_k,), _ = curve_fit(decay_curve, x, y_empirical, p0=[15.0], bounds=(0.1, 500))
        fitted_k = round(float(fitted_k), 2)

        y_fitted = decay_curve(x, fitted_k)
        y_original = decay_curve(x, ORIGINAL_SHARED_K)

        rows.append({
            "position": position,
            "n_picks": len(picks),
            "n_classes": len(CALIBRATION_SEASONS),
            "fitted_k": fitted_k,
            "r_squared_fitted": _r_squared(y_empirical, y_fitted),
            "r_squared_original_shared_k": _r_squared(y_empirical, y_original),
        })

    return pd.DataFrame(rows)


def main() -> int:
    result = calibrate_draft_capital()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
