"""
Step 4b: divergence features -- explicit MISPRICING measures.

Step 4 tested opportunity as levels ("is this player good?"), which ADP
already prices, and found no lift at any position. These features instead
ask "is this player better than his price / better than his results
suggest?" -- each one is a percentile rank minus another percentile rank,
computed within (season, position).

Mechanism (see research/archetype_engine_design.md, candidates N-1/N-2/
WR-1/N-7): year-over-year, opportunity is sticky and efficiency reverts to
the mean. Two WRs who both finish with 900 yards -- one on 150 targets, one
on 80 -- are priced alike by the market but have very different outlooks.
The high-volume/low-efficiency player is the buy; the inverse is the fade.

Leakage: every input is a prior_* column (prior season) or ADP (known
pre-draft). Percentile ranks are computed within a season across players,
which uses only that season's pre-draft-known values -- no outcome data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

INPUT_PATH = VALIDATION_DIR / "predraft_validation_dataset_workload_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "predraft_validation_dataset_divergence_v1.csv"

DIVERGENCE_COLS = [
    "opportunity_percentile",
    "efficiency_percentile",
    "div_opportunity_minus_efficiency",
    "div_wopr_minus_adp",
    "div_redzone_minus_overall_share",
]

# Minimum prior-season volume before an efficiency rate is trustworthy. A
# player with 3 targets and 2 catches has a gaudy catch rate that means
# nothing; including him would put pure noise at the top of the efficiency
# percentile and corrupt the divergence signal.
MIN_VOLUME_FOR_EFFICIENCY = 20


def _pct_within_season_position(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank of `col` within each (season, position) cohort."""
    values = pd.to_numeric(df[col], errors="coerce")
    return values.groupby([df["season"], df["position"]]).rank(pct=True)


def _mean_of_available(frame: pd.DataFrame) -> pd.Series:
    """Row-wise mean over whichever component percentiles exist.

    Deliberately NOT fillna(0) -- a missing component should not be read as
    'bottom of the league'. Rows with no components at all stay NaN and the
    downstream model's median imputation handles them.
    """
    return frame.mean(axis=1, skipna=True)


def build_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    targets = pd.to_numeric(out.get("prior_targets_pbp"), errors="coerce")
    carries = pd.to_numeric(out.get("prior_carries_pbp"), errors="coerce")
    rec_yards = pd.to_numeric(out.get("prior_receiving_yards"), errors="coerce")
    rush_yards = pd.to_numeric(out.get("prior_rushing_yards"), errors="coerce")
    total_tds = pd.to_numeric(out.get("prior_total_tds"), errors="coerce")
    total_touches = targets.fillna(0) + carries.fillna(0)

    # ---------- Efficiency block (per-touch results) ----------
    enough_volume = total_touches >= MIN_VOLUME_FOR_EFFICIENCY
    out["eff_yards_per_target"] = (rec_yards / targets.replace(0, np.nan)).where(enough_volume)
    out["eff_yards_per_carry"] = (rush_yards / carries.replace(0, np.nan)).where(enough_volume)
    out["eff_td_rate"] = (total_tds / total_touches.replace(0, np.nan)).where(enough_volume)

    efficiency_pcts = pd.DataFrame(index=out.index)
    for col in ["eff_yards_per_target", "eff_yards_per_carry", "eff_td_rate"]:
        efficiency_pcts[col] = _pct_within_season_position(out, col)
    out["efficiency_percentile"] = _mean_of_available(efficiency_pcts)

    # ---------- Opportunity block (volume / role) ----------
    out["opp_total_touches"] = total_touches.where(total_touches > 0)
    opportunity_pcts = pd.DataFrame(index=out.index)
    for col in ["prior_wopr", "prior_snap_share", "prior_target_share", "opp_total_touches"]:
        if col in out.columns:
            opportunity_pcts[col] = _pct_within_season_position(out, col)
    out["opportunity_percentile"] = _mean_of_available(opportunity_pcts)

    # ---------- Divergence 1: opportunity vs. efficiency (N-1 / N-2) ----------
    # Positive = lots of work, poor per-touch results -> volume persists,
    # bad luck reverts -> BUY. Negative = the fade.
    out["div_opportunity_minus_efficiency"] = (
        out["opportunity_percentile"] - out["efficiency_percentile"]
    )

    # ---------- Divergence 2: real workload vs. market price (WR-1) ----------
    # ADP is "lower is better", so invert it before ranking so that a high
    # percentile means an expensive player in both series.
    adp = pd.to_numeric(out.get("overall_adp"), errors="coerce")
    if adp.isna().all():
        adp = pd.to_numeric(out.get("preseason_adp"), errors="coerce")
    out["_adp_inverted"] = -adp
    adp_pct = _pct_within_season_position(out, "_adp_inverted")
    wopr_pct = _pct_within_season_position(out, "prior_wopr")
    # Positive = real workload the market is pricing cheaply -> BUY.
    out["div_wopr_minus_adp"] = wopr_pct - adp_pct

    # ---------- Divergence 3: red zone vs. overall usage (N-7) ----------
    # Negative = gets team targets but not red-zone looks -> TD-starved,
    # positive-regression candidate. Positive = TD-leveraged, higher
    # variance and a TD-regression risk.
    rz_pct = _pct_within_season_position(out, "prior_redzone_target_share")
    share_pct = _pct_within_season_position(out, "prior_target_share")
    out["div_redzone_minus_overall_share"] = rz_pct - share_pct

    return out.drop(columns=["_adp_inverted"])


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found -- run build_workload_features_v1.py first."
        )
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    out = build_divergence_features(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Divergence dataset written: {OUTPUT_PATH}")
    print(f"Rows: {len(out)}")
    for col in DIVERGENCE_COLS:
        series = out[col]
        print(
            f"  {col:35s} notna={int(series.notna().sum()):6d}  "
            f"mean={series.mean():+.4f}  sd={series.std():.4f}"
        )


if __name__ == "__main__":
    main()
