"""projection_model_iteration_plan.pdf, Iteration 7: market-mispricing signal.

Compares the model's own opinion of a player (model_prediction) to where
ADP thinks they belong (adp_only_prediction) -- both already real,
validated outputs of run_position() reused as-is, not rebuilt. Restricted
to the real ADP-covered subset (same constraint as every ADP comparison
since Iteration 3).

market_gap = model_prediction - adp_only_prediction
  positive -> model likes this player more than the market ("undervalued")
  negative -> model likes this player less than the market ("overvalued")

Two validations, not one, per the standard already locked in tonight for
this exact kind of claim (draft_deanchor plan, Amendment 2): the real ask
is whether divergence from ADP correlates directionally with real outcomes
more than chance, NOT whether flagged players outright beat ADP's own
accuracy -- a different, harder claim nobody asked for here.

  1. Plan's own literal template: top/bottom tercile of market_gap per
     position-season flagged as undervalued/overvalued; check whether each
     group's real outcome landed on the correct side of adp_only_prediction,
     and by how much, on average.
  2. Directional-correlation check (Amendment 2 standard): Spearman
     correlation between market_gap and (actual - adp_only_prediction)
     across ALL ADP-subset rows, not just the flagged extremes -- does the
     model's degree of disagreement with the market predict the real
     direction/size of ADP's miss. Checked per-season and reported via the
     locked-in win-count-majority rule (majority of the 5 real seasons with
     a positive correlation = signal holds), not a single pooled number.

Usage:
    python build_market_mispricing_v1.py
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from build_baseline_projection_v1 import VALIDATION_TEST_SEASONS, run_position
from build_trend_layer_v1 import build_features_with_trend, current_best_features
from validation_utils import VALIDATION_DIR

REPORT_PATH = VALIDATION_DIR / "market_mispricing_v1_report.csv"
FLAGGED_PATH = VALIDATION_DIR / "market_mispricing_v1_flagged.csv"

POSITIONS = ["QB", "RB", "WR", "TE"]
TERCILE = 1 / 3

# Locked in per review, before any live wiring: market_gap is NOT
# symmetrically trustworthy. Validated directly (both the aggregate stats
# AND a real-player spot check -- see module docstring) that "undervalued"
# (buy-side) carries real signal only at RB; at WR/TE it's statistically
# indistinguishable from noise (+0.04/+0.09 ppg) even though the position's
# overall correlation is positive, because that correlation is being
# carried almost entirely by the sell side. QB has no usable signal in
# EITHER direction -- its "overvalued" flag is actively backwards (1/5
# seasons positive correlation; spot-check found the model calling real,
# well-known QB breakouts -- Jordan Love 2023, Jalen Hurts 2021 -- sells).
# Same gating pattern as POSITIONS_WITH_TREND_LAYER (build_trend_layer_v1.py):
# a position not being validated for a given direction means that
# combination is OFF, not silently degraded. Any live surface (Iteration 8)
# must route through is_signal_trustworthy() rather than displaying
# market_gap as one symmetric number.
#
# RB's buy side spot-checked SEPARATELY from the sell side (per review --
# the sell-side examples below don't by themselves ground the buy claim):
# 48/80 (60%) of all RB undervalued flags landed correctly, real but
# meaningfully weaker than the sell side's dramatic, recurring pattern.
# Strongest real wins: James Conner 2021 (predicted 10.7 vs ADP 7.7, actual
# 15.9 -- real post-Pittsburgh bounce-back with Arizona), Alvin Kamara 2023
# (real suspension-depressed-ADP bounce-back). Real misses exist too (Najee
# Harris 2025, Raheem Mostert 2024) -- the +1.15ppg average is a genuine net
# positive built from a mixed 60% hit rate, not a one-sided story.
#
# CAVEAT: this gate is validated against the CURRENT Ridge-based
# model_prediction (build_baseline_projection_v1.py / build_trend_layer_v1.py)
# specifically. If a future iteration swaps in a different model class
# (tree-based, ensemble) or a meaningfully changed feature set, market_gap
# itself changes and there is no guarantee this same asymmetry (RB both
# directions, WR/TE sell-only, QB excluded) still holds. Re-validate this
# gate against the new model's own real residuals/flags before carrying it
# forward -- do not assume it transfers.
MARKET_SIGNAL_GATES = {
    "QB": set(),                          # no direction survived validation -- fully excluded
    "RB": {"undervalued", "overvalued"},  # both directions real: 5/5 seasons each; spot-check: Taylor sell x3 real busts, Conner/Kamara buy real wins (60% overall buy-side hit rate)
    "WR": {"overvalued"},                 # sell-side only: 5/5 seasons, +1.05ppg; spot-check: Thomas x2, Moore x2 real misses on the sell side
    "TE": {"overvalued"},                 # sell-side only, and thinner than WR (3/5 seasons) -- carries TE's standing sample-size caveat
}


def is_signal_trustworthy(position: str, flag: str) -> bool:
    """Whether a given (position, undervalued/overvalued) combination
    cleared validation and is safe to surface live. flag="neutral" is
    always False (nothing to surface)."""
    return flag in MARKET_SIGNAL_GATES.get(position, set())


def _collect_adp_subset(df: pd.DataFrame, position: str) -> pd.DataFrame:
    parts = []
    for test_season in VALIDATION_TEST_SEASONS:
        test, report = run_position(df, position, test_season=test_season, extra_features=current_best_features(position))
        if report["status"] != "fit":
            continue
        sub = test[test["adp_only_prediction"].notna()].copy()
        sub["position"] = position
        parts.append(sub[["season", "player_id", "player_name", "position", "final_fantasy_ppg", "model_prediction", "adp_only_prediction"]])
    out = pd.concat(parts, ignore_index=True)
    out["market_gap"] = out["model_prediction"] - out["adp_only_prediction"]
    out["adp_miss"] = out["final_fantasy_ppg"] - out["adp_only_prediction"]
    return out


def _flag_terciles(subset: pd.DataFrame) -> pd.DataFrame:
    """Terciles computed WITHIN each season -- avoids one unusually
    optimistic/pessimistic season skewing what counts as "undervalued"
    for another."""
    out = subset.copy()
    out["flag"] = "neutral"
    for season, group in out.groupby("season"):
        lower_cut = group["market_gap"].quantile(TERCILE)
        upper_cut = group["market_gap"].quantile(1 - TERCILE)
        out.loc[group.index[group["market_gap"] >= upper_cut], "flag"] = "undervalued"
        out.loc[group.index[group["market_gap"] <= lower_cut], "flag"] = "overvalued"
    return out


def main() -> None:
    df = build_features_with_trend()
    all_flagged = []
    report_rows = []

    print("Iteration 7 -- market-mispricing signal\n")
    for position in POSITIONS:
        subset = _collect_adp_subset(df, position)
        flagged = _flag_terciles(subset)
        flagged["trustworthy"] = flagged["flag"].apply(lambda f: is_signal_trustworthy(position, f))
        all_flagged.append(flagged)

        undervalued = flagged[flagged["flag"].eq("undervalued")]
        overvalued = flagged[flagged["flag"].eq("overvalued")]
        under_beat = undervalued["adp_miss"].mean()
        over_missed = -overvalued["adp_miss"].mean()  # sign-flip: report as "missed by X", X>0 means they underperformed as predicted

        # Per-season directional correlation, win-count-majority (Amendment 2 standard)
        season_corrs = []
        for season, group in subset.groupby("season"):
            if len(group) < 8:
                continue
            corr, _ = spearmanr(group["market_gap"], group["adp_miss"])
            season_corrs.append((season, corr))
        n_seasons = len(season_corrs)
        n_positive = sum(1 for _, c in season_corrs if c > 0)
        signal_holds = n_positive > n_seasons / 2 if n_seasons else False

        report_rows.append({
            "position": position,
            "n_undervalued": len(undervalued),
            "n_overvalued": len(overvalued),
            "undervalued_beat_adp_by_ppg": round(under_beat, 2),
            "overvalued_missed_by_ppg": round(over_missed, 2),
            "n_seasons_checked": n_seasons,
            "n_seasons_positive_corr": n_positive,
            "signal_holds_directionally": signal_holds,
            "gated_directions_live": sorted(MARKET_SIGNAL_GATES.get(position, set())) or "NONE (excluded)",
        })

        template_verdict = "signal holds" if (under_beat > 0 and over_missed > 0) else "isn't real yet"
        print(f"{position}:")
        print(
            f"  Undervalued players (n={len(undervalued)}) beat ADP expectation by {under_beat:+.2f} pts on average; "
            f"overvalued players (n={len(overvalued)}) missed by {over_missed:+.2f} pts. [{template_verdict}]"
        )
        print(
            f"  Directional correlation (market_gap vs real ADP-miss): positive in {n_positive}/{n_seasons} "
            f"individual seasons -> {'signal holds directionally' if signal_holds else 'not a majority -- unresolved'} "
            f"(per-season: {', '.join(f'{s}={c:+.2f}' for s, c in season_corrs)})"
        )
        gate = MARKET_SIGNAL_GATES.get(position, set())
        gate_desc = " + ".join(sorted(gate)) if gate else "NONE -- excluded entirely from any live surface"
        print(f"  LIVE GATE: {gate_desc}")
        print()

    pd.concat(all_flagged, ignore_index=True).to_csv(FLAGGED_PATH, index=False)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)
    print(f"Flagged players written: {FLAGGED_PATH}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
