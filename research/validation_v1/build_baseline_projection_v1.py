"""projection_model_iteration_plan.pdf, Iteration 3: season-level baseline
projection model (v1 core).

Per position (QB/RB/WR/TE), predicts final_fantasy_ppg (half-PPR, corrected
in Iteration 2) from three real, pre-draft-safe input groups:

  1. Talent baseline: a recency-weighted average of the player's own
     final_fantasy_ppg over up to the last 3 real seasons, renormalized over
     however many are actually available -- most 2nd/3rd-year players won't
     have all 3. Weights are now PER-POSITION, locked in by Iteration 4's
     multi-season sweep (build_weighting_scheme_sweep_v1.py, validated
     across 3 independent holdout seasons, not just one) against 4
     candidate schemes: QB=equal weight (genuine tie with moderate_recency,
     kept as the simpler default), RB/WR/TE=strong_recency [0.7,0.2,0.1]
     -- see POSITION_RECENCY_WEIGHTS below for the exact values and each
     scheme's real, measured MAE.
  2. Efficiency stats: position-specific real usage-quality signals from
     stats_player_reg_by_season (target_share/air_yards_share/wopr for
     WR/TE, carries for RB, attempts for QB), read from the season BEFORE
     the target season -- no same-season leakage.
  3. Team context: offense_environment_team_seasons.csv's prior_team_*
     columns (already trailing-season by construction), joined on
     (season, team).

Target is points-PER-GAME, not season total -- games-played uncertainty is
an injury/availability question this repo already handles separately via
draftkit/injury_history.py, not something this projection model should
re-solve by conflating "how good is this player" with "will they stay
healthy."

Model: Ridge regression per position (simple, interpretable, handles the
real collinearity between e.g. target_share and wopr better than plain OLS).
Not a from-scratch weight search -- that's what "v1 core" is: a real,
fitted baseline to compare later iterations against, not a finished model.

Validation: hold out the most recent complete season (2025), train on
everything before it (<=2024), report MAE vs two benchmarks -- a plain
unweighted historical average (ablation of this model's own recency
weighting) and an ADP-only regression (real market baseline).

Usage:
    python build_baseline_projection_v1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

from validation_utils import PROJECT_ROOT, VALIDATION_DIR

STATS_DIR = VALIDATION_DIR / "data" / "stats_player_reg_by_season"
OFFENSE_CSV = VALIDATION_DIR / "data" / "offense_environment_team_seasons.csv"
E6_PATH = VALIDATION_DIR / "predraft_validation_dataset.csv"
OUTPUT_PATH = VALIDATION_DIR / "baseline_projection_v1_predictions.csv"
REPORT_PATH = VALIDATION_DIR / "baseline_projection_v1_report.csv"

TEST_SEASON = 2025  # kept for backward compatibility (run_position()'s default); NOT the reported baseline figure -- see VALIDATION_TEST_SEASONS
# Extended from [2023,2024,2025] to 5 seasons after RB's ADP-subset gap was
# found to swing +8.15% / -9.70% / +3.03% year to year on only 3 seasons --
# too wide a swing to resolve whether RB is genuinely near-parity with ADP
# or the 3-season average just landed near zero by cancellation. Checked
# real coverage back through 2018 (build_baseline_projection_v1.py's
# run_position() directly, comparable adp_subset sizes every year back to
# at least 2018) before extending -- capped at 5 seasons/2021 rather than
# going further back because RB usage patterns (workload share, receiving
# role) drift with era; older seasons trade noise reduction for relevance.
VALIDATION_TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
# Standing caveat, not a bug to fix: TE's ADP-covered subset is ~19
# players/season, roughly a third of WR's (~59) and RB's (~47) -- a durable
# fact about how few fantasy-relevant TEs exist per season, not a sample
# gap this pipeline can close. TE's per-season spread has consistently been
# the widest of any position (34.0pp on the ADP comparison, 2.5% scheme
# spread vs RB/WR's 1.7-3.0%) BECAUSE of this, not despite good methodology
# -- treat any TE-specific win-count or MAE claim as carrying more
# irreducible uncertainty than WR/RB's, even at the same 5-season window.
RECENCY_WEIGHTS = [0.5, 0.3, 0.2]  # 1 season back, 2 back, 3 back -- Iteration 3's original reasoned default

# Iteration 4 (build_weighting_scheme_sweep_v1.py) tested equal/moderate_recency/
# strong_recency/smooth_decline per position. First pass used only the 2025
# holdout and found QB/TE margins between schemes of 0.7-1.1% of the best
# score -- too thin to trust from one season. Re-run against 3 independent
# holdouts (2023/2024/2025, train strictly before each) to check stability:
# RB confirmed strong_recency (won 3/3 seasons), WR confirmed strong_recency
# (won 2/3, clear average win) -- both unchanged from the single-season
# picks. QB's original "equal" pick turned out to be a genuine tie with
# moderate_recency (avg MAE identical to 3 decimals, 4.364 vs 4.364) --
# kept equal as the simpler default per Occam's razor, not because the data
# actually preferred it. TE's original single-season pick (moderate_recency)
# did NOT hold up -- it only won the 2025 season it was picked on;
# strong_recency won 2023 and 2024 and ties smooth_decline on the 3-season
# average (1.559 vs 1.559, both ahead of moderate_recency's 1.562) --
# relocked to strong_recency.
#
# Re-checked again (per review, before Iteration 5) against the same 5-season
# window (2021-2025) VALIDATION_TEST_SEASONS now uses, since the 3-season
# window had just been shown to mislabel RB's ADP comparison as a tie when
# a real 4/5-season loss was one extension away -- the same risk applied to
# these locks and hadn't been checked. Selection criterion also upgraded to
# win-count-primary (build_weighting_scheme_sweep_v1.py), not lowest average.
# All four locks HELD, with stronger support than the 3-season pass: QB=equal
# (3/5, was a literal tie at 3 seasons), RB=strong_recency (4/5, was 3/3),
# WR=strong_recency (4/5, was 2/3), TE=strong_recency (4/5, was 2/3). No
# changes to the weights below.
POSITION_RECENCY_WEIGHTS = {
    "QB": [1 / 3, 1 / 3, 1 / 3],
    "RB": [0.7, 0.2, 0.1],
    "WR": [0.7, 0.2, 0.1],
    "TE": [0.7, 0.2, 0.1],
}

POSITION_EFFICIENCY_COLS = {
    "WR": ["target_share", "air_yards_share", "wopr"],
    "TE": ["target_share", "air_yards_share", "wopr"],
    "RB": ["carries", "target_share"],
    "QB": ["attempts", "passing_epa"],
}
TEAM_CONTEXT_COLS = ["prior_team_pass_rate", "prior_team_tds_pg", "prior_team_epa_pg"]


def _load_efficiency_by_season_player() -> pd.DataFrame:
    all_eff_cols = sorted({c for cols in POSITION_EFFICIENCY_COLS.values() for c in cols})
    rows = []
    for path in sorted(STATS_DIR.glob("*.csv")):
        cols = pd.read_csv(path, nrows=0).columns
        keep = ["season", "player_id"] + [c for c in all_eff_cols if c in cols]
        df = pd.read_csv(path, usecols=keep)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    for c in all_eff_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out.groupby(["season", "player_id"], as_index=False)[all_eff_cols].mean()


def _load_team_context() -> pd.DataFrame:
    df = pd.read_csv(OFFENSE_CSV)
    return df[["season", "team"] + TEAM_CONTEXT_COLS].copy()


def _recency_weighted_baseline_single_scheme(e6: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    """Core per-scheme computation, factored out so both a single flat
    weight list (Iteration 3, Iteration 4's sweep) and per-position locked-in
    schemes (Iteration 4's final pick, see _build_recency_weighted_baseline)
    share the exact same join/merge logic.

    reset_index() is required, not cosmetic: `result.merge(...)` inside the
    loop below always returns a fresh 0..N-1 RangeIndex regardless of the
    input's index, but weighted_sum/weight_total/n_seasons_available are
    built once before the loop on e6's original index. When e6 is the full,
    already-contiguous dataset (Iteration 3) those two indices coincide by
    accident and the bug is invisible; when e6 is a position-filtered slice
    (Iteration 4's per-position dict path, whose index is NOT contiguous --
    e.g. [3, 4, 8, 15, ...]), the post-merge positional add silently
    misaligned ~84% of QB rows to NaN before this fix (caught by comparing
    Iteration 4's locked-in run against its own sweep's numbers -- they
    didn't match, which is exactly why that check matters)."""
    e6 = e6.reset_index(drop=True)
    base = e6[["season", "player_id", "position", "final_fantasy_ppg", "games_played"]].copy()
    result = e6[["season", "player_id", "position"]].copy()
    weighted_sum = pd.Series(0.0, index=result.index)
    weight_total = pd.Series(0.0, index=result.index)
    n_seasons_available = pd.Series(0, index=result.index)
    for lag, weight in zip([1, 2, 3], weights):
        lagged = base.copy()
        lagged["season"] = lagged["season"] + lag
        lagged = lagged.rename(columns={"final_fantasy_ppg": f"ppg_lag{lag}", "games_played": f"games_lag{lag}"})
        result = result.merge(
            lagged[["season", "player_id", "position", f"ppg_lag{lag}", f"games_lag{lag}"]],
            on=["season", "player_id", "position"], how="left",
        )
        has_lag = result[f"ppg_lag{lag}"].notna() & (result[f"games_lag{lag}"].fillna(0) > 0)
        weighted_sum = weighted_sum + np.where(has_lag, result[f"ppg_lag{lag}"] * weight, 0.0)
        weight_total = weight_total + np.where(has_lag, weight, 0.0)
        n_seasons_available = n_seasons_available + has_lag.astype(int)
    result["recency_weighted_ppg_baseline"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
    result["historical_average_ppg_baseline"] = result[["ppg_lag1", "ppg_lag2", "ppg_lag3"]].mean(axis=1)
    result["n_prior_seasons"] = n_seasons_available
    return result[["season", "player_id", "position", "recency_weighted_ppg_baseline", "historical_average_ppg_baseline", "n_prior_seasons"]]


def _build_recency_weighted_baseline(
    e6: pd.DataFrame, weights: list[float] | dict[str, list[float]] = RECENCY_WEIGHTS
) -> pd.DataFrame:
    """Self-join E6 to itself at lags 1/2/3 seasons on (player_id, position)
    to build the 3-season recency-weighted talent baseline. Uses E6 (not raw
    stats_player_reg_by_season) so the target of this model and the history
    it's built from share the exact same half-PPR correction from
    Iteration 2 -- no format mismatch between input and target.

    `weights` accepts either one flat [lag1, lag2, lag3] list applied to
    every position (Iteration 3's default; also how Iteration 4's sweep
    tests one scheme at a time), or a dict keyed by position for the final
    per-position locked-in scheme Iteration 4 picked -- QB/RB/WR/TE do not
    all share the same winning scheme, so POSITION_RECENCY_WEIGHTS below
    computes each position's baseline under its own scheme and concatenates,
    rather than forcing one global scheme post-selection."""
    if isinstance(weights, dict):
        parts = []
        for position, pos_weights in weights.items():
            pos_e6 = e6[e6["position"].eq(position)]
            parts.append(_recency_weighted_baseline_single_scheme(pos_e6, pos_weights))
        return pd.concat(parts, ignore_index=True)
    return _recency_weighted_baseline_single_scheme(e6, weights)


def build_features(weights: list[float] | dict[str, list[float]] = RECENCY_WEIGHTS) -> pd.DataFrame:
    e6 = pd.read_csv(E6_PATH)
    baseline = _build_recency_weighted_baseline(e6, weights=weights)
    df = e6.merge(baseline, on=["season", "player_id", "position"], how="left")

    efficiency = _load_efficiency_by_season_player()
    efficiency_prior = efficiency.copy()
    efficiency_prior["season"] = efficiency_prior["season"] + 1
    efficiency_prior = efficiency_prior.rename(columns={c: f"{c}_prior" for c in efficiency_prior.columns if c not in ("season", "player_id")})
    df = df.merge(efficiency_prior, on=["season", "player_id"], how="left")

    team_ctx = _load_team_context()
    df["team"] = df["team"].astype(str).replace({"nan": np.nan})
    team_ctx["team"] = team_ctx["team"].astype(str)
    df = df.merge(team_ctx, on=["season", "team"], how="left")
    return df


def _feature_list(position: str) -> list[str]:
    eff = [f"{c}_prior" for c in POSITION_EFFICIENCY_COLS[position]]
    return ["recency_weighted_ppg_baseline"] + eff + TEAM_CONTEXT_COLS


def run_position(
    df: pd.DataFrame, position: str, test_season: int = TEST_SEASON, extra_features: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """extra_features (Iteration 5): appended on top of the position's base
    feature list without touching it, so Iteration 3/4's behavior is
    unchanged when this is None (the default) -- Iteration 5's trend layer
    calls this with its new snap-trend columns instead of duplicating this
    function's fit/score/benchmark logic."""
    pos_df = df[df["position"].eq(position) & df["n_prior_seasons"].gt(0)].copy()
    features = _feature_list(position) + (extra_features or [])
    for c in features:
        pos_df[c] = pd.to_numeric(pos_df[c], errors="coerce")

    train = pos_df[pos_df["season"] < test_season].copy()
    test = pos_df[pos_df["season"].eq(test_season)].copy()
    test = test[test["final_fantasy_ppg"].notna()]

    if train.empty or test.empty:
        return pd.DataFrame(), {"position": position, "test_season": test_season, "status": "insufficient_data"}

    train_y = pd.to_numeric(train["final_fantasy_ppg"], errors="coerce")
    train_valid = train_y.notna()
    train = train[train_valid]
    train_y = train_y[train_valid]

    medians = train[features].median(numeric_only=True)
    train_x = train[features].fillna(medians)
    test_x = test[features].fillna(medians)

    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    test = test.copy()
    test["model_prediction"] = model.predict(test_x)

    # Benchmark A: plain unweighted historical average (ablation -- same
    # source data, no recency weighting, no efficiency/team features).
    test["historical_average_prediction"] = test["historical_average_ppg_baseline"]

    # Benchmark B: ADP-only regression, fit fresh on train rows that have
    # real ADP (can't evaluate players ADP has no opinion on).
    adp_train = train[train["overall_adp"].notna()]
    adp_test_mask = test["overall_adp"].notna()
    if len(adp_train) >= 10 and adp_test_mask.any():
        adp_model = Ridge(alpha=1.0)
        adp_x_train = np.log(adp_train[["overall_adp"]].to_numpy() + 1.0)
        adp_model.fit(adp_x_train, adp_train["final_fantasy_ppg"])
        adp_x_test = np.log(test.loc[adp_test_mask, ["overall_adp"]].to_numpy() + 1.0)
        test.loc[adp_test_mask, "adp_only_prediction"] = adp_model.predict(adp_x_test)
    else:
        test["adp_only_prediction"] = np.nan

    def mae(col: str, mask: pd.Series | None = None) -> float:
        sub = test if mask is None else test[mask]
        valid = sub[col].notna() & sub["final_fantasy_ppg"].notna()
        if valid.sum() == 0:
            return float("nan")
        return float((sub.loc[valid, col] - sub.loc[valid, "final_fantasy_ppg"]).abs().mean())

    adp_subset_mask = test["adp_only_prediction"].notna()

    report = {
        "position": position,
        "test_season": test_season,
        "status": "fit",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_rows_with_adp": int(adp_test_mask.sum()),
        "model_mae_all_test_rows": round(mae("model_prediction"), 3),
        "historical_average_mae_all_test_rows": round(mae("historical_average_prediction"), 3),
        "model_mae_adp_subset": round(mae("model_prediction", adp_subset_mask), 3),
        "historical_average_mae_adp_subset": round(mae("historical_average_prediction", adp_subset_mask), 3),
        "adp_only_mae_adp_subset": round(mae("adp_only_prediction", adp_subset_mask), 3),
        "beats_historical_average": bool(mae("model_prediction") < mae("historical_average_prediction")),
        "beats_adp_only": bool(mae("model_prediction", adp_subset_mask) < mae("adp_only_prediction", adp_subset_mask)) if adp_subset_mask.any() else None,
        "ridge_coefficients": dict(zip(features, [round(float(c), 4) for c in model.coef_])),
    }
    return test, report


def compare_model_versions(
    old_report: pd.DataFrame, new_report: pd.DataFrame, metric_col: str = "model_mae_all_test_rows"
) -> pd.DataFrame:
    """Locked-in success criterion (per user review, before Iteration 5):
    an "improvement" claim must NOT rest on a shift in the multi-season
    AVERAGE alone -- RB's ADP-subset gap swung +8.15% / -9.70% / +3.03%
    across only 3 seasons, so an average-only comparison can flip sign
    purely from which seasons happen to be in the window, independent of
    whether the new version is actually better. Require the new version to
    win a strict MAJORITY of individual, matched test seasons (both old and
    new report on the same season) before calling it an improvement.

    Every later iteration that claims "beats the prior version" should
    route that claim through this function rather than eyeballing an
    average-MAE delta.
    """
    rows = []
    for position in old_report["position"].unique():
        old_pos = old_report[(old_report["position"] == position) & (old_report["status"] == "fit")]
        new_pos = new_report[(new_report["position"] == position) & (new_report["status"] == "fit")]
        merged = old_pos[["test_season", metric_col]].merge(
            new_pos[["test_season", metric_col]], on="test_season", suffixes=("_old", "_new")
        )
        if merged.empty:
            rows.append({"position": position, "status": "no_matched_seasons"})
            continue
        wins = int((merged[f"{metric_col}_new"] < merged[f"{metric_col}_old"]).sum())
        n = len(merged)
        rows.append({
            "position": position,
            "status": "fit",
            "n_matched_seasons": n,
            "new_version_wins": wins,
            "old_avg": round(merged[f"{metric_col}_old"].mean(), 3),
            "new_avg": round(merged[f"{metric_col}_new"].mean(), 3),
            "improved": wins > n / 2,  # strict majority, e.g. 2 of 3, 3 of 5
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_features(weights=POSITION_RECENCY_WEIGHTS)
    all_predictions = []
    reports = []
    for position in ["QB", "RB", "WR", "TE"]:
        for test_season in VALIDATION_TEST_SEASONS:
            test, report = run_position(df, position, test_season=test_season)
            reports.append(report)
            if not test.empty:
                all_predictions.append(test)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
    report_df = pd.DataFrame(reports)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"Test seasons: {VALIDATION_TEST_SEASONS} (per-season detail below; see AVERAGE for the real baseline-to-beat)\n")
    fit = report_df[report_df["status"] == "fit"]
    for position in ["QB", "RB", "WR", "TE"]:
        pos_rows = fit[fit["position"] == position]
        if pos_rows.empty:
            print(f"{position}: insufficient_data across all test seasons")
            continue
        for _, r in pos_rows.iterrows():
            print(
                f"{r['position']} {r['test_season']}: train={r['train_rows']} test={r['test_rows']} "
                f"(adp_subset={r['test_rows_with_adp']}) | model_MAE={r['model_mae_all_test_rows']} "
                f"hist_avg_MAE={r['historical_average_mae_all_test_rows']} "
                f"| adp_subset: model={r['model_mae_adp_subset']} "
                f"hist_avg={r['historical_average_mae_adp_subset']} "
                f"adp_only={r['adp_only_mae_adp_subset']}"
            )
        avg_model_mae = pos_rows["model_mae_all_test_rows"].mean()
        avg_hist_mae = pos_rows["historical_average_mae_all_test_rows"].mean()
        print(
            f"{position} AVERAGE (full test set) across {len(pos_rows)} seasons -- model_MAE={avg_model_mae:.3f} "
            f"hist_avg_MAE={avg_hist_mae:.3f}  <-- baseline-to-beat for Iteration 5, NOT any single season"
        )

        # ADP-only benchmark. Verdict is WIN-COUNT based (locked-in success
        # criterion, per review before Iteration 5), not an average-gap
        # threshold -- an average can land near zero purely from a big win
        # and a big loss canceling out (RB: +8.15/-9.70/+3.03% across just 3
        # seasons looked like a tie on average but was really a 1-in-3 win
        # rate). Requires a strict majority of individual seasons to call
        # "beats" or "loses"; anything else is reported as volatile/
        # unresolved rather than rounded to a verdict. All three MAE columns
        # here are computed on the identical adp_subset_mask per season (see
        # run_position()), so the comparison stays apples-to-apples.
        per_season_gap = (pos_rows["model_mae_adp_subset"] - pos_rows["adp_only_mae_adp_subset"]) / pos_rows["adp_only_mae_adp_subset"] * 100.0
        n = len(pos_rows)
        wins = int((per_season_gap < 0).sum())
        losses = int((per_season_gap > 0).sum())
        spread = per_season_gap.max() - per_season_gap.min()
        if wins > n / 2:
            verdict = "beats ADP"
        elif losses > n / 2:
            verdict = "loses to ADP"
        else:
            verdict = "no majority -- unresolved"
        avg_adp_n = pos_rows["test_rows_with_adp"].mean()
        avg_model_adp = pos_rows["model_mae_adp_subset"].mean()
        avg_hist_adp = pos_rows["historical_average_mae_adp_subset"].mean()
        avg_adp_only = pos_rows["adp_only_mae_adp_subset"].mean()
        avg_gap_pct = (avg_model_adp - avg_adp_only) / avg_adp_only * 100.0
        print(
            f"{position} AVERAGE (ADP subset, ~{avg_adp_n:.0f} players/season) across {n} seasons -- "
            f"model={avg_model_adp:.3f} hist_avg={avg_hist_adp:.3f} adp_only={avg_adp_only:.3f} "
            f"(avg gap {avg_gap_pct:+.1f}%, won {wins}/{n} individual seasons, "
            f"per-season spread {spread:.1f}pp -> {verdict})\n"
        )

    print(f"Predictions written: {OUTPUT_PATH}")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
