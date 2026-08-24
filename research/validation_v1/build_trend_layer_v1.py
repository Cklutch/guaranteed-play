"""projection_model_iteration_plan.pdf, Iteration 5: recent-trend layer (v1.1).

Scope note, checked directly before building: the plan calls for "game-level
usage trends (last 3, last 5 games)." Tonight's data inventory sweep (A2)
already found no weekly points/yards/TDs file exists anywhere in this repo,
and rechecking now confirms the ONLY real game-level signal available is
snap counts (research/validation_v1/data/raw_snap_counts_2012_2025.csv,
2013-2025, player-week-game). No weekly targets/carries/receptions exist to
trend on. This iteration is therefore scoped to a SNAP-SHARE trend layer
specifically, not a general usage-trend layer -- disclosed here rather than
silently building something narrower than "usage trends" might imply.

Three new features per player-season, computed from the season BEFORE the
target season's REG-season games only (no leakage), ordered by week:
  - last_5_games_snap_pct: mean offense_pct over the final 5 games
  - last_3_games_snap_pct: mean offense_pct over the final 3 games
  - snap_pct_trend_delta: mean(final 3 games) - mean(first 3 games) --
    catches a real in-season role change the season average can't see
    (e.g. a Week 15 promotion to starter buried inside a season-long
    average snap share)

ID crosswalk: raw_snap_counts uses a "player" full-name column + PFR ID, not
the gsis-style player_id E6/stats_player_reg_by_season use, and no existing
crosswalk between them exists in this repo (checked: build_risk_variables.py's
_build_id_name_crosswalk() only maps gsis player_id -> name, not PFR ID ->
gsis ID). Built a real, verified one here instead: (clean_name(full name),
position) -> player_id via stats_player_reg_by_season's player_display_name,
per season. Verified directly: 91-93% match rate on 3 sampled seasons
(2019/2022/2024), zero (name, position) collisions to different player_ids
in any of them -- safe join key, real coverage gap disclosed as such (not
hidden as 100%).

Validation: reuses build_baseline_projection_v1.py's exact fit/score/
benchmark logic (run_position()'s new extra_features parameter) across the
same 5-season window (2021-2025) and routes the "did this help" verdict
through compare_model_versions() -- the locked-in win-count-majority
criterion, not an average-MAE shift, comparing directly against Iteration
4's own frozen report (weighting_scheme_sweep_v1_report.csv's winning
schemes, reproduced via baseline_projection_v1_report.csv).

Usage:
    python build_trend_layer_v1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from build_baseline_projection_v1 import (
    POSITION_RECENCY_WEIGHTS,
    VALIDATION_TEST_SEASONS,
    build_features,
    compare_model_versions,
    run_position,
)
from validation_utils import VALIDATION_DIR, clean_name

SNAP_COUNTS_PATH = VALIDATION_DIR / "data" / "raw_snap_counts_2012_2025.csv"
STATS_DIR = VALIDATION_DIR / "data" / "stats_player_reg_by_season"
BASELINE_REPORT_PATH = VALIDATION_DIR / "baseline_projection_v1_report.csv"
REPORT_PATH = VALIDATION_DIR / "trend_layer_v1_report.csv"

POSITIONS = ["QB", "RB", "WR", "TE"]
TREND_COLS_RAW = ["last_5_games_snap_pct", "last_3_games_snap_pct", "snap_pct_trend_delta"]
TREND_FEATURES = [f"{c}_prior" for c in TREND_COLS_RAW]  # "_prior" suffix matches efficiency-features convention (build_baseline_projection_v1.py)
TREND_FEATURES_WITH_FLAG = TREND_FEATURES + ["has_snap_trend"]

# Locked in after running the win-count-majority comparison below: WR won
# 5/5 individual seasons (real, small, consistent improvement every year --
# never once reversed, the opposite pattern from this session's noisy
# ties). QB/RB/TE did not benefit (1/5, 1/5, 0/5) with PLAIN median-imputed
# trend features.
#
# Re-checked after finding the missing-trend rows are NOT random (average
# ~2x lower ppg than matched rows at every position, skewed toward players
# with <3 prior seasons -- median-imputing was telling the model a fringe
# player's role looked "average"). Added has_snap_trend as an explicit
# feature so Ridge can discount the imputed value: WR's win got stronger
# (5/5, avg 2.217->2.204, better than the unflagged 2.211) and TE flipped
# from a clean loss (0/5) to a real if thinner win (3/5: lost 2021 clearly,
# ~tied 2022, won 2023-2025). QB improved but still lost the majority (1/5
# -> 2/5); RB was unaffected (1/5 -> 1/5). Both stay on v1.
#
# TE's win is real but weaker than WR's -- keep the standing TE sample-size
# caveat in mind here specifically; this is the kind of TE claim that
# caveat was written for.
#
# RB is notably NOT in this list despite getting the same has_snap_trend
# fix as QB/TE: RB stayed at exactly 1/5 wins before and after the flag --
# zero movement, unlike QB (1/5->2/5) and TE (0/5->3/5), which both moved
# in the fix's direction. That RB alone didn't respond to a correction that
# helped the other two positions suggests snap share itself may carry
# little real predictive signal for RB specifically, not just an
# imputation artifact -- a defensible football story, since RB fantasy
# value tracks red-zone usage/game script more than raw snap counts, unlike
# WR/TE where snaps proxy for target opportunity fairly directly. Treated
# as a real finding, not a bug to chase further right now. Worth
# revisiting if a real weekly RB-specific stat (carries, red-zone touches)
# ever becomes available in this repo to test instead of snap share.
#
# Single source of truth for "the current best model per position" that
# Iteration 6+ should build from.
POSITIONS_WITH_TREND_LAYER = ["WR", "TE"]


def current_best_features(position: str) -> list[str] | None:
    """extra_features to pass to run_position() for "the current best
    model" at this position -- None for QB/RB (trend layer didn't help,
    stay on Iteration 3/4's feature set), TREND_FEATURES_WITH_FLAG (trend
    + missing-data indicator, not the plain unflagged version) for WR/TE."""
    return TREND_FEATURES_WITH_FLAG if position in POSITIONS_WITH_TREND_LAYER else None


def build_current_best_features() -> pd.DataFrame:
    """Feature frame for whichever positions need trend columns -- safe to
    call unconditionally since QB/RB/TE simply carry unused trend columns
    that current_best_features() never passes to run_position() for them."""
    return build_features_with_trend()


def _build_name_position_crosswalk(season: int) -> dict[tuple[str, str], str]:
    """(clean_name, position) -> player_id, for one season's stats file."""
    stats = pd.read_csv(
        STATS_DIR / f"{season}.csv",
        usecols=["player_id", "player_display_name", "position"],
    )
    stats = stats[stats["position"].isin(POSITIONS)].dropna(subset=["player_display_name", "player_id"])
    stats["key"] = stats["player_display_name"].apply(clean_name)
    # Verified zero collisions on (key, position) in the seasons checked
    # (see module docstring) -- drop_duplicates as a defensive backstop
    # only, not expected to fire.
    stats = stats.drop_duplicates(["key", "position"], keep="first")
    return {(row.key, row.position): row.player_id for row in stats.itertuples()}


def _compute_snap_trends_for_season(season_df: pd.DataFrame, crosswalk: dict[tuple[str, str], str]) -> pd.DataFrame:
    season_df = season_df[season_df["game_type"].eq("REG")].copy()
    season_df["key"] = season_df["player"].apply(clean_name)
    season_df["player_id"] = list(zip(season_df["key"], season_df["position"]))
    season_df["player_id"] = season_df["player_id"].map(crosswalk)
    season_df = season_df.dropna(subset=["player_id"])
    season_df["offense_pct"] = pd.to_numeric(season_df["offense_pct"], errors="coerce")
    season_df = season_df.sort_values(["player_id", "week"])

    rows = []
    for player_id, group in season_df.groupby("player_id"):
        pct = group["offense_pct"].dropna()
        if len(pct) < 3:
            continue
        last5 = pct.tail(5).mean()
        last3 = pct.tail(3).mean()
        first3 = pct.head(3).mean()
        rows.append({
            "player_id": player_id,
            "last_5_games_snap_pct": last5,
            "last_3_games_snap_pct": last3,
            "snap_pct_trend_delta": last3 - first3,
        })
    return pd.DataFrame(rows)


def _load_snap_trend_by_season_player() -> pd.DataFrame:
    snaps = pd.read_csv(
        SNAP_COUNTS_PATH,
        usecols=["season", "week", "game_type", "player", "position", "offense_pct"],
    )
    snaps = snaps[snaps["position"].isin(POSITIONS)]
    parts = []
    for season, season_df in snaps.groupby("season"):
        crosswalk = _build_name_position_crosswalk(season)
        trend = _compute_snap_trends_for_season(season_df, crosswalk)
        if trend.empty:
            continue
        trend["season"] = season
        parts.append(trend)
    return pd.concat(parts, ignore_index=True)


def build_features_with_trend(weights=POSITION_RECENCY_WEIGHTS) -> pd.DataFrame:
    df = build_features(weights=weights)
    trend = _load_snap_trend_by_season_player()
    trend_prior = trend.copy()
    trend_prior["season"] = trend_prior["season"] + 1
    trend_prior = trend_prior.rename(columns={c: f"{c}_prior" for c in TREND_COLS_RAW})
    df = df.merge(trend_prior[["season", "player_id"] + TREND_FEATURES], on=["season", "player_id"], how="left")
    # Missing-data flag (per review): rows missing real trend data are NOT
    # random -- they skew heavily toward lower-production, fewer-prior-
    # season players (checked directly: has_trend rows average ~2x the ppg
    # of missing rows at every position). Blind median-imputing tells the
    # model these players' role looked "average," which is specifically
    # wrong for the subset most likely to be inconsistent bench/fringe
    # players. This flag lets Ridge learn to discount the imputed value
    # instead of trusting it as real.
    df["has_snap_trend"] = df["last_5_games_snap_pct_prior"].notna().astype(float)
    return df


def main() -> None:
    print("Iteration 5 -- snap-share trend layer, coverage check\n")
    trend_raw = _load_snap_trend_by_season_player()
    print(f"Real snap-trend rows built: {len(trend_raw)} (across {trend_raw['season'].nunique()} seasons)\n")

    df = build_features_with_trend()
    old_report = pd.read_csv(BASELINE_REPORT_PATH)

    all_rows = []
    for position in POSITIONS:
        # Denominator is the MODEL'S population (n_prior_seasons>0), not
        # every E6 row -- an earlier version of this coverage check included
        # true rookies with zero NFL history, who run_position() excludes
        # from the model entirely, and reported coverage as 66.8-71.7% as a
        # result (wrong). Corrected denominator: 79.4/85.6/90.0/89.3% for
        # QB/RB/WR/TE -- still real missingness, but roughly half what was
        # originally reported.
        coverage = df[df["position"].eq(position) & df["season"].isin(VALIDATION_TEST_SEASONS) & df["n_prior_seasons"].gt(0)]
        n_total = len(coverage)
        n_with_trend = coverage["last_5_games_snap_pct_prior"].notna().sum()
        print(f"{position}: trend feature coverage (model population only) = {n_with_trend}/{n_total} ({n_with_trend/max(n_total,1)*100:.1f}%)")
        for test_season in VALIDATION_TEST_SEASONS:
            _, report = run_position(df, position, test_season=test_season, extra_features=TREND_FEATURES)
            all_rows.append(report)
    print()

    new_report = pd.DataFrame(all_rows)
    new_report.to_csv(REPORT_PATH, index=False)

    comparison = compare_model_versions(old_report, new_report, metric_col="model_mae_all_test_rows")
    print("Iteration 5 vs Iteration 4 (win-count-majority verdict, full test set MAE):\n")
    print(comparison.to_string(index=False))
    print()
    for _, row in comparison.iterrows():
        if row["status"] != "fit":
            continue
        verdict = "worth keeping" if row["improved"] else "not worth the complexity -- staying on v1"
        print(
            f"{row['position']}: old_avg={row['old_avg']} new_avg={row['new_avg']} "
            f"(won {row['new_version_wins']}/{row['n_matched_seasons']} seasons) -> {verdict}"
        )

    print(f"\nReport written: {REPORT_PATH}")

    # Flagged variant (per review): missing-trend rows checked directly and
    # found NOT random -- has_snap_trend rows average ~2x the ppg of
    # missing rows at every position, skewed toward players with a full
    # 3-season track record. Blind median imputation was telling the model
    # a fringe/bench player's role looked "average." Test whether adding an
    # explicit has_snap_trend flag (so Ridge can discount the imputed
    # value) changes the QB/RB/TE verdict.
    flagged_rows = []
    for position in POSITIONS:
        for test_season in VALIDATION_TEST_SEASONS:
            _, report = run_position(df, position, test_season=test_season, extra_features=TREND_FEATURES_WITH_FLAG)
            flagged_rows.append(report)
    flagged_report = pd.DataFrame(flagged_rows)
    flagged_report.to_csv(VALIDATION_DIR / "trend_layer_v1_flagged_report.csv", index=False)

    flagged_comparison = compare_model_versions(old_report, flagged_report, metric_col="model_mae_all_test_rows")
    print("\nFlagged variant (trend + has_snap_trend indicator) vs Iteration 4:\n")
    print(flagged_comparison.to_string(index=False))
    print()
    for _, row in flagged_comparison.iterrows():
        if row["status"] != "fit":
            continue
        verdict = "worth keeping" if row["improved"] else "still not worth it -- staying on v1"
        print(
            f"{row['position']}: old_avg={row['old_avg']} new_avg={row['new_avg']} "
            f"(won {row['new_version_wins']}/{row['n_matched_seasons']} seasons) -> {verdict}"
        )

    # Final locked-in state: v1.1 (with trend) for WR, v1 (Iteration 3/4,
    # no trend) for QB/RB/TE -- reusing current_best_features() so this
    # block can never silently drift from the POSITIONS_WITH_TREND_LAYER
    # lock above.
    print("\nFinal locked-in model per position (for Iteration 6+):")
    final_rows = []
    for position in POSITIONS:
        version = "v1.1 (+trend+flag)" if position in POSITIONS_WITH_TREND_LAYER else "v1 (no trend)"
        for test_season in VALIDATION_TEST_SEASONS:
            _, report = run_position(df, position, test_season=test_season, extra_features=current_best_features(position))
            final_rows.append(report)
        pos_rows = [r for r in final_rows if r["position"] == position and r["status"] == "fit"]
        avg_mae = sum(r["model_mae_all_test_rows"] for r in pos_rows) / len(pos_rows)
        print(f"  {position}: {version}, avg_MAE={avg_mae:.3f} across {len(pos_rows)} seasons")
    pd.DataFrame(final_rows).to_csv(VALIDATION_DIR / "current_best_model_report.csv", index=False)
    print(f"  Written: {VALIDATION_DIR / 'current_best_model_report.csv'}")


if __name__ == "__main__":
    main()
