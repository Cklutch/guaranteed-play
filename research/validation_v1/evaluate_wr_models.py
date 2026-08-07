from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from validation_utils import (
    VALIDATION_DIR,
    adp_baseline_scores,
    bucket_rows,
    ensure_columns,
    fit_predict,
    model_specs,
    summarize_model,
    top_pick_rows,
)
from build_predraft_dataset import build_dataset
from workload_feature_groups import workload_feature_groups

POSITION = "WR"
TARGETS = ["WR_Top24", "WR_Top12", "WR_Underpriced_Top24", "WR_Underpriced_Top12", "WR_Beat_ADP_By_12"]

FEATURE_GROUPS = {
    "adp_only_baseline": ["overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round"],
    "projections_only": [
        "preseason_projection", "projected_fantasy_points", "projected_positional_rank",
        "projected_points_over_adp_expectation", "projection_minus_adp_implied_expectation",
    ],
    "adp_projections": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "preseason_projection", "projected_fantasy_points", "projected_positional_rank",
        "projected_points_over_adp_expectation", "projection_minus_adp_implied_expectation",
    ],
    "prior_production_only_baseline": [
        "prior_targets", "prior_receiving_yards", "prior_receiving_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
    ],
    "age_prior_production_baseline": [
        "age", "prior_targets", "prior_receiving_yards", "prior_receiving_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
    ],
    "adp_prior_production_baseline": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round", "prior_targets", "prior_receiving_yards", "prior_receiving_tds",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
    ],
    "adp_role_opportunity": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "prior_targets", "prior_target_share", "prior_wr_targets_expanded", "prior_wr_target_share_expanded",
        "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded",
        "target_competition_score", "same_team_better_adp_count", "same_team_top120_count", "adp_gap_to_next_teammate",
    ],
    "adp_team_context": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "prior_team_pass_attempts", "prior_wr_team_targets_expanded", "team_position_adp_count",
        "target_competition_score", "wr_team_change_from_prior_opportunity", "adp_stdev", "adp_uncertainty_score",
    ],
    "sportsbook_features_only": [
        "sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_receiving_tds",
        "sportsbook_projected_anytime_td_probability", "sportsbook_wr_volume_score", "sportsbook_value_over_adp",
        "sportsbook_rank_minus_positional_adp", "sportsbook_projection_available_flag", "sportsbook_volume_available_flag",
    ],
    "adp_sportsbook_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_receiving_tds",
        "sportsbook_projected_anytime_td_probability", "sportsbook_wr_volume_score", "sportsbook_value_over_adp",
        "sportsbook_rank_minus_positional_adp", "sportsbook_projection_available_flag", "sportsbook_volume_available_flag",
    ],
    "adp_sportsbook_prior_production": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_receiving_tds",
        "sportsbook_projected_anytime_td_probability", "sportsbook_wr_volume_score", "sportsbook_value_over_adp",
        "prior_targets", "prior_receiving_yards", "prior_receiving_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
    ],
    "adp_sportsbook_expanded_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_receiving_tds",
        "sportsbook_projected_anytime_td_probability", "sportsbook_wr_volume_score", "sportsbook_value_over_adp", "sportsbook_rank_minus_positional_adp",
        "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "target_competition_score", "adp_uncertainty_score",
    ],
    "adp_all_predraft_safe_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_receiving_tds",
        "sportsbook_projected_anytime_td_probability", "sportsbook_wr_volume_score", "sportsbook_value_over_adp", "sportsbook_rank_minus_positional_adp",
        "preseason_projection", "projected_fantasy_points", "projected_positional_rank",
        "prior_targets", "prior_receiving_yards", "prior_receiving_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
        "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "target_competition_score",
        "same_team_better_adp_count", "same_team_top120_count", "adp_stdev", "adp_uncertainty_score", "years_in_league_proxy", "rookie_or_first_year_flag", "age_bucket_code",
    ],    "adp_all_expanded": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "preseason_projection", "projected_fantasy_points", "projected_positional_rank",
        "projected_points_over_adp_expectation", "projection_minus_adp_implied_expectation",
        "prior_targets", "prior_receiving_yards", "prior_receiving_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share",
        "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded",
        "target_competition_score", "same_team_better_adp_count", "same_team_top48_count", "same_team_top120_count",
        "teammate_best_adp", "teammate_second_best_adp", "adp_gap_to_next_teammate", "adp_stdev", "adp_uncertainty_score",
        "wr_team_change_from_prior_opportunity", "years_in_league_proxy", "rookie_or_first_year_flag", "age_bucket_code", "prior_games_missed_proxy",
    ],
}


PROJECTION_VOLUME_FEATURES = [
    "projection_available_flag", "projected_fantasy_points", "projected_positional_rank",
    "projection_rank_minus_positional_adp", "projection_value_over_adp", "projection_points_per_adp",
    "projected_receptions", "projected_receiving_yards", "projected_receiving_tds",
    "projected_carries", "projected_rushing_yards", "projected_rushing_tds", "projected_total_tds",
    "projected_volume_score", "projected_touch_score", "projected_receiving_role_score",
]

ADP_FEATURES = ["overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round"]
PRIOR_FEATURES = [
    "prior_targets", "prior_receptions", "prior_receiving_yards", "prior_receiving_tds",
    "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
]
EXPANDED_FEATURES = [
    "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded",
    "target_competition_score", "same_team_better_adp_count", "same_team_top120_count",
    "adp_stdev", "adp_uncertainty_score", "years_in_league_proxy", "rookie_or_first_year_flag", "age_bucket_code",
]

FEATURE_GROUPS.update({
    "projections_only": PROJECTION_VOLUME_FEATURES,
    "adp_projections": ADP_FEATURES + PROJECTION_VOLUME_FEATURES,
    "adp_projection_prior_production": ADP_FEATURES + PROJECTION_VOLUME_FEATURES + PRIOR_FEATURES,
    "adp_projection_expanded_features": ADP_FEATURES + PROJECTION_VOLUME_FEATURES + EXPANDED_FEATURES,
    "adp_all_available_predraft_safe_features": ADP_FEATURES + PROJECTION_VOLUME_FEATURES + PRIOR_FEATURES + EXPANDED_FEATURES,
})
FEATURE_GROUPS.update(workload_feature_groups(POSITION, PRIOR_FEATURES))
REQUIRED_OUTPUT_COLUMNS = [
    "test_season", "target", "model_name", "model_type", "feature_group", "status", "sample_size", "positive_rate",
    "baseline_hit_rate", "model_hit_rate", "lift_over_baseline", "auc", "adp_auc", "top_decile_hit_rate", "adp_available", "beats_adp",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    df, _metadata = build_dataset()
    df.to_csv(path, index=False)
    return df


def evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = dataset[dataset["position"].eq(POSITION)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    specs = model_specs(FEATURE_GROUPS)
    result_rows: list[dict[str, object]] = []
    bucket_result_rows: list[dict[str, object]] = []
    top_pick_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []

    for target in TARGETS:
        if target not in df.columns:
            continue
        target_non_null = pd.to_numeric(df[target], errors="coerce").notna().sum()
        for test_season in seasons:
            train = df[df["season"] < test_season].copy()
            test = df[df["season"] == test_season].copy()
            if train.empty or test.empty or target_non_null == 0:
                continue
            baseline = adp_baseline_scores(test)
            for spec in specs:
                features = [f for f in spec["features"] if f in df.columns]
                if target.startswith(f"{POSITION}_Underpriced") or target.endswith("Beat_ADP_By_12"):
                    if df.get("positional_adp", pd.Series(np.nan, index=df.index)).notna().sum() == 0:
                        status = "skipped_target_requires_historical_adp"
                        scores = pd.Series(np.nan, index=test.index)
                    else:
                        scores, status = fit_predict(train, test, target, features, str(spec["kind"]))
                elif str(spec["feature_group"] if "feature_group" in spec else spec["model_name"]).startswith("adp"):
                    scores, status = fit_predict(train, test, target, features, str(spec["kind"]))
                else:
                    scores, status = fit_predict(train, test, target, features, str(spec["kind"]))
                result_rows.append(summarize_model(
                    test, target, scores, baseline, test_season, str(spec["model_name"]), str(spec["model_type"]), str(spec["model_name"]).rsplit("_", 2)[0], status
                ))
                bucket_result_rows.extend(bucket_rows(test, target, scores, test_season, str(spec["model_name"])))
                top = top_pick_rows(test, target, scores, baseline, str(spec["model_name"]), test_season, top_n=20)
                if not top.empty:
                    top_pick_frames.append(top)
                    if target == f"{POSITION}_Top24":
                        signal = top.copy()
                        signal["ADP"] = signal["overall_adp"].fillna(signal["preseason_adp"])
                        signal["model_score"] = signal["model_score"]
                        signal["target_probability"] = signal["model_score"]
                        if signal["ADP"].notna().any():
                            adp_rank_score = -pd.to_numeric(signal["ADP"], errors="coerce")
                            signal["ADP_baseline_probability"] = adp_rank_score.rank(pct=True)
                            signal["edge_over_ADP"] = signal["target_probability"] - signal["ADP_baseline_probability"]
                        else:
                            signal["ADP_baseline_probability"] = np.nan
                            signal["edge_over_ADP"] = np.nan
                        signal["primary_signal"] = np.where(signal["ADP"].notna(), "model_vs_adp", "not_app_ready_no_historical_adp")
                        signal["risk_notes"] = np.where(signal["ADP"].isna(), "No historical ADP available; signal is not app-ready", "Review lift over ADP and ADP bucket coverage before app use")
                        signal["feature_explanation"] = "Predraft-safe prior production/age/market features only; same-season outcome used as label."
                        signal_frames.append(signal)

    results = ensure_columns(pd.DataFrame(result_rows), REQUIRED_OUTPUT_COLUMNS)
    buckets = pd.DataFrame(bucket_result_rows)
    top_picks = pd.concat(top_pick_frames, ignore_index=True) if top_pick_frames else pd.DataFrame()
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return results, buckets, top_picks, signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WR pre-draft validation models with walk-forward validation.")
    # See the same note in evaluate_rb_models.py -- defaults to the Step 3a
    # workload dataset (base + new opportunity features, all four positions),
    # not the stale WR/RB-only projected chain file.
    parser.add_argument("--dataset", default=str(VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"))
    args = parser.parse_args()
    dataset = load_dataset(Path(args.dataset))
    results, buckets, top_picks, signals = evaluate(dataset)
    results.to_csv(VALIDATION_DIR / "wr_validation_results.csv", index=False)
    buckets.to_csv(VALIDATION_DIR / "wr_adp_bucket_results.csv", index=False)
    top_picks.to_csv(VALIDATION_DIR / "wr_top_model_picks.csv", index=False)
    if not signals.empty:
        signals_out = signals[[
            "season", "player_name", "position", "team", "ADP", "positional_adp", "model_score", "target_probability",
            "ADP_baseline_probability", "edge_over_ADP", "primary_signal", "risk_notes", "feature_explanation",
        ]].copy()
    else:
        signals_out = pd.DataFrame(columns=[
            "season", "player_name", "position", "team", "ADP", "positional_adp", "model_score", "target_probability",
            "ADP_baseline_probability", "edge_over_ADP", "primary_signal", "risk_notes", "feature_explanation",
        ])
    signals_out.to_csv(VALIDATION_DIR / "wr_unified_signal_candidates.csv", index=False)
    print(f"WR rows evaluated: {int((dataset['position'] == POSITION).sum())}")
    print(f"WR result rows: {len(results)}")
    print("Best WR model found: not app-ready; see report for ADP availability and lift status.")


if __name__ == "__main__":
    main()




