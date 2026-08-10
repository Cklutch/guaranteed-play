from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from validation_utils import (
    STACKED_WEIGHT_KINDS,
    VALIDATION_DIR,
    adp_baseline_scores,
    append_stacked_weight_rows,
    bucket_rows,
    ensure_columns,
    fit_predict,
    model_specs,
    stacked_model_specs,
    summarize_model,
    top_pick_rows,
)
from build_predraft_dataset import build_dataset
from workload_feature_groups import workload_feature_groups

POSITION = "RB"
TARGETS = ["RB_Top24", "RB_Top12", "RB_Underpriced_Top24", "RB_Underpriced_Top12", "RB_Beat_ADP_By_12"]

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
        "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
    "age_prior_production_baseline": [
        "age", "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
    "adp_prior_production_baseline": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round", "prior_carries", "prior_rushing_yards", "prior_rushing_tds",
        "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
    "adp_role_opportunity": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
        "backfield_competition_score", "same_team_better_adp_count", "same_team_top120_count", "adp_gap_to_next_teammate",
    ],
    "adp_team_context": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "team_position_adp_count", "backfield_competition_score", "same_team_better_adp_count",
        "same_team_top48_count", "same_team_top120_count", "adp_stdev", "adp_uncertainty_score",
    ],
    "sportsbook_features_only": [
        "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability",
        "sportsbook_rb_touch_score", "sportsbook_rb_receiving_role_score", "sportsbook_value_over_adp", "sportsbook_rank_minus_positional_adp",
        "sportsbook_projection_available_flag", "sportsbook_volume_available_flag",
    ],
    "adp_sportsbook_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability",
        "sportsbook_rb_touch_score", "sportsbook_rb_receiving_role_score", "sportsbook_value_over_adp", "sportsbook_rank_minus_positional_adp",
    ],
    "adp_sportsbook_prior_production": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability",
        "sportsbook_rb_touch_score", "sportsbook_rb_receiving_role_score", "sportsbook_value_over_adp",
        "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_games",
    ],
    "adp_sportsbook_expanded_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability",
        "sportsbook_rb_touch_score", "sportsbook_rb_receiving_role_score", "backfield_competition_score", "adp_uncertainty_score",
    ],
    "adp_all_predraft_safe_features": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability",
        "sportsbook_rb_touch_score", "sportsbook_rb_receiving_role_score", "sportsbook_value_over_adp", "sportsbook_rank_minus_positional_adp",
        "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_games",
        "backfield_competition_score", "same_team_better_adp_count", "same_team_top120_count", "adp_stdev", "adp_uncertainty_score", "years_in_league_proxy", "rookie_or_first_year_flag", "age_bucket_code",
    ],    "adp_all_expanded": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "preseason_projection", "projected_fantasy_points", "projected_positional_rank",
        "projected_points_over_adp_expectation", "projection_minus_adp_implied_expectation",
        "prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "backfield_competition_score",
        "same_team_better_adp_count", "same_team_top48_count", "same_team_top120_count",
        "teammate_best_adp", "teammate_second_best_adp", "adp_gap_to_next_teammate", "adp_stdev", "adp_uncertainty_score",
        "adp_team_change_from_prior_adp", "years_in_league_proxy", "rookie_or_first_year_flag", "age_bucket_code", "prior_games_missed_proxy",
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


def evaluate(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    df = dataset[dataset["position"].eq(POSITION)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    specs = model_specs(FEATURE_GROUPS)
    specs += stacked_model_specs("adp_all_v1", FEATURE_GROUPS.get("adp_all_v1", []))
    result_rows: list[dict[str, object]] = []
    bucket_result_rows: list[dict[str, object]] = []
    top_pick_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []

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
                kind = str(spec["kind"])
                # Feature weights are only collected on the primary target,
                # for the two stacked-only kinds -- see stacked_model_specs()
                # in validation_utils.py for why these kinds/scope exist.
                collect = kind in STACKED_WEIGHT_KINDS and target == f"{POSITION}_Top24"
                weights: dict[str, float] | None = None
                if target.startswith(f"{POSITION}_Underpriced") or target.endswith("Beat_ADP_By_12"):
                    if df.get("positional_adp", pd.Series(np.nan, index=df.index)).notna().sum() == 0:
                        status = "skipped_target_requires_historical_adp"
                        scores = pd.Series(np.nan, index=test.index)
                    elif collect:
                        scores, status, weights = fit_predict(train, test, target, features, kind, collect_weights=True)
                    else:
                        scores, status = fit_predict(train, test, target, features, kind)
                elif collect:
                    scores, status, weights = fit_predict(train, test, target, features, kind, collect_weights=True)
                else:
                    scores, status = fit_predict(train, test, target, features, kind)
                if weights:
                    for feat, w in weights.items():
                        weight_rows.append({"position": POSITION, "test_season": test_season, "kind": kind, "feature": feat, "weight": w})
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
                        signal["primary_signal"] = np.where(signal["ADP"].notna(), "model_vs_adp", "prior_production_no_adp")
                        signal["risk_notes"] = np.where(signal["ADP"].isna(), "No historical ADP available for baseline comparison", "Check ADP bucket lift before app use")
                        signal["feature_explanation"] = "Predraft-safe prior production/age/market features only; same-season outcome used as label."
                        signal_frames.append(signal)

    results = ensure_columns(pd.DataFrame(result_rows), REQUIRED_OUTPUT_COLUMNS)
    buckets = pd.DataFrame(bucket_result_rows)
    top_picks = pd.concat(top_pick_frames, ignore_index=True) if top_pick_frames else pd.DataFrame()
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return results, buckets, top_picks, signals, weight_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RB pre-draft validation models with walk-forward validation.")
    # Defaults to the Step 3a workload dataset (base + snap share / draft
    # capital / red zone / air yards / vacated volume, all four positions).
    # NOTE: this is base+workload, NOT the older projected/sportsbook chain
    # file -- those chain outputs are stale (built before QB/TE and before
    # the prior_target_share join fix) and WR/RB-only. Feature groups whose
    # columns aren't present are skipped as "skipped_no_features", so the
    # comparison here is deliberately ADP-baseline vs. new workload data.
    parser.add_argument("--dataset", default=str(VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"))
    args = parser.parse_args()
    dataset = load_dataset(Path(args.dataset))
    results, buckets, top_picks, signals, weight_rows = evaluate(dataset)
    append_stacked_weight_rows(weight_rows, POSITION)
    results.to_csv(VALIDATION_DIR / "rb_validation_results.csv", index=False)
    buckets.to_csv(VALIDATION_DIR / "rb_adp_bucket_results.csv", index=False)
    top_picks.to_csv(VALIDATION_DIR / "rb_top_model_picks.csv", index=False)
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
    signals_out.to_csv(VALIDATION_DIR / "rb_unified_signal_candidates.csv", index=False)
    print(f"RB rows evaluated: {int((dataset['position'] == POSITION).sum())}")
    print(f"RB result rows: {len(results)}")
    print("Best RB model found: not app-ready; see report for ADP availability and lift status.")


if __name__ == "__main__":
    main()









