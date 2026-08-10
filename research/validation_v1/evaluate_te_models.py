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

POSITION = "TE"
# TE uses Top12/Top6 finish lines (see POSITION_TOP_N_THRESHOLDS in
# build_predraft_dataset.py) rather than WR/RB's Top24/Top12 -- only ~1 TE
# per team is fantasy-relevant, so 24/12 would be meaningless here.
PRIMARY_TARGET = "TE_Top12"
TARGETS = ["TE_Top12", "TE_Top6", "TE_Underpriced_Top12", "TE_Underpriced_Top6", "TE_Beat_ADP_By_12"]

# Scoped to the base predraft_validation_dataset.csv columns only -- see the
# same note in evaluate_qb_models.py. TE does not get the WR-opportunity
# target-share merge (that source is WR-specific), so prior production here
# is limited to receiving yards/TDs/fantasy points from the outcome cache.
FEATURE_GROUPS = {
    "adp_only_baseline": ["overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round"],
    "prior_production_only_baseline": [
        "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
    "age_prior_production_baseline": [
        "age", "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
    "adp_prior_production_baseline": [
        "overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round",
        "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
        "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
    ],
}

TE_PRIOR_PRODUCTION = [
    "prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
    "prior_fantasy_points", "prior_fantasy_ppg", "prior_games",
]
FEATURE_GROUPS.update(workload_feature_groups(POSITION, TE_PRIOR_PRODUCTION))

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
                collect = kind in STACKED_WEIGHT_KINDS and target == PRIMARY_TARGET
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
                    if target == PRIMARY_TARGET:
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
    parser = argparse.ArgumentParser(description="Evaluate TE pre-draft validation models with walk-forward validation.")
    parser.add_argument("--dataset", default=str(VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"))
    args = parser.parse_args()
    dataset = load_dataset(Path(args.dataset))
    results, buckets, top_picks, signals, weight_rows = evaluate(dataset)
    append_stacked_weight_rows(weight_rows, POSITION)
    results.to_csv(VALIDATION_DIR / "te_validation_results.csv", index=False)
    buckets.to_csv(VALIDATION_DIR / "te_adp_bucket_results.csv", index=False)
    top_picks.to_csv(VALIDATION_DIR / "te_top_model_picks.csv", index=False)
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
    signals_out.to_csv(VALIDATION_DIR / "te_unified_signal_candidates.csv", index=False)
    print(f"TE rows evaluated: {int((dataset['position'] == POSITION).sum())}")
    print(f"TE result rows: {len(results)}")
    print("Best TE model found: not app-ready; see report for ADP availability and lift status.")


if __name__ == "__main__":
    main()
