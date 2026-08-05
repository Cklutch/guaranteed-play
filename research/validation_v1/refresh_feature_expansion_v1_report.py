from __future__ import annotations

from pathlib import Path
import pandas as pd

D = Path("research/validation_v1")
inv = pd.read_csv(D / "feature_expansion_inventory.csv")
cov = pd.read_csv(D / "feature_coverage_report.csv")
wr = pd.read_csv(D / "wr_validation_results.csv")
rb = pd.read_csv(D / "rb_validation_results.csv")
wrb = pd.read_csv(D / "wr_bucket_lift_analysis.csv")
rbb = pd.read_csv(D / "rb_bucket_lift_analysis.csv")


def infer_group(model_name: object) -> str:
    name = str(model_name)
    for suffix in ["_regularized_logistic", "_random_forest", "_logistic"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    fit = df[df["status"].eq("fit")].copy()
    fit["expanded_feature_group"] = fit["model_name"].apply(infer_group)
    grouped = fit.groupby(["target", "expanded_feature_group"], dropna=False).agg(
        seasons=("test_season", "nunique"),
        model_hit_rate=("model_hit_rate", "mean"),
        adp_hit_rate=("baseline_hit_rate", "mean"),
        lift_over_adp=("lift_over_baseline", "mean"),
        auc=("auc", "mean"),
        adp_auc=("adp_auc", "mean"),
        seasons_model_beats_adp=("beats_adp", "sum"),
        adp_seasons=("adp_available", "sum"),
    ).reset_index()
    grouped["pct_seasons_model_beats_adp"] = grouped["seasons_model_beats_adp"] / grouped["seasons"].clip(lower=1) * 100.0
    return grouped.sort_values(["target", "lift_over_adp"], ascending=[True, False])


def best_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(
        ["bucket_type", "bucket", "lift_over_adp", "pct_seasons_model_beats_adp", "sample_size"],
        ascending=[True, True, False, False, False],
    ).drop_duplicates(["position", "bucket_type", "bucket"]).copy()
    return out


wr_summary = summarize_results(wr)
rb_summary = summarize_results(rb)
wr_best = wr_summary.sort_values(["target", "lift_over_adp"], ascending=[True, False]).groupby("target", as_index=False).head(3)
rb_best = rb_summary.sort_values(["target", "lift_over_adp"], ascending=[True, False]).groupby("target", as_index=False).head(3)
wr_bucket_best = best_by_bucket(wrb)
rb_bucket_best = best_by_bucket(rbb)
usable = inv[inv["status"].isin(["usable", "questionable", "poor coverage"])].copy()
unavailable = inv[inv["status"].eq("unavailable")].copy()
coverage_pivot = cov[cov["feature_name"].isin(usable["feature_name"])].pivot_table(index="feature_name", columns="position", values="coverage_pct", aggfunc="first").reset_index().fillna(0)

wr_useful = wr_bucket_best[wr_bucket_best["classification"].ne("Not Useful")]
rb_useful = rb_bucket_best[rb_bucket_best["classification"].ne("Not Useful")]

lines = [
    "# Feature Expansion V1 Report",
    "",
    "Date: 2026-07-07",
    "",
    "Scope: research-only feature expansion in `research/validation_v1`. No Streamlit app changes, no UI, and no new model families.",
    "",
    "## Executive Summary",
    "",
    "Preseason Feature Expansion V1 added the historical features that were actually available locally and marked the requested-but-missing features as unavailable. Historical projection data was not found, so projection feature columns remain empty and current 2026 app projections were excluded as unsafe for historical validation.",
    "",
    "The expanded features did not create a repeatable edge over ADP. WR and RB full-pool validation still have negative average lift over ADP, and the expanded draft-window analysis still has zero buckets classified as Tie-Breaker Only, Strong Draft Signal, or App-Ready.",
    "",
    "## Features Added Or Populated",
    "",
    usable[["feature_name", "source", "safety_classification", "status", "coverage_rows", "coverage_pct"]].to_markdown(index=False),
    "",
    "## Coverage By Position",
    "",
    coverage_pivot.to_markdown(index=False, floatfmt=".2f"),
    "",
    "## Requested Features Unavailable Or Excluded",
    "",
    unavailable[["feature_name", "source", "safety_classification", "status"]].to_markdown(index=False),
    "",
    "## Validation Groups Tested",
    "",
    "The existing logistic, regularized logistic, and random forest model families were reused. No new model family was added.",
    "",
    "Feature groups compared:",
    "",
    "- ADP only",
    "- projections only",
    "- ADP + projections",
    "- ADP + prior production",
    "- ADP + role/opportunity features",
    "- ADP + team context",
    "- ADP + all expanded features",
    "",
    "Projection-only rows mostly skipped because historical projection inputs are unavailable. ADP + projections therefore behaves like ADP-only in this run.",
    "",
    "## WR Full-Pool Results",
    "",
    wr_best[["target", "expanded_feature_group", "seasons", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "auc", "adp_auc", "seasons_model_beats_adp", "pct_seasons_model_beats_adp"]].to_markdown(index=False, floatfmt=".3f"),
    "",
    "WR answer: expanded features did not improve WR enough to beat ADP. The best WR full-pool rows are still negative on average lift over ADP.",
    "",
    "## RB Full-Pool Results",
    "",
    rb_best[["target", "expanded_feature_group", "seasons", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "auc", "adp_auc", "seasons_model_beats_adp", "pct_seasons_model_beats_adp"]].to_markdown(index=False, floatfmt=".3f"),
    "",
    "RB answer: expanded team/context features helped RB Top12 get closer to ADP, but the result is still negative lift and not repeatable enough for a usable signal.",
    "",
    "## Draft Window Results",
    "",
    "Best WR bucket rows after expansion:",
    "",
    wr_bucket_best[["bucket_type", "bucket", "best_target_in_bucket", "model_name", "seasons_tested", "sample_size", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "pct_seasons_model_beats_adp", "classification"]].to_markdown(index=False, floatfmt=".3f"),
    "",
    "Best RB bucket rows after expansion:",
    "",
    rb_bucket_best[["bucket_type", "bucket", "best_target_in_bucket", "model_name", "seasons_tested", "sample_size", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "pct_seasons_model_beats_adp", "classification"]].to_markdown(index=False, floatfmt=".3f"),
    "",
    f"WR buckets at least Tie-Breaker Only: {len(wr_useful)}",
    "",
    f"RB buckets at least Tie-Breaker Only: {len(rb_useful)}",
    "",
    "No expanded draft range became repeatably positive by the classification rules.",
    "",
    "## Required Answers",
    "",
    "Which new features were successfully added?",
    "",
    "- FFCalc ADP raw team metadata, ADP spread/uncertainty, same-team positional competition, WR prior targets/target share/team targets/games/PPG, WR team-change proxy, age bucket, years-in-data proxy, first-year proxy, and prior games-missed proxy for WR rows with prior opportunity data.",
    "",
    "Which requested features were unavailable?",
    "",
    "- Historical preseason projections, projected volume stats, routes, air yards, snap share, red-zone and goal-line usage, true vacated touches/targets, team implied points, projected team points, pace, QB/coach changes, draft capital, preseason injury flags, and suspension flags were not found locally as historical pre-draft-safe sources.",
    "",
    "Which features had usable coverage?",
    "",
    "- ADP raw metadata covers about the same 2010-2024 ADP-matched rows as the ADP baseline. WR prior opportunity covers 1,447 rows. Competition scores cover 1,021 WR rows and 860 RB rows. Projection features have 0 coverage.",
    "",
    "Did expanded features improve WR lift over ADP?",
    "",
    "- No. WR remains negative. Some bucket averages improved, but repeatability stayed weak.",
    "",
    "Did expanded features improve RB lift over ADP?",
    "",
    "- Slightly for RB Top12 team-context comparisons, but not enough to beat ADP. RB remains negative overall.",
    "",
    "Did any draft range become repeatably positive?",
    "",
    "- No. Every WR and RB bucket is still classified Not Useful.",
    "",
    "Which feature group helped most?",
    "",
    "- WR: ADP + prior production remains the best broad group, not the new expansion features. RB: ADP + team context was closest for RB Top12, but still negative. No group created usable edge.",
    "",
    "Should we continue WR, RB, or both?",
    "",
    "- Continue both for research only. WR is still closer to market in some full-pool comparisons; RB has larger positive bucket averages but weaker repeatability. Neither should be integrated.",
    "",
    "What is the next research build?",
    "",
    "- Historical Projection Import V1. The biggest missing ingredient is real preseason projections and projected volume from a dated historical source. Without those, the model mostly has ADP plus prior production, which the market already prices well.",
    "",
    "## Output Files",
    "",
    "- `research/validation_v1/build_feature_expansion_v1.py`",
    "- `research/validation_v1/feature_expansion_inventory.csv`",
    "- `research/validation_v1/predraft_validation_dataset_expanded.csv`",
    "- `research/validation_v1/feature_coverage_report.csv`",
    "- `research/validation_v1/feature_expansion_v1_report.md`",
    "- `research/validation_v1/wr_validation_results.csv`",
    "- `research/validation_v1/rb_validation_results.csv`",
    "- `research/validation_v1/wr_bucket_lift_analysis.csv`",
    "- `research/validation_v1/rb_bucket_lift_analysis.csv`",
    "- `research/validation_v1/draft_window_edge_report.md`",
]
(D / "feature_expansion_v1_report.md").write_text("\n".join(lines), encoding="utf-8")
print("feature expansion report updated")
