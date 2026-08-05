from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, fit_predict


DATASET = VALIDATION_DIR / "predraft_validation_dataset_projected.csv"
WR_OUTPUT = VALIDATION_DIR / "wr_projection_bucket_window_lightweight.csv"
RB_OUTPUT = VALIDATION_DIR / "rb_projection_bucket_window_lightweight.csv"
REPORT = VALIDATION_DIR / "projection_bucket_window_lightweight_report.md"

OVERALL_BUCKETS = [
    ("1-24", 1, 24),
    ("25-48", 25, 48),
    ("49-72", 49, 72),
    ("73-96", 73, 96),
    ("97-120", 97, 120),
    ("121+", 121, math.inf),
]

POSITIONAL_BUCKETS = [
    ("1-12", 1, 12),
    ("13-24", 13, 24),
    ("25-36", 25, 36),
    ("37-48", 37, 48),
    ("49+", 49, math.inf),
]

ADP_FEATURES = ["overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round"]
PROJECTION_VOLUME_FEATURES = [
    "projection_available_flag",
    "projected_fantasy_points",
    "projected_positional_rank",
    "projection_rank_minus_positional_adp",
    "projection_value_over_adp",
    "projection_points_per_adp",
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_carries",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_total_tds",
    "projected_volume_score",
    "projected_touch_score",
    "projected_receiving_role_score",
]
EXPANDED_FEATURES = [
    "prior_wr_targets_expanded",
    "prior_wr_target_share_expanded",
    "prior_wr_team_targets_expanded",
    "target_competition_score",
    "same_team_better_adp_count",
    "same_team_top120_count",
    "adp_stdev",
    "adp_uncertainty_score",
    "years_in_league_proxy",
    "rookie_or_first_year_flag",
    "age_bucket_code",
]
FEATURES = ADP_FEATURES + PROJECTION_VOLUME_FEATURES + EXPANDED_FEATURES

CONFIG = {
    "WR": {
        "target": "WR_Beat_ADP_By_12",
        "model_name": "adp_projection_expanded_features_logistic",
        "feature_group": "adp_projection_expanded",
        "output": WR_OUTPUT,
    },
    "RB": {
        "target": "RB_Beat_ADP_By_12",
        "model_name": "adp_projection_expanded_features_logistic",
        "feature_group": "adp_projection_expanded",
        "output": RB_OUTPUT,
    },
}


def bucket_label(value: object, buckets: list[tuple[str, float, float]]) -> str | None:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    for label, low, high in buckets:
        if float(val) >= float(low) and float(val) <= float(high):
            return label
    return None


def top_count(n: int) -> int:
    return max(1, int(math.ceil(n * 0.10))) if n else 0


def hit_rate(df: pd.DataFrame, target: str) -> float:
    if df.empty:
        return np.nan
    y = pd.to_numeric(df[target], errors="coerce")
    return float(y.mean()) if y.notna().any() else np.nan


def classify(row: pd.Series) -> tuple[str, bool, str]:
    seasons = int(row["seasons_tested"])
    sample = int(row["sample_size"])
    selected = int(row["model_selected_count"])
    lift = float(row["lift_over_adp"]) if pd.notna(row["lift_over_adp"]) else np.nan
    pct = float(row["pct_seasons_model_beats_adp"]) if pd.notna(row["pct_seasons_model_beats_adp"]) else 0.0
    small_sample = seasons < 3 or sample < 50 or selected < 10
    repeatable = bool(pd.notna(lift) and lift > 0 and pct >= 50.0 and not small_sample)
    if pd.isna(lift) or lift <= 0 or pct < 50.0 or small_sample:
        return "Not Useful", repeatable, "small_sample" if small_sample else ""
    if seasons >= 5 and sample >= 100 and lift >= 0.08 and pct >= 65.0:
        return "Strong Draft Signal", repeatable, ""
    return "Tie-Breaker Only", repeatable, ""


def examples(rows: pd.DataFrame, target: str) -> str:
    if rows.empty:
        return ""
    winners = rows[pd.to_numeric(rows[target], errors="coerce").eq(1)].copy()
    if winners.empty:
        winners = rows.copy()
    winners = winners.sort_values(["season", "model_score"], ascending=[False, False]).head(8)
    parts = []
    for _, row in winners.iterrows():
        adp = pd.to_numeric(pd.Series([row.get("overall_adp")]), errors="coerce").iloc[0]
        adp_text = "?" if pd.isna(adp) else f"{adp:.1f}"
        parts.append(f"{int(row['season'])} {row['player_name']} ADP {adp_text}")
    return "; ".join(parts)


def score_position(dataset: pd.DataFrame, position: str) -> pd.DataFrame:
    cfg = CONFIG[position]
    target = cfg["target"]
    df = dataset[dataset["position"].eq(position)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    for col in ["overall_adp", "positional_adp", "projection_available_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        df["season"].between(2014, 2024)
        & df["overall_adp"].notna()
        & df["positional_adp"].notna()
        & df["projection_available_flag"].eq(1)
        & pd.to_numeric(df[target], errors="coerce").notna()
    ].copy()
    df["overall_bucket"] = df["overall_adp"].apply(lambda x: bucket_label(x, OVERALL_BUCKETS))
    df["positional_bucket"] = df["positional_adp"].apply(lambda x: bucket_label(x, POSITIONAL_BUCKETS))
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    available_features = [f for f in FEATURES if f in df.columns]

    scored_frames = []
    for test_season in seasons:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"].eq(test_season)].copy()
        if train.empty or test.empty:
            continue
        scores, status = fit_predict(train, test, target, available_features, "logistic")
        test["model_score"] = scores
        test["fit_status"] = status
        scored_frames.append(test)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if scored.empty:
        return pd.DataFrame()

    season_bucket_rows = []
    selected_examples = []
    for bucket_type, bucket_col, adp_col in [
        ("overall_adp", "overall_bucket", "overall_adp"),
        ("positional_adp", "positional_bucket", "positional_adp"),
    ]:
        for (season, bucket), group in scored.groupby(["season", bucket_col], dropna=True):
            group = group[group["model_score"].notna()].copy()
            if group.empty:
                continue
            n = len(group)
            pick_n = top_count(n)
            model_top = group.sort_values("model_score", ascending=False).head(pick_n).copy()
            adp_top = group.sort_values(adp_col, ascending=True).head(pick_n).copy()
            model_hit = hit_rate(model_top, target)
            adp_hit = hit_rate(adp_top, target)
            lift = model_hit - adp_hit if pd.notna(model_hit) and pd.notna(adp_hit) else np.nan
            season_bucket_rows.append(
                {
                    "position": position,
                    "test_season": int(season),
                    "bucket_type": bucket_type,
                    "bucket": f"{position}{bucket}" if bucket_type == "positional_adp" else bucket,
                    "target": target,
                    "model_name": cfg["model_name"],
                    "feature_group": cfg["feature_group"],
                    "sample_size": n,
                    "selected_count": pick_n,
                    "model_hit_rate": model_hit,
                    "adp_hit_rate": adp_hit,
                    "lift_over_adp": lift,
                    "model_beats_adp": bool(pd.notna(lift) and lift > 0),
                    "average_model_score": float(model_top["model_score"].mean()),
                }
            )
            model_top["bucket_type"] = bucket_type
            model_top["bucket"] = f"{position}{bucket}" if bucket_type == "positional_adp" else bucket
            selected_examples.append(model_top)

    season_df = pd.DataFrame(season_bucket_rows)
    selected_df = pd.concat(selected_examples, ignore_index=True) if selected_examples else pd.DataFrame()
    rows = []
    grouped = season_df.groupby(["position", "bucket_type", "bucket", "target", "model_name", "feature_group"], dropna=False)
    for keys, group in grouped:
        selected = group["selected_count"].sum()
        model_hits = (group["model_hit_rate"] * group["selected_count"]).sum()
        adp_hits = (group["adp_hit_rate"] * group["selected_count"]).sum()
        row = {
            "position": keys[0],
            "bucket_type": keys[1],
            "bucket": keys[2],
            "target": keys[3],
            "model_name": keys[4],
            "feature_group": keys[5],
            "seasons_tested": int(group["test_season"].nunique()),
            "sample_size": int(group["sample_size"].sum()),
            "model_selected_count": int(selected),
            "adp_selected_count": int(selected),
            "model_hit_rate": model_hits / selected if selected else np.nan,
            "adp_hit_rate": adp_hits / selected if selected else np.nan,
            "lift_over_adp": (model_hits - adp_hits) / selected if selected else np.nan,
            "seasons_model_beats_adp": int(group["model_beats_adp"].sum()),
            "pct_seasons_model_beats_adp": float(group["model_beats_adp"].sum() / group["test_season"].nunique() * 100.0),
            "average_model_score": float(group["average_model_score"].mean()),
        }
        c, repeatable, small = classify(pd.Series(row))
        row["repeatable_lift"] = repeatable
        row["small_sample_flag"] = small
        row["classification"] = c
        if not selected_df.empty:
            mask = selected_df["bucket_type"].eq(row["bucket_type"]) & selected_df["bucket"].eq(row["bucket"])
            row["top_historical_player_examples"] = examples(selected_df[mask], target)
        else:
            row["top_historical_player_examples"] = ""
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["bucket_type", "bucket", "classification", "lift_over_adp"], ascending=[True, True, True, False])


def write_report(wr: pd.DataFrame, rb: pd.DataFrame) -> None:
    def section(label: str, df: pd.DataFrame) -> list[str]:
        if df.empty:
            return [f"## {label}", "", "No buckets were tested.", ""]
        useful = df[df["classification"].ne("Not Useful")]
        strong = df[df["classification"].eq("Strong Draft Signal")]
        lines = [
            f"## {label}",
            "",
            f"Buckets tested: `{len(df)}`.",
            f"Tie-Breaker Only buckets: `{int(df['classification'].eq('Tie-Breaker Only').sum())}`.",
            f"Strong Draft Signal buckets: `{int(df['classification'].eq('Strong Draft Signal').sum())}`.",
            "",
        ]
        if useful.empty:
            lines.append("No draft window cleared the repeatability gate.")
        else:
            lines.append(useful[[
                "bucket_type", "bucket", "seasons_tested", "sample_size", "model_hit_rate",
                "adp_hit_rate", "lift_over_adp", "pct_seasons_model_beats_adp", "classification",
                "top_historical_player_examples",
            ]].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        if not strong.empty:
            lines.append("Strong buckets require additional review before any app use.")
            lines.append("")
        return lines

    combined = pd.concat([wr, rb], ignore_index=True)
    any_strong = bool(not combined.empty and combined["classification"].eq("Strong Draft Signal").any())
    any_app = False
    lines = [
        "# Projection Bucket/Window Lightweight Report",
        "",
        "Date: 2026-07-07",
        "",
        "Scope: research-only. This lightweight script fits only the best WR/RB projection candidate models and does not rerun the full model grid or modify the app.",
        "",
        "## Required Answers",
        "",
        f"Does the WR projection Tie-Breaker signal work in any draft window? {'Yes, in the WR buckets classified above Not Useful.' if not wr.empty and wr['classification'].ne('Not Useful').any() else 'No repeatable WR bucket cleared the gate.'}",
        "",
        f"Does the RB projection Tie-Breaker signal work in any draft window? {'Yes, in the RB buckets classified above Not Useful.' if not rb.empty and rb['classification'].ne('Not Useful').any() else 'No repeatable RB bucket cleared the gate.'}",
        "",
        f"Does any bucket reach Strong Draft Signal? {'Yes' if any_strong else 'No'}.",
        "",
        "Is anything App-Ready? No. This remains research-only and the projection model is weaker than the best ADP-only/non-projection full-pool models.",
        "",
        "Did projected volume help in similarly priced player comparisons? The bucket tables below answer that directly; only buckets with positive lift and at least 50% season repeatability are promoted.",
        "",
        "Is the signal coming from projection features or mostly ADP? The tested model includes ADP plus projection and expanded features. Because prior full-pool validation found stronger ADP-only/non-projection models, any promoted bucket should be treated as secondary evidence, not proof that projected volume alone is the source.",
        "",
    ]
    lines.extend(section("WR Buckets", wr))
    lines.extend(section("RB Buckets", rb))
    not_useful = combined[combined["classification"].eq("Not Useful")]
    tie = combined[combined["classification"].eq("Tie-Breaker Only")]
    strong = combined[combined["classification"].eq("Strong Draft Signal")]
    lines.extend([
        "## Classification Summary",
        "",
        f"Not Useful buckets: `{len(not_useful)}`.",
        f"Tie-Breaker Only buckets: `{len(tie)}`.",
        f"Strong Draft Signal buckets: `{len(strong)}`.",
        "App-Ready buckets: `0`.",
        "",
        "## Next Research Step",
        "",
        "Since no projection bucket was promoted, the next step is to compare the best near-miss buckets against ADP-only and non-projection models, then check whether projection fields add signal beyond ADP or mostly travel with ADP.",
        "",
        "Recommended next Codex prompt:",
        "",
        "```text",
        "Continue research-only validation in research/validation_v1. Compare the best near-miss projection buckets against ADP-only and non-projection models in the same buckets, then identify whether projection volume fields add signal beyond ADP. Do not modify the app.",
        "```",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset = pd.read_csv(DATASET)
    wr = score_position(dataset, "WR")
    rb = score_position(dataset, "RB")
    wr.to_csv(WR_OUTPUT, index=False)
    rb.to_csv(RB_OUTPUT, index=False)
    write_report(wr, rb)
    wr_tie = int(wr["classification"].eq("Tie-Breaker Only").sum()) if not wr.empty else 0
    rb_tie = int(rb["classification"].eq("Tie-Breaker Only").sum()) if not rb.empty else 0
    any_strong = bool(
        (not wr.empty and wr["classification"].eq("Strong Draft Signal").any())
        or (not rb.empty and rb["classification"].eq("Strong Draft Signal").any())
    )
    print(f"WR buckets tested: {len(wr)}")
    print(f"RB buckets tested: {len(rb)}")
    print(f"WR Tie-Breaker buckets: {wr_tie}")
    print(f"RB Tie-Breaker buckets: {rb_tie}")
    print(f"any Strong Draft Signal: {'yes' if any_strong else 'no'}")
    print("anything App-Ready: no")
    print(f"report path: {REPORT}")
    print("recommended next Codex prompt: Continue research-only validation in research/validation_v1. Compare the best near-miss projection buckets against ADP-only and non-projection models in the same buckets, then identify whether projection volume fields add signal beyond ADP. Do not modify the app.")


if __name__ == "__main__":
    main()

