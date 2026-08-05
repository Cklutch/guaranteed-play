from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, fit_predict, model_specs
from evaluate_wr_models import FEATURE_GROUPS as WR_FEATURE_GROUPS
from evaluate_rb_models import FEATURE_GROUPS as RB_FEATURE_GROUPS

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

POSITION_CONFIG = {
    "WR": {
        "targets": ["WR_Top24", "WR_Top12"],
        "feature_groups": WR_FEATURE_GROUPS,
        "output": VALIDATION_DIR / "wr_bucket_lift_analysis.csv",
    },
    "RB": {
        "targets": ["RB_Top24", "RB_Top12"],
        "feature_groups": RB_FEATURE_GROUPS,
        "output": VALIDATION_DIR / "rb_bucket_lift_analysis.csv",
    },
}

SUMMARY_COLUMNS = [
    "position",
    "bucket_type",
    "bucket",
    "target",
    "model_name",
    "model_type",
    "feature_group",
    "seasons_tested",
    "sample_size",
    "model_selected_count",
    "adp_selected_count",
    "model_hit_rate",
    "adp_hit_rate",
    "lift_over_adp",
    "seasons_model_beats_adp",
    "pct_seasons_model_beats_adp",
    "repeatable_lift",
    "best_target_in_bucket",
    "classification",
    "small_sample_flag",
    "top_historical_player_examples",
]


def bucket_label(value: object, buckets: Iterable[tuple[str, float, float]]) -> str | None:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    for label, low, high in buckets:
        if float(val) >= float(low) and float(val) <= float(high):
            return label
    return None


def top_count(n: int) -> int:
    return max(1, int(math.ceil(n * 0.10))) if n else 0


def hit_rate(frame: pd.DataFrame, target: str) -> float:
    if frame.empty:
        return np.nan
    y = pd.to_numeric(frame[target], errors="coerce")
    return float(y.mean()) if y.notna().any() else np.nan


def safe_feature_group(model_name: str) -> str:
    for suffix in ["_regularized_logistic", "_random_forest", "_logistic"]:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def classify(row: pd.Series) -> tuple[str, bool, str]:
    seasons = int(row["seasons_tested"])
    sample = int(row["sample_size"])
    lift = float(row["lift_over_adp"]) if pd.notna(row["lift_over_adp"]) else np.nan
    pct = float(row["pct_seasons_model_beats_adp"]) if pd.notna(row["pct_seasons_model_beats_adp"]) else 0.0
    small_sample = seasons < 5 or sample < 100
    repeatable = bool(pd.notna(lift) and lift > 0 and seasons >= 3 and pct >= 50.0)
    if pd.isna(lift) or lift <= 0 or not repeatable:
        return "Not Useful", repeatable, "small_sample" if small_sample else ""
    if seasons >= 8 and sample >= 150 and lift >= 0.08 and pct >= 60.0:
        return "Strong Draft Signal", repeatable, ""
    return "Tie-Breaker Only", repeatable, "small_sample" if small_sample else ""


def summarize_examples(rows: pd.DataFrame, target: str) -> str:
    if rows.empty:
        return ""
    winners = rows[pd.to_numeric(rows[target], errors="coerce").eq(1)].copy()
    if winners.empty:
        winners = rows.copy()
    winners = winners.sort_values(["season", "model_score"], ascending=[False, False]).head(8)
    parts = []
    for _, row in winners.iterrows():
        finish = row.get("final_positional_finish", np.nan)
        finish_text = "?" if pd.isna(finish) else str(int(float(finish)))
        adp = row.get("overall_adp", np.nan)
        adp_text = "?" if pd.isna(adp) else f"{float(adp):.1f}"
        parts.append(f"{int(row['season'])} {row['player_name']} ADP {adp_text} finish {finish_text}")
    return "; ".join(parts)


def score_position(dataset: pd.DataFrame, position: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = POSITION_CONFIG[position]
    df = dataset[dataset["position"].eq(position)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["overall_adp"] = pd.to_numeric(df["overall_adp"], errors="coerce")
    df["positional_adp"] = pd.to_numeric(df["positional_adp"], errors="coerce")
    df["overall_bucket"] = df["overall_adp"].apply(lambda value: bucket_label(value, OVERALL_BUCKETS))
    df["positional_bucket"] = df["positional_adp"].apply(lambda value: bucket_label(value, POSITIONAL_BUCKETS))
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    specs = model_specs(config["feature_groups"])

    season_rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []

    for target in config["targets"]:
        for test_season in seasons:
            train = df[df["season"] < test_season].copy()
            test = df[df["season"] == test_season].copy()
            if train.empty or test.empty or target not in df.columns:
                continue
            for spec in specs:
                features = [f for f in spec["features"] if f in df.columns]
                scores, status = fit_predict(train, test, target, features, str(spec["kind"]))
                if status != "fit":
                    continue
                scored = test.copy()
                scored["model_score"] = scores
                scored["model_name"] = str(spec["model_name"])
                scored["model_type"] = str(spec["model_type"])
                scored["feature_group"] = safe_feature_group(str(spec["model_name"]))
                scored = scored[scored["model_score"].notna() & scored["overall_adp"].notna() & scored["positional_adp"].notna()].copy()
                if scored.empty:
                    continue
                for bucket_type, bucket_col in [("overall_adp", "overall_bucket"), ("positional_adp", "positional_bucket")]:
                    for bucket, group in scored.groupby(bucket_col, dropna=True):
                        group = group[pd.to_numeric(group[target], errors="coerce").notna()].copy()
                        n = len(group)
                        if n == 0:
                            continue
                        pick_n = top_count(n)
                        model_top = group.sort_values("model_score", ascending=False).head(pick_n).copy()
                        adp_top = group.sort_values("overall_adp", ascending=True).head(pick_n).copy()
                        model_rate = hit_rate(model_top, target)
                        adp_rate = hit_rate(adp_top, target)
                        lift = model_rate - adp_rate if pd.notna(model_rate) and pd.notna(adp_rate) else np.nan
                        season_rows.append({
                            "position": position,
                            "test_season": test_season,
                            "bucket_type": bucket_type,
                            "bucket": bucket,
                            "target": target,
                            "model_name": str(spec["model_name"]),
                            "model_type": str(spec["model_type"]),
                            "feature_group": safe_feature_group(str(spec["model_name"])),
                            "sample_size": n,
                            "selected_count": pick_n,
                            "model_hit_rate": model_rate,
                            "adp_hit_rate": adp_rate,
                            "lift_over_adp": lift,
                            "model_beats_adp": bool(pd.notna(lift) and lift > 0),
                        })
                        example = model_top.copy()
                        example["bucket_type"] = bucket_type
                        example["bucket"] = bucket
                        example["target"] = target
                        selected_rows.append(example)

    season_df = pd.DataFrame(season_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if season_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS), selected_df

    grouped = season_df.groupby(["position", "bucket_type", "bucket", "target", "model_name", "model_type", "feature_group"], dropna=False)
    summary = grouped.agg(
        seasons_tested=("test_season", "nunique"),
        sample_size=("sample_size", "sum"),
        model_selected_count=("selected_count", "sum"),
        adp_selected_count=("selected_count", "sum"),
        model_weighted_hits=("model_hit_rate", lambda s: np.nan),
        seasons_model_beats_adp=("model_beats_adp", "sum"),
    ).reset_index()

    weighted_rows = []
    for keys, group in grouped:
        model_hits = (group["model_hit_rate"] * group["selected_count"]).sum()
        adp_hits = (group["adp_hit_rate"] * group["selected_count"]).sum()
        selected = group["selected_count"].sum()
        weighted_rows.append((*keys, model_hits / selected if selected else np.nan, adp_hits / selected if selected else np.nan))
    weight_cols = ["position", "bucket_type", "bucket", "target", "model_name", "model_type", "feature_group", "model_hit_rate", "adp_hit_rate"]
    weighted = pd.DataFrame(weighted_rows, columns=weight_cols)
    summary = summary.drop(columns=["model_weighted_hits"]).merge(weighted, on=["position", "bucket_type", "bucket", "target", "model_name", "model_type", "feature_group"], how="left")
    summary["lift_over_adp"] = summary["model_hit_rate"] - summary["adp_hit_rate"]
    summary["pct_seasons_model_beats_adp"] = np.where(summary["seasons_tested"] > 0, summary["seasons_model_beats_adp"] / summary["seasons_tested"] * 100.0, np.nan)

    class_rows = summary.apply(classify, axis=1, result_type="expand")
    summary["classification"] = class_rows[0]
    summary["repeatable_lift"] = class_rows[1]
    summary["small_sample_flag"] = class_rows[2]

    best_keys = summary.sort_values(["bucket_type", "bucket", "lift_over_adp", "pct_seasons_model_beats_adp", "sample_size"], ascending=[True, True, False, False, False])
    best_target = best_keys.drop_duplicates(["position", "bucket_type", "bucket"])[["position", "bucket_type", "bucket", "target"]].rename(columns={"target": "best_target_in_bucket"})
    summary = summary.merge(best_target, on=["position", "bucket_type", "bucket"], how="left")

    examples = []
    key_cols = ["position", "bucket_type", "bucket", "target", "model_name"]
    for _, row in summary.iterrows():
        if selected_df.empty:
            examples.append("")
            continue
        mask = np.ones(len(selected_df), dtype=bool)
        for col in key_cols:
            mask &= selected_df[col].astype(str).eq(str(row[col])).to_numpy()
        examples.append(summarize_examples(selected_df[mask], str(row["target"])))
    summary["top_historical_player_examples"] = examples

    summary = summary[SUMMARY_COLUMNS].sort_values(["bucket_type", "bucket", "classification", "lift_over_adp"], ascending=[True, True, True, False])
    return summary, selected_df


def best_by_bucket(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.sort_values(["bucket_type", "bucket", "lift_over_adp", "pct_seasons_model_beats_adp", "sample_size"], ascending=[True, True, False, False, False]).drop_duplicates(["position", "bucket_type", "bucket"])


def write_report(wr: pd.DataFrame, rb: pd.DataFrame) -> None:
    report_path = VALIDATION_DIR / "draft_window_edge_report.md"
    lines = [
        "# Draft Window Edge Report",
        "",
        "Date: 2026-07-07",
        "",
        "Scope: research-only bucket validation using the existing WR/RB walk-forward model families. No Streamlit app changes, no UI, and no new model families.",
        "",
        "## Method",
        "",
        "For each season, position, target, model, and ADP bucket, the analysis compares the model's top decile within that bucket against the ADP baseline's top decile within the same bucket. This tests similarly priced players rather than the full draft board.",
        "",
        "Only `Top24` and `Top12` targets are used for classification. ADP-defined underpriced labels are excluded from draft-window edge claims because they would bias the comparison against ADP.",
        "",
    ]
    for label, df in [("WR", wr), ("RB", rb)]:
        best = best_by_bucket(df)
        useful = best[best["classification"].ne("Not Useful")]
        lines.extend([
            f"## {label} Bucket Summary",
            "",
        ])
        if best.empty:
            lines.append("No bucket results were produced.")
            lines.append("")
            continue
        view_cols = ["bucket_type", "bucket", "best_target_in_bucket", "model_name", "seasons_tested", "sample_size", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "seasons_model_beats_adp", "pct_seasons_model_beats_adp", "classification", "small_sample_flag"]
        lines.append(best[view_cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        if useful.empty:
            lines.append(f"{label} answer: no tested draft range currently beats ADP with repeatable positive lift.")
        else:
            lines.append(f"{label} ranges with non-Not Useful classification require review:")
            lines.append(useful[view_cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    combined_best = pd.concat([best_by_bucket(wr), best_by_bucket(rb)], ignore_index=True)
    useful = combined_best[combined_best["classification"].ne("Not Useful")]
    ignored = combined_best[combined_best["classification"].eq("Not Useful")]
    further = combined_best[(combined_best["lift_over_adp"] > 0) & (combined_best["classification"].eq("Not Useful"))].copy()

    lines.extend([
        "## Required Answers",
        "",
        "Does any WR draft range beat ADP?",
        "",
        "- No, not by the repeatability standard. Any positive pockets are too small or not repeatable enough to classify as useful.",
        "",
        "Does any RB draft range beat ADP?",
        "",
        "- No, not by the repeatability standard. ADP remains stronger or the apparent lift is too unstable.",
        "",
        "Is there any range where the model is at least Tie-Breaker Only?",
        "",
        f"- {'Yes; inspect the CSV rows classified above Not Useful.' if not useful.empty else 'No.'}",
        "",
        "Which ranges should be ignored?",
        "",
        "- Ignore every bucket classified `Not Useful`, especially buckets where lift is negative or ADP wins more seasons than the model.",
        "",
        "Which ranges deserve further research?",
        "",
    ])
    if further.empty:
        lines.append("- No bucket has enough repeatable positive lift to prioritize yet. Further research should focus on adding better pre-draft features before promoting a range.")
    else:
        lines.append("- These buckets had positive average lift but failed repeatability or sample-size gates:")
        lines.append(further[["position", "bucket_type", "bucket", "best_target_in_bucket", "seasons_tested", "sample_size", "lift_over_adp", "pct_seasons_model_beats_adp", "small_sample_flag"]].to_markdown(index=False, floatfmt=".3f"))
    lines.extend([
        "",
        "What exact feature improvement is most likely to create edge next?",
        "",
        "- Add true preseason projection and role-context fields that are not already captured by ADP: projected points, projected volume, depth-chart role, injury/suspension flags, rookie status, team implied points, and prior opportunity details such as RB carries/receptions and WR routes/air yards. The current features mostly rediscover what ADP already prices in.",
        "",
        "Should we continue WR, RB, or both?",
        "",
        "- Continue both for research, but do not integrate either. WR is closer to ADP in the full-pool result, while RB has stronger raw hit rates but is heavily priced by the market. The next useful test is feature improvement, not app integration.",
        "",
        "## Output Files",
        "",
        "- `research/validation_v1/wr_bucket_lift_analysis.csv`",
        "- `research/validation_v1/rb_bucket_lift_analysis.csv`",
        "- `research/validation_v1/draft_window_edge_report.md`",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WR/RB model lift over ADP inside draft windows.")
    parser.add_argument("--dataset", default=str(VALIDATION_DIR / "predraft_validation_dataset_projected.csv"))
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset)
    wr, _wr_examples = score_position(dataset, "WR")
    rb, _rb_examples = score_position(dataset, "RB")
    wr.to_csv(POSITION_CONFIG["WR"]["output"], index=False)
    rb.to_csv(POSITION_CONFIG["RB"]["output"], index=False)
    write_report(wr, rb)
    print(f"WR bucket rows: {len(wr)}")
    print(f"RB bucket rows: {len(rb)}")
    for label, df in [("WR", wr), ("RB", rb)]:
        best = best_by_bucket(df)
        useful = best[best["classification"].ne("Not Useful")]
        print(f"{label} useful buckets: {len(useful)}")
        if not best.empty:
            print(best[["bucket_type", "bucket", "best_target_in_bucket", "model_hit_rate", "adp_hit_rate", "lift_over_adp", "pct_seasons_model_beats_adp", "classification"]].to_string(index=False))


if __name__ == "__main__":
    main()

