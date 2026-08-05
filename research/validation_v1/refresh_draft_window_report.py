from __future__ import annotations

from pathlib import Path
import pandas as pd

D = Path("research/validation_v1")
wr = pd.read_csv(D / "wr_bucket_lift_analysis.csv")
rb = pd.read_csv(D / "rb_bucket_lift_analysis.csv")


def best(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            ["bucket_type", "bucket", "lift_over_adp", "pct_seasons_model_beats_adp", "sample_size"],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(["position", "bucket_type", "bucket"])
        .copy()
    )


wrb = best(wr)
rbb = best(rb)
combined = pd.concat([wrb, rbb], ignore_index=True)
useful = combined[combined["classification"].ne("Not Useful")]
further = combined[(combined["lift_over_adp"] > 0) & combined["classification"].eq("Not Useful")].copy()

lines: list[str] = []
lines += [
    "# Draft Window Edge Report",
    "",
    "Date: 2026-07-07",
    "",
    "Scope: research-only bucket validation using the existing WR/RB walk-forward model families. No Streamlit app changes, no UI, and no new model families.",
    "",
    "## Method",
    "",
    "For each season, position, target, model, and ADP bucket, the analysis compares the model top decile within that bucket against the ADP baseline top decile within the same bucket. This tests similarly priced players rather than the full draft board.",
    "",
    "Only `Top24` and `Top12` targets are used for classification. ADP-defined underpriced labels are excluded from draft-window edge claims because they would bias the comparison against ADP.",
    "",
    "Repeatability gate: a bucket needs positive lift across multiple seasons and at least 50% of tested seasons beating ADP before it can become `Tie-Breaker Only`. No bucket met that gate.",
    "",
]

for label, df in [("WR", wrb), ("RB", rbb)]:
    lines += [f"## {label} Bucket Summary", ""]
    view = [
        "bucket_type", "bucket", "best_target_in_bucket", "model_name", "seasons_tested", "sample_size",
        "model_hit_rate", "adp_hit_rate", "lift_over_adp", "seasons_model_beats_adp",
        "pct_seasons_model_beats_adp", "classification", "small_sample_flag",
    ]
    lines.append(df[view].to_markdown(index=False, floatfmt=".3f"))
    lines += ["", f"{label} answer: no tested draft range currently beats ADP with repeatable positive lift.", ""]
    examples = ["bucket_type", "bucket", "best_target_in_bucket", "model_name", "top_historical_player_examples"]
    lines += [f"### {label} Examples", "", df[examples].to_markdown(index=False), ""]

lines += [
    "## Similarly Priced Player Test",
    "",
    "The bucket method is the similarly priced player test. It does show some positive average lift pockets, especially RB picks 25-48 and WR positional ADP 1-12 or 25-48. However, these pockets fail repeatability: the model beats ADP in only about 7-40% of tested seasons depending on the bucket. Several best rows also come from ADP-feature or ADP-only models, which means the apparent lift is not yet independent player-evaluation edge.",
    "",
    "## Required Answers",
    "",
    "Does any WR draft range beat ADP?",
    "",
    "- No. WR has positive average pockets, but none are repeatable enough for Tie-Breaker Only.",
    "",
    "Does any RB draft range beat ADP?",
    "",
    "- No. RB has larger positive average pockets, but the season-to-season repeatability is weak.",
    "",
    "Is there any range where the model is at least Tie-Breaker Only?",
    "",
    "- " + ("Yes; inspect the CSV rows classified above Not Useful." if not useful.empty else "No."),
    "",
    "Which ranges should be ignored?",
    "",
    "- Ignore all buckets for draft recommendations right now because every best bucket row is classified `Not Useful`.",
    "- Especially ignore WR overall ADP 97-120, WR positional ADP 49+, and any bucket where lift is zero or negative.",
    "",
    "Which ranges deserve further research?",
    "",
]
if further.empty:
    lines.append("- None yet.")
else:
    lines.append("- These ranges had positive average lift but failed repeatability gates:")
    lines.append(
        further[
            [
                "position", "bucket_type", "bucket", "best_target_in_bucket", "seasons_tested",
                "sample_size", "lift_over_adp", "pct_seasons_model_beats_adp", "small_sample_flag",
            ]
        ].to_markdown(index=False, floatfmt=".3f")
    )
lines += [
    "",
    "What exact feature improvement is most likely to create edge next?",
    "",
    "- Add true preseason projection and role-context fields that ADP does not fully encode: projected points, projected targets/carries/receptions, depth-chart role, injury/suspension flags, rookie status, team implied points, offensive environment, RB carries/receptions, WR routes/air yards, and vacated opportunity. The current features mostly rediscover what ADP already prices in.",
    "",
    "Should we continue WR, RB, or both?",
    "",
    "- Continue both for research, but do not integrate either. WR is closer to market in full-pool validation; RB shows bigger bucket-level average lift pockets but weaker repeatability. The next useful step is better pre-draft features, not a new UI.",
    "",
    "## Output Files",
    "",
    "- `research/validation_v1/wr_bucket_lift_analysis.csv`",
    "- `research/validation_v1/rb_bucket_lift_analysis.csv`",
    "- `research/validation_v1/draft_window_edge_report.md`",
]
(D / "draft_window_edge_report.md").write_text("\n".join(lines), encoding="utf-8")
print("report refreshed with examples")

