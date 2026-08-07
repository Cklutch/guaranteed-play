"""
Step 4 / 4b summary: does a candidate feature block (Step 3a workload data,
Step 4b divergence features) actually add lift over the ADP baseline?

Compares feature groups on MATCHED SEASONS ONLY (seasons where every group
actually ran), because different feature groups can run in different season
ranges depending on which of their inputs exist -- comparing raw means
across groups with different season coverage is not apples-to-apples.
"""
from __future__ import annotations

import pandas as pd

from validation_utils import VALIDATION_DIR

POSITIONS = ["qb", "rb", "wr", "te"]
PRIMARY_TARGET = {"qb": "QB_Top12", "rb": "RB_Top24", "wr": "WR_Top24", "te": "TE_Top12"}

# A candidate group must clear the best baseline group by at least this much
# mean lift AND beat it in a majority of individual seasons before we call it
# a win. The first Step 4 run reported "WORKLOAD HELPS" for WR on a +0.0018
# delta -- far inside season-to-season noise, and still *negative* lift
# overall (it merely lost to ADP slightly less than the alternative). Sign
# alone is not evidence.
MIN_MEANINGFUL_DELTA = 0.01
MIN_SEASON_WIN_RATE = 0.5

# Feature blocks under test, as opposed to the pre-existing baselines.
CANDIDATE_MARKERS = ("workload", "divergence", "archetype", "adp_all_v1")


def _strip_model_suffix(name: str) -> str:
    for suffix in ["_regularized_logistic", "_random_forest", "_logistic"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_candidate(group: str) -> bool:
    return any(marker in group for marker in CANDIDATE_MARKERS)


def load_matched(position: str) -> pd.DataFrame | None:
    """Fitted rows for the position's primary target, restricted to seasons
    in which every feature group ran."""
    path = VALIDATION_DIR / f"{position}_validation_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    target = PRIMARY_TARGET[position]
    fit = df[(df["status"] == "fit") & df["lift_over_baseline"].notna() & (df["target"] == target)].copy()
    if fit.empty:
        return None
    fit["group"] = fit["model_name"].apply(_strip_model_suffix)

    per_season_groups = fit.groupby("test_season")["group"].nunique()
    matched = per_season_groups[per_season_groups == fit["group"].nunique()].index
    matched_fit = fit[fit["test_season"].isin(matched)]
    return matched_fit if not matched_fit.empty else fit


def summarize_position(matched_fit: pd.DataFrame, position: str) -> pd.DataFrame:
    out = matched_fit.groupby("group").agg(
        mean_lift_over_adp=("lift_over_baseline", "mean"),
        beats_adp_rate=("beats_adp", "mean"),
        mean_auc=("auc", "mean"),
        mean_adp_auc=("adp_auc", "mean"),
        seasons=("test_season", "nunique"),
    ).reset_index()
    out.insert(0, "position", position.upper())
    out.insert(1, "target", PRIMARY_TARGET[position])
    out["is_candidate_group"] = out["group"].apply(_is_candidate)
    return out.sort_values("mean_lift_over_adp", ascending=False)


def verdict(matched_fit: pd.DataFrame, summary: pd.DataFrame, position: str) -> str:
    candidates = summary[summary["is_candidate_group"]]
    baselines = summary[~summary["is_candidate_group"]]
    if candidates.empty or baselines.empty:
        return f"{position.upper()}: no comparison possible (missing candidate or baseline groups)"

    best_c = candidates.loc[candidates["mean_lift_over_adp"].idxmax()]
    best_b = baselines.loc[baselines["mean_lift_over_adp"].idxmax()]
    delta = best_c["mean_lift_over_adp"] - best_b["mean_lift_over_adp"]

    # Season-by-season head-to-head, best candidate vs. best baseline.
    per_season = matched_fit.groupby(["group", "test_season"])["lift_over_baseline"].mean().unstack(0)
    season_win_rate = float("nan")
    if best_c["group"] in per_season.columns and best_b["group"] in per_season.columns:
        head_to_head = per_season[[best_c["group"], best_b["group"]]].dropna()
        if not head_to_head.empty:
            season_win_rate = float(
                (head_to_head[best_c["group"]] > head_to_head[best_b["group"]]).mean()
            )

    passes_delta = delta >= MIN_MEANINGFUL_DELTA
    passes_seasons = pd.notna(season_win_rate) and season_win_rate > MIN_SEASON_WIN_RATE

    if passes_delta and passes_seasons:
        label = "CANDIDATE HELPS"
    elif delta > 0:
        label = f"no evidence (delta below {MIN_MEANINGFUL_DELTA} and/or season win rate not >{MIN_SEASON_WIN_RATE:.0%})"
    else:
        label = "no improvement"

    win_rate_text = "n/a" if pd.isna(season_win_rate) else f"{season_win_rate:.0%}"
    return (
        f"{position.upper()}: best_candidate={best_c['mean_lift_over_adp']:+.4f} ({best_c['group']}) | "
        f"best_baseline={best_b['mean_lift_over_adp']:+.4f} ({best_b['group']}) | "
        f"delta={delta:+.4f} | candidate wins {win_rate_text} of seasons -> {label}"
    )


def main() -> None:
    frames = []
    verdicts = []
    for position in POSITIONS:
        matched_fit = load_matched(position)
        if matched_fit is None:
            print(f"[skip] {position.upper()}: no results file yet")
            continue
        summary = summarize_position(matched_fit, position)
        frames.append(summary)
        print(f"\n=== {position.upper()} ({summary['target'].iloc[0]}, {int(summary['seasons'].iloc[0])} matched seasons) ===")
        print(summary.drop(columns=["position", "target", "seasons"]).to_string(index=False))
        verdicts.append(verdict(matched_fit, summary, position))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = VALIDATION_DIR / "workload_v1_step4_summary.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSummary written: {out_path}")

        print(
            f"\n=== VERDICT (needs delta >= {MIN_MEANINGFUL_DELTA} AND candidate winning "
            f">{MIN_SEASON_WIN_RATE:.0%} of seasons) ==="
        )
        for line in verdicts:
            print(line)


if __name__ == "__main__":
    main()
