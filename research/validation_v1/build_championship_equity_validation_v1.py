"""Championship Equity Score V1 Validation (2026-08-27).

Direct historical backtest of draftkit.championship_equity.calculate_
championship_equity_score() -- the actual live blend (5 components,
CHAMPIONSHIP_EQUITY_WEIGHTS: 25/25/25/10/15), not a description of it --
against research/validation_v1/predraft_validation_dataset.csv (14,771
player-seasons, 1999-2025, all 4 positions), using the same
validation_utils.py methodology (AUC, top-decile hit rate vs. ADP
baseline, lift-over-baseline, per-season + aggregate) as the existing
market_disagreement_score_v1 and floor_ceiling_v1 studies, for direct
comparability.

No model is fit -- calculate_championship_equity_score() is a fixed
heuristic, not a trained model, so there's no train/test split in the ML
sense. Every row is scored once from its own preseason inputs; each season
is then evaluated as a held-out test slice, exactly the way this repo's
non-fitted heuristic scores (simple_score() et al.) are already evaluated
elsewhere in this file.

WHERE THIS IS FAITHFUL TO THE LIVE FORMULA, AND WHERE IT'S A DISCLOSED PROXY
(be honest reading the results -- this is not a literal replay of
production inputs for two of the five components):

  - age_curve_score: FAITHFUL. Real age + position, calls the actual
    draftkit function unmodified.
  - adp_outperformance_score: PROXY. The live version compares this
    year's internal model projection_rank against adp_rank. No internal
    model exists for 1999-2025 seasons, so this substitutes a real,
    available ex-ante analog: rank of PRIOR-SEASON fantasy_ppg (the
    closest thing to "what a simple model would have expected" using
    only information known before the season).
  - breakout_probability: PARTIALLY FAITHFUL. age_score real; adp_rank
    bucket real (from preseason_adp); boom_score/stability_score/
    archetype have no historical source and default to the same neutral
    values they default to for ~87% of the LIVE player pool anyway (see
    CHANGELOG.md), so this isn't introducing a new gap, just inheriting
    the live one.
  - ceiling_score: PROXY, and the weaker one. The live version (as of the
    2026-08-27 rewrite) uses role_usage_td_score + volatility_diagnostic
    from risk_variables.csv. No historical equivalent exists in this
    dataset -- prior_red_zone_usage/prior_goal_line_usage/prior_snap_share
    are 0% populated here. Substituted a real, well-covered (69.4%) TD-
    dependence analog instead: what SHARE of the player's prior-season
    fantasy points came from touchdowns (prior_total_tds * 6 /
    prior_fantasy_points), percentile-scaled to the live function's
    expected 1-5 range. volatility_diagnostic has no analog at all and is
    left unset -- ceiling_score's own "use whichever signal is present"
    logic means it scores off the TD-share proxy alone here. Read
    ceiling_score's historical AUC with that in mind.
  - sportsbook_advantage_score: PROXY. No sportsbook data exists for
    1999-2025. Substituted the ADP-rank-vs-production-rank gap, the SAME
    proxy the already-validated market_disagreement_score_v1 uses (for
    comparability), tier-normalized by ADP-quantile the same way the live
    2026-08-27 fix does.

Run: python research/validation_v1/build_championship_equity_validation_v1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from draftkit.championship_equity import (
    CHAMPIONSHIP_EQUITY_WEIGHTS,
    calculate_age_curve_score,
    calculate_adp_outperformance_score,
    calculate_breakout_probability,
    calculate_ceiling_score,
    calculate_championship_equity_score,
    calculate_sportsbook_advantage_score,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import (  # noqa: E402
    adp_baseline_scores,
    summarize_model,
)

DATASET_PATH = REPO_ROOT / "research/validation_v1/predraft_validation_dataset.csv"
REPORT_PATH = REPO_ROOT / "research/validation_v1/championship_equity_v1_validation_report.md"
PREDICTIONS_PATH = REPO_ROOT / "research/validation_v1/championship_equity_v1_predictions.csv"

SPORTSBOOK_EDGE_TIER_COUNT = 6  # matches draftkit/championship_equity.py

TARGETS = [
    "Beat_ADP_By_12", "Underpriced_Top24", "Underpriced_Top12", "Top24", "Top12",
    "WR_Beat_ADP_By_12", "WR_Underpriced_Top24", "WR_Underpriced_Top12", "WR_Top24", "WR_Top12",
    "RB_Beat_ADP_By_12", "RB_Underpriced_Top24", "RB_Underpriced_Top12", "RB_Top24", "RB_Top12",
    "QB_Beat_ADP_By_12", "QB_Underpriced_Top12", "QB_Top12",
    "TE_Beat_ADP_By_12", "TE_Underpriced_Top12", "TE_Top12",
]

SCORE_COLUMNS = [
    "age_curve_score", "adp_outperformance_score", "breakout_probability",
    "ceiling_score", "sportsbook_advantage_score", "championship_equity_score",
]


def build_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["position"] = out["position"].astype(str).str.upper()

    # Ex-ante "projection" proxy: prior-season fantasy_ppg. Real, available,
    # known before the season -- the closest thing to what a simple model
    # would have projected using only information on hand at draft time.
    prior_ppg = pd.to_numeric(out["prior_fantasy_ppg"], errors="coerce")
    adp = pd.to_numeric(out["preseason_adp"], errors="coerce")

    grp = out.groupby(["season", "position"])
    out["_proxy_projection_rank"] = prior_ppg.groupby([out["season"], out["position"]]).rank(
        ascending=False, method="average"
    )
    out["_proxy_adp_rank"] = adp.groupby([out["season"], out["position"]]).rank(
        ascending=True, method="average"
    )

    # ceiling_score's relevance damping (see calculate_ceiling_score's
    # docstring for why this is required, not optional) -- percentile of
    # prior production, same role projection percentile plays live.
    out["_ceiling_relevance"] = prior_ppg.groupby([out["season"], out["position"]]).rank(
        pct=True
    ).fillna(0.0)

    # TD-dependence proxy for role_usage_td_score (1-5 scale): share of
    # prior-season fantasy points that came from touchdowns. 6 pts/TD is a
    # standard approximation across scoring formats -- exact value doesn't
    # matter since this gets percentile-ranked, not used as a raw number.
    prior_tds = pd.to_numeric(out["prior_total_tds"], errors="coerce")
    prior_pts = pd.to_numeric(out["prior_fantasy_points"], errors="coerce")
    td_points = prior_tds * 6.0
    td_share = (td_points / prior_pts.replace(0, np.nan)).clip(0.0, 1.0)
    td_share_pct = td_share.groupby([out["season"], out["position"]]).rank(pct=True)
    out["role_usage_td_score"] = 1.0 + td_share_pct * 4.0  # -> live's expected 1-5 range
    out["volatility_diagnostic"] = np.nan  # no historical analog -- left unset, see module docstring
    out["archetype_primary"] = np.nan  # no historical analog -- no event lift applied

    # Sportsbook-edge proxy: same ADP-rank-vs-production-rank gap the
    # already-validated market_disagreement_score_v1 uses, tier-normalized
    # by ADP quantile the same way the live 2026-08-27 fix does.
    gap = out["_proxy_adp_rank"] - out["_proxy_projection_rank"]
    has_both = out["_proxy_adp_rank"].notna() & out["_proxy_projection_rank"].notna()
    gap = gap.where(has_both)

    def _tier_rank(group: pd.DataFrame) -> pd.Series:
        adp_rank_for_tier = group["_proxy_adp_rank"]
        g = group["_gap"]
        try:
            tier = pd.qcut(adp_rank_for_tier, q=SPORTSBOOK_EDGE_TIER_COUNT, duplicates="drop")
        except ValueError:
            return g.rank(pct=True)
        return g.groupby(tier, observed=True).rank(pct=True)

    out["_gap"] = gap
    out["_sportsbook_edge_percentile"] = (
        out.groupby(["season", "position"], group_keys=False).apply(_tier_rank)
    )
    out = out.drop(columns=["_gap"])

    return out


def score_row(row: pd.Series) -> pd.Series:
    columns = {
        "position": "position",
        "age": "age",
        "projection_rank": "_proxy_projection_rank",
        "adp_rank": "_proxy_adp_rank",
        "boom": None,
        "stability": None,
        "archetype": None,
    }
    age = calculate_age_curve_score(row, columns)
    adp_out = calculate_adp_outperformance_score(row, columns)
    breakout = calculate_breakout_probability(row, columns)
    ceiling = calculate_ceiling_score(row)
    sportsbook = calculate_sportsbook_advantage_score(row)
    blended = calculate_championship_equity_score(row, columns, CHAMPIONSHIP_EQUITY_WEIGHTS)
    return pd.Series({
        "age_curve_score": age,
        "adp_outperformance_score": adp_out,
        "breakout_probability": breakout,
        "ceiling_score": ceiling,
        "sportsbook_advantage_score": sportsbook,
        "championship_equity_score": blended,
    })


def evaluate(df: pd.DataFrame, score_col: str, targets: list[str]) -> pd.DataFrame:
    rows = []
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    for target in targets:
        if target not in df.columns:
            continue
        for season in seasons:
            test = df[df["season"] == season].copy()
            if test.empty or target not in test.columns:
                continue
            y = pd.to_numeric(test[target], errors="coerce")
            if y.notna().sum() < 5 or y.dropna().nunique() < 2:
                continue
            baseline = adp_baseline_scores(test)  # checks overall_adp, falls back to preseason_adp
            summary = summarize_model(
                test=test, target=target, scores=test[score_col], baseline_scores=baseline,
                test_season=season, model_name=score_col, model_type="heuristic",
                feature_group=score_col, status="evaluated",
            )
            rows.append(summary)
    return pd.DataFrame(rows)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    agg_rows = []
    for (target, model_name), group in results.groupby(["target", "model_name"]):
        seasons_tested = int(group["test_season"].nunique())
        beat = group[group["beats_adp"] == True]
        agg_rows.append({
            "target": target,
            "model_name": model_name,
            "seasons_tested": seasons_tested,
            "seasons_beat_adp": int(beat["test_season"].nunique()),
            "mean_auc": round(float(group["auc"].mean()), 4),
            "mean_adp_auc": round(float(group["adp_auc"].mean()), 4),
            "mean_lift_over_baseline": round(float(group["lift_over_baseline"].mean()), 4),
            "median_lift_over_baseline": round(float(group["lift_over_baseline"].median()), 4),
            "mean_sample_size": round(float(group["sample_size"].mean()), 1),
        })
    return pd.DataFrame(agg_rows).sort_values(["model_name", "mean_lift_over_baseline"], ascending=[True, False])


def classify(mean_auc: float, seasons_beat_adp: int, seasons_tested: int) -> str:
    if pd.isna(mean_auc) or seasons_tested == 0:
        return "Insufficient Data"
    beat_rate = seasons_beat_adp / seasons_tested
    if mean_auc >= 0.60 and beat_rate >= 0.60:
        return "Real Signal"
    if mean_auc >= 0.53 and beat_rate >= 0.45:
        return "Weak Research Signal"
    return "No Signal"


def main() -> int:
    print(f"Loading {DATASET_PATH.name}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"{len(df)} player-seasons, {df['season'].nunique()} seasons ({df['season'].min()}-{df['season'].max()})")

    print("Building proxy features (see module docstring for what's faithful vs. proxied)...")
    df = build_proxy_features(df)

    print(f"Scoring {len(df)} rows through the real calculate_championship_equity_score()...")
    scores = df.apply(score_row, axis=1)
    df = pd.concat([df, scores], axis=1)

    df[["player_name", "season", "position"] + SCORE_COLUMNS].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions written to {PREDICTIONS_PATH.name}")

    print("Evaluating against real historical outcomes (AUC, hit rate vs. ADP baseline)...")
    all_results = []
    for score_col in SCORE_COLUMNS:
        all_results.append(evaluate(df, score_col, TARGETS))
    results = pd.concat(all_results, ignore_index=True)
    agg = aggregate(results)
    agg["final_classification"] = agg.apply(
        lambda r: classify(r["mean_auc"], r["seasons_beat_adp"], r["seasons_tested"]), axis=1
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Championship Equity Score V1 Validation\n\n")
        f.write(
            "Direct backtest of the LIVE `calculate_championship_equity_score()` blend "
            "against real historical outcomes, 1999-2025. See the script's module "
            "docstring for exactly which inputs are faithful to production vs. "
            "disclosed proxies (ceiling_score and sportsbook_advantage_score both "
            "substitute proxies -- no historical role_usage_td_score/volatility_"
            "diagnostic/sportsbook data exists).\n\n"
        )
        f.write("## Championship Equity Score (final blend) vs. each sub-component\n\n")
        blend_summary = agg[agg["model_name"] == "championship_equity_score"].groupby("model_name").agg(
            mean_auc=("mean_auc", "mean"),
            targets_tested=("target", "nunique"),
        ).reset_index()
        f.write(blend_summary.to_markdown(index=False) + "\n\n")
        f.write("## Full results by target and component\n\n")
        f.write(agg.to_markdown(index=False) + "\n")

    print(f"\nReport written to {REPORT_PATH.name}")
    print("\n=== SUMMARY: championship_equity_score (the actual blend) ===")
    blend = agg[agg["model_name"] == "championship_equity_score"]
    print(blend.to_string(index=False))
    print("\n=== By component (mean across all targets) ===")
    comp_summary = agg.groupby("model_name").agg(
        mean_auc=("mean_auc", "mean"),
        mean_lift=("mean_lift_over_baseline", "mean"),
        seasons_beat_adp_rate=("seasons_beat_adp", lambda s: (s / agg.loc[s.index, "seasons_tested"]).mean()),
    ).round(4).sort_values("mean_auc", ascending=False)
    print(comp_summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
