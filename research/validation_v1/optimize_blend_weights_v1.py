"""
Optimal ADP-anchored blend weighting.

Every prior test let a model choose its own feature weights freely, with
ADP as just one input among many. The fitted weights show why that fails:
at QB only 13.8% of total weight mass landed on ADP features (see
data/stacked_model_feature_weights.csv), so the model actively down-weighted
the single signal that beats the field (+0.032 lift) in favor of draft
capital and prior production. Free weighting wandered off the anchor.

This tests the constrained alternative instead:

    final_percentile = (1 - lam) * adp_percentile + lam * model_percentile

where the model is trained on NON-ADP features only, so `lam` is literally
"how far do we deviate from consensus." lam=0 is pure ADP; lam=1 is pure
model. Sweeping lam directly answers "what is the optimal weighting of
these signals together," in the one form the regression tests could not
express -- a structurally guaranteed ADP anchor.

Two numbers are reported per position:

  walk_forward  -- lam chosen using only seasons BEFORE the test season,
                   then applied to the test season. This is the honest,
                   deployable number.
  oracle        -- lam chosen WITH hindsight, per test season, to maximize
                   that season's own lift. Not achievable in practice; it
                   is the ceiling. If even the oracle can't beat ADP, then
                   no blend weighting exists that would have helped, and
                   the question is closed by evidence rather than by
                   assumption.

Scope note: evaluation is restricted to test rows that HAVE an ADP, since
the ADP baseline cannot rank rows without one. The baseline hit rate is
recomputed on that same restricted population, so lift numbers here are
internally consistent but not directly comparable to the row counts in
workload_v1_step4_summary.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import (
    VALIDATION_DIR,
    auc_or_nan,
    fit_predict,
    hit_rate,
    top_decile_threshold_count,
)
from workload_feature_groups import (
    DIVERGENCE,
    DURABILITY,
    POSITION_ARCHETYPE_FEATURES,
    POSITION_WORKLOAD_FEATURES,
)

DATASET_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "blend_weight_results.csv"

PRIMARY_TARGET = {"QB": "QB_Top12", "RB": "RB_Top24", "WR": "WR_Top24", "TE": "TE_Top12"}

PRIOR_PRODUCTION = {
    "QB": ["prior_passing_yards", "prior_passing_tds", "prior_rushing_yards", "prior_rushing_tds",
           "prior_total_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games"],
    "RB": ["prior_carries", "prior_rushing_yards", "prior_rushing_tds", "prior_receiving_yards",
           "prior_receiving_tds", "prior_total_tds", "prior_fantasy_points", "prior_fantasy_ppg", "prior_games"],
    "WR": ["prior_targets", "prior_receptions", "prior_receiving_yards", "prior_receiving_tds",
           "prior_fantasy_points", "prior_fantasy_ppg", "prior_games", "prior_target_share"],
    "TE": ["prior_receiving_yards", "prior_receiving_tds", "prior_total_tds",
           "prior_fantasy_points", "prior_fantasy_ppg", "prior_games"],
}

LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

# Estimator for the non-ADP signal model. lasso_logistic because it at
# least drops dead features rather than carrying them at small weight.
MODEL_KIND = "lasso_logistic"


def _non_adp_features(position: str) -> list[str]:
    """Everything the project has built, minus ADP itself."""
    feats = (
        PRIOR_PRODUCTION[position]
        + POSITION_WORKLOAD_FEATURES[position]
        + DIVERGENCE
        + POSITION_ARCHETYPE_FEATURES[position]
        + DURABILITY
    )
    seen: dict[str, None] = {}
    for f in feats:
        seen.setdefault(f, None)
    return list(seen)


def _adp_series(frame: pd.DataFrame) -> pd.Series:
    adp = pd.to_numeric(frame.get("overall_adp"), errors="coerce")
    fallback = pd.to_numeric(frame.get("preseason_adp"), errors="coerce")
    return adp.fillna(fallback)


def _lift_for_lambda(test: pd.DataFrame, target: str, adp_pct: pd.Series,
                     model_pct: pd.Series, lam: float, baseline_hit: float) -> float:
    blended = (1.0 - lam) * adp_pct + lam * model_pct
    scored = test.copy()
    scored["_blend"] = blended
    top_n = top_decile_threshold_count(len(scored))
    if not top_n:
        return float("nan")
    top = scored[scored["_blend"].notna()].sort_values("_blend", ascending=False).head(top_n)
    model_hit = hit_rate(top, target)
    if pd.isna(model_hit) or pd.isna(baseline_hit):
        return float("nan")
    return float(model_hit - baseline_hit)


def evaluate_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    target = PRIMARY_TARGET[position]
    pos_df = df[df["position"].eq(position)].copy()
    pos_df["season"] = pd.to_numeric(pos_df["season"], errors="coerce")
    if target not in pos_df.columns:
        return pd.DataFrame()

    features = [f for f in _non_adp_features(position) if f in pos_df.columns]
    seasons = sorted(pos_df["season"].dropna().astype(int).unique().tolist())
    rows: list[dict[str, object]] = []

    for test_season in seasons:
        train = pos_df[pos_df["season"] < test_season].copy()
        test = pos_df[pos_df["season"] == test_season].copy()
        if train.empty or test.empty:
            continue

        # Restrict to rows the ADP baseline can actually rank.
        test_adp = _adp_series(test)
        test = test[test_adp.notna()].copy()
        if len(test) < 10:
            continue
        test_adp = _adp_series(test)

        # Lower ADP = better, so negate before ranking to percentile.
        adp_pct = (-test_adp).rank(pct=True)

        baseline_top_n = top_decile_threshold_count(len(test))
        baseline_top = test.assign(_adp=adp_pct).sort_values("_adp", ascending=False).head(baseline_top_n)
        baseline_hit = hit_rate(baseline_top, target)
        if pd.isna(baseline_hit):
            continue
        baseline_auc = auc_or_nan(test[target], adp_pct)

        scores, status = fit_predict(train, test, target, features, MODEL_KIND)
        if scores.notna().sum() == 0:
            continue
        model_pct = scores.rank(pct=True)

        for lam in LAMBDA_GRID:
            lam_f = float(lam)
            # AUC alongside top-decile lift, because top-decile hit rate is
            # too granular to measure this at these sample sizes: QB/TE test
            # sets run 13-29 players, so the top decile is 2-3 players and a
            # single swap registers as +/-0.333. Measured directly -- 12 to 14
            # of 16 seasons come back exactly 0.000, i.e. the metric usually
            # cannot see the re-ranking at all. AUC uses every pair, so it
            # can detect a small consistent improvement that the top-decile
            # cut rounds away.
            blended = (1.0 - lam_f) * adp_pct + lam_f * model_pct
            rows.append({
                "position": position,
                "test_season": test_season,
                "lam": lam_f,
                "lift_over_adp": _lift_for_lambda(test, target, adp_pct, model_pct, lam_f, baseline_hit),
                "auc": auc_or_nan(test[target], blended),
                "adp_auc": baseline_auc,
                "auc_gain": auc_or_nan(test[target], blended) - baseline_auc,
                "baseline_hit_rate": baseline_hit,
                "n_test": len(test),
                "status": status,
            })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward lambda selection vs. the hindsight oracle ceiling."""
    out_rows: list[dict[str, object]] = []

    for position, pos_res in results.groupby("position"):
        seasons = sorted(pos_res["test_season"].unique().tolist())
        wf_lifts: list[float] = []
        oracle_lifts: list[float] = []
        chosen_lams: list[float] = []

        for test_season in seasons:
            this_season = pos_res[pos_res["test_season"] == test_season]
            prior = pos_res[pos_res["test_season"] < test_season]

            # Oracle: best lambda for this season, chosen with hindsight.
            if this_season["lift_over_adp"].notna().any():
                oracle_lifts.append(float(this_season["lift_over_adp"].max()))

            # Walk-forward: pick lambda on prior seasons only.
            if prior.empty or prior["lift_over_adp"].notna().sum() == 0:
                continue
            mean_by_lam = prior.groupby("lam")["lift_over_adp"].mean()
            if mean_by_lam.notna().sum() == 0:
                continue
            best_lam = float(mean_by_lam.idxmax())
            realized = this_season[np.isclose(this_season["lam"], best_lam)]["lift_over_adp"]
            if realized.notna().any():
                wf_lifts.append(float(realized.iloc[0]))
                chosen_lams.append(best_lam)

        # Best FIXED lambda across all seasons, also hindsight-selected --
        # a middle ground between walk-forward and the per-season oracle.
        mean_by_lam_all = pos_res.groupby("lam")["lift_over_adp"].mean()
        best_fixed_lam = float(mean_by_lam_all.idxmax()) if mean_by_lam_all.notna().any() else float("nan")
        best_fixed_lift = float(mean_by_lam_all.max()) if mean_by_lam_all.notna().any() else float("nan")
        pure_adp_lift = float(mean_by_lam_all.get(0.0, float("nan")))

        out_rows.append({
            "position": position,
            "seasons": len(seasons),
            "pure_adp_lift": pure_adp_lift,
            "walk_forward_lift": float(np.mean(wf_lifts)) if wf_lifts else float("nan"),
            "walk_forward_win_rate": float(np.mean([l > 0 for l in wf_lifts])) if wf_lifts else float("nan"),
            "median_chosen_lam": float(np.median(chosen_lams)) if chosen_lams else float("nan"),
            "best_fixed_lam": best_fixed_lam,
            "best_fixed_lift": best_fixed_lift,
            "oracle_lift": float(np.mean(oracle_lifts)) if oracle_lifts else float("nan"),
        })

    return pd.DataFrame(out_rows)


def main() -> None:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    frames = [evaluate_position(df, pos) for pos in ["QB", "RB", "WR", "TE"]]
    frames = [f for f in frames if not f.empty]
    if not frames:
        print("No results produced.")
        return

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Per-season/per-lambda results written: {OUTPUT_PATH}")

    summary = summarize(results)
    print("\n=== ADP-ANCHORED BLEND: does any weighting beat pure ADP? ===")
    print("(pure_adp_lift is 0 by construction -- it IS the baseline.)")
    print(summary.to_string(index=False))

    print("\n=== Mean lift by lambda (0.0 = pure ADP, 1.0 = pure model) ===")
    pivot = results.pivot_table(index="lam", columns="position", values="lift_over_adp", aggfunc="mean")
    print(pivot.round(4).to_string())

    print("\n=== Mean AUC GAIN over ADP by lambda (sensitive metric) ===")
    auc_pivot = results.pivot_table(index="lam", columns="position", values="auc_gain", aggfunc="mean")
    print(auc_pivot.round(4).to_string())

    print("\n=== AUC-gain significance at each position's best lambda ===")
    rng = np.random.default_rng(0)
    for position, pos_res in results.groupby("position"):
        by_lam = pos_res.groupby("lam")["auc_gain"].mean()
        if by_lam.notna().sum() == 0:
            continue
        best_lam = float(by_lam.idxmax())
        vals = pos_res[np.isclose(pos_res["lam"], best_lam)]["auc_gain"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wins = float((vals > 0).mean())
        flag = "significant" if lo > 0 else "CI includes 0"
        print(f"{position}: best_lam={best_lam:.2f} mean_auc_gain={vals.mean():+.4f} "
              f"CI=[{lo:+.4f},{hi:+.4f}] wins {wins:.0%} of {len(vals)} seasons -> {flag}")

    print("\n=== VERDICT ===")
    for _, r in summary.iterrows():
        if pd.notna(r["walk_forward_lift"]) and r["walk_forward_lift"] > 0.01 and r["walk_forward_win_rate"] > 0.5:
            verdict = "BLEND HELPS"
        elif pd.notna(r["oracle_lift"]) and r["oracle_lift"] <= 0:
            verdict = "CLOSED -- even the hindsight oracle cannot beat ADP"
        else:
            verdict = "no deployable gain (oracle ceiling exists but walk-forward selection can't reach it)"
        print(f"{r['position']}: walk_forward={r['walk_forward_lift']:+.4f} "
              f"(wins {r['walk_forward_win_rate']:.0%}, median lam={r['median_chosen_lam']}) | "
              f"oracle_ceiling={r['oracle_lift']:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
