"""Breakout Score V1 Validation (2026-08-27).

Nested comparison for BREAKOUT, mirroring wr_bust_final_validation.py's
Task 1 methodology exactly (same pipeline, same guard contracts, same
season-blocked bootstrap CIs) so the result is directly comparable to the
already-completed, real WR bust-risk finding -- see research/MODEL_REGISTRY.md
and research/validation_v1/WR_BUST_DECISION_MEMO.md for that precedent and
its caveats, which apply here with equal force.

    baseline:  P(breakout) = f(continuous ADP)
    expanded:  P(breakout) = f(continuous ADP, features)

Continuous ADP retained in BOTH arms -- this tests INCREMENTAL value beyond
the market, not whether features alone beat ADP (a different, easier-to-lose
question the bust study's v1 retraction already demonstrates the danger of
conflating with this one).

TARGET: `Beat_ADP_By_12` (already a real column in predraft_validation_
dataset_archetypes_v1.csv, used by market_disagreement_score_v1 too) --
finished the season having beaten the player's own draft slot by 12+
overall spots. The most literal definition of "breakout" available without
inventing a new threshold.

UNIVERSE: overall_adp <= 150, same bound the bust study and market_
disagreement study both use. Tested per-position (WR, RB -- the two with
comparable sample size to the bust study's own 925-row WR universe) rather
than pooled, so results are directly interpretable against precedent rather
than mixing position-specific base rates and feature availability.

FEATURES: same CORE/FULL split as the bust study, for the same reason
(route-participation features only exist from 2017+, so FULL trades 3
seasons of power for 2 extra features) -- not because breakout and bust
should share a feature set on first principles, but because this is the
established, validated split for this exact dataset and changing it without
cause would just be a fresh, untested guess:
    CORE (2014+): prior_snap_share, prior_garbage_time_share,
                  div_opportunity_minus_efficiency, prior_durability_score, age
    FULL (2017+): CORE + prior_route_participation_rate, prior_targets_per_route_run

EXTENSION BEYOND THE BUST STUDY'S METHODOLOGY (added 2026-08-27, second
pass): after the first pass found weak-to-negative whole-population AUC
across all three specs, tested whether the model's own most-CONFIDENT
calls are still real even when the aggregate ranking isn't -- a
whole-population AUC can be null while a model correctly identifies a
specific high-confidence subset. Top ~10% by predicted probability,
computed WITHIN each test season (not pooled), season-blocked bootstrapped
the same way as AUC/Brier/logloss. This is now part of the confirmatory
pipeline, not a separate exploratory afterthought. A second, genuinely
exploratory check (one pre-specified experience-stage subgroup split) is
also included and is NOT bootstrapped -- see experience_stage_split()'s
docstring for why that one stays exploratory.

NOT INCLUDED: the bust study's Task 2 (2x2 fade/replacement policy
decomposition). That's a materially bigger, separate undertaking, and the
bust study's own finding there was "infeasible under the matched design" --
worth attempting only if the results above show something worth a policy
test in the first place.

Run: python research/validation_v1/build_breakout_validation_v1.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALIDATION_DIR = Path(__file__).resolve().parent

SEED = 17
N_SEASON_BOOTSTRAP = 5000
ADP_MAX = 150
TARGET = "Beat_ADP_By_12"

DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTDIR = VALIDATION_DIR / "data" / "breakout_v1"

ADP_CONTINUOUS = ["overall_adp"]
FEATURES_FULL = [
    "prior_snap_share", "prior_targets_per_route_run",
    "div_opportunity_minus_efficiency", "prior_garbage_time_share",
    "prior_route_participation_rate", "prior_durability_score", "age",
]
FEATURES_CORE = [
    "prior_snap_share", "div_opportunity_minus_efficiency",
    "prior_garbage_time_share", "prior_durability_score", "age",
]
# Draft capital / development-stage angle (added 2026-08-27, after CORE/FULL
# above showed weak-to-negative signal): a genuinely different feature
# category from prior-season USAGE -- this is about talent investment and
# career stage, which a rookie/sophomore breakout candidate has real values
# for even with zero fantasy-relevant usage yet. Real, well-covered fields
# already native to this dataset (no merge needed): draft_pick_overall/
# years_since_drafted at 84-94% coverage EVERY season 2010-2025 (no
# staggered-coverage problem, unlike the usage features), estimated_
# draft_round at 100%.
FEATURES_CAPITAL = ["estimated_draft_round", "draft_pick_overall", "years_since_drafted", "age"]
SPECS = {
    "FULL": (FEATURES_FULL, 2017),
    "CORE": (FEATURES_CORE, 2014),
    "CAPITAL": (FEATURES_CAPITAL, 2010),
}
POSITIONS = ["WR", "RB"]  # comparable sample size to the bust study's own 925-row WR universe


class FoldSchemaError(RuntimeError):
    pass


def load(position: str) -> pd.DataFrame:
    d = pd.read_csv(DATASET, low_memory=False)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d = d[d["overall_adp"].notna() & d[TARGET].notna()].copy()
    d = d[d["position"] == position].copy()
    d["is_breakout"] = pd.to_numeric(d[TARGET], errors="coerce").astype(int)
    return d[d["overall_adp"] <= ADP_MAX].copy()


def make_pipe(seed=SEED):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def folds(d, start, min_train=120, min_test=15):
    d = d[d["season"] >= start]
    for ts in sorted(d["season"].dropna().astype(int).unique()):
        tr, te = d[d["season"] < ts], d[d["season"] == ts]
        if len(tr) < min_train or len(te) < min_test or te["is_breakout"].nunique() < 2:
            continue
        yield ts, tr, te


def assert_fold_contract(tr, feats, ts, schema_ref):
    for f in feats:
        if tr[f].notna().sum() == 0:
            raise FoldSchemaError(f"season {ts}: '{f}' has no non-missing training value")
    imp = SimpleImputer(strategy="median").fit(tr[feats])
    if np.isnan(imp.statistics_).any():
        bad = [f for f, s in zip(feats, imp.statistics_) if np.isnan(s)]
        raise FoldSchemaError(f"season {ts}: imputer would drop {bad}")
    schema = tuple(imp.get_feature_names_out())
    if schema_ref and schema != schema_ref[0]:
        raise FoldSchemaError(f"season {ts}: transformed schema differs from earlier folds")
    if not schema_ref:
        schema_ref.append(schema)


def assert_no_train_test_overlap(tr, te, key=("season", "player_name")):
    kt = set(map(tuple, tr[list(key)].to_numpy()))
    ke = set(map(tuple, te[list(key)].to_numpy()))
    both = kt & ke
    if both:
        raise FoldSchemaError(f"train/test overlap on {len(both)} player-seasons")


def season_bootstrap(per_season_values, rng, n=N_SEASON_BOOTSTRAP):
    v = np.asarray(per_season_values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    draws = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def nested_comparison(d, feats, start, label, position, rng):
    schema_ref, rows, player_preds = [], [], []
    expanded_feats = ADP_CONTINUOUS + feats

    for ts, tr, te in folds(d, start):
        assert_no_train_test_overlap(tr, te)
        assert_fold_contract(tr, expanded_feats, ts, schema_ref)

        base = make_pipe().fit(tr[ADP_CONTINUOUS], tr["is_breakout"])
        exp = make_pipe().fit(tr[expanded_feats], tr["is_breakout"])
        pb = base.predict_proba(te[ADP_CONTINUOUS])[:, 1]
        pe = exp.predict_proba(te[expanded_feats])[:, 1]
        y = te["is_breakout"].to_numpy()

        # Per-season top-decile hit rate for BOTH arms, computed the same
        # way each season (not pooled) so it can go through the identical
        # season-blocked bootstrap as AUC/Brier/logloss below -- promotes
        # the exploratory pooled top-decile check from the previous pass
        # into a properly uncertainty-quantified metric instead of a bare
        # point estimate. top_n ~= round(10% of that season's test size);
        # min 3 so a single-player season can't produce a 0%-or-100% rate.
        top_n = max(3, int(round(len(te) * 0.10)))
        top_idx_base = np.argsort(-pb)[:top_n]
        top_idx_exp = np.argsort(-pe)[:top_n]
        topdecile_base = float(y[top_idx_base].mean())
        topdecile_exp = float(y[top_idx_exp].mean())

        rows.append({
            "season": ts, "n": len(te), "breakout_rate": round(float(y.mean()), 3),
            "auc_base": roc_auc_score(y, pb), "auc_exp": roc_auc_score(y, pe),
            "brier_base": brier_score_loss(y, pb), "brier_exp": brier_score_loss(y, pe),
            "logloss_base": log_loss(y, pb, labels=[0, 1]),
            "logloss_exp": log_loss(y, pe, labels=[0, 1]),
            "topdecile_n": top_n,
            "topdecile_base": topdecile_base, "topdecile_exp": topdecile_exp,
        })

        # Out-of-sample per-player predictions, kept for two follow-up
        # checks the season-level AUC/Brier/logloss summary above can't
        # answer on its own: (1) does the model's own TOP-CONFIDENCE calls
        # actually hit more than baseline, regardless of whole-population
        # AUC; (2) does a pre-specified subgroup (experience stage) show a
        # real effect the pooled average washes out. Both exploratory,
        # reported separately from the confirmatory nested comparison above.
        fold_preds = pd.DataFrame({
            "season": ts, "player_name": te["player_name"].to_numpy(),
            "is_breakout": y, "pred_base": pb, "pred_exp": pe,
        })
        if "years_since_drafted" in te.columns:
            fold_preds["years_since_drafted"] = te["years_since_drafted"].to_numpy()
        player_preds.append(fold_preds)

    if not rows:
        return None, position, label, pd.DataFrame()

    player_preds_df = pd.concat(player_preds, ignore_index=True) if player_preds else pd.DataFrame()

    per = pd.DataFrame(rows)
    per["auc_delta"] = per.auc_exp - per.auc_base
    per["brier_delta"] = per.brier_exp - per.brier_base
    per["logloss_delta"] = per.logloss_exp - per.logloss_base
    per["topdecile_delta"] = per.topdecile_exp - per.topdecile_base

    summary = {"position": position, "spec": label, "start_season": start,
               "test_seasons": len(per), "features": expanded_feats}
    for m, better_is in (("auc_delta", "higher"), ("brier_delta", "lower"), ("logloss_delta", "lower"),
                        ("topdecile_delta", "higher")):
        lo, hi = season_bootstrap(per[m], rng)
        wins = float((per[m] > 0).mean()) if better_is == "higher" else float((per[m] < 0).mean())
        summary[m] = {"mean": round(float(per[m].mean()), 4), "ci_lo": round(lo, 4),
                      "ci_hi": round(hi, 4), "seasons_improved": round(wins, 3),
                      "excludes_zero": bool(lo > 0 or hi < 0)}
    summary["auc_base_pooled"] = round(float(per.auc_base.mean()), 4)
    summary["auc_exp_pooled"] = round(float(per.auc_exp.mean()), 4)
    summary["topdecile_base_pooled"] = round(float(per.topdecile_base.mean()), 4)
    summary["topdecile_exp_pooled"] = round(float(per.topdecile_exp.mean()), 4)
    return {"per_season": per, "summary": summary}, position, label, player_preds_df


def experience_stage_split(player_preds: pd.DataFrame, score_col: str, rng) -> pd.DataFrame:
    """EXPLORATORY, not confirmatory -- a single pre-specified subgroup split
    (years_since_drafted <= 2 = "developing" vs. > 2 = "established"), the
    classic breakout-happens-in-year-2-3 theory, chosen before looking at
    results and tested only once here (not scanned across many cut points,
    which would just be noise-fishing dressed up as a finding). Compares AUC
    within EACH subgroup on the same pooled out-of-sample predictions the
    top-decile check uses -- not season-blocked (too few rows per season
    per subgroup to bootstrap meaningfully), so no CI is reported here;
    treat this as hypothesis-generating, matching how this repo's other
    studies (WR bust memo, section 12) explicitly separate "what was
    confirmatorily tested" from "what's recorded for future research only.\""""
    if "years_since_drafted" not in player_preds.columns:
        return pd.DataFrame()
    rows = []
    for stage_label, mask in [
        ("developing (<=2 yrs)", player_preds["years_since_drafted"] <= 2),
        ("established (>2 yrs)", player_preds["years_since_drafted"] > 2),
    ]:
        sub = player_preds[mask & player_preds["years_since_drafted"].notna()]
        if len(sub) < 20 or sub["is_breakout"].nunique() < 2:
            rows.append({"stage": stage_label, "n": len(sub), "status": "insufficient_rows"})
            continue
        y = sub["is_breakout"].to_numpy()
        auc_base = roc_auc_score(y, sub["pred_base"])
        auc_exp = roc_auc_score(y, sub[score_col])
        rows.append({
            "stage": stage_label, "n": len(sub), "breakout_rate": round(float(y.mean()), 3),
            "auc_base": round(float(auc_base), 4), "auc_exp": round(float(auc_exp), 4),
            "auc_delta": round(float(auc_exp - auc_base), 4), "status": "evaluated",
        })
    return pd.DataFrame(rows)


def calibration_report(d, feats, start):
    expanded = ADP_CONTINUOUS + feats
    preds, actuals = [], []
    for ts, tr, te in folds(d, start):
        est = make_pipe().fit(tr[expanded], tr["is_breakout"])
        preds.extend(est.predict_proba(te[expanded])[:, 1])
        actuals.extend(te["is_breakout"])
    preds, actuals = np.array(preds), np.array(actuals)
    if len(preds) == 0:
        return None
    cl = np.clip(preds, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(np.log(cl / (1 - cl)).reshape(-1, 1), actuals)
    return {"n": int(len(preds)), "brier": round(float(brier_score_loss(actuals, preds)), 4),
            "intercept": round(float(lr.intercept_[0]), 4),
            "slope": round(float(lr.coef_[0][0]), 4),
            "mean_pred": round(float(preds.mean()), 4),
            "actual_rate": round(float(actuals.mean()), 4)}


def classify(summary: dict) -> str:
    auc = summary["auc_delta"]
    if auc["excludes_zero"] and auc["mean"] > 0:
        return "Real Signal (CORE-style: CI excludes zero)"
    if auc["mean"] > 0 and auc["seasons_improved"] >= 0.6:
        return "Weak Research Signal"
    return "No Signal"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_summaries = []
    calibrations = []
    all_top_decile = []
    all_experience_splits = []

    for position in POSITIONS:
        d = load(position)
        print(f"\n{'=' * 74}\n{position}: {len(d)} rows, seasons {int(d['season'].min())}-{int(d['season'].max())}, "
              f"breakout rate {round(d['is_breakout'].mean() * 100, 1)}%\n{'=' * 74}")
        for label, (feats, start) in SPECS.items():
            try:
                result, pos, lbl, player_preds = nested_comparison(d, feats, start, label, position, rng)
            except FoldSchemaError as exc:
                print(f"  {label}: SKIPPED -- {exc}")
                continue
            if result is None:
                print(f"  {label}: no valid folds")
                continue
            summary = result["summary"]
            summary["final_classification"] = classify(summary)
            all_summaries.append(summary)
            cal = calibration_report(d, feats, start)
            if cal:
                cal["position"] = position
                cal["spec"] = label
                calibrations.append(cal)
            print(f"\n  --- {position} / {label} ({summary['test_seasons']} seasons, "
                  f"since {start}) ---")
            print(f"  AUC:     base={summary['auc_base_pooled']:.4f}  exp={summary['auc_exp_pooled']:.4f}  "
                  f"delta={summary['auc_delta']['mean']:+.4f} "
                  f"[{summary['auc_delta']['ci_lo']:+.4f},{summary['auc_delta']['ci_hi']:+.4f}]  "
                  f"({summary['auc_delta']['seasons_improved']:.0%} seasons improved)")
            print(f"  Brier delta: {summary['brier_delta']['mean']:+.4f} "
                  f"[{summary['brier_delta']['ci_lo']:+.4f},{summary['brier_delta']['ci_hi']:+.4f}]")
            print(f"  LogLoss delta: {summary['logloss_delta']['mean']:+.4f} "
                  f"[{summary['logloss_delta']['ci_lo']:+.4f},{summary['logloss_delta']['ci_hi']:+.4f}]")
            print(f"  Classification: {summary['final_classification']}")
            per_path = OUTDIR / f"per_season_{position}_{label}.csv"
            result["per_season"].to_csv(per_path, index=False)

            # --- top-decile hit rate: NOW promoted to the same season-blocked
            # bootstrap as AUC/Brier/logloss above, per-season (not pooled) --
            # this is the confirmatory-grade version of last pass's exploratory
            # pooled point estimate.
            td = summary["topdecile_delta"]
            print(f"  Top-decile hit rate: base={summary['topdecile_base_pooled']:.1%} "
                  f"exp={summary['topdecile_exp_pooled']:.1%}  delta={td['mean']:+.1%} "
                  f"[{td['ci_lo']:+.1%},{td['ci_hi']:+.1%}]  "
                  f"({td['seasons_improved']:.0%} seasons improved)"
                  f"{'  ** excludes zero **' if td['excludes_zero'] else ''}")
            all_top_decile.append({
                "position": position, "spec": label,
                "topdecile_base": summary["topdecile_base_pooled"],
                "topdecile_exp": summary["topdecile_exp_pooled"],
                "topdecile_delta": td["mean"],
                "topdecile_ci": f"[{td['ci_lo']:+.4f},{td['ci_hi']:+.4f}]",
                "seasons_improved": td["seasons_improved"],
                "excludes_zero": td["excludes_zero"],
            })

            exp_split = experience_stage_split(player_preds, "pred_exp", rng)
            if not exp_split.empty:
                exp_split.insert(0, "spec", label)
                exp_split.insert(0, "position", position)
                all_experience_splits.append(exp_split)
                for _, r in exp_split.iterrows():
                    if r["status"] == "evaluated":
                        print(f"  Experience split [{r['stage']}]: n={r['n']}, "
                              f"AUC base={r['auc_base']:.4f} exp={r['auc_exp']:.4f} "
                              f"delta={r['auc_delta']:+.4f}")

    with open(OUTDIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target": TARGET, "adp_max": ADP_MAX, "summaries": all_summaries,
            "calibrations": calibrations,
        }, f, indent=2, default=str)

    report_path = VALIDATION_DIR / "breakout_v1_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Breakout Score V1 Validation\n\n")
        f.write(
            f"Nested comparison, `P({TARGET}) = f(ADP)` vs. `f(ADP, features)`, "
            "mirroring wr_bust_final_validation.py's Task 1 methodology exactly. "
            "See this script's module docstring for the full spec and what is/isn't "
            "covered (no policy-decomposition test, unlike the bust study).\n\n"
        )
        f.write("## Results (nested comparison, AUC/Brier/log-loss deltas over continuous-ADP baseline)\n\n")
        rows = []
        for s in all_summaries:
            rows.append({
                "position": s["position"], "spec": s["spec"], "test_seasons": s["test_seasons"],
                "auc_base": s["auc_base_pooled"], "auc_exp": s["auc_exp_pooled"],
                "auc_delta": s["auc_delta"]["mean"],
                "auc_ci": f"[{s['auc_delta']['ci_lo']:+.4f},{s['auc_delta']['ci_hi']:+.4f}]",
                "seasons_improved": s["auc_delta"]["seasons_improved"],
                "brier_delta": s["brier_delta"]["mean"],
                "logloss_delta": s["logloss_delta"]["mean"],
                "classification": s["final_classification"],
            })
        f.write(pd.DataFrame(rows).to_markdown(index=False) + "\n\n")
        f.write("## Calibration (expanded model, pooled out-of-sample predictions)\n\n")
        f.write(pd.DataFrame(calibrations).to_markdown(index=False) + "\n\n")

        f.write("## Top-decile hit rate: does the model's own most-confident calls beat baseline?\n\n")
        f.write(
            "A whole-population AUC can be null or reliably negative while the model's own "
            "highest-confidence calls are still real -- this asks that question directly instead "
            "of inferring it from AUC. Same rigor as the confirmatory nested comparison above, "
            "not a pooled point estimate: top ~10% by predicted probability computed WITHIN each "
            "test season (min 3 players), then season-blocked bootstrapped exactly like the "
            "AUC/Brier/logloss deltas. `excludes_zero=True` rows are the real finding.\n\n"
        )
        if all_top_decile:
            f.write(pd.DataFrame(all_top_decile).to_markdown(index=False) + "\n\n")
        else:
            f.write("No data.\n\n")

        f.write("## Exploratory: experience-stage subgroup split\n\n")
        f.write(
            "EXPLORATORY, NOT CONFIRMATORY. One pre-specified split -- years_since_drafted <= 2 "
            "(\"developing\") vs. > 2 (\"established\"), the standard breakout-happens-in-year-2-3 "
            "theory -- tested once, not scanned across cut points. No season-blocked bootstrap CI "
            "(too few rows per season per subgroup for a stable estimate), so treat this as "
            "hypothesis-generating only, same standard this repo's other studies hold themselves to "
            "for anything not run through the full confirmatory pipeline.\n\n"
        )
        if all_experience_splits:
            f.write(pd.concat(all_experience_splits, ignore_index=True).to_markdown(index=False) + "\n")
        else:
            f.write("No data (years_since_drafted not available for this spec).\n")

    print(f"\n\nReport: {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
