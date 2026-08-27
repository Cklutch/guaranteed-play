"""Breakout Score V1 -- frozen scorer (2026-08-27).

The concrete "model" implied by build_breakout_validation_v1.py's WR/FULL
result and build_breakout_robustness_checks_v1.py's stress tests, per the
five requirements laid out for turning a validated effect into a scoring
function:

1. FROZEN TARGET -- Beat_ADP_By_12 (finished the season 12+ overall spots
   ahead of preseason ADP). Same definition used throughout validation and
   robustness testing. Do not redefine "breakout" here or anywhere that
   consumes this scorer's output.

2. FEATURE SET -- WR FULL spec MINUS div_opportunity_minus_efficiency.
   That feature's coefficient sign was only 57% consistent across the 7
   walk-forward folds (vs. 100% for the other six) in the original
   robustness check -- closer to noise than signal at this sample size.
   build_breakout_trimmed_feature_check_v1.py re-ran the full battery with
   it dropped and every metric improved: top-decile delta +14.3% (was
   +11.9%), permutation p=0.0042 (was p=0.0188 -- this one clears a
   Bonferroni family-wise threshold of 0.0083 across the 6 originally
   tested position x spec combinations; the original 7-feature version did
   not), and the effect now excludes zero at every tested threshold
   (5/10/15/20%), not just three of four. This is the trimmed 6-feature
   set: overall_adp, prior_snap_share, prior_targets_per_route_run,
   prior_garbage_time_share, prior_route_participation_rate,
   prior_durability_score, age. CORE and CAPITAL specs did not clear the
   bar for WR and are not folded in here.

3. MODEL FAMILY -- logistic regression, not a boosted tree. With 7 usable
   seasons (route-participation data starts 2017) a shallow linear model
   generalizes better than a many-parameter GBM chasing a small sample;
   this was also the model class the validated AUC/top-decile/permutation
   results above were actually measured on -- switching families now would
   silently invalidate every check already run.

   NOT `class_weight="balanced"`, unlike the walk-forward validation
   pipeline in build_breakout_validation_v1.py. That flag was checked here
   (2026-08-27) and found to badly miscalibrate predict_proba (mean
   predicted probability 42.5% vs. actual observed rate 16.1%, Brier
   0.211, calibration slope 0.63) while buying no discrimination benefit
   over an unweighted fit (AUC 0.737 vs. 0.738 walk-forward, comparable
   top-decile hit rates). Dropping it brings mean predicted probability to
   15.4% (actual 16.1%), Brier to 0.127, slope to 0.75 -- this scorer uses
   the unweighted fit so its probability outputs are actually usable as
   probabilities, not just a ranking signal.

4. EVALUATION PROTOCOL STAYS SEASON-BLOCKED, PERMANENTLY -- every time a
   new season of data is added, rerun build_breakout_validation_v1.py and
   build_breakout_robustness_checks_v1.py (both still use the balanced
   pipeline for direct comparability with the frozen historical result;
   this file's unweighted variant is scoring-only, not a replacement
   validation protocol). Do not treat a single historical pass as
   permanent license to skip re-checking.

5. CALIBRATION -- see point 3 above and calibration_report_final() below,
   which reports the SAME walk-forward, pooled, out-of-sample calibration
   check build_breakout_validation_v1.calibration_report() already uses,
   just against the unweighted model instead of the balanced one.

STATUS: RESEARCH_ONLY, matching research/MODEL_REGISTRY.md's default. This
file provides score_players() as a usable scoring function, but nothing
in this repo calls it yet -- it is not wired into Home.py, draft_analysis.py,
or any UI path. Wiring it into a live recommendation surface is a separate
decision requiring its own explicit sign-off, not an automatic next step
just because a scorer function now exists.

Run: python research/validation_v1/breakout_score_v1_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALIDATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VALIDATION_DIR))

from build_breakout_validation_v1 import ADP_CONTINUOUS, FEATURES_FULL, folds, load  # noqa: E402

POSITION = "WR"
START_SEASON = 2017
DROPPED_FEATURE = "div_opportunity_minus_efficiency"
FEATURES = ADP_CONTINUOUS + [f for f in FEATURES_FULL if f != DROPPED_FEATURE]
TARGET_DEFINITION = "Beat_ADP_By_12"
SEED = 17


def make_scoring_pipe(seed=SEED) -> Pipeline:
    """Unweighted logistic regression -- see module docstring point 3 for why
    this differs from build_breakout_validation_v1.make_pipe()'s
    class_weight="balanced"."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=seed)),
    ])


def calibration_report_final(d: pd.DataFrame) -> dict:
    """Walk-forward, pooled, out-of-sample calibration of the UNWEIGHTED
    model -- same method as build_breakout_validation_v1.calibration_report,
    run here against the scoring pipeline actually used in fit_final_model()."""
    preds, actuals, aucs = [], [], []
    for ts, tr, te in folds(d, START_SEASON):
        est = make_scoring_pipe().fit(tr[FEATURES], tr["is_breakout"])
        p = est.predict_proba(te[FEATURES])[:, 1]
        y = te["is_breakout"].to_numpy()
        preds.extend(p)
        actuals.extend(y)
        aucs.append(roc_auc_score(y, p))
    preds, actuals = np.array(preds), np.array(actuals)
    cl = np.clip(preds, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(np.log(cl / (1 - cl)).reshape(-1, 1), actuals)
    return {
        "n": int(len(preds)),
        "mean_walkforward_auc": round(float(np.mean(aucs)), 4),
        "brier": round(float(brier_score_loss(actuals, preds)), 4),
        "mean_pred": round(float(preds.mean()), 4),
        "actual_rate": round(float(actuals.mean()), 4),
        "calibration_slope": round(float(lr.coef_[0][0]), 4),
        "calibration_intercept": round(float(lr.intercept_[0]), 4),
    }


def fit_final_model(d: pd.DataFrame) -> Pipeline:
    """Fit on ALL available WR/FULL rows (2017+) -- this is the model
    score_players() actually uses. Not itself walk-forward evaluated
    (nothing held out); calibration_report_final() above is what stands in
    for "does this model's probability output mean what it says," using
    walk-forward folds instead."""
    d = d[d["season"] >= START_SEASON]
    return make_scoring_pipe().fit(d[FEATURES], d["is_breakout"])


def score_players(model: Pipeline, players_df: pd.DataFrame) -> pd.DataFrame:
    """players_df must contain FEATURES (overall_adp + the 7 FULL columns).
    Returns players_df with breakout_probability_v1 (raw calibrated
    probability) and breakout_decile_v1 (1=top 10% by that probability,
    ties broken arbitrarily) appended."""
    missing = [c for c in FEATURES if c not in players_df.columns]
    if missing:
        raise ValueError(f"players_df missing required columns: {missing}")
    out = players_df.copy()
    out["breakout_probability_v1"] = model.predict_proba(out[FEATURES])[:, 1]
    out["breakout_decile_v1"] = pd.qcut(
        out["breakout_probability_v1"].rank(method="first", ascending=False),
        10, labels=False,
    ) + 1
    return out


def main() -> int:
    d = load(POSITION)
    cal = calibration_report_final(d)
    print(f"Walk-forward calibration ({POSITION}/FULL, unweighted logistic regression):")
    for k, v in cal.items():
        print(f"  {k}: {v}")

    model = fit_final_model(d)
    print(f"\nFinal model fit on {int((d['season'] >= START_SEASON).sum())} rows "
          f"({START_SEASON}-{int(d['season'].max())}).")
    coefs = model.named_steps["model"].coef_[0]
    print("Final coefficients (standardized scale):")
    for feat, c in zip(FEATURES, coefs):
        print(f"  {feat}: {c:+.4f}")

    report_path = VALIDATION_DIR / "breakout_v1_scorer_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Breakout Score V1 -- Frozen Scorer\n\n")
        f.write(f"Target: `{TARGET_DEFINITION}`. Position: {POSITION}. Spec: FULL "
                f"({START_SEASON}+). Model: unweighted logistic regression "
                "(see module docstring point 3 for why this differs from the "
                "class_weight=\"balanced\" pipeline used in validation).\n\n")
        f.write("## Walk-forward calibration (out-of-sample, pooled across folds)\n\n")
        f.write(pd.DataFrame([cal]).to_markdown(index=False) + "\n\n")
        f.write("## Final model (fit on all available rows, used by score_players())\n\n")
        f.write(f"Fit on {int((d['season'] >= START_SEASON).sum())} rows, "
                f"{START_SEASON}-{int(d['season'].max())}.\n\n")
        f.write(pd.DataFrame({"feature": FEATURES, "coef_standardized": coefs}).to_markdown(index=False) + "\n\n")
        f.write(
            "**Status: RESEARCH_ONLY.** `score_players()` is a usable function but nothing in "
            "this repo calls it yet. See research/MODEL_REGISTRY.md.\n"
        )
    print(f"\nReport: {report_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
