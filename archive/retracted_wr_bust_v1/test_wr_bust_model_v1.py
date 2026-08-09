"""
Stress-test the WR conditional-bust model before building anything on it.

Claim under test: among WRs drafted at similar ADP, can we identify who is
unusually likely to finish below replacement?

This is NOT a ranking model and is not evaluated as one. Prior top-N
objectives failed and are not revived here.

Tests, ordered by what could actually invalidate the finding:
  1. LEAKAGE     -- shuffle labels within the test season; real skill must
                    collapse to AUC ~0.50, or the harness leaks.
  2. PER-SEASON  -- pooled AUC can hide one or two carrying seasons.
  3. CALIBRATION -- are the probabilities usable, or only ranks?
  4. STABILITY   -- coefficient sign across folds. A sign that flips is
                    predictive at best, never interpretable as a driver.
  5. SUPPRESSION -- div_opportunity_minus_efficiency entered POSITIVE here
                    but its BOTTOM decile was the earlier fade signal.
  6. DECISION    -- the test that matters. Does avoiding flagged WRs improve
                    a draft? AUC cannot answer that.

IMPORTANT correction baked in: an earlier pooled result ("AUC 0.7498 vs
0.7224, 85% of 13 seasons") was inflated. Feature coverage by season is:
    2010-2013  essentially nothing but age
    2014-2016  snap share / divergence / durability, NO route data
    2017-2025  everything at 85-93%
sklearn's imputer silently drops all-NaN columns, so early folds quietly fit
a SMALLER model and the pooled number mixed different specifications. Both
specs below are therefore run separately, with a guard requiring every
feature to be observed in the training fold.

Everything (imputer, scaler, calibrator, thresholds) is fit strictly inside
the training fold. Bootstraps resample SEASONS, respecting clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validation_utils import VALIDATION_DIR

SEED = 17
DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 14}
ADP_MAX = 150  # draftable WR range; past this, bust is near-certain and uninformative

FEATURES_FULL = [
    "prior_snap_share", "prior_targets_per_route_run",
    "div_opportunity_minus_efficiency", "prior_garbage_time_share",
    "prior_route_participation_rate", "prior_durability_score", "age",
]
# Dropping the two route features buys three extra seasons of history.
FEATURES_CORE = [
    "prior_snap_share", "div_opportunity_minus_efficiency",
    "prior_garbage_time_share", "prior_durability_score", "age",
]
SPECS = {"FULL (2017+)": (FEATURES_FULL, 2017), "CORE (2014+, no route)": (FEATURES_CORE, 2014)}
ADP_CONTROL = ["overall_adp"]
FADE_THRESHOLDS = (0.10, 0.15, 0.20)  # pre-specified; all reported
SEP = "=" * 72


def load() -> pd.DataFrame:
    d = pd.read_csv(DATASET, low_memory=False)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d = d[d["final_fantasy_points"].notna() & d["overall_adp"].notna()].copy()
    baselines = {}
    for (season, position), g in d.groupby(["season", "position"]):
        if position not in REPLACEMENT_RANK:
            continue
        pts = g["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        n = REPLACEMENT_RANK[position]
        baselines[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])
    d = d[d["position"] == "WR"].copy()
    d["replacement"] = [baselines.get((s, "WR"), np.nan) for s in d["season"]]
    d["is_bust"] = (d["final_fantasy_points"] < d["replacement"]).astype(int)
    d["vor"] = d["final_fantasy_points"] - d["replacement"]
    return d[d["overall_adp"] <= ADP_MAX].copy()


def make_pipe(seed=SEED):
    # Imputer inside the pipeline so medians are learned per training fold.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def folds(d, feats, start_season, min_train=120, min_test=15):
    """Walk-forward, with every feature required to be observed in train."""
    d = d[d["season"] >= start_season]
    for ts in sorted(d["season"].dropna().astype(int).unique()):
        tr, te = d[d["season"] < ts], d[d["season"] == ts]
        if len(tr) < min_train or len(te) < min_test or te["is_bust"].nunique() < 2:
            continue
        if any(tr[f].notna().sum() == 0 for f in feats):
            continue
        yield ts, tr, te


def season_bootstrap(values, rng, n=20000):
    v = np.asarray(values, dtype=float)
    return np.percentile([rng.choice(v, len(v), replace=True).mean() for _ in range(n)], [2.5, 97.5])


def evaluate_spec(d, feats, start, rng, label):
    rows, shuf = [], []
    for ts, tr, te in folds(d, feats, start):
        m = make_pipe().fit(tr[feats], tr["is_bust"])
        a = make_pipe().fit(tr[ADP_CONTROL], tr["is_bust"])
        pm = m.predict_proba(te[feats])[:, 1]
        rows.append({"season": ts, "n": len(te),
                     "model": roc_auc_score(te["is_bust"], pm),
                     "adp": roc_auc_score(te["is_bust"], a.predict_proba(te[ADP_CONTROL])[:, 1])})
        y = te["is_bust"].sample(frac=1.0, random_state=ts).to_numpy()
        if len(np.unique(y)) > 1:
            shuf.append(roc_auc_score(y, pm))
    if not rows:
        print("  " + label + ": no usable folds")
        return None
    per = pd.DataFrame(rows)
    per["gain"] = per.model - per.adp
    lo, hi = season_bootstrap(per.gain, rng)
    print("\n--- " + label + " | " + str(len(per)) + " test seasons ---")
    print(per.round(3).to_string(index=False))
    verdict = "PASS" if abs(np.mean(shuf) - 0.5) < 0.06 else "FAIL (harness leaks)"
    print("  shuffled-label AUC = {:.4f}  {}".format(np.mean(shuf), verdict))
    print("  model {:.4f} vs ADP {:.4f} | gain {:+.4f} CI=[{:+.4f},{:+.4f}] | beats {:.0%} -> {}".format(
        per.model.mean(), per.adp.mean(), per.gain.mean(), lo, hi,
        (per.gain > 0).mean(), "SIGNIFICANT" if lo > 0 else "CI includes 0"))
    return per


def main() -> None:
    rng = np.random.default_rng(SEED)
    d = load()
    print("WR player-seasons, ADP<={}: {} | {}-{} | bust rate {:.1%}".format(
        ADP_MAX, len(d), int(d.season.min()), int(d.season.max()), d.is_bust.mean()))

    print("\n" + SEP)
    print("1-2. LEAKAGE + PER-SEASON AUC, both specifications")
    print(SEP)
    best = None
    for label, (feats, start) in SPECS.items():
        feats = [f for f in feats if f in d.columns]
        per = evaluate_spec(d, feats, start, rng, label)
        if per is not None and (best is None or len(per) > len(best[1])):
            best = (label, per, feats, start)
    if best is None:
        return
    label, per, feats, start = best

    print("\n" + SEP)
    print("3. CALIBRATION -- spec: " + label + " (calibrator fit on TRAIN only)")
    print(SEP)
    rows = []
    for method in ("uncalibrated", "sigmoid", "isotonic"):
        briers, preds, actuals = [], [], []
        for ts, tr, te in folds(d, feats, start):
            est = make_pipe() if method == "uncalibrated" else CalibratedClassifierCV(
                make_pipe(), method=method, cv=3)
            est.fit(tr[feats], tr["is_bust"])
            p = est.predict_proba(te[feats])[:, 1]
            briers.append(brier_score_loss(te["is_bust"], p))
            preds.extend(p)
            actuals.extend(te["is_bust"])
        preds, actuals = np.array(preds), np.array(actuals)
        cl = np.clip(preds, 1e-6, 1 - 1e-6)
        lr = LogisticRegression(max_iter=1000).fit(np.log(cl / (1 - cl)).reshape(-1, 1), actuals)
        rows.append({"method": method, "brier": np.mean(briers),
                     "slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]),
                     "mean_pred": preds.mean(), "actual": actuals.mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("  (slope 1.0 / intercept 0.0 = calibrated; lower Brier better)")

    print("\n" + SEP)
    print("4. COEFFICIENT STABILITY (sign flips => not interpretable as a driver)")
    print(SEP)
    C = pd.DataFrame([
        pd.Series(make_pipe().fit(tr[feats], tr["is_bust"]).named_steps["model"].coef_[0], index=feats)
        for _, tr, _ in folds(d, feats, start)])
    print(pd.DataFrame({
        "median": C.median(), "p10": C.quantile(.10), "p90": C.quantile(.90),
        "sign_consistency": C.apply(lambda s: max((s > 0).mean(), (s < 0).mean())),
    }).sort_values("median", key=abs, ascending=False).round(3).to_string())

    print("\n" + SEP)
    print("5. SUPPRESSION TEST -- div_opportunity_minus_efficiency")
    print(SEP)
    DIV = "div_opportunity_minus_efficiency"
    variants = {
        "all features": feats,
        "drop snap_share": [f for f in feats if f != "prior_snap_share"],
        "drop TPRR": [f for f in feats if f != "prior_targets_per_route_run"],
        "div + adp only": [DIV, "overall_adp"],
    }
    for nm, fs in variants.items():
        if DIV not in fs:
            continue
        cs, au = [], []
        for ts, tr, te in folds(d, fs, start):
            p = make_pipe().fit(tr[fs], tr["is_bust"])
            cs.append(pd.Series(p.named_steps["model"].coef_[0], index=fs)[DIV])
            au.append(roc_auc_score(te["is_bust"], p.predict_proba(te[fs])[:, 1]))
        if cs:
            c = np.array(cs)
            print("  {:18s} div_coef={:+.3f}  sign_consistency={:.0%}  AUC={:.4f}".format(
                nm, float(np.median(c)), max((c > 0).mean(), (c < 0).mean()), float(np.mean(au))))

    print("\n" + SEP)
    print("6. DECISION BACKTEST -- pre-specified thresholds, all reported")
    print("   Fade top X% by predicted risk; replace with lowest-risk WR within +/-12 ADP.")
    print(SEP)
    out = []
    for thr in FADE_THRESHOLDS:
        dv, db = [], []
        for ts, tr, te in folds(d, feats, start):
            p = make_pipe().fit(tr[feats], tr["is_bust"])
            te = te.copy()
            te["risk"] = p.predict_proba(te[feats])[:, 1]
            faded = te[te.risk >= te.risk.quantile(1 - thr)]
            bv, bb, sv, sb = [], [], [], []
            for _, r in faded.iterrows():
                pool = te[te.overall_adp.between(r.overall_adp - 12, r.overall_adp + 12)
                          & (te.index != r.name)]
                if pool.empty:
                    continue
                rp = pool.loc[pool.risk.idxmin()]
                bv.append(r.vor); bb.append(r.is_bust)
                sv.append(rp.vor); sb.append(rp.is_bust)
            if bv:
                dv.append(np.mean(sv) - np.mean(bv))
                db.append(np.mean(sb) - np.mean(bb))
        if dv:
            lv, hv = season_bootstrap(dv, rng)
            lb, hb = season_bootstrap(db, rng)
            out.append({"fade": "{:.0%}".format(thr), "seasons": len(dv),
                        "vor_gain": round(float(np.mean(dv)), 2),
                        "vor_ci": "[{:+.1f},{:+.1f}]".format(lv, hv),
                        "bust_change": round(float(np.mean(db)), 3),
                        "bust_ci": "[{:+.2f},{:+.2f}]".format(lb, hb),
                        "improved": "{:.0%}".format((np.array(dv) > 0).mean())})
    if out:
        print(pd.DataFrame(out).to_string(index=False))
        print("\n  vor_gain>0 with CI excluding 0 => real draft value.")
        print("  bust_change<0 => the policy avoids busts.")


if __name__ == "__main__":
    main()
