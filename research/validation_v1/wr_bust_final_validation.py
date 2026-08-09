"""
Final validation of the WR conditional-bust model. Diagnosis, not rescue.

The v1 claim (AUC 0.7498 vs 0.7224, +0.0274) is retracted -- see
archive/retracted_wr_bust_v1/README.md. It failed because staggered feature
coverage let sklearn silently drop all-NaN columns, so early folds fit a
smaller model and the pooled figure mixed specifications.

Two questions remain, and this script answers both.

TASK 1 -- NESTED comparison. The prior test compared a feature-only model
against an ADP-only model, which can only show the features are a worse
*substitute* for ADP. The question that matters is whether they add anything
*on top of* ADP:

    baseline:  P(bust) = f(continuous ADP)
    expanded:  P(bust) = f(continuous ADP, WR features)

Continuous ADP is retained in BOTH arms. ADP bands appear only as a secondary
robustness check, never as a substitute for continuous ADP.

TASK 2 -- 2x2 policy decomposition. The v1 policy result (+42.5 VOR) was
confounded: random replacement scored +48.9, so the "gain" came from the
replacement window, not the model. Crossing fade-selection against
replacement-selection isolates which component (if either) carries value:

           fades \ replacements |  model  |  random
        ------------------------+---------+---------
                          model |   1     |    2
                         random |   3     |    4

    (2 vs 4) isolates FADE-selection value.
    (3 vs 4) isolates REPLACEMENT-selection value.

ELIGIBILITY: a replacement must satisfy replacement_ADP >= faded_ADP -- you
cannot draft someone already off the board. Identical rule for model and
random arms. Exclusion rates are reported per policy; the rule is never
loosened after seeing swap counts.

GUARDS: transformed feature schema asserted identical across every fold;
every raw feature asserted to have >=1 non-missing training value per fold;
per-feature train/test missingness recorded. All fitting is train-only.

Nothing here is tuned on results. Seeds are fixed and recorded in metadata.

Run:  python research/validation_v1/wr_bust_final_validation.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from validation_utils import VALIDATION_DIR

# ---------------------------------------------------------------- config
SEED = 17
N_RANDOM_REPS = 1000        # null distribution per season per random policy
N_SEASON_BOOTSTRAP = 5000   # season-blocked resampling
FADE_THRESHOLD = 0.15       # pre-specified; not tuned
ADP_MAX = 150

DATASET = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"
OUTDIR = VALIDATION_DIR / "data" / "wr_bust_final"

REPLACEMENT_RANK = {"QB": 12, "RB": 29, "WR": 29, "TE": 14}
ADP_CONTINUOUS = ["overall_adp"]          # retained in BOTH arms
ADP_BANDS = [(1, 36), (37, 84), (85, 150)]  # secondary robustness only

FEATURES_FULL = [
    "prior_snap_share", "prior_targets_per_route_run",
    "div_opportunity_minus_efficiency", "prior_garbage_time_share",
    "prior_route_participation_rate", "prior_durability_score", "age",
]
FEATURES_CORE = [
    "prior_snap_share", "div_opportunity_minus_efficiency",
    "prior_garbage_time_share", "prior_durability_score", "age",
]
SPECS = {"FULL": (FEATURES_FULL, 2017), "CORE": (FEATURES_CORE, 2014)}
SEP = "=" * 74


class FoldSchemaError(RuntimeError):
    """Raised when a fold cannot honour the availability contract."""


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    d = pd.read_csv(DATASET, low_memory=False)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d = d[d["final_fantasy_points"].notna() & d["overall_adp"].notna()].copy()
    base = {}
    for (season, position), g in d.groupby(["season", "position"]):
        if position not in REPLACEMENT_RANK:
            continue
        pts = g["final_fantasy_points"].sort_values(ascending=False).to_numpy()
        n = REPLACEMENT_RANK[position]
        base[(season, position)] = float(pts[n - 1]) if len(pts) >= n else float(pts[-1])
    d = d[d["position"] == "WR"].copy()
    d["replacement"] = [base.get((s, "WR"), np.nan) for s in d["season"]]
    d["is_bust"] = (d["final_fantasy_points"] < d["replacement"]).astype(int)
    d["vor"] = d["final_fantasy_points"] - d["replacement"]
    return d[d["overall_adp"] <= ADP_MAX].copy()


def make_pipe(seed=SEED):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def folds(d, feats, start, min_train=120, min_test=15):
    d = d[d["season"] >= start]
    for ts in sorted(d["season"].dropna().astype(int).unique()):
        tr, te = d[d["season"] < ts], d[d["season"] == ts]
        if len(tr) < min_train or len(te) < min_test or te["is_bust"].nunique() < 2:
            continue
        yield ts, tr, te


def assert_fold_contract(tr, te, feats, ts, schema_ref, miss_log):
    """
    Hard availability contract, per the stricter guard.

    (a) every raw feature has >=1 non-missing TRAINING value -- this is exactly
        the condition under which SimpleImputer drops a column, which is what
        invalidated v1;
    (b) the transformed schema is identical to every other fold in the spec.
    Missingness is recorded either way: 79-92% coverage is not proof of stable
    measurement quality and the imputation burden is reported as a limitation.
    """
    for f in feats:
        if tr[f].notna().sum() == 0:
            raise FoldSchemaError(f"season {ts}: '{f}' has no non-missing training value")
        miss_log.append({
            "season": ts, "feature": f,
            "train_missing_pct": round(float(tr[f].isna().mean() * 100), 1),
            "test_missing_pct": round(float(te[f].isna().mean() * 100), 1),
        })
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
    """
    Season-blocked folds make overlap structurally impossible, but asserting it
    keeps that guarantee from silently lapsing if the split ever changes.
    """
    kt = set(map(tuple, tr[list(key)].to_numpy()))
    ke = set(map(tuple, te[list(key)].to_numpy()))
    both = kt & ke
    if both:
        raise FoldSchemaError(f"train/test overlap on {len(both)} player-seasons, e.g. {sorted(both)[:3]}")


def assert_continuous_adp_present(*feature_sets):
    """
    Continuous ADP must appear in BOTH nested arms. Dropping it from either
    turns the nested comparison back into the substitution comparison, which
    is the mistake this whole study exists to avoid. ADP bands are never a
    substitute for the continuous term.
    """
    for fs in feature_sets:
        missing = [c for c in ADP_CONTINUOUS if c not in fs]
        if missing:
            raise FoldSchemaError(f"continuous ADP {missing} absent from arm {list(fs)}")


def assert_shared_candidate_pool(pool_model, pool_control, faded_idx):
    """
    Model and control replacement policies must draw from one identical pool.
    Divergent universes were the confound that invalidated v1's policy claim.
    """
    a, b = set(pool_model.index), set(pool_control.index)
    if a != b:
        raise FoldSchemaError(
            f"candidate universes differ for faded player {faded_idx}: "
            f"model-only={sorted(a - b)[:3]} control-only={sorted(b - a)[:3]}")


def season_bootstrap(per_season_values, rng, n=N_SEASON_BOOTSTRAP):
    v = np.asarray(per_season_values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    draws = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- Task 1
def nested_comparison(d, feats, start, label, rng):
    """Baseline f(ADP) vs expanded f(ADP, features). Paired on identical test rows."""
    schema_ref, miss_log, rows = [], [], []
    expanded_feats = ADP_CONTINUOUS + feats

    for ts, tr, te in folds(d, feats, start):
        assert_fold_contract(tr, te, expanded_feats, ts, schema_ref, miss_log)

        base = make_pipe().fit(tr[ADP_CONTINUOUS], tr["is_bust"])
        exp = make_pipe().fit(tr[expanded_feats], tr["is_bust"])
        pb = base.predict_proba(te[ADP_CONTINUOUS])[:, 1]
        pe = exp.predict_proba(te[expanded_feats])[:, 1]
        y = te["is_bust"].to_numpy()

        rows.append({
            "season": ts, "n": len(te), "bust_rate": round(float(y.mean()), 3),
            "auc_base": roc_auc_score(y, pb), "auc_exp": roc_auc_score(y, pe),
            "brier_base": brier_score_loss(y, pb), "brier_exp": brier_score_loss(y, pe),
            "logloss_base": log_loss(y, pb, labels=[0, 1]),
            "logloss_exp": log_loss(y, pe, labels=[0, 1]),
        })

    if not rows:
        return None, pd.DataFrame(miss_log)

    per = pd.DataFrame(rows)
    per["auc_delta"] = per.auc_exp - per.auc_base
    per["brier_delta"] = per.brier_exp - per.brier_base      # negative = better
    per["logloss_delta"] = per.logloss_exp - per.logloss_base  # negative = better

    summary = {"spec": label, "start_season": start, "test_seasons": len(per),
               "features": expanded_feats}
    for m, better_is in (("auc_delta", "higher"), ("brier_delta", "lower"), ("logloss_delta", "lower")):
        lo, hi = season_bootstrap(per[m], rng)
        wins = float((per[m] > 0).mean()) if better_is == "higher" else float((per[m] < 0).mean())
        summary[m] = {"mean": round(float(per[m].mean()), 4), "ci_lo": round(lo, 4),
                      "ci_hi": round(hi, 4), "seasons_improved": round(wins, 3),
                      "excludes_zero": bool(lo > 0 or hi < 0)}
    summary["auc_base_pooled"] = round(float(per.auc_base.mean()), 4)
    summary["auc_exp_pooled"] = round(float(per.auc_exp.mean()), 4)
    return {"per_season": per, "summary": summary}, pd.DataFrame(miss_log)


def calibration_report(d, feats, start):
    """Calibration intercept/slope of the expanded model on pooled OOS predictions."""
    expanded = ADP_CONTINUOUS + feats
    preds, actuals = [], []
    for ts, tr, te in folds(d, feats, start):
        est = make_pipe().fit(tr[expanded], tr["is_bust"])
        preds.extend(est.predict_proba(te[expanded])[:, 1])
        actuals.extend(te["is_bust"])
    preds, actuals = np.array(preds), np.array(actuals)
    cl = np.clip(preds, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(np.log(cl / (1 - cl)).reshape(-1, 1), actuals)
    return {"n": int(len(preds)), "brier": round(float(brier_score_loss(actuals, preds)), 4),
            "intercept": round(float(lr.intercept_[0]), 4),
            "slope": round(float(lr.coef_[0][0]), 4),
            "mean_pred": round(float(preds.mean()), 4),
            "actual_rate": round(float(actuals.mean()), 4)}


# ---------------------------------------------------------------- Task 2
def eligible_pool(te, faded_idx):
    """
    Replacements must satisfy replacement_ADP >= faded_ADP: you cannot draft a
    player already off the board. Identical rule for model and random arms.
    """
    faded_adp = te.loc[faded_idx, "overall_adp"]
    return te[(te["overall_adp"] >= faded_adp) & (te.index != faded_idx)]


def run_policy(te, fade_idx, replacement_rule, rng):
    """Returns realized deltas (swap minus faded) plus eligibility diagnostics."""
    dv, dp, db, dadp, same_band = [], [], [], [], []
    excluded = 0
    for fi in fade_idx:
        pool = eligible_pool(te, fi)
        if pool.empty:
            excluded += 1
            continue
        pick = (pool.loc[pool["risk"].idxmin()] if replacement_rule == "model"
                else pool.iloc[rng.integers(len(pool))])
        f = te.loc[fi]
        dv.append(pick.vor - f.vor)
        dp.append(pick.final_fantasy_points - f.final_fantasy_points)
        db.append(pick.is_bust - f.is_bust)
        dadp.append(pick.overall_adp - f.overall_adp)
        same_band.append(int(any(lo <= f.overall_adp <= hi and lo <= pick.overall_adp <= hi
                                 for lo, hi in ADP_BANDS)))
    if not dv:
        return None
    return {"vor": float(np.mean(dv)), "points": float(np.mean(dp)),
            "bust": float(np.mean(db)), "adp_shift_mean": float(np.mean(dadp)),
            "adp_shift_median": float(np.median(dadp)),
            "same_band_frac": float(np.mean(same_band)),
            "n_swaps": len(dv), "n_excluded": excluded}


def policy_2x2(d, feats, start, label, rng):
    """Cross fade-selection against replacement-selection."""
    expanded = ADP_CONTINUOUS + feats
    season_rows, null_rows = [], []

    for ts, tr, te in folds(d, feats, start):
        est = make_pipe().fit(tr[expanded], tr["is_bust"])
        te = te.copy()
        te["risk"] = est.predict_proba(te[expanded])[:, 1]
        k = max(1, int(round(len(te) * FADE_THRESHOLD)))
        model_fades = te.nlargest(k, "risk").index.tolist()

        # (1) model fade + model replacement, (2) model fade + random replacement
        r1 = run_policy(te, model_fades, "model", np.random.default_rng(SEED))
        if r1:
            season_rows.append({"season": ts, "policy": "1_modelfade_modelrepl", **r1})
        reps2 = [run_policy(te, model_fades, "random", np.random.default_rng(SEED + r))
                 for r in range(N_RANDOM_REPS)]
        reps2 = [x for x in reps2 if x]
        if reps2:
            season_rows.append({"season": ts, "policy": "2_modelfade_randrepl",
                                **{k2: float(np.mean([x[k2] for x in reps2])) for k2 in reps2[0]}})
            null_rows += [{"season": ts, "policy": "2_modelfade_randrepl", "rep": i,
                           "vor": x["vor"], "bust": x["bust"]} for i, x in enumerate(reps2)]

        # random fades matched on ADP band to the model's fade set
        bands_needed = [next((j for j, (lo, hi) in enumerate(ADP_BANDS)
                              if lo <= te.loc[i, "overall_adp"] <= hi), None) for i in model_fades]
        r3s, r4s = [], []
        for rep in range(N_RANDOM_REPS):
            rr = np.random.default_rng(SEED + 10_000 + rep)
            picks = []
            for b in bands_needed:
                if b is None:
                    continue
                lo, hi = ADP_BANDS[b]
                cand = te[(te.overall_adp.between(lo, hi)) & (~te.index.isin(picks))]
                if not cand.empty:
                    picks.append(cand.index[rr.integers(len(cand))])
            if not picks:
                continue
            a = run_policy(te, picks, "model", rr)
            b_ = run_policy(te, picks, "random", rr)
            if a:
                r3s.append(a)
            if b_:
                r4s.append(b_)
        for nm, reps in (("3_randfade_modelrepl", r3s), ("4_randfade_randrepl", r4s)):
            if reps:
                season_rows.append({"season": ts, "policy": nm,
                                    **{k2: float(np.mean([x[k2] for x in reps])) for k2 in reps[0]}})
                null_rows += [{"season": ts, "policy": nm, "rep": i,
                               "vor": x["vor"], "bust": x["bust"]} for i, x in enumerate(reps)]

    return pd.DataFrame(season_rows), pd.DataFrame(null_rows)


def contrast(per_policy, a, b, rng, metric="vor"):
    """Season-blocked paired contrast between two policies."""
    pa = per_policy[per_policy.policy == a].set_index("season")[metric]
    pb = per_policy[per_policy.policy == b].set_index("season")[metric]
    shared = pa.index.intersection(pb.index)
    if len(shared) < 3:
        return None
    diffs = (pa.loc[shared] - pb.loc[shared]).to_numpy()
    lo, hi = season_bootstrap(diffs, rng)
    return {"contrast": f"{a} - {b}", "metric": metric, "seasons": int(len(shared)),
            "mean": round(float(diffs.mean()), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "seasons_positive": round(float((diffs > 0).mean()), 3),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- main
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load()
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
            "n_random_reps": N_RANDOM_REPS, "n_season_bootstrap": N_SEASON_BOOTSTRAP,
            "fade_threshold": FADE_THRESHOLD, "adp_max": ADP_MAX,
            "eligibility_rule": "replacement_ADP >= faded_ADP",
            "rows": int(len(d)), "seasons": [int(s) for s in sorted(d.season.unique())]}

    print(SEP)
    print("WR BUST MODEL -- FINAL VALIDATION")
    print(SEP)
    print(f"WR player-seasons (ADP<={ADP_MAX}): {len(d)} | bust rate {d.is_bust.mean():.1%}")

    all_summaries, all_contrasts = [], []
    for label, (feats, start) in SPECS.items():
        feats = [f for f in feats if f in d.columns]
        print(f"\n{SEP}\nSPEC {label} (from {start})\n{SEP}")
        try:
            res, miss = nested_comparison(d, feats, start, label, rng)
        except FoldSchemaError as e:
            print(f"  SPECIFICATION FAILED: {e}")
            continue
        if res is None:
            print("  no usable folds")
            continue

        per, summ = res["per_season"], res["summary"]
        print("\nTASK 1 -- nested: f(ADP) vs f(ADP + features), paired on identical test rows")
        print(per[["season", "n", "bust_rate", "auc_base", "auc_exp", "auc_delta",
                   "brier_delta", "logloss_delta"]].round(4).to_string(index=False))
        for m in ("auc_delta", "brier_delta", "logloss_delta"):
            s = summ[m]
            print(f"  {m:14s} mean={s['mean']:+.4f} CI=[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] "
                  f"improved {s['seasons_improved']:.0%} -> "
                  f"{'EXCLUDES ZERO' if s['excludes_zero'] else 'includes zero'}")

        cal = calibration_report(d, feats, start)
        summ["calibration"] = cal
        print(f"  calibration: brier={cal['brier']} slope={cal['slope']} "
              f"intercept={cal['intercept']} (mean_pred {cal['mean_pred']} vs actual {cal['actual_rate']})")

        imp_burden = miss.groupby("feature").train_missing_pct.mean().round(1)
        summ["mean_train_missing_pct"] = imp_burden.to_dict()
        print(f"  imputation burden (mean train missing %): {imp_burden.to_dict()}")
        miss.to_csv(OUTDIR / f"missingness_{label}.csv", index=False)
        per.to_csv(OUTDIR / f"nested_per_season_{label}.csv", index=False)

        print(f"\nTASK 2 -- 2x2 policy ({N_RANDOM_REPS} seeded reps/season)")
        pp, nulls = policy_2x2(d, feats, start, label, rng)
        if pp.empty:
            print("  no policy folds")
        else:
            agg = pp.groupby("policy").agg(
                vor=("vor", "mean"), points=("points", "mean"), bust=("bust", "mean"),
                adp_shift=("adp_shift_mean", "mean"), same_band=("same_band_frac", "mean"),
                swaps=("n_swaps", "mean"), excluded=("n_excluded", "mean"),
                seasons=("season", "nunique")).round(3)
            print(agg.to_string())
            pp.to_csv(OUTDIR / f"policy_per_season_{label}.csv", index=False)
            nulls.to_csv(OUTDIR / f"policy_null_distributions_{label}.csv", index=False)

            print("\n  Isolating components (each vs its matched random control):")
            for a, b, what in (
                ("2_modelfade_randrepl", "4_randfade_randrepl", "FADE selection"),
                ("3_randfade_modelrepl", "4_randfade_randrepl", "REPLACEMENT selection"),
                ("1_modelfade_modelrepl", "4_randfade_randrepl", "FULL policy"),
            ):
                c = contrast(pp, a, b, rng)
                if c:
                    c["component"] = what
                    c["spec"] = label
                    all_contrasts.append(c)
                    print(f"    {what:22s} vor {c['mean']:+7.2f} CI=[{c['ci_lo']:+.2f},{c['ci_hi']:+.2f}] "
                          f"pos {c['seasons_positive']:.0%} -> "
                          f"{'REAL' if c['excludes_zero'] else 'no evidence'}")
        all_summaries.append(summ)

    (OUTDIR / "summary.json").write_text(json.dumps(
        {"metadata": meta, "nested": all_summaries, "policy_contrasts": all_contrasts}, indent=2))
    if all_contrasts:
        pd.DataFrame(all_contrasts).to_csv(OUTDIR / "policy_contrasts.csv", index=False)
    print(f"\n{SEP}\nArtifacts written to {OUTDIR}\n{SEP}")


if __name__ == "__main__":
    main()
