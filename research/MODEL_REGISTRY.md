# Model Registry

Canonical index of research models and their production eligibility.

**Default status for anything listed here is RESEARCH_ONLY.** An entry is
eligible for live use only if it explicitly says so. If you are wiring
something into rankings, tiers, recommendations, or UI, check this file
first.

---

## `breakout_score_v1_research`

**Status:** `RESEARCH_ONLY — NOT ELIGIBLE FOR LIVE DRAFT RECOMMENDATIONS`

**Scope:** WR and RB; historical nested-comparison research (2010-2025). No current-season scoring path exists. Deliberately mirrors `wr_conditional_bust_risk_research`'s Task 1 methodology (same pipeline, same guard contracts, same season-blocked bootstrap CIs) for direct comparability, using the same feature source (`predraft_validation_dataset_archetypes_v1.csv`) and the same CORE/FULL feature-availability split.

**Question tested:** Whether prior-season usage/opportunity features (`prior_snap_share`, `prior_garbage_time_share`, `div_opportunity_minus_efficiency`, `prior_durability_score`, `age`, plus route-participation features in the FULL spec) add incremental predictive information beyond continuous ADP for predicting `Beat_ADP_By_12` (a literal breakout definition: finished the season having beaten draft slot by 12+ overall spots).

**Primary finding:** Weaker and more uneven than the analogous bust-risk result, and position-dependent. Two genuinely different feature categories were tested, both negative-to-null:

*Usage/opportunity features (CORE/FULL, same spec as the bust study):*
- **WR FULL** (2017+, 7 seasons): AUC delta +0.035, CI `[-0.0068,+0.0764]` — does not exclude zero, but 71% of seasons improved. Directionally positive, underpowered, same shape as the bust study's own inconclusive FULL result.
- **WR CORE** (2014+, 10 seasons): AUC delta +0.0155, CI includes zero, only 50% of seasons improved — no reliable ranking improvement. Brier (−0.0165) and log-loss (−0.0354) deltas **do** exclude zero — the features measurably improve probability calibration without improving rank-ordering.
- **RB FULL**: AUC delta −0.0349, CI includes zero, only 33% of seasons improved.
- **RB CORE**: AUC delta **−0.0248, CI `[-0.0472,-0.0023]` — excludes zero, negative.** Reliably *hurts* predictive power relative to ADP alone.

*Draft capital / development stage (`estimated_draft_round`, `draft_pick_overall`, `years_since_drafted`, `age` — 84-100% coverage every season 2010-2025, no staggered-coverage issue; added same day after the usage-based specs showed weak-to-negative results):*
- **WR CAPITAL** (2010+, 13 seasons): AUC delta **−0.0124, CI `[-0.0254,-0.0004]` — excludes zero, negative.**
- **RB CAPITAL** (2010+, 13 seasons): AUC delta **−0.0112, CI `[-0.0204,-0.0019]` — excludes zero, negative.** (Brier/log-loss deltas here do exclude zero in the improving direction — better-calibrated on average while reliably worse at rank-ordering, a real if counterintuitive split.)

Three of six position × spec combinations (RB CORE, WR CAPITAL, RB CAPITAL) show a **statistically significant negative** effect, not merely an absence of a positive one. Plausible explanation for CAPITAL specifically: draft capital and experience are already substantially priced into ADP itself (the market already knows a player's draft pedigree and tenure), so adding them on top is largely redundant information that adds noise rather than signal.

**Interpretation:** Two structurally different feature categories -- usage/opportunity level, and talent/experience stage -- were tried and neither shows reliable incremental value for breakout prediction on this dataset. Usage features that carry real signal for predicting who *busts* (mechanically tied to who doesn't get volume) do not transfer to predicting who *breaks out* (a swing up usually driven by an opportunity change the prior season's data doesn't yet reflect). Further undirected feature search on this dataset is unlikely to be productive without either a different target definition or genuinely new data (situational/opportunity-change signals like vacated targets or depth-chart shifts exist as column names in this dataset but are 0% populated).

**Follow-up: does the model correctly call a specific SUBSET even where the aggregate is null? One real result, one that didn't survive proper testing.** A whole-population AUC can hide a model that's genuinely good at its own most-confident calls. Tested via top-~10%-by-predicted-probability hit rate, computed WITHIN each test season (not pooled) and season-blocked bootstrapped exactly like the AUC/Brier/logloss deltas above -- same rigor as the confirmatory result, not a separate lower bar.

- **WR FULL: delta +11.9%, CI `[+7.1%,+16.7%]` -- excludes zero, 71% of seasons improved.** The one real, confirmed finding in this whole study. The model's top decile hits meaningfully more than ADP's own top decile, even though WR FULL's whole-population AUC delta itself didn't clear significance. Caveat: only 7 test seasons (route-participation data starts 2017), and this is 1 of 6 tested position×spec combinations with no multiple-comparison correction applied -- a real, promising lead, not an unconditionally proven fact. Worth a genuine out-of-sample check once new seasons are available, not further mining of the same 7 seasons.
- **RB CORE reversed on proper testing -- a real methodological lesson, not just a null result.** An earlier PASS using a pooled (not season-blocked) point estimate showed an apparently promising +6.6pp lift for RB CORE's top decile, despite RB CORE's aggregate AUC being reliably negative. Under the same season-blocked bootstrap used everywhere else, that reversed: delta −6.7%, CI `[-15.6%,0.0%]`, and the expanded model's top decile beat ADP-only's top decile in **0 of 9 test seasons.** The pooled estimate was actively misleading here, not just noisier -- exactly the class of mistake the WR bust study's own guard-test discipline exists to catch, reproduced and caught within this study.
- All other position×spec combinations: CI includes zero, no confirmed effect either direction.

**Robustness checks on the WR FULL top-decile finding (added 2026-08-27):** four independent stress tests, run because a single CI-excludes-zero result from 1 of 6 tested combinations warrants more than a bare confidence interval before it's treated as trustworthy.
- **Permutation test:** shuffled `is_breakout` within each season 5,000 times against the fixed real model predictions, rebuilding a null distribution of the same topdecile-delta statistic. Observed +0.119 vs. null mean +0.0004 (std 0.052); **one-sided p=0.0188** — survives at p<0.05, and this test implicitly answers the multiple-comparison concern the CI alone did not.
- **Leave-one-season-out:** dropping any single one of the 7 test seasons shifts the mean effect by at most ±0.02 off the full +0.119 — no one season is carrying the result.
- **Feature coefficient sign consistency (fold-level, matching WR_BUST_DECISION_MEMO.md's own check):** 6 of 7 features are 100% sign-consistent across all 7 walk-forward folds. **`div_opportunity_minus_efficiency` is only 57% consistent** (positive in 4 of 7 folds) — the one weak link in an otherwise stable feature set.
- **Threshold sensitivity:** effect excludes zero at top-10%, top-15%, and top-20% cutoffs; the top-5% cutoff does not exclude zero (CI `[-0.048,+0.476]`), most plausibly an underpowered-sample artifact at that narrower cut rather than evidence against the 10% result specifically.
- **Verdict: the finding survives all four checks with one flagged weakness** (the one unstable feature). Still a single historical sample, not new out-of-sample evidence — see reproduction note below.
- **Authoritative reference:** [`validation_v1/breakout_v1_robustness_report.md`](validation_v1/breakout_v1_robustness_report.md)

**Trimmed-feature re-validation (added 2026-08-27):** `div_opportunity_minus_efficiency`'s 57%-consistent coefficient sign (above) was investigated by dropping it and re-running the full battery on the remaining 6 features. Every metric improved: top-decile delta **+14.3% [+7.1%,+21.4%]** (was +11.9%), AUC delta +0.0383 [-0.0058,+0.0820] (was +0.0350), permutation test **p=0.0042** (was p=0.0188 — this now clears a Bonferroni family-wise threshold of 0.0083 across the 6 originally tested position×spec combinations; the 7-feature version did not), and threshold sensitivity now excludes zero at **all four** tested cutoffs (5/10/15/20%), not three of four. This 6-feature spec supersedes the original 7-feature FULL spec as the stronger, cleaner result and is what the frozen scorer (below) uses. **Authoritative reference:** [`validation_v1/breakout_v1_trimmed_feature_report.md`](validation_v1/breakout_v1_trimmed_feature_report.md)

**Frozen scorer + calibration check (added 2026-08-27):** `breakout_score_v1_scorer.py` turns the trimmed WR result into an actual scoring function (`score_players()`), per five requirements: frozen target (`Beat_ADP_By_12`), the trimmed 6-feature set above (no CORE/CAPITAL features folded back in), logistic regression (not a boosted tree — 7 usable seasons is too small a sample for a many-parameter model), season-blocked evaluation kept permanent, and calibration checked explicitly.
- **Calibration finding: the validation pipeline's `class_weight="balanced"` badly miscalibrates raw probabilities** — mean predicted probability 42.5% vs. actual observed rate 16.1% (Brier 0.211, calibration slope 0.63). Dropping that weighting (unweighted logistic regression) fixes this almost completely — mean predicted 15.3% vs. actual 16.1%, Brier 0.126, slope 0.79 — while walk-forward AUC is unchanged or slightly better (0.744 vs. 0.738) and top-decile performance is comparable. The scorer uses the unweighted fit; the validation/robustness scripts above still use the balanced pipeline (kept as-is for exact comparability with the frozen historical result) and should not be read as producing usable probability estimates on their own.
- **Authoritative reference:** [`validation_v1/breakout_v1_scorer_report.md`](validation_v1/breakout_v1_scorer_report.md)

**Prospective prediction locked in (added 2026-08-27):** `score_current_wr_pool_v1.py` scored the actual 2026 WR draft pool (63 players, ADP≤150) using the frozen scorer, timestamped and written to `validation_v1/data/breakout_v1/current_wr_predictions_2026.csv`. This is the genuine out-of-sample test every check above is a proxy for: every prior check (nested comparison, permutation test, leave-one-season-out, coefficient stability, threshold sensitivity, and the decision to drop `div_opportunity_minus_efficiency` itself) was computed on seasons that already existed when this study was built, so the overall spec was iteratively shaped by what worked on those same 7 seasons — a form of look-ahead no amount of within-sample testing fully rules out. A prediction locked in before Week 1 2026 is not subject to that risk. **Reopen criterion for this entry: compare `current_wr_predictions_2026.csv` against real `Beat_ADP_By_12` outcomes once the 2026 season is final** — that comparison, not another backtest pass, is what should move this model's production status.

**Production decision:** **Still Do not use in any live UI, ranking, or recommendation path.** A concrete, calibrated scoring function now exists, has passed real robustness checks (including a stronger trimmed-feature version that clears multiple-comparison correction), and has a timestamped prospective prediction on file — which is substantially more than this entry could say before 2026-08-27. But everything that produced it is still backtested, and wiring it into Home.py, draft_analysis.py, or any player-facing surface is a separate, explicit decision this entry does not make on its own. The bar for that decision is the reopen criterion above, not additional backtest rigor.

**Not attempted:** the bust study's Task 2 (policy decomposition) — a materially bigger undertaking, not worth attempting given Task 1 didn't clear the bar that motivated it for bust.

**Authoritative reference:** [`validation_v1/breakout_v1_validation_report.md`](validation_v1/breakout_v1_validation_report.md)

**Reproduction:**
```bash
python research/validation_v1/build_breakout_validation_v1.py
python research/validation_v1/build_breakout_robustness_checks_v1.py
python research/validation_v1/build_breakout_trimmed_feature_check_v1.py
python research/validation_v1/breakout_score_v1_scorer.py
python research/validation_v1/score_current_wr_pool_v1.py
```

---

## `wr_conditional_bust_risk_research`

**Status:** `RESEARCH_ONLY — NOT ELIGIBLE FOR LIVE DRAFT RECOMMENDATIONS`

**Scope:** WR only; historical walk-forward research. No current-season scoring path exists.

**Question tested:** Whether WR features add conditional bust-risk information beyond continuous ADP.

**Primary finding:** CORE features showed incremental out-of-sample predictive information beyond continuous ADP in the 2014+ sample. Nested comparison (`f(ADP)` vs `f(ADP, features)`) improved AUC, Brier score and log-loss with season-blocked CIs excluding zero.

**Important limitation:** Incremental predictive value did **not** establish an actionable draft-policy edge. Improving a conditional probability estimate and improving a draft decision are different results requiring different tests. Only the former was demonstrated.

**Policy status:** **Unvalidated.** The confirmatory policy test was **infeasible / underpowered** under the preregistered exact matching design, because replacement opportunity count was structurally collinear with ADP rank under the defined candidate pool. Matching on both reduced to exact ADP-rank matching and retained only tied-ADP observations. This is a property of that candidate-pool construction, not of draft-policy evaluation generally.

**FULL specification:** **Inconclusive, not negative proof.** Positive point estimates with insufficient precision across seven seasons.

**Production decision:** **Do not use** this model, its probabilities, feature coefficients, risk labels, rankings, or outputs in live draft recommendations, player ranks, tiers, or UI explanations.

**Allowed use:** Historical research reference; data-pipeline validation; methodology example; candidate input for a future separately preregistered study.

**Reopen criteria** — a separately preregistered, realistic choice-set policy study that:
1. defines the candidate set available at each simulated draft slot;
2. compares ADP-only versus ADP-plus-model decisions within identical candidate sets;
3. pre-specifies roster constraints, decision rules, and outcomes;
4. demonstrates a policy edge beyond matched / random / ADP controls.

**Frozen commits:**

| commit | purpose |
|---|---|
| `8d2862ae6bcbc2ec2aec6c3134eec270b9a7439b` | Analysis snapshot — code, tests, memo, verified input dataset, result artifacts, archived v1 retraction record |
| `056d7da6aaab5a2d5a5395b6e40a615863f6e703` | Metadata binding — ties provenance metadata to the analysis snapshot |
| `cc04ea2f59cd2a19fa30a6a2c2c8b0db7f2aeac5` | Provenance-scope clarification — cleanliness claims are package-scoped, not repository-scoped |

**Authoritative references:**
- Memo (findings, limitations, recommendation): [`validation_v1/WR_BUST_DECISION_MEMO.md`](validation_v1/WR_BUST_DECISION_MEMO.md)
- Provenance record (commits, seeds, environment, input-data identities): [`validation_v1/data/wr_bust_final/feasibility_metadata.json`](validation_v1/data/wr_bust_final/feasibility_metadata.json)
- Superseded v1 and why it was retracted: [`../archive/retracted_wr_bust_v1/README.md`](../archive/retracted_wr_bust_v1/README.md)

**Reproduction:**
```bash
python research/validation_v1/wr_bust_final_validation.py
```
```bash
python research/validation_v1/wr_bust_policy_feasibility.py
```
```bash
cd research/validation_v1 && python -m unittest discover -v
```

**Integration status (verified at registry creation):** no reference to this
model or its artifacts exists in `Home.py`, `draftkit/`, `pages_archive/`,
`utils.py`, or `app.py`. It is not wired into any production path.
