# WR Conditional-Bust Model — Final Decision Memo

**Recommendation: Research-only — limited evidence, not decision-ready.**

## Conclusion, stated precisely

> **CORE WR features show incremental predictive information beyond continuous ADP in the current 2014+ walk-forward sample, but the draft-policy implementation is not yet validated because ADP-distance and eligibility differences confound the policy decomposition.**

This is a reversal of a *narrow* conclusion, not a restoration of the original production claim. Specifically:

| statement | status |
|---|---|
| "features replace ADP" | **false** — feature-only models lose to ADP-only |
| "there is no incremental signal" | **no longer supported** — nested test, CORE, CIs exclude zero |
| "the model is ready for live draft decisions" | **still unsupported** — policy legs are confounded |

The **nested comparison is the primary research result.** FULL is **inconclusive, not negative proof**: it has a smaller sample (7 vs 10 test seasons) and a different specification and period, so its wider CIs cannot distinguish a real effect from none.

Reproduce with:
```bash
python research/validation_v1/wr_bust_final_validation.py      # nested comparison + calibration
python research/validation_v1/wr_bust_policy_feasibility.py    # feasibility audit (structure only)
cd research/validation_v1 && python -m unittest discover -v    # guard tests
```

Artifacts: `research/validation_v1/data/wr_bust_final/`

**Frozen provenance — `feasibility_metadata.json` is the authoritative record of:**

| item | field |
|---|---|
| Git commit and dirty status | `git_commit`, `git_dirty` |
| Python and package versions | `python`, `platform`, `packages` |
| Input-data hashes (SHA-256) | `input_data` |
| Random seeds | `seeds` (SEED=17) |
| Timestamp | `timestamp_utc` |
| Reproduction command | `command` |

`summary.json` carries the same seed and run metadata for the nested-comparison artifacts.

---

## 1. Headline: the nested test reverses the earlier conclusion

The prior retraction compared a **feature-only** model against an **ADP-only** model. That can only show whether features are a worse *substitute* for ADP — which they are. It cannot answer whether they add value *on top of* ADP.

The correct nested comparison retains continuous ADP in **both** arms:

- baseline: `P(bust) = f(overall_adp)`
- expanded: `P(bust) = f(overall_adp, WR features)`

| spec | seasons | AUC base | AUC exp | ΔAUC | ΔBrier | Δlog-loss | improved |
|---|---|---|---|---|---|---|---|
| **CORE** (2014+) | 10 | 0.7318 | 0.7695 | **+0.0377** `[+0.0109,+0.0611]` | **−0.0144** `[−0.0229,−0.0053]` | **−0.0327** `[−0.0535,−0.0096]` | 80% |
| FULL (2017+) | 7 | 0.7395 | 0.7650 | +0.0255 `[−0.0115,+0.0594]` | −0.0094 `[−0.0227,+0.0046]` | −0.0239 `[−0.0540,+0.0100]` | 71% |

**CORE excludes zero on all three metrics.** FULL does not. Season-blocked bootstrap, 5,000 resamples, paired on identical held-out rows.

**The earlier "-0.021 / -0.048" figures were not wrong** — they answered the substitution question. This answers the incremental one, and the sign flips.

## 2. Data eligibility

- Universe: WR player-seasons with realized outcomes and `overall_adp ≤ 150`. **925 rows, 2010–2025.** Base bust rate 51.6%.
- Bust = finished below that season's WR replacement level (WR29, reflecting 12 teams × 2 WR + 0.4 FLEX share).
- Walk-forward: train strictly on prior seasons; ≥120 train rows, ≥15 test rows, both classes present in test.

### Feature availability by specification

| spec | features | start | reason |
|---|---|---|---|
| FULL | snap share, TPRR, opp−eff divergence, garbage-time share, route participation, durability, age | 2017 | route data begins 2017 |
| CORE | snap share, opp−eff divergence, garbage-time share, durability, age | 2014 | drops route features, buys 3 seasons |

Coverage is staggered: 2010–13 has essentially nothing but `age`; 2014–16 lacks route data; 2017+ is complete. This is exactly what invalidated v1.

## 3. Confirmation that no feature is silently removed

Enforced per fold, raising `FoldSchemaError` rather than degrading silently:

1. every raw feature has ≥1 non-missing **training** value (the precise condition under which `SimpleImputer` drops a column);
2. `imputer.statistics_` contains no NaN;
3. the transformed feature schema is byte-identical to every other fold in the specification;
4. all imputers, scalers and models are fit on training rows only.

**No fold raised.** Both specifications are legal.

### Imputation burden (mean % of training rows imputed) — a stated limitation

| feature | CORE | FULL |
|---|---|---|
| `div_opportunity_minus_efficiency` | **22.8%** | 19.2% |
| `prior_durability_score` | **22.5%** | 16.5% |
| `prior_targets_per_route_run` | — | 12.4% |
| `prior_garbage_time_share` | 12.8% | 9.8% |
| `prior_snap_share` | 9.8% | 7.2% |

Roughly a fifth of training values for the two weakest-covered features are median-imputed. **83% coverage is not evidence of stable measurement quality**, and this is a material caveat on both the point estimates and the driver interpretation.

## 4. Calibration

| spec | Brier | slope | intercept | mean pred | actual |
|---|---|---|---|---|---|
| CORE | 0.1975 | **1.033** | 0.074 | 0.518 | 0.533 |
| FULL | 0.2001 | 0.908 | 0.165 | 0.491 | 0.525 |

CORE is close to well-calibrated (slope ≈ 1, small positive intercept — mildly under-predicting). Usable as probabilities, not merely ranks.

## 5. 2×2 policy decomposition

Eligibility: `replacement_ADP ≥ faded_ADP` — you cannot draft a player already off the board. Identical rule in both arms. 1,000 seeded repetitions per season for every random policy; season-blocked bootstrap on top.

### CORE (10 seasons)

| policy | VOR | bust Δ | ADP shift | same-band | swaps | excluded |
|---|---|---|---|---|---|---|
| 1 model fade + model repl | **+30.6** | −0.166 | **+8.0** | 0.975 | 8.40 | 0.50 |
| 2 model fade + random repl | +17.8 | −0.106 | +18.7 | 0.867 | 8.40 | 0.50 |
| 3 random fade + model repl | +0.5 | −0.018 | **+8.3** | 0.980 | 8.59 | 0.32 |
| 4 random fade + random repl | −16.1 | +0.082 | +21.8 | 0.874 | 8.59 | 0.32 |

| component | contrast | VOR | 95% CI | seasons + | verdict |
|---|---|---|---|---|---|
| **Fade selection** | 2 − 4 | +33.91 | `[+22.53,+44.93]` | **100%** | excludes zero |
| **Replacement selection** | 3 − 4 | +16.59 | `[+4.00,+28.32]` | 60% | excludes zero |
| Full policy | 1 − 4 | +46.76 | `[+25.16,+67.70]` | 90% | excludes zero |

### FULL (7 seasons) — inconclusive

| component | VOR | 95% CI | verdict |
|---|---|---|---|
| Fade selection | +20.81 | `[−3.71,+42.43]` | inconclusive |
| Replacement selection | +21.93 | `[−6.06,+47.03]` | inconclusive |
| Full policy | +37.00 | `[−2.24,+72.70]` | inconclusive |

Point estimates are positive and directionally consistent with CORE; the CIs
are simply too wide at 7 seasons to resolve. **This is not evidence against
the effect.**

## 6. The confound that prevents a stronger recommendation

**Model-selected replacements sit systematically closer in ADP than random ones**, and closer means better:

| | model replacement | random replacement |
|---|---|---|
| mean ADP shift | **+8.0 / +8.3** | **+18.7 / +21.8** |
| same-band fraction | 0.975 / 0.980 | 0.867 / 0.874 |

Under `replacement_ADP ≥ faded_ADP`, every replacement is later. The model picks ones ~8 slots later; random picks ~21. **The replacement-selection result (+16.59) is therefore not cleanly attributable to risk assessment** — picking a nearer-ADP player is largely sufficient to produce it. This is the same class of confound that invalidated the v1 policy claim, reappearing in a new form.

The fade contrast is less affected but not immune: model fades draw replacements ~3 slots closer than random fades (18.7 vs 21.8), and exclusion rates differ (**0.50 vs 0.32** — model-selected fades more often have no legal replacement, indicating they sit later in ADP).

Per the pre-commitment, no rule was changed after seeing these numbers. Fixing this needs an ADP-shift-matched replacement control, specified in advance and run as a separate study.

## 7. Evidence summary

| claim | evidence | verdict |
|---|---|---|
| **Incremental predictive edge over continuous ADP** | CORE: ΔAUC +0.038, ΔBrier −0.014, Δlog-loss −0.033, all CIs exclude zero, 80% of seasons | **Supported, CORE only** |
| Same, FULL specification | all three CIs include zero | Not supported |
| **Fade-selection edge** | +33.91 VOR `[+22.53,+44.93]`, 100% of seasons | **Supported**, mild ADP-shift confound |
| **Replacement-selection edge** | +16.59 VOR `[+4.00,+28.32]` | **Confounded** — model replacements are 13 slots nearer in ADP |

## 8. Limitations

1. **Specification-dependent.** CORE clears; FULL does not. CORE has 10 seasons vs 7, so this may be power rather than a better feature set — the two cannot be separated with this data.
2. **Low power.** 7–10 test seasons. Wide CIs; a modest true effect and no effect are hard to distinguish.
3. **Imputation burden** of ~20% on two CORE features.
4. **ADP-shift confound** on the replacement component (§6).
5. **Two negative seasons in both specs** (2023, 2025) — 2025 being the most recent is worth noting.
6. **No causal claim.** Coefficients are conditional associations under collinearity. A prior suppression test showed `div_opportunity_minus_efficiency` carries 50% sign consistency with snap share present but −0.336 at 100% consistency without it — snap share was carrying it.
7. **Stable correlates are not independently actionable.** Snap share, garbage-time share, durability and age all show 100% sign consistency, but the nested test is the only thing establishing they add value beyond ADP.

## 8b. Final evidence classification

| Claim | Status | Interpretation |
|---|---|---|
| Features replace ADP | Not supported | Feature-only models should not replace the market baseline |
| CORE features add information beyond continuous ADP | Supported in current sample | Nested CORE model improves out-of-sample predictive metrics in 2014+ data |
| FULL model adds information beyond continuous ADP | Inconclusive | Positive point estimates but insufficient precision with seven seasons |
| Model identifies a validated fade policy | Unvalidated | Confirmatory policy test is infeasible under the exact matching design |
| Model identifies validated replacement choices | Unvalidated | Earlier result was confounded by ADP distance |
| Model is ready for live drafting | No | Research-only; no production use |

## 8c. Predictive value and policy value are separate claims

These are routinely conflated and must not be here.

- **Incremental predictive value does not automatically imply an actionable draft-policy edge.** Improving a conditional probability estimate and improving a draft decision are different results, established by different tests.
- The model may improve **conditional bust-probability estimation** without any demonstration that it improves **real draft choices**. Section 1 supports the former for CORE. Nothing in this study supports the latter.
- The policy test must be described as **"infeasible / underpowered as specified"** — never as evidence for or against a policy edge. It did not run to a result; the matched design admits too few controls to produce one.

Sections 1 and 4 (nested comparison, calibration) are therefore the study's findings. Sections 5, 6 and 10.5 record that the policy question remains open, not answered.

## 9. Recommendation

**Research-only: limited evidence, not decision-ready.**

There is now genuine evidence of incremental predictive value beyond continuous ADP — but only in one of two specifications, on 10 seasons, with ~20% imputation on key features, and with an unresolved ADP-shift confound in the policy decomposition. That is enough to justify further work; it is not enough to change draft decisions.

**Do not productionize.**

**The next decision gate is not another broad model search.** It is whether the incremental CORE signal survives a policy test with ADP-distance-matched and candidate-set-matched controls.

---

## 10. PREREGISTERED policy-validation phase (not yet run)

Specified in advance of any results. Current validation numbers and thresholds are frozen and must not be altered by this phase.

### 10.1 ADP-shift-matched replacement control
For each model-selected replacement, compute `replacement_ADP − faded_ADP`. Random controls are drawn from the **same faded-player-specific pool** and matched to the model replacement's shift bin. Exact ADP delta where feasible; otherwise the pre-specified bins:

`0–3 · 4–7 · 8–12 · 13–18 · 19+`

Results reported **within each bin**, not only pooled. A model replacement demonstrates edge only if it beats random replacements **at the same ADP distance**.

### 10.2 Candidate-set comparison
For every faded player, construct **one fixed eligible pool before scoring**, and apply that identical pool to both model and random policies. Model selection is evaluated against repeated random draws from that exact pool. Valid-swap and exclusion rules must be identical across arms — enforced by `assert_shared_candidate_pool()` and covered by guard test 7.

### 10.3 Matched fade-selection control
Random control fades are drawn from the **same season** and exact-or-tightly-binned ADP range as each model fade, and additionally **matched on number of available replacements** — model fades currently show a **0.50 exclusion rate vs 0.32** for random fades, so replacement opportunity is not currently held constant. Both arms use the same replacement rule when testing the fade leg. This isolates whether the model finds unusually poor picks *at the same market price and the same practical replacement opportunity*.

### 10.4 Exactly three reported quantities
1. **Fade-selection value** — controlling for ADP and candidate-pool size.
2. **Replacement-selection value** — controlling for ADP shift and exact candidate pool.
3. **Full-policy value** — under both matched controls.

If any component loses significance under these controls, that is stated directly.

**The current +16.6 replacement result must not be cited as evidence of a replacement edge.** It is presently explained by the model selecting closer-to-ADP alternatives (+8 slots vs +19–22 for random).

---

## 10.5 Policy Feasibility — audit run BEFORE the confirmatory test

`wr_bust_policy_feasibility.py` — structure only. **No VOR, bust rate, or policy contrast is computed**, so inspecting it cannot influence the confirmatory result. Provenance (git commit, package versions, input SHA-256, seeds, timestamp, command) recorded in `feasibility_metadata.json`.

### Attrition through the matched design

| | CORE | FULL |
|---|---|---|
| eligible WRs / season | 53–63 | 53–63 |
| fade candidates (15%) | 89 | 62 |
| after tight ADP-range match (±6) | 86 (97%) | 62 (100%) |
| **after EXACT opportunity-count match** | **22 (25%)** | **11 (18%)** |
| effective seasons retained | **8 of 10** | **5 of 7** |
| matched swaps / season (median) | 2.5 (min 0, max 5) | 2.0 (min 0, max 3) |
| season concentration (top 3) | 50% | **73%** |

### Legal replacements by ADP-shift bin

| bin | CORE fades with ≥1 | share of all legal slots |
|---|---|---|
| 0–3 | 73% | **9%** |
| 4–7 | 65% | 9% |
| 8–12 | 73% | 10% |
| 13–18 | 67% | 11% |
| **19+** | 60% | **62%** |

Near-ADP bins — the ones that matter most for a realistic replacement — hold only ~9% of legal slots each. Bin-stratified estimates would be extremely thin exactly where they are most informative.

### Exact opportunity-count matching is structurally degenerate

**75% of CORE fades (67 of 89) and 82% of FULL fades have ZERO matched controls.** This is not bad luck; it follows from the eligibility rule:

**Under the current candidate-pool construction — where eligible replacements are all WRs with ADP at or later than the faded player — replacement opportunity count is a deterministic function of ADP rank. Exact matching on both ADP rank and opportunity count therefore reduces to exact ADP-rank matching and retains only tied-ADP observations.**

Concretely: a player at ADP-rank *k* of *n* has exactly *n − k* eligible replacements, and ADP ranks are unique within a season, so a control can exist only where two players share an identical ADP value.

**This is a property of the current policy design, not a universal property of fantasy-draft decision settings.** A different candidate-pool construction — for example a fixed choice set at a simulated draft slot, or a window bounded on both sides — would not induce this collinearity. The finding constrains *this* matched design; it does not generalise to policy evaluation as such.

Verified directly: the 22 CORE matches are all ADP ties (`52.3/52.3`, `105.3/105.3`, …). They are not a random subsample of fades — they are specifically the duplicate-ADP subgroup, which is both tiny and unrepresentative.

### Verdict, per the preregistered interpretive rule

> If the matched sample is too sparse or too concentrated to support a stable season-blocked estimate, the policy phase will be reported as **infeasible / underpowered** rather than as evidence for or against a draft-policy edge.

**The confirmatory policy test as specified is INFEASIBLE.** 22 matched swaps across 8 seasons (median 2.5/season, two seasons contributing zero), drawn entirely from ADP ties, cannot support a stable season-blocked estimate. No sample-size cutoff was invented after the fact — the design admits almost no controls by construction.

**This says nothing about whether a policy edge exists.** It says this particular matched design cannot test for one.

### Status of coarsened matching — exploratory only

Coarse opportunity-count bins (1 / 2–3 / 4–6 / 7+) exist and remain heavily concentrated (CORE: `7+` holds 56 of 89 fades, 63%). Their status is fixed:

- Coarsened opportunity-count matching is **exploratory only**.
- It is **not a relaxation or rescue** of the preregistered confirmatory design.
- **No sensitivity result may be cited as policy validation.**

A coarsened analysis was not run. Even if it were, it could not convert an infeasible confirmatory test into a validated one.

---

## 12. Future research design — OUT OF SCOPE for this study

Recorded for continuity only. **Not authorised, not designed, not run.** Nothing below is part of the completed study, and none of it may be cited as a result.

> A future policy study would require a separately preregistered **choice-set design**: define the realistic candidate set available at each simulated draft slot; compare an ADP-only choice rule with an ADP-plus-model choice rule within the identical candidate set; condition on draft slot or ADP range rather than matching on the downstream opportunity-count variable; and pre-specify roster constraints, choice rules, and outcomes.

Conditioning on draft slot rather than matching on opportunity count is what avoids the collinearity documented in §10.5 — opportunity count is downstream of ADP rank under the current pool construction, whereas a fixed choice set at a slot is not.

---

## 11. Guard tests

`test_wr_bust_guards.py` — standalone, stdlib `unittest`, no pytest dependency:

```bash
cd research/validation_v1 && python -m unittest discover -v
```

**14 tests, all passing.** Each deliberately constructs a violation and asserts a loud failure:

| # | violation | guard |
|---|---|---|
| 1 | raw feature all-missing in training | `assert_fold_contract` |
| 2 | transformed schema mismatch across folds | `assert_fold_contract` |
| 3 | NaN imputer statistic (the v1 failure mode) | `assert_fold_contract` |
| 4 | train/test player-season overlap | `assert_no_train_test_overlap` |
| 5 | continuous ADP absent from a nested arm | `assert_continuous_adp_present` |
| 6 | replacement ADP earlier than faded ADP | `eligible_pool` |
| 7 | divergent candidate universes | `assert_shared_candidate_pool` |
| 8 | non-determinism under fixed seed | `run_policy` |

Plus negative controls: clean folds pass, disjoint seasons pass, ADP bands are rejected as a substitute for continuous ADP, tail players correctly yield no legal replacement, and different seeds genuinely differ (so "deterministic" cannot be satisfied by a constant).
