# Player Archetype Engine — Design Document

Purpose: engineer **predictive features** that help the model identify players who outperform projections and ADP. Not user-facing labels. Every archetype here must be quantifiable, auto-assignable from historical data, reproducible across eras, and computable **before** the season from prior-season/known-now data only.

Status: design only. Nothing here is validated yet. Confidence ratings are priors to be tested against the existing walk-forward harness (`research/validation_v1/validation_utils.py`), not claims.

---

## 0. The organizing principle

Most fantasy "archetypes" are descriptive vocabulary. The ones with predictive value all reduce to one of three mechanisms:

1. **Volume persistence.** Opportunity (snaps, routes, carries, targets) is far stickier year-over-year than efficiency (yards/target, TD rate). An archetype is predictive when it identifies *durable opportunity* the market underprices.
2. **Efficiency mean-reversion.** Archetypes built on unsustainable rate stats predict *negative* regression. This is as valuable as finding risers, and less crowded.
3. **Role-change leading indicators.** Vacated volume, depth-chart displacement, draft capital, and late-season role trajectory foreshadow opportunity shifts the market prices slowly.

An archetype that doesn't map to one of these three is probably descriptive. That's the filter I applied below.

**The single most important framing:** because we score against an ADP baseline, an archetype only helps if it is *systematically mispriced by the market*. "Bellcow RBs are good" is true and useless — ADP already knows. "Bellcow RBs whose prior-season TD rate was unsustainably low" is a claim about mispricing. Every hypothesis below is written as a mispricing claim.

---

## 1. Data we already have vs. what these need

Built and verified this session (`predraft_validation_dataset_workload_v1.csv`):

| Available now | Column |
|---|---|
| Snap share (2012+) | `prior_snap_share` |
| Red zone target/carry share | `prior_redzone_target_share`, `prior_redzone_carry_share` |
| Air yards + share | `prior_air_yards`, `prior_air_yards_share` |
| WOPR | `prior_wopr` |
| Draft capital | `draft_round`, `draft_pick_overall`, `years_since_drafted` |
| Team vacated volume | `prior_team_vacated_target_share`, `prior_team_vacated_carry_share` |
| Prior production | `prior_targets`, `prior_carries`, `prior_receiving_yards`, `prior_target_share`, etc. |
| Age, ADP | `age`, `overall_adp`, `positional_adp` |

Needed but **not yet ingested** (blocks several archetypes below):

- Route participation (routes run / team dropbacks) — the single highest-value missing input for all pass-catcher archetypes
- aDOT (average depth of target) — derivable from existing pbp pull, not yet extracted
- Target/carry **counts** per game, not just season totals (for rate-vs-volume separation)
- Depth-chart competition (partially — `depth_chart_competition` is still a stub column)
- Offensive line quality, coaching/scheme continuity

---

## 2. Running Back archetypes

### RB-1. Three-Down Bellcow

**Description.** Player absorbs a dominant share of his team's backfield snaps, carries, *and* passing-down work.

**Why it exists.** Coaching staffs concentrate touches in a back they trust in pass protection; pass-pro competence is the gate that separates two-down backs from three-down backs.

**Predictive hypothesis.** Not "bellcows are good" (ADP knows). The mispricing is in **entry and exit**: the market is slow to price a back *becoming* a bellcow (mid-season role consolidation shows up in second-half snap share before it shows up in next-year ADP), and slow to price one *losing* it. Late-season snap-share trajectory should beat full-season average as a predictor.

**Identifying variables.** `prior_snap_share` (>0.65), carry share (>0.60), target share among RBs (>0.15), `prior_redzone_carry_share` (>0.50).

**Effects.** Median ↑↑ · Ceiling ↑↑ · Floor ↑↑ · Volatility ↓ (volume smooths weekly variance) · Injury risk ↑ (workload accumulation, and injury is *the* bellcow failure mode) · P(beat ADP) ≈ neutral at market price, ↑↑ when acquired below RB1 ADP.

**Career arc.** Typically emerges years 1–3, holds 3–5 years, degrades sharply — RB aging cliffs are real and steep relative to other positions. The existing `case_studies/output/rb_elite_age_study.md` in this repo already found RB Top-5 finishes peak at age 23 and Top-3 at 22.

**Examples.** Emmitt Smith (90s) · LaDainian Tomlinson, Shaun Alexander (2000s) · Adrian Peterson, Le'Veon Bell (2010s) · Christian McCaffrey, Derrick Henry (2020s).

**Team-context interactions.** Requires a competent offensive line and a coaching staff that doesn't rotate by philosophy. A bellcow behind a bad line retains volume (good) but loses efficiency (less bad than it looks — volume is what pays). Offense that trails often → more passing-down value.

**Where it breaks down.** New coaching staff (committee philosophy imports), a drafted Day-2 RB, or a team signing a passing-down specialist. Also breaks on the injury tail: the archetype's median is great and its 10th percentile is catastrophic.

**Do NOT use (leakage).** Same-season snaps/carries/targets, same-season games played, same-season team pass rate.

**Feature engineering.** Snap share *trajectory* (final 4 games vs. season average) as a separate feature from level. Carry-share-minus-target-share as a "three-down-ness" axis. Interaction: `snap_share × (1 − prior_team_vacated_carry_share)` to distinguish "earned role" from "role by default."

**Confidence: Medium-High** for the trajectory/entry-exit version. **Low** for the naive level version (ADP prices it).

---

### RB-2. Passing-Down / Receiving Specialist

**Description.** Low carry share, disproportionate target share, high third-down and two-minute snap rate.

**Why it exists.** Pass protection and route-running are separable skills from between-tackles rushing; teams roster for the split.

**Predictive hypothesis.** Systematically **underpriced in PPR relative to standard**, because ADP is mostly set by half-PPR/standard consensus and casual perception keys on rushing yardage. The archetype's value is also *injury-contingent*: it converts to a bellcow role on a starter injury more often than a two-down back does.

**Identifying variables.** Target share among RBs (>0.20), carry share (<0.40), `prior_snap_share` moderate (0.35–0.60), reception-to-carry ratio.

**Effects.** Median ↑ (PPR) · Ceiling ↔ (capped without goal-line work) · Floor ↑↑ (receptions are the most stable fantasy input) · Volatility ↓↓ · Injury risk ↓ (fewer collisions between tackles) · P(beat ADP) ↑ in full PPR.

**Career arc.** Longer than bellcows — lower collision volume. Often peaks later (age 26–29) and declines gently.

**Examples.** Marshall Faulk (as the elite hybrid), Darren Sproles (2000s–10s) · Danny Woodhead, James White (2010s) · Austin Ekeler, Alvin Kamara (2020s).

**Team-context interactions.** Enormous QB dependency — a checkdown-prone QB or a bad offensive line inflates RB targets. Negative game script (bad team, trailing often) *helps* this archetype, which is the inverse of RB-1.

**Where it breaks down.** Scoring format (worthless in standard relative to price). Also collapses if the team drafts a three-down back.

**Do NOT use.** Same-season receptions/targets/game script.

**Feature engineering.** Interaction with `prior_team_pass_attempts` and team pass rate over expected. Explicit `format_is_ppr` interaction term.

**Confidence: Medium-High** — mechanism is clear and the format-mispricing is plausible and testable.

---

### RB-3. Early-Down Grinder (Two-Down Back)

**Description.** High carry share, near-zero target share, exits on obvious passing downs.

**Predictive hypothesis.** **Systematically overpriced**, because raw rushing volume looks like bellcow volume in box scores while carrying much worse PPR value and much higher TD-dependence. This is a *negative* signal archetype — arguably more useful than a positive one, since fade candidates are less contested.

**Identifying variables.** Carry share >0.55, target share among RBs <0.10, `prior_snap_share` <0.55 despite high carry share (the tell: touches concentrated into few snaps).

**Effects.** Median ↔ · Ceiling ↓ · Floor ↓↓ (a 12-carry, 40-yard, no-TD game is a real weekly outcome) · Volatility ↑↑ (TD-dependent) · Injury risk ↑ · P(beat ADP) ↓.

**Examples.** Jamal Lewis (2000s) · Michael Turner, Jordan Howard (2010s) · Sony Michel, early-career Rhamondre Stevenson.

**Where it breaks down.** Elite talent overrides it — Derrick Henry was a two-down back who produced RB1 seasons on sheer volume and TD equity. Guard against by interacting with team run rate and goal-line share.

**Confidence: Medium** as a fade signal; the Henry-style exception is real and needs the interaction term.

---

### RB-4. Contingent-Volume Handcuff

**Description.** Currently low-role back with a credible path to bellcow work if the starter misses time.

**Predictive hypothesis.** The market prices the *current* role; the true value is `P(starter misses time) × value_of_inherited_role`. Both terms are estimable pre-draft — the first from the starter's durability history (the Step 3b injury feature), the second from team backfield structure. This is one of the cleanest genuinely-mispriced archetypes because the payoff is bimodal and ADP is a mean.

**Identifying variables.** Low `prior_snap_share` (<0.30) but meaningful `draft_pick_overall`, teammate's durability score, backfield depth count, `prior_team_vacated_carry_share`.

**Effects.** Median ↓ · Ceiling ↑↑ · Floor ↓↓ · Volatility ↑↑ · P(beat ADP) ↑↑ (cheap price, fat right tail).

**Examples.** Jordan Howard (2016, after Langford injury) · Phillip Lindsay (2018) · James Robinson (2020) · Jaylen Warren, Chuba Hubbard-type ascensions (2020s).

**Where it breaks down.** Teams that sign a veteran free agent after the fantasy draft. Also fails when the "handcuff" is a committee-mate rather than a true next-man-up.

**Feature engineering.** This is the archetype that most needs the Step 3b injury/durability feature — build `teammate_starter_durability_penalty` as an explicit input. **Requires a same-team join, which the current dataset supports via the `team` column added this session.**

**Confidence: High** on mechanism, **Medium** on execution — depends entirely on getting the teammate-linkage and injury-history features right.

---

## 3. Wide Receiver archetypes

### WR-1. Alpha X / True Number One

**Description.** Dominant target share, dominant air-yards share, high WOPR.

**Predictive hypothesis.** ADP prices this well at the top. The exploitable version is **WOPR-vs-ADP divergence**: a receiver with alpha-level opportunity metrics but non-alpha ADP (usually because his prior-season *efficiency* or team quality was poor) is the highest-value target in the pool. Opportunity persists; efficiency and team quality regress.

**Identifying variables.** `prior_target_share` >0.25, `prior_air_yards_share` >0.30, `prior_wopr` >0.60, `prior_redzone_target_share` >0.20.

**Effects.** Median ↑↑ · Ceiling ↑↑ · Floor ↑ · Volatility ↔ · Injury risk ↔ · P(beat ADP) ↑↑ *only in the divergence case*.

**Career arc.** WR breakouts cluster years 2–3; the existing repo age study found WR Top-12 peaks at age 25, Top-5 at 26 — later and flatter than RB. Alphas hold value into early 30s.

**Examples.** Jerry Rice (80s–90s) · Marvin Harrison, Torry Holt (2000s) · Calvin Johnson, Julio Jones, Antonio Brown (2010s) · Davante Adams, Justin Jefferson, Ja'Marr Chase (2020s).

**Team-context interactions.** QB quality is the dominant modifier and is *partially* priced. Target competition (a second alpha, or a target-hog TE) caps the ceiling.

**Where it breaks down.** QB downgrade, or offense that concentrates targets in the slot/TE. Also breaks when a team adds a high-draft-capital rookie WR.

**Do NOT use.** Same-season targets/receptions/air yards/team pass attempts.

**Feature engineering.** `prior_wopr` percentile-within-position **minus** ADP percentile-within-position. This single divergence feature is my highest-conviction candidate in this document.

**Confidence: High** for the divergence formulation.

---

### WR-2. Deep Threat / Field Stretcher

**Description.** High aDOT, low catch rate, TD- and explosive-play dependent.

**Predictive hypothesis.** **Overpriced after a high-TD season, underpriced after a low-TD season** — deep TD rate is among the least stable stats in football. Also structurally higher weekly variance than ADP (a mean-based price) reflects, which matters more for best-ball/DFS than season-long.

**Identifying variables.** aDOT >14 (needs extraction from the pbp pull), catch rate <0.58, high `prior_air_yards_share` relative to modest `prior_target_share` (the tell: few targets, lots of air yards).

**Effects.** Median ↔ · Ceiling ↑↑ · Floor ↓↓ · Volatility ↑↑ · Injury risk ↔ (soft-tissue risk somewhat elevated) · P(beat ADP) depends entirely on prior-year TD luck.

**Examples.** Randy Moss (elite outlier) · DeSean Jackson, Ted Ginn (2000s–10s) · Will Fuller, Marquez Valdes-Scantling, Tyquan Thornton-type roles (2020s).

**Feature engineering.** Explicit **TD-regression flag**: prior-season TDs minus expected TDs given air yards and red-zone targets. High positive residual → fade.

**Confidence: Medium-High** — TD mean-reversion is one of the best-supported effects in football analytics.

---

### WR-3. Possession / Slot Technician

**Description.** High target volume, low aDOT, high catch rate, heavy slot alignment.

**Predictive hypothesis.** **Underpriced in full PPR, overpriced in standard**, same format-mispricing mechanism as RB-2. Reception volume is the most stable week-to-week fantasy input in the sport, giving this archetype the best floor-per-dollar in the pool.

**Identifying variables.** Catch rate >0.70, aDOT <9, `prior_target_share` >0.20 with *low* `prior_air_yards_share` (the inverse tell of WR-2).

**Effects.** Median ↑ · Ceiling ↓ · Floor ↑↑ · Volatility ↓↓ · Injury risk ↔ · P(beat ADP) ↑ in PPR.

**Examples.** Wes Welker (2000s–10s) · Julian Edelman, Jarvis Landry (2010s) · Amon-Ra St. Brown, Hunter Renfrow (2020s).

**Where it breaks down.** Ceiling is genuinely capped without red-zone or deep work — do not expect league-winning upside. Slot-heavy players are also more scheme-dependent (a new OC can move them or bench them).

**Confidence: Medium** — real and stable, but the low ceiling limits how much edge is available.

---

### WR-4. Post-Hype Third-Year Ascender

**Description.** High draft capital, disappointing years 1–2, still young, role trending up late in year 2.

**Predictive hypothesis.** The market overreacts to two years of counting-stat disappointment and underweights retained draft capital + age. Draft capital is a *team-commitment* signal that persists — teams keep giving early-round picks opportunity. This is a genuine ADP-fade-then-buy pattern.

**Identifying variables.** `draft_round` ≤2, `years_since_drafted` == 2, `age` ≤24, positive second-half snap/route trajectory in year 2, ADP outside the top 40 at position.

**Effects.** Median ↔ · Ceiling ↑↑ · Floor ↓ · P(beat ADP) ↑↑ (cheap by construction).

**Examples.** Historically the "year-three WR breakout" heuristic; modern examples include several 2nd-round WRs who consolidated roles in year 3 after quiet starts.

**Feature engineering.** Interaction: `draft_capital_percentile × (1 − prior_production_percentile) × young_age_flag`. Explicitly the *conjunction* of high investment and low output.

**Confidence: Medium** — plausible and cheap to test, but the classic "third-year breakout" rule has weakened as rookie WRs contribute earlier in the modern game. Test, don't assume.

---

## 4. Tight End archetypes

### TE-1. Move TE / De-Facto Slot Receiver

**Description.** Deployed as a receiver rather than an in-line blocker; route participation approaching WR levels.

**Predictive hypothesis.** TE scoring is extremely top-heavy, so the binary question "is this TE actually a receiver?" carries more information than any gradation within it. Route participation is the cleanest separator and is **not** in our data yet — this is the strongest argument for prioritizing route-participation ingestion.

**Identifying variables.** Route participation >0.70 (missing), `prior_target_share` >0.18, `prior_snap_share` >0.75, `prior_air_yards_share` meaningful.

**Effects.** Median ↑↑ · Ceiling ↑↑ · Floor ↑ · Volatility ↓ · P(beat ADP) ↑ at the position's steep scarcity curve.

**Examples.** Tony Gonzalez, Antonio Gates (2000s) · Rob Gronkowski, Jimmy Graham (2010s) · Travis Kelce, George Kittle, Mark Andrews (2020s).

**Confidence: High** on the mechanism, but **blocked** — needs route participation to separate cleanly from TE-2.

---

### TE-2. In-Line Blocking TE — **fade signal**

High snap share, low target share, low air yards. Snap share alone would misclassify these as valuable; the target-share-per-snap ratio is the discriminator. **Confidence: High** (as a negative filter).

---

### TE-3. Rookie TE — **structural fade**

Rookie TEs historically produce very poorly relative to draft capital, because the position has the steepest NFL learning curve (blocking assignments + route nuance). Encode as `position == TE AND years_since_drafted == 0`. Recent exceptions (Bowers, LaPorta, Pitts-level draft capital) suggest the effect may be weakening — **test whether the era interaction is real** rather than assuming the old rule holds. **Confidence: Medium-High**, with an explicit era-drift caveat.

---

## 5. Quarterback archetypes

### QB-1. Designed-Rushing QB ("Konami Code")

**Description.** Meaningful designed-run and scramble volume on top of passing production.

**Predictive hypothesis.** Rushing yards and especially rushing TDs are worth far more per-unit than passing in standard fantasy scoring (typically 4pt pass TD vs 6pt rush TD, 25 pass yds/pt vs 10 rush yds/pt). Rushing volume is *also* stickier than passing efficiency. The market has largely caught up to this — so the residual edge is in the **second tier**: mid-ADP QBs with 40+ rush attempts, not the obvious elite ones.

**Identifying variables.** Prior rush attempts >60, `prior_redzone_carry_share` non-trivial for a QB, prior rushing TDs ≥4.

**Effects.** Median ↑↑ · Ceiling ↑↑ · Floor ↑↑ (rushing floor is why this archetype dominates) · Volatility ↓ · Injury risk ↑ (contact exposure) · P(beat ADP) ↑ in the second tier only.

**Examples.** Randall Cunningham (80s–90s) · Michael Vick (2000s) · Cam Newton, Russell Wilson (2010s) · Lamar Jackson, Josh Allen, Jalen Hurts (2020s).

**Where it breaks down.** Coaching change to a pocket-oriented scheme; a QB who "stops running" after an injury scare. Also: rushing-QB injury risk is real and the archetype's floor advantage assumes availability.

**Confidence: High** on the mechanism, **Medium** on remaining edge (heavily priced at the top).

---

### QB-2. High-Volume Pocket Passer

Passing attempts are the input; attempts correlate with *negative* game script (trailing teams throw). Slight contrarian implication: a good QB on a mediocre defense is a better fantasy asset than the same QB on a great one. **Confidence: Medium.**

### QB-3. Efficient Low-Volume Game Manager — **fade signal**

Good real-football QB, poor fantasy QB — low attempts, no rushing, TD-dependent. **Confidence: Medium-High** as a negative filter.

---

## 6. Proposed NEW archetypes (less commonly discussed)

These are the ones I'd actually prioritize testing, because they're mechanism-driven and less likely to be already priced.

### N-1. Volume-Rich / Efficiency-Poor "Regression Buy" ⭐

**Hypothesis.** Volume persists; efficiency reverts. A player in the top quartile of opportunity (WOPR, snap share, carry share) and the bottom quartile of efficiency (yards per touch, TD rate, catch rate over expected) had his *counting stats* suppressed by variance that will not repeat — but ADP is set largely off counting stats. This should be a systematic, repeatable buy.

**Variables.** Opportunity percentile − efficiency percentile, within position and season.
**Confidence: High.** This is the single strongest candidate here — it's a direct application of the best-established regression finding in football analytics, and it's *specifically* a mispricing claim rather than a quality claim.

### N-2. Efficiency-Rich / Volume-Poor "Regression Fade" ⭐

The mirror image. Elite per-touch numbers on modest opportunity, priced up on last year's highlights. **Confidence: High**, same mechanism, and fade candidates are less contested than buys.

### N-3. Vacuum Inheritor

Team lost a large share of prior-season targets/carries (`prior_team_vacated_*`, built this session) **and** did not replace it with high draft capital or a notable free agent. Existing incumbents inherit by default. Distinct from a generic "good situation" — it's specifically *uncontested* vacated volume. **Confidence: Medium-High**; data exists now, cheap to test.

### N-4. Late-Season Role Consolidator

Snap/route/target share in the final 4–5 games materially exceeds the full-season average. The market anchors on full-season totals; the late-season role is the better estimate of the *entering* role. **Note: this repo already has `build_late_season_role_growth_score_v1.py`** — that work should be revived and folded in rather than rebuilt. **Confidence: Medium-High.**

### N-5. Displaced Alpha (Involuntary Team Change)

A player who changes teams via free agency/trade *into* an uncontested target/carry vacuum, versus one who changes into a crowded room. Team-change is visible pre-draft; the market prices the *name*, and prices the *new situation* slowly. **Confidence: Medium.**

### N-6. Age-Cliff Straddler

Position-specific age thresholds interacted with workload accumulation (career touches), not age alone. An RB at 28 with 1,800 career touches is a different asset than one at 28 with 900. Career-touch accumulation is computable from the historical data we now have. **Confidence: Medium-High** for RB specifically; weak for WR/QB.

### N-7. Red-Zone Share vs. Overall Share Divergence

A player whose red-zone share substantially exceeds his overall share is TD-leveraged (higher ceiling, higher variance, and a TD-regression risk); the inverse is TD-starved and a positive-regression candidate. We built both inputs this session. **Confidence: Medium-High.**

---

## 7. Hand-engineered vs. clustered vs. supervised

**Hand-engineer (do these first):** every ratio and divergence feature above — WOPR-minus-ADP percentile, opportunity-minus-efficiency percentile, red-zone-share-minus-overall-share, snap-share trajectory, career-touch accumulation. Rationale: they're interpretable, directly encode the hypothesized mechanism, are stable across eras, and cost nothing to compute. **Most of the available signal is probably here, not in the clustering.**

**Learn by clustering (soft membership only):** the multi-dimensional role types where boundaries are genuinely fuzzy — RB three-down-ness, WR alignment/route-tree type, TE receiver-vs-blocker. A player is rarely purely one archetype; hard labels destroy that information.

**Learn supervised (let the model do it):** all interactions between archetype and team context. Do **not** hand-build "bellcow × good O-line × pass-heavy offense" terms — that's what gradient boosting is for. Hand-build the *axes*, let the model find the *interactions*.

---

## 8. Method selection

Honest constraint first: **this is small data.** ~15k player-seasons total, but only ~2,700 have real ADP, and per-position-per-season the modeling sample is a few hundred rows. That single fact rules out most of the fancier options.

| Method | Verdict for this dataset |
|---|---|
| **K-Means** | Weak fit. Assumes spherical, equal-variance clusters and forces hard assignment. Fantasy roles are continuous — hard labels throw away exactly the gradation that carries signal. Useful only as a fast sanity baseline. |
| **Gaussian Mixture Models** | **Recommended.** Soft membership is the right representation: a back can be 0.7 bellcow / 0.3 passing-down, and those probabilities become continuous features. Handles elliptical/unequal-variance clusters. Cheap at this sample size. |
| **Hierarchical clustering** | Useful for *exploration* — dendrogram to decide how many archetypes actually exist per position. Not for production features. |
| **DBSCAN** | Poor fit. Fantasy role-space has no clean density separation, and DBSCAN's noise label (−1) is awkward as a model feature. Skip. |
| **PCA** | **Recommended as preprocessing.** Snap share, target share, air-yards share, and WOPR are heavily collinear; PCA decorrelates them into 3–5 interpretable axes (roughly: total opportunity, receiving-vs-rushing tilt, depth-of-usage). The components are usable features in their own right. |
| **UMAP** | Not for production. Excellent for visual exploration, but the embedding is unstable across refits, `transform()` on new seasons is fragile, and non-linear components can't be reasoned about or audited. The interpretability loss isn't worth it here. |
| **Self-Organizing Maps** | Skip. Adds complexity and hyperparameters over GMM with no clear gain at this sample size, and it's a less-maintained tool ecosystem. |

### Recommended pipeline

1. Hand-engineer the ratio/divergence features (Section 7) — **and validate these alone first.** If they don't add lift, clustering on top of them won't save it.
2. PCA the correlated workload block → 3–5 components per position, retaining ~90% variance.
3. Fit a **GMM** per position on those components; take **posterior membership probabilities** as features (not the argmax label).
4. Feed hand-engineered features + PCA components + GMM probabilities into the existing supervised harness.
5. Ablate: ADP-only → +hand-engineered → +PCA → +GMM. Keep only what earns its place on lift-over-ADP.

### The critical methodological trap

**Fit PCA and the GMM inside each walk-forward fold, on training seasons only, then `transform()` the test season.** Fitting the cluster definitions on all seasons — including the test season — leaks future information into the archetype boundaries themselves. It won't look like leakage (no outcome variable is touched) and it will quietly inflate every result. This is the most likely way this whole effort produces a fake positive, so it should be enforced in code, not just remembered.

Two supporting guardrails:
- **Cluster count via training-fold-only BIC**, not by eyeballing all-season structure.
- **Archetype stability check across eras** — refit per decade and confirm cluster centroids are recognizably similar. If 2005 archetypes don't resemble 2020 archetypes, the archetype isn't reproducible and shouldn't be a feature (the NFL's structural pass-rate shift is a real confound here).

---

## 9. Priority order

1. **N-1 / N-2 (opportunity-vs-efficiency divergence)** — highest conviction, uses only data we already have.
2. **WR-1 divergence (WOPR percentile − ADP percentile)** — one feature, directly a mispricing measure.
3. **N-7 (red-zone vs. overall share)** — inputs built this session.
4. **N-3 (vacuum inheritor)** — inputs built this session.
5. **Route participation ingestion** — unblocks TE-1/TE-2 separation and sharpens every WR archetype. Highest-value *missing* data.
6. **N-4 (late-season consolidation)** — revive the existing `build_late_season_role_growth_score_v1.py` rather than rebuilding.
7. **GMM soft-membership layer** — only after 1–4 have been ablated individually.

Everything above is a hypothesis. The harness in `research/validation_v1/` already answers "did this beat ADP" honestly; the discipline is to let it kill the ones that don't.
