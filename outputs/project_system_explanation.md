# Guaranteed Play System Explanation

This is a plain-English explanation of how the current Guaranteed Play platform works. It describes the app as it exists now, not a proposed rebuild.

## 1. What Guaranteed Play Currently Does

- Guaranteed Play is a Streamlit fantasy football draft assistant.
- It loads a local player pool, removes drafted players, and updates recommendations as the draft state changes.
- It gives a recommended pick based on projection value, roster need, ADP value, tier urgency, team fit, roster construction pressure, and data-trust guardrails.
- It shows player cards with a quick draft case: grade, upside, risk, value, durability, fit reasons, and watch points.
- It tracks the user's roster, draft pick number, draft slot, league size, roster settings, and drafted-player list in Streamlit session state.
- It estimates whether a player will survive to the user's next pick using ADP-weighted Monte Carlo draft simulations.
- It estimates championship impact for top candidates with a fast simulation layer that models floor, median, ceiling, injury risk, age risk, roster shape, and league-winning upside.
- It includes tier and scarcity tools that highlight positions where waiting may cause a projection drop.
- It still contains older rankings/comparison helpers in `utils.py`, which power some pages but do not use the newer master recommendation engine.
- It also contains research outputs for age curves, WR opportunity, and WR/RB underpriced studies. Those are research-only unless explicitly wired into the app.

## 2. Main App Pages

### Home

- Purpose: landing/control page.
- Data used: no player calculations directly; initializes shared session state.
- Calculations: none beyond sidebar league settings.
- User sees: title, short instruction, and the shared league-settings sidebar.
- Status: current but simple.

### Draft Mode

- Purpose: main draft command center.
- Data used: player pool from `data/processed/master_players.csv` if present; otherwise fallback CSV/SQLite paths. Uses current drafted players and roster settings from session state.
- Calculations: master recommendations, conviction, availability probability, optional future-impact simulations, championship equity V2, consensus score, roster needs, and best available board.
- User sees: recommended pick, championship impact, availability advice, confidence, model score, reasons, risks, alternatives, draft plan, roster snapshot, draft status, and draft board.
- Status: current primary app surface. Some advanced simulations are intentionally cached/limited for speed.

### Player Cards

- Purpose: scout one available player and understand the draft case.
- Data used: available players plus recommendation rows from the master recommendation engine. It can also reuse latest championship equity results already stored by Draft Mode.
- Calculations: recommendation grade, fit label, value label, risk label, upside/value/durability grades, decision-card reasons, and advanced score breakdown.
- User sees: player hero, recommendation grade, championship impact if available, reasons to draft, player profile, best fits, avoid-if notes, and draft buttons.
- Status: current, but championship impact may be missing unless Draft Mode has already generated it in the same session.

### Team Outlook

- Purpose: compare the user's roster to estimated leaguemate rosters.
- Data used: local player pool, my team, drafted non-my-team players, roster settings, and team profile from `draft_analysis`.
- Calculations: total projection, starter projection, safe/risk/boom-bust/depth scores, estimated opponent rosters, and ranks versus league estimate.
- User sees: roster strength metrics, league comparison table, starter projection chart, roster construction, profile read, current roster, and remove buttons.
- Status: useful directional page. Opponent teams are estimated, not imported live rosters.

### Player Compare

- Purpose: compare 2-4 available players.
- Data used: older `utils.py` data loader from `guaranteed_play.db`, not the newer `draftkit.data_access` loader.
- Calculations: old Best Fit Score, Best Value Score, and Best Available Score.
- User sees: comparison table, simple cards, and draft buttons.
- Status: needs review. This page uses the older scoring layer and may disagree with Draft Mode.

### Live Rankings

- Purpose: show a sortable live board with quick draft actions.
- Data used: older `utils.py` data loader from `guaranteed_play.db`.
- Calculations: old Best Fit Score, Best Value Score, and Best Available Score.
- User sees: current pick, players drafted, filters, rankings table, and draft buttons.
- Status: needs review. This page is app-facing but not aligned with the newer master recommendation engine.

### Tier Desperation

- Purpose: show tier cliffs, scarcity, fall-off risk, and priority targets.
- Data used: current available players from the newer `draftkit.data_access` loader.
- Calculations: position tiers, tier summary, tier warning, fall-off by next turn/manual wait, scarcity, desperation score, and recommendation rankings.
- User sees: tier warnings, best overall pick, best desperation target, priority targets, tiers, recommendations, fall-off, and scarcity tables.
- Status: current diagnostic/support page. It is useful, but "desperation" should be treated as a supporting signal rather than the main draft answer.

### Component Audit

- Purpose: diagnose what features are driving the top-100 recommendation pool.
- Data used: master recommendations, old championship equity, and availability simulation.
- Calculations: estimated feature influence, correlation to final score, variance share, missing/unsupported feature inputs.
- User sees: feature influence ranking, variance flags, unsupported features, and player contribution table.
- Status: research/diagnostic app page. Helpful for auditing, not a user-facing draft decision page.

## 3. Draft Mode

- Draft recommendations are generated by `build_recommendation_rankings_df()`, which calls the master recommendation engine.
- Inputs include available players, projections, ADP, position, team, injury risk, durability, archetype, roster settings, current roster, current pick, league size, and draft slot.
- Player value starts with position-adjusted value over replacement. Raw projection is compared with a position baseline based on league size, starting roster slots, and flex assumptions.
- The core master score blends five normalized components:
  - projection value: 30%
  - position need: 25%
  - ADP value: 15%
  - tier urgency: 15%
  - team fit: 15%
- Roster construction affects recommendations in two ways:
  - position need weights reward positions where the roster is short
  - construction pressure multiplies final scores for positions with high or severe roster needs
- RB/WR construction mandates exist for extreme imbalance, such as having several WRs and zero RBs, or several RBs and zero WRs.
- ADP value compares projection rank to ADP rank. A player drafted later than his projection rank receives a stronger value signal.
- Fall risk uses current pick, next-pick distance, and ADP to label whether a player is likely to make it back.
- Tier urgency looks at how many players remain in the current tier, how large the next tier drop is, roster need, and how far away the next pick is.
- Risk/upside are shown mostly through injury risk, durability, archetype, boom/bust/stability fields, safety score, and championship-equity upside.
- Availability is simulated with ADP-weighted Monte Carlo paths between the current pick and the user's next pick.
- Championship equity V2 is simulated for a small candidate set in Draft Mode for speed. It estimates the change in title probability after adding a player.
- Future impact simulations are optional behind a sidebar toggle because they are slower.
- Several results are cached with Streamlit cache keys based on current pick, roster, drafted players, league settings, simulation count, and seed.
- Potentially outdated pieces: older `utils.py` scoring pages, old championship equity used by Component Audit/Conviction fallback, and research-only outputs that are not part of the Draft Mode score.

## 4. Player Scores and Signals

### Consensus Score

- Source: `draftkit/recommendation_consensus.py`.
- Measures: the top-level pick score in Draft Mode.
- Rough calculation: weighted blend of base recommendation score, ADP/value score, roster fit, future-pick impact, championship equity, risk/safety, and survival probability.
- Higher is better.
- App-facing: yes, in Draft Mode.
- Status: trusted as the current decision layer, but still directional.

### Model Score / Final Score

- Source: `draftkit/draft_analysis.py`.
- Measures: the master recommendation score before/inside the consensus layer.
- Rough calculation: normalized projection value, position need, ADP value, tier urgency, and team fit are blended, then adjusted for construction pressure, data trust, and single-QB value.
- Higher is better.
- App-facing: yes, in Draft Mode, Player Cards, Tier Desperation, and draft board.
- Status: trusted core app score.

### Conviction Score

- Source: `draftkit/conviction.py`.
- Measures: how clearly the top recommendation separates from alternatives.
- Rough calculation: combines score gap, same-position gap, replacement gap, alternative strength, signal confidence, championship equity, and construction pressure.
- Higher is better.
- App-facing: yes, in Draft Mode recommendation details.
- Status: diagnostic/trusted as a confidence aid, not a standalone rank.

### Championship Equity

- Source: `draftkit/championship_equity_v2.py` in Draft Mode; older `draftkit/championship_equity.py` still exists.
- Measures: simulated chance that a roster becomes title-caliber after adding a candidate.
- Rough calculation: builds floor/median/ceiling player distributions using projection, volatility, injury risk, role uncertainty, and age risk, then simulates roster outcomes and converts points to playoff/advancement/championship probability.
- Higher is better.
- App-facing: yes, in Draft Mode and Player Cards when available.
- Status: trusted directionally, not literal odds.

### Equity Delta

- Source: `championship_equity_v2`.
- Measures: how much a candidate improves or hurts simulated championship equity versus the current roster baseline.
- Rough calculation: candidate equity minus current team equity.
- Higher is better.
- App-facing: yes.
- Status: trusted directionally.

### League-Winning Upside

- Source: `championship_equity_v2`.
- Measures: ceiling/portfolio value of a player.
- Rough calculation: combines ceiling-over-median, volatility, positional scarcity, roster position duplication, and late-value bonus.
- Higher is better.
- App-facing: yes, mostly in Draft Mode details and consensus categories.
- Status: directional.

### Construction Score / Construction Pressure

- Source: `draftkit/construction_pressure.py` and future-impact simulation.
- Measures: whether the roster is missing important positions.
- Rough calculation: compares current position counts to starter targets, applies round pressure, flex pressure, and RB/WR imbalance checks.
- Higher construction pressure means the roster needs that position more. Higher expected construction score means the resulting roster is healthier.
- App-facing: yes.
- Status: trusted and important.

### Value Score / ADP Value

- Source: `draftkit/draft_analysis.py`.
- Measures: whether a player is cheaper than his projected value.
- Rough calculation: projection rank is compared to ADP rank; positive delta means the player is being drafted later than his projection rank. Projection strength and current-pick cost also contribute.
- Higher is better.
- App-facing: yes.
- Status: trusted when ADP data is clean; guarded by signal trust checks.

### Upside Score / Boom Score

- Source: `data/processed/master_players.csv` and `draftkit/feature_engineering.py`.
- Measures: ceiling potential.
- Rough calculation: uses direct boom/ceiling fields if available; otherwise combines projection percentile and volatility.
- Higher is better.
- App-facing: partly. Player Cards translate it into an upside grade.
- Status: app-facing but less central than model/consensus score.

### Floor / Safety / Stability Score

- Source: `feature_engineering.py`, `recommendation_consensus.py`, and player data fields.
- Measures: downside insulation and dependability.
- Rough calculation: durability, inverse injury risk, inverse volatility, and archetype are combined into stability/safety.
- Higher is better.
- App-facing: partly. Player Cards and consensus risk use these ideas.
- Status: useful but partly inferred from imperfect data.

### Bust Risk / Injury Risk

- Source: player data and `feature_engineering.py`.
- Measures: downside, injury, or volatility risk.
- Rough calculation: uses direct injury/bust fields if available; otherwise infers from missed games, status, age, durability, and volatility.
- Lower risk is better. For safety scores, higher is better because risk is inverted.
- App-facing: yes, mostly through Player Cards and risk messages.
- Status: useful guardrail, but depends heavily on source data quality.

### Availability / Survival Probability

- Source: `draftkit/draft_simulation.py`.
- Measures: chance a player survives until the user's next pick.
- Rough calculation: simulates picks before the user's next turn using ADP-weighted randomness plus projection weighting.
- Higher survival probability means the user may be able to wait.
- App-facing: yes.
- Status: trusted directionally; sensitive to ADP and draft room behavior.

### Future Impact

- Source: `draftkit/draft_simulator.py`.
- Measures: what the board may look like after choosing a candidate now.
- Rough calculation: simulates opponent picks before the next turn, tracks surviving/lost top targets, expected roster value, expected championship equity, construction score, and future pick quality.
- Higher is better for expected roster value, championship equity, construction score, and future pick quality.
- App-facing: optional in Draft Mode.
- Status: experimental/slow; useful for advanced checks.

### Tier Urgency / Desperation Score

- Source: `draftkit/draft_analysis.py`.
- Measures: how urgent a position or specific tier is.
- Rough calculation: uses players left in tier, projected tier drop, wait distance, and roster need.
- Higher is more urgent.
- App-facing: yes, especially Tier Desperation and Draft Mode warnings.
- Status: trusted as a support signal.

### Signal Trust

- Source: `draftkit/signal_trust.py`.
- Measures: whether ADP, projection, sportsbook/market, and disagreement signals look reliable.
- Rough calculation: starts from 100 and penalizes missing data, out-of-range ranks, extreme ADP/projection gaps, poor coverage, and sportsbook provider issues.
- Higher is better.
- App-facing: mostly behind the scenes through score dampening and audit/debug outputs.
- Status: trusted guardrail.

### Market Disagreement / Hidden Value

- Source: `draftkit/market_disagreement.py`.
- Measures: whether projection/market rank likes a player more or less than public rank/ADP.
- Rough calculation: blends projection-rank gap and ADP-rank gap; confidence rises when more sources are present.
- Positive is hidden value; negative is overvalued.
- App-facing: not a main visible Draft Mode score.
- Status: diagnostic/experimental.

### Best Fit, Best Value, Best Available

- Source: older `utils.py`.
- Measures: older simplified rankings for Player Compare and Live Rankings.
- Rough calculation: normalized projection, inverse ADP, and simple roster need weights.
- Higher is better.
- App-facing: yes, but only on older pages.
- Status: stale/needs review because it can disagree with the master recommendation engine.

### WR Underpriced Signals

- Source: research outputs requested in the project brief; not wired into the current app.
- Measures: probability that a WR drafted in a target range outperforms into a higher fantasy tier.
- Rough calculation: historical underpriced-player modeling by draft range/tier and signal set.
- Higher is better.
- App-facing: no.
- Status: research-only. Separate WR signals are trusted more than blended scores.

### RB Underpriced Signals

- Source: research outputs requested in the project brief; not wired into the current app.
- Measures: probability that an RB drafted outside a target tier later hits that tier.
- Rough calculation: best result named in the brief is Underpriced_RB2 using Regularized Logistic Regression with Usage + ADP.
- Higher is better.
- App-facing: no.
- Status: promising research-only; needs benchmark/freeze audit before app use.

### Age Scores / Age Curves

- Source: `case_studies/` scripts and outputs, plus age-risk logic in championship equity.
- Measures: how player age relates to elite fantasy outcomes and risk.
- Rough calculation: research groups historical player seasons by age and finish thresholds; championship equity V2 separately applies position-specific age risk.
- Higher age-study rates are better; higher age risk is worse.
- App-facing: partly through championship equity age risk, but age-study dashboards are research-only.
- Status: age risk is app-facing directional; age studies are research-only.

## 5. Data Sources

### Main Player Data

- Used for: nearly all app pages, recommendations, player cards, simulations, team outlook, and tiers.
- Loaded from: `data/processed/master_players.csv` first, then fallback CSV paths, then SQLite fallback paths.
- Features: player name, position, team, bye, age, injury status, projection, projection rank, position rank, tier, ceiling/floor, ADP, ADP rank, injury risk, durability, boom/bust/stability, archetype.
- App-facing: yes.

### Legacy SQLite Player Data

- Used for: older `utils.py` pages, especially Player Compare and Live Rankings.
- Loaded from: `guaranteed_play.db`.
- Features: whatever the `players` table contains; older helpers look for player, position, team, ADP, and projection.
- App-facing: yes on older pages.
- Status: needs review because it bypasses newer `draftkit.data_access`.

### FantasyPros Raw Rankings

- Used for: source/reference data and possible data prep.
- Loaded from: `data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`.
- Features: rank, tier, player, team, position rank, bye, upside/bust labels, schedule strength, ECR vs ADP.
- App-facing: not directly through the current main loader.
- Status: source/raw input.

### Sportsbook / Market Data

- Used for: projection engine, sportsbook fallback, market disagreement, and signal trust.
- Loaded from: provider modules for DraftKings, FanDuel, Pinnacle, FantasyPros fallback, and internal fallback.
- Features: prop lines, market projections, fantasy-point projection, sportsbook projection rank, provider/debug coverage, projection gaps.
- App-facing: partly, mostly through trust/debug and optional projection enrichment.
- Status: experimental/diagnostic unless refreshed and validated.

### Historical Stats / Case Study Data

- Used for: age studies and WR opportunity research.
- Loaded from: `case_studies/data/` and duplicated nested `case_studies/case_studies/data/`.
- Features: historical seasons, age, position, fantasy points, positional finish, target share, targets.
- App-facing: no, except indirect age-risk concepts in app code.
- Status: research-only.

### Research Outputs

- Used for: owner review, research dashboards, and future model design.
- Loaded from: `case_studies/output/`, nested `case_studies/case_studies/output/`, root `opportunity_wr_*.html`, and performance reports.
- Features: age curves, opportunity dashboards, target-share outcome tables, runtime profile metrics.
- App-facing: mostly no.
- Status: research-only/diagnostic.

## 6. Research Outputs

### WR

- WR2 baseline: V1 raw WR2 model, WR1-WR48, 41.2% hit rate, 3.60x lift.
- Practical WR2 diagnostic: V2 focused WR2, WR25-WR42, 50.0% hit rate, 1.93x lift.
- WR1 ceiling model: V2 WR1, WR13-WR24, 20.0% hit rate, 1.53x lift.
- Conclusion: separate signals are trusted more than blended scores.
- Score V4 is delayed because score compression lost signal.
- App status: not currently app-facing.

### RB

- RB V1 best result: Underpriced_RB2, Regularized Logistic Regression, Usage + ADP.
- Result: 41.4% hit rate, 10.3% base rate, 4.02x lift, ROC AUC 0.852.
- Conclusion: promising, but needs benchmark/freeze audit before app use.
- App status: not currently app-facing.

### Opportunity / Age Studies

- WR opportunity dashboards study target share and raw targets versus historical WR outcomes.
- Age studies summarize historical elite-finish rates by age for QB/RB/WR/TE.
- These are useful research references, not current Draft Mode scoring inputs.

## 7. What Is Trusted Right Now

- Draft Mode as the primary decision surface.
- Main player data from `data/processed/master_players.csv`, assuming the file is refreshed intentionally.
- Master recommendation score as the core ranking signal.
- Consensus score as the top-level pick signal in Draft Mode.
- ADP value when ADP/projection ranks pass trust checks.
- Roster construction pressure and position need.
- Tier urgency as a supporting signal.
- Availability probability as directional wait/take guidance.
- Championship equity V2 as directional portfolio/upside guidance.
- Signal trust as a useful guardrail.

## 8. What Is Experimental or Should Not Be Used Yet

- WR underpriced probabilities should remain research-only.
- RB underpriced probabilities should remain research-only until benchmark/freeze audit is done.
- Score V4 should remain paused because compression lost signal.
- Future impact simulation is useful but slower and still experimental.
- Sportsbook/market projection enrichment should be treated as diagnostic unless provider coverage is validated.
- Component Audit is a diagnostic page, not a production decision page.
- Older `utils.py` Best Fit/Best Value/Best Available rankings should not be treated as the same thing as Draft Mode recommendations.

## 9. What Might Be Confusing or Duplicated

- There are two data-loading paths:
  - newer `draftkit.data_access` prefers `data/processed/master_players.csv`
  - older `utils.py` reads `guaranteed_play.db`
- There are two scoring generations:
  - newer master recommendation engine in `draftkit/draft_analysis.py`
  - older Best Fit/Best Value/Best Available helpers in `utils.py`
- Player Compare and Live Rankings are app-facing but use the older scoring layer.
- Championship equity has an older module and a newer V2 module. Draft Mode uses V2; some diagnostics/fallbacks still call the older version.
- Research outputs are duplicated in both `case_studies/output/` and `case_studies/case_studies/output/`.
- Root-level opportunity HTML files duplicate files under `case_studies/output/`.
- `pages/5_Draft_Lab.py` exists but is not currently in the app navigation.
- Raw FantasyPros data exists, but the current app primarily loads the processed master file.
- Some labels like Model Score, Consensus Score, Final Score, Best Fit Score, and Value Score can sound similar but refer to different layers.

## 10. How To Resume Later

- Where we paused: the app is explanation-mapped, with Draft Mode identified as the trusted current surface and older rankings/research clearly separated.
- Next research task: finish the RB benchmark/freeze audit before using Underpriced_RB2 in the app. Keep WR signals separate rather than forcing a compressed blended score.
- Next app cleanup task: align Player Compare and Live Rankings with the newer `draftkit` data loader and master recommendation score, or mark them clearly as legacy.
- Do not touch yet: do not wire WR/RB underpriced research into Draft Mode until the research signals have frozen definitions, benchmarks, and clear display rules.
- Also avoid changing championship equity labels until the app consistently uses V2 or explicitly distinguishes V1 diagnostics from V2 decision signals.
