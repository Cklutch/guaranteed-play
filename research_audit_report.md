# Guaranteed Play Research Audit Report

Date: 2026-07-06

Project inspected: `C:\Users\cklut\Desktop\Projects\Guaranteed Play`

Note: the configured Codex workspace folder `C:\Users\cklut\Documents\FF` was empty. The actual project files were found under the Desktop project path above.

## Executive Summary

The research side currently contains useful descriptive studies, but I did not find trained WR/RB breakout models, saved model files, notebooks, WR signal exports, WR signal loss audits, research freeze reports, or underpriced WR/RB model scripts in this project tree.

The strongest confirmed research asset is the WR opportunity study in `case_studies/opportunity_modeling.py`. It studies WR target share and raw targets against actual WR finishes from 2016-2025 nflverse data. It is useful as research evidence, but it is not a predictive draft model yet because it uses same-season opportunity and same-season fantasy finish. That makes it descriptive and likely not pre-draft safe unless the opportunity inputs are replaced with projections.

The age-curve work is also useful, especially `case_studies/run_fantasy_age_study.py` and `case_studies/rb_elite_age_analysis.py`, but it is descriptive. It identifies which ages have historically occupied elite finish slots. It does not validate draft pick recommendations.

RB research is behind WR research. RB has an age study, but I did not find RB opportunity, workload, goal-line, durability, underpriced RB2, or breakout models comparable to the WR opportunity work.

Recommended next step: build a feature documentation and evaluation layer before integrating research into the app. The current research is not ready to be treated as a trusted draft signal in Draft Mode.

## Repository Structure Audit

### App-Facing Files

| Path | Role | Status |
| --- | --- | --- |
| `app.py` | Small Streamlit entry file | App-facing |
| `Home.py` | Streamlit home page | App-facing |
| `pages/1_Draft_Mode.py` | Main draft command center | App-facing, primary |
| `pages/2_Player_Cards.py` | Player card view | App-facing |
| `pages/3_Team_Outlook.py` | Team/roster outlook | App-facing |
| `pages/5_Draft_Lab.py` | Draft simulation lab | App-facing/experimental |
| `pages/5_Player_Compare.py` | Player comparison page | App-facing, legacy scoring |
| `pages/7_Live_Rankings.py` | Live board page | App-facing, legacy scoring |
| `pages/8_Tier_Desperation.py` | Tier and scarcity support page | App-facing support |
| `pages/9_Component_Audit.py` | Recommendation component diagnostics | App-facing diagnostic |
| `draftkit/*.py` | Current recommendation, data, simulation, scoring modules | App-facing engine |
| `utils.py` | Older helper/scoring layer | App-facing legacy |

### Active/Useful Research Outputs

| Path | What it is | Trust level |
| --- | --- | --- |
| `case_studies/opportunity_modeling.py` | WR targets and target-share descriptive study | Useful research, not app-ready |
| `case_studies/data/opportunity_wr_target_share_player_seasons_half_ppr.csv` | Cached WR opportunity player-seasons, 2,169 rows | Useful source table |
| `case_studies/output/opportunity_wr_target_share.html` | Generated WR target-share report | Useful research output |
| `case_studies/output/opportunity_wr_targets.html` | Generated WR raw-targets report | Useful research output |
| `case_studies/run_fantasy_age_study.py` | Multi-position age curve generator | Useful descriptive research |
| `case_studies/rb_elite_age_analysis.py` | Data pull and age summary helper | Useful descriptive research |
| `case_studies/output/wr_elite_age_study.md` | Generated WR age report | Useful reference |
| `case_studies/output/rb_elite_age_study.md` | Generated RB age report | Useful reference |
| `case_studies/output/wr_elite_age_rates.csv` | WR age rates table, 16 rows | Useful reference |
| `case_studies/output/rb_elite_age_rates.csv` | RB age rates table, 14 rows | Useful reference |

### Experimental/Old Versions

| Path | Why it looks experimental or older |
| --- | --- |
| `case_studies/RB_Age_Study.py` | Older/smaller RB-only age study script compared with `run_fantasy_age_study.py` |
| `case_studies/run_rb_age_study_last10.py` | Narrow RB-only last-10 script, likely superseded by multi-position age study |
| `case_studies/run_rb_age_share_study.py` | Narrow RB-only share script, likely superseded |
| `case_studies/case_studies/...` | Nested duplicate `case_studies` tree with duplicated data/output artifacts |
| root `opportunity_wr_targets.html` | Duplicate copy of `case_studies/output/opportunity_wr_targets.html` |
| root `opportunity_wr_target_share.html` | Duplicate copy of `case_studies/output/opportunity_wr_target_share.html` |
| `draftkit/championship_equity.py` | Older app model exists next to `championship_equity_v2.py` |
| older `utils.py` ranking helpers | Still used by some pages, but not aligned with the newer Draft Mode engine |

### Duplicate or Redundant Files

| Duplicate pattern | Example files |
| --- | --- |
| Root copies of WR opportunity reports | `opportunity_wr_targets.html`, `case_studies/output/opportunity_wr_targets.html` |
| Nested `case_studies/case_studies` copy | `case_studies/case_studies/data/*`, `case_studies/case_studies/output/*` |
| Multiple RB age scripts | `RB_Age_Study.py`, `run_rb_age_study_last10.py`, `run_rb_age_share_study.py`, `run_fantasy_age_study.py` |
| PPR and half-PPR historical caches | Multiple `*_player_seasons_ppr.csv` and `*_player_seasons_half_ppr.csv` files |
| Generated chart variants | PNG, SVG, HTML, and CSV outputs for same studies |

### Data/Raw/Intermediate Files

| Path | Contents |
| --- | --- |
| `data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv` | FantasyPros rankings source, 456 rows |
| `data/processed/master_players.csv` | Main processed app player table, 3,931 rows |
| `data/players.csv`, `data/players_expanded_sample.csv`, `data/players_cleaned.csv` | Small/sample player files |
| `guaranteed_play.db` | SQLite database used by older loaders |
| `case_studies/data/*.csv` | Cached historical research datasets |
| `case_studies/output/*.csv` | Generated summary tables |

### Files That Should Be Archived or Ignored

| Path | Recommendation |
| --- | --- |
| `.venv/` | Ignore. Dependency environment, not project source |
| `.idea/` | Ignore. Editor metadata |
| `__pycache__/` | Ignore/generated |
| `case_studies/case_studies/` | Archive or delete after confirming no unique newer outputs |
| Root copies of opportunity HTML | Archive or remove after keeping canonical output folder copies |
| Older RB-only scripts | Archive after confirming `run_fantasy_age_study.py` replaces them |

## Current Research Models and Scoring Systems

### WR Target Share Opportunity Study

| Field | Finding |
| --- | --- |
| File | `case_studies/opportunity_modeling.py` |
| Purpose | Study relationship between WR target share buckets and fantasy outcomes |
| Target/label | Same-season WR positional finish thresholds: Top 36, Top 24, Top 12, Top 5, WR1 overall |
| Position | WR |
| Features used | Targets, team targets, target share, target share percent, games, receiving TDs, age, fantasy points, fantasy PPG |
| Data source | nflverse-data GitHub releases: `stats_player` plus `rosters`; cached locally |
| Model type | Descriptive bucket analysis and weighted score, not trained ML |
| Outputs | `case_studies/data/opportunity_wr_target_share_player_seasons_half_ppr.csv`, `case_studies/output/opportunity_wr_target_share.html`, root duplicate HTML |
| Validation | No train/test validation found. Same-season descriptive grouping only |
| Metrics | Bucket sample size, average PPG, median PPG, Top 36/24/12/5 rates, WR1 rate, target-share score |
| Task type | Descriptive analysis/scoring |
| Draft usefulness | Useful evidence, not draft-ready because same-season target share is not known pre-draft |

### WR Raw Targets Opportunity Study

| Field | Finding |
| --- | --- |
| File | `case_studies/opportunity_modeling.py` |
| Purpose | Study relationship between raw WR target buckets and fantasy outcomes |
| Target/label | Same-season WR positional finish thresholds: Top 36, Top 24, Top 12, Top 5, WR1 overall |
| Position | WR |
| Features used | Targets, receiving TDs, games, fantasy points, PPG, positional finish; also target bucket |
| Data source | Same nflverse cache as target-share study |
| Model type | Descriptive bucket analysis and weighted score, not trained ML |
| Outputs | `case_studies/output/opportunity_wr_targets.html`, root duplicate HTML |
| Validation | No train/test validation found |
| Metrics | Bucket sample size, average PPG, median PPG, Top 36/24/12/5 rates, WR1 rate, targets score |
| Task type | Descriptive analysis/scoring |
| Draft usefulness | Useful for understanding opportunity thresholds, not app-ready as a pre-draft signal |

### Multi-Position Age Curve Study

| Field | Finding |
| --- | --- |
| File | `case_studies/run_fantasy_age_study.py` plus `case_studies/rb_elite_age_analysis.py` |
| Purpose | Identify age shares inside elite positional finish buckets |
| Target/label | Top-N positional finishes: RB/WR Top 36, Top 24, Top 12, Top 5, Top 3; QB/TE Top 20, Top 15, Top 12, Top 6, Top 3; Top 1 overall comparison also exists |
| Position | RB, WR, QB, TE |
| Features used | Age, position, fantasy points, positional finish, production stat columns where available |
| Data source | nflverse-data GitHub releases: `stats_player` plus `rosters`; cached locally |
| Model type | Descriptive age distribution, not predictive ML |
| Outputs | `case_studies/output/*_elite_age_rates.csv`, `*_elite_age_study.md`, chart PNG/SVG files, dashboard HTML |
| Validation | No train/test validation found |
| Metrics | Age share inside Top-N slots, peak age, player-seasons analyzed |
| Task type | Descriptive analysis |
| Draft usefulness | Useful as an age prior or guardrail, not enough alone for pick recommendations |

Confirmed output metrics from generated reports:

| Position | Key confirmed peaks |
| --- | --- |
| WR | Top 36 peak age 24, Top 24 peak age 24, Top 12 peak age 25, Top 5 peak age 26, Top 3 peak age 24 |
| RB | Top 36 peak age 24, Top 24 peak age 24, Top 12 peak age 25, Top 5 peak age 23, Top 3 peak age 22 |

### App Recommendation Score

| Field | Finding |
| --- | --- |
| File | `draftkit/draft_analysis.py` and related `draftkit` modules |
| Purpose | Rank current draft candidates in the app |
| Target/label | No historical target found; this is a rules/heuristic recommendation engine |
| Position | General: QB, RB, WR, TE |
| Features used | Projection value, roster need, ADP value, tier urgency, team fit, construction pressure, signal trust, risk/upside fields |
| Data source | `data/processed/master_players.csv`, fallback CSV/SQLite sources, Streamlit session draft state |
| Model type | Heuristic weighted scoring/ranking |
| Outputs | Streamlit recommendation tables/cards; no saved research model output found |
| Validation | No historical validation found in inspected files |
| Metrics | Model/final score, consensus score, conviction, equity delta, availability probability |
| Task type | Ranking/scoring/simulation |
| Draft usefulness | Current app decision layer, but not validated as historical research |

### Championship Equity V2

| Field | Finding |
| --- | --- |
| File | `draftkit/championship_equity_v2.py` |
| Purpose | Simulate candidate impact on championship probability |
| Target/label | Simulated roster outcome, not historical label |
| Position | General |
| Features used | Projection, floor/ceiling, injury risk, durability, age risk, role uncertainty, roster shape, ADP late-value bonus |
| Data source | App player table and draft state |
| Model type | Monte Carlo simulation/scoring |
| Outputs | App-facing equity reports/tables |
| Validation | No historical validation found |
| Metrics | Championship probability, equity delta, upside/portfolio values |
| Task type | Simulation/scoring |
| Draft usefulness | Directional app signal, not a research-validated model |

## WR Research Audit

Files inspected or searched for WR research:

| Area requested | Status |
| --- | --- |
| WR breakout model | Not found |
| WR upside/value score | App has generic upside/value; no WR-specific research model found |
| Underpriced WR2 / WR1 / Top24 models | Not found |
| ADP-based WR model | Not found as research model; ADP is used in app scoring |
| Usage-based WR model | Found descriptive target and target-share opportunity studies |
| Explainability reports | App has component audit/system explanation; no WR model explainability report found |
| Signal export files | Not found |
| Evaluation universe benchmark | Not found |
| WR signal loss audit | Not found |
| Research model freeze report | Not found |

Strongest WR model currently found: the WR opportunity study. It is the strongest because it has a clear dataset, clear outcome thresholds, and generated reports. But it is not a predictive model and should not be called a breakout model.

Most useful target definition found: Top 24 and Top 12 WR finish rates are the most draft-relevant labels in the existing files. Top 24 maps to usable starter outcomes. Top 12 maps to real WR1 outcomes. WR1 overall is too rare to be a stable primary label.

Is the WR work finding an edge or rediscovering ADP: ADP is not part of the WR opportunity study, so it is not rediscovering ADP directly. However, it also does not prove a draft edge because it uses realized same-season opportunity. The missing step is testing whether pre-draft available features can predict those opportunity buckets or Top 24/Top 12 outcomes better than ADP.

Useful non-ADP features found: targets, team targets, target share, age, receiving TDs, games, fantasy PPG, and positional finish. Of these, only age is clearly pre-draft available as-is. Targets and target share need projected versions to become draft-safe.

WR integration readiness: not ready for direct app integration as a player-level draft signal. It is ready to inform feature design, target definitions, and threshold language.

## RB Research Audit

Files inspected or searched for RB research:

| Area requested | Status |
| --- | --- |
| RB breakout model | Not found |
| Underpriced RB2 target | Not found |
| RB age curve research | Found |
| RB opportunity/workload/team/goal-line/durability features | Not found as RB research model inputs; app has generic risk/durability heuristics |
| RB scoring or signal exports | Age-study CSV/Markdown/HTML outputs found only |

Completed RB work:

- RB age curve study over 2016-2025 nflverse data.
- Generated RB age report and age-rate CSV.
- Older RB-only age scripts and generated charts exist.

Missing compared with WR:

- No RB opportunity study comparable to the WR target-share study.
- No RB workload feature table for carries, targets, route/snap role, goal-line usage, team scoring environment, or backfield competition.
- No RB underpriced target model.
- No RB validation framework.

RB integration readiness: not ready. The RB age curve can be used as a weak prior or context note, but not as a draft recommendation model.

Best next RB research step if RB is prioritized: build a pre-draft-safe RB opportunity/workload dataset and validate a Top 24 or RB2 finish target against ADP baseline.

## Feature Inventory

| Feature | Position/model | Represents | Source/calculation | Objective or subjective | Pre-draft available | Leakage risk | Missing risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADP | App scoring | Market cost | `master_players.csv` / FantasyPros raw rankings | Objective market-derived | Yes | Low if current preseason ADP | Medium | Trust with signal checks |
| ECR vs ADP | Raw data | Expert rank gap vs ADP | FantasyPros raw CSV | Objective market-derived | Yes | Low | Medium | Investigate for app use |
| Projection points | App scoring | Expected fantasy points | `master_players.csv` | Depends on source | Yes | Low if preseason projection | Medium | Trust directionally |
| Projection rank | App scoring | Rank by projection | `master_players.csv` / app calc | Objective from projection | Yes | Low | Medium | Trust directionally |
| Position rank | App/data | Positional rank | `master_players.csv` | Objective from source | Yes | Low | Medium | Trust directionally |
| Tier | App/data | Player tier | `master_players.csv` | Source unclear | Yes if provided | Low | Medium | Improve documentation |
| Ceiling projection | App/risk/upside | Upside estimate | `master_players.csv` or inferred | Source unclear | Yes if preseason | Low | Medium | Investigate source |
| Floor projection | App/risk | Downside estimate | `master_players.csv` or inferred | Source unclear | Yes if preseason | Low | Medium | Investigate source |
| Injury risk | App/risk | Injury/downside risk | `master_players.csv` or `feature_engineering.py` inference | Mixed | Yes if preseason | Medium if derived from current-season results | High | Improve source documentation |
| Durability grade | App/risk | Availability safety | Direct field or games/missed-games inference | Mixed | Yes if based on prior history | Medium | High | Improve source documentation |
| Boom score | App/upside | Ceiling/upside | Direct field or projection percentile + volatility | Mixed | Yes if inputs are preseason | Medium | Medium | Useful but audit source |
| Bust score | App/risk | Downside | Direct field or injury/volatility/durability | Mixed | Yes if inputs are preseason | Medium | Medium | Useful but audit source |
| Stability score | App/risk | Safety | Data field and app risk logic | Mixed | Yes if preseason | Medium | Medium | Useful but audit source |
| Archetype | App/profile | Player style/risk label | `master_players.csv` | Likely subjective/inferred | Yes | Low | Medium | Keep as explanatory, not core |
| Expected finish | Research target concept | Future positional finish | Not found as pre-draft feature; positional finish is outcome in research | Objective outcome | No if same-season actual | High | N/A | Use only as label, not feature |
| Positional finish | WR/RB age/opportunity labels | Actual season finish | Rank fantasy points within season/position | Objective | No | High if used as feature | Low in historical data | Trust as label only |
| Prior PPG | Requested feature | Previous scoring rate | Not found as research feature | Objective if built | Yes | Low if prior season only | Medium | Build/investigate |
| Fantasy PPG | WR opportunity output | Same-season per-game scoring | Fantasy points / games | Objective outcome | No | High if used as feature | Medium | Label/diagnostic only |
| Targets | WR opportunity | Same-season passing opportunity | nflverse stats_player | Objective | No as actual, yes if projected | High if actual current season | Low historical | Use projected targets for draft model |
| Target share | WR opportunity | Share of team targets | targets / team_targets | Objective | No as actual, yes if projected | High if actual current season | Medium | Use projected target share only |
| Team targets | WR opportunity | Team pass-game volume to WR/team | Sum targets by team/season | Objective | No as actual, yes if projected | High | Low historical | Convert to projected team pass volume |
| Receiving TDs | WR opportunity | TD production | nflverse stats_player | Objective | No actual | High | Low historical | Label/diagnostic only unless projected |
| Games | WR opportunity | Games counted in player season | Derived from weekly rows/targets | Objective | No actual | High | Medium | Label/diagnostic only |
| Age | WR/RB/QB/TE age studies, app data | Player age | rosters age or birth date | Objective | Yes | Low | Medium | Trust |
| Draft capital | Requested feature | NFL draft investment | Not found | Objective | Yes | N/A | N/A | Add later |
| Air yards | Requested WR feature | Downfield opportunity | Not found | Objective | Yes if prior/projected | N/A | N/A | Add later |
| Route participation | Requested WR feature | Route role | Not found | Objective | Yes if projected/prior | N/A | N/A | Add later |
| Snap share | Requested feature | Playing time | Not found | Objective | Yes if projected/prior | N/A | N/A | Add later |
| Red zone usage | Requested feature | Scoring opportunity | Not found | Objective | Yes if projected/prior | N/A | N/A | Add later |
| Team pass attempts | Requested feature | Team environment | Not found directly | Objective | Yes if projected | N/A | N/A | Add later |
| QB/team/OC context | Requested feature | Environment quality/change | Not found as research feature | Mixed | Yes | N/A | N/A | Add later, document carefully |
| Competition score | Requested feature | Depth chart competition | Not found | Mixed/objective possible | Yes | N/A | N/A | Add later |
| Workload/security | Requested RB feature | Role safety | Not found as RB research | Mixed/objective possible | Yes | N/A | N/A | Add later |
| Goal-line usage | Requested RB feature | TD opportunity | Not found | Objective | Yes if projected/prior | N/A | N/A | Add later |

## Validation Quality Review

| Model/study | Validation found | Leakage risk | Sample/universe | Metrics | Practical usefulness |
| --- | --- | --- | --- | --- | --- |
| WR target-share study | No train/test or walk-forward validation found | High if used for draft decisions because same-season target share and finish are paired | 2,169 WR player-seasons in cached half-PPR file | Top-N finish rates by bucket, average/median PPG, score | Good descriptive evidence, not draft-ready |
| WR targets study | No train/test or walk-forward validation found | High if actual targets are used as features | Same WR opportunity dataset | Top-N finish rates by bucket, average/median PPG, score | Good threshold evidence, not draft-ready |
| Age curve study | No train/test or walk-forward validation found | Low if used only as descriptive historical context; high if age buckets are overfit into a predictive model without testing | 5,929 multi-position player-seasons in generated PPR report | Age share in Top-N slots, peak age | Useful prior, not standalone model |
| App recommendation engine | No historical validation found | Depends on projection/ADP freshness; no historical backtest found | Current draft pool | Heuristic scores, simulation outputs | Useful current app layer, but not research-validated |
| Championship equity V2 | No historical validation found | Simulation assumptions may create false precision | Current draft pool and roster state | Simulated championship probability/equity delta | Directional only |

Misleading metric risks:

- Age share is not hit rate. It measures what share of elite slots came from an age, not the probability that a player of that age becomes elite.
- WR opportunity bucket rates are descriptive, not pre-draft predictive, unless using projected targets/target share available before the draft.
- WR1 overall rate is rare and likely unstable in small buckets.
- AUC, R-squared, lift, and top-decile hit rate were not found for the research studies.
- No ADP baseline comparison was found, so no edge over market is proven.

## Trusted Research Assets

Trusted means the file is useful as an honest baseline or source of evidence, not that it is ready to drive app recommendations.

### Trusted Scripts

1. `case_studies/opportunity_modeling.py`
2. `case_studies/run_fantasy_age_study.py`
3. `case_studies/rb_elite_age_analysis.py`

### Trusted Output Files

1. `case_studies/data/opportunity_wr_target_share_player_seasons_half_ppr.csv`
2. `case_studies/output/opportunity_wr_target_share.html`
3. `case_studies/output/opportunity_wr_targets.html`
4. `case_studies/output/wr_elite_age_study.md`
5. `case_studies/output/rb_elite_age_study.md`

### Trusted Target Definitions

- WR Top 24 finish as a starter/value target.
- WR Top 12 finish as a true WR1 upside target.
- RB Top 24 finish as a future RB2/starter target, though no RB model exists yet.
- Top 5 and Top 3 finishes as descriptive ceiling outcomes, not primary model labels.

### Trusted Features

- Age, as a pre-draft safe context feature.
- ADP, if sourced cleanly and checked for missing/outlier values.
- Projection points/rank, if source is documented.
- Targets and target share only as historical labels/diagnostics unless converted to projected pre-draft features.

### Official Baselines

- Use `case_studies/opportunity_modeling.py` as the WR opportunity baseline.
- Use `case_studies/run_fantasy_age_study.py` as the age-curve baseline.
- Use `data/processed/master_players.csv` as the current app player pool baseline.

## Do Not Trust Yet

- Any implied WR breakout model. I did not find one.
- Any implied underpriced WR2/WR1/Top24 model. I did not find one.
- Any implied underpriced RB2 model. I did not find one.
- WR opportunity scores as draft inputs without projected opportunity data.
- Championship probability as literal odds.
- Older `utils.py` Best Fit/Best Value/Best Available scores as primary recommendations.

## Needs Rebuild

- Evaluation framework: no walk-forward, ADP baseline, top-decile lift, or consistent universe benchmark was found.
- Feature documentation: many app-facing columns exist without clear source/provenance documentation.
- Research output naming: duplicate root/case_studies outputs make it unclear which file is canonical.
- RB research: needs opportunity/workload features before any model integration.

## Archive / Old Version

Likely archive candidates:

1. `case_studies/case_studies/`
2. `case_studies/RB_Age_Study.py`
3. `case_studies/run_rb_age_study_last10.py`
4. `case_studies/run_rb_age_share_study.py`
5. Root `opportunity_wr_targets.html` and `opportunity_wr_target_share.html` duplicates, after confirming canonical copies are under `case_studies/output/`

## Useful But Not App-Ready

- WR target-share opportunity report.
- WR raw targets opportunity report.
- WR/RB age curve reports.
- Generated charts and dashboards.

## Duplication and Cleanup Opportunities

Recommended clean structure, without moving files yet:

```text
research/
  data/
    raw/
    interim/
    processed/
  studies/
    wr_opportunity/
    age_curves/
    rb_workload/
  reports/
  exports/
  archive/
```

Specific cleanup recommendations:

- Pick one canonical output folder: `case_studies/output` or `research/reports`.
- Move duplicate root HTML reports into archive or remove them after confirming no differences.
- Archive `case_studies/case_studies/` if it is a stale nested copy.
- Rename age-study columns from `total_rb_seasons` and `top24_rb_seasons` when used for WR/QB/TE, because those names are confusing in WR reports.
- Add a README to each research study folder with source data, target label, leakage notes, and exact outputs.
- Add a model registry or research inventory CSV before adding new models.

## Recommended Next Step

Ranked recommendation:

1. Build feature documentation layer.
2. Rebuild evaluation framework.
3. Advance RB model to WR-level validation.
4. Build unified player signal export.
5. Continue WR research.
6. Freeze WR model and integrate signals into app.

Immediate next step: build feature documentation layer.

Why this is next: the project currently has research outputs, app scores, and feature columns, but not enough provenance to know which features are pre-draft safe, which are labels, and which are inferred. Without that, any WR/RB model risks leakage or false confidence.

Exact files to use:

- `case_studies/opportunity_modeling.py`
- `case_studies/run_fantasy_age_study.py`
- `case_studies/rb_elite_age_analysis.py`
- `case_studies/data/opportunity_wr_target_share_player_seasons_half_ppr.csv`
- `data/processed/master_players.csv`
- `draftkit/feature_engineering.py`
- `draftkit/draft_analysis.py`
- `draftkit/championship_equity_v2.py`

Exact files to avoid:

- `case_studies/case_studies/` until duplicate status is resolved.
- Root duplicate opportunity HTML files as canonical sources.
- `case_studies/RB_Age_Study.py`, `run_rb_age_study_last10.py`, and `run_rb_age_share_study.py` unless specifically comparing old versions.
- Older `utils.py` ranking helpers for new research conclusions.

What success looks like:

- A single markdown or CSV feature dictionary exists.
- Every feature has source, calculation, pre-draft availability, leakage risk, missingness risk, and trust status.
- Labels are clearly separated from features.
- Current app fields are mapped to research fields where possible.
- The next model-building prompt can safely choose a target and input set.

Expected output:

- `research_feature_dictionary.md` or `research_feature_dictionary.csv`
- Optional `research_inventory.md` listing canonical scripts, outputs, and archive candidates

## Codex-Ready Prompt for Next Phase

```text
You are working inside my Guaranteed Play fantasy football project. Do not build new models yet.

Create a research feature documentation layer. Inspect the current app data, research scripts, and generated research outputs. Produce `research_feature_dictionary.md` and, if useful, `research_feature_dictionary.csv`.

For every feature or label used in `data/processed/master_players.csv`, `draftkit/feature_engineering.py`, `draftkit/draft_analysis.py`, `draftkit/championship_equity_v2.py`, `case_studies/opportunity_modeling.py`, `case_studies/run_fantasy_age_study.py`, and `case_studies/rb_elite_age_analysis.py`, document:

- feature or label name
- source file
- source data
- calculation
- position applicability
- whether it is a model feature, label, diagnostic metric, or app display field
- whether it is available before a fantasy draft
- leakage risk
- missing value risk
- trust status: trusted, usable with caution, investigate, remove, or label only
- recommendation for WR modeling
- recommendation for RB modeling

Keep labels separate from input features. Clearly flag same-season actual stats that cannot be used as pre-draft model inputs. Do not change the Streamlit app. Do not build models. End with the recommended target and feature set for the next WR or RB model validation pass.
```
