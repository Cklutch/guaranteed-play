# Guaranteed Play Performance Audit

Date: 2026-06-12

## Executive Summary

Draft Mode latency was primarily caused by eager advanced modeling on every Streamlit rerun:

- Championship Equity V2 ran 250 simulations per candidate for 12 candidates on page load.
- Recommendation decision cards were built eagerly for 8 players, including availability/opponent/tier explanation paths.
- The best-pick panel called `get_best_pick_recommendation()`, which rebuilt the master recommendation table after `build_recommendation_rankings_df()` had already run.
- Draft Lab ran full single and batch simulations immediately when the page loaded or any widget changed.
- Available-player filtering, board sorting, and lookup construction repeated across pages.

The optimization pass adds cached Draft Mode outputs, fast-mode simulation counts, top-candidate advanced simulation limits, lazy detailed recommendation cards, cached available-player filtering, submit-driven Draft Lab execution, and optional runtime profiling.

## Page Audit

### `pages/1_Draft_Mode.py`

Execution path:

1. Initialize session state and sidebar settings.
2. Load raw and available player data.
3. Resolve column names.
4. Build recommendation rankings.
5. Build board recommendation lookup.
6. Build tier warning.
7. Build conviction report.
8. Run availability simulation.
9. Optionally run future-impact Monte Carlo.
10. Run Championship Equity V2.
11. Build consensus report.
12. Render recommendation summary, cards, draft plan, status, queue, board, and draft search.

Functions called:

- `load_players_df`
- `get_available_players_df`
- `build_recommendation_rankings_df`
- `get_tier_warning`
- `build_conviction_report`
- `calculate_availability_probability`
- `build_simulation_report`
- `build_championship_equity_v2_df`
- `build_consensus_recommendations`
- `build_player_decision_card` on demand
- `build_draft_strategy`
- `get_turn_aware_falloff_recommendation`

Estimated runtime before optimization:

- Normal rerun: 2.5-8.0s
- With future-impact simulation: 5.0-15.0s
- Championship Equity V2 alone: 1.5-6.0s depending player pool

Estimated runtime after optimization:

- Warm rerun: under 1s for unchanged draft state
- Recommendation refresh: under 1s after cache warmup
- Championship Equity V2 fast mode: top 5 candidates x 35 simulations
- Future Impact only when enabled: top 5 candidates x 50 simulations

Simulations executed:

- Availability: 35 in Draft Mode fast path
- Championship Equity V2: 35 per top-5 candidate
- Future Impact: 50 per top-5 candidate, only when toggled

Duplicate computations found:

- `build_recommendation_rankings_df()` and `get_best_pick_recommendation()` both rebuilt recommendations.
- Decision cards generated for all top 8 recommendations even when not viewed.
- Championship Equity V2 ran for 12 candidates even though only top recommendation and visible cards needed it.
- Available player filtering repeated despite unchanged `drafted_players`.

Fixes implemented:

- Cached recommendation table by draft compute key.
- Replaced duplicate best-pick rebuild with top row from existing recommendation table.
- Cached conviction, availability, championship, future impact, and consensus outputs.
- Restricted advanced simulation to top 5 candidates.
- Reduced Draft Mode Championship Equity V2 from 250 to 35 simulations.
- Reduced Draft Mode availability from 75 to 35 simulations.
- Reduced future-impact toggle from 105 to 50 simulations.
- Added on-demand detailed card loading.
- Stored latest recommendation, simulation, equity, and consensus outputs in `st.session_state`.
- Added optional developer performance metrics.

### `pages/2_Player_Cards.py`

Execution path:

1. Load raw and available players.
2. Build full recommendation rankings.
3. Resolve selected player.
4. Render card and recommendation metrics.

Functions called:

- `load_players_df`
- `get_available_players_df`
- `build_recommendation_rankings_df`

Estimated runtime:

- 0.5-2.0s depending recommendation cache state.

Simulations executed:

- None directly.

Duplicate computations:

- Builds recommendation rankings even if Draft Mode already produced same state.

Recommended fix:

- Read `st.session_state["latest_recommendation_df"]` when present and draft state matches.
- Add cached page-local recommendation wrapper keyed by draft state.
- Generate player deep dive only after player selection.

### `pages/3_Team_Outlook.py`

Execution path:

1. Load player data.
2. Build roster projection tables.
3. Sort roster and league comparison views.
4. Render team outlook summaries.

Functions called:

- `load_players_df`
- local roster filtering/sorting helpers

Estimated runtime:

- 0.2-1.0s.

Simulations executed:

- None.

Duplicate computations:

- Repeated roster filtering and projection sorting.

Recommended fix:

- Cache roster lookup and position summary by `(my_team, roster_settings, player_data_mtime)`.

### `pages/4_Draft_Queue.py`

Execution path:

1. Load available players.
2. Render queue controls.
3. Filter/search available players.

Functions called:

- `get_available_players_df`
- queue state handlers

Estimated runtime:

- 0.1-0.8s.

Simulations executed:

- None.

Duplicate computations:

- Available-player filtering repeated before this optimization.

Fix implemented:

- `get_available_players_df()` now uses cached filtering keyed by drafted-player tuple.

### `pages/5_Player_Compare.py`

Execution path:

1. Load available players through legacy `utils`.
2. Filter by position.
3. Build selected player comparison.

Functions called:

- `utils.get_available_players_df`
- `utils.build_player_comparison_df`

Estimated runtime:

- 0.2-1.0s.

Simulations executed:

- None.

Duplicate computations:

- Legacy `utils` data access can duplicate `draftkit.data_access` loading.

Recommended fix:

- Migrate page to `draftkit.data_access` and cache comparison rows by selected player tuple.

### `pages/5_Draft_Lab.py`

Execution path before optimization:

1. Load players.
2. Run one complete draft.
3. Run batch simulation.
4. Evaluate batch.
5. Render outputs.

Execution path after optimization:

1. Load players.
2. Render controls in a form.
3. Run single draft and batch only after `Run Simulation`.
4. Store results in `st.session_state`.
5. Render cached last results.

Functions called:

- `load_players_df`
- `simulate_complete_draft`
- `run_simulation_batch`
- `evaluate_draft_lab_results`

Estimated runtime before optimization:

- 10 simulations: 2-6s
- 50 simulations: 8-30s
- 100 simulations: 15-60s
- 500 simulations: multi-minute on large pools

Estimated runtime after optimization:

- Page load: under 1s before running.
- Runs only on form submit.

Simulations executed:

- Draft Lab complete-draft batches: 10, 50, 100, or 500.
- Championship Equity inside draft grading: 250 per simulated final roster.

Duplicate computations:

- Player pool preparation repeated per simulation.

Fixes implemented:

- Batch runner now prepares the draft pool once and reuses it.
- Page now uses a form and submit button to prevent widget rerun storms.

### `pages/7_Live_Rankings.py`

Execution path:

1. Load rankings table.
2. Sort by selected field.
3. Render ranking table.

Estimated runtime:

- 0.1-0.7s.

Simulations executed:

- None.

Duplicate computations:

- Repeated table sort on widget changes.

Recommended fix:

- Cache sorted ranking views by `(sort_by, ascending, drafted_tuple)`.

### `pages/8_Tier_Desperation.py`

Execution path:

1. Load player data.
2. Build recommendation rankings.
3. Build tier and desperation tables.
4. Render urgency/desperation output.

Functions called:

- `load_players_df`
- `build_recommendation_rankings_df`
- tier helpers in `draft_analysis`

Estimated runtime:

- 0.8-3.0s.

Simulations executed:

- None.

Duplicate computations:

- Rebuilds recommendation table and tier tables separately.

Recommended fix:

- Use cached recommendation output from Draft Mode when draft state matches.
- Cache tier tables by player pool and roster settings.

## Bottleneck Breakdown

### Monte Carlo Bottlenecks

- `calculate_availability_probability`: repeated simulation of all available players until next pick.
- `build_simulation_report`: path simulation per candidate.
- `simulate_draft_paths`: nested candidate x simulation loop.

Fixes:

- Draft Mode availability reduced to 35 simulations.
- Future Impact remains toggle-gated and reduced to 50 simulations.
- Future Impact restricted to top 5.

### Championship Equity V2 Bottlenecks

- `build_championship_equity_v2_df`: candidate loop.
- `TeamOutcomeSimulator.simulate`: per-candidate season outcome loop.
- Previously forced 250 simulation minimum even when caller requested fewer.

Fixes:

- Simulator now honors caller-provided simulation count.
- Draft Mode uses 35 simulations.
- Draft Lab still uses 250+.
- Draft Mode candidate set restricted to top 5.

### Consensus Engine Bottlenecks

- Mostly lightweight DataFrame and ranking work.
- Cost grows with candidate count and repeated JSON/DataFrame conversion.

Fixes:

- Cached consensus report by draft compute key and input records.
- Candidate limit remains 12 for consensus, while expensive upstream inputs are top 5.

### Data Loading Bottlenecks

- `load_players_df` was already cached by source path and mtime.
- `get_available_players_df` filtered repeatedly.

Fixes:

- Added cached available-player filtering keyed by drafted tuple.

### DataFrame Bottlenecks

- Repeated `sort_values` in Draft Mode board, draft search, tier tables, and recommendation generation.
- Repeated merges in recommendation/trust/equity helpers.

Fixes implemented:

- Cached high-cost page-level outputs.
- Reused existing recommendation table for best-pick panel.

Recommended next fixes:

- Cache board display sorted views.
- Cache tier tables.
- Cache trust/equity lookups as dicts rather than repeatedly merging.

### Streamlit Rendering Bottlenecks

- Draft Lab widgets triggered full simulation reruns.
- Recommendation expanders still executed all inner code eagerly.
- Large dataframes rendered on every rerun.

Fixes:

- Draft Lab wrapped in `st.form`.
- Detailed recommendation card generation moved behind button.
- Developer metrics hidden behind sidebar checkbox.

## Top 20 Slowest Functions

| Rank | Function | Estimated Runtime Contribution | Cause | Recommended Fix |
|---:|---|---:|---|---|
| 1 | `TeamOutcomeSimulator.simulate` | 25-45% Draft Mode before fix | 250 simulations per candidate | Fast mode 35 sims in Draft Mode, keep 250+ in Draft Lab |
| 2 | `build_championship_equity_v2_df` | 20-40% Draft Mode before fix | Candidate loop plus simulator | Restrict to top 5 candidates |
| 3 | `run_simulation_batch` | 60-95% Draft Lab | Full draft batches | Run only on submit, reuse prepared pool |
| 4 | `simulate_complete_draft` | 10-40% Draft Lab | Full snake draft pick loop | Reuse prepared pool, keep page lazy |
| 5 | `build_simulation_report` | 10-35% when enabled | Monte Carlo path simulation | Toggle gated, top 5, 50 sims |
| 6 | `simulate_draft_paths` | 10-35% when enabled | Candidate x simulation nested loop | Candidate restriction |
| 7 | `calculate_availability_probability` | 5-20% Draft Mode | Repeated survival simulations | 35 sims and cached by draft key |
| 8 | `build_master_recommendations_df` | 15-35% normal Draft Mode | Many ranking, merge, tier, trust passes | Cache by draft key |
| 9 | `_build_base_recommendation_rankings_df` | 5-15% | Replacement baseline and row loop | Cache via master recommendation wrapper |
| 10 | `build_position_urgency_df` | 5-15% | Tier table generation | Cache tier tables by player pool |
| 11 | `build_championship_equity_df` | 5-15% older modules | Static equity calculation over pool | Prefer V2 top candidates or cached lookup |
| 12 | `build_conviction_report` | 5-12% | Adds equity and conviction layers | Cached by recommendation records |
| 13 | `build_player_decision_cards` | 10-25% before fix | Eager 8-card explanation | Replaced with single-card lazy loading |
| 14 | `build_player_decision_card` | 2-8% per card | Explanation dependencies | On-demand only |
| 15 | `build_draft_strategy` | 2-8% | Construction and history tracking | Could cache by draft key |
| 16 | `get_best_pick_recommendation` | 10-25% before fix | Rebuilt recommendations | Removed duplicate call |
| 17 | `get_available_players_df` | 2-10% | Repeated filtering/copying | Cached by drafted tuple |
| 18 | Draft board sorting | 2-8% | Sort/filter on every widget change | Cache sorted board views |
| 19 | Draft search sorting | 1-6% | Re-sorts filtered available pool | Cache or limit search results |
| 20 | `evaluate_draft_lab_results` | 1-8% Draft Lab | Aggregates batch results | Only runs after submitted simulation |

## Cache Inventory

Existing:

- `data_access._load_players_df_cached`
- Draft Mode availability/future/championship reports
- Draft Lab single/batch functions

Added:

- Draft Mode recommendation table cache.
- Draft Mode conviction report cache.
- Draft Mode consensus report cache.
- Data access available-player cache.
- Session-state storage for latest recommendations, availability, future impact, championship equity, and consensus.
- `st.cache_resource` for immutable Draft Mode runtime model limits and Draft Lab option/config lookups.

Cache invalidation:

- Draft Mode cache key includes current pick, league size, draft slot, roster, drafted players, queue, and roster settings.
- Player dataset cache invalidates on source file mtime.
- Available-player cache invalidates on player DataFrame hash and drafted tuple.
- Draft Lab cache invalidates on players DataFrame, settings, strategy, slot, league size, batch size, and seed.

## Remaining Recommendations

1. Add a shared `DraftComputationContext` object so every page can reuse one canonical recommendations/equity bundle.
2. Cache tier tables directly in `draft_analysis`.
3. Migrate legacy `utils` pages to `draftkit.data_access`.
4. Add an app-level “Fast/Full” toggle for Draft Mode advanced models.
5. Store board sorted views by sort/filter/search keys.
6. Move long Draft Lab batches into background jobs if Streamlit responsiveness becomes a priority.

## Success Criteria Status

| Target | Status |
|---|---|
| Draft Mode actions under 1s | Expected after cache warmup; advanced V2 reduced substantially |
| Recommendation refresh under 1s | Expected after cache warmup |
| Draft player action under 1s | Expected after cache warmup because state-keyed outputs invalidate once |
| Page load under 2s | Draft Lab fixed; Draft Mode depends on first cold recommendation build |
| Draft Lab may remain slower | Preserved, now submit-driven |
