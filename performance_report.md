# Guaranteed Play Runtime Performance Report

Date: 2026-06-12

## Method

I ran a real runtime profiling audit with `scripts/run_runtime_profile.py`.

Because the bundled Python runtime does not include Streamlit, the profiling harness installs a lightweight Streamlit shim that supports `session_state`, `st.cache_data`, and `st.cache_resource`. This lets the actual engines run against the real local player data without launching the UI.

Raw output:

- `data/processed/runtime_profile_results.json`

Dataset and run shape:

- Raw player rows: 3,931
- Available player rows: 3,931
- Recommendation rows: 27
- Recommendation summary rows prepared: 8
- Consensus rows prepared: 12
- Championship Equity V2 candidates: 5
- Championship Equity V2 simulations: 35 per candidate
- Availability simulations: 35
- Future Impact candidates: 5
- Future Impact simulations: 50 per candidate
- Draft Lab batch measured: 10 simulations

The audit ran three Draft Mode-style passes: `cold`, `warm_1`, and `warm_2`, plus one Draft Lab batch.

## Timing Breakdown

| Engine / Function Group | Runs | Average ms | Max ms | Total ms | Runtime Share |
|---|---:|---:|---:|---:|---:|
| future_impact_simulation | 3 | 6504.96 | 6567.78 | 19514.87 | 65.08% |
| draft_lab_batch_10 | 1 | 4610.32 | 4610.32 | 4610.32 | 15.38% |
| recommendation_generation | 3 | 1068.11 | 1116.09 | 3204.34 | 10.69% |
| conviction_report | 3 | 434.06 | 464.38 | 1302.18 | 4.34% |
| availability_prediction | 3 | 118.60 | 120.82 | 355.79 | 1.19% |
| championship_equity_v2 | 3 | 116.36 | 122.34 | 349.07 | 1.16% |
| consensus_generation | 3 | 95.49 | 97.10 | 286.46 | 0.96% |
| draft_strategy | 3 | 56.50 | 60.12 | 169.49 | 0.57% |
| roster_construction_evaluation | 3 | 55.55 | 57.49 | 166.64 | 0.56% |
| dataframe_filtering | 3 | 4.44 | 6.28 | 13.31 | 0.04% |
| data_loading | 3 | 2.37 | 6.70 | 7.12 | 0.02% |
| rendering_preparation | 3 | 1.86 | 2.22 | 5.59 | 0.02% |

## Ranked Runtime Bottlenecks

### 1. Future Impact Simulation

Measured:

- Average: 6504.96 ms
- Max: 6567.78 ms
- Share: 65.08%

Inputs:

- 5 candidates
- 50 simulations
- 3,931-player pool

Finding:

This is still by far the slowest measured path. Even after the previous sprint reduced candidate count and simulation count, `build_simulation_report()` / `simulate_draft_paths()` remains expensive. The likely cause is repeated future-board simulation over a large available-player DataFrame for every candidate and simulation.

Cache behavior:

- This is acceptable only when the Streamlit cache hits.
- Any draft-state change or toggle cache miss costs roughly 6.5 seconds.

Recommended fixes:

- Keep Future Impact disabled by default in Draft Mode.
- Add a second “Deep Future Impact” path for Draft Lab only.
- For Draft Mode, replace path simulation with a precomputed opponent pick probability table.
- Pre-sample draft paths once per draft state, then evaluate candidates against the same path set.
- Restrict future-impact targets to top 3 candidates if it must run live.

Estimated savings:

- Top 3 candidates instead of 5: about 35-45% savings.
- Shared pre-sampled paths: likely 50-75% savings.
- Replacing full path sim with probability lookup: likely 80-90% savings.

### 2. Draft Lab Batch

Measured:

- 10 simulations: 4610.32 ms
- Average per draft simulation: about 461 ms
- Share in total measured run: 15.38%

Finding:

This is acceptable because Draft Lab is now submit-driven and not part of live Draft Mode. The cost is dominated by full draft simulation plus final roster grade and Championship Equity simulation.

Cache behavior:

- Results persist in `st.session_state` after running.
- Re-running with same settings uses page cache.

Recommended fixes:

- Keep as-is for now.
- For 100/500 batches, consider reducing final roster equity simulations or performing grading in a vectorized approximation.

Estimated savings:

- Replacing per-draft final equity simulation with a grade proxy could reduce Draft Lab batch time by 30-60%.

### 3. Recommendation Generation

Measured:

- Average: 1068.11 ms
- Max: 1116.09 ms
- Share: 10.69%

Finding:

This is the largest always-needed Draft Mode cost. It is not catastrophic, but a cache miss is already above the target for a full recommendation refresh. The recommendation engine performs repeated sorting, merging, construction-pressure work, tier urgency work, trust adjustments, and single-QB adjustments.

Cache behavior:

- Page-level cache avoids recomputation when draft key is unchanged.
- Any draft-state change invalidates it, so the miss cost matters.

Repeated computations:

- Tier tables and position urgency are rebuilt inside recommendation generation and elsewhere.
- Construction pressure is computed inside the recommendation path and again later for draft strategy.
- Signal trust and championship/trust-related lookups are built through DataFrame merges.

Recommended fixes:

- Cache tier tables by available-player pool and roster settings.
- Cache construction pressure by roster position counts and round.
- Convert signal trust/equity joins to keyed lookup maps where possible.
- Split recommendation generation into static player features and dynamic draft-state adjustments.

Estimated savings:

- Caching tier/construction subresults: 150-300 ms.
- Static feature table reuse: 300-500 ms.
- Lookup-map replacement for repeated merges: 50-150 ms.

### 4. Conviction Report

Measured:

- Average: 434.06 ms
- Max: 464.38 ms
- Share: 4.34%

Finding:

Conviction is moderately expensive relative to its display role. It likely recomputes or merges equity/trust/recommendation context.

Cache behavior:

- Page wrapper now caches conviction by recommendation records.

Recommended fixes:

- Consume already-built recommendation/equity/trust lookups from the Draft Mode computation context.
- Do not rebuild static championship equity inside conviction if V2 equity is already available.

Estimated savings:

- 200-350 ms on cache misses.

### 5. Availability Prediction

Measured:

- Average: 118.60 ms
- Max: 120.82 ms
- Share: 1.19%

Finding:

Availability fast mode is now acceptable. It still returns 3,931 rows, so rendering or converting the whole result can become wasteful even if simulation is cheap.

Recommended fixes:

- Return top-N availability rows for Draft Mode where possible.
- Build dict lookup only for recommendation candidates and alternatives.

Estimated savings:

- 20-50 ms.

### 6. Championship Equity V2

Measured:

- Average: 116.36 ms
- Max: 122.34 ms
- Share: 1.16%

Inputs:

- Top 5 candidates
- 35 simulations per candidate

Finding:

Fast-mode Championship Equity V2 is no longer a major bottleneck. The previous forced 250-simulation minimum was successfully removed.

Recommended fixes:

- Keep current Draft Mode settings.
- Keep 250+ only for Draft Lab and deeper evaluation contexts.

Estimated savings:

- No immediate fix needed.

### 7. Consensus Generation

Measured:

- Average: 95.49 ms
- Max: 97.10 ms
- Share: 0.96%

Finding:

Consensus is not a meaningful runtime problem. It is DataFrame and ranking work over 12 candidates.

Recommended fixes:

- Keep cached.
- Avoid JSON/DataFrame conversions inside Streamlit wrappers if a shared computation context is introduced.

Estimated savings:

- 20-40 ms.

### 8. Draft Strategy and Roster Construction

Measured:

- Draft strategy average: 56.50 ms
- Roster construction average: 55.55 ms

Finding:

Both are acceptable but slightly duplicated. Construction pressure is evaluated separately in recommendation generation, explicit construction evaluation, and strategy.

Recommended fixes:

- Cache construction pressure by roster counts, settings, and round.
- Pass construction report into strategy/recommendation consumers.

Estimated savings:

- 50-100 ms per cache miss.

### 9. DataFrame Filtering, Data Loading, Rendering Preparation

Measured:

- Dataframe filtering average: 4.44 ms
- Data loading average: 2.37 ms
- Rendering preparation average: 1.86 ms

Finding:

These are no longer bottlenecks in the measured engine path. The cached available-player filtering is working well.

Recommended fixes:

- No immediate action required.

## Cache Misses and Repeated Computations

### Cache Misses Observed

The profiler intentionally called the engines directly, so the measured values represent raw cache-miss engine costs rather than Streamlit page cache hits.

Remaining expensive cache misses:

- Future Impact: about 6.5s
- Recommendation generation: about 1.1s
- Conviction report: about 0.43s

Meaning:

- Warm UI reruns should feel fast if Streamlit page caches hit.
- Any draft action that changes the draft compute key will still pay at least recommendation-generation miss cost.
- If Future Impact is enabled, any relevant draft-state change will still feel slow.

### Repeated Computations Identified

1. Construction pressure:
   - Used in recommendation generation.
   - Used in draft strategy.
   - Measured separately at about 55 ms.

2. Tier/urgency tables:
   - Built in recommendation generation.
   - Used by warning/falloff helpers.

3. Conviction/equity/trust context:
   - Conviction report can rebuild or rejoin context already represented in recommendation/equity outputs.

4. Future path simulation:
   - Candidate simulations repeatedly estimate future board states.
   - Paths are not shared across candidate evaluations.

## Expensive DataFrame Operations

Measured DataFrame filtering is cheap, but recommendation generation still likely spends meaningful time in DataFrame operations:

- repeated `sort_values` in recommendation/tier/value builders
- DataFrame merges for ADP value and trust/equity context
- repeated lookup DataFrame construction for conviction

Recommended fixes:

- Split master recommendation into:
  - static player feature table
  - dynamic draft-state adjustment table
  - final lightweight scoring join
- Cache tier and urgency tables.
- Replace repeated DataFrame merges with keyed dict lookups where table size is small.

## Expensive Simulation Loops

### Future Impact

Current measured loop:

- 5 candidates x 50 simulations
- about 6.5 seconds total
- about 26 ms per candidate-simulation unit

This is the only remaining live-path simulation that violates the responsiveness goal.

Recommended next implementation:

1. Generate opponent draft paths once per draft state.
2. Store path outcomes as sets of lost/surviving player keys.
3. Evaluate each candidate against the precomputed paths.
4. Reduce Draft Mode to 3 candidates or 25 simulations if still needed.

### Championship Equity V2

Current measured loop:

- 5 candidates x 35 simulations
- about 116 ms total

This is acceptable.

### Availability

Current measured loop:

- 35 simulations over full available pool
- about 119 ms total

This is acceptable.

## Recommended Fix Priority

1. Rework Future Impact simulation to share pre-sampled paths.
   - Estimated savings: 3.5-5.5s.

2. Keep Future Impact fully manual in Draft Mode and never trigger it from draft actions.
   - Estimated savings on normal draft actions: 6.5s when users leave it off.

3. Cache tier, urgency, and construction subresults inside recommendation generation.
   - Estimated savings: 250-600 ms.

4. Split recommendation generation into static and dynamic phases.
   - Estimated savings: 300-700 ms on draft-state changes.

5. Make conviction consume existing lookups instead of rebuilding context.
   - Estimated savings: 200-350 ms.

6. Avoid converting full availability output to dict for all 3,931 rows.
   - Estimated savings: 20-50 ms.

## Success Criteria Assessment

| Scenario | Measured / Inferred Runtime | Status |
|---|---:|---|
| Draft Mode normal warm rerun with caches | Expected under 1s | Likely pass |
| Draft Mode recommendation cache miss | 1.1s recommendation + 0.43s conviction + small engines | Borderline |
| Draft Mode with Future Impact enabled | 6.5s Future Impact alone | Fail |
| Draft player action with Future Impact off | likely near 1-2s on cache miss | Needs recommendation subcache work |
| Draft player action with Future Impact on | 6s+ | Fail |
| Championship Equity V2 fast mode | 116 ms | Pass |
| Availability fast mode | 119 ms | Pass |
| Draft Lab batch 10 | 4.6s | Acceptable for lab |

## Conclusion

The optimization sprint successfully moved Championship Equity V2, availability, consensus, data loading, and rendering preparation out of the critical bottleneck zone.

The remaining live-path risk is Future Impact. It is still too expensive for Draft Mode if enabled during drafting. Recommendation generation is the next meaningful cache-miss cost, averaging 1.07s, and should be split into cached static features plus dynamic draft-state adjustments.
