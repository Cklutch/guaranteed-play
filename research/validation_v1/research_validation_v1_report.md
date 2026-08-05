# Research Validation V1 Report

Date: 2026-07-07

Folder: `research/validation_v1/`

## Executive Summary

Historical preseason ADP has now been sourced, imported, merged, and evaluated against the fitted WR/RB validation models. The ADP source used in this pass is Fantasy Football Calculator archived 12-team PPR ADP for seasons 2010-2024.

The important answer: the current WR/RB models do not beat ADP. They still show decent predictive AUC, especially when prior production is included, but their top-decile draft picks do not outperform an ADP-only market baseline on average. For a beginner fantasy player, that means the model can be interesting research, but it should not drive draft decisions yet. ADP remains the better main guide.

Current classification:

| Area | Classification | Reason |
| --- | --- | --- |
| WR models | Not Useful for app integration | Best WR target has negative average lift over ADP. |
| RB models | Not Useful for app integration | Best RB target has negative average lift over ADP. |
| Unified signal export | Not Useful for app integration | Real ADP fields exist, but rows are marked not app-ready because models do not beat ADP. |

No model is Tie-Breaker Only, Strong Draft Signal, or App-Ready.

## ADP Source

ADP file used:

- `research/validation_v1/historical_adp.csv`

ADP source:

- `FantasyFootballCalculator`

Raw source folder:

- `research/validation_v1/source_adp_raw/`

Source seasons:

- 2010-2024

Rows imported:

- 1,834 WR/RB ADP rows

Source quality checks:

| Check | Result |
| --- | ---: |
| Duplicate player-season-position rows before dedupe | 8 |
| Missing overall ADP rows | 0 |
| Invalid positions | 0 |
| Suspicious ADP values | 0 |

Important limitation: the public Fantasy Football Calculator endpoint returned no 2025 ADP data in this run, so 2025 is not covered by ADP.

## ADP Merge Coverage

Outcome dataset:

- `case_studies/data/qb_rb_te_wr_elite_age_player_seasons_ppr.csv`

Generated dataset:

- `research/validation_v1/predraft_validation_dataset.csv`

Dataset size:

| Item | Count |
| --- | ---: |
| Seasons used | 27 |
| Season range | 1999-2025 |
| Total WR/RB rows | 10,131 |
| WR rows | 6,008 |
| RB rows | 4,123 |
| Rows with ADP | 1,900 |
| WR rows with ADP | 1,029 / 6,008, 17.13% |
| RB rows with ADP | 871 / 4,123, 21.13% |

ADP is concentrated in 2010-2024. Within those seasons, WR coverage is commonly around 20-34% of outcome rows and RB coverage is commonly around 29-47%. This is expected because ADP only covers drafted fantasy-relevant players, while the outcome table contains many low-volume players.

Diagnostics updated:

- `research/validation_v1/adp_merge_coverage_by_season.csv`
- `research/validation_v1/adp_unmatched_player_examples.csv`
- `research/validation_v1/adp_market_rows_not_matched_examples.csv`
- `research/validation_v1/adp_merge_diagnostics.json`

## WR Results Vs ADP

Best WR market comparisons:

| Target | Best model by average lift | Avg model hit rate | Avg ADP hit rate | Avg lift over ADP | Avg AUC | ADP AUC | ADP seasons | Beats ADP seasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `WR_Top24` | ADP-only regularized logistic | 0.5310 | 0.5650 | -0.0340 | 0.7693 | 0.7488 | 14 | 1 |
| `WR_Top12` | ADP-only regularized logistic | 0.3419 | 0.3481 | -0.0062 | 0.7846 | 0.7945 | 14 | 1 |

Does WR beat ADP?

- No.

Best WR target:

- `WR_Top12` is closest to ADP, but still negative lift.

Beginner interpretation:

- For WRs, the model may rank good players reasonably well, but it is not finding enough better values than the draft market. If ADP says a WR is expensive, the model mostly agrees; when it disagrees, the disagreement has not paid off enough.

Classification:

- Not Useful for app integration.

## RB Results Vs ADP

Best RB market comparisons:

| Target | Best model by average lift | Avg model hit rate | Avg ADP hit rate | Avg lift over ADP | Avg AUC | ADP AUC | ADP seasons | Beats ADP seasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RB_Top24` | ADP-only random forest | 0.6555 | 0.6773 | -0.0218 | 0.7693 | 0.7698 | 14 | 3 |
| `RB_Top12` | ADP-only random forest | 0.4412 | 0.4554 | -0.0142 | 0.7420 | 0.7994 | 14 | 3 |

Does RB beat ADP?

- No.

Best RB target:

- `RB_Top12` is closer by lift, but still negative. `RB_Top24` has the higher raw hit rate, but ADP does better.

Beginner interpretation:

- RB production is easier to predict than WR production in raw terms, but the draft market already knows that. The model is not adding enough new information beyond ADP.

Classification:

- Not Useful for app integration.

## Required Answers

Does WR beat ADP?

- No. The best WR target still has negative average lift over ADP.

Does RB beat ADP?

- No. The best RB target still has negative average lift over ADP.

Which target works best?

- Closest to useful: `WR_Top12`, because it is only slightly below ADP by average lift.
- Best raw hit-rate target: `RB_Top24`, but ADP still beats it.

Which position has the stronger signal?

- Neither has a proven draft edge. RB has stronger raw hit rates, but WR is closer to ADP on lift. Since the goal is beating the market, neither position currently has a strong signal.

Which draft ranges are useful?

- None are ready to use. ADP bucket files now contain real non-missing ADP buckets, but the overall model-vs-ADP result is negative. Do not promote any draft range until bucket-level lift is positive and repeatable across seasons.

Is anything Tie-Breaker Only?

- No. Tie-Breaker Only requires at least slight positive lift over ADP. Current average lift is negative.

Is anything a Strong Draft Signal?

- No. Strong Draft Signal requires meaningful lift over ADP across multiple seasons and useful draft ranges.

Is anything App-Ready?

- No. App-Ready requires positive lift over ADP, stable coverage, useful draft ranges, and clear interpretability.

Can it ever become a main decision maker?

- Yes, but not in its current form. To become a main decision maker, it must repeatedly beat ADP across multiple seasons and in the draft ranges where managers actually face hard choices. The realistic path is: first become a tie-breaker between similarly priced players, then prove consistent lift, then earn app integration.

## Unified Signal Export

Updated file:

- `research/validation_v1/unified_player_signal_export.csv`

Current export size:

| Item | Count |
| --- | ---: |
| Total signal rows | 13,800 |
| Rows with ADP | 8,247 |
| Rows without ADP | 5,553 |

Signal labels:

| Signal | Count | Meaning |
| --- | ---: | --- |
| `not_app_ready_does_not_beat_adp` | 8,247 | ADP is present, but validation did not prove edge. |
| `not_app_ready_no_historical_adp` | 5,553 | No ADP baseline exists for that row. |

The export contains the required market comparison fields:

- `ADP`
- `positional_adp`
- `model_score`
- `target_probability`
- `ADP_baseline_probability`
- `edge_over_ADP`
- `primary_signal`
- `risk_notes`
- `feature_explanation`

## Final Classification

| Position | Best current target | Classification | Why |
| --- | --- | --- | --- |
| WR | `WR_Top12` | Not Useful | Closest to ADP, but still negative lift. |
| RB | `RB_Top12` | Not Useful | Closest to ADP, but still negative lift. |

Nothing is Tie-Breaker Only.

Nothing is a Strong Draft Signal.

Nothing is App-Ready.

## Next Step

The next research step is not UI integration. The next step is better market validation:

1. Improve ADP coverage, especially 2025 and any missing/ambiguous name matches.
2. Evaluate narrower draft windows, such as picks 49-120 or positional ADP 25-48, where market inefficiencies are more plausible.
3. Add preseason projection data if it is clearly pre-season and source-dated.
4. Test whether model edge exists only as a tie-breaker between similarly priced players rather than across the whole player pool.

Recommended next Codex prompt:

```text
Continue research-only validation in `research/validation_v1`. Do not modify the Streamlit app. Using the ADP-backed dataset, analyze model lift over ADP by narrower draft ranges and positional ADP buckets for WR/RB. Identify whether any range has repeatable positive lift across multiple seasons. Do not build new model families and do not claim app readiness unless the model beats ADP consistently.
```
