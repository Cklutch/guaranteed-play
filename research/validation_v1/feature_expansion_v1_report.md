# Feature Expansion V1 Report

Date: 2026-07-07

Scope: research-only feature expansion in `research/validation_v1`. No Streamlit app changes, no UI, and no new model families.

## Executive Summary

Preseason Feature Expansion V1 added the historical features that were actually available locally and marked the requested-but-missing features as unavailable. Historical projection data was not found, so projection feature columns remain empty and current 2026 app projections were excluded as unsafe for historical validation.

The expanded features did not create a repeatable edge over ADP. WR and RB full-pool validation still have negative average lift over ADP, and the expanded draft-window analysis still has zero buckets classified as Tie-Breaker Only, Strong Draft Signal, or App-Ready.

## Features Added Or Populated

| feature_name                          | source                                  | safety_classification                 | status        |   coverage_rows |   coverage_pct |
|:--------------------------------------|:----------------------------------------|:--------------------------------------|:--------------|----------------:|---------------:|
| adp_team                              | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1881 |          18.57 |
| adp_source_total_drafts               | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1900 |          18.75 |
| adp_times_drafted                     | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1900 |          18.75 |
| adp_high                              | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1900 |          18.75 |
| adp_low                               | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1900 |          18.75 |
| adp_stdev                             | FFCalc raw ADP archive                  | pre-draft-safe                        | usable        |            1900 |          18.75 |
| adp_uncertainty_score                 | derived from FFCalc raw ADP             | pre-draft-safe                        | usable        |            1900 |          18.75 |
| team_position_adp_count               | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1881 |          18.57 |
| same_team_better_adp_count            | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1881 |          18.57 |
| same_team_top48_count                 | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1881 |          18.57 |
| same_team_top120_count                | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1881 |          18.57 |
| teammate_best_adp                     | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1619 |          15.98 |
| teammate_second_best_adp              | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |             613 |           6.05 |
| adp_gap_to_next_teammate              | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |             907 |           8.95 |
| target_competition_score              | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |            1021 |          10.08 |
| backfield_competition_score           | derived from FFCalc raw ADP team        | pre-draft-safe                        | usable        |             860 |           8.49 |
| prior_wr_targets_expanded             | WR opportunity study shifted one season | pre-draft-safe                        | usable        |            1447 |          14.28 |
| prior_wr_target_share_expanded        | WR opportunity study shifted one season | pre-draft-safe                        | usable        |            1447 |          14.28 |
| prior_wr_team_targets_expanded        | WR opportunity study shifted one season | pre-draft-safe                        | usable        |            1447 |          14.28 |
| prior_wr_games_expanded               | WR opportunity study shifted one season | pre-draft-safe                        | usable        |            1447 |          14.28 |
| prior_wr_fantasy_ppg_expanded         | WR opportunity study shifted one season | pre-draft-safe but production-derived | usable        |            1447 |          14.28 |
| prior_team_from_opportunity           | WR opportunity study shifted one season | pre-draft-safe                        | usable        |            1447 |          14.28 |
| adp_team_change_from_prior_adp        | FFCalc raw ADP archive                  | pre-draft-safe proxy                  | usable        |               0 |           0    |
| wr_team_change_from_prior_opportunity | FFCalc plus prior WR opportunity        | pre-draft-safe                        | usable        |             443 |           4.37 |
| player_first_dataset_season           | local historical outcome cache          | questionable proxy                    | questionable  |           10131 |         100    |
| years_in_league_proxy                 | local historical outcome cache          | questionable proxy                    | questionable  |           10131 |         100    |
| rookie_or_first_year_flag             | local historical outcome cache          | questionable proxy                    | questionable  |           10131 |         100    |
| age_bucket_code                       | age field                               | pre-draft-safe                        | usable        |           10131 |         100    |
| prior_games_missed_proxy              | prior season games                      | pre-draft-safe if games available     | poor coverage |            1447 |          14.28 |

## Coverage By Position

| feature_name                          |    ALL |     RB |     WR |
|:--------------------------------------|-------:|-------:|-------:|
| adp_gap_to_next_teammate              |   8.95 |   9.12 |   8.84 |
| adp_high                              |  18.75 |  21.13 |  17.13 |
| adp_low                               |  18.75 |  21.13 |  17.13 |
| adp_source_total_drafts               |  18.75 |  21.13 |  17.13 |
| adp_stdev                             |  18.75 |  21.13 |  17.13 |
| adp_team                              |  18.57 |  20.86 |  16.99 |
| adp_team_change_from_prior_adp        |   0.00 |   0.00 |   0.00 |
| adp_times_drafted                     |  18.75 |  21.13 |  17.13 |
| adp_uncertainty_score                 |  18.75 |  21.13 |  17.13 |
| age_bucket_code                       | 100.00 | 100.00 | 100.00 |
| backfield_competition_score           |   8.49 |  20.86 |   0.00 |
| player_first_dataset_season           | 100.00 | 100.00 | 100.00 |
| prior_games_missed_proxy              |  14.28 |   0.00 |  24.08 |
| prior_team_from_opportunity           |  14.28 |   0.00 |  24.08 |
| prior_wr_fantasy_ppg_expanded         |  14.28 |   0.00 |  24.08 |
| prior_wr_games_expanded               |  14.28 |   0.00 |  24.08 |
| prior_wr_target_share_expanded        |  14.28 |   0.00 |  24.08 |
| prior_wr_targets_expanded             |  14.28 |   0.00 |  24.08 |
| prior_wr_team_targets_expanded        |  14.28 |   0.00 |  24.08 |
| rookie_or_first_year_flag             | 100.00 | 100.00 | 100.00 |
| same_team_better_adp_count            |  18.57 |  20.86 |  16.99 |
| same_team_top120_count                |  18.57 |  20.86 |  16.99 |
| same_team_top48_count                 |  18.57 |  20.86 |  16.99 |
| target_competition_score              |  10.08 |   0.00 |  16.99 |
| team_position_adp_count               |  18.57 |  20.86 |  16.99 |
| teammate_best_adp                     |  15.98 |  17.34 |  15.05 |
| teammate_second_best_adp              |   6.05 |   4.71 |   6.97 |
| wr_team_change_from_prior_opportunity |   4.37 |   0.00 |   7.37 |
| years_in_league_proxy                 | 100.00 | 100.00 | 100.00 |

## Requested Features Unavailable Or Excluded

| feature_name                             | source                                                        | safety_classification                                                 | status      |
|:-----------------------------------------|:--------------------------------------------------------------|:----------------------------------------------------------------------|:------------|
| projected_fantasy_points                 | Historical projection file not found                          | pre-draft-safe if sourced historically                                | unavailable |
| projected_positional_rank                | Historical projection file not found                          | pre-draft-safe if sourced historically                                | unavailable |
| projected_points_over_adp_expectation    | Requires historical projections                               | pre-draft-safe if sourced historically                                | unavailable |
| projection_minus_adp_implied_expectation | Requires historical projections                               | pre-draft-safe if sourced historically                                | unavailable |
| prior_wr_late_season_target_growth       | Weekly historical target split not found                      | pre-draft-safe if prior-season weekly data                            | unavailable |
| projected_targets                        | Historical projected target file not found                    | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_receptions                     | Historical projected reception file not found                 | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_receiving_yards                | Historical projected receiving-yard file not found            | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_receiving_tds                  | Historical projected receiving-TD file not found              | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_carries                        | Historical projected carry file not found                     | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_rushing_yards                  | Historical projected rushing-yard file not found              | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_total_tds                      | Historical projected TD file not found                        | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_routes                             | No local historical route data found                          | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_route_participation                | No local historical route participation data found            | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_air_yards                          | No local historical air-yards data found                      | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_snap_share                         | No local historical snap share data found                     | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_red_zone_usage                     | No local historical red-zone usage data found                 | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_goal_line_usage                    | No local historical goal-line usage data found                | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| team_implied_points                      | No historical preseason betting/team implied point file found | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| projected_team_points                    | No historical projected team scoring file found               | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_team_rush_attempts                 | No local team rush attempts file found                        | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| prior_team_touchdowns                    | No local team TD file found                                   | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| offensive_pace                           | No local pace file found                                      | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| qb_change_flag                           | No historical QB/team continuity file found                   | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| coach_change_flag                        | No historical coach/coordinator file found                    | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| draft_capital                            | No historical NFL draft capital file found                    | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| injury_flag_entering_season              | No historical preseason injury file found                     | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |
| suspension_flag_entering_season          | No historical preseason suspension file found                 | pre-draft-safe if sourced historically; unsafe if current/post-season | unavailable |

## Validation Groups Tested

The existing logistic, regularized logistic, and random forest model families were reused. No new model family was added.

Feature groups compared:

- ADP only
- projections only
- ADP + projections
- ADP + prior production
- ADP + role/opportunity features
- ADP + team context
- ADP + all expanded features

Projection-only rows mostly skipped because historical projection inputs are unavailable. ADP + projections therefore behaves like ADP-only in this run.

## WR Full-Pool Results

| target   | expanded_feature_group        |   seasons |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   auc |   adp_auc |   seasons_model_beats_adp |   pct_seasons_model_beats_adp |
|:---------|:------------------------------|----------:|-----------------:|---------------:|----------------:|------:|----------:|--------------------------:|------------------------------:|
| WR_Top12 | adp_team_context              |        15 |            0.320 |          0.348 |          -0.011 | 0.767 |     0.794 |                         6 |                        40.000 |
| WR_Top12 | adp_only_baseline             |        14 |            0.333 |          0.348 |          -0.015 | 0.778 |     0.794 |                         3 |                        21.429 |
| WR_Top12 | adp_projections               |        14 |            0.333 |          0.348 |          -0.015 | 0.778 |     0.794 |                         3 |                        21.429 |
| WR_Top24 | adp_prior_production_baseline |        25 |            0.528 |          0.567 |          -0.036 | 0.841 |     0.750 |                         6 |                        24.000 |
| WR_Top24 | adp_role_opportunity          |        15 |            0.521 |          0.565 |          -0.042 | 0.836 |     0.749 |                         4 |                        26.667 |
| WR_Top24 | adp_only_baseline             |        14 |            0.521 |          0.565 |          -0.044 | 0.773 |     0.749 |                         3 |                        21.429 |

WR answer: expanded features did not improve WR enough to beat ADP. The best WR full-pool rows are still negative on average lift over ADP.

## RB Full-Pool Results

| target   | expanded_feature_group   |   seasons |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   auc |   adp_auc |   seasons_model_beats_adp |   pct_seasons_model_beats_adp |
|:---------|:-------------------------|----------:|-----------------:|---------------:|----------------:|------:|----------:|--------------------------:|------------------------------:|
| RB_Top12 | adp_team_context         |        14 |            0.445 |          0.455 |          -0.010 | 0.807 |     0.799 |                         6 |                        42.857 |
| RB_Top12 | adp_only_baseline        |        14 |            0.438 |          0.455 |          -0.017 | 0.781 |     0.799 |                         5 |                        35.714 |
| RB_Top12 | adp_projections          |        14 |            0.438 |          0.455 |          -0.017 | 0.781 |     0.799 |                         5 |                        35.714 |
| RB_Top24 | adp_only_baseline        |        14 |            0.651 |          0.677 |          -0.026 | 0.768 |     0.770 |                         6 |                        42.857 |
| RB_Top24 | adp_projections          |        14 |            0.651 |          0.677 |          -0.026 | 0.768 |     0.770 |                         6 |                        42.857 |
| RB_Top24 | adp_role_opportunity     |        25 |            0.636 |          0.680 |          -0.032 | 0.823 |     0.774 |                         4 |                        16.000 |

RB answer: expanded team/context features helped RB Top12 get closer to ADP, but the result is still negative lift and not repeatable enough for a usable signal.

## Draft Window Results

Best WR bucket rows after expansion:

| bucket_type    | bucket   | best_target_in_bucket   | model_name                                  |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   pct_seasons_model_beats_adp | classification   |
|:---------------|:---------|:------------------------|:--------------------------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|------------------------------:|:-----------------|
| overall_adp    | 1-24     | WR_Top24                | age_prior_production_baseline_random_forest |               15 |           157 |            0.913 |          0.783 |           0.130 |                        26.667 | Not Useful       |
| overall_adp    | 121+     | WR_Top12                | adp_team_context_logistic                   |               14 |           225 |            0.037 |          0.037 |           0.000 |                         7.143 | Not Useful       |
| overall_adp    | 25-48    | WR_Top24                | prior_production_only_baseline_logistic     |               15 |           193 |            0.741 |          0.630 |           0.111 |                        26.667 | Not Useful       |
| overall_adp    | 49-72    | WR_Top24                | adp_all_expanded_random_forest              |               15 |           142 |            0.579 |          0.474 |           0.105 |                        33.333 | Not Useful       |
| overall_adp    | 73-96    | WR_Top24                | adp_role_opportunity_random_forest          |               14 |           116 |            0.333 |          0.200 |           0.133 |                        21.429 | Not Useful       |
| overall_adp    | 97-120   | WR_Top24                | adp_role_opportunity_logistic               |               14 |           129 |            0.500 |          0.350 |           0.150 |                        35.714 | Not Useful       |
| positional_adp | 1-12     | WR_Top24                | age_prior_production_baseline_random_forest |               15 |           202 |            0.889 |          0.741 |           0.148 |                        33.333 | Not Useful       |
| positional_adp | 13-24    | WR_Top24                | age_prior_production_baseline_logistic      |               15 |           212 |            0.633 |          0.533 |           0.100 |                        33.333 | Not Useful       |
| positional_adp | 25-36    | WR_Top12                | adp_team_context_random_forest              |               14 |           168 |            0.192 |          0.077 |           0.115 |                        28.571 | Not Useful       |
| positional_adp | 37-48    | WR_Top24                | adp_prior_production_baseline_logistic      |               14 |           185 |            0.250 |          0.143 |           0.107 |                        35.714 | Not Useful       |
| positional_adp | 49+      | WR_Top12                | adp_prior_production_baseline_random_forest |               14 |           249 |            0.067 |          0.100 |          -0.033 |                         0.000 | Not Useful       |

Best RB bucket rows after expansion:

| bucket_type    | bucket   | best_target_in_bucket   | model_name                            |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   pct_seasons_model_beats_adp | classification   |
|:---------------|:---------|:------------------------|:--------------------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|------------------------------:|:-----------------|
| overall_adp    | 1-24     | RB_Top12                | adp_team_context_random_forest        |               14 |           173 |            0.680 |          0.560 |           0.120 |                        21.429 | Not Useful       |
| overall_adp    | 121+     | RB_Top24                | adp_team_context_logistic             |               14 |           192 |            0.269 |          0.154 |           0.115 |                        35.714 | Not Useful       |
| overall_adp    | 25-48    | RB_Top24                | adp_only_baseline_logistic            |               14 |           112 |            0.800 |          0.533 |           0.267 |                        35.714 | Not Useful       |
| overall_adp    | 49-72    | RB_Top24                | adp_only_baseline_logistic            |               14 |           101 |            0.500 |          0.357 |           0.143 |                        21.429 | Not Useful       |
| overall_adp    | 73-96    | RB_Top12                | adp_only_baseline_random_forest       |               14 |           112 |            0.250 |          0.062 |           0.188 |                        21.429 | Not Useful       |
| overall_adp    | 97-120   | RB_Top24                | adp_role_opportunity_random_forest    |               15 |           101 |            0.375 |          0.188 |           0.188 |                        26.667 | Not Useful       |
| positional_adp | 1-12     | RB_Top12                | adp_team_context_random_forest        |               14 |           166 |            0.680 |          0.520 |           0.160 |                        28.571 | Not Useful       |
| positional_adp | 13-24    | RB_Top24                | adp_only_baseline_logistic            |               14 |           156 |            0.760 |          0.720 |           0.040 |                        28.571 | Not Useful       |
| positional_adp | 25-36    | RB_Top24                | adp_role_opportunity_random_forest    |               15 |           175 |            0.556 |          0.333 |           0.222 |                        40.000 | Not Useful       |
| positional_adp | 37-48    | RB_Top12                | adp_team_context_regularized_logistic |               13 |           154 |            0.227 |          0.091 |           0.136 |                        30.769 | Not Useful       |
| positional_adp | 49+      | RB_Top24                | adp_only_baseline_logistic            |               13 |           168 |            0.167 |          0.083 |           0.083 |                        23.077 | Not Useful       |

WR buckets at least Tie-Breaker Only: 0

RB buckets at least Tie-Breaker Only: 0

No expanded draft range became repeatably positive by the classification rules.

## Required Answers

Which new features were successfully added?

- FFCalc ADP raw team metadata, ADP spread/uncertainty, same-team positional competition, WR prior targets/target share/team targets/games/PPG, WR team-change proxy, age bucket, years-in-data proxy, first-year proxy, and prior games-missed proxy for WR rows with prior opportunity data.

Which requested features were unavailable?

- Historical preseason projections, projected volume stats, routes, air yards, snap share, red-zone and goal-line usage, true vacated touches/targets, team implied points, projected team points, pace, QB/coach changes, draft capital, preseason injury flags, and suspension flags were not found locally as historical pre-draft-safe sources.

Which features had usable coverage?

- ADP raw metadata covers about the same 2010-2024 ADP-matched rows as the ADP baseline. WR prior opportunity covers 1,447 rows. Competition scores cover 1,021 WR rows and 860 RB rows. Projection features have 0 coverage.

Did expanded features improve WR lift over ADP?

- No. WR remains negative. Some bucket averages improved, but repeatability stayed weak.

Did expanded features improve RB lift over ADP?

- Slightly for RB Top12 team-context comparisons, but not enough to beat ADP. RB remains negative overall.

Did any draft range become repeatably positive?

- No. Every WR and RB bucket is still classified Not Useful.

Which feature group helped most?

- WR: ADP + prior production remains the best broad group, not the new expansion features. RB: ADP + team context was closest for RB Top12, but still negative. No group created usable edge.

Should we continue WR, RB, or both?

- Continue both for research only. WR is still closer to market in some full-pool comparisons; RB has larger positive bucket averages but weaker repeatability. Neither should be integrated.

What is the next research build?

- Historical Projection Import V1. The biggest missing ingredient is real preseason projections and projected volume from a dated historical source. Without those, the model mostly has ADP plus prior production, which the market already prices well.

## Output Files

- `research/validation_v1/build_feature_expansion_v1.py`
- `research/validation_v1/feature_expansion_inventory.csv`
- `research/validation_v1/predraft_validation_dataset_expanded.csv`
- `research/validation_v1/feature_coverage_report.csv`
- `research/validation_v1/feature_expansion_v1_report.md`
- `research/validation_v1/wr_validation_results.csv`
- `research/validation_v1/rb_validation_results.csv`
- `research/validation_v1/wr_bucket_lift_analysis.csv`
- `research/validation_v1/rb_bucket_lift_analysis.csv`
- `research/validation_v1/draft_window_edge_report.md`