# Draft Window Edge Report

Date: 2026-07-07

Scope: research-only bucket validation using the existing WR/RB walk-forward model families. No Streamlit app changes, no UI, and no new model families.

## Method

For each season, position, target, model, and ADP bucket, the analysis compares the model's top decile within that bucket against the ADP baseline's top decile within the same bucket. This tests similarly priced players rather than the full draft board.

Only `Top24` and `Top12` targets are used for classification. ADP-defined underpriced labels are excluded from draft-window edge claims because they would bias the comparison against ADP.

## WR Bucket Summary

| bucket_type    | bucket   | best_target_in_bucket   | model_name                                        |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   seasons_model_beats_adp |   pct_seasons_model_beats_adp | classification   | small_sample_flag   |
|:---------------|:---------|:------------------------|:--------------------------------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|--------------------------:|------------------------------:|:-----------------|:--------------------|
| overall_adp    | 1-24     | WR_Top24                | age_prior_production_baseline_random_forest       |               15 |           157 |            0.913 |          0.783 |           0.130 |                         4 |                        26.667 | Not Useful       |                     |
| overall_adp    | 121+     | WR_Top24                | projections_only_logistic                         |               15 |           247 |            0.267 |          0.233 |           0.033 |                         2 |                        13.333 | Not Useful       |                     |
| overall_adp    | 25-48    | WR_Top24                | prior_production_only_baseline_logistic           |               15 |           193 |            0.741 |          0.630 |           0.111 |                         4 |                        26.667 | Not Useful       |                     |
| overall_adp    | 49-72    | WR_Top24                | adp_all_expanded_random_forest                    |               15 |           142 |            0.632 |          0.474 |           0.158 |                         7 |                        46.667 | Not Useful       |                     |
| overall_adp    | 73-96    | WR_Top24                | adp_sportsbook_expanded_features_random_forest    |               14 |           116 |            0.400 |          0.200 |           0.200 |                         4 |                        28.571 | Not Useful       |                     |
| overall_adp    | 97-120   | WR_Top24                | adp_role_opportunity_logistic                     |               14 |           129 |            0.500 |          0.350 |           0.150 |                         5 |                        35.714 | Not Useful       |                     |
| positional_adp | 1-12     | WR_Top24                | age_prior_production_baseline_random_forest       |               15 |           202 |            0.889 |          0.741 |           0.148 |                         5 |                        33.333 | Not Useful       |                     |
| positional_adp | 13-24    | WR_Top24                | age_prior_production_baseline_logistic            |               15 |           212 |            0.633 |          0.533 |           0.100 |                         5 |                        33.333 | Not Useful       |                     |
| positional_adp | 25-36    | WR_Top12                | adp_all_available_predraft_safe_features_logistic |               15 |           181 |            0.214 |          0.071 |           0.143 |                         4 |                        26.667 | Not Useful       |                     |
| positional_adp | 37-48    | WR_Top24                | adp_sportsbook_expanded_features_logistic         |               13 |           172 |            0.269 |          0.154 |           0.115 |                         3 |                        23.077 | Not Useful       |                     |
| positional_adp | 49+      | WR_Top24                | projections_only_regularized_logistic             |               14 |           249 |            0.333 |          0.333 |           0.000 |                         1 |                         7.143 | Not Useful       |                     |

WR answer: no tested draft range currently beats ADP with repeatable positive lift.

## RB Bucket Summary

| bucket_type    | bucket   | best_target_in_bucket   | model_name                                |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   seasons_model_beats_adp |   pct_seasons_model_beats_adp | classification   | small_sample_flag   |
|:---------------|:---------|:------------------------|:------------------------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|--------------------------:|------------------------------:|:-----------------|:--------------------|
| overall_adp    | 1-24     | RB_Top12                | adp_team_context_random_forest            |               14 |           173 |            0.680 |          0.560 |           0.120 |                         3 |                        21.429 | Not Useful       |                     |
| overall_adp    | 121+     | RB_Top24                | adp_team_context_logistic                 |               14 |           192 |            0.269 |          0.154 |           0.115 |                         5 |                        35.714 | Not Useful       |                     |
| overall_adp    | 25-48    | RB_Top24                | adp_only_baseline_logistic                |               14 |           112 |            0.800 |          0.533 |           0.267 |                         5 |                        35.714 | Not Useful       |                     |
| overall_adp    | 49-72    | RB_Top12                | projections_only_logistic                 |               15 |           109 |            0.267 |          0.067 |           0.200 |                         3 |                        20.000 | Not Useful       |                     |
| overall_adp    | 73-96    | RB_Top24                | adp_projection_expanded_features_logistic |               15 |           121 |            0.529 |          0.294 |           0.235 |                         4 |                        26.667 | Not Useful       |                     |
| overall_adp    | 97-120   | RB_Top24                | adp_role_opportunity_random_forest        |               15 |           101 |            0.375 |          0.188 |           0.188 |                         4 |                        26.667 | Not Useful       |                     |
| positional_adp | 1-12     | RB_Top12                | adp_team_context_random_forest            |               14 |           166 |            0.680 |          0.520 |           0.160 |                         4 |                        28.571 | Not Useful       |                     |
| positional_adp | 13-24    | RB_Top24                | adp_only_baseline_logistic                |               14 |           156 |            0.760 |          0.720 |           0.040 |                         4 |                        28.571 | Not Useful       |                     |
| positional_adp | 25-36    | RB_Top24                | adp_projections_random_forest             |               15 |           175 |            0.556 |          0.333 |           0.222 |                         7 |                        46.667 | Not Useful       |                     |
| positional_adp | 37-48    | RB_Top12                | adp_team_context_regularized_logistic     |               13 |           154 |            0.227 |          0.091 |           0.136 |                         4 |                        30.769 | Not Useful       |                     |
| positional_adp | 49+      | RB_Top24                | projections_only_logistic                 |               14 |           185 |            0.231 |          0.077 |           0.154 |                         4 |                        28.571 | Not Useful       |                     |

RB answer: no tested draft range currently beats ADP with repeatable positive lift.

## Required Answers

Does any WR draft range beat ADP?

- No, not by the repeatability standard. Any positive pockets are too small or not repeatable enough to classify as useful.

Does any RB draft range beat ADP?

- No, not by the repeatability standard. ADP remains stronger or the apparent lift is too unstable.

Is there any range where the model is at least Tie-Breaker Only?

- No.

Which ranges should be ignored?

- Ignore every bucket classified `Not Useful`, especially buckets where lift is negative or ADP wins more seasons than the model.

Which ranges deserve further research?

- These buckets had positive average lift but failed repeatability or sample-size gates:
| position   | bucket_type    | bucket   | best_target_in_bucket   |   seasons_tested |   sample_size |   lift_over_adp |   pct_seasons_model_beats_adp | small_sample_flag   |
|:-----------|:---------------|:---------|:------------------------|-----------------:|--------------:|----------------:|------------------------------:|:--------------------|
| WR         | overall_adp    | 1-24     | WR_Top24                |               15 |           157 |           0.130 |                        26.667 |                     |
| WR         | overall_adp    | 121+     | WR_Top24                |               15 |           247 |           0.033 |                        13.333 |                     |
| WR         | overall_adp    | 25-48    | WR_Top24                |               15 |           193 |           0.111 |                        26.667 |                     |
| WR         | overall_adp    | 49-72    | WR_Top24                |               15 |           142 |           0.158 |                        46.667 |                     |
| WR         | overall_adp    | 73-96    | WR_Top24                |               14 |           116 |           0.200 |                        28.571 |                     |
| WR         | overall_adp    | 97-120   | WR_Top24                |               14 |           129 |           0.150 |                        35.714 |                     |
| WR         | positional_adp | 1-12     | WR_Top24                |               15 |           202 |           0.148 |                        33.333 |                     |
| WR         | positional_adp | 13-24    | WR_Top24                |               15 |           212 |           0.100 |                        33.333 |                     |
| WR         | positional_adp | 25-36    | WR_Top12                |               15 |           181 |           0.143 |                        26.667 |                     |
| WR         | positional_adp | 37-48    | WR_Top24                |               13 |           172 |           0.115 |                        23.077 |                     |
| RB         | overall_adp    | 1-24     | RB_Top12                |               14 |           173 |           0.120 |                        21.429 |                     |
| RB         | overall_adp    | 121+     | RB_Top24                |               14 |           192 |           0.115 |                        35.714 |                     |
| RB         | overall_adp    | 25-48    | RB_Top24                |               14 |           112 |           0.267 |                        35.714 |                     |
| RB         | overall_adp    | 49-72    | RB_Top12                |               15 |           109 |           0.200 |                        20.000 |                     |
| RB         | overall_adp    | 73-96    | RB_Top24                |               15 |           121 |           0.235 |                        26.667 |                     |
| RB         | overall_adp    | 97-120   | RB_Top24                |               15 |           101 |           0.188 |                        26.667 |                     |
| RB         | positional_adp | 1-12     | RB_Top12                |               14 |           166 |           0.160 |                        28.571 |                     |
| RB         | positional_adp | 13-24    | RB_Top24                |               14 |           156 |           0.040 |                        28.571 |                     |
| RB         | positional_adp | 25-36    | RB_Top24                |               15 |           175 |           0.222 |                        46.667 |                     |
| RB         | positional_adp | 37-48    | RB_Top12                |               13 |           154 |           0.136 |                        30.769 |                     |
| RB         | positional_adp | 49+      | RB_Top24                |               14 |           185 |           0.154 |                        28.571 |                     |

What exact feature improvement is most likely to create edge next?

- Add true preseason projection and role-context fields that are not already captured by ADP: projected points, projected volume, depth-chart role, injury/suspension flags, rookie status, team implied points, and prior opportunity details such as RB carries/receptions and WR routes/air yards. The current features mostly rediscover what ADP already prices in.

Should we continue WR, RB, or both?

- Continue both for research, but do not integrate either. WR is closer to ADP in the full-pool result, while RB has stronger raw hit rates but is heavily priced by the market. The next useful test is feature improvement, not app integration.

## Output Files

- `research/validation_v1/wr_bucket_lift_analysis.csv`
- `research/validation_v1/rb_bucket_lift_analysis.csv`
- `research/validation_v1/draft_window_edge_report.md`