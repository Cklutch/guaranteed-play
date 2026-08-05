# Age Curve Edge Score V1 Validation Report

Research-only. Do not promote this score into the app; do not claim app-readiness.

## Executive Summary

Final classification: **Weak Research Signal**. Some improvement exists, but season repeatability is weak.

## Targets Tested

WR_Beat_ADP_By_12, RB_Beat_ADP_By_12, WR_Underpriced_Top24, RB_Underpriced_Top24, WR_Underpriced_Top12, RB_Underpriced_Top12, Underpriced_Top24, Underpriced_Top12, Beat_ADP_By_12

## Validation Methodology

Used season-based walk-forward logistic regression comparing ADP-only, ADP + age_curve_edge_score, and age-only. Also ran bucket-level target rates and ADP draft-window comparisons where sample rules allowed.

## Bucket-Level Validation

| validation_type   | target               | position   | bucket_type           | bucket             |   rows |   positive_count |   positive_rate |   baseline_positive_rate |   lift_over_baseline |
|:------------------|:---------------------|:-----------|:----------------------|:-------------------|-------:|-----------------:|----------------:|-------------------------:|---------------------:|
| bucket            | WR_Beat_ADP_By_12    | WR         | age_curve_edge_bucket | Elite Age Window   |    231 |               39 |           0.169 |                    0.162 |                0.007 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_curve_edge_bucket | Major Age Risk     |    153 |               32 |           0.209 |                    0.162 |                0.047 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_curve_edge_bucket | Mild Age Risk      |    129 |               20 |           0.155 |                    0.162 |               -0.007 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_curve_edge_bucket | Neutral Age Window |    287 |               37 |           0.129 |                    0.162 |               -0.033 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_curve_edge_bucket | Strong Age Window  |    229 |               39 |           0.170 |                    0.162 |                0.008 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 0                  |    128 |               24 |           0.188 |                    0.162 |                0.025 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 1                  |    154 |               28 |           0.182 |                    0.162 |                0.020 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 2                  |    173 |               22 |           0.127 |                    0.162 |               -0.035 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 3                  |    114 |               15 |           0.132 |                    0.162 |               -0.031 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 4                  |    118 |               22 |           0.186 |                    0.162 |                0.024 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 6                  |    227 |               37 |           0.163 |                    0.162 |                0.001 |
| bucket            | WR_Beat_ADP_By_12    | WR         | age_score_decile      | 7                  |    115 |               19 |           0.165 |                    0.162 |                0.003 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_curve_edge_bucket | Elite Age Window   |    262 |               49 |           0.187 |                    0.170 |                0.017 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_curve_edge_bucket | Major Age Risk     |    230 |               35 |           0.152 |                    0.170 |               -0.018 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_curve_edge_bucket | Neutral Age Window |    160 |               27 |           0.169 |                    0.170 |               -0.001 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_curve_edge_bucket | Strong Age Window  |    219 |               37 |           0.169 |                    0.170 |               -0.001 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 0                  |     93 |               18 |           0.194 |                    0.170 |                0.024 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 1                  |    137 |               17 |           0.124 |                    0.170 |               -0.046 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 3                  |     79 |               10 |           0.127 |                    0.170 |               -0.043 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 4                  |     81 |               17 |           0.210 |                    0.170 |                0.040 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 5                  |    219 |               37 |           0.169 |                    0.170 |               -0.001 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 8                  |    129 |               21 |           0.163 |                    0.170 |               -0.007 |
| bucket            | RB_Beat_ADP_By_12    | RB         | age_score_decile      | 9                  |    133 |               28 |           0.211 |                    0.170 |                0.041 |
| bucket            | WR_Underpriced_Top24 | WR         | age_curve_edge_bucket | Elite Age Window   |    231 |               24 |           0.104 |                    0.102 |                0.002 |
| bucket            | WR_Underpriced_Top24 | WR         | age_curve_edge_bucket | Major Age Risk     |    153 |               19 |           0.124 |                    0.102 |                0.022 |
| bucket            | WR_Underpriced_Top24 | WR         | age_curve_edge_bucket | Mild Age Risk      |    129 |               15 |           0.116 |                    0.102 |                0.014 |
| bucket            | WR_Underpriced_Top24 | WR         | age_curve_edge_bucket | Neutral Age Window |    287 |               27 |           0.094 |                    0.102 |               -0.008 |
| bucket            | WR_Underpriced_Top24 | WR         | age_curve_edge_bucket | Strong Age Window  |    229 |               20 |           0.087 |                    0.102 |               -0.015 |
| bucket            | WR_Underpriced_Top24 | WR         | age_score_decile      | 0                  |    128 |               14 |           0.109 |                    0.102 |                0.007 |
| bucket            | WR_Underpriced_Top24 | WR         | age_score_decile      | 1                  |    154 |               20 |           0.130 |                    0.102 |                0.028 |

## ADP-Only vs ADP + Age Validation

| validation_type   | target               | position   | model                      | status    |   rows |   positive_rate |   seasons_tested |   auc |   top_decile_hit_rate |   baseline_hit_rate |   lift_over_baseline |   improvement_over_adp_only |   seasons_adp_age_beats_adp_only |   average_yearly_lift |   median_yearly_lift |
|:------------------|:---------------------|:-----------|:---------------------------|:----------|-------:|----------------:|-----------------:|------:|----------------------:|--------------------:|---------------------:|----------------------------:|---------------------------------:|----------------------:|---------------------:|
| walk_forward      | WR_Underpriced_Top12 | WR         | age_curve_edge_score only  | completed |   1029 |           0.078 |               14 | 0.461 |                 0.103 |               0.080 |                0.022 |                       0.103 |                                0 |                 0.022 |                0.003 |
| walk_forward      | RB_Underpriced_Top12 | RB         | age_curve_edge_score only  | completed |    871 |           0.079 |               14 | 0.506 |                 0.083 |               0.080 |                0.003 |                       0.083 |                                2 |                 0.003 |               -0.062 |
| walk_forward      | Underpriced_Top12    | ALL        | age_curve_edge_score only  | completed |   1900 |           0.078 |               14 | 0.511 |                 0.065 |               0.080 |               -0.015 |                       0.065 |                                2 |                -0.015 |               -0.007 |
| walk_forward      | Underpriced_Top24    | ALL        | age_curve_edge_score only  | completed |   1900 |           0.108 |               14 | 0.510 |                 0.124 |               0.107 |                0.017 |                       0.058 |                                2 |                 0.017 |                0.030 |
| walk_forward      | WR_Underpriced_Top24 | WR         | age_curve_edge_score only  | completed |   1029 |           0.102 |               14 | 0.529 |                 0.112 |               0.101 |                0.011 |                       0.050 |                                2 |                 0.011 |               -0.018 |
| walk_forward      | RB_Underpriced_Top24 | RB         | age_curve_edge_score only  | completed |    871 |           0.115 |               13 | 0.467 |                 0.134 |               0.121 |                0.012 |                       0.048 |                                1 |                 0.012 |                0.000 |
| walk_forward      | RB_Underpriced_Top12 | RB         | ADP + age_curve_edge_score | completed |    871 |           0.079 |               14 | 0.544 |                 0.022 |               0.080 |               -0.058 |                       0.022 |                                2 |                -0.058 |               -0.070 |
| walk_forward      | Beat_ADP_By_12       | ALL        | ADP + age_curve_edge_score | completed |   1900 |           0.166 |               14 | 0.718 |                 0.260 |               0.163 |                0.097 |                       0.015 |                                4 |                 0.097 |                0.058 |
| walk_forward      | Underpriced_Top12    | ALL        | ADP + age_curve_edge_score | completed |   1900 |           0.078 |               14 | 0.545 |                 0.010 |               0.080 |               -0.070 |                       0.010 |                                2 |                -0.070 |               -0.075 |
| walk_forward      | WR_Beat_ADP_By_12    | WR         | ADP + age_curve_edge_score | completed |   1029 |           0.162 |               14 | 0.713 |                 0.219 |               0.158 |                0.061 |                       0.009 |                                3 |                 0.061 |                0.083 |
| walk_forward      | WR_Underpriced_Top24 | WR         | ADP + age_curve_edge_score | completed |   1029 |           0.102 |               14 | 0.660 |                 0.070 |               0.101 |               -0.031 |                       0.008 |                                2 |                -0.031 |               -0.087 |
| walk_forward      | Underpriced_Top24    | ALL        | ADP-only                   | completed |   1900 |           0.108 |               14 | 0.646 |                 0.066 |               0.107 |               -0.041 |                       0.000 |                                2 |                -0.041 |               -0.041 |
| walk_forward      | RB_Underpriced_Top12 | RB         | ADP-only                   | completed |    871 |           0.079 |               14 | 0.552 |                 0.000 |               0.080 |               -0.080 |                       0.000 |                                2 |                -0.080 |               -0.074 |
| walk_forward      | WR_Underpriced_Top12 | WR         | ADP-only                   | completed |   1029 |           0.078 |               14 | 0.551 |                 0.000 |               0.080 |               -0.080 |                       0.000 |                                0 |                -0.080 |               -0.081 |
| walk_forward      | WR_Underpriced_Top12 | WR         | ADP + age_curve_edge_score | completed |   1029 |           0.078 |               14 | 0.544 |                 0.000 |               0.080 |               -0.080 |                       0.000 |                                0 |                -0.080 |               -0.081 |
| walk_forward      | WR_Underpriced_Top24 | WR         | ADP-only                   | completed |   1029 |           0.102 |               14 | 0.665 |                 0.062 |               0.101 |               -0.039 |                       0.000 |                                2 |                -0.039 |               -0.087 |
| walk_forward      | RB_Beat_ADP_By_12    | RB         | ADP-only                   | completed |    871 |           0.170 |               13 | 0.740 |                 0.374 |               0.180 |                0.194 |                       0.000 |                                1 |                 0.194 |                0.247 |
| walk_forward      | WR_Beat_ADP_By_12    | WR         | ADP-only                   | completed |   1029 |           0.162 |               14 | 0.717 |                 0.210 |               0.158 |                0.052 |                       0.000 |                                3 |                 0.052 |                0.084 |
| walk_forward      | RB_Underpriced_Top24 | RB         | ADP-only                   | completed |    871 |           0.115 |               13 | 0.624 |                 0.086 |               0.121 |               -0.035 |                       0.000 |                                1 |                -0.035 |               -0.068 |
| walk_forward      | Beat_ADP_By_12       | ALL        | ADP-only                   | completed |   1900 |           0.166 |               14 | 0.719 |                 0.246 |               0.163 |                0.083 |                       0.000 |                                4 |                 0.083 |                0.051 |
| walk_forward      | Underpriced_Top12    | ALL        | ADP-only                   | completed |   1900 |           0.078 |               14 | 0.555 |                 0.000 |               0.080 |               -0.080 |                       0.000 |                                2 |                -0.080 |               -0.078 |
| walk_forward      | RB_Underpriced_Top24 | RB         | ADP + age_curve_edge_score | completed |    871 |           0.115 |               13 | 0.619 |                 0.084 |               0.121 |               -0.037 |                      -0.002 |                                1 |                -0.037 |               -0.068 |
| walk_forward      | Underpriced_Top24    | ALL        | ADP + age_curve_edge_score | completed |   1900 |           0.108 |               14 | 0.646 |                 0.060 |               0.107 |               -0.047 |                      -0.006 |                                2 |                -0.047 |               -0.094 |
| walk_forward      | RB_Beat_ADP_By_12    | RB         | ADP + age_curve_edge_score | completed |    871 |           0.170 |               13 | 0.737 |                 0.361 |               0.180 |                0.181 |                      -0.013 |                                1 |                 0.181 |                0.259 |
| walk_forward      | WR_Beat_ADP_By_12    | WR         | age_curve_edge_score only  | completed |   1029 |           0.162 |               14 | 0.494 |                 0.158 |               0.158 |               -0.001 |                      -0.052 |                                3 |                -0.001 |               -0.029 |
| walk_forward      | Beat_ADP_By_12       | ALL        | age_curve_edge_score only  | completed |   1900 |           0.166 |               14 | 0.488 |                 0.164 |               0.163 |                0.002 |                      -0.081 |                                4 |                 0.002 |                0.032 |
| walk_forward      | RB_Beat_ADP_By_12    | RB         | age_curve_edge_score only  | completed |    871 |           0.170 |               13 | 0.493 |                 0.247 |               0.180 |                0.067 |                      -0.126 |                                1 |                 0.067 |                0.080 |

## Draft-Window Validation

| validation_type   | target               | position   | draft_window   | status    |   rows |   positive_rate |   adp_only_top_decile_hit_rate |   adp_age_top_decile_hit_rate |   improvement_over_adp_only |   seasons_tested |   seasons_adp_age_beats_adp_only |
|:------------------|:---------------------|:-----------|:---------------|:----------|-------:|----------------:|-------------------------------:|------------------------------:|----------------------------:|-----------------:|---------------------------------:|
| draft_window      | WR_Underpriced_Top24 | WR         | 73-96          | completed |    123 |           0.203 |                          0.375 |                         0.625 |                       0.250 |            4.000 |                            1.000 |
| draft_window      | RB_Underpriced_Top12 | RB         | 25-48          | completed |    121 |           0.198 |                          0.000 |                         0.250 |                       0.250 |            2.000 |                            1.000 |
| draft_window      | RB_Underpriced_Top24 | RB         | 73-96          | completed |    121 |           0.256 |                          0.333 |                         0.500 |                       0.167 |            3.000 |                            1.000 |
| draft_window      | Underpriced_Top24    | ALL        | 73-96          | completed |    244 |           0.230 |                          0.202 |                         0.369 |                       0.167 |           14.000 |                            5.000 |
| draft_window      | Underpriced_Top12    | ALL        | 73-96          | completed |    244 |           0.098 |                          0.083 |                         0.200 |                       0.117 |           10.000 |                            3.000 |
| draft_window      | WR_Underpriced_Top24 | WR         | 49-72          | completed |    142 |           0.169 |                          0.200 |                         0.300 |                       0.100 |            5.000 |                            1.000 |
| draft_window      | Underpriced_Top12    | ALL        | 25-48          | completed |    314 |           0.166 |                          0.083 |                         0.179 |                       0.095 |           14.000 |                            3.000 |
| draft_window      | Underpriced_Top24    | ALL        | 151+           | completed |    200 |           0.075 |                          0.048 |                         0.143 |                       0.095 |            7.000 |                            1.000 |
| draft_window      | WR_Underpriced_Top24 | WR         | 97-120         | completed |    139 |           0.187 |                          0.125 |                         0.188 |                       0.062 |            8.000 |                            2.000 |
| draft_window      | Underpriced_Top24    | ALL        | 49-72          | completed |    251 |           0.179 |                          0.250 |                         0.292 |                       0.042 |           12.000 |                            3.000 |
| draft_window      | RB_Beat_ADP_By_12    | RB         | 25-48          | completed |    121 |           0.091 |                          0.250 |                         0.250 |                       0.000 |            2.000 |                            0.000 |
| draft_window      | RB_Beat_ADP_By_12    | RB         | 151+           | completed |    104 |           0.298 |                          0.000 |                         0.000 |                       0.000 |            2.000 |                            0.000 |
| draft_window      | WR_Beat_ADP_By_12    | WR         | 151+           | completed |     96 |           0.281 |                          0.167 |                         0.167 |                       0.000 |            3.000 |                            0.000 |
| draft_window      | WR_Beat_ADP_By_12    | WR         | 49-72          | completed |    142 |           0.162 |                          0.250 |                         0.250 |                       0.000 |            4.000 |                            0.000 |
| draft_window      | Beat_ADP_By_12       | ALL        | 151+           | completed |    200 |           0.290 |                          0.212 |                         0.212 |                       0.000 |           11.000 |                            0.000 |
| draft_window      | Underpriced_Top12    | ALL        | 151+           | completed |    200 |           0.015 |                          0.000 |                         0.000 |                       0.000 |            2.000 |                            0.000 |
| draft_window      | Beat_ADP_By_12       | ALL        | 121-150        | completed |    253 |           0.261 |                          0.167 |                         0.167 |                       0.000 |           13.000 |                            1.000 |
| draft_window      | Underpriced_Top12    | ALL        | 1-24           | completed |    341 |           0.026 |                          0.000 |                         0.000 |                       0.000 |            8.000 |                            0.000 |
| draft_window      | WR_Underpriced_Top12 | WR         | 25-48          | completed |    193 |           0.145 |                          0.042 |                         0.042 |                       0.000 |           12.000 |                            0.000 |
| draft_window      | RB_Beat_ADP_By_12    | RB         | 97-120         | completed |    101 |           0.317 |                          0.500 |                         0.500 |                       0.000 |            2.000 |                            0.000 |
| draft_window      | RB_Beat_ADP_By_12    | RB         | 73-96          | completed |    121 |           0.198 |                          0.333 |                         0.333 |                       0.000 |            3.000 |                            0.000 |
| draft_window      | WR_Underpriced_Top24 | WR         | 121-150        | completed |    150 |           0.120 |                          0.062 |                         0.062 |                       0.000 |            8.000 |                            1.000 |
| draft_window      | RB_Underpriced_Top12 | RB         | 121-150        | completed |    103 |           0.039 |                          0.000 |                         0.000 |                       0.000 |            1.000 |                            0.000 |
| draft_window      | Underpriced_Top24    | ALL        | 25-48          | completed |    314 |           0.003 |                          0.000 |                         0.000 |                       0.000 |            1.000 |                            0.000 |
| draft_window      | WR_Underpriced_Top12 | WR         | 97-120         | completed |    139 |           0.072 |                          0.300 |                         0.300 |                       0.000 |            5.000 |                            0.000 |
| draft_window      | Underpriced_Top12    | ALL        | 49-72          | completed |    251 |           0.127 |                          0.167 |                         0.167 |                       0.000 |           12.000 |                            1.000 |
| draft_window      | RB_Underpriced_Top24 | RB         | 121-150        | completed |    103 |           0.107 |                          0.000 |                         0.000 |                       0.000 |            2.000 |                            0.000 |
| draft_window      | RB_Underpriced_Top12 | RB         | 1-24           | completed |    184 |           0.043 |                          0.000 |                         0.000 |                       0.000 |            7.000 |                            0.000 |
| draft_window      | WR_Underpriced_Top12 | WR         | 1-24           | completed |    157 |           0.006 |                          0.000 |                         0.000 |                       0.000 |            1.000 |                            0.000 |
| draft_window      | Underpriced_Top24    | ALL        | 121-150        | completed |    253 |           0.115 |                          0.030 |                         0.000 |                      -0.030 |           11.000 |                            0.000 |

## Season Repeatability

Repeatability is measured by seasons where ADP + age beats ADP-only in top-decile hit rate.

## Best Result

{"validation_type":"walk_forward","target":"RB_Underpriced_Top12","position":"RB","model":"ADP + age_curve_edge_score","status":"completed","rows":871,"positive_rate":0.0792192882,"seasons_tested":14,"auc":0.5438057988,"top_decile_hit_rate":0.0221088435,"baseline_hit_rate":0.0798414194,"lift_over_baseline":-0.0577325758,"improvement_over_adp_only":0.0221088435,"seasons_adp_age_beats_adp_only":2,"average_yearly_lift":-0.0577325758,"median_yearly_lift":-0.0695704779}

## Best Draft-Window Result

{"validation_type":"draft_window","target":"WR_Underpriced_Top24","position":"WR","draft_window":"73-96","status":"completed","rows":123,"positive_rate":0.2032520325,"adp_only_top_decile_hit_rate":0.375,"adp_age_top_decile_hit_rate":0.625,"improvement_over_adp_only":0.25,"seasons_tested":4.0,"seasons_adp_age_beats_adp_only":1.0}

## Whether Age Adds Independent Signal Beyond ADP

This is answered by `improvement_over_adp_only`. Positive values indicate ADP + age beat ADP-only in the tested walk-forward setup.

## Recommendation For Next Research Step

If the score shows useful lift, test it inside combined-feature models alongside projections and role features. If weak, keep it as a diagnostic/risk feature only and do not app-integrate.