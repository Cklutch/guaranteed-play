# Projection Validation V1 Report

Date: 2026-07-07

Scope: research-only validation. This does not modify the Streamlit app, add UI, build a recommendation engine, use 2025 projections, or claim model edge unless the completed ADP comparison supports it.

## Executive Summary

Projection data imported successfully from `FantasyPros_Wayback`. The FantasyPros Wayback source adds preseason projected volume for WR/RB seasons 2014-2024, but it does not include projected fantasy points.

WR classification: **Tie-Breaker Only**. Best projection result: WR_Beat_ADP_By_12 / adp_projection_expanded_features_logistic / adp_projection_expanded: model hit 0.211, ADP hit 0.030, lift 0.181, AUC 0.542, ADP AUC 0.283, beats ADP 13/15 seasons (86.7%).

RB classification: **Tie-Breaker Only**. Best projection result: RB_Beat_ADP_By_12 / adp_projection_expanded_features_logistic / adp_projection_expanded: model hit 0.213, ADP hit 0.017, lift 0.195, AUC 0.571, ADP AUC 0.261, beats ADP 11/15 seasons (73.3%).

No result is App-Ready because bucket/window projection validation was not completed after the refit-heavy bucket script timed out.

## Data Used

Projection file used: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\historical_projections.csv`
Projection source: `FantasyPros_Wayback`
Seasons covered: 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
Total normalized projection rows: `1473`
WR projection rows: `725`
RB projection rows: `748`
Base projected dataset: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_expanded.csv`

## Projection Source And Coverage

WR validation rows with projections: `760`
RB validation rows with projections: `714`

|   season | position   |   rows |   rows_with_projection |   projection_coverage_rate |
|---------:|:-----------|-------:|-----------------------:|---------------------------:|
|     1999 | RB         |    140 |                      0 |                      0.000 |
|     1999 | WR         |    180 |                      0 |                      0.000 |
|     2000 | RB         |    140 |                      0 |                      0.000 |
|     2000 | WR         |    188 |                      0 |                      0.000 |
|     2001 | RB         |    143 |                      0 |                      0.000 |
|     2001 | WR         |    199 |                      0 |                      0.000 |
|     2002 | RB         |    171 |                      0 |                      0.000 |
|     2002 | WR         |    230 |                      0 |                      0.000 |
|     2003 | RB         |    172 |                      0 |                      0.000 |
|     2003 | WR         |    236 |                      0 |                      0.000 |
|     2004 | RB         |    151 |                      0 |                      0.000 |
|     2004 | WR         |    199 |                      0 |                      0.000 |
|     2005 | RB         |    144 |                      0 |                      0.000 |
|     2005 | WR         |    203 |                      0 |                      0.000 |
|     2006 | RB         |    144 |                      0 |                      0.000 |
|     2006 | WR         |    206 |                      0 |                      0.000 |
|     2007 | RB         |    138 |                      0 |                      0.000 |
|     2007 | WR         |    203 |                      0 |                      0.000 |
|     2008 | RB         |    145 |                      0 |                      0.000 |
|     2008 | WR         |    216 |                      0 |                      0.000 |
|     2009 | RB         |    135 |                      0 |                      0.000 |
|     2009 | WR         |    204 |                      0 |                      0.000 |
|     2010 | RB         |    136 |                      0 |                      0.000 |
|     2010 | WR         |    213 |                      0 |                      0.000 |
|     2011 | RB         |    132 |                      0 |                      0.000 |
|     2011 | WR         |    224 |                      0 |                      0.000 |
|     2012 | RB         |    146 |                      0 |                      0.000 |
|     2012 | WR         |    225 |                      0 |                      0.000 |
|     2013 | RB         |    152 |                      0 |                      0.000 |
|     2013 | WR         |    217 |                      0 |                      0.000 |
|     2014 | RB         |    151 |                     94 |                      0.623 |
|     2014 | WR         |    224 |                     77 |                      0.344 |
|     2015 | RB         |    158 |                     79 |                      0.500 |
|     2015 | WR         |    232 |                     91 |                      0.392 |
|     2016 | RB         |    163 |                     83 |                      0.509 |
|     2016 | WR         |    223 |                     92 |                      0.413 |
|     2017 | RB         |    152 |                     57 |                      0.375 |
|     2017 | WR         |    226 |                     70 |                      0.310 |
|     2018 | RB         |    155 |                     57 |                      0.368 |
|     2018 | WR         |    238 |                     62 |                      0.261 |
|     2019 | RB         |    156 |                     62 |                      0.397 |
|     2019 | WR         |    247 |                     60 |                      0.243 |
|     2020 | RB         |    173 |                     59 |                      0.341 |
|     2020 | WR         |    256 |                     71 |                      0.277 |
|     2021 | RB         |    184 |                     61 |                      0.332 |
|     2021 | WR         |    268 |                     60 |                      0.224 |
|     2022 | RB         |    176 |                     54 |                      0.307 |
|     2022 | WR         |    242 |                     56 |                      0.231 |
|     2023 | RB         |    160 |                     52 |                      0.325 |
|     2023 | WR         |    224 |                     59 |                      0.263 |
|     2024 | RB         |    152 |                     56 |                      0.368 |
|     2024 | WR         |    239 |                     62 |                      0.259 |
|     2025 | RB         |    154 |                      0 |                      0.000 |
|     2025 | WR         |    246 |                      0 |                      0.000 |

## Available Projection Features

projected_receptions, projected_receiving_yards, projected_receiving_tds, projected_carries, projected_rushing_yards, projected_rushing_tds, projected_total_tds, projected_volume_score, projected_touch_score, projected_receiving_role_score

## Missing Projection Features

projected_fantasy_points, projected_targets, projected_games

## WR Validation Results

Best projection feature result: WR_Beat_ADP_By_12 / adp_projection_expanded_features_logistic / adp_projection_expanded: model hit 0.211, ADP hit 0.030, lift 0.181, AUC 0.542, ADP AUC 0.283, beats ADP 13/15 seasons (86.7%).

Best ADP-only result: WR_Beat_ADP_By_12 / adp_only_baseline_random_forest / adp_only_baseline: model hit 0.276, ADP hit 0.030, lift 0.246, AUC 0.717, ADP AUC 0.283, beats ADP 14/14 seasons (100.0%).

Best non-projection result: WR_Beat_ADP_By_12 / adp_only_baseline_logistic / adp_only: model hit 0.276, ADP hit 0.030, lift 0.246, AUC 0.717, ADP AUC 0.283, beats ADP 14/14 seasons (100.0%).

Projected volume improved over ADP-only by top aggregate lift: no.
Projected volume improved over prior non-projection models by top aggregate lift: no.
Does WR beat ADP by repeatability rule? yes.
WR Tie-Breaker Only? yes.
WR Strong Draft Signal? no.
WR App-Ready? no.

Top WR aggregate rows:

| target            | model_name                                   | feature_group                 |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   auc |   adp_auc |   seasons_model_beats_adp |   pct_seasons_model_beats_adp |
|:------------------|:---------------------------------------------|:------------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|------:|----------:|--------------------------:|------------------------------:|
| WR_Beat_ADP_By_12 | adp_only_baseline_logistic                   | adp_only                      |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_only_baseline_random_forest              | adp_only_baseline             |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_only_baseline_regularized_logistic       | adp_only_baseline             |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_sportsbook_features_logistic             | adp_sportsbook                |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_sportsbook_features_random_forest        | adp_sportsbook_features       |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_sportsbook_features_regularized_logistic | adp_sportsbook_features       |               14 |          3285 |            0.276 |          0.030 |           0.246 | 0.717 |     0.283 |                        14 |                       100.000 |
| WR_Beat_ADP_By_12 | adp_prior_production_baseline_logistic       | adp_prior_production          |               15 |          3531 |            0.272 |          0.030 |           0.241 | 0.679 |     0.283 |                        14 |                        93.333 |
| WR_Beat_ADP_By_12 | adp_prior_production_baseline_random_forest  | adp_prior_production_baseline |               15 |          3531 |            0.272 |          0.030 |           0.241 | 0.679 |     0.283 |                        14 |                        93.333 |

## RB Validation Results

Best projection feature result: RB_Beat_ADP_By_12 / adp_projection_expanded_features_logistic / adp_projection_expanded: model hit 0.213, ADP hit 0.017, lift 0.195, AUC 0.571, ADP AUC 0.261, beats ADP 11/15 seasons (73.3%).

Best ADP-only result: RB_Beat_ADP_By_12 / adp_only_baseline_random_forest / adp_only_baseline: model hit 0.301, ADP hit 0.017, lift 0.284, AUC 0.739, ADP AUC 0.261, beats ADP 14/14 seasons (100.0%).

Best non-projection result: RB_Beat_ADP_By_12 / adp_role_opportunity_logistic / adp_role: model hit 0.327, ADP hit 0.017, lift 0.310, AUC 0.703, ADP AUC 0.261, beats ADP 14/15 seasons (93.3%).

Projected volume improved over ADP-only by top aggregate lift: no.
Projected volume improved over prior non-projection models by top aggregate lift: no.
Does RB beat ADP by repeatability rule? yes.
RB Tie-Breaker Only? yes.
RB Strong Draft Signal? no.
RB App-Ready? no.

Top RB aggregate rows:

| target            | model_name                                | feature_group           |   seasons_tested |   sample_size |   model_hit_rate |   adp_hit_rate |   lift_over_adp |   auc |   adp_auc |   seasons_model_beats_adp |   pct_seasons_model_beats_adp |
|:------------------|:------------------------------------------|:------------------------|-----------------:|--------------:|-----------------:|---------------:|----------------:|------:|----------:|--------------------------:|------------------------------:|
| RB_Beat_ADP_By_12 | adp_role_opportunity_logistic             | adp_role                |               15 |          2364 |            0.327 |          0.017 |           0.310 | 0.703 |     0.261 |                        14 |                        93.333 |
| RB_Beat_ADP_By_12 | adp_role_opportunity_random_forest        | adp_role_opportunity    |               15 |          2364 |            0.327 |          0.017 |           0.310 | 0.703 |     0.261 |                        14 |                        93.333 |
| RB_Beat_ADP_By_12 | adp_role_opportunity_regularized_logistic | adp_role_opportunity    |               15 |          2364 |            0.327 |          0.017 |           0.310 | 0.703 |     0.261 |                        14 |                        93.333 |
| RB_Beat_ADP_By_12 | adp_only_baseline_logistic                | adp_only                |               14 |          2210 |            0.301 |          0.017 |           0.284 | 0.739 |     0.261 |                        14 |                       100.000 |
| RB_Beat_ADP_By_12 | adp_only_baseline_random_forest           | adp_only_baseline       |               14 |          2210 |            0.301 |          0.017 |           0.284 | 0.739 |     0.261 |                        14 |                       100.000 |
| RB_Beat_ADP_By_12 | adp_only_baseline_regularized_logistic    | adp_only_baseline       |               14 |          2210 |            0.301 |          0.017 |           0.284 | 0.739 |     0.261 |                        14 |                       100.000 |
| RB_Beat_ADP_By_12 | adp_sportsbook_features_logistic          | adp_sportsbook          |               14 |          2210 |            0.301 |          0.017 |           0.284 | 0.739 |     0.261 |                        14 |                       100.000 |
| RB_Beat_ADP_By_12 | adp_sportsbook_features_random_forest     | adp_sportsbook_features |               14 |          2210 |            0.301 |          0.017 |           0.284 | 0.739 |     0.261 |                        14 |                       100.000 |

## Whether Projections Improved Over ADP

WR: no by best aggregate lift comparison. RB: no by best aggregate lift comparison.

This is not the same as App-Ready edge. A draft tool needs repeatable bucket/window lift and stable player-level output, which remains incomplete.

## Whether Projections Improved Over Prior Models

WR: no. RB: no.

## Bucket/Window Status

Bucket/window projection validation completed: **no**. The previous refit-heavy bucket script timed out. Current WR/RB validator bucket outputs exist, but they are not a full lightweight repeatability report comparing projected-volume feature groups against ADP in each draft window. No bucket is promoted.

## Unified Signal Export Status

Unified signal export refreshed: yes. Projection fields were merged into `unified_player_signal_export.csv` where player-season rows matched.

## Final Classifications

WR: **Tie-Breaker Only**
RB: **Tie-Breaker Only**
Any Tie-Breaker Only: yes
Any Strong Draft Signal: no
Anything App-Ready: no

## Recommended Next Step

Build a lightweight bucket/window projection validation script that reuses completed model scores or restricts the model grid to the best projection-vs-ADP candidates. Then classify draft windows only if positive lift over ADP repeats across multiple seasons.

Recommended next Codex prompt:

```text
Continue research-only validation in research/validation_v1. Build a lightweight projection bucket/window analysis that reuses completed WR/RB validation outputs or limits to the best ADP + projection models, then classify whether any draft window has repeatable lift over ADP. Do not modify the app.
```
