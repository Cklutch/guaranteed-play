# Breakout Score V1 -- Robustness Checks (WR / FULL top-decile finding)

Four independent checks against the one result from breakout_v1_validation_report.md that cleared a season-blocked bootstrap CI. See this script's module docstring for what each checks and why.

## 1. Permutation test

|   observed_delta |   null_mean |   null_std |   null_p95 |   null_p99 |   p_value_one_sided |   n_permutations |
|-----------------:|------------:|-----------:|-----------:|-----------:|--------------------:|-----------------:|
|            0.119 |      0.0004 |     0.0519 |     0.0952 |      0.119 |              0.0188 |             5000 |

**Verdict: SURVIVES at p<0.05.**

## 2. Leave-one-season-out

Full mean (all seasons): +0.1190

|   excluded_season |   mean_delta_without_this_season |   shift_from_full_mean |
|------------------:|---------------------------------:|-----------------------:|
|              2019 |                           0.1111 |                -0.0079 |
|              2020 |                           0.1111 |                -0.0079 |
|              2021 |                           0.1111 |                -0.0079 |
|              2022 |                           0.1389 |                 0.0199 |
|              2023 |                           0.1389 |                 0.0199 |
|              2024 |                           0.1111 |                -0.0079 |
|              2025 |                           0.1111 |                -0.0079 |

**Verdict: ROBUST.**

## 3. Feature coefficient sign consistency

| feature                          |   mean_coef |   pct_positive |   pct_negative |   sign_consistency_pct |
|:---------------------------------|------------:|---------------:|---------------:|-----------------------:|
| age                              |     -0.3043 |            0   |          100   |                  100   |
| overall_adp                      |      1.2315 |          100   |            0   |                  100   |
| prior_durability_score           |      0.2522 |          100   |            0   |                  100   |
| prior_garbage_time_share         |     -0.3702 |            0   |          100   |                  100   |
| prior_snap_share                 |      1.1739 |          100   |            0   |                  100   |
| prior_route_participation_rate   |     -0.5873 |            0   |          100   |                  100   |
| prior_targets_per_route_run      |      0.2668 |          100   |            0   |                  100   |
| div_opportunity_minus_efficiency |      0.0857 |           57.1 |           42.9 |                   57.1 |

## 4. Threshold sensitivity

|   top_pct |   mean_delta |   ci_lo |   ci_hi | excludes_zero   |   seasons_improved |
|----------:|-------------:|--------:|--------:|:----------------|-------------------:|
|      0.05 |       0.1905 | -0.0476 |  0.4762 | False           |              0.429 |
|      0.1  |       0.119  |  0.0714 |  0.1667 | True            |              0.714 |
|      0.15 |       0.0813 |  0.0317 |  0.1429 | True            |              0.571 |
|      0.2  |       0.0877 |  0.0357 |  0.1548 | True            |              0.714 |

**Excludes zero at 3 of 4 thresholds tested.**
