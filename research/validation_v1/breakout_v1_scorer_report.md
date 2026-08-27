# Breakout Score V1 -- Frozen Scorer

Target: `Beat_ADP_By_12`. Position: WR. Spec: FULL (2017+). Model: unweighted logistic regression (see module docstring point 3 for why this differs from the class_weight="balanced" pipeline used in validation).

## Walk-forward calibration (out-of-sample, pooled across folds)

|   n |   mean_walkforward_auc |   brier |   mean_pred |   actual_rate |   calibration_slope |   calibration_intercept |
|----:|-----------------------:|--------:|------------:|--------------:|--------------------:|------------------------:|
| 415 |                 0.7438 |  0.1263 |      0.1533 |        0.1614 |              0.7914 |                 -0.2409 |

## Final model (fit on all available rows, used by score_players())

Fit on 538 rows, 2017-2025.

| feature                        |   coef_standardized |
|:-------------------------------|--------------------:|
| overall_adp                    |           0.97071   |
| prior_snap_share               |           0.815759  |
| prior_targets_per_route_run    |           0.0806899 |
| prior_garbage_time_share       |          -0.158872  |
| prior_route_participation_rate |          -0.262255  |
| prior_durability_score         |           0.191037  |
| age                            |          -0.249035  |

**Status: RESEARCH_ONLY.** `score_players()` is a usable function but nothing in this repo calls it yet. See research/MODEL_REGISTRY.md.
