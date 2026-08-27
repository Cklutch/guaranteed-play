# Breakout Score V1 -- Trimmed-Feature Re-validation

Dropped `div_opportunity_minus_efficiency` (57% coefficient-sign consistency in the original robustness check) and re-ran the full battery on the remaining 6 features + ADP. The original 7-feature FULL result stays the frozen historical record in breakout_v1_validation_report.md; this is a separate comparison.

## Nested comparison

| metric | FULL (7 feat, frozen) | FULL_TRIMMED (6 feat) |
|---|---|---|
| AUC delta | +0.0350 [-0.0068,+0.0764] | +0.0383 [-0.0058,+0.0820] |
| Top-decile delta | +11.9% [+7.1%,+16.7%] | +14.3% [+7.1%,+21.4%] |

## Permutation test

Observed delta: +0.1429. P(null >= observed), one-sided: 0.0042. Bonferroni family-wise threshold across the 6 originally tested combinations: 0.0083 -- clears it.

## Leave-one-season-out

Full mean: +0.1429

|   excluded_season |   mean_delta_without_this_season |   shift_from_full_mean |
|------------------:|---------------------------------:|-----------------------:|
|              2019 |                           0.1389 |                -0.004  |
|              2020 |                           0.1389 |                -0.004  |
|              2021 |                           0.1389 |                -0.004  |
|              2022 |                           0.1667 |                 0.0238 |
|              2023 |                           0.1667 |                 0.0238 |
|              2024 |                           0.1111 |                -0.0318 |
|              2025 |                           0.1389 |                -0.004  |

## Threshold sensitivity

|   top_pct |   mean_delta |   ci_lo |   ci_hi | excludes_zero   |   seasons_improved |
|----------:|-------------:|--------:|--------:|:----------------|-------------------:|
|      0.05 |       0.2381 |  0.0476 |  0.4762 | True            |              0.429 |
|      0.1  |       0.1429 |  0.0714 |  0.2143 | True            |              0.714 |
|      0.15 |       0.0992 |  0.0317 |  0.1706 | True            |              0.571 |
|      0.2  |       0.0877 |  0.0357 |  0.1537 | True            |              0.714 |
