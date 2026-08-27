# Breakout Score V1 Validation

Nested comparison, `P(Beat_ADP_By_12) = f(ADP)` vs. `f(ADP, features)`, mirroring wr_bust_final_validation.py's Task 1 methodology exactly. See this script's module docstring for the full spec and what is/isn't covered (no policy-decomposition test, unlike the bust study).

## Results (nested comparison, AUC/Brier/log-loss deltas over continuous-ADP baseline)

| position   | spec    |   test_seasons |   auc_base |   auc_exp |   auc_delta | auc_ci            |   seasons_improved |   brier_delta |   logloss_delta | classification       |
|:-----------|:--------|---------------:|-----------:|----------:|------------:|:------------------|-------------------:|--------------:|----------------:|:---------------------|
| WR         | FULL    |              7 |     0.7032 |    0.7382 |      0.035  | [-0.0068,+0.0764] |              0.714 |       -0.0098 |         -0.0104 | Weak Research Signal |
| WR         | CORE    |             10 |     0.7102 |    0.7258 |      0.0155 | [-0.0310,+0.0604] |              0.5   |       -0.0165 |         -0.0354 | No Signal            |
| WR         | CAPITAL |             13 |     0.7136 |    0.7011 |     -0.0124 | [-0.0261,-0.0003] |              0.308 |        0.0029 |          0.0074 | No Signal            |
| RB         | FULL    |              6 |     0.7503 |    0.7153 |     -0.0349 | [-0.0932,+0.0111] |              0.333 |       -0.0065 |         -0.0063 | No Signal            |
| RB         | CORE    |              9 |     0.7398 |    0.715  |     -0.0248 | [-0.0476,-0.0016] |              0.222 |        0.0012 |          0.0026 | No Signal            |
| RB         | CAPITAL |             13 |     0.7427 |    0.7316 |     -0.0112 | [-0.0205,-0.0021] |              0.308 |       -0.0029 |         -0.008  | No Signal            |

## Calibration (expanded model, pooled out-of-sample predictions)

|   n |   brier |   intercept |   slope |   mean_pred |   actual_rate | position   | spec    |
|----:|--------:|------------:|--------:|------------:|--------------:|:-----------|:--------|
| 415 |  0.2113 |     -1.6031 |  0.6313 |      0.4254 |        0.1614 | WR         | FULL    |
| 599 |  0.2038 |     -1.5906 |  0.7625 |      0.419  |        0.1553 | WR         | CORE    |
| 771 |  0.2201 |     -1.6311 |  0.7978 |      0.4481 |        0.1595 | WR         | CAPITAL |
| 299 |  0.221  |     -1.7766 |  0.6437 |      0.4398 |        0.1438 | RB         | FULL    |
| 446 |  0.2176 |     -1.8181 |  0.6932 |      0.4296 |        0.1345 | RB         | CORE    |
| 661 |  0.2226 |     -1.7497 |  0.7345 |      0.4438 |        0.1498 | RB         | CAPITAL |

## Top-decile hit rate: does the model's own most-confident calls beat baseline?

A whole-population AUC can be null or reliably negative while the model's own highest-confidence calls are still real -- this asks that question directly instead of inferring it from AUC. Same rigor as the confirmatory nested comparison above, not a pooled point estimate: top ~10% by predicted probability computed WITHIN each test season (min 3 players), then season-blocked bootstrapped exactly like the AUC/Brier/logloss deltas. `excludes_zero=True` rows are the real finding.

| position   | spec    |   topdecile_base |   topdecile_exp |   topdecile_delta | topdecile_ci      |   seasons_improved | excludes_zero   |
|:-----------|:--------|-----------------:|----------------:|------------------:|:------------------|-------------------:|:----------------|
| WR         | FULL    |           0.1905 |          0.3095 |            0.119  | [+0.0708,+0.1667] |              0.714 | True            |
| WR         | CORE    |           0.2167 |          0.2667 |            0.05   | [-0.0667,+0.1500] |              0.5   | False           |
| WR         | CAPITAL |           0.259  |          0.2205 |           -0.0385 | [-0.1026,+0.0385] |              0.231 | False           |
| RB         | FULL    |           0.2667 |          0.2417 |           -0.025  | [-0.2000,+0.1417] |              0.333 | False           |
| RB         | CORE    |           0.2667 |          0.2    |           -0.0667 | [-0.1556,+0.0000] |              0     | False           |
| RB         | CAPITAL |           0.2846 |          0.2577 |           -0.0269 | [-0.1077,+0.0423] |              0.077 | False           |

## Exploratory: experience-stage subgroup split

EXPLORATORY, NOT CONFIRMATORY. One pre-specified split -- years_since_drafted <= 2 ("developing") vs. > 2 ("established"), the standard breakout-happens-in-year-2-3 theory -- tested once, not scanned across cut points. No season-blocked bootstrap CI (too few rows per season per subgroup for a stable estimate), so treat this as hypothesis-generating only, same standard this repo's other studies hold themselves to for anything not run through the full confirmatory pipeline.

| position   | spec    | stage                |   n |   breakout_rate |   auc_base |   auc_exp |   auc_delta | status    |
|:-----------|:--------|:---------------------|----:|----------------:|-----------:|----------:|------------:|:----------|
| WR         | FULL    | developing (<=2 yrs) | 157 |           0.204 |     0.6303 |    0.6735 |      0.0432 | evaluated |
| WR         | FULL    | established (>2 yrs) | 237 |           0.127 |     0.7444 |    0.762  |      0.0176 | evaluated |
| WR         | CORE    | developing (<=2 yrs) | 210 |           0.195 |     0.6656 |    0.678  |      0.0124 | evaluated |
| WR         | CORE    | established (>2 yrs) | 345 |           0.13  |     0.7229 |    0.7541 |      0.0311 | evaluated |
| WR         | CAPITAL | developing (<=2 yrs) | 267 |           0.187 |     0.6776 |    0.6636 |     -0.0141 | evaluated |
| WR         | CAPITAL | established (>2 yrs) | 440 |           0.145 |     0.733  |    0.728  |     -0.005  | evaluated |
| RB         | FULL    | developing (<=2 yrs) | 122 |           0.197 |     0.6605 |    0.6569 |     -0.0036 | evaluated |
| RB         | FULL    | established (>2 yrs) | 144 |           0.097 |     0.7808 |    0.7725 |     -0.0082 | evaluated |
| RB         | CORE    | developing (<=2 yrs) | 197 |           0.162 |     0.6863 |    0.6869 |      0.0007 | evaluated |
| RB         | CORE    | established (>2 yrs) | 199 |           0.106 |     0.7394 |    0.7335 |     -0.0059 | evaluated |
| RB         | CAPITAL | developing (<=2 yrs) | 277 |           0.17  |     0.6684 |    0.6636 |     -0.0049 | evaluated |
| RB         | CAPITAL | established (>2 yrs) | 296 |           0.118 |     0.7835 |    0.7599 |     -0.0236 | evaluated |
