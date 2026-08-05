# Late-Season Role Growth Score V1 Validation

## Executive summary
Final classification: **No Signal**.

ADP plus late-season role growth did not repeatably improve over ADP-only.

This is research-only. The score was not promoted to the app, Draft Mode, rankings, recommendations, or any production-facing file.

## Targets tested
- `WR_Beat_ADP_By_12`
- `RB_Beat_ADP_By_12`
- `WR_Underpriced_Top24`
- `RB_Underpriced_Top24`
- `WR_Underpriced_Top12`
- `RB_Underpriced_Top12`
- `Underpriced_Top24`
- `Underpriced_Top12`
- `Beat_ADP_By_12`

## Validation methodology
Validation used season-based walk-forward splits only. Each test season was predicted using earlier seasons. Models were simple LogisticRegression pipelines with median imputation and standardization.

Minimum sample rules were applied for full-pool, per-season, and draft-window tests. Skipped rows are retained in the validation CSV with a skip reason where applicable.

## Bucket-level validation
Bucket and score-decile target rates are included in the validation CSV as `bucket` and `score_decile` result types.

## ADP-only vs ADP + late-season role growth
No evaluated ADP + late-season role growth result cleared the sample rules.

## ADP + age vs ADP + age + late-season role growth
No evaluated ADP + age + late-season role growth result cleared the sample rules, or age score was unavailable.

## Draft-window validation
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | test_season | reason | model_name | draft_window | seasons_tested | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | status | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_window | Underpriced_Top12 | ALL |  | 24.0 |  | 0.08333333333333333 |  | 0.3333333333333333 |  |  | ADP + age | 1-24 | 1.0 |  | 0.3333333333333333 | 0.0 | 0.3333333333333333 | 1.0 | 0.3333333333333333 | 0.3333333333333333 | evaluated | No Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 22.0 |  | 0.13636363636363635 |  | 0.3333333333333333 |  |  | ADP + age | 151+ | 1.0 |  | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 1.0 | 0.3333333333333333 | 0.3333333333333333 | evaluated | No Signal |
| draft_window | Underpriced_Top24 | ALL |  | 41.0 |  | 0.14634146341463414 |  | 0.2 |  |  | ADP + age | 73-96 | 2.0 |  | 0.2 | 0.0 | 0.2 | 1.0 | 0.08333333333333334 | 0.08333333333333334 | evaluated | No Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.2 |  |  | ADP + age | 73-96 | 2.0 |  | 0.2 | 0.0 | 0.2 | 1.0 | 0.08333333333333334 | 0.08333333333333334 | evaluated | No Signal |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.2 |  |  | ADP + age | 73-96 | 2.0 |  | 0.2 | 0.0 | 0.2 | 1.0 | 0.08333333333333334 | 0.08333333333333334 | evaluated | No Signal |
| draft_window | Underpriced_Top12 | ALL |  | 24.0 |  | 0.08333333333333333 |  | 0.0 |  |  | ADP-only | 1-24 | 1.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | evaluated | No Signal |
| draft_window | WR_Underpriced_Top12 | WR |  | 25.0 |  | 0.16 |  | 0.0 |  |  | ADP-only | 25-48 | 1.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | evaluated | No Signal |
| draft_window | WR_Underpriced_Top12 | WR |  | 25.0 |  | 0.16 |  | 0.0 |  |  | ADP + age | 25-48 | 1.0 |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | evaluated | No Signal |

## Season repeatability
Repeatability is measured by `seasons_where_model_beat_adp_only`, `average_yearly_lift`, and `median_yearly_lift` in the validation CSV.

## Best result
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | test_season | reason | model_name | draft_window | seasons_tested | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | status | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_pool | WR_Beat_ADP_By_12 | WR |  | 957.0 |  | 0.15987460815047022 |  | 0.06249999999999997 |  |  | ADP + age |  | 14.0 |  | 0.2708333333333333 | 0.20833333333333334 | 0.06249999999999997 | 5.0 | 0.04702380952380952 | 0.0 | evaluated | No Signal |
| full_pool | Underpriced_Top24 | ALL |  | 1764.0 |  | 0.10770975056689343 |  | 0.03954802259887005 |  |  | ADP + age |  | 14.0 |  | 0.10734463276836158 | 0.06779661016949153 | 0.03954802259887005 | 8.0 | 0.03176823176823178 | 0.06904761904761905 | evaluated | No Signal |
| full_pool | WR_Underpriced_Top12 | WR |  | 957.0 |  | 0.07836990595611286 |  | 0.03125 |  |  | ADP + age |  | 14.0 |  | 0.041666666666666664 | 0.010416666666666666 | 0.03125 | 4.0 | 0.0494047619047619 | 0.0 | evaluated | No Signal |
| full_pool | Beat_ADP_By_12 | ALL |  | 1764.0 |  | 0.1649659863945578 |  | 0.02824858757062146 |  |  | ADP + age |  | 14.0 |  | 0.2768361581920904 | 0.24858757062146894 | 0.02824858757062146 | 6.0 | 0.041227819799248375 | 0.0 | evaluated | No Signal |
| full_pool | RB_Underpriced_Top24 | RB |  | 773.0 |  | 0.1203104786545925 |  | 0.025641025641025647 |  |  | ADP + age |  | 13.0 |  | 0.11538461538461539 | 0.08974358974358974 | 0.025641025641025647 | 5.0 | 0.046245421245421255 | 0.0 | evaluated | No Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.011299435028248588 |  |  | ADP + age |  | 14.0 |  | 0.022598870056497175 | 0.011299435028248588 | 0.011299435028248588 | 3.0 | 0.009927572427572428 | 0.0 | evaluated | No Signal |
| full_pool | WR_Underpriced_Top24 | WR |  | 957.0 |  | 0.10031347962382445 |  | 0.010416666666666671 |  |  | ADP + age |  | 14.0 |  | 0.07291666666666667 | 0.0625 | 0.010416666666666671 | 6.0 | 0.0244047619047619 | 0.0 | evaluated | No Signal |
| full_pool | WR_Beat_ADP_By_12 | WR |  | 957.0 |  | 0.15987460815047022 |  | 0.0 |  |  | ADP-only |  | 14.0 |  | 0.20833333333333334 | 0.20833333333333334 | 0.0 | 0.0 | 0.0 | 0.0 | evaluated | No Signal |

## Independent signal beyond ADP
The score only has independent signal if ADP + late-season role growth beats ADP-only repeatedly across seasons and survives draft-window checks. Do not infer edge from a single positive average.

## Data concerns
- No RB-specific touch/carry late-season growth columns were detected, so RB validation is coverage-limited.
- No route participation growth columns were detected.
- No snap-share growth columns were detected.
- Late-season role growth score coverage is low across the full dataset.
- The score uses only prior-season opportunity-growth columns and excludes all outcome/label columns.

## Final classification
**No Signal**: ADP plus late-season role growth did not repeatably improve over ADP-only.

## Recommended next research step
Add real prior-season weekly opportunity data for RB and WR, especially weeks 10-17 targets, routes, snaps, carries, red-zone touches, and team-level opportunity changes. Then rebuild this score with balanced position coverage and rerun the same ADP walk-forward validation.