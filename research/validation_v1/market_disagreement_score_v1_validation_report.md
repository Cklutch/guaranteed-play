# Market Disagreement Score V1 Validation

## Executive summary
Final classification: **Weak Research Signal**.

Positive evidence exists, but the effect size or repeatability is limited.

This is research-only. No app-readiness is claimed.

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

## Best full-pool validation results
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | status | test_season | reason | draft_window | model_name | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_tested | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_pool | WR_Underpriced_Top24 | WR |  | 957.0 |  | 0.10031347962382445 |  | 0.13541666666666666 | evaluated |  |  |  | market disagreement only | 0.6233909214092141 | 0.1875 | 0.052083333333333336 | 0.13541666666666666 | 14.0 | 8.0 | 0.11692176870748298 | 0.125 | Weak Research Signal |
| full_pool | RB_Underpriced_Top12 | RB |  | 807.0 |  | 0.07806691449814127 |  | 0.07407407407407407 | evaluated |  |  |  | ADP + age + market disagreement | 0.5614439324116743 | 0.09876543209876543 | 0.024691358024691357 | 0.07407407407407407 | 14.0 | 7.0 | 0.10076530612244898 | 0.0625 | Weak Research Signal |
| full_pool | Underpriced_Top24 | ALL |  | 1764.0 |  | 0.10770975056689343 |  | 0.06779661016949153 | evaluated |  |  |  | market disagreement only | 0.6044455962014311 | 0.13559322033898305 | 0.06779661016949153 | 0.06779661016949153 | 14.0 | 8.0 | 0.07107178535749965 | 0.06904761904761905 | Weak Research Signal |
| full_pool | WR_Underpriced_Top12 | WR |  | 957.0 |  | 0.07836990595611286 |  | 0.0625 | evaluated |  |  |  | market disagreement only | 0.481307634164777 | 0.08333333333333333 | 0.020833333333333332 | 0.0625 | 14.0 | 7.0 | 0.08333333333333333 | 0.0625 | Weak Research Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.05084745762711865 | evaluated |  |  |  | ADP + market disagreement | 0.6015138955737382 | 0.07909604519774012 | 0.02824858757062147 | 0.05084745762711865 | 14.0 | 7.0 | 0.07647649969078539 | 0.03333333333333333 | Weak Research Signal |
| full_pool | RB_Underpriced_Top12 | RB |  | 807.0 |  | 0.07806691449814127 |  | 0.04938271604938271 | evaluated |  |  |  | ADP + market disagreement | 0.5946620583717358 | 0.07407407407407407 | 0.024691358024691357 | 0.04938271604938271 | 14.0 | 6.0 | 0.1096938775510204 | 0.0 | Weak Research Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.04519774011299435 | evaluated |  |  |  | market disagreement only | 0.4809236679323315 | 0.07344632768361582 | 0.02824858757062147 | 0.04519774011299435 | 14.0 | 7.0 | 0.04848722705865564 | 0.03571428571428571 | Weak Research Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.03954802259887005 | evaluated |  |  |  | ADP + age + market disagreement | 0.5915980355455729 | 0.06779661016949153 | 0.02824858757062147 | 0.03954802259887005 | 14.0 | 7.0 | 0.08455473098330239 | 0.03333333333333333 | Weak Research Signal |
| full_pool | RB_Underpriced_Top24 | RB |  | 773.0 |  | 0.1203104786545925 |  | 0.038461538461538464 | evaluated |  |  |  | market disagreement only | 0.5800521821631879 | 0.11538461538461539 | 0.07692307692307693 | 0.038461538461538464 | 13.0 | 2.0 | -0.0018315018315018315 | 0.0 | Weak Research Signal |
| full_pool | WR_Underpriced_Top24 | WR |  | 957.0 |  | 0.10031347962382445 |  | 0.031249999999999993 | evaluated |  |  |  | ADP + age + market disagreement | 0.6646039005032908 | 0.08333333333333333 | 0.052083333333333336 | 0.031249999999999993 | 14.0 | 5.0 | 0.05425170068027211 | 0.0 | Weak Research Signal |

## Best draft-window validation results
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | status | test_season | reason | draft_window | model_name | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_tested | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_window | Underpriced_Top12 | ALL |  | 24.0 |  | 0.08333333333333333 |  | 0.6666666666666666 | evaluated |  |  | 1-24 | market disagreement only | 1.0 | 0.6666666666666666 | 0.0 | 0.6666666666666666 | 1.0 | 1.0 | 0.6666666666666666 | 0.6666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top24 | ALL |  | 41.0 |  | 0.14634146341463414 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.7285714285714285 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + market disagreement | 0.6888888888888889 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.6944444444444444 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + market disagreement | 0.7333333333333333 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.7222222222222222 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 22.0 |  | 0.13636363636363635 |  | 0.3333333333333333 | evaluated |  |  | 151+ | ADP + age + market disagreement | 0.7017543859649122 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 1.0 | 1.0 | 0.3333333333333333 | 0.3333333333333333 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 22.0 |  | 0.13636363636363635 |  | 0.3333333333333333 | evaluated |  |  | 151+ | ADP + age | 0.7017543859649122 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 1.0 | 1.0 | 0.3333333333333333 | 0.3333333333333333 | Weak Research Signal |
| draft_window | Underpriced_Top24 | ALL |  | 41.0 |  | 0.14634146341463414 |  | 0.2 | evaluated |  |  | 73-96 | market disagreement only | 0.7476190476190476 | 0.4 | 0.2 | 0.2 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.2 | evaluated |  |  | 73-96 | market disagreement only | 0.7 | 0.4 | 0.2 | 0.2 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |

## ADP + market disagreement
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | status | test_season | reason | draft_window | model_name | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_tested | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_window | Beat_ADP_By_12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + market disagreement | 0.7333333333333333 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + market disagreement | 0.6888888888888889 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top24 | ALL |  | 41.0 |  | 0.14634146341463414 |  | 0.2 | evaluated |  |  | 73-96 | ADP + market disagreement | 0.738095238095238 | 0.4 | 0.2 | 0.2 | 2.0 | 1.0 | 0.16666666666666666 | 0.16666666666666666 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 167.0 |  | 0.15568862275449102 |  | 0.11764705882352941 | evaluated |  |  | 25-48 | ADP + market disagreement | 0.5317785051827605 | 0.11764705882352941 | 0.0 | 0.11764705882352941 | 7.0 | 1.0 | -0.05952380952380952 | 0.0 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 103.0 |  | 0.10679611650485436 |  | 0.09090909090909091 | evaluated |  |  | 25-48 | ADP + market disagreement | 0.5143280632411067 | 0.18181818181818182 | 0.09090909090909091 | 0.09090909090909091 | 4.0 | 0.0 | 0.0 | 0.0 | Weak Research Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.05084745762711865 | evaluated |  |  |  | ADP + market disagreement | 0.6015138955737382 | 0.07909604519774012 | 0.02824858757062147 | 0.05084745762711865 | 14.0 | 7.0 | 0.07647649969078539 | 0.03333333333333333 | Weak Research Signal |
| full_pool | RB_Underpriced_Top12 | RB |  | 807.0 |  | 0.07806691449814127 |  | 0.04938271604938271 | evaluated |  |  |  | ADP + market disagreement | 0.5946620583717358 | 0.07407407407407407 | 0.024691358024691357 | 0.04938271604938271 | 14.0 | 6.0 | 0.1096938775510204 | 0.0 | Weak Research Signal |
| full_pool | WR_Underpriced_Top24 | WR |  | 957.0 |  | 0.10031347962382445 |  | 0.020833333333333336 | evaluated |  |  |  | ADP + market disagreement | 0.668892760356175 | 0.07291666666666667 | 0.052083333333333336 | 0.020833333333333336 | 14.0 | 2.0 | 0.02040816326530612 | 0.0 | Weak Research Signal |
| full_pool | Underpriced_Top24 | ALL |  | 1764.0 |  | 0.10770975056689343 |  | 0.016949152542372878 | evaluated |  |  |  | ADP + market disagreement | 0.6408663813281615 | 0.0847457627118644 | 0.06779661016949153 | 0.016949152542372878 | 14.0 | 2.0 | 0.005834641548927264 | 0.0 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 25.0 |  | 0.28 |  | 0.0 | evaluated |  |  | 97-120 | ADP + market disagreement | 0.24603174603174605 | 0.3333333333333333 | 0.3333333333333333 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | Weak Research Signal |

## ADP + age + market disagreement
| result_type | target | position | bucket | rows | positives | positive_rate | baseline_positive_rate | lift_over_baseline | status | test_season | reason | draft_window | model_name | auc | top_decile_hit_rate | baseline_hit_rate | improvement_over_adp_only | seasons_tested | seasons_where_model_beat_adp_only | average_yearly_lift | median_yearly_lift | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_window | Underpriced_Top12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.6944444444444444 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 41.0 |  | 0.12195121951219512 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.7222222222222222 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Underpriced_Top24 | ALL |  | 41.0 |  | 0.14634146341463414 |  | 0.39999999999999997 | evaluated |  |  | 73-96 | ADP + age + market disagreement | 0.7285714285714285 | 0.6 | 0.2 | 0.39999999999999997 | 2.0 | 1.0 | 0.33333333333333337 | 0.33333333333333337 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 22.0 |  | 0.13636363636363635 |  | 0.3333333333333333 | evaluated |  |  | 151+ | ADP + age + market disagreement | 0.7017543859649122 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 | 1.0 | 1.0 | 0.3333333333333333 | 0.3333333333333333 | Weak Research Signal |
| draft_window | Beat_ADP_By_12 | ALL |  | 103.0 |  | 0.10679611650485436 |  | 0.09090909090909091 | evaluated |  |  | 25-48 | ADP + age + market disagreement | 0.5918972332015809 | 0.18181818181818182 | 0.09090909090909091 | 0.09090909090909091 | 4.0 | 1.0 | 0.08333333333333333 | 0.0 | Weak Research Signal |
| full_pool | RB_Underpriced_Top12 | RB |  | 807.0 |  | 0.07806691449814127 |  | 0.07407407407407407 | evaluated |  |  |  | ADP + age + market disagreement | 0.5614439324116743 | 0.09876543209876543 | 0.024691358024691357 | 0.07407407407407407 | 14.0 | 7.0 | 0.10076530612244898 | 0.0625 | Weak Research Signal |
| draft_window | Underpriced_Top12 | ALL |  | 167.0 |  | 0.15568862275449102 |  | 0.058823529411764705 | evaluated |  |  | 25-48 | ADP + age + market disagreement | 0.5941080196399346 | 0.058823529411764705 | 0.0 | 0.058823529411764705 | 7.0 | 2.0 | -0.035714285714285705 | 0.0 | Weak Research Signal |
| full_pool | Underpriced_Top12 | ALL |  | 1764.0 |  | 0.0782312925170068 |  | 0.03954802259887005 | evaluated |  |  |  | ADP + age + market disagreement | 0.5915980355455729 | 0.06779661016949153 | 0.02824858757062147 | 0.03954802259887005 | 14.0 | 7.0 | 0.08455473098330239 | 0.03333333333333333 | Weak Research Signal |
| full_pool | WR_Underpriced_Top24 | WR |  | 957.0 |  | 0.10031347962382445 |  | 0.031249999999999993 | evaluated |  |  |  | ADP + age + market disagreement | 0.6646039005032908 | 0.08333333333333333 | 0.052083333333333336 | 0.031249999999999993 | 14.0 | 5.0 | 0.05425170068027211 | 0.0 | Weak Research Signal |
| full_pool | Underpriced_Top24 | ALL |  | 1764.0 |  | 0.10770975056689343 |  | 0.016949152542372878 | evaluated |  |  |  | ADP + age + market disagreement | 0.637231659198823 | 0.0847457627118644 | 0.06779661016949153 | 0.016949152542372878 | 14.0 | 4.0 | 0.015762213976499694 | 0.0 | Weak Research Signal |

## Data concerns
- Market disagreement coverage is low; projection data only exists for imported FantasyPros Wayback seasons.
- The score uses preseason projections and ADP only; final outcomes and target labels are excluded from score construction.

## Recommended next step
Use this as the main projection-vs-market research branch: inspect which projection components drive lift, then retest in draft windows and by positional ADP bucket before any app integration.