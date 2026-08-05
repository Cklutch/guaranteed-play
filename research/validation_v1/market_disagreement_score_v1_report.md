# Market Disagreement Score V1

## Executive summary
Market Disagreement Score V1 compares archived preseason FantasyPros projection volume/rank signals against historical ADP inside each season-position group.

This is research-only and was not promoted into the app.

## Inputs and outputs
- Input: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_age_edge_score.csv`
- Output: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_market_disagreement_score.csv`
- Validation CSV: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\market_disagreement_score_v1_validation.csv`

## Columns detected
- ADP column used: `overall_adp`
- Positional ADP column available: `positional_adp`
- Age score column: `age_curve_edge_score`

## Projection columns used
- WR: `projected_receptions`, `projected_receiving_yards`, `projected_receiving_tds`, `projected_receiving_role_score`, `projected_volume_score`, `projected_total_tds`
- RB: `projected_carries`, `projected_rushing_yards`, `projected_rushing_tds`, `projected_touch_score`, `projected_volume_score`, `projected_total_tds`, `projected_receptions`
- TE: `projected_receptions`, `projected_receiving_yards`, `projected_receiving_tds`, `projected_receiving_role_score`, `projected_volume_score`, `projected_total_tds`

## Projection columns skipped
- WR: none
- RB: none
- TE: none

## Score distribution by position
| position | rows | valid_scores | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| RB | 4123 | 871 | 50.23480546077214 | 50.202025987912116 | 25.0 | 79.55232558139535 |
| WR | 6008 | 1029 | 49.7032556393108 | 50.0 | 5.882352941176471 | 75.0 |

## Bucket counts
| position | market_disagreement_bucket | rows |
| --- | --- | --- |
| RB | Negative Market Disagreement | 225 |
| RB | Neutral / Fairly Priced | 623 |
| RB | Positive Market Disagreement | 23 |
| RB | Unknown | 3252 |
| WR | Major Negative Market Disagreement | 6 |
| WR | Negative Market Disagreement | 219 |
| WR | Neutral / Fairly Priced | 778 |
| WR | Positive Market Disagreement | 26 |
| WR | Unknown | 4979 |

## Data concerns
- Market disagreement coverage is low; projection data only exists for imported FantasyPros Wayback seasons.
- The score uses preseason projections and ADP only; final outcomes and target labels are excluded from score construction.