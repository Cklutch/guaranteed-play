# Late-Season Role Growth Score V1

## Executive summary
Late-Season Role Growth Score V1 was built as a research-only pre-draft-safe feature. It preserves existing age and projection fields and does not modify any app-facing files.

The available dataset has limited explicit late-season usage data. The only strong discovered late-season source column is primarily WR-specific, so RB coverage is expected to be sparse or unavailable.

## Input/output files
- Input: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_age_edge_score.csv`
- Output: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_late_role_growth_score.csv`
- Validation output: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\late_season_role_growth_score_v1_validation.csv`

## Columns detected
- Season column: `season`
- Player column: `player_name`
- Position column: `position`
- ADP column: `overall_adp`
- Age score column: `age_curve_edge_score`

## Source columns used
- target_growth_component: none
- touch_growth_component: none
- snap_growth_component: none
- route_growth_component: none
- red_zone_growth_component: none
- direct_late_role_growth_component: none

## Source columns missing/skipped
- target_growth_component: prior_wr_late_season_target_growth
- touch_growth_component: none
- snap_growth_component: none
- route_growth_component: none
- red_zone_growth_component: none
- direct_late_role_growth_component: prior_wr_late_season_target_growth

## Scoring logic by position
- WR emphasizes target growth, route growth, snap growth, red-zone target growth, and direct late-role growth.
- RB emphasizes touch growth, target growth, snap growth, red-zone touch growth, and direct late-role growth.
- TE uses target, route, snap, red-zone, and direct growth if rows exist.
- QB is not forced unless direct late-role growth data exists.
- Component weights are renormalized when components are missing. Missing source data becomes NaN, not zero.

## Component weights
- WR: target_growth_component: 35%, route_growth_component: 25%, snap_growth_component: 15%, red_zone_growth_component: 15%, direct_late_role_growth_component: 10%
- RB: touch_growth_component: 35%, target_growth_component: 20%, snap_growth_component: 20%, red_zone_growth_component: 15%, direct_late_role_growth_component: 10%
- TE: route_growth_component: 30%, target_growth_component: 30%, snap_growth_component: 15%, red_zone_growth_component: 15%, direct_late_role_growth_component: 10%
- QB: direct_late_role_growth_component: 100%

## Missingness diagnostics
- direct_late_role_growth_component: 100.0% missing
- red_zone_growth_component: 100.0% missing
- route_growth_component: 100.0% missing
- snap_growth_component: 100.0% missing
- target_growth_component: 100.0% missing
- touch_growth_component: 100.0% missing
- late_season_role_growth_score: 100.0% missing

## Score distribution by position
| position | rows | valid_scores | mean | median | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| RB | 4123 | 0 |  |  |  |  |
| WR | 6008 | 0 |  |  |  |  |

## Bucket definitions
- Major Late-Season Role Growth: score >= 85
- Strong Late-Season Role Growth: 70 to 84.99
- Moderate Late-Season Role Growth: 55 to 69.99
- Stable Role: 40 to 54.99
- Role Decline: below 40
- Unknown: not enough source data

## Data concerns
- Coverage is narrow because most requested late-season opportunity columns are not present in the current dataset.
- `prior_wr_late_season_target_growth` is pre-draft-safe because it describes the prior season, but it is position-specific and should not be generalized to RB without RB-specific usage data.
- No Beat_ADP, Underpriced, Breakout, Tier_Jump, final ranking, or final fantasy outcome columns are used in score construction.

## Research-only status
This feature is not app-ready and was not promoted into production.