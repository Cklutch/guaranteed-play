# WR Elite Season Age Study

This study measures each age's share of elite WR finishes. It does not use average PPG by age.

Rate % = players at that age inside Top N WR slots / all Top N WR slots.

## Data Source

- Source: nflverse-data GitHub releases: stats_player + rosters
- Releases: stats_player, rosters
- Timeline: 2016-2025
- Seasons used: 10
- Scoring: Full PPR
- Player-seasons analyzed: 5,929
- Cache file: `case_studies\data\qb_rb_te_wr_elite_age_player_seasons_ppr.csv`

## Peaks

- Peak Age for Top 36 Rate: Age 24 (13.89%)
- Peak Age for Top 24 Rate: Age 24 (15.42%)
- Peak Age for Top 12 Rate: Age 25 (14.17%)
- Peak Age for Top 5 Rate: Age 26 (16.00%)
- Peak Age for Top 3 Rate: Age 24 (16.67%)

## Table

| Age  | player_seasons_at_age | top36_age_count | top36_rate_pct | top24_age_count | top24_rate_pct | top12_age_count | top12_rate_pct | top5_age_count | top5_rate_pct | top3_age_count | top3_rate_pct |
| ---- | --------------------- | --------------- | -------------- | --------------- | -------------- | --------------- | -------------- | -------------- | ------------- | -------------- | ------------- |
| 21.0 | 45.0                  | 10.0            | 2.78           | 6.0             | 2.5            | 3.0             | 2.5            | 1.0            | 2.0           | 0.0            | 0.0           |
| 22.0 | 170.0                 | 25.0            | 6.94           | 17.0            | 7.08           | 6.0             | 5.0            | 3.0            | 6.0           | 0.0            | 0.0           |
| 23.0 | 324.0                 | 46.0            | 12.78          | 23.0            | 9.58           | 14.0            | 11.67          | 6.0            | 12.0          | 4.0            | 13.33         |
| 24.0 | 375.0                 | 50.0            | 13.89          | 37.0            | 15.42          | 14.0            | 11.67          | 7.0            | 14.0          | 5.0            | 16.67         |
| 25.0 | 329.0                 | 47.0            | 13.06          | 31.0            | 12.92          | 17.0            | 14.17          | 7.0            | 14.0          | 5.0            | 16.67         |
| 26.0 | 275.0                 | 40.0            | 11.11          | 30.0            | 12.5           | 17.0            | 14.17          | 8.0            | 16.0          | 5.0            | 16.67         |
| 27.0 | 227.0                 | 41.0            | 11.39          | 27.0            | 11.25          | 10.0            | 8.33           | 3.0            | 6.0           | 1.0            | 3.33          |
| 28.0 | 184.0                 | 32.0            | 8.89           | 19.0            | 7.92           | 13.0            | 10.83          | 5.0            | 10.0          | 4.0            | 13.33         |
| 29.0 | 141.0                 | 25.0            | 6.94           | 23.0            | 9.58           | 11.0            | 9.17           | 5.0            | 10.0          | 3.0            | 10.0          |
| 30.0 | 98.0                  | 19.0            | 5.28           | 13.0            | 5.42           | 7.0             | 5.83           | 3.0            | 6.0           | 2.0            | 6.67          |
| 31.0 | 59.0                  | 10.0            | 2.78           | 6.0             | 2.5            | 3.0             | 2.5            | 1.0            | 2.0           | 1.0            | 3.33          |
| 32.0 | 51.0                  | 7.0             | 1.94           | 3.0             | 1.25           | 1.0             | 0.83           | 0.0            | 0.0           | 0.0            | 0.0           |
| 33.0 | 27.0                  | 5.0             | 1.39           | 4.0             | 1.67           | 3.0             | 2.5            | 0.0            | 0.0           | 0.0            | 0.0           |
| 34.0 | 14.0                  | 1.0             | 0.28           | 1.0             | 0.42           | 1.0             | 0.83           | 1.0            | 2.0           | 0.0            | 0.0           |
| 35.0 | 7.0                   | 1.0             | 0.28           | 0.0             | 0.0            | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
| 36.0 | 5.0                   | 1.0             | 0.28           | 0.0             | 0.0            | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
