# RB Elite Season Age Study

This study measures each age's share of elite RB finishes. It does not use average PPG by age.

Rate % = players at that age inside Top N RB slots / all Top N RB slots.

## Data Source

- Source: nflverse-data GitHub releases: stats_player + rosters
- Releases: stats_player, rosters
- Timeline: 2016-2025
- Seasons used: 10
- Scoring: Full PPR
- Player-seasons analyzed: 5,929
- Cache file: `case_studies\data\qb_rb_te_wr_elite_age_player_seasons_ppr.csv`

## Peaks

- Peak Age for Top 36 Rate: Age 24 (17.22%)
- Peak Age for Top 24 Rate: Age 24 (16.25%)
- Peak Age for Top 12 Rate: Age 25 (15.83%)
- Peak Age for Top 5 Rate: Age 23 (18.00%)
- Peak Age for Top 3 Rate: Age 22 (20.00%)

## Table

| Age  | player_seasons_at_age | top36_age_count | top36_rate_pct | top24_age_count | top24_rate_pct | top12_age_count | top12_rate_pct | top5_age_count | top5_rate_pct | top3_age_count | top3_rate_pct |
| ---- | --------------------- | --------------- | -------------- | --------------- | -------------- | --------------- | -------------- | -------------- | ------------- | -------------- | ------------- |
| 21.0 | 39.0                  | 12.0            | 3.33           | 10.0            | 4.17           | 6.0             | 5.0            | 2.0            | 4.0           | 2.0            | 6.67          |
| 22.0 | 131.0                 | 38.0            | 10.56          | 25.0            | 10.42          | 15.0            | 12.5           | 7.0            | 14.0          | 6.0            | 20.0          |
| 23.0 | 223.0                 | 52.0            | 14.44          | 33.0            | 13.75          | 18.0            | 15.0           | 9.0            | 18.0          | 5.0            | 16.67         |
| 24.0 | 248.0                 | 62.0            | 17.22          | 39.0            | 16.25          | 18.0            | 15.0           | 9.0            | 18.0          | 5.0            | 16.67         |
| 25.0 | 237.0                 | 57.0            | 15.83          | 38.0            | 15.83          | 19.0            | 15.83          | 8.0            | 16.0          | 5.0            | 16.67         |
| 26.0 | 200.0                 | 47.0            | 13.06          | 33.0            | 13.75          | 14.0            | 11.67          | 6.0            | 12.0          | 3.0            | 10.0          |
| 27.0 | 154.0                 | 35.0            | 9.72           | 31.0            | 12.92          | 13.0            | 10.83          | 3.0            | 6.0           | 3.0            | 10.0          |
| 28.0 | 101.0                 | 19.0            | 5.28           | 12.0            | 5.0            | 6.0             | 5.0            | 3.0            | 6.0           | 0.0            | 0.0           |
| 29.0 | 79.0                  | 18.0            | 5.0            | 6.0             | 2.5            | 5.0             | 4.17           | 1.0            | 2.0           | 1.0            | 3.33          |
| 30.0 | 45.0                  | 8.0             | 2.22           | 5.0             | 2.08           | 3.0             | 2.5            | 1.0            | 2.0           | 0.0            | 0.0           |
| 31.0 | 33.0                  | 6.0             | 1.67           | 4.0             | 1.67           | 2.0             | 1.67           | 1.0            | 2.0           | 0.0            | 0.0           |
| 32.0 | 18.0                  | 1.0             | 0.28           | 0.0             | 0.0            | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
| 33.0 | 11.0                  | 3.0             | 0.83           | 3.0             | 1.25           | 1.0             | 0.83           | 0.0            | 0.0           | 0.0            | 0.0           |
| 34.0 | 4.0                   | 2.0             | 0.56           | 1.0             | 0.42           | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
