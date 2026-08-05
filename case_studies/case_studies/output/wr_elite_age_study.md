# WR Elite Season Age Study

This study measures each age's share of elite WR finishes. It does not use average PPG by age.

Rate % = players at that age inside Top N WR slots / all Top N WR slots.

## Data Source

- Source: nflverse-data GitHub releases: stats_player + rosters
- Releases: stats_player, rosters
- Timeline: 2023-2025
- Seasons used: 3
- Scoring: Half PPR
- Player-seasons analyzed: 1,783
- Cache file: `case_studies\data\qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv`

## Peaks

- Peak Age for Top 36 Rate: Age 24 (14.81%)
- Peak Age for Top 24 Rate: Age 24 (15.28%)
- Peak Age for Top 12 Rate: Age 24 (22.22%)
- Peak Age for Top 5 Rate: Age 24 (33.33%)
- Peak Age for Top 3 Rate: Age 24 (44.44%)

## Table

| Age  | player_seasons_at_age | top36_age_count | top36_rate_pct | top24_age_count | top24_rate_pct | top12_age_count | top12_rate_pct | top5_age_count | top5_rate_pct | top3_age_count | top3_rate_pct |
| ---- | --------------------- | --------------- | -------------- | --------------- | -------------- | --------------- | -------------- | -------------- | ------------- | -------------- | ------------- |
| 21.0 | 10.0                  | 3.0             | 2.78           | 2.0             | 2.78           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 22.0 | 39.0                  | 8.0             | 7.41           | 5.0             | 6.94           | 3.0             | 8.33           | 2.0            | 13.33         | 0.0            | 0.0           |
| 23.0 | 90.0                  | 15.0            | 13.89          | 6.0             | 8.33           | 3.0             | 8.33           | 2.0            | 13.33         | 1.0            | 11.11         |
| 24.0 | 116.0                 | 16.0            | 14.81          | 11.0            | 15.28          | 8.0             | 22.22          | 5.0            | 33.33         | 4.0            | 44.44         |
| 25.0 | 89.0                  | 15.0            | 13.89          | 11.0            | 15.28          | 7.0             | 19.44          | 3.0            | 20.0          | 2.0            | 22.22         |
| 26.0 | 77.0                  | 10.0            | 9.26           | 9.0             | 12.5           | 4.0             | 11.11          | 2.0            | 13.33         | 1.0            | 11.11         |
| 27.0 | 73.0                  | 10.0            | 9.26           | 7.0             | 9.72           | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
| 28.0 | 64.0                  | 8.0             | 7.41           | 3.0             | 4.17           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 29.0 | 44.0                  | 7.0             | 6.48           | 6.0             | 8.33           | 2.0             | 5.56           | 1.0            | 6.67          | 1.0            | 11.11         |
| 30.0 | 44.0                  | 5.0             | 4.63           | 4.0             | 5.56           | 3.0             | 8.33           | 0.0            | 0.0           | 0.0            | 0.0           |
| 31.0 | 28.0                  | 5.0             | 4.63           | 4.0             | 5.56           | 2.0             | 5.56           | 0.0            | 0.0           | 0.0            | 0.0           |
| 32.0 | 18.0                  | 3.0             | 2.78           | 2.0             | 2.78           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 33.0 | 8.0                   | 3.0             | 2.78           | 2.0             | 2.78           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
