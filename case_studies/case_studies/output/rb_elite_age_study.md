# RB Elite Season Age Study

This study measures each age's share of elite RB finishes. It does not use average PPG by age.

Rate % = players at that age inside Top N RB slots / all Top N RB slots.

## Data Source

- Source: nflverse-data GitHub releases: stats_player + rosters
- Releases: stats_player, rosters
- Timeline: 2023-2025
- Seasons used: 3
- Scoring: Half PPR
- Player-seasons analyzed: 1,783
- Cache file: `case_studies\data\qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv`

## Peaks

- Peak Age for Top 36 Rate: Age 26 (17.59%)
- Peak Age for Top 24 Rate: Age 24 (18.06%)
- Peak Age for Top 12 Rate: Age 24 (16.67%)
- Peak Age for Top 5 Rate: Age 22 (20.00%)
- Peak Age for Top 3 Rate: Age 22 (33.33%)

## Table

| Age  | player_seasons_at_age | top36_age_count | top36_rate_pct | top24_age_count | top24_rate_pct | top12_age_count | top12_rate_pct | top5_age_count | top5_rate_pct | top3_age_count | top3_rate_pct |
| ---- | --------------------- | --------------- | -------------- | --------------- | -------------- | --------------- | -------------- | -------------- | ------------- | -------------- | ------------- |
| 21.0 | 14.0                  | 2.0             | 1.85           | 2.0             | 2.78           | 2.0             | 5.56           | 0.0            | 0.0           | 0.0            | 0.0           |
| 22.0 | 32.0                  | 10.0            | 9.26           | 6.0             | 8.33           | 4.0             | 11.11          | 3.0            | 20.0          | 3.0            | 33.33         |
| 23.0 | 49.0                  | 12.0            | 11.11          | 7.0             | 9.72           | 4.0             | 11.11          | 3.0            | 20.0          | 2.0            | 22.22         |
| 24.0 | 62.0                  | 18.0            | 16.67          | 13.0            | 18.06          | 6.0             | 16.67          | 3.0            | 20.0          | 1.0            | 11.11         |
| 25.0 | 79.0                  | 18.0            | 16.67          | 11.0            | 15.28          | 5.0             | 13.89          | 0.0            | 0.0           | 0.0            | 0.0           |
| 26.0 | 73.0                  | 19.0            | 17.59          | 12.0            | 16.67          | 4.0             | 11.11          | 1.0            | 6.67          | 0.0            | 0.0           |
| 27.0 | 45.0                  | 9.0             | 8.33           | 8.0             | 11.11          | 3.0             | 8.33           | 2.0            | 13.33         | 2.0            | 22.22         |
| 28.0 | 36.0                  | 9.0             | 8.33           | 5.0             | 6.94           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 29.0 | 30.0                  | 6.0             | 5.56           | 4.0             | 5.56           | 4.0             | 11.11          | 1.0            | 6.67          | 1.0            | 11.11         |
| 30.0 | 13.0                  | 3.0             | 2.78           | 2.0             | 2.78           | 1.0             | 2.78           | 1.0            | 6.67          | 0.0            | 0.0           |
| 31.0 | 8.0                   | 2.0             | 1.85           | 2.0             | 2.78           | 2.0             | 5.56           | 1.0            | 6.67          | 0.0            | 0.0           |
