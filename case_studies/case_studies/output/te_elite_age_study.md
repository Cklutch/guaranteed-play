# TE Elite Season Age Study

This study measures each age's share of elite TE finishes. It does not use average PPG by age.

Rate % = players at that age inside Top N TE slots / all Top N TE slots.

## Data Source

- Source: nflverse-data GitHub releases: stats_player + rosters
- Releases: stats_player, rosters
- Timeline: 2023-2025
- Seasons used: 3
- Scoring: Half PPR
- Player-seasons analyzed: 1,783
- Cache file: `case_studies\data\qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv`

## Peaks

- Peak Age for Top 20 Rate: Age 24 (15.00%)
- Peak Age for Top 15 Rate: Age 24 (13.33%)
- Peak Age for Top 12 Rate: Age 24 (13.89%)
- Peak Age for Top 6 Rate: Age 26 (16.67%)
- Peak Age for Top 3 Rate: Age 22 (22.22%)

## Table

| Age  | player_seasons_at_age | top20_age_count | top20_rate_pct | top15_age_count | top15_rate_pct | top12_age_count | top12_rate_pct | top6_age_count | top6_rate_pct | top3_age_count | top3_rate_pct |
| ---- | --------------------- | --------------- | -------------- | --------------- | -------------- | --------------- | -------------- | -------------- | ------------- | -------------- | ------------- |
| 21.0 | 4.0                   | 2.0             | 3.33           | 2.0             | 4.44           | 2.0             | 5.56           | 1.0            | 5.56          | 0.0            | 0.0           |
| 22.0 | 13.0                  | 3.0             | 5.0            | 3.0             | 6.67           | 2.0             | 5.56           | 2.0            | 11.11         | 2.0            | 22.22         |
| 23.0 | 34.0                  | 5.0             | 8.33           | 5.0             | 11.11          | 3.0             | 8.33           | 1.0            | 5.56          | 0.0            | 0.0           |
| 24.0 | 54.0                  | 9.0             | 15.0           | 6.0             | 13.33          | 5.0             | 13.89          | 0.0            | 0.0           | 0.0            | 0.0           |
| 25.0 | 62.0                  | 4.0             | 6.67           | 3.0             | 6.67           | 2.0             | 5.56           | 2.0            | 11.11         | 2.0            | 22.22         |
| 26.0 | 57.0                  | 6.0             | 10.0           | 4.0             | 8.89           | 4.0             | 11.11          | 3.0            | 16.67         | 1.0            | 11.11         |
| 27.0 | 37.0                  | 2.0             | 3.33           | 2.0             | 4.44           | 2.0             | 5.56           | 1.0            | 5.56          | 0.0            | 0.0           |
| 28.0 | 31.0                  | 7.0             | 11.67          | 3.0             | 6.67           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 29.0 | 29.0                  | 8.0             | 13.33          | 6.0             | 13.33          | 5.0             | 13.89          | 3.0            | 16.67         | 1.0            | 11.11         |
| 30.0 | 24.0                  | 4.0             | 6.67           | 3.0             | 6.67           | 3.0             | 8.33           | 1.0            | 5.56          | 0.0            | 0.0           |
| 31.0 | 20.0                  | 2.0             | 3.33           | 2.0             | 4.44           | 2.0             | 5.56           | 1.0            | 5.56          | 1.0            | 11.11         |
| 32.0 | 10.0                  | 2.0             | 3.33           | 1.0             | 2.22           | 0.0             | 0.0            | 0.0            | 0.0           | 0.0            | 0.0           |
| 33.0 | 4.0                   | 1.0             | 1.67           | 1.0             | 2.22           | 1.0             | 2.78           | 0.0            | 0.0           | 0.0            | 0.0           |
| 34.0 | 3.0                   | 2.0             | 3.33           | 2.0             | 4.44           | 2.0             | 5.56           | 1.0            | 5.56          | 1.0            | 11.11         |
| 35.0 | 3.0                   | 2.0             | 3.33           | 1.0             | 2.22           | 1.0             | 2.78           | 1.0            | 5.56          | 0.0            | 0.0           |
| 36.0 | 1.0                   | 1.0             | 1.67           | 1.0             | 2.22           | 1.0             | 2.78           | 1.0            | 5.56          | 1.0            | 11.11         |
