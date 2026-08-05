# Age Curve Edge Score V1 Feature Report

Research-only. This does not modify the app, Draft Mode, rankings, recommendations, player cards, or app-facing scores.

## Executive Summary

Built `age_curve_edge_score` from verified age-study artifacts. Dashboard found: `True`. Dashboard parsed: `True`. Fallback curves needed: `False`.

## Input/Output Files

- Input: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_projected.csv`
- Output: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\predraft_validation_dataset_age_edge_score.csv`
- Primary dashboard: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\fantasy_age_study_dashboard.html`

## Related Source Files Found

- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\RB_Age_Study.py`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\qb_rb_wr_elite_age_player_seasons_half_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\rb_elite_age_player_seasons_dad.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\rb_elite_age_player_seasons_half_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\rb_elite_age_player_seasons_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\data\rb_wr_elite_age_player_seasons_half_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\number_one_overall_age_share.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\qb_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\qb_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_age_share_study\rb_age_share_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_age_share_study\rb_age_share_study.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_elite_age_study.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_wr_elite_age_study.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_wr_qb_elite_age_study.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\rb_wr_qb_te_elite_age_study.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\te_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\te_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\wr_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\case_studies\output\wr_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\qb_rb_te_wr_elite_age_player_seasons_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\qb_rb_wr_elite_age_player_seasons_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\rb_elite_age_player_seasons_dad.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\rb_elite_age_player_seasons_half_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\rb_elite_age_player_seasons_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\data\rb_wr_elite_age_player_seasons_ppr.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\fantasy_age_study_dashboard.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\number_one_overall_age_share.html`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\qb_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\qb_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\rb_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\rb_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\te_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\te_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\wr_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\wr_elite_age_study.md`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\rb_elite_age_analysis.py`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\run_fantasy_age_study.py`

## Related Source Files Used

- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\rb_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\wr_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\te_elite_age_rates.csv`
- `C:\Users\cklut\Desktop\Projects\Guaranteed Play\case_studies\output\qb_elite_age_rates.csv`

## Extracted Age Findings

- RB: strongest ages by normalized blended elite/useful finish rates: 24 (97.9), 25 (94.9), 23 (89.1), 26 (78.1), 22 (69.3)
- WR: strongest ages by normalized blended elite/useful finish rates: 25 (93.9), 24 (90.9), 26 (89.2), 23 (76.0), 27 (67.7)
- TE: strongest ages by normalized blended elite/useful finish rates: 26 (98.8), 25 (90.7), 29 (84.0), 24 (80.5), 27 (74.6)
- QB: strongest ages by normalized blended elite/useful finish rates: 24 (100.0), 25 (93.0), 23 (83.9), 28 (69.6), 27 (66.0)

## Score Conversion

For each position, age-level Top12, Top24, Top36, and Top6 rates were normalized within position to 0-100 and blended with weights 40%, 30%, 20%, and 10% when available. Small sample ages were penalized. Player-season scoring uses only age and position.

## Columns Detected

{
  "age": "age",
  "position": "position",
  "season": "season",
  "player": "player_name",
  "adp": "overall_adp",
  "positional_adp": "positional_adp"
}

## Bucket Definitions

- 90-100: Elite Age Window
- 75-89: Strong Age Window
- 55-74: Neutral Age Window
- 35-54: Mild Age Risk
- 0-34: Major Age Risk
- NaN: Unknown

## Missingness Diagnostics

Rows loaded/saved: `10131`. Valid age scores: `10131`. Missing age scores: `0`.

## Score Distribution By Position

| position   |   count |   mean |   min |   max |
|:-----------|--------:|-------:|------:|------:|
| RB         |    4123 |  68.64 |  0.00 | 97.89 |
| WR         |    6008 |  67.90 |  0.00 | 93.91 |

## Bucket Counts By Position

| position   | age_curve_edge_bucket   |   rows |
|:-----------|:------------------------|-------:|
| RB         | Elite Age Window        |   1286 |
| RB         | Major Age Risk          |   1036 |
| RB         | Neutral Age Window      |    719 |
| RB         | Strong Age Window       |   1082 |
| WR         | Elite Age Window        |   1770 |
| WR         | Major Age Risk          |    694 |
| WR         | Mild Age Risk           |    668 |
| WR         | Neutral Age Window      |   1375 |
| WR         | Strong Age Window       |   1501 |

## Data Concerns

- The score is calibrated from historical outcome rates in the age study, but player-season assignment uses only age and position.
- This is a standalone feature, not an app-facing recommendation.
- TE/QB curves are built, but validation targets in the current WR/RB dataset are mostly WR/RB specific.