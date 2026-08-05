# Wayback FantasyPros Projection Expansion Report

Date: 2026-07-07

Scope: research-only historical projection extraction. This does not modify the Streamlit app, add UI, build recommendations, rerun model validation, or claim model edge.

## Summary

Seasons attempted: 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.
Seasons successfully parsed: 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024.
Seasons included in the validation-safe file: 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024.
Seasons excluded for safety: 2025.
Seasons failed technically: 2025.
Total validation-safe WR rows: `725`.
Total validation-safe RB rows: `748`.
Were all selected captures preseason-safe? No. 2020 and 2025 are excluded.
Were projected fantasy points available? No.
Projected volume features available: projected_targets, projected_carries, projected_receptions, projected_receiving_yards, projected_receiving_tds, projected_rushing_yards, projected_rushing_tds.
Is this data valid enough for Historical Projection Import V1? Yes, for preseason WR/RB volume features; projected fantasy points were not found.
Was `historical_projections.csv` created? Yes.
Projection import rerun? No.
Projection feature build rerun? No.
Import/build notes: Skipped because import_historical_projections.py and build_projection_features_v1.py do not exist.

## Season Coverage

| Season | Safety status | Included | Page accessible | Tables | Rows | WR | RB | Fantasy points | Volume stats | Output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2014 | preseason_safe | true | true | 11 | 600 | 86 | 91 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2014.csv` |
| 2015 | preseason_safe | true | true | 11 | 600 | 85 | 85 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2015.csv` |
| 2016 | preseason_safe | true | true | 11 | 600 | 86 | 92 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2016.csv` |
| 2017 | preseason_safe | true | true | 11 | 360 | 57 | 61 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2017.csv` |
| 2018 | same_day_pre_kickoff_verify | true | true | 11 | 360 | 53 | 61 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2018.csv` |
| 2019 | preseason_safe | true | true | 11 | 360 | 54 | 59 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2019.csv` |
| 2020 | preseason_safe | true | true | 11 | 360 | 56 | 62 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2020.csv` |
| 2021 | preseason_safe | true | true | 11 | 360 | 62 | 62 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2021.csv` |
| 2022 | preseason_safe | true | true | 11 | 360 | 59 | 58 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2022.csv` |
| 2023 | preseason_safe | true | true | 11 | 360 | 63 | 56 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2023.csv` |
| 2024 | preseason_safe | true | true | 11 | 360 | 64 | 61 | false | true | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2024.csv` |
| 2025 | unknown_exclude | false | false | 0 | 0 | 0 | 0 | false | false | `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\projections_wayback_fantasypros_2025.csv` |

## Safety Notes

The Wayback URL timestamp is the primary safety check. Captures after the season opener are excluded even if the page content looks like projections. The provided 2020 capture is excluded because it is after kickoff. The 2025 season is excluded because no preseason-safe August 16 capture for the exact leaders URL was found in the tiny lookup, and September 15 captures are post-kickoff.

## Next Validation Step

Create the projection import and feature-build scripts, or add these projection volume columns directly into a dedicated expanded predraft dataset builder. Then run a separate validation comparing ADP-only versus ADP plus preseason projected volume. No WR/RB edge classification should be made until that validation is complete.

Recommended next Codex prompt:

```text
Continue research-only validation in research/validation_v1. Use historical_projections.csv to build preseason projection volume features, merge them into the predraft validation dataset without leakage, then rerun WR/RB validation comparing ADP-only against ADP plus projected volume features.
```
