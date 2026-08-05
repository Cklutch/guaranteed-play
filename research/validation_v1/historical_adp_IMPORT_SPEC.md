# Historical ADP Import Spec

The validation framework will automatically use the first existing file from this list:

1. `research/validation_v1/historical_adp.csv`
2. `research/validation_v1/historical_market.csv`
3. `data/research/historical_adp.csv`

Do not use final season rankings, final fantasy finishes, post-season ranks, rest-of-season rankings, or current 2026 rankings as historical ADP. ADP must be data that would have been known before that fantasy season started.

## Minimum Required Columns

| Column | Required | Notes |
| --- | --- | --- |
| `season` | yes | Fantasy/NFL season being drafted, for example `2021`. |
| `player_name` | yes | Player name. The builder normalizes punctuation, suffixes, and case. |
| `position` | yes | `WR` or `RB` for the current validation. |
| `overall_adp` or `adp` | yes | Overall preseason average draft position. Lower means more expensive. |
| `positional_adp` | strongly recommended | Positional preseason ADP/rank, such as WR31 or RB18 as numeric `31` or `18`. Required for underpriced and beat-ADP labels. |

## Optional Columns

| Column | Notes |
| --- | --- |
| `preseason_projection` | Preseason projected fantasy points. |
| `projection_source` | Source name for projections. |
| `adp_source` | Source name for ADP, such as FantasyPros, Underdog, FFCalc, etc. |
| `half_ppr_adp` | Half-PPR ADP if available. |
| `ppr_adp` | PPR ADP if available. |
| `standard_adp` | Standard-scoring ADP if available. |

## Template

Use `research/validation_v1/historical_adp_TEMPLATE.csv` as the column template.

## Importer Workflow

A helper script is available:

- `research/validation_v1/import_historical_adp.py`

Single file example:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\import_historical_adp.py" --input "path\to\source.csv" --scoring ppr --source "FantasyFootballCalculator" --derive-positional-adp
```

Folder example:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\import_historical_adp.py" --input-dir "research\validation_v1\source_adp_raw" --scoring ppr --source "FantasyFootballCalculator" --derive-positional-adp
```

The importer accepts common source column names such as `player`, `name`, `pos`, `adp`, `average draft position`, `overall`, `position_rank`, and scoring-specific ADP columns. It writes normalized rows to `research/validation_v1/historical_adp.csv`.

## Importer Validation Outputs

After importing source CSVs, inspect:

- `historical_adp_validation_report.md`
- `historical_adp_validation_coverage.csv`
- `historical_adp_validation_duplicates.csv`
- `historical_adp_validation_invalid_positions.csv`
- `historical_adp_validation_missing_required.csv`
- `historical_adp_validation_missing_adp.csv`
- `historical_adp_validation_suspicious_adp.csv`
- `historical_adp_validation_unmatched_name_examples.csv`

## Diagnostics Written By The Builder

After running `build_predraft_dataset.py`, inspect:

- `dataset_metadata.csv`
- `adp_merge_diagnostics.json`
- `adp_merge_coverage_by_season.csv`
- `adp_unmatched_player_examples.csv`
- `adp_market_rows_not_matched_examples.csv`

The validation results are not allowed to claim an edge unless ADP coverage is good enough and model lift over ADP is positive across multiple seasons and useful draft ranges.
