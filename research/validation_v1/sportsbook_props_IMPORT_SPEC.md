# Sportsbook Props Import Spec

Purpose: research-only fantasy football validation. This framework imports historical, pre-draft-safe sportsbook player props and converts them into fantasy draft features. It must not be used to recommend wagers, place bets, scrape sportsbooks, or connect to sportsbook accounts.

The importer uses the first existing file from this list:

1. `research/validation_v1/historical_sportsbook_props.csv`
2. `research/validation_v1/historical_player_props.csv`
3. `data/research/historical_sportsbook_props.csv`

Do not use current 2026 props as historical validation data. Do not use live sportsbook pages, undocumented APIs, final stats, final fantasy rankings, post-season data, or in-season closing lines as preseason draft features.

## Minimum Required Columns

| Column | Required | Notes |
| --- | --- | --- |
| `season` | yes | NFL/fantasy season the prop snapshot applies to. |
| `player_name` | yes | Player name. For team markets, use team name or leave player blank only if `market_type=team`. |
| `position` | yes | `WR`, `RB`, or `TEAM` for team context rows. |
| `market` | yes | Market name such as `season_receiving_yards` or `rushing_yards`. |
| `line` | yes | Numeric prop line. |
| `sportsbook` | yes | Book/source name. |
| `odds` | yes | American odds if only one side is known, or over odds if `over_odds` is absent. |
| `odds_format` | yes | Use `american` for this V1 framework. |
| `snapshot_date` | yes | Date the line was captured. Must be before the season or before fantasy draft context. |

## Strongly Recommended Columns

| Column | Notes |
| --- | --- |
| `team` | Player team at snapshot. |
| `opponent` | Opponent for game markets, if applicable. |
| `market_type` | `player`, `team`, or `game`. |
| `over_odds` | American odds for over side. |
| `under_odds` | American odds for under side. |
| `source` | Provider or dataset name. |
| `source_url_or_file` | URL or file provenance. |
| `is_season_long` | `true`/`false`; season-long markets are preferred for draft validation. |
| `is_preseason_snapshot` | `true`/`false`; must be true for app-ready research claims. |

## Supported Markets

WR:

- `receiving_yards`
- `receptions`
- `receiving_tds`
- `anytime_td`
- `season_receiving_yards`
- `season_receptions`
- `season_receiving_tds`

RB:

- `rushing_yards`
- `rushing_attempts`
- `receptions`
- `receiving_yards`
- `rushing_tds`
- `anytime_td`
- `season_rushing_yards`
- `season_rushing_tds`
- `season_receptions`

Team context:

- `team_win_total`
- `team_total_points`
- `game_total`
- `team_implied_points`

## Odds Handling

The importer converts American odds to raw implied probability and computes no-vig probability when both `over_odds` and `under_odds` are available. These probabilities are fantasy expectation features only. They are not betting recommendations.

## Diagnostics

After import, inspect:

- `sportsbook_props_import_diagnostics.json`
- `sportsbook_props_coverage_by_season.csv`
- `sportsbook_props_unmatched_player_examples.csv`
- `sportsbook_props_market_rows_not_matched_examples.csv`

## Next Step After Import

Run:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\import_sportsbook_props.py"
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\build_sportsbook_features_v1.py"
```
