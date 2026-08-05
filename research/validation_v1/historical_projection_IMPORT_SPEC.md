# Historical Projection Import Spec

Purpose: research-only fantasy football validation. This importer loads historical preseason projection rows and prepares them for ADP-backed validation. Do not use final stats, post-season rankings, weekly projections, post-kickoff projections, or current-season projections as historical draft features.

Accepted input paths, in priority order:

1. `research/validation_v1/historical_projections.csv`
2. `research/validation_v1/historical_projection_market.csv`
3. `data/research/historical_projections.csv`

## Minimum Required Columns

- `season`
- `player_name`
- `position`

At least one projected stat column must exist:

- `projected_fantasy_points`
- `projected_receptions`
- `projected_receiving_yards`
- `projected_receiving_tds`
- `projected_carries`
- `projected_rushing_yards`
- `projected_rushing_tds`
- `projected_total_tds`

Projected fantasy points are optional. The FantasyPros Wayback source currently provides projected WR/RB volume stats but not fantasy-point projections.

## Recommended Provenance Columns

- `projection_source`
- `scoring_format`
- `projection_date`
- `source_url_or_file`
- `is_preseason_projection`
- `projection_safety_status`
- `projected_team`
- `projected_positional_rank`

Rows should be preseason-safe before being used in validation.
