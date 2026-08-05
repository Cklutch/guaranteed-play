# Sleeper Projection Feasibility Report

Date: 2026-07-07

Scope: research-only fantasy football validation. This report does not build a projection dataset, does not modify the Streamlit app, does not add UI, does not scrape websites, and does not make betting recommendations.

## Decision

Gate A - true historical preseason season-long projections: failed.

Gate B - limited Week 1 projection pilot: partially true, but not safe enough to build without explicit approval.

Data build status: skipped.

Reason: Sleeper's official public API documentation does not document projection endpoints. An unofficial weekly projection endpoint responded for tiny 2023 and 2024 Week 1 samples, but the returned `last_modified` timestamps were after Week 1 started, and there is no documented snapshot/as-of parameter to prove the data was captured before kickoff or before the season. That fails the pre-draft safety standard for Historical Projection Import V1.

## Official Sleeper API Review

The official Sleeper API documentation describes a read-only API for users, leagues, drafts, rosters, matchups, transactions, NFL state, players, and trending players. It documents `https://api.sleeper.app/v1/players/nfl` for player metadata, but it does not document season-long projections, weekly projections, projection snapshots, or historical projection endpoints.

Existing project code matches that boundary:

- `draftkit/sleeper.py` imports league settings only.
- `draftkit/sleeper_client.py` imports draft, roster, league, and debug information.
- `draftkit/data_sources/sleeper_source.py` imports player metadata from `https://api.sleeper.app/v1/players/nfl` and intentionally leaves `projection_points`, `projection_rank`, `adp`, and `adp_rank` empty.

## Tiny Endpoint Probe

These endpoints are not documented in Sleeper's official public docs, so they must be treated as unofficial investigation only.

Tested:

| Endpoint | Status | Result |
| --- | ---: | --- |
| `https://api.sleeper.app/projections/nfl/2024/1?season_type=regular` | 200 | Returned Week 1 projection rows. |
| `https://api.sleeper.com/projections/nfl/2024/1?season_type=regular` | 200 | Returned Week 1 projection rows. |
| `https://api.sleeper.app/projections/nfl/2023/1?season_type=regular` | 200 | Returned Week 1 projection rows. |
| `https://api.sleeper.com/projections/nfl/2023/1?season_type=regular` | 200 | Returned Week 1 projection rows. |
| `https://api.sleeper.app/projections/nfl/regular/2024/1` | 404 | No usable response. |
| `https://api.sleeper.app/projections/nfl/2024/1` | 400 | Missing or invalid query shape. |
| `https://api.sleeper.app/projections/nfl/2024/season?season_type=regular` | 500 | No usable season-long response. |

The tiny sample did not download a full dataset.

## Fields Observed

The unofficial Week 1 rows included:

- top-level fields: `status`, `date`, `stats`, `category`, `last_modified`, `week`, `sport`, `season_type`, `season`, `player`
- fantasy point projections: `pts_half_ppr`, `pts_ppr`, `pts_std`
- receiving/WR-relevant volume fields: `rec`, `rec_tgt`, `rec_yd`, `rec_td`
- rushing/RB-relevant volume fields: `rush_att`, `rush_yd`, `rush_td`
- other fields in samples: `gp`, `adp_dd_ppr`, `pos_adp_dd_ppr`, first-down and yardage-bucket fields

Observed sample timing:

- 2024 Week 1 sample `date`: `2024-09-05`
- 2024 Week 1 sample `last_modified`: `2024-09-10T03:45:16.291Z`
- 2023 Week 1 sample `date`: `2023-09-07`
- 2023 Week 1 sample `last_modified`: `2023-09-12T03:45:22.454Z`

Those `last_modified` timestamps are not pre-kickoff proof. They are after the Thursday opener and around/after the Week 1 slate. Without an official snapshot parameter, these rows cannot be treated as pre-Week-1 projections.

## Gate A - Historical Preseason Season-Long Projections

| Requirement | Result |
| --- | --- |
| Season-long player projections available | Not confirmed. No official endpoint found. |
| Historical seasons available | Not confirmed for season-long projections. |
| Available before each NFL season | Not verifiable. |
| Projected fantasy points included | Not confirmed for season-long projections. |
| Volume stats included | Not confirmed for season-long projections. |
| Projection date/snapshot timing verifiable | No. |

Gate A result: failed.

## Gate B - Limited Week 1 Projection Pilot

| Requirement | Result |
| --- | --- |
| Historical Week 1 projections available | Partially yes via unofficial endpoint for 2023 and 2024 samples. |
| Seasons available | 2023 and 2024 Week 1 responded in tiny tests; older years may respond but were not safely enumerated. |
| Documented endpoint | No. Endpoint is not in official Sleeper docs. |
| Fantasy points present | Yes: `pts_half_ppr`, `pts_ppr`, `pts_std`. |
| Underlying stat projections present | Yes in samples: targets, receptions, receiving yards, receiving TDs, carries, rushing yards, rushing TDs. |
| Pre-kickoff / pre-Week-1 verifiable | No. Sample `last_modified` timestamps were after Week 1 began, and no snapshot parameter is documented. |
| Can be labeled limited pilot | Yes, but only as unofficial and not pre-kickoff-safe. |

Gate B result: partially true, not build-eligible under the current safety rules.

## Draft Validation Safety

Sleeper should not be used for Historical Projection Import V1 right now.

For a beginner fantasy football framing: this is like finding an old cheat sheet but not knowing whether it was printed before the draft or updated after the games started. It may contain useful-looking numbers, but if we cannot prove when those numbers existed, it is not fair evidence that the model could have helped you draft better than ADP.

## Recommended Data Source Instead

Use a projection archive that provides explicit preseason/as-of timestamps and season-long fantasy projections, ideally with underlying player volume stats. Good candidates:

- FantasyPros historical preseason projections CSVs, if available with archived preseason dates
- DraftSharks historical preseason projections, if exportable with source date
- FantasyData or SportsDataIO historical projections, if licensed and timestamped
- Manually archived preseason projection CSVs from known draft dates

Minimum acceptable import fields remain:

- `season`
- `player_name`
- `position`
- `projection_source`
- `snapshot_date`
- `projected_fantasy_points`
- volume stats such as targets, receptions, carries, receiving yards, rushing yards, and TDs where available

## Next Codex Prompt

```text
Continue research-only validation in research/validation_v1. I have added a timestamped historical preseason season-long projection CSV. Validate its snapshot dates, merge it into the predraft dataset, rerun WR/RB validation against ADP, and classify signals without modifying the app.
```

## Sources

- Sleeper official API docs: `https://docs.sleeper.com/`
- Existing project files inspected: `draftkit/sleeper.py`, `draftkit/sleeper_client.py`, `draftkit/data_sources/sleeper_source.py`
