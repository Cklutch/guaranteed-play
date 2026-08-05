# The Odds API Historical Props Pilot V1

Date: 2026-07-07

Scope: research-only fantasy football validation. This is not a betting system, does not recommend wagers, does not scrape sportsbooks, does not connect to sportsbook accounts, and uses only documented The Odds API endpoints.

## Build Status

`BUILD_STATUS: built_preseason_sportsbook_pilot`

No live API pull was run during this investigation. The project now has a documented downloader at `research/validation_v1/download_theoddsapi_historical_props.py`, but it only writes `historical_sportsbook_props.csv` when run manually with `THE_ODDS_API_KEY` and explicit Week 1 snapshot/window arguments.

## Gate Decision

Gate A - true preseason season-long props: failed.

The official betting-market documentation lists NFL event-level player props such as receptions, receiving yards, rushing attempts, rushing yards, rushing touchdowns, receiving touchdowns, and anytime touchdown. It does not document NFL season-long player markets such as season receiving yards, season receptions, season rushing yards, or season rushing touchdowns. The general `outrights` market is documented for final tournament/competition outcomes, not player season-long WR/RB volume props.

Gate B - pre-Week-1 game props: passed as a limited pilot only.

The official historical event odds endpoint returns historical odds for a single event at a specified timestamp. It accepts available market keys, including player props, after `2023-05-03T05:30:00Z`, and historical snapshots are available at 5-minute intervals. The response contains a snapshot `timestamp` and event `commence_time`, so rows can be excluded unless `timestamp < commence_time`. Historical events can be filtered by `commenceTimeFrom` and `commenceTimeTo`, which lets the pilot target NFL Week 1 games.

This is not a true fantasy draft projection source. It is a pre-kickoff Week 1 game-prop source. If used, it should be labeled as a limited role/usage signal for Week 1, not as a preseason season-long market.

## Documented Capabilities Reviewed

Historical NFL odds coverage: supported through historical odds/events/event-odds endpoints on paid plans.

Historical player prop availability: documented for historical event odds after `2023-05-03T05:30:00Z`.

Available seasons: practically limited for player props to 2023 and later because additional markets, including player props, are documented as available after `2023-05-03T05:30:00Z`.

Available bookmakers: region/bookmaker dependent. Coverage is mainly US sports and US bookmakers for player props; the downloader supports `--regions` or explicit `--bookmakers`.

Available WR/RB-relevant markets:

- `player_receptions`
- `player_reception_yds`
- `player_reception_tds`
- `player_rush_attempts`
- `player_rush_yds`
- `player_rush_tds`
- `player_rush_reception_yds`
- `player_rush_reception_tds`
- `player_anytime_td`

Season-long player props: not documented for NFL player volume markets.

Timestamps/snapshots: documented. Historical endpoints return snapshot `timestamp`, `previous_timestamp`, and `next_timestamp`; historical event odds snapshots are documented at 5-minute intervals.

Pre-game filtering: supported by comparing snapshot `timestamp` to event `commence_time`. The downloader also writes `snapshot_before_commence_time`, `is_pre_week1_snapshot`, and `sportsbook_data_safety_status`.

## Downloader Behavior

Created:

`research/validation_v1/download_theoddsapi_historical_props.py`

The script:

- uses `THE_ODDS_API_KEY` from the environment
- never hardcodes or prints the API key
- queries documented historical events and historical event odds endpoints only
- saves raw JSON responses under `research/validation_v1/source_sportsbook_raw/theoddsapi/`
- writes normalized rows to `research/validation_v1/historical_sportsbook_props.csv`
- writes only rows classified as `pre_week1_game_prop`
- excludes rows where the snapshot is not before kickoff
- labels rows as non-season-long with `is_season_long=false`

Example command for a small 2024 Week 1 pilot:

```powershell
$env:THE_ODDS_API_KEY="..."
& ".\.venv\Scripts\python.exe" "research\validation_v1\download_theoddsapi_historical_props.py" `
  --season 2024 `
  --snapshot "2024-09-04T12:00:00Z" `
  --commence-from "2024-09-05T00:00:00Z" `
  --commence-to "2024-09-10T12:00:00Z" `
  --regions us
```

## Validation Results

Seasons covered: not run yet.

Markets covered: not run yet.

Sportsbooks/bookmakers covered: not run yet.

Safe rows: not run yet.

Unsafe rows excluded: not run yet.

WR coverage: not run yet.

RB coverage: not run yet.

Did sportsbook features improve WR lift over ADP? Not tested yet.

Did sportsbook features improve RB lift over ADP? Not tested yet.

Did any bucket become Tie-Breaker Only? Not tested yet.

Did anything become Strong Draft Signal? Not tested yet.

Did anything become App-Ready? No. A limited Week 1 game-prop pilot cannot be app-ready until it shows repeatable lift over ADP across multiple seasons and useful draft ranges.

## Limitations

This is a limited pilot, not final validation.

Week 1 game props are not the same thing as preseason season-long fantasy projections. They may encode health, role, team context, and matchup expectations for the opening game, but they do not directly answer full-season fantasy value.

Historical player props are documented only after `2023-05-03T05:30:00Z`, so the available sample is likely too short for App-Ready classification by itself.

Availability depends on paid-plan access, bookmaker coverage, market coverage, and whether books posted a given player market at the chosen snapshot.

Player positions are inferred from existing validation datasets. Any player not matched to WR/RB will be labeled `UNKNOWN` and should not drive validation claims.

## Next Step

Run one small season pilot first, inspect raw coverage, and only then decide whether to spend quota on multiple seasons. The most useful next data source remains true historical preseason season-long player projections or season-long player prop archives, because those align directly with fantasy draft decisions.

## Sources

- The Odds API V4 documentation: `https://the-odds-api.com/liveapi/guides/v4/`
- The Odds API betting markets documentation: `https://the-odds-api.com/sports-odds-data/betting-markets.html`
