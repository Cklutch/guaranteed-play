# Sportsbook-Implied Projection V1 Report

Date: 2026-07-07

Scope: research-only fantasy football validation. This is not a betting system, does not recommend wagers, does not scrape sportsbooks, and does not connect to sportsbook accounts.

## Data Status

Sportsbook data file used: `not_found`

Sportsbook source: `not_found`

Seasons covered: none

Markets covered: none

WR sportsbook coverage: 0 / 6008

RB sportsbook coverage: 0 / 4123

## Required CSV Format

No historical sportsbook prop data was found locally.

Use `research/validation_v1/historical_sportsbook_props_TEMPLATE.csv` and the format documented in `sportsbook_props_IMPORT_SPEC.md`.

Minimum columns: `season`, `player_name`, `position`, `market`, `line`, `sportsbook`, `odds`, `odds_format`, `snapshot_date`.

## Validator Preparation

The WR/RB validators now include sportsbook feature groups for future runs: sportsbook-only, ADP + sportsbook, ADP + sportsbook + prior production, ADP + sportsbook + expanded features, and ADP + all pre-draft-safe features. Because sportsbook feature coverage is currently 0, rerunning those groups would only produce skipped/no-feature rows and no valid edge claim.

## Validation Verdict

Did sportsbook features improve WR lift over ADP? Not tested; no sportsbook prop rows are available.

Did sportsbook features improve RB lift over ADP? Not tested; no sportsbook prop rows are available.

Did any bucket become Tie-Breaker Only? No.

Did any bucket become Strong Draft Signal? No.

Did anything become App-Ready? No.

Which sportsbook-derived features helped most? Not answerable without historical props coverage.

Which markets are most useful? Not answerable without historical props coverage. Season-long volume markets are expected to be more draft-relevant than single-game anytime touchdown markets, but that must be validated before use.

## Next Research Build

Import a legal, documented or manually provided historical preseason prop CSV with season-long WR/RB volume markets. Then rerun `import_sportsbook_props.py`, `build_sportsbook_features_v1.py`, WR/RB validators, and draft-window analysis.

Recommended next Codex prompt:

```text
Continue research-only validation in research/validation_v1. I have added historical sportsbook prop CSV data. Import it, validate preseason safety and market coverage, build sportsbook-implied fantasy features, rerun WR/RB validators and bucket analysis, and classify signals without making betting recommendations.
```
