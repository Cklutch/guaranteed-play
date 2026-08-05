# Historical ADP Acquisition Plan

Date: 2026-07-07

Purpose: source real historical preseason ADP for the research-only validation framework. This must represent what fantasy drafters knew before each NFL season, not final finishes, final rankings, post-season rankings, or any result-derived table.

## Data Needed

Target file:

- `research/validation_v1/historical_adp.csv`

Minimum columns:

| Column | Meaning |
| --- | --- |
| `season` | NFL fantasy season being drafted, for example `2021`. |
| `player_name` | Player name as listed by the ADP source. |
| `position` | `WR` or `RB`. |
| `overall_adp` or `adp` | Overall preseason average draft position. Lower means drafted earlier. |
| `positional_adp` | Positional ADP/rank, such as WR31 or RB18 as numeric `31` or `18`. |

Optional but useful:

- `preseason_projection`
- `projection_source`
- `adp_source`
- `half_ppr_adp`
- `ppr_adp`
- `standard_adp`

## Local Search Result

No usable historical preseason ADP file was found in the project.

Found but rejected:

- `data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`: current 2026 rankings, not historical preseason ADP.
- Current app source modules for FantasyPros, Sleeper, Underdog, and sportsbook sources: useful for the app, but not historical preseason market data.
- Existing validation outputs: diagnostics/results only, not source ADP.

## Recommended Source Order

1. Fantasy Football Calculator archive/API

Fantasy Football Calculator exposes archived PPR ADP pages with year selectors from 2010 through the current year and CSV/JSON links. The 2024 archived page states it is archive data and shows a draft window immediately before the season. Example URLs:

- `https://fantasyfootballcalculator.com/adp/ppr/12-team/all/2024`
- `https://fantasyfootballcalculator.com/api/v1/adp/ppr?position=all&teams=12&year=2024`
- `https://fantasyfootballcalculator.com/adp/csv/ppr.csv?position=all&teams=12`

Acquisition approach:

- Prefer the JSON API for each season if historical `year=` works.
- Save one raw CSV/JSON per season under a separate manual download folder, for example `research/validation_v1/source_adp_raw/ffcalc_ppr_2024.csv`.
- Include `adp_source=FantasyFootballCalculator`.
- Use PPR first if half-PPR is unavailable historically; mark scoring in the imported columns.
- Do not use 2026 as historical validation data.

2. FantasyPros historical ADP exports

FantasyPros publishes current ADP pages and may provide historical exports through downloaded CSVs, paid archive access, or saved historical ranking pages. Use only files that clearly represent preseason ADP for that season.

Acquisition approach:

- Export/download per season and scoring format if available.
- Keep source filenames with season and scoring format.
- Include `adp_source=FantasyPros`.
- Reject files labeled final, rest-of-season, weekly, in-season, or post-season.

3. MyFantasyLeague public ADP exports

MFL has public ADP-style exports in some seasons/leagues. These can be useful if they include season, player, position, and ADP before Week 1.

Acquisition approach:

- Prefer preseason snapshots or exported date windows ending before kickoff.
- Add `adp_source=MyFantasyLeague`.
- Validate player-name matching carefully because MFL naming can differ from local outcome files.

4. Curated public datasets or paid vendors

Kaggle, GitHub, or paid fantasy data vendors may have historical ADP/projection archives. Use only if the dataset documents source, scoring, and preseason timing.

Acquisition approach:

- Keep raw files unchanged.
- Add source metadata in `adp_source` and `projection_source`.
- Reject datasets that blend ADP with final rank, final points, auction value after the season, or expert re-draft rankings.

## Import Workflow

After obtaining one or more source CSVs, run:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\import_historical_adp.py" --input "path\to\source.csv" --scoring ppr --source "FantasyFootballCalculator" --derive-positional-adp
```

For a folder of CSVs:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\import_historical_adp.py" --input-dir "research\validation_v1\source_adp_raw" --scoring ppr --source "FantasyFootballCalculator" --derive-positional-adp
```

The importer writes:

- `research/validation_v1/historical_adp.csv`
- `research/validation_v1/historical_adp_validation_report.md`
- `research/validation_v1/historical_adp_validation_coverage.csv`
- `research/validation_v1/historical_adp_validation_duplicates.csv`
- `research/validation_v1/historical_adp_validation_invalid_positions.csv`
- `research/validation_v1/historical_adp_validation_missing_required.csv`
- `research/validation_v1/historical_adp_validation_missing_adp.csv`
- `research/validation_v1/historical_adp_validation_suspicious_adp.csv`
- `research/validation_v1/historical_adp_validation_unmatched_name_examples.csv`

## Validation Gate

Before rerunning model evaluation, inspect the importer report and require:

- Seasons covered are the intended historical seasons.
- WR and RB rows exist for each covered season.
- Duplicate player-season-position rows are explainable or deduped correctly.
- Missing ADP values are near zero.
- Invalid positions are zero.
- Suspicious ADP values are explainable.
- Name unmatched examples are not dominated by key fantasy players.

Then run:

```powershell
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\build_predraft_dataset.py"
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\evaluate_wr_models.py"
& "C:\Users\cklut\Desktop\Projects\Guaranteed Play\.venv\Scripts\python.exe" "research\validation_v1\evaluate_rb_models.py"
```

## Decision Rules

Do not claim an edge unless the model beats ADP.

Classification rules:

| Classification | Requirement |
| --- | --- |
| Not Useful | No ADP comparison, poor coverage, or no lift over ADP. |
| Tie-Breaker Only | Slight lift over ADP, limited range or limited season consistency. |
| Strong Draft Signal | Meaningful lift over ADP across multiple seasons and useful draft ranges. |
| App-Ready | Strong Draft Signal plus stable coverage, interpretable outputs, and no major leakage/source concerns. |
