# Wayback FantasyPros Projection POC Report

Date: 2026-07-07

Scope: research-only 2019 proof of concept. This does not modify the Streamlit app, add UI, build recommendations, or claim model edge.

## Summary

Did the scraper access the page? Yes.
Was the archive preseason-safe? Yes; archive date `2019-07-15` is before the 2019 NFL regular season.
Were projection tables visible in HTML? Yes; `11` stat tables were found.
Sections extracted: Rushing Attempts, Rushing Yards, Rushing TDs, Receptions, Receiving Yards, Receiving TDs.
Raw player-stat rows extracted: `360`.
Clean WR/RB rows created: `113` total, `54` WR and `59` RB.
Projected fantasy points available? No. The page exposed stat-leader projection sections, not a fantasy-points table.
Projected volume stats available? Yes: rushing attempts, rushing yards, rushing TDs, receptions, receiving yards, and receiving TDs.
Was position resolved successfully? Yes for WR/RB rows using local validation/master datasets; unresolved extracted player rows: `25`.

## Diagnostics

- Source URL: `https://web.archive.org/web/20190715211021/https://www.fantasypros.com/nfl/projections/leaders.php`
- Raw HTML: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\source_projection_raw\wayback_fantasypros_2019.html`
- Output CSV: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\historical_projections_wayback_2019.csv`
- Diagnostics JSON: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\wayback_fantasypros_2019_parse_diagnostics.json`
- Rows extracted by section: `{"Receiving TDs": 60, "Receiving Yards": 60, "Receptions": 60, "Rushing Attempts": 60, "Rushing TDs": 60, "Rushing Yards": 60}`
- Missing projected fantasy points count: `113`

## Viability

Is this scraper/cleaner approach viable? Yes, as a volume-stat projection source proof of concept.

Before expanding to more years, the parser should be tested against several archived dates because FantasyPros page layout and Wayback captures may differ by year. The next version should also look for archived FantasyPros position-specific projection pages or downloadable tables that include fantasy points and positions directly, reducing reliance on local position matching.

Expand to more years: yes.
