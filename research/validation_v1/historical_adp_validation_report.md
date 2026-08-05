# Historical ADP Validation Report

Output file: `C:\Users\cklut\Desktop\Projects\Guaranteed Play\research\validation_v1\historical_adp.csv`
Source files: research\validation_v1\source_adp_raw\ffcalc_ppr_2010.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2011.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2012.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2013.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2014.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2015.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2016.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2017.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2018.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2019.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2020.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2021.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2022.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2023.csv, research\validation_v1\source_adp_raw\ffcalc_ppr_2024.csv
Rows imported: 1,834
Seasons covered: 2010-2024 (15 seasons)

## Validation Counts

Duplicate player-season-position rows found before dedupe: 8
Rows with missing overall ADP: 0
Rows with invalid positions: 0
Rows with suspicious ADP values: 0
Rows not matched to local outcome player names: 200 shown in sample file

## Coverage By Season/Position

|   season | position   |   rows |   rows_with_overall_adp |   rows_with_positional_adp |   median_overall_adp |   max_overall_adp |
|---------:|:-----------|-------:|------------------------:|---------------------------:|---------------------:|------------------:|
|     2010 | RB         |     64 |                      64 |                         64 |                82.25 |             160.4 |
|     2010 | WR         |     72 |                      72 |                         72 |                97.75 |             165.9 |
|     2011 | RB         |     57 |                      57 |                         57 |                71.4  |             165.2 |
|     2011 | WR         |     62 |                      62 |                         62 |                84.55 |             157.9 |
|     2012 | RB         |     33 |                      33 |                         33 |                41.7  |             152.8 |
|     2012 | WR         |     36 |                      36 |                         36 |                56.1  |             145   |
|     2013 | RB         |     60 |                      60 |                         60 |                59.45 |             152.2 |
|     2013 | WR         |     56 |                      56 |                         56 |                81.95 |             156.5 |
|     2014 | RB         |     60 |                      60 |                         60 |                73.1  |             157.5 |
|     2014 | WR         |     65 |                      65 |                         65 |                76.1  |             161.5 |
|     2015 | RB         |     61 |                      61 |                         61 |                75.5  |             156.8 |
|     2015 | WR         |     68 |                      68 |                         68 |                84.35 |             163.1 |
|     2016 | RB         |     57 |                      57 |                         57 |                73.8  |             159.4 |
|     2016 | WR         |     65 |                      65 |                         65 |                70.6  |             159.3 |
|     2017 | RB         |     62 |                      62 |                         62 |                76.6  |             160.7 |
|     2017 | WR         |     63 |                      63 |                         63 |                70    |             162.9 |
|     2018 | RB         |     60 |                      60 |                         60 |                67.85 |             164.3 |
|     2018 | WR         |     65 |                      65 |                         65 |                79.5  |             166.8 |
|     2019 | RB         |     61 |                      61 |                         61 |                69.8  |             159.2 |
|     2019 | WR         |     66 |                      66 |                         66 |                81.1  |             166.9 |
|     2020 | RB         |     65 |                      65 |                         65 |                74.6  |             161.4 |
|     2020 | WR         |     66 |                      66 |                         66 |                80    |             164.2 |
|     2021 | RB         |     65 |                      65 |                         65 |                79.1  |             172.7 |
|     2021 | WR         |     69 |                      69 |                         69 |                85.7  |             170.9 |
|     2022 | RB         |     53 |                      53 |                         53 |                61.7  |             130.5 |
|     2022 | WR         |     58 |                      58 |                         58 |                69.95 |             153.8 |
|     2023 | RB         |     59 |                      59 |                         59 |                79.2  |             163   |
|     2023 | WR         |     74 |                      74 |                         74 |                80.7  |             169.1 |
|     2024 | RB         |     60 |                      60 |                         60 |                83.55 |             174   |
|     2024 | WR         |     72 |                      72 |                         72 |                78.3  |             170.2 |

## Decision

Structurally valid for import. Next step: run `build_predraft_dataset.py` and inspect merge coverage.