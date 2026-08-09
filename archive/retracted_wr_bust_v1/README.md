# RETRACTED — WR bust model v1

**The results in `bust_model_results.csv` are invalid. Do not cite them.**

The headline claim was:

> WR conditional-bust model, AUC 0.7498 vs 0.7224 for ADP alone
> (+0.0274, CI [+0.0094, +0.0459]), beating ADP in 85% of 13 seasons.

## Why it is wrong

Feature coverage is staggered across history:

| seasons | availability |
|---|---|
| 2010-2013 | essentially nothing but `age` |
| 2014-2016 | snap share / divergence / durability, but **no route data** |
| 2017-2025 | all features, 79-92% |

`sklearn.impute.SimpleImputer` **silently drops columns with no observed
values**. Early walk-forward folds therefore fit a *smaller model* than late
folds, and the pooled figure averaged across different specifications. It was
not comparing like with like.

## What the corrected result is

With a guard requiring every feature to be observed in the training fold:

| specification | model AUC | ADP AUC | gain |
|---|---|---|---|
| FULL (2017+) | 0.717 | 0.738 | **-0.021** |
| CORE (2014+) | 0.684 | 0.732 | **-0.048** |

ADP alone is better. A leakage check passed (shuffled-label AUC 0.50), so the
evaluation harness was sound — the specification was not.

A companion policy result (+42.5 VOR from fading high-risk WRs) was also
invalid as evidence of model skill: random replacement in the same ADP window
scored **+48.9**, better than the model's selection.

## Status

Retained only for reproducibility of the retraction. **No active script,
document, or output imports or cites these files.** Superseded by
`research/validation_v1/wr_bust_final_validation.py`.

## `test_wr_bust_model_v1.py`

This test harness belongs to the **superseded v1 analysis**. Its original
pooled result was invalid because fold specifications differed under
staggered feature coverage — early folds silently fit a smaller model.

Retained for **forensic reproducibility only**. No active workflow, memo
conclusion, or current result may import, execute, or cite it as valid
evidence. The superseding analysis is
`research/validation_v1/wr_bust_final_validation.py`.
