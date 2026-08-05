# Projection Bucket/Window Lightweight Report

Date: 2026-07-07

Scope: research-only. This lightweight script fits only the best WR/RB projection candidate models and does not rerun the full model grid or modify the app.

## Required Answers

Does the WR projection Tie-Breaker signal work in any draft window? No repeatable WR bucket cleared the gate.

Does the RB projection Tie-Breaker signal work in any draft window? No repeatable RB bucket cleared the gate.

Does any bucket reach Strong Draft Signal? No.

Is anything App-Ready? No. This remains research-only and the projection model is weaker than the best ADP-only/non-projection full-pool models.

Did projected volume help in similarly priced player comparisons? The bucket tables below answer that directly; only buckets with positive lift and at least 50% season repeatability are promoted.

Is the signal coming from projection features or mostly ADP? The tested model includes ADP plus projection and expanded features. Because prior full-pool validation found stronger ADP-only/non-projection models, any promoted bucket should be treated as secondary evidence, not proof that projected volume alone is the source.

## WR Buckets

Buckets tested: `11`.
Tie-Breaker Only buckets: `0`.
Strong Draft Signal buckets: `0`.

No draft window cleared the repeatability gate.

## RB Buckets

Buckets tested: `11`.
Tie-Breaker Only buckets: `0`.
Strong Draft Signal buckets: `0`.

No draft window cleared the repeatability gate.

## Classification Summary

Not Useful buckets: `22`.
Tie-Breaker Only buckets: `0`.
Strong Draft Signal buckets: `0`.
App-Ready buckets: `0`.

## Next Research Step

Since no projection bucket was promoted, the next step is to compare the best near-miss buckets against ADP-only and non-projection models, then check whether projection fields add signal beyond ADP or mostly travel with ADP.

Recommended next Codex prompt:

```text
Continue research-only validation in research/validation_v1. Compare the best near-miss projection buckets against ADP-only and non-projection models in the same buckets, then identify whether projection volume fields add signal beyond ADP. Do not modify the app.
```

