# Model Registry

Canonical index of research models and their production eligibility.

**Default status for anything listed here is RESEARCH_ONLY.** An entry is
eligible for live use only if it explicitly says so. If you are wiring
something into rankings, tiers, recommendations, or UI, check this file
first.

---

## `wr_conditional_bust_risk_research`

**Status:** `RESEARCH_ONLY — NOT ELIGIBLE FOR LIVE DRAFT RECOMMENDATIONS`

**Scope:** WR only; historical walk-forward research. No current-season scoring path exists.

**Question tested:** Whether WR features add conditional bust-risk information beyond continuous ADP.

**Primary finding:** CORE features showed incremental out-of-sample predictive information beyond continuous ADP in the 2014+ sample. Nested comparison (`f(ADP)` vs `f(ADP, features)`) improved AUC, Brier score and log-loss with season-blocked CIs excluding zero.

**Important limitation:** Incremental predictive value did **not** establish an actionable draft-policy edge. Improving a conditional probability estimate and improving a draft decision are different results requiring different tests. Only the former was demonstrated.

**Policy status:** **Unvalidated.** The confirmatory policy test was **infeasible / underpowered** under the preregistered exact matching design, because replacement opportunity count was structurally collinear with ADP rank under the defined candidate pool. Matching on both reduced to exact ADP-rank matching and retained only tied-ADP observations. This is a property of that candidate-pool construction, not of draft-policy evaluation generally.

**FULL specification:** **Inconclusive, not negative proof.** Positive point estimates with insufficient precision across seven seasons.

**Production decision:** **Do not use** this model, its probabilities, feature coefficients, risk labels, rankings, or outputs in live draft recommendations, player ranks, tiers, or UI explanations.

**Allowed use:** Historical research reference; data-pipeline validation; methodology example; candidate input for a future separately preregistered study.

**Reopen criteria** — a separately preregistered, realistic choice-set policy study that:
1. defines the candidate set available at each simulated draft slot;
2. compares ADP-only versus ADP-plus-model decisions within identical candidate sets;
3. pre-specifies roster constraints, decision rules, and outcomes;
4. demonstrates a policy edge beyond matched / random / ADP controls.

**Frozen commits:**

| commit | purpose |
|---|---|
| `8d2862ae6bcbc2ec2aec6c3134eec270b9a7439b` | Analysis snapshot — code, tests, memo, verified input dataset, result artifacts, archived v1 retraction record |
| `056d7da6aaab5a2d5a5395b6e40a615863f6e703` | Metadata binding — ties provenance metadata to the analysis snapshot |
| `cc04ea2f59cd2a19fa30a6a2c2c8b0db7f2aeac5` | Provenance-scope clarification — cleanliness claims are package-scoped, not repository-scoped |

**Authoritative references:**
- Memo (findings, limitations, recommendation): [`validation_v1/WR_BUST_DECISION_MEMO.md`](validation_v1/WR_BUST_DECISION_MEMO.md)
- Provenance record (commits, seeds, environment, input-data identities): [`validation_v1/data/wr_bust_final/feasibility_metadata.json`](validation_v1/data/wr_bust_final/feasibility_metadata.json)
- Superseded v1 and why it was retracted: [`../archive/retracted_wr_bust_v1/README.md`](../archive/retracted_wr_bust_v1/README.md)

**Reproduction:**
```bash
python research/validation_v1/wr_bust_final_validation.py
```
```bash
python research/validation_v1/wr_bust_policy_feasibility.py
```
```bash
cd research/validation_v1 && python -m unittest discover -v
```

**Integration status (verified at registry creation):** no reference to this
model or its artifacts exists in `Home.py`, `draftkit/`, `pages_archive/`,
`utils.py`, or `app.py`. It is not wired into any production path.
