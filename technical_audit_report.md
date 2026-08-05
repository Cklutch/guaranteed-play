# Guaranteed Play — Technical Audit: Why Correlation to ADP/Expert Consensus Is Weak

Date: 2026-08-05
Scope: `draftkit/`, `data/`, `scripts/`, `utils.py`, `pages/`, plus supporting research under `research/validation_v1/`.

This audit traces the live Streamlit app's ranking pipeline end to end: raw source → merged dataset → engineered features → recommendation score. Every claim below is tied to a specific file/line/snippet. Where I had to infer intent (e.g., "this looks like placeholder data"), that is called out explicitly rather than stated as fact.

**Headline finding, confirmed by direct execution (see Section 5):** the live recommendation engine (`draftkit.draft_analysis.build_recommendation_rankings_df`) only ever ranks **27 of 3,931 players** in the current dataset, because it silently drops every player missing `projection_points` — and `projection_points` is populated for exactly 27 rows, which trace back to a small hand-authored sample file, not a real projections feed. Within that 27-player universe, the model's own `final_score` correlates to ADP at only **Spearman ρ = -0.48** (Pearson r = -0.39), well short of what a market-aligned ranking should show. This one pipeline defect explains most of the "weak correlation" symptom on its own.

---

## 1. Target / Label

**There is no trained model and no explicit prediction target.** `draftkit` is a hand-weighted heuristic scoring/ranking engine, not a regression or classifier. There is no `y` variable, no season-total or per-game fantasy-points label being fit against, and no held-out evaluation set in the live app path.

- The closest thing to a "target" is `final_score` in [draftkit/draft_analysis.py:1480](draftkit/draft_analysis.py:1480), a weighted blend of five hand-normalized components (see Section 5). It is not fit to any historical outcome; the weights are hardcoded constants:

```python
DEFAULT_MASTER_COMPONENT_WEIGHTS = {
    "projection": 0.30,
    "position_need": 0.25,
    "adp_value": 0.15,
    "tier_urgency": 0.15,
    "team_fit": 0.15,
}
```
[draftkit/draft_analysis.py:46-52](draftkit/draft_analysis.py:46)

- The underlying `projection_points` input (the one real "prediction" in the system) is a **preseason, full-season total** — a single static number per player (e.g., Josh Allen = 337.5). There is no per-game rate anywhere in the app pipeline, and no explicit statement of scoring format (PPR/half-PPR/standard) in code or column names.
- **Time window:** single upcoming season, pre-draft, one point-in-time snapshot. There is no weekly/game-log grain anywhere in `draftkit` or `data/` (confirmed in Section 3) — so there is nothing to build a "per-game average" or in-season target from even if you wanted to.
- Research-side work (`research/validation_v1/`, not wired into the app) does define real historical targets — `WR_Top24`, `WR_Top12`, `RB_Top24`, `RB_Top12` season-finish thresholds — but that code path never feeds `draftkit`. See Section 5 for what happened when those targets were actually validated against ADP.

### ADP benchmark — and a critical mislabeling bug

Yes, an "ADP comparison" exists in the app (`adp_delta`, `value_score`, `adp_bonus` in `draft_analysis.py`), but **the "ADP" field feeding all of it is not real market ADP.**

- `draftkit/data_pipeline.py` merges three sources in priority order: Sleeper (no ADP field at all), FantasyPros (`data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv`), then "Underdog" (`draftkit/data_sources/underdog_source.py`).
- The FantasyPros source file's actual header is:
  ```
  "RK",TIERS,"PLAYER NAME",TEAM,"POS","BYE WEEK","UPSIDE ","BUST ","SOS SEASON","ECR VS. ADP"
  ```
  There is **no ADP column in this file at all.** But `map_fantasypros_columns()` in [draftkit/data_sources/fantasypros_source.py:91-112](draftkit/data_sources/fantasypros_source.py:91) does this:
  ```python
  "adp": safe_col(df, ["adp", "ADP", "avg_adp", "Average ADP"]),
  "adp_rank": safe_col(df, ["adp_rank", "ADP Rank", "rank", "Rank", "RANK", "RK"]),
  ```
  and then, when building the row:
  ```python
  "adp": (
      row.get(column_map["adp"])
      if column_map.get("adp")
      else row.get(column_map["adp_rank"])
      if column_map.get("adp_rank")
      else pd.NA
  ),
  ```
  [draftkit/data_sources/fantasypros_source.py:288-294](draftkit/data_sources/fantasypros_source.py:288)

  Since there's no real `adp`/`ADP` column, `column_map["adp"]` is `None`, so this falls through to `column_map["adp_rank"]`, which resolves to `"RK"` — **the FantasyPros expert consensus rank column.** I verified this directly: Josh Allen's row in the raw CSV is `"26",4,"Josh Allen",BUF,"QB1",...` (RK=26), and his row in `data/processed/master_players.csv` is `adp=26.0, adp_rank=26.0`. **The "ADP" field for every FantasyPros-sourced player is literally that same source's own expert rank**, not an independent market number.

- The "Underdog" source (`draftkit/data_sources/underdog_source.py`) never actually calls an Underdog API. `load_underdog_adp()` just reads the first existing file from:
  ```python
  UNDERDOG_SOURCE_CANDIDATES = [
      Path("data/raw/underdog_adp.csv"),
      Path("data/underdog_adp.csv"),
      Path("data/adp.csv"),
      Path("data/players.csv"),
  ]
  ```
  [draftkit/data_sources/underdog_source.py:27-32](draftkit/data_sources/underdog_source.py:27). None of the first three exist in this repo (confirmed via directory listing); it silently falls back to `data/players.csv`, a 27-row hand-authored sample file. In the current merge, FantasyPros already populates `adp` first for its 405 players, so this fallback mostly doesn't even get used — but the function name and module name (`underdog_source.py`, "Load Underdog ADP data") actively mislead anyone reading the code into thinking real market ADP is present.

**Net effect:** the app has no real, independent ADP/market benchmark anywhere in its live pipeline. What it calls "ADP" is FantasyPros' own rank column relabeled. Any "ADP value" or "market disagreement" score computed downstream (`calculate_adp_bonus`, `calculate_adp_delta`, `market_disagreement.py`) is comparing FantasyPros against a mislabeled copy of FantasyPros, not against the actual fantasy market. This is a target/benchmark mismatch at the data-integrity level, separate from and prior to any modeling problem.

---

## 2. Features / Input Variables

The full canonical schema, as defined in [draftkit/data_pipeline.py:13-35](draftkit/data_pipeline.py:13):

```python
CANONICAL_COLUMNS = [
    "player_id", "player_name", "position", "team", "bye_week", "age",
    "injury_status", "projection_points", "projection_rank", "position_rank",
    "tier", "ceiling_projection", "floor_projection", "adp", "adp_rank",
    "injury_risk", "durability_grade", "boom_score", "bust_score",
    "stability_score", "archetype",
]
```

Below is what each one actually is, as implemented, plus its real coverage in the current `data/processed/master_players.csv` (3,931 rows, measured directly):

| Feature | Formula / source as implemented | Season total or rate? | Coverage (non-null / 3,931) |
|---|---|---|---|
| `projection_points` | Raw value from `data/players.csv` fallback (see below); no real projections source is wired up | Full-season total | **27 (0.7%)** |
| `adp` / `adp_rank` | = FantasyPros `RK` column (expert rank), mislabeled as ADP — see Section 1 | N/A (rank) | 405 (10.3%) |
| `tier` | FantasyPros `TIERS` column, passed through unchanged | N/A | 405 (10.3%) |
| `position_rank` | Parsed digits from FantasyPros `POS` field (e.g. `"QB1"` → `1`) via `_parse_position_rank()`, [fantasypros_source.py:121-124](draftkit/data_sources/fantasypros_source.py:121) | N/A | tied to FantasyPros coverage |
| `ceiling_projection` / `floor_projection` | Canonical columns exist but are **never populated by any source** (Sleeper: `None`; FantasyPros: no matching column; Underdog fallback: `None`) | — | **0 (0%)** |
| `injury_risk` | If no direct column: bucketed from Sleeper `injury_status` text via `_normalize_injury_risk()` — `out/ir/pup/doubtful/nfi → 85`, `questionable/limited → 65`, `probable/active/healthy → 30`, else `50` ([sleeper_source.py:127-137](draftkit/data_sources/sleeper_source.py:127); mirrored in [feature_engineering.py:151-182](draftkit/feature_engineering.py:151)) | 4-bucket categorical dressed as a 0–100 score | 3,928 (99.9%), but **94% of all rows (3,693) are the single value 30.0** |
| `durability_grade` | Falls back to a **hardcoded constant `70.0`** when no games-played/missed-games data exists ([feature_engineering.py:132-148](draftkit/feature_engineering.py:132)); no games-played or missed-games column exists anywhere in the pipeline, so this is always the fallback | — | **27 (0.7%), and every single non-null value is exactly 70.0** (measured: `value_counts()` → `{70.0: 27}`) |
| `boom_score` | `(projection_percentile * 0.70) + (volatility_score * 0.30)` — [feature_engineering.py:224-245](draftkit/feature_engineering.py:224) | Derived from projection + a synthetic volatility score, not real ceiling outcomes | 27 (0.7%) |
| `bust_score` | `(injury_risk * 0.40) + (volatility_score * 0.35) + ((100 - durability_grade) * 0.25)` — [feature_engineering.py:248-266](draftkit/feature_engineering.py:248) | Derived entirely from other derived fields (see leakage note below) | 27 (0.7%) |
| `stability_score` | `(durability * 0.45) + ((100 - injury_risk) * 0.35) + ((100 - volatility) * 0.20)` — [feature_engineering.py:269-282](draftkit/feature_engineering.py:269) | Same as above | 27 (0.7%) |
| `archetype` | Rule cascade over boom/bust/stability/injury thresholds — [feature_engineering.py:285-305](draftkit/feature_engineering.py:285) | Categorical label, not a feature | 27 (0.7%) |
| `age` | Direct value or derived from Sleeper `birth_date` — [sleeper_source.py:99-124](draftkit/data_sources/sleeper_source.py:99) | Static | Sleeper-wide (not projection-gated) |

### The `data/players.csv` fallback is the entire predictive signal

`draftkit/data_pipeline.py` defines:
```python
LOCAL_PROJECTION_FALLBACK_PATH = Path("data/players.csv")
...
def _fill_projection_fallback(master_df):
    if "projection_points" in master_df.columns and master_df["projection_points"].notna().any():
        return master_df
    fallback_df = _load_local_projection_fallback()
    ...
```
[draftkit/data_pipeline.py:38, 310-349](draftkit/data_pipeline.py:38)

`data/players.csv` is a 27-row file with suspiciously round, hand-typed-looking numbers (Josh Allen 337.5 pts / ADP 24.0, Jalen Hurts 329.1 / 31.0, etc. — no decimals suggesting real-world source noise, and values don't reconcile with the FantasyPros ranks for the same players). **I cannot confirm whether this is a manually-entered placeholder or a genuine but stale export** — I did not find any generation script for it — but it is the *only* source of `projection_points`, `ceiling_projection`/`floor_projection` fallback, and (indirectly) `boom_score`/`bust_score`/`stability_score`/`archetype` for the entire 3,931-player pool. This should be verified with you directly before any rebuild.

### Season totals vs. per-game rates

**Everything is a season total or a static single-number field.** There is no per-game column anywhere in the schema, and no games-played denominator used to convert anything to a rate. `calculate_durability_grade()` does reference `games_played`/`missed_games` columns, but these columns don't exist in any actual data source (`GAMES_PLAYED_COLS`, `MISSED_GAMES_COLS` in [feature_engineering.py:15-16](draftkit/feature_engineering.py:15) never match anything in `master_players.csv`), so that branch is dead code in practice and the function always hits its default fallback.

### Recency weighting

**None exists, and none is possible with the current data.** The dataset has no weekly/game-log rows — it is a single point-in-time draft-prep table (`data/processed/master_players.csv`, one row per player, no season or week dimension). There is nothing to weight "recent" vs. "old" within this pipeline. (Recency-aware work does exist in `research/validation_v1/build_late_season_role_growth_score_v1.py`, but it is a standalone research script, not wired into `draftkit` or the app.)

### Leakage, redundancy, and miscalculation flags

- **Circular/self-referential features:** `bust_score` and `stability_score` are linear combinations of `injury_risk`, `durability_grade`, and `volatility_score` — but `volatility_score` itself is derived from `projection_points` percentile vs. `adp` ([feature_engineering.py:185-221](draftkit/feature_engineering.py:185)), and `durability_grade` is a flat constant. So three of the five "risk" features you'd expect to be independent signals are actually just re-weighted transformations of projection and ADP, which are already the two primary scoring inputs elsewhere. This inflates the appearance of a rich multi-factor model while adding close to zero independent information.
- **`market_disagreement.py` compares a source to itself:** `_resolve_columns()` sets `"fantasypros_rank": fantasypros_rank_col or adp_rank_col` ([draftkit/market_disagreement.py:61](draftkit/market_disagreement.py:61)). Since no literal `fantasypros_rank`/`RK` column survives into the merged dataset (it was renamed to `adp_rank` upstream), this always falls back to `adp_rank` — which, per Section 1, *is* the FantasyPros rank. So "market disagreement" is computing `adp_rank (FantasyPros RK) − projection_rank`, i.e., comparing FantasyPros's own rank against a rank derived from the 27-row fallback file. It is not an independent market signal.
- **Zero opportunity/usage features anywhere in the live app.** I grepped `draftkit/`, `utils.py`, and `scripts/` for target share, red zone usage, air yards, snap share, route participation, and Vegas/implied-total signals. The only hit is a dead column-name candidate, never populated:
  ```python
  "columns": ["workload_score", "workload", "touch_projection", "snap_share"],
  ```
  [draftkit/ranking_component_audit.py:29](draftkit/ranking_component_audit.py:29). No source ever writes to any of these columns.
- **QB scoring is flattened to a single constant.** In single-QB leagues, `apply_single_qb_value_adjustments()` caps every non-elite-tier QB's `final_score` at exactly `50.0`:
  ```python
  SINGLE_QB_SCORE_CAP = 50.0
  ...
  qb_caps = adjusted_df.loc[qb_mask, "position_pressure"].map(
      lambda level: 58.0 if str(level).upper() == "SEVERE"
      else 55.0 if str(level).upper() == "HIGH"
      else SINGLE_QB_SCORE_CAP
  )
  ```
  [draftkit/draft_analysis.py:45, 1217-1260](draftkit/draft_analysis.py:45). I confirmed this directly: Josh Allen (ADP rank 26), Jalen Hurts (ADP rank 55), Lamar Jackson (33), Joe Burrow (45), and C.J. Stroud (139) **all score exactly 50.00**, despite a 113-slot ADP spread between them (see Section 5 output). Any correlation-to-ADP measurement that includes QBs in a single-QB league context will be mechanically dragged down by this — it's not a projection problem, it's a hard floor/ceiling collapsing five very differently-valued players to one score.

---

## 3. Data Pipeline

**Raw sources** (all defined in `draftkit/data_sources/`):

1. **Sleeper** — live API call to `https://api.sleeper.app/v1/players/nfl` ([sleeper_source.py:9](draftkit/data_sources/sleeper_source.py:9)). Provides player ID/name/position/team/bye/age/injury_status metadata for the full NFL player universe. No projections, no ADP.
2. **FantasyPros** — local CSV, `data/raw/FantasyPros_2026_Draft_ALL_Rankings.csv` (456 rows, glob-matched via `*fantasypros*ranking*.csv` pattern in [fantasypros_source.py:27-33](draftkit/data_sources/fantasypros_source.py:27)). Provides rank, tier, position-rank, bye week. **No projection points and no real ADP**, as established in Section 1.
3. **"Underdog"** — not a real API integration; reads whichever of `data/raw/underdog_adp.csv`, `data/underdog_adp.csv`, `data/adp.csv`, `data/players.csv` exists first. Only `data/players.csv` (27 rows) exists in this repo.

**Ingestion entry point:** `draftkit/data_pipeline.py::merge_player_sources()` → `build_master_dataset()`, which name-normalizes (`normalize_player_name()`, strips suffixes/punctuation/diacritics) and merges the three sources by `player_id` then normalized name key, then writes `data/processed/master_players.csv`. This is what `draftkit.data_access.load_players_df()` reads at runtime (cached by file mtime, [data_access.py:196-259](draftkit/data_access.py:196)).

**Grain:** one row per player, season-level, single point-in-time snapshot. There is no weekly grain, no per-game log, and no season-over-season history anywhere in the app-facing pipeline. (Weekly-grain nflverse data does exist under `case_studies/data/*_player_seasons_*.csv`, but that's research-side, not part of `draftkit`'s runtime path.)

**Missing games / injuries / byes:**
- `bye_week` is carried through as metadata only; I found no code path where bye week affects any score (it's not in `PROJECTION_COLS`, not referenced in `draft_analysis.py` scoring functions).
- There is no games-played or missed-games column populated anywhere, so `calculate_durability_grade()`'s games-based branches are unreachable in practice (see Section 2) — injury-shortened seasons cannot distort per-game stats because **there are no per-game stats in this pipeline at all**; everything is a single season-total number, so a shortened season would only show up if the source projection itself accounted for it (unverifiable, since the 27-row fallback file's provenance is unknown).
- Injury status only affects `injury_risk` via the 4-bucket text heuristic in Section 2 — it does not modify `projection_points`, `ceiling_projection`, or `floor_projection`.

**Row-count sanity check (measured directly against `data/processed/master_players.csv`):**

| Population | Row count |
|---|---|
| Total merged rows (mostly bare Sleeper metadata) | 3,931 |
| Position breakdown | WR 1,744 / RB 896 / TE 826 / QB 465 |
| Rows with `adp`/`adp_rank` (i.e., FantasyPros-ranked players) | 405 |
| Rows with `tier` | 405 |
| Rows with `projection_points` | **27** |
| Rows with `ceiling_projection` or `floor_projection` | **0** |

So roughly 90% of the merged pool is Sleeper-only metadata rows with no draft-relevant signal at all, and even within the 405 "ranked" players, 378 (93%) have no projection.

---

## 4. Positional Handling

**Pooled, not separately trained/scored.** There is no per-position model, coefficient set, or independent scoring function — one scoring pipeline (`_build_base_recommendation_rankings_df()` → `build_master_recommendations_df()`) runs across QB/RB/WR/TE together, filtered only by `SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]` ([draft_analysis.py:25](draftkit/draft_analysis.py:25)).

**Partial, ad hoc positional adjustment exists in three places, all hardcoded constants, none derived from data:**

```python
POSITION_VALUE_MULTIPLIERS = {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 0.68}
POSITION_URGENCY_MULTIPLIERS = {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 0.70}
POSITION_NEED_WEIGHT_CAPS = {"TE": 1.18}
```
[draft_analysis.py:27-41](draftkit/draft_analysis.py:27)

- `position_value_score` (the projection component) *is* position-relative in one specific sense: `build_position_replacement_baselines()` computes a replacement-level projection per position from league roster settings and league size ([draft_analysis.py:887-914](draftkit/draft_analysis.py:887)), then scores `value_over_replacement * position_multiplier`. This is legitimate value-over-replacement logic — the one part of the pipeline that is meaningfully position-aware.
- Everything else — `injury_risk`, `durability_grade`, `adp_bonus`, `team_fit_bonus`, `tier_bonus` — is computed on an absolute 0–100 (or ratio) scale with **no position-specific normalization**. A TE's `injury_risk` and a QB's `injury_risk` are computed with the identical bucket thresholds and directly compared/combined.
- Championship Equity V2 similarly pools all positions through one `TeamOutcomeSimulator` with position-specific constants (`POSITION_REPLACEMENT_POINTS`, `POSITION_SCARCITY_MULTIPLIERS` in [championship_equity_v2.py:52-64](draftkit/championship_equity_v2.py:52)) rather than separate models.

Net: the "positional handling" is a handful of manually-tuned multipliers layered onto a single pooled scoring function, not genuine per-position modeling or within-position feature normalization (e.g., no z-scoring of projection within position group before combining across positions).

---

## 5. Model / Scoring Logic

**No ML model of any kind runs in the live app.** It's a deterministic, hand-weighted multiplicative/additive formula. Core chain in `draftkit/draft_analysis.py`:

1. `position_value_score` = value-over-replacement × position multiplier ([calculate_position_value_score, line 917](draftkit/draft_analysis.py:917))
2. `need_bonus` = roster-need weight (0.2–1.5) from `get_position_need_weights()`
3. `adp_bonus` = `1.0 + clip(-0.10, 0.15, (adp − projection_rank)/100)` ([calculate_adp_bonus, line 638](draftkit/draft_analysis.py:638))
4. `tier_bonus` = 1.0, or 1.06–1.10 if the player's tier is "thin"/"shrinking"
5. `team_fit_bonus` = 0.80–1.20 multiplier from archetype/roster-volatility matching
6. Base score: `final_score = position_value_score × need_bonus × adp_bonus × tier_bonus × team_fit_bonus` ([calculate_recommendation_score, line 867](draftkit/draft_analysis.py:867))
7. This base score is then **discarded and replaced** by a second, differently-weighted formula: each of five components is normalized to 0–100 (`normalize_component_scores()`, line 1069) and recombined via `DEFAULT_MASTER_COMPONENT_WEIGHTS` (30/25/15/15/15 split, shown in Section 1) inside `calculate_final_recommendation_score()` (line 1102). **The step-6 multiplicative score is computed but never used** as the final output — worth flagging as either dead code or an unintentional duplicate scoring path.
8. Post-processing: construction-pressure multiplier, signal-trust damping (caps score at 58–68 if data-quality flags fire), then the single-QB cap described in Section 2.

No coefficients here were fit to data — every weight, multiplier, and cap is a literal constant chosen by hand.

### Existing validation / backtest scripts

- **`draftkit/model_evaluator.py`** and **`draftkit/ranking_component_audit.py`** compute Pearson correlation, calibration error, and rank-accuracy — but only between the app's own internal signals (e.g., `consensus_score` vs. `final_draft_grade`, or a feature vs. `final_score`) from **simulated Draft Lab runs**, never against real historical player outcomes or real ADP. This is self-referential: it validates internal consistency of the heuristic, not predictive accuracy.
- **`scripts/run_runtime_profile.py`** exists but measures wall-clock performance, not accuracy (it's the basis of the existing `performance_report.md`/`performance_audit.md` in this repo, which are about latency, not correlation).
- **The one genuine backtest against real historical ADP lives outside the app**, in `research/validation_v1/` (Fantasy Football Calculator archived ADP, 2010–2024, 1,834 rows, merged against real season-finish outcomes from 1999–2025 nflverse data). It evaluates different research signals (age-curve edge score, market disagreement score, late-season role growth score) — **not** the live `draftkit` engine. Its dated conclusion (`research_validation_v1_report.md`, 2026-07-07):

  | Target | Avg model hit rate | Avg ADP hit rate | Avg lift over ADP | Avg AUC | ADP AUC |
  |---|---:|---:|---:|---:|---:|
  | `WR_Top24` | 0.531 | 0.565 | **-0.034** | 0.769 | 0.749 |
  | `WR_Top12` | 0.342 | 0.348 | **-0.006** | 0.785 | 0.795 |
  | `RB_Top24` | 0.656 | 0.677 | **-0.022** | 0.769 | 0.770 |
  | `RB_Top12` | 0.441 | 0.455 | **-0.014** | 0.742 | 0.799 |

  Its own conclusion: *"the current WR/RB models do not beat ADP... ADP remains the better main guide."* Every research signal tested has negative lift vs. ADP; none is classified above "Not Useful for app integration." This confirms that even the most rigorously-tested piece of this project has never beaten the market it's being compared to — which is a useful, sobering baseline for expectations on any rebuild.

### Actual current correlation number (measured live, this session)

Since no script in the repo directly measures the live app's correlation to its own ADP field, I ran the actual engine (`draftkit.draft_analysis.build_recommendation_rankings_df()`) against the current `data/processed/master_players.csv`, using a Streamlit-shim harness modeled on the existing `scripts/run_runtime_profile.py` approach:

```
Raw master_players.csv rows: 3931
Rows with non-null projection_points: 27
Rows with non-null adp: 405

build_recommendation_rankings_df() output rows: 27   <- entire ranking pool

Rows usable for correlation (final_score & adp both present): 27
Pearson correlation (final_score vs adp):  -0.3880
Spearman correlation (final_score vs adp): -0.4801
```

Two findings from this run:

1. **The recommendation engine only ever outputs 27 players**, because `_build_base_recommendation_rankings_df()` does `df = df.dropna(subset=[proj_col])` ([draft_analysis.py:954-955](draftkit/draft_analysis.py:954)) — and the same drop happens in `_prep_ranked_df()` (line 220-221) and `build_adp_value_rankings_df()` (line 740). Since only 27 of 3,931 rows have `projection_points`, **378 of the 405 players with real FantasyPros-derived rank data — including Jahmyr Gibbs (rank 2), Ashton Jeanty, Malik Nabers, Brock Bowers, Chase Brown, and dozens of other clearly-relevant draft picks — are silently excluded from every recommendation table the app produces.** This is not a modeling weakness; it's a pipeline defect that makes the "model" blind to ~93% of the players it has rank/tier data for.
2. Even restricted to the 27 players it *does* rank, `final_score` vs. `adp` (both directions oriented so "better" is higher-score/lower-adp) shows only a moderate Spearman ρ = -0.48. A ranking system that mostly agreed with consensus would show something much closer to -0.9 to -1.0 in this small, top-heavy sample. Part of this gap is mechanical: all 5 QBs in the sample are hard-capped to `final_score = 50.0` regardless of their ADP spread (26 to 139), which alone suppresses the correlation (see Section 2).

---

## 6. Gaps and Risks

**Referenced but never populated:**

- `ceiling_projection`, `floor_projection` — canonical columns, 0% coverage, never written by any source.
- Snap share, target share, red zone usage, air yards, route participation, Vegas/implied point totals, team pass-attempt volume — none exist as columns anywhere in `draftkit` or `data/`; only one dead reference (`ranking_component_audit.py` line 29, never populated).
- `games_played`/`missed_games` — referenced in `feature_engineering.py` fallback logic but never present in any actual data source, so that logic path is dead in practice.
- Real market ADP — referenced conceptually everywhere but not actually present anywhere in the live pipeline (Section 1).

**Hardcoded / suspicious values:**

- `durability_grade` = flat `70.0` constant for all 27 rows that have it at all ([feature_engineering.py:148](draftkit/feature_engineering.py:148)).
- `injury_risk` = `30.0` for 3,693 of 3,931 rows (94%) — effectively a constant, providing almost no discriminative signal despite looking like a real per-player score.
- `SINGLE_QB_SCORE_CAP = 50.0` collapses all non-elite QBs to an identical score (Section 2/5).
- `data/players.csv` — 27 rows of suspiciously round, unverified projection/ADP numbers that are the sole source of `projection_points` for the entire pool. **Flagging explicitly rather than guessing:** I could not find a generation script or documented source for this file; it may be intentional placeholder/test data left over from early development, but as written it is silently powering production output.
- Legacy `scripts/rank_players.py` computes `value_score = projection_points - adp * 10` directly in SQL — an arbitrary, unnormalized formula with no statistical grounding, kept in the repo though apparently disconnected from the current Streamlit pages (worth confirming it's fully dead).

**Dead / duplicate logic worth flagging:**

- `calculate_recommendation_score()` (the multiplicative score, step 6 in Section 5) is fully computed per-row but its result is immediately overwritten by the separately-weighted `calculate_final_recommendation_score()` — two different scoring formulas exist in the same function, only one of which is load-bearing.
- `draftkit/championship_equity.py` (V1) still exists alongside `championship_equity_v2.py`; per the existing `research_audit_report.md`, V1 appears superseded but not removed.
- `utils.py` contains an older independent "Best Fit/Best Value/Best Available" scoring layer still imported by `pages/5_Player_Compare.py`, parallel to and inconsistent with the `draftkit` engine used everywhere else (flagged previously in `performance_audit.md`, not yet resolved as of this audit).
- No `TODO`/`FIXME` comments were found in `draftkit/*.py`, but the code's own docstring in `calculate_context_score()` — *"Version 1 intentionally stays simple so future ADP, tier, scarcity, or ML weighting can replace this"* ([draft_analysis.py:369-375](draftkit/draft_analysis.py:369)) — is a direct admission that this is a placeholder formula, not a finished model, and it still ships in the live scoring path via `build_context_rankings_df()`.

---

## Most Likely Causes of Weak Correlation to ADP/Expert Consensus, Ranked

1. **Coverage collapse in the ranking pool.** The live engine drops every player without `projection_points`, leaving only 27 of 3,931 players (and only 27 of 405 players with real rank/tier data). Any correlation you compute against ADP is being measured on a tiny, non-representative, top-heavy sample — the other 93% of ranked players simply never appear in a recommendation table to be compared. This is very likely the single largest contributor and should be fixed before anything else.
2. **"ADP" is not ADP.** The benchmark you're correlating against is FantasyPros' own expert-rank column, mislabeled through the pipeline as `adp`/`adp_rank`. This doesn't just weaken correlation — it makes the reported correlation number itself borderline meaningless as an ADP comparison, since there's no independent market signal in the system to correlate against.
3. **The one real predictive input (`projection_points`) traces to a 27-row file of unknown, unverified provenance** rather than a real projections feed — so even for the 27 players that do get ranked, the core signal quality is unverified.
4. **Several "independent" risk/upside features are circular re-derivations of projection and ADP**, not real signal (`volatility_score`, `boom_score`, `bust_score`, `stability_score`), which dilutes the model's effective feature count without adding information — and can actively work against ADP alignment when they disagree with it for reasons that aren't grounded in real data.
5. **No recency weighting and no per-game rates are possible**, because the pipeline has no weekly grain at all — this isn't a design flaw in the scoring formula so much as a data-pipeline limitation that forecloses an entire class of standard fantasy-projection technique (workload trend, opportunity-based projection).
6. **Zero opportunity-based features** (target share, snap share, red zone usage, Vegas lines) are wired into the app, despite candidate columns existing in the code — so the model has no lever to disagree with ADP in a way that's likely to be *right*, only in ways driven by the hardcoded multipliers/caps above (e.g., the flat TE discount, the single-QB cap).
7. **Independent, dated evidence (`research/validation_v1/research_validation_v1_report.md`) shows that even a properly-validated version of similar signal ideas, tested against real historical ADP across 2010–2024, does not beat ADP** — so before rebuilding, it's worth deciding whether the goal is to *match* consensus more closely (fix the pipeline/labeling bugs above) or to *beat* it (which the project's own most rigorous research to date has not yet achieved for any tested signal).
