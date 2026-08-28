# Guaranteed Play — Fantasy Football Draft Kit

Streamlit app for 2026 fantasy football draft prep: player rankings, tiers, and
draft-pick recommendations built from a custom valuation engine (not raw
consensus rankings).

## Running the app

```bash
.venv/Scripts/streamlit.exe run Home.py --server.headless true
```

Or via the Claude Code preview tools: `preview_start` with `{name: "guaranteed-play"}`
(config already in `.claude/launch.json`, port 8501).

First-time setup on a new machine:
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

## Architecture

- **`Home.py`** — Streamlit UI: rankings table rendering, CSS, tier bands, cell
  formatting (`_render_cell`), position filter chips, search, sort links,
  hover popovers (risk scorecard, archetype scorecard). All server-rendered
  HTML via `st.markdown(unsafe_allow_html=True)` — no JS, no native Streamlit
  dataframe widget.
- **`draftkit/`** — core logic package:
  - `draft_analysis.py` — Base Value scoring engine (VOR + projection +
    market + risk components), `build_master_recommendations_df()`
  - `draft_state.py` — session state init
  - `projection_engine.py`, `signal_trust.py` — projection blending/trust logic
  - `risk_scoring.py`, `risk_constraints.py`, `risk_weights.json` — risk model
  - `qb_archetypes.py`, `rb_archetypes.py`, `wr_archetypes.py`, `te_archetypes.py`
    — per-position usage archetype classification
  - `rookie_projection.py`, `injury_history.py`
  - `live_draft.py` — in-draft pick engine (`recommend_picks()`, one player
    at a time); `draft_center.py` — all Draft Command Center HTML rendering
  - `survival.py` — "will he still be there?" model, shared by the pick
    engine and the turn optimizer. See Gotchas.
  - `turn_optimizer.py` — best two-player COMBINATION for back-to-back picks
    (slots 1/12, plus near-turn 2/3 and 10/11). Pure logic; rendered by
    `draft_center.render_turn_planner()`. See Gotchas on survival.
  - `scripts/` — data pipeline build scripts (props pulls, combine measurables,
    draft capital calibration, backtests)
  - `tests/` — pytest coverage for archetypes, risk, rookie projections
- **`research/validation_v1/`** — research pipeline: feature engineering,
  model validation, and `build_live_projections_v1.py` (holds
  `MODEL_PROJECTION_CORRECTIONS`, manual per-player projection overrides —
  see Gotchas below)
- **`data/raw/`**, **`data/processed/`** — pipeline inputs/outputs
- **`pages/`** — additional Streamlit pages (Draft Mode, etc.)

## Base Value scoring system

`final_score` in the rankings board = weighted sum of four components,
computed in `draft_analysis.py`:

- **VOR** (value-over-replacement) — weight 1.0 — cross-position scarcity
  from actual league format (QB12/RB34/WR46/TE17 replacement ranks)
- **Projection component** — max bonus 20.0 — within-position percentile
  rank (not raw points, avoids QB scoring-scale bias)
- **Market swing component** — max 12.0 — sportsbook vs. ADP signal
- **Risk penalty component** — max 15.0 — subtracted

These four constants were sensitivity-tested (see
`research/validation_v1/data/base_value_sensitivity_2026-08-21.csv`) — the
board is stable (Spearman ρ > 0.999) across ±25% variation in any single
weight. Don't re-tune them without re-running that check.

## Tier system

`_assign_automatic_tiers()` in `Home.py` places tier breaks at the 9 largest
gaps in `final_score` across the whole ranked board, producing exactly
`TARGET_TIER_COUNT = 10` tiers. This is self-calibrating — it adapts to
whatever the score distribution looks like after any future projection
change, so don't replace it with a fixed point-gap threshold (that was the
bug that produced 19+ fragmented tiers before this redesign).

## Gotchas

- **`MODEL_PROJECTION_CORRECTIONS` is a plain dict literal** in
  `build_live_projections_v1.py` — if a player ID appears twice, the later
  entry silently wins. Sequencing matters when doing pool redistributions
  (e.g. docking one player and crediting others).
- **Streamlit caching**: after editing `draft_analysis.py` or the projection
  correction data, a stale cache can serve old scores. If values don't
  update after a code change, restart the Streamlit process rather than
  just reloading the page. This applies to `turn_optimizer.py` too — it is
  imported *inside* `render_turn_planner()` (to avoid a cycle with
  `draft_center`), so Streamlit's file watcher never reloads it and edits
  need a full restart.
- **Only ever run the app from this checkout.** There is a second, stale
  clone at `C:\Users\19054\guaranteed-play` (frozen at 2026-08-24) with the
  same git remote. A server started from there serves old projections while
  every edit lands here, which reads exactly like a caching bug and is not
  one. If numbers won't update, check the serving process's working
  directory before anything else — and check for more than one server, since
  Streamlit silently takes the next free port when 8501 is busy.
- **`draftkit/survival.py` owns the survival model — three views of ONE
  curve, and they must never diverge.** (1) `raw_survival(adp, pick)` is
  UNCONDITIONAL — right for the board's survival gauge only. (2)
  `survival_between(adp, from_pick, opponent_picks)` is CONDITIONAL on the
  player being available now and counts only OPPONENT picks; between your own
  picks 12 and 13 nobody else selects, so it returns exactly 1.0 where the
  unconditional curve wrongly reports ~8%. (3) `survival_with_needs(...)`
  scales each intervening pick by whether the team on the clock still needs
  that position. (3) reduces to (2) exactly when there's no roster info, and
  (2) is the telescoping product of (3)'s per-pick hazards — both pinned by
  `test_survival.py`. If you add a fourth, prove it reduces to these.
- **Opponent-need survival needs Sleeper.** `drafted_by` ({player_name:
  slot}) only exists in Sleeper mode; Manual mode tracks *that* a player was
  drafted, not *by whom*, so the model degrades to plain ADP rather than
  guessing. `context["needs_aware"]` / the absence of a multiplier tells you
  which regime you're in.
- **Two players on the same NFL team is worse than their scores suggest** —
  shared bye, shared offense, and a RB/WR pair splits the same touches. The
  per-player pass cannot see it (neither is on your roster yet), so
  `turn_optimizer.SAME_TEAM_PENALTY` handles it at the pair level. QB +
  pass-catcher is exempt: that stack is correlated the way you want.
- **Secrets are not in git.** `.github_token` (optional, raises the nflverse
  GitHub API rate limit via `NFLVERSE_GITHUB_TOKEN`/`GITHUB_TOKEN`) and
  `.odds_api.token` (required for `draftkit/scripts/pull_sportsbook_props.py`,
  via `THE_ODDS_API_KEY`) must be copied to the repo root manually on each
  machine — they're gitignored (`*.token`, `.github_token`).
- **No requirements.txt was tracked historically** — it was generated from
  the working venv on 2026-08-24. If you add a new dependency, regenerate it:
  `.venv/Scripts/pip freeze > requirements.txt`.
