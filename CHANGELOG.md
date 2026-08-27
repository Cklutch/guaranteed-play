# Changelog

Notable removals and structural changes not obvious from a diff alone. Not a full history — see `git log` for that.

## 2026-08-26

- Removed `pages_archive/5_Player_Compare.py` — superseded by the Draft Command Center's built-in player search/compare panel; was the last file still importing the legacy root `utils.py`.
- Removed `pages_archive/7_Live_Rankings.py` — functionally superseded by `Home.py`, kept around only as dead weight.
- Removed `utils.py` — the legacy root-level module, fully orphaned once the two pages above were gone.
- Removed `pages_archive/8_Tier_Desperation.py`, `pages_archive/2_Player_Cards.py`, `pages_archive/5_Draft_Lab.py` — three more shelved pages, not needed and not being revived.
- Removed `draftkit/recommendation_explainer.py` and `draftkit/model_evaluator.py` — orphaned once Player Cards and Draft Lab were gone (nothing else imported either).
- Kept (do not delete despite being paired with the pages above): `draftkit/conviction.py`, `draft_lab.py`, `draft_simulator.py`, `draft_simulation.py`, `opponent_model.py`, `draft_strategy.py` — all still imported by `scripts/run_runtime_profile.py` (a real runtime-profiling dev tool), and `draft_simulation.py` additionally by the still-shelved `pages_archive/9_Component_Audit.py`.
- Trimmed `app.py`'s page list (Player Cards, Tier Desperation) to match. Note: `app.py` is itself a dead, unused alternate Streamlit entry point — 4 of its remaining 6 page paths already pointed at `pages/` instead of `pages_archive/` before this change and still do; the real nav is `Home.py`'s `TOOL_PAGES` dropdown (`draftkit/ui_helpers.py`). Worth deleting `app.py` outright in a future pass.
- Revived `pages_archive/9_Component_Audit.py` as `pages/3_Component_Audit.py` — wired into the `TOOL_PAGES` nav dropdown, verified end to end against live data (recommendations, championship equity, availability simulation, and the audit build itself all run cleanly). Unlike the other live pages it keeps its sidebar (`render_league_settings_sidebar()`) rather than hiding it — real, load-bearing UI here, not leftover chrome; only the auto page-list is suppressed.
