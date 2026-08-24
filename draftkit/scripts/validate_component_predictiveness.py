"""Real component-predictiveness validation (concept from a user-supplied
validate_component_predictiveness.py, extended to use every real player this
repo actually has a confirmed outcome for, not just the ~0-2 the naive
version's join would find).

The naive version joins data/rookie_inputs.csv (the CURRENT 2026 board)
against rb_archetypes.csv/wr_archetypes.csv's confirmed outcomes -- but 2026
rookies have zero real NFL snaps yet by definition, so essentially none of
them have a confirmed archetype (that join finds Bijan/Gibbs only because
they happen to still be present from an earlier, since-superseded version of
that file; the current data/raw/rookie_inputs.csv is 2026-only). The real
calibration set already exists elsewhere in this repo:
data/raw/backtest_rookie_inputs.csv (built this session by
build_backtest_rookie_inputs.py for the draft-capital calibration work) is
exactly "real 2023-2025 draftees with real pre-draft inputs," 116 players,
and build_rookie_projections() already joins every one of them to
rb_archetypes.csv/wr_archetypes.csv's real confirmed outcomes. This script
reuses that machinery (same real components, same real confirmed join, same
real data run_rookie_backtest.py's accuracy report already depends on)
instead of re-deriving a parallel, smaller pipeline.

Correlates the COMPONENT SCORES (component_draft_capital, component_dominator,
etc. -- the actual 0-100 values the composite formula weights), not the raw
combine/draft-pick inputs -- this directly answers the question the concept
was after: does each component's current composite weight match its real
predictive power against a real confirmed outcome? Every one of these
component scores is already oriented "higher = better" by
draftkit/rookie_projection.py's own design (draft_capital_score,
dominator_score, athletic_score_rb, competition_score, and
team_context_quality's inversion fix all do this already), so no manual
sign-flip is needed here the way the naive draft_pick version needed one.

Only status=="confirmed" rows are used as ground truth (a player with real
games clearing the sample floor) -- "blended" rows are excluded from ground
truth on purpose: for sample_weight<0.5, blend_rookie_tag() returns the
PROJECTED tier as tag, which would correlate the model's own projection
against itself, not against a real outcome.

Usage:
    python -m draftkit.scripts.validate_component_predictiveness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_backtest_rookie_inputs import OUTPUT_CSV as BACKTEST_INPUTS_CSV  # noqa: E402
from draftkit.scripts.build_rookie_projections import build_rookie_projections  # noqa: E402

# Real, position-specific ordinal ranking of each position's own real
# confirmed archetype vocabulary (rb_archetypes.py/wr_archetypes.py) -- the
# ground truth every component gets checked against. "unconfirmed" is a
# real, meaningful outcome (cleared the real sample floor but never cleared
# any real usage tier), ranked at the bottom rather than excluded --
# dropping it would throw away real information about which components
# predict a player never developing a role, not just which tier they land in.
# WR: "possession" replaced by boom_bust/high_floor
# (claude_code_plan_possession_split.pdf) -- both real, co-equal
# replacements for the same real rank (neither is "better" than the
# other, just a different profile shape), so both take Possession's old
# rank=2 rather than one winning it.
WR_OUTCOME_RANK = {"alpha": 3, "boom_bust": 2, "high_floor": 2, "complementary": 1, "unconfirmed": 0}
# RB: receiving_back/goal_line_specialist/explosive_back removed from the
# real confirmed vocabulary entirely (explicit user correction, 2026-08-15
# -- primary is a usage tier only, those became secondary lean tags, see
# rb_archetypes.py). Handcuff moves from a tied rank=0 (lumped with
# Unconfirmed under the old 6-way scale) to its own rank=1 -- it's a real,
# decisive usage read (clear backup to an identified starter), genuinely
# more informative than "no role identified," and deserves its own step
# on a cleaner 4-level scale now that it's one of only 4 real outcomes.
RB_OUTCOME_RANK = {"bellcow": 3, "committee_back": 2, "handcuff": 1, "unconfirmed": 0}

# Real composite weights currently live in draftkit/rookie_projection.py's
# wr_rookie_projection()/rb_rookie_projection() -- printed next to each
# component's real correlation so the two can be compared directly, which
# is the entire point of this script.
WR_WEIGHTS = {
    "component_draft_capital": 0.35, "component_dominator": 0.25, "component_breakout_age": 0.10,
    "component_roster_competition": 0.15, "component_offense_environment": 0.10, "component_schedule": 0.05,
}
RB_WEIGHTS = {
    "component_draft_capital": 0.30, "component_dominator": 0.15, "component_athletic": 0.15,
    "component_roster_competition": 0.20, "component_offense_environment": 0.15, "component_schedule": 0.05,
}

MIN_SAMPLE_FOR_TRUST = 5


def _correlation(xs: np.ndarray, ys: np.ndarray) -> float | None:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _position_report(board: pd.DataFrame, position: str, outcome_rank: dict, weights: dict) -> pd.DataFrame:
    confirmed = board[(board["position"] == position) & (board["status"] == "confirmed")].copy()
    blended_n = len(board[(board["position"] == position) & (board["status"] == "blended")])
    confirmed["outcome_rank"] = confirmed["tag"].map(outcome_rank)
    pool = confirmed.dropna(subset=["outcome_rank"])

    print(f"\n{position}: {len(pool)} real confirmed players used as ground truth "
          f"(status=='confirmed' only; excludes {blended_n} real 'blended' players -- "
          f"see module docstring for why).")
    if len(pool) < MIN_SAMPLE_FOR_TRUST:
        print(f"  WARNING: n={len(pool)} is below the {MIN_SAMPLE_FOR_TRUST}-player trust floor -- "
              f"treat every correlation below as directional, not conclusive.")

    print(f"  {'component':30s} {'current weight':>15s} {'real correlation':>18s} {'n':>5s}")
    for component, weight in weights.items():
        if component not in pool.columns:
            continue
        xs = pd.to_numeric(pool[component], errors="coerce")
        ys = pool["outcome_rank"].astype(float)
        valid = xs.notna()
        n = int(valid.sum())
        r = _correlation(xs[valid].to_numpy(), ys[valid].to_numpy())
        r_str = f"{r:+.3f}" if r is not None else ("no real variance yet" if n >= 3 else "insufficient data")
        print(f"  {component:30s} {weight:>14.0%} {r_str:>18s} {n:>5d}")
    return pool


def _sourced_vs_placeholder_report(pool: pd.DataFrame, component: str, raw_column: str, label: str) -> None:
    """Splits a real confirmed pool by whether COMPONENT's underlying RAW
    input (raw_column, from backtest_rookie_inputs.csv) is a real sourced
    value or still null -- checked against the raw input's own nullness,
    NOT against the derived component score's value, since a real sourced
    value can legitimately land near the neutral default by coincidence
    (e.g. a real dominator share of ~22.7% scores component_dominator=50.0,
    identical to the "no data" neutral default). First built for
    rookie_data_backfill_plan.pdf's WR-dominator follow-up question (does a
    correlation hold, disappear, or reverse once placeholder rows are
    excluded); reused here for breakout_age per rookie_model_next_plan.pdf's
    Tier 0 item 2, which flagged the same likely contamination.

    raw_column is read directly off `pool` -- build_rookie_projections()'s
    output now carries the raw prospect fields itself (added for Home.py's
    Prospect Profile section), so no separate re-merge from
    BACKTEST_INPUTS_CSV is needed (an earlier version of this function did
    that merge manually and it silently collided/suffixed once the board
    started carrying the same column)."""
    sourced = pool[pool[raw_column].notna()]
    placeholder = pool[pool[raw_column].isna()]

    print(f"\n{label} split -- sourced (real {raw_column} on file) vs. placeholder "
          f"(neutral default, no real data sourced yet):")
    for split_label, subset in (("sourced", sourced), ("placeholder", placeholder)):
        xs = pd.to_numeric(subset[component], errors="coerce")
        ys = subset["outcome_rank"].astype(float)
        valid = xs.notna()
        n = int(valid.sum())
        r = _correlation(xs[valid].to_numpy(), ys[valid].to_numpy())
        r_str = f"{r:+.3f}" if r is not None else ("no real variance" if n >= 3 else "insufficient data")
        print(f"  {split_label:12s} n={n:<4d} real correlation={r_str}")


def main() -> int:
    board = build_rookie_projections(inputs_csv=BACKTEST_INPUTS_CSV)
    print(f"{len(board)} real backtest players available (2023-2025 draft classes, "
          f"draftkit/scripts/build_backtest_rookie_inputs.py's real 116-player scope).")

    wr_pool = _position_report(board, "WR", WR_OUTCOME_RANK, WR_WEIGHTS)
    _position_report(board, "RB", RB_OUTCOME_RANK, RB_WEIGHTS)
    _sourced_vs_placeholder_report(wr_pool, "component_dominator", "college_dominator_final_year", "WR dominator")
    _sourced_vs_placeholder_report(wr_pool, "component_breakout_age", "breakout_age", "WR breakout_age")

    print("\nRead this as directional evidence, not a final reweighting. draft_pick's own "
          "predictive power is already handled by calibrate_draft_capital.py's real "
          "positional k fit -- this is the same real-data-first approach extended to the "
          "OTHER components (dominator/athletic/breakout_age/roster_competition/team "
          "context) that calibration never touched. Re-run as the 72-candidate backfill "
          "list (find_backfill_candidates.py) gets more real combine/season-stats data "
          "sourced -- component_dominator and component_athletic's correlations are "
          "currently diluted by real, documented data gaps (no rb_season_stats_2024.csv "
          "exists yet; the 2023 class's dominator is entirely blocked; see "
          "report_backfill_data_coverage.py), not by the components themselves being "
          "unpredictive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
