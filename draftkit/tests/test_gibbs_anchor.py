"""Gibbs #1 anchor regression check (plan_addendum_gibbs_anchor.pdf,
Option B -- regression/sanity-check anchor, the plan's stated default).

This is deliberately NOT a hard override on final_score or rank. It is a
check on the OUTCOME of the real scoring pipeline (position_value_score,
normalize_component_scores(), calculate_final_recommendation_score()),
run against Jahmyr Gibbs's real, current, healthy inputs (337.2 proj pts,
ADP 1.1, RB1 consensus) -- the plan's own explicit rationale for rejecting
Option A: a hard clamp that ignores real signal is the exact class of bug
this whole session spent fixing (position_value_score's hard clamp,
normalize_component_scores()'s linear-range flattening, the 2-decimal
precision bottleneck). Forcing Gibbs to #1 unconditionally would keep him
there even if his real situation changed (injury, role loss, suspension,
holdout) -- actively wrong, and the kind of silent special case a human
has to remember to remove.

If this assertion ever fails after a future scoring-pipeline change: that
is a signal to investigate WHY the change moved him (did it unfairly
penalize elite-tier RBs? compress the top of the distribution the
tier-calibration fix worked to protect?) -- not a signal to hard-code the
ranking back. If Gibbs's real inputs change (injury, depth chart, etc.),
this test should be updated or removed by a human who has confirmed that
change is real, not silently bypassed.

Real fixture, captured 2026-08-17 (today's real board, post hard-clamp
fix, post log-transform normalization fix, post 6-decimal precision fix):
    position_value_score            166.5
    projection_component_score      100.0
    adp_value_component_score       65.27
    tier_urgency_component_score    70.5
    team_fit_component_score        50.0
    position_need_component_score   100.0
    final_score                     84.53
    Rank                            1

Runnable directly:
    python -m draftkit.tests.test_gibbs_anchor
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.draft_state import init_session_state  # noqa: E402
from draftkit.draft_analysis import build_recommendation_rankings_df  # noqa: E402

GIBBS_NAME = "Jahmyr Gibbs"


def _current_board():
    init_session_state()
    df = build_recommendation_rankings_df()
    has_adp = pd.to_numeric(df.get("adp"), errors="coerce").notna()
    has_real_proj = df.get("projection_source", pd.Series(dtype=object)) == "real"
    board = df[has_adp | has_real_proj].copy()
    board = board.sort_values("final_score", ascending=False).reset_index(drop=True)
    board["Rank"] = range(1, len(board) + 1)
    return board


def test_gibbs_ranks_first_under_current_real_inputs():
    board = _current_board()
    matches = board[board["player_name"] == GIBBS_NAME]
    assert not matches.empty, f"expected {GIBBS_NAME} in the current board, found none"
    row = matches.iloc[0]

    assert row["Rank"] == 1, (
        f"Gibbs's real, current inputs (proj={row['projection_points']}, adp={row.get('adp')}) "
        f"no longer rank him #1 (got Rank={row['Rank']}, final_score={row['final_score']}). "
        f"Per plan_addendum_gibbs_anchor.pdf: this is a signal to investigate WHY the most "
        f"recent scoring-pipeline change moved him -- NOT a signal to hard-code the ranking "
        f"back. Check whether the change unfairly penalized elite-tier RBs or compressed the "
        f"top of the distribution the tier-calibration fix protected."
    )
    print(
        f"PASS -- Gibbs ranks #1 under current real inputs: proj={row['projection_points']:.1f}, "
        f"adp={row.get('adp')}, position_value_score={row['position_value_score']}, "
        f"projection_component_score={row['projection_component_score']}, "
        f"final_score={row['final_score']:.2f}"
    )


def main() -> int:
    test_gibbs_ranks_first_under_current_real_inputs()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
