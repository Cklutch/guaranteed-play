"""Opportunity-cost / replaceability regression tests (2026-08-28).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_opportunity_cost

Pins the substitutes-vs-complements split in recommend_picks(). Both halves
are real bugs that were shipped and caught live, so both directions are
asserted -- fixing one by breaking the other is the failure mode here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from draftkit.live_draft import recommend_picks  # noqa: E402


def _pool(rows):
    """rows: (name, pos, our_score, adp)"""
    return pd.DataFrame([
        {
            "player_name": n, "position": p, "team": f"T{i}", "bye_week": 7,
            "our_score": s, "injury_risk": 0.0, "overall_risk": 0.0,
            "position_tier": 1, "adp": float(a),
        }
        for i, (n, p, s, a) in enumerate(rows)
    ])


def _score(df, name, col="draft_score"):
    return float(df.loc[df["player_name"] == name, col].iloc[0])


# One slot (QB): a low ADP means "goes soon", a high ADP means "will last".
_QB_CASE = [
    ("Lawrence", "QB", 40.4, 97.0),   # better, and likely to last
    ("Herbert", "QB", 35.8, 80.0),    # worse, and likely gone
    ("FillerRB", "RB", 30.0, 200.0),
    ("FillerWR", "WR", 30.0, 201.0),
]

# Several slots (WR): same shape, but you want three of them.
_WR_CASE = [
    ("Addison", "WR", 42.5, 101.0),   # better, 97% to return
    ("Metcalf", "WR", 41.1, 78.0),    # slightly worse, 7% to return
    ("FillerWR3", "WR", 20.0, 300.0),
    ("FillerRB", "RB", 30.0, 200.0),
]


def test_substitute_position_never_prefers_the_worse_player():
    """THE reported bug (2026-08-28): the board offered Justin Herbert (35.8,
    5% to return) over Trevor Lawrence (40.4, the #1 QB left, 89% to return).
    You roster one QB, so they are substitutes -- Herbert's scarcity buys
    nothing when a BETTER QB is nearly certain to be sitting there next pick.
    """
    df = recommend_picks(
        _pool(_QB_CASE), drafted_players=[], my_team=[], top_n=50,
        current_round=7, current_pick=78, my_next_pick=90,
    )
    assert _score(df, "Lawrence") > _score(df, "Herbert"), df[
        ["player_name", "our_score", "replaceable", "survival_pts", "draft_score"]
    ]
    # Lawrence is the best QB left, so nobody can replace him -- he keeps his
    # own survival. Herbert inherits Lawrence's, which is the whole fix.
    assert _score(df, "Herbert", "replaceable") > _score(df, "Lawrence", "replaceable")


def test_replaceability_floors_at_the_players_own_survival():
    """A player is always replaceable BY HIMSELF. An earlier attempt used a
    'k better players survive' tail that lost this floor and returned 0.0 for
    the best player at a needed position -- which zeroed the penalty on
    Jordan Addison at 97% to return and ranked him 2nd overall."""
    df = recommend_picks(
        _pool(_WR_CASE), drafted_players=[], my_team=[], top_n=50,
        current_round=7, current_pick=78, my_next_pick=90,
    )
    # Addison is ~certain to return; his penalty must reflect that, not be 0.
    assert _score(df, "Addison", "replaceable") > 0.5, df[
        ["player_name", "replaceable", "survival_pts", "draft_score"]
    ]


def test_multi_slot_position_still_takes_the_scarce_player_first():
    """With three WR slots open they are COMPLEMENTS, not substitutes: take
    the scarce one now and the survivor later and you get BOTH. A better WR
    surviving must NOT make a scarce one pointless."""
    df = recommend_picks(
        _pool(_WR_CASE), drafted_players=[], my_team=[], top_n=50,
        current_round=7, current_pick=78, my_next_pick=90,
    )
    assert _score(df, "Metcalf") > _score(df, "Addison"), df[
        ["player_name", "our_score", "replaceable", "survival_pts", "draft_score"]
    ]


def test_position_switches_to_substitute_logic_as_it_fills():
    """Once only one WR slot is left, the substitute rule should take over --
    a better WR being there really does make a scarcer lesser one a waste."""
    pool = _pool(_WR_CASE)
    # Roster two WRs so only one starter slot remains.
    pool = pd.concat([pool, _pool([("MineA", "WR", 10.0, 400.0), ("MineB", "WR", 10.0, 401.0)])])
    df = recommend_picks(
        pool, drafted_players=[], my_team=["MineA", "MineB"], top_n=50,
        current_round=7, current_pick=78, my_next_pick=90,
    )
    # Metcalf now inherits Addison's near-certain return.
    assert _score(df, "Metcalf", "replaceable") > _score(df, "Addison", "replaceable"), df[
        ["player_name", "replaceable", "draft_score"]
    ]


def test_no_next_pick_means_no_opportunity_cost():
    """With nowhere to defer to, nothing is replaceable -- the term is off.
    turn_optimizer relies on this (it passes my_next_pick=None so it can do
    the survival maths itself at the pair level)."""
    df = recommend_picks(
        _pool(_QB_CASE), drafted_players=[], my_team=[], top_n=50,
        current_round=7, current_pick=78, my_next_pick=None,
    )
    assert (df["survival_pts"] == 0).all()
    assert _score(df, "Lawrence") > _score(df, "Herbert")   # pure value order


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} opportunity-cost tests passed.")


if __name__ == "__main__":
    _run()
