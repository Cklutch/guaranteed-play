"""turn_optimizer regression tests (2026-08-28).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_turn_optimizer

Built on a small synthetic pool rather than the real board so the assertions
pin BEHAVIOR (the survival math, the need correction, the roster caps) and
don't drift every time a projection is edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from draftkit.turn_optimizer import (  # noqa: E402
    MAX_TURN_GAP,
    MIN_LANDING_ODDS,
    _survival_array,
    expected_best_available,
    find_pick_cluster,
    optimize_turn,
    survival_between,
    upcoming_picks,
)


def _pool():
    """Synthetic 12-player board: 4 RB, 4 WR, 2 TE, 2 QB, descending value."""
    rows = []
    spec = [
        ("RB", 4, 90.0), ("WR", 4, 88.0), ("TE", 2, 70.0), ("QB", 2, 65.0),
    ]
    adp = 1
    for pos, count, top in spec:
        for i in range(count):
            rows.append({
                "player_name": f"{pos}{i + 1}",
                "position": pos,
                "team": f"T{adp}",
                "bye_week": 7,
                "our_score": top - i * 5.0,
                "injury_risk": 20.0,
                "overall_risk": 20.0,
                "position_tier": 1,
                "adp": float(adp),
            })
            adp += 1
    return pd.DataFrame(rows)


def test_back_to_back_survival_is_certain():
    """THE core invariant. Between two of MY consecutive picks no opponent
    selects, so the second player cannot be sniped -- survival must be
    exactly 1.0, not the ~76% the raw unconditional curve reports."""
    assert survival_between(5, 12, 0) == 1.0
    assert survival_between(200, 12, 0) == 1.0
    assert survival_between(None, 12, 0) == 1.0


def test_survival_is_conditional_and_monotonic():
    """More opponent picks -> strictly less likely to survive, and an
    already-passed-ADP player is scored on what's LEFT, not on the stale
    unconditional number."""
    probs = [survival_between(14, 11, n) for n in (0, 1, 2, 4, 10)]
    assert probs == sorted(probs, reverse=True), probs
    assert probs[0] == 1.0
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Conditioning matters: a player well past his ADP who is somehow still
    # here is far likelier to last one more pick than the raw curve implies.
    assert survival_between(5, 12, 1) > 0.5

    # Unknown ADP is genuinely uncertain, not a confident yes or no.
    assert survival_between(None, 11, 3) == 0.5


def test_vectorized_survival_matches_scalar():
    """_survival_array() is a performance shortcut for the pair loop; it must
    not become a second, subtly different model."""
    adps = [1.0, 14.0, 55.0, None, 200.0]
    for gap in (0, 2, 4):
        vec = _survival_array(adps, 11, gap)
        for got, adp in zip(vec, adps):
            assert abs(float(got) - survival_between(adp, 11, gap)) < 1e-9, (adp, gap)


def test_expected_best_available_math():
    """Exact expectation under independence: walk best-first, weighting each
    by P(it survives) x P(everything better is gone)."""
    # 10 at 50%, 8 guaranteed -> 10*.5 + 8*1*(1-.5) = 9.0
    assert abs(expected_best_available([10.0, 8.0], [0.5, 1.0]) - 9.0) < 1e-9
    # A guaranteed best option makes everything below it irrelevant.
    assert abs(expected_best_available([10.0, 8.0], [1.0, 1.0]) - 10.0) < 1e-9
    assert expected_best_available([], []) == 0.0


def test_cluster_detection_matches_snake_seats():
    """In a 12-team league exactly the seats the user named are turn seats:
    1/12 back-to-back, 2/11 and 3/10 near-turn, everything else single."""
    kinds = {slot: find_pick_cluster(12, slot, 1)["kind"] for slot in range(1, 13)}
    assert kinds[12] == "turn", kinds
    assert kinds[11] == "near_turn" and kinds[10] == "near_turn", kinds
    assert all(kinds[s] == "single" for s in range(4, 10)), kinds

    # Slot 1's back-to-back is at 24/25, not at pick 1 -- it only becomes a
    # turn once the draft snakes around.
    assert find_pick_cluster(12, 1, 1)["kind"] == "single"
    turn = find_pick_cluster(12, 1, 24)
    assert turn["kind"] == "turn" and (turn["first_pick"], turn["second_pick"]) == (24, 25)

    # Slots 2 and 3 likewise wrap into near-turns.
    assert find_pick_cluster(12, 2, 23)["gap"] == 2
    assert find_pick_cluster(12, 3, 22)["gap"] == 4
    assert find_pick_cluster(12, 3, 22)["gap"] <= MAX_TURN_GAP


def test_upcoming_picks_snake_order():
    assert upcoming_picks(12, 12, 1)[:4] == [12, 13, 36, 37]
    assert upcoming_picks(12, 1, 1)[:3] == [1, 24, 25]
    assert upcoming_picks(12, 0, 1) == []      # slot not tracked
    assert upcoming_picks(12, 99, 1) == []     # nonsense slot


def test_need_bonus_not_double_counted():
    """Running the single-pick engine twice hands the SAME positional need
    bonus to both halves of a same-position pair. With one RB already
    rostered, a second RB fills the last starter slot and a third would
    saturate -- so the pair's second RB must be corrected downward."""
    pool = _pool()
    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=["RB1"], teams=12, slot=12, current_pick=12,
    )
    assert not plans.empty
    rb_rb = plans[plans["shape"] == "RB/RB"]
    if not rb_rb.empty:
        assert (rb_rb["need_delta"] < 0).all(), rb_rb[["first", "second", "need_delta"]]

    # With an EMPTY roster the RB target is 2, so taking one RB does not
    # remove the need for another -- no correction should apply.
    plans_empty, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=12,
    )
    rb_rb_empty = plans_empty[plans_empty["shape"] == "RB/RB"]
    if not rb_rb_empty.empty:
        assert (rb_rb_empty["need_delta"] == 0).all()


def test_roster_caps_block_duplicate_capped_positions():
    """1-QB / 1-TE leagues: a pair may never be two QBs or two TEs."""
    pool = _pool()
    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=12,
        max_plans=99,
    )
    assert "QB/QB" not in set(plans["shape"])
    assert "TE/TE" not in set(plans["shape"])


def test_true_turn_pairs_are_guaranteed():
    """At a real back-to-back, every recommended pair lands with certainty --
    that is the whole reason the turn is worth optimizing as a pair."""
    pool = _pool()
    plans, ctx = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=12,
    )
    assert ctx["cluster"]["gap"] == 0
    assert (plans["land_pct"] == 100).all()


def test_near_turn_reports_real_landing_odds():
    """With opponents picking in between, the second player is a gamble and
    the plan must say so rather than implying a guarantee."""
    pool = _pool()
    plans, ctx = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=11, current_pick=11,
    )
    assert ctx["cluster"]["gap"] == 2
    assert not plans.empty
    assert (plans["land_pct"] <= 100).all()
    assert (plans["land_pct"] > 0).all()


def test_non_turn_slot_declines_cleanly():
    """A mid-round seat has no pair to lock in -- say so, don't invent one."""
    pool = _pool()
    plans, ctx = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=6, current_pick=6,
    )
    assert plans.empty
    assert ctx["cluster"]["kind"] == "single"
    assert ctx["reason"]


def test_hides_until_the_turn_is_in_view():
    """Naming a specific pair 11 picks out is theatre -- every plan converges
    on the fallback and the ranking degenerates into noise (observed live:
    top four plans within 0.3 points, recommending 8%-to-be-there players).
    The panel must decline rather than guess."""
    pool = _pool()
    plans, ctx = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=1,
    )
    assert plans.empty
    assert ctx["opponents_until_turn"] == 11
    assert "picks away" in ctx["reason"]

    # Once it's close, it engages.
    near, ctx_near = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=8,
    )
    assert not near.empty
    assert ctx_near["opponents_until_turn"] == 4


def test_only_realistically_available_players_are_offered():
    """A name you have almost no chance of getting is a wish, not a plan."""
    pool = _pool()
    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=8,
        max_plans=99,
    )
    assert not plans.empty
    floor = int(MIN_LANDING_ODDS * 100)
    assert (plans["first_land_pct"] >= floor).all(), plans["first_land_pct"].tolist()
    assert (plans["land_pct"] >= floor).all(), plans["land_pct"].tolist()


def test_same_team_pair_is_penalised():
    """Real miss found live 2026-08-28: the top pair was Parker Washington +
    Bhayshul Tuten, both Jacksonville -- shared bye, shared offense, and a RB
    and WR splitting the same touches. The old bye penalty missed it entirely
    because it only fired for two players at the SAME position."""
    pool = _pool()
    # Put the best RB and the best WR on one team.
    pool.loc[pool["player_name"] == "RB1", "team"] = "JAX"
    pool.loc[pool["player_name"] == "WR1", "team"] = "JAX"

    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=12,
        max_plans=99,
    )
    same = plans[(plans["first"].isin(["RB1", "WR1"])) & (plans["second"].isin(["RB1", "WR1"]))]
    assert not same.empty
    assert (same["stack_penalty"] < 0).all(), same[["first", "second", "stack_penalty"]]
    assert bool(same.iloc[0]["same_team"])

    # Everyone else is untouched.
    others = plans[~plans.index.isin(same.index)]
    assert (others["stack_penalty"] == 0).all()


def test_qb_receiver_stack_is_not_penalised():
    """A QB paired with his own pass-catcher is correlated in the direction
    you WANT -- a real strategy, not the mistake above."""
    pool = _pool()
    pool.loc[pool["player_name"] == "QB1", "team"] = "CIN"
    pool.loc[pool["player_name"] == "WR1", "team"] = "CIN"

    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=12, current_pick=12,
        max_plans=99,
    )
    stack = plans[(plans["first"].isin(["QB1", "WR1"])) & (plans["second"].isin(["QB1", "WR1"]))]
    if not stack.empty:
        assert (stack["stack_penalty"] == 0).all(), stack[["first", "second", "stack_penalty"]]


def test_one_row_per_unordered_pair():
    """Both orderings are evaluated so the better one wins on merit, but the
    output must not list the same two players twice."""
    pool = _pool()
    plans, _ = optimize_turn(
        pool, drafted_players=[], my_team=[], teams=12, slot=11, current_pick=11,
        max_plans=99,
    )
    keys = plans.apply(lambda r: tuple(sorted([r["first"], r["second"]])), axis=1)
    assert len(keys) == len(set(keys))


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} turn_optimizer tests passed.")


if __name__ == "__main__":
    _run()
