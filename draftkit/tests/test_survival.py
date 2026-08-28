"""survival model tests (2026-08-28).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_survival
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.survival import (  # noqa: E402
    NEED_MULT_CAPPED,
    make_need_multiplier,
    pick_hazard,
    raw_survival,
    rosters_by_slot,
    survival_array,
    survival_between,
    survival_with_needs,
    team_on_clock,
)

STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
CAPS = {"QB": 1, "TE": 1}


def test_hazards_telescope_to_survival_between():
    """survival_between IS the product of per-pick hazards. If these ever
    disagree the module has quietly grown a second model."""
    for adp in (5.0, 22.0, 60.0):
        for m in (1, 2, 4, 9):
            prod = 1.0
            for j in range(m):
                prod *= (1.0 - pick_hazard(adp, 11 + j))
            assert abs(prod - survival_between(adp, 11, m)) < 1e-9, (adp, m)


def test_needs_model_reduces_to_adp_only():
    """THE contract that keeps this one model instead of two: with no roster
    information, the need-aware path must return exactly the ADP answer."""
    for adp in (3.0, 18.0, 44.0, 120.0):
        for m in (0, 1, 3, 8):
            assert abs(
                survival_with_needs(adp, "WR", 12, m, None) - survival_between(adp, 12, m)
            ) < 1e-9, (adp, m)


def test_back_to_back_still_certain_under_needs():
    """No opponent picks in between -> nobody can take him, whatever the
    opposing rosters look like."""
    mult = make_need_multiplier(
        {"X": 1}, {"X": "WR"}, 12, 12, STARTERS, CAPS,
    )
    assert survival_with_needs(5.0, "WR", 12, 0, mult) == 1.0


def test_opponent_need_moves_the_odds():
    """The case this exists for: a team with no WRs is far likelier to take
    the WR than ADP alone says, and a team that already has its one QB is far
    less likely to take a QB."""
    teams, my_slot = 12, 12
    # Slot 11 picks next (pick 14 in a 12-team snake round 2). Give them three
    # RBs and a QB and no WR -- the user's real 5.11 situation.
    drafted_by = {"r1": 11, "r2": 11, "r3": 11, "q1": 11}
    position_of = {"r1": "RB", "r2": "RB", "r3": "RB", "q1": "QB"}
    mult = make_need_multiplier(drafted_by, position_of, teams, my_slot, STARTERS, CAPS)

    adp = 20.0
    wr_needs = survival_with_needs(adp, "WR", 12, 3, mult)
    plain = survival_between(adp, 12, 3)
    assert wr_needs < plain, (wr_needs, plain)

    # Same player, same picks, but a position that team is done with.
    qb_needs = survival_with_needs(adp, "QB", 12, 3, mult)
    assert qb_needs > plain, (qb_needs, plain)


def test_my_own_picks_cannot_take_from_me():
    """A pick belonging to my slot is not a threat -- multiplier 0."""
    mult = make_need_multiplier({"x": 3}, {"x": "RB"}, 12, 1, STARTERS, CAPS)
    # Pick 1 in a 12-team snake belongs to slot 1, which is me.
    assert mult(1, "RB") == 0.0
    assert mult(2, "RB") != 0.0


def test_capped_position_is_nearly_no_threat():
    mult = make_need_multiplier({"q": 5}, {"q": "QB"}, 12, 12, STARTERS, CAPS)
    assert mult(5, "QB") == NEED_MULT_CAPPED


def test_rosters_by_slot_and_manual_mode_degrades():
    got = rosters_by_slot({"a": 3, "b": 3, "c": 7}, {"a": "RB", "b": "wr", "c": "TE"})
    assert got == {3: {"RB": 1, "WR": 1}, 7: {"TE": 1}}
    # Manual mode has no attribution -> no multiplier, plain ADP behaviour.
    assert make_need_multiplier({}, {}, 12, 4, STARTERS, CAPS) is None


def test_snake_seat_math():
    assert [team_on_clock(p, 12) for p in (1, 12, 13, 24, 25)] == [1, 12, 12, 1, 1]


def test_vectorized_matches_scalar():
    adps = [1.0, 14.0, 55.0, None, 200.0]
    for gap in (0, 2, 5):
        for got, adp in zip(survival_array(adps, 11, gap), adps):
            assert abs(float(got) - survival_between(adp, 11, gap)) < 1e-9


def test_raw_survival_is_unconditional_and_differs():
    """Kept deliberately distinct from the conditional model -- the board's
    gauge wants the unconditional number. Guard against them being merged."""
    assert raw_survival(None, 10) is None
    # A player 7 picks past ADP: unconditional is grim, conditional-on-still-
    # being-here for a single pick is not.
    assert raw_survival(5, 12) < 0.2
    assert survival_between(5, 12, 1) > 0.5


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} survival tests passed.")


if __name__ == "__main__":
    _run()
