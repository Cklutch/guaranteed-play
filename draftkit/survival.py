"""Survival modelling -- will this player still be on the board?

Three models live here, in increasing order of what they know. They are the
same curve decomposed three ways, so they can never disagree by accident:

  1. `raw_survival(adp, pick)` -- the unconditional logistic (k=0.30) the
     board's survival gauge has always used. "Across many drafts, how often
     is this player still there at pick N."

  2. `survival_between(adp, from_pick, opponent_picks)` -- CONDITIONAL on the
     player being available right now, and measured in OPPONENT selections
     rather than raw pick numbers. Between your own back-to-back picks nobody
     else chooses, so this returns exactly 1.0 where the unconditional curve
     would wrongly shave ~20% off. This is the identity
     S(from + m) / S(from), i.e. the telescoping product of the per-pick
     hazards below.

  3. `survival_with_needs(...)` -- the same hazard product, but each
     intervening pick is attributed to the team actually on the clock for it,
     and its hazard is scaled by whether that team still NEEDS the position.
     With no roster information it reduces EXACTLY to (2) -- see
     test_needs_model_reduces_to_adp_only -- so this is a strict
     generalization, not a competing model.

WHY (3) EXISTS. Pure-ADP survival assumes every team drafts the consensus
board. Real drafts don't work that way, and the gap is at its widest exactly
when it matters most -- deciding between two players at your pick. The case
that motivated it (user, 2026-08-28, pick 5.11): Davante Adams was NOT the
best player on their board, but the team picking next had three RBs and a QB
and no WRs. ADP said Adams might last; roster logic said he was certain to be
taken. Leaving him meant losing him. A model that can't see the opponent's
roster cannot represent that, and will keep advising "take the better player,
he'll come back" right up until he doesn't.

Deliberately free of any dependency on live_draft/draft_center so both can
import it (turn_optimizer already imports from draft_center, so anything
shared has to sit below both). Roster targets/caps stay with their owners and
reach this module through the `mult_for` callable that make_need_multiplier()
builds.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

# Logistic steepness. Matches live_draft.SURVIVAL_K and the survival gauge in
# draft_center -- one curve, three decompositions.
SURVIVAL_K = 0.30

# How much a team's roster need moves their odds of taking a given player.
# These scale a per-pick hazard, so they compound across several picks: a WR
# facing four WR-hungry teams before your next pick decays far faster than
# ADP alone implies, which is the whole point.
NEED_MULT_HUNGRY = 1.8    # starter slot at that position still open
NEED_MULT_FULL = 0.40     # starters filled -- he'd be depth, much less likely
NEED_MULT_CAPPED = 0.05   # 1-QB/1-TE league and they already have theirs
NEED_MULT_UNKNOWN = 1.0   # no roster info for this team -> behave like ADP


def raw_survival(adp, pick, k=SURVIVAL_K):
    """Unconditional P(available at `pick`). None for an unpriced player."""
    if adp is None or pd.isna(adp):
        return None
    # A player 250+ picks past ADP overflows math.exp() otherwise, and this
    # runs across the whole pool.
    x = max(-60.0, min(60.0, -k * (float(pick) - float(adp))))
    return 1.0 - 1.0 / (1.0 + math.exp(x))


def pick_hazard(adp, pick, k=SURVIVAL_K):
    """P(taken AT `pick` | still available going into it), from ADP alone.

    The building block both conditional models are products of:
        h(p) = 1 - S(p+1)/S(p)
    """
    s_now = raw_survival(adp, pick, k)
    s_next = raw_survival(adp, pick + 1, k)
    if s_now is None or s_next is None or s_now <= 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - s_next / s_now))


def survival_between(adp, from_pick, opponent_picks, k=SURVIVAL_K):
    """P(survives `opponent_picks` more OPPONENT selections | available now).

    Exactly 1.0 when no opponent picks intervene -- the back-to-back turn.
    Unknown ADP returns 0.5: genuinely uncertain, and both confident answers
    would be wrong in opposite directions.
    """
    if opponent_picks is None or opponent_picks <= 0:
        return 1.0
    if adp is None or pd.isna(adp):
        return 0.5

    s_from = raw_survival(adp, from_pick, k)
    s_to = raw_survival(adp, from_pick + opponent_picks, k)
    if s_from is None or s_to is None or s_from <= 1e-9:
        return 0.0
    return float(min(1.0, max(0.0, s_to / s_from)))


def survival_array(adp_values, from_pick, opponent_picks, k=SURVIVAL_K):
    """Vectorized survival_between() over a column of ADPs -- same formula,
    run once per horizon instead of once per candidate per pair."""
    adp = pd.to_numeric(pd.Series(list(adp_values)), errors="coerce").to_numpy(dtype=float)
    if opponent_picks is None or opponent_picks <= 0:
        return np.ones(len(adp))

    x_from = np.clip(-k * (from_pick - adp), -60.0, 60.0)
    x_to = np.clip(-k * (from_pick + opponent_picks - adp), -60.0, 60.0)
    s_from = 1.0 - 1.0 / (1.0 + np.exp(x_from))
    s_to = 1.0 - 1.0 / (1.0 + np.exp(x_to))

    out = np.divide(s_to, s_from, out=np.zeros_like(s_to), where=s_from > 1e-9)
    out = np.clip(out, 0.0, 1.0)
    out[np.isnan(adp)] = 0.5
    return out


def team_on_clock(pick, teams):
    """Snake-draft seat picking at `pick` (1-indexed both ways)."""
    rnd = math.ceil(pick / teams)
    idx = (pick - 1) % teams
    return idx + 1 if rnd % 2 == 1 else teams - idx


def rosters_by_slot(drafted_by, position_of):
    """{slot: {POS: count}} from Sleeper's {player_name: slot} map.

    Manual mode has no per-team attribution, so this comes back empty and
    every downstream multiplier falls through to NEED_MULT_UNKNOWN -- the
    model degrades to plain ADP rather than guessing.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for name, slot in (drafted_by or {}).items():
        pos = position_of.get(name)
        if pos:
            counts[int(slot)][str(pos).upper()] += 1
    return {slot: dict(pos_counts) for slot, pos_counts in counts.items()}


def make_need_multiplier(drafted_by, position_of, teams, my_slot,
                         starter_targets, roster_caps):
    """Build `mult_for(pick, position) -> float` for survival_with_needs().

    Returns None when there's nothing to go on (no per-team pick attribution),
    which callers pass straight through to get plain ADP behaviour.

    `starter_targets` and `roster_caps` are injected rather than imported so
    this module stays free of live_draft (which imports it).
    """
    rosters = rosters_by_slot(drafted_by, position_of)
    if not rosters:
        return None

    def mult_for(pick, position):
        slot = team_on_clock(pick, teams)
        # My own pick can't take a player away from me.
        if my_slot and slot == my_slot:
            return 0.0
        counts = rosters.get(slot)
        if counts is None:
            return NEED_MULT_UNKNOWN
        pos = str(position).upper()
        filled = counts.get(pos, 0)
        cap = roster_caps.get(pos)
        if cap is not None and filled >= cap:
            return NEED_MULT_CAPPED
        if filled < starter_targets.get(pos, 0):
            return NEED_MULT_HUNGRY
        return NEED_MULT_FULL

    return mult_for


def survival_with_needs(adp, position, from_pick, opponent_picks,
                        mult_for=None, k=SURVIVAL_K):
    """survival_between(), but each intervening pick is scaled by whether the
    team on the clock for it actually needs this position.

    Step j uses the ADP curve at index `from_pick + j` (which is what makes
    the no-info case telescope back to exactly S(from+m)/S(from)) and is
    attributed to the real pick `from_pick + 1 + j`, the next selection after
    mine. With `mult_for=None` this IS survival_between -- verified by test,
    so the two can never quietly diverge.
    """
    if opponent_picks is None or opponent_picks <= 0:
        return 1.0
    if adp is None or pd.isna(adp):
        return 0.5
    if mult_for is None:
        return survival_between(adp, from_pick, opponent_picks, k)

    surv = 1.0
    for j in range(int(opponent_picks)):
        hazard = pick_hazard(adp, from_pick + j, k)
        hazard = min(1.0, hazard * float(mult_for(from_pick + 1 + j, position)))
        surv *= (1.0 - hazard)
        if surv <= 1e-9:
            return 0.0
    return float(min(1.0, max(0.0, surv)))
