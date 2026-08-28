"""Turn-pick optimizer -- the best COMBINATION for back-to-back picks.

At the turn (slot 1 or 12 in a 12-team league) you pick twice in a row with
NOBODY picking in between, then wait ~22 picks for your next pair. That makes
the right question "which PAIR maximizes value" rather than "who is the best
player right now," and the two answers genuinely differ:

  * Running the single-pick engine twice double-counts positional need. It
    hands the same +6 RB need bonus to both halves of an RB/RB pair even
    though taking the first one fills that hole -- so greedy-twice
    systematically over-rates doubling up. This module recomputes the second
    player's need term against the roster you'd have AFTER the first pick.

  * The single-pick engine's opportunity-cost term (survival_pts in
    live_draft.py) is actively wrong inside a turn. It penalizes a player for
    being likely to "return at your next pick" -- but at a true turn your next
    pick is the very next selection, so nothing gets sniped and there is no
    opportunity cost to trade off at all. We call recommend_picks() with
    my_next_pick=None to switch that term off, and model survival properly
    here instead.

WHY A SEPARATE SURVIVAL MODEL (the mathematical core of this module):

draft_center.survival_pct(adp, pick) is an UNCONDITIONAL curve indexed by raw
pick number. It answers "across many drafts, what fraction of the time is this
player still on the board at pick N." That is the wrong quantity here, twice
over:

  1. It ignores what you already know. If a player with ADP 5 is somehow still
     sitting there at pick 12, the unconditional 11% is stale -- you have
     observed him surviving. What matters is the CONDITIONAL probability given
     he is available right now.

  2. It counts your own picks as chances to lose him. Between your pick 12 and
     your pick 13 no opponent selects, yet the raw curve still advances one
     pick and quietly shaves ~24% off. Applied to a back-to-back pair that is
     nonsense: the second player is guaranteed.

survival_between() fixes both by conditioning and by measuring elapsed time in
OPPONENT picks rather than pick numbers:

        P(survives | available now) = S(from + opponents) / S(from)

which collapses to exactly 1.0 when opponents == 0 -- the true turn -- and
degrades smoothly for the near-turn slots. This is what lets one code path
serve both cases without special-casing the turn.

SCOPE: slots 1/12 are the real target (0 opponent picks between). Slots 2/3
and 10/11 get 2 and 4 opponent picks respectively and are handled by the same
math, just with landing probabilities below 1.0 -- which is also why those
slots get an explicit "take this one FIRST" ordering, something a true turn
never needs.

Pure functions over DataFrames + plain draft state, no Streamlit, so this is
unit-testable the same way live_draft.py is.
"""

from __future__ import annotations

import itertools
import math

import pandas as pd

from draftkit.draft_center import my_pick_numbers
from draftkit.live_draft import (
    DEFAULT_WEIGHTS,
    NEED_BONUS,
    NEED_BONUS_OVERRIDES,
    ROSTER_CAPS,
    SATURATION_PENALTY,
    STARTER_TARGETS,
    recommend_picks,
)
# The survival model lives in its own module so live_draft and this one can
# both use it (turn_optimizer already imports draft_center, so anything shared
# has to sit below both). Re-exported here because this module's callers and
# tests have always reached for survival_between through it.
from draftkit.survival import (  # noqa: F401
    SURVIVAL_K,
    make_need_multiplier,
    survival_array as _survival_array,
    survival_between,
    survival_with_needs,
)

# How many opponent picks may sit between my two picks and still count as a
# "turn" worth optimizing as a pair. 4 is not arbitrary -- in a 12-team league
# it selects exactly the slots the user cares about and no others:
#   slot 1/12 -> 0 opponents (true turn)
#   slot 2/11 -> 2 opponents
#   slot 3/10 -> 4 opponents
#   slot 4/9  -> 16/6 opponents  (excluded)
# Slots 5-8 are mid-round seats whose picks are far enough apart that pair
# optimization has nothing to add over the normal one-at-a-time board.
MAX_TURN_GAP = 4

# Weight on the next-turn lookahead term. Deliberately secondary: it answers
# "what does this pair leave me at my NEXT pair" and is built on a forecast of
# other teams' behavior, which is much softer evidence than the value of two
# players you can actually see on the board right now. Large enough to break
# ties between otherwise-equal pairs (which is the real job -- RB/RB vs RB/WR
# when both score within a point or two), too small to override a genuine
# value gap.
NEXT_TURN_WEIGHT = 0.35

# Candidate pool for pair enumeration, per side. Cheap enough to run on every
# rerun of a live draft page.
PAIR_TOP_K = 16

# Don't plan a specific pair from further out than this many opponent picks.
# See the guard in optimize_turn() for the failure this prevents.
MAX_PLANNING_HORIZON = 6

# A pair is only advice if you can actually get it. Below this landing
# probability a name is a wish, not a plan, and it is dropped rather than
# ranked. 0.25 is deliberately permissive -- a one-in-four shot at a stud is
# still worth planning around; a one-in-twelve shot is not.
MIN_LANDING_ODDS = 0.25

# Two non-QB players from the SAME NFL team is a worse pair than their
# individual scores suggest, and nothing in the per-player pass can see it
# (neither is on your roster yet when that runs). Real miss found live
# 2026-08-28: the planner's top pair was Parker Washington + Bhayshul Tuten,
# both Jacksonville. Three separate problems compound there --
#   * they share a bye, so your lineup takes both holes in the same week;
#   * they share one offense, so a bad season for that offense busts both
#     picks together (the correlation you do NOT want on two early picks);
#   * a RB and a WR on the same team split the same pool of touches and
#     targets, so they cap each other's ceiling.
# The old bye penalty missed this entirely because it only fired when both
# players were the SAME position.
#
# QB + pass-catcher is the deliberate exception: that stack is correlated in
# the direction you want (the QB's good games ARE the receiver's good games),
# which is a real strategy rather than a mistake, so it is left alone.
SAME_TEAM_PENALTY = 7.0
_STACK_OK = ({"QB", "WR"}, {"QB", "TE"})

# Rounds of snake picks to project when locating clusters.
PROJECTED_ROUNDS = 16


def _entries(names, values, survivals):
    """(name, value, survival) sorted best-value-first, NaN values dropped --
    the precomputed shape the hot loop walks instead of re-filtering a frame."""
    rows = [
        (n, float(v), float(s))
        for n, v, s in zip(names, values, survivals)
        if pd.notna(v)
    ]
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows


def _expected_best_from_entries(entries, exclude=()):
    """expected_best_available() over a precomputed entry list, skipping the
    one or two players this pair already took. Breaks out once it is
    near-certain something above has survived, so it touches only the top
    handful of candidates rather than the whole pool."""
    expected = 0.0
    none_yet = 1.0
    for name, value, surv in entries:
        if name in exclude:
            continue
        expected += value * surv * none_yet
        none_yet *= (1.0 - surv)
        if none_yet <= 1e-6:
            break
    return expected


def upcoming_picks(teams, slot, current_pick, rounds=PROJECTED_ROUNDS):
    """My remaining pick numbers, current pick included if it is mine."""
    if not teams or not slot or slot < 1 or slot > teams:
        return []
    return [p for p in my_pick_numbers(teams, slot, rounds) if p >= (current_pick or 1)]


def find_pick_cluster(teams, slot, current_pick, max_gap=MAX_TURN_GAP,
                      rounds=PROJECTED_ROUNDS):
    """Describe my next two picks and whether they form a turn.

    Returns None when the seat has no upcoming pair to optimize. `gap` is the
    number of OPPONENT picks between them -- 0 is a true back-to-back turn.
    """
    picks = upcoming_picks(teams, slot, current_pick, rounds)
    if len(picks) < 2:
        return None

    first, second = picks[0], picks[1]
    gap = second - first - 1
    # The pick after the pair -- where the next-turn lookahead lands.
    following = picks[2] if len(picks) > 2 else None

    if gap == 0:
        kind, label = "turn", "True turn -- back-to-back picks"
    elif gap <= max_gap:
        kind = "near_turn"
        label = f"Near turn -- {gap} opponent pick{'s' if gap != 1 else ''} in between"
    else:
        kind, label = "single", f"Single pick -- {gap} opponent picks until your next"

    return {
        "first_pick": first,
        "second_pick": second,
        "following_pick": following,
        "gap": gap,
        "kind": kind,
        "label": label,
        "is_turn": kind in ("turn", "near_turn"),
        "round": math.ceil(first / teams) if teams else None,
    }


def _need_pts(pos, have):
    """live_draft.recommend_picks()'s positional need term, re-derived here so
    the SECOND pick of a pair can be scored against the roster the first pick
    creates. Must stay in sync with that function's need_pts()."""
    filled = have.get(pos, 0)
    target = STARTER_TARGETS.get(pos, 0)
    if filled < target:
        return NEED_BONUS_OVERRIDES.get(pos, NEED_BONUS)
    return -SATURATION_PENALTY * (filled - target + 1)


def expected_best_available(values, survivals):
    """Expected value of the best player still on the board, given each
    player's independent survival probability.

    Walks candidates best-first and accumulates value x P(this one survives)
    x P(everyone better is gone). This is the exact expectation under
    independence, and it is the right shape for the lookahead: it naturally
    rewards positions with DEPTH (several near-equal options, so something
    good almost certainly survives) over positions that are one-deep behind a
    cliff -- which is exactly the "if you take two RBs now, what WR is left at
    36?" question this term exists to answer.
    """
    pairs = sorted(
        ((v, s) for v, s in zip(values, survivals) if pd.notna(v)),
        key=lambda t: t[0], reverse=True,
    )
    expected = 0.0
    none_yet = 1.0
    for value, surv in pairs:
        surv = float(min(1.0, max(0.0, surv)))
        expected += float(value) * surv * none_yet
        none_yet *= (1.0 - surv)
        if none_yet <= 1e-6:
            break
    return expected


def _next_turn_outlook(entries_by_pos, exclude_names, have_after):
    """What the board is expected to hand me at my NEXT pair, given the
    positions this pair leaves unfilled.

    Only unfilled starter slots count -- depth at a position you've already
    filled is not what makes a pair good or bad here. `entries_by_pos` is
    precomputed against the next turn's pick horizon.
    """
    needed = [p for p, target in STARTER_TARGETS.items() if have_after.get(p, 0) < target]
    if not needed or not entries_by_pos:
        return 0.0, {}

    by_pos = {
        pos: _expected_best_from_entries(entries_by_pos.get(pos, ()), exclude_names)
        for pos in needed
    }
    # Two picks at the next turn -> the two best things waiting for me.
    top_two = sorted(by_pos.values(), reverse=True)[:2]
    return float(sum(top_two)), by_pos


def optimize_turn(pool, drafted_players=None, my_team=None, teams=12, slot=None,
                  current_pick=None, top_k=PAIR_TOP_K, max_plans=6,
                  scoring_mode="standard", weights=None, drafted_by=None):
    """Rank the best two-player combinations for my next pair of picks.

    Returns (plans_df, context). `context` carries the cluster description
    even when no plans can be built, so callers can explain why.
    """
    context = {"cluster": None, "reason": None}

    if pool is None or pool.empty:
        context["reason"] = "No candidate pool available."
        return pd.DataFrame(), context

    cluster = find_pick_cluster(teams, slot, current_pick)
    context["cluster"] = cluster
    if cluster is None:
        context["reason"] = "Set your draft slot and league size to plan the turn."
        return pd.DataFrame(), context
    if not cluster["is_turn"]:
        context["reason"] = (
            f"Slot {slot} isn't a turn seat -- {cluster['gap']} opponent picks sit "
            f"between your next two picks, so there's no pair to lock in."
        )
        return pd.DataFrame(), context

    first, second = cluster["first_pick"], cluster["second_pick"]
    gap, following = cluster["gap"], cluster["following_pick"]

    # How many OPPONENT picks stand between now and each half of the pair.
    # `first` is mine, so it never counts toward the wait for `second`. On the
    # clock these are 0 and `gap` -- the case everything else is tuned for.
    now = current_pick or first
    opp_to_first = max(0, first - now)
    opp_to_second = max(0, second - now - 1)
    context["opponents_until_turn"] = opp_to_first

    # Naming a specific pair from across the room is theatre: 11 picks out,
    # every plan converges on "you'll get roughly the best available anyway"
    # (the fallback term swamps the difference) and the ranking degenerates
    # into noise -- observed live at pick 1 planning pick 12, where the top
    # four plans landed within 0.3 points of each other while recommending
    # players who were 8% to still be there. Stay quiet until the turn is
    # actually in view.
    if opp_to_first > MAX_PLANNING_HORIZON:
        context["reason"] = (
            f"Your turn is at picks {first} + {second} -- still {opp_to_first} picks away. "
            f"The board will turn over before then; this fills in as it gets close."
        )
        return pd.DataFrame(), context

    # my_next_pick=None deliberately switches OFF the single-pick
    # opportunity-cost penalty -- this module models survival itself, at the
    # pair level, and letting both apply would double-count it. top_n is pulled
    # wide because the landing-odds filter below removes the unreachable names.
    scored = recommend_picks(
        pool,
        drafted_players=drafted_players,
        my_team=my_team,
        weights=weights,
        top_n=max(top_k * 3, 45),
        current_round=cluster["round"],
        current_pick=first,
        my_next_pick=None,
        scoring_mode=scoring_mode,
    )
    if scored.empty or len(scored) < 2:
        context["reason"] = "Not enough available players to build a pair."
        return pd.DataFrame(), context

    # Only players who can realistically still be there are advice; the rest
    # are wishes. Filtering here (rather than ranking them and letting the
    # expected-value math bury them) is what keeps the list actionable. On the
    # clock every probability is 1.0, so this is a no-op in the common case.
    # Opponent-need-aware landing odds where the draft feed gives us per-team
    # attribution (Sleeper). ADP alone assumes every team drafts the consensus
    # board; knowing that the two teams picking before you have no WR and
    # three RBs is what separates "he might last" from "he is certainly gone."
    # Falls back to plain ADP when drafted_by is empty (manual mode).
    position_of = dict(zip(pool["player_name"], pool["position"]))
    mult_for = make_need_multiplier(
        drafted_by, position_of, teams, slot, STARTER_TARGETS, ROSTER_CAPS,
    )
    context["needs_aware"] = mult_for is not None

    scored = scored.copy()
    scored["_p_first"] = [
        survival_with_needs(adp, pos, now, opp_to_first, mult_for)
        for adp, pos in zip(scored["adp"], scored["position"])
    ]
    scored["_p_second"] = [
        survival_with_needs(adp, pos, now, opp_to_second, mult_for)
        for adp, pos in zip(scored["adp"], scored["position"])
    ]

    first_cands = scored[scored["_p_first"] >= MIN_LANDING_ODDS].head(top_k)
    second_cands = scored[scored["_p_second"] >= MIN_LANDING_ODDS].head(top_k)
    if first_cands.empty or second_cands.empty:
        context["reason"] = "Nothing on the board is likely enough to still be there."
        return pd.DataFrame(), context

    # Roster I already have, for the need recomputation.
    mine = pool[pool["player_name"].isin(my_team or [])]
    have = mine["position"].astype(str).str.upper().value_counts().to_dict()

    # The full available board (not just top_k) backs the lookahead, so the
    # "what's left at my next turn" estimate sees real depth rather than only
    # the handful of players in contention right now.
    taken = set(drafted_players or []) | set(my_team or [])
    avail_all = pool[~pool["player_name"].isin(taken)].copy()

    bye_w = {**DEFAULT_WEIGHTS, **(weights or {})}["bye_conflict"]
    # Both orderings get evaluated (a from the first-pick set, b from the
    # second-pick set) so the better one wins on merit -- which matters at a
    # near turn, where taking the guy who won't last FIRST is the whole
    # decision. The duplicate unordered pair is collapsed at the end.
    first_rows = first_cands.to_dict("records")
    second_rows = second_cands.to_dict("records")
    plans = []

    # --- precompute the survival horizons ONCE ----------------------------
    # Everything the pair loop needs is a function of a few fixed horizons,
    # not of the pair, so building these here turns a per-pair DataFrame
    # filter (300-row isin + copy, hundreds of times) into a walk over a
    # short list.
    names_all = avail_all["player_name"].tolist()
    scores_all = avail_all["our_score"].tolist()
    adp_all = avail_all["adp"] if "adp" in avail_all.columns else pd.Series([None] * len(avail_all))

    first_entries = _entries(names_all, scores_all, _survival_array(adp_all, now, opp_to_first))
    gap_entries = _entries(names_all, scores_all, _survival_array(adp_all, now, opp_to_second))

    # Lookahead horizon: surviving from the cluster to my following pick.
    next_gap = (following - second - 1) if following else None
    entries_by_pos = {}
    if next_gap is not None:
        surv_next = _survival_array(adp_all, second, next_gap)
        pos_all = avail_all["position"].astype(str).str.upper().tolist()
        for pos in STARTER_TARGETS:
            idx = [i for i, p in enumerate(pos_all) if p == pos]
            entries_by_pos[pos] = _entries(
                [names_all[i] for i in idx],
                [scores_all[i] for i in idx],
                [surv_next[i] for i in idx],
            )

    for a, b in itertools.product(first_rows, second_rows):
        if a["player_name"] == b["player_name"]:
            continue
        pos_a = str(a.get("position", "")).upper()
        pos_b = str(b.get("position", "")).upper()

        # Can't roster two of a capped position (1-QB / 1-TE leagues).
        if pos_a == pos_b and have.get(pos_a, 0) + 2 > ROSTER_CAPS.get(pos_a, 99):
            continue

        # Both halves are scored on the odds they're actually still there
        # when the pick comes up. On the clock at a true turn that is 1.0 and
        # 1.0 -- the pair is a guarantee, which is the whole point of the
        # turn. Planning from a few picks out, both discount honestly.
        p_first = float(a["_p_first"])
        p_land = float(b["_p_second"])

        # Correct the second player's need term for the roster the first pick
        # creates. Without this, RB/RB collects the RB need bonus twice.
        have_after_a = dict(have)
        have_after_a[pos_a] = have_after_a.get(pos_a, 0) + 1
        need_delta = _need_pts(pos_b, have_after_a) - _need_pts(pos_b, have)

        # Same position AND same bye is a real roster problem the per-player
        # pass can't see, since neither is on my roster yet when it runs.
        bye_pen = 0.0
        if pos_a == pos_b and pd.notna(a.get("bye_week")) and pd.notna(b.get("bye_week")):
            if int(a["bye_week"]) == int(b["bye_week"]):
                bye_pen = -bye_w

        # Shared bye + shared offense + split touches -- see SAME_TEAM_PENALTY.
        stack_pen = 0.0
        team_a, team_b = a.get("team"), b.get("team")
        if (
            pd.notna(team_a) and pd.notna(team_b)
            and str(team_a).upper() == str(team_b).upper()
            and {pos_a, pos_b} not in _STACK_OK
        ):
            stack_pen = -SAME_TEAM_PENALTY

        score_b = float(b["draft_score"]) + need_delta + bye_pen + stack_pen

        # If `b` is gone, I take the best thing still there instead. Folding
        # that in is what stops the optimizer from recommending a pair built
        # on a player who realistically will not last the gap.
        # If a player is gone, I take the best thing still on the board
        # instead. Folding that in is what stops the optimizer from building
        # a plan around someone who realistically will not last.
        pair_names = frozenset((a["player_name"], b["player_name"]))

        if p_first >= 0.999:
            expected_first = float(a["draft_score"])
        else:
            expected_first = (
                p_first * float(a["draft_score"])
                + (1.0 - p_first) * _expected_best_from_entries(first_entries, pair_names)
            )

        if p_land >= 0.999:
            fallback = 0.0
            expected_second = score_b
        else:
            fallback = _expected_best_from_entries(gap_entries, pair_names)
            expected_second = p_land * score_b + (1.0 - p_land) * fallback

        have_after = dict(have_after_a)
        have_after[pos_b] = have_after.get(pos_b, 0) + 1
        outlook, outlook_by_pos = _next_turn_outlook(
            entries_by_pos, pair_names, have_after,
        )

        total = expected_first + expected_second + NEXT_TURN_WEIGHT * outlook

        plans.append({
            "first": a["player_name"],
            "first_pos": pos_a,
            "first_team": a.get("team"),
            "first_score": round(float(a["draft_score"]), 2),
            "first_adp": a.get("adp"),
            "first_land_pct": int(round(p_first * 100)),
            "expected_first": round(expected_first, 2),
            "second": b["player_name"],
            "second_pos": pos_b,
            "second_team": b.get("team"),
            "second_score": round(score_b, 2),
            "second_adp": b.get("adp"),
            "land_pct": int(round(p_land * 100)),
            "expected_second": round(expected_second, 2),
            "next_turn_outlook": round(outlook, 2),
            "next_turn_by_pos": outlook_by_pos,
            "combo_score": round(total, 2),
            "shape": "/".join(sorted([pos_a, pos_b])),
            "need_delta": round(need_delta, 2),
            "bye_penalty": round(bye_pen, 2),
            "stack_penalty": round(stack_pen, 2),
            "same_team": stack_pen != 0.0,
            "fallback_value": round(fallback, 2),
        })

    if not plans:
        context["reason"] = "No legal pair available under your roster caps."
        return pd.DataFrame(), context

    df = pd.DataFrame(plans).sort_values("combo_score", ascending=False)

    # One entry per unordered pair -- permutations() produced both orderings so
    # the better one could win on merit (it matters whenever the two have
    # different landing odds), but showing both as separate plans would be
    # noise. Keeping the first occurrence keeps the higher-scoring order.
    df["_key"] = df.apply(lambda r: "|".join(sorted([r["first"], r["second"]])), axis=1)
    df = df.drop_duplicates("_key", keep="first").drop(columns="_key")

    df = df.head(max_plans).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df, context
