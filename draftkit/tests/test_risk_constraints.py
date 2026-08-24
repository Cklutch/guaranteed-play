"""Spec test cases A-D (fantasy_draft_risk_scorecard_master_spec.pdf, Section 8).

v2: category names updated to the 4-category restructure (RiskCategory.ROLE
became ROLE_USAGE_TD; RiskCategory.VOLATILITY was removed entirely -- v2's
volatility is a display-only diagnostic, not a weighted RiskCategory, so
test B's ceiling-tightening mechanism check was moved to
OFFENSE_ENVIRONMENT instead. That test was never really "about" volatility
specifically -- dynamic_risk_ceiling()'s tightening logic is category-
agnostic, so any valid RiskCategory member exercises the same mechanism.

Plain assert-based, no pytest (not installed in this project's venv) --
matches the loose test_*.py convention already used elsewhere in this repo
(e.g. research/validation_v1/test_offense_environment_v1.py). Runnable
directly:

    python -m draftkit.tests.test_risk_constraints
"""

from __future__ import annotations

from draftkit.risk_constraints import (
    DraftPool,
    TeamRiskBudget,
    check_correlated_risk,
    draft_pick,
    dynamic_risk_ceiling,
    evaluate_pick_fit,
    undo_pick,
)
from draftkit.risk_scoring import (
    DraftStatus,
    Player,
    Position,
    RiskCategory,
    RiskVariable,
    SEVERITY_FLOOR_THRESHOLD,
    SEVERITY_FLOOR_VALUE,
    risk_index,
)


def _player(name, position, category_scores: dict) -> Player:
    """Build a Player with exactly one variable per given category."""
    variables = [
        RiskVariable(name=f"{cat}_var", category=RiskCategory(cat), score=score)
        for cat, score in category_scores.items()
    ]
    return Player(name=name, position=position, adp=50.0, variables=variables)


def test_a_correlated_role_risk_red():
    """Roster has an RB with role_usage_td >= 4. Candidate RB also has
    role_usage_td >= 4."""
    existing_rb = _player("Existing RB", Position.RB, {"role_usage_td": 4.5, "injury": 2.0})
    budget = TeamRiskBudget(picks=[existing_rb])

    candidate = _player("Candidate RB", Position.RB, {"role_usage_td": 4.2, "injury": 2.0})
    flags = check_correlated_risk(candidate, budget)
    assert flags["role_usage_td"] is True, f"expected role_usage_td correlation flag True, got {flags}"

    fit = evaluate_pick_fit(candidate, budget)
    assert fit == "red", f"expected red, got {fit}"
    print("A PASS -- correlated role_usage_td risk -> red")


def test_b_draft_sequence_tightening_yellow():
    """First 3 picks are WRs with offense_environment >= 4. Candidate RB has
    offense_environment = 4.2, no role conflict. Ceiling tightens to 3.0 ->
    yellow. (v1 used volatility for this mechanism check; v2 dropped
    volatility as a weighted RiskCategory, so offense_environment
    substitutes -- the tightening mechanism itself is category-agnostic.)"""
    wrs = [
        _player(f"WR {i}", Position.WR, {"offense_environment": 4.0 + i * 0.1, "role_usage_td": 2.0})
        for i in range(3)
    ]
    budget = TeamRiskBudget(picks=wrs)

    ceilings = dynamic_risk_ceiling(budget, base_ceiling=4.0)
    assert ceilings["offense_environment"] == 3.0, f"expected tightened ceiling 3.0, got {ceilings}"

    candidate = _player("Candidate RB", Position.RB, {"offense_environment": 4.2, "role_usage_td": 2.0})
    # No RB on the roster yet, so no correlated-risk conflict -- this must
    # fail purely on the tightened ceiling, not on correlation.
    assert not any(check_correlated_risk(candidate, budget).values())

    fit = evaluate_pick_fit(candidate, budget)
    assert fit == "yellow", f"expected yellow, got {fit}"
    print("B PASS -- draft-sequence tightening -> yellow")


def test_c_clean_diversifying_pick_green():
    """Roster has one high-Role/Usage/TD-Risk RB. Candidate RB has low
    role_usage_td risk (clear bell-cow) and moderate injury risk only ->
    green."""
    existing_rb = _player("Existing RB", Position.RB, {"role_usage_td": 4.5})
    budget = TeamRiskBudget(picks=[existing_rb])

    candidate = _player("Bell Cow RB", Position.RB, {"role_usage_td": 1.5, "injury": 2.5})
    flags = check_correlated_risk(candidate, budget)
    # candidate's own role_usage_td score (1.5) is below the 4.0 threshold,
    # so it can never trigger a correlation flag regardless of the roster.
    assert not any(flags.values()), f"expected no correlation flags, got {flags}"

    fit = evaluate_pick_fit(candidate, budget)
    assert fit == "green", f"expected green, got {fit}"
    print("C PASS -- clean diversifying pick -> green")


def test_d_draft_pool_integrity():
    """A player marked taken_by_other never appears in board or affects the
    budget. mine appends to both pool and budget. undo cleanly reverses
    either case."""
    players = {
        "jadarian price": Player("Jadarian Price", Position.RB, 120.0),
        "treveyon henderson": Player(
            "TreVeyon Henderson", Position.RB, 45.0,
            variables=[RiskVariable("role_usage_td", RiskCategory.ROLE_USAGE_TD, 4)],
        ),
    }
    pool = DraftPool(all_players=players)
    budget = TeamRiskBudget()

    # taken by opponent
    draft_pick(pool, budget, "jadarian price", by_me=False)
    assert pool.status["jadarian price"] == DraftStatus.TAKEN_BY_OTHER
    assert "Jadarian Price" not in [p.name for p in pool.available_players()]
    assert len(budget.picks) == 0, "opponent pick must not touch my budget"

    # double-entry must raise
    try:
        draft_pick(pool, budget, "jadarian price", by_me=False)
        raise AssertionError("expected ValueError on double mark_taken")
    except ValueError:
        pass

    # mine appends to both pool status and budget
    draft_pick(pool, budget, "treveyon henderson", by_me=True)
    assert pool.status["treveyon henderson"] == DraftStatus.MY_TEAM
    assert len(budget.picks) == 1
    assert budget.picks[0].name == "TreVeyon Henderson"
    assert "TreVeyon Henderson" not in [p.name for p in pool.available_players()]

    # undo reverses the single most recent pick (the "mine" one) in BOTH
    # pool status AND the budget -- not just pool status.
    undone = undo_pick(pool, budget)
    assert undone == "treveyon henderson"
    assert pool.status["treveyon henderson"] == DraftStatus.AVAILABLE
    assert "TreVeyon Henderson" in [p.name for p in pool.available_players()]
    assert len(budget.picks) == 0, "undo must also remove the pick from the budget"

    fuzzy = pool.resolve_name("Jadarian Prise")  # 1-char typo
    assert fuzzy == "jadarian price", f"fuzzy match failed, got {fuzzy}"

    print("D PASS -- draft pool integrity")


def test_e_risk_index_floor_subtracted():
    """claude_code_plan_risk_index_floor_fix.pdf: risk_index() used to weight
    the raw 1-5 score directly against a 5*sum(weights) ceiling, so a raw
    score of 1 (the safest possible value) still contributed 1/5=20% of its
    own weight -- inflating the final number regardless of real risk.

    James Cook is the real, locked-in case this fix was diagnosed and
    verified against: real raw scores (1.0, 1.1, 1.0, 4.0) against real RB
    weights (0.38/0.32/0.20/0.10) used to produce risk_index=26.6 under the
    old formula -- hand-verified under the new floor-subtracted formula:
    weighted_sum = ((1.0-1)/4)*0.38 + ((1.1-1)/4)*0.32 + ((1.0-1)/4)*0.20
    + ((4.0-1)/4)*0.10 = 0.083, max_possible = 1.00, risk_index = 8.3 --
    a meaningfully lower, more honest number given 3 of 4 categories sit
    at the real safest floor."""
    weights = {"RB": {"injury": 0.38, "role_usage_td": 0.32, "offense_environment": 0.20, "schedule_weather_venue": 0.10}}
    cook = _player("James Cook", Position.RB, {
        "injury": 1.0, "role_usage_td": 1.1, "offense_environment": 1.0, "schedule_weather_venue": 4.0,
    })
    result = risk_index(cook, weights)
    assert result == 8.3, f"expected 8.3 (real, locked-in James Cook case), got {result}"
    print(f"E PASS -- James Cook: raw (1.0, 1.1, 1.0, 4.0) -> risk_index={result} (was 26.6 pre-fix)")


def test_f_risk_index_missing_category_not_negative():
    """Real bug caught during verification of the floor-subtracted fix
    above, not in the original fix spec: category_score() returns 0.0 --
    not the 1.0 floor -- for a category with no RiskVariable at all (a
    real, common case: offense_environment/schedule_weather_venue only
    cover 625/730 real players). Floor-subtracting that 0.0 unconditionally
    -- (0-1)/4 = -0.25 -- turned "no data for this category" into a
    NEGATIVE contribution, producing real risk_index values below zero
    (e.g. -4.4) for players missing 2 of 4 categories. risk_index() now
    only iterates the categories the player actually has data for,
    re-normalizing both weighted_sum and max_possible over just those --
    missing data is excluded, not penalized.

    Real case this reproduces: a real backup/practice-squad-tier QB with
    only injury+role_usage_td data (no real team-level offense/schedule
    context on file) -- e.g. Austin Reed, Diego Pavia, Emory Jones all hit
    this exact path with identical real inputs."""
    weights = {"QB": {"injury": 0.20, "role_usage_td": 0.25, "offense_environment": 0.45, "schedule_weather_venue": 0.10}}
    partial = _player("Backup QB", Position.QB, {"injury": 1.0, "role_usage_td": 2.5})
    result = risk_index(partial, weights)
    assert result >= 0, f"expected a non-negative risk_index for a player missing 2 categories, got {result}"
    assert result == 20.8, f"expected 20.8 (re-normalized over injury+role_usage_td only), got {result}"
    print(f"F PASS -- player missing offense_environment/schedule_weather_venue entirely -> "
          f"risk_index={result} (non-negative, re-normalized over present categories only)")


def test_g_severity_floor_outlier_not_diluted():
    """plan_risk_index_reweight_dynamic.pdf Phase 1: a linear weighted
    average dilutes a genuine outlier toward "average" -- real case: Joe
    Burrow's real pre-decay injury_score=4.0 (Grade 3 turf toe requiring
    surgery) blended down to a composite of just 49.6 under the old
    formula, since QB's offense_environment weight (0.45) swamped it.
    Synthetic case mirrors Burrow's real pre-decay category scores."""
    weights = {"QB": {"injury": 0.20, "role_usage_td": 0.25, "offense_environment": 0.45, "schedule_weather_venue": 0.10}}
    burrow_like = _player("Burrow-like QB", Position.QB, {
        "injury": 4.0, "role_usage_td": 3.4, "offense_environment": 2.5, "schedule_weather_venue": 2.1,
    })
    result = risk_index(burrow_like, weights)
    assert result >= SEVERITY_FLOOR_VALUE, f"expected the severity floor to trigger (>={SEVERITY_FLOOR_VALUE}), got {result}"

    just_under = _player("Sub-threshold QB", Position.QB, {
        "injury": SEVERITY_FLOOR_THRESHOLD - 0.1, "role_usage_td": 3.4, "offense_environment": 2.5, "schedule_weather_venue": 2.1,
    })
    result_under = risk_index(just_under, weights)
    assert result_under < SEVERITY_FLOOR_VALUE, f"expected no floor trigger just under the threshold, got {result_under}"
    assert result_under < result, "expected the floored composite to exceed the sub-threshold one"
    print(f"G PASS -- severity floor: injury={SEVERITY_FLOOR_THRESHOLD} -> risk_index={result} (floored); "
          f"injury={SEVERITY_FLOOR_THRESHOLD - 0.1} -> {result_under} (not floored)")


def test_h_severity_floor_clean_player_unaffected():
    """A genuinely clean/low-risk player must stay low -- the floor rule
    only ever raises a composite, never inflates one that has no real
    outlier category."""
    weights = {"RB": {"injury": 0.38, "role_usage_td": 0.32, "offense_environment": 0.20, "schedule_weather_venue": 0.10}}
    clean = _player("Clean RB", Position.RB, {
        "injury": 1.2, "role_usage_td": 1.3, "offense_environment": 1.5, "schedule_weather_venue": 1.4,
    })
    result = risk_index(clean, weights)
    assert result < 20, f"expected a genuinely low composite for a clean player (no floor trigger), got {result}"
    print(f"H PASS -- clean player stays low, floor doesn't trigger: risk_index={result}")


def main() -> int:
    test_a_correlated_role_risk_red()
    test_b_draft_sequence_tightening_yellow()
    test_c_clean_diversifying_pick_green()
    test_d_draft_pool_integrity()
    test_e_risk_index_floor_subtracted()
    test_f_risk_index_missing_category_not_negative()
    test_g_severity_floor_outlier_not_diluted()
    test_h_severity_floor_clean_player_unaffected()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
