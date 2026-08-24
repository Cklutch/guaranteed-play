"""Rookie Projection Model v1 (rookie_projection_model_v1_spec.pdf).

Produces a pre-season projected archetype tier for rookies (WR: Alpha/
Possession/Complementary; RB: Bellcow/Committee -- same tier names as
draftkit/wr_archetypes.py/rb_archetypes.py's own real usage tiers), then
blends it toward the CONFIRMED in-season tier from those same modules as
real game sample accumulates (blend_rookie_tag()).

Ports the spec's component-score and composite functions close to verbatim,
with two real, confirmed corrections:

1. Team-context inversion+rescale fix (team_context_quality(), replacing
   the spec's literal team_context_scores()). The spec's docstring claims
   build_risk_variables.offense_environment_score()/
   schedule_weather_venue_score() are "already 0-1." Directly re-read both
   functions: they're real, clamped to 1.0-5.0, and RISK-direction (higher
   = worse -- epa_rank=1 is the best real offense and produces the LOWEST
   score). Used literally as positive composite contributors, a rookie
   landing with an elite real offense would score LOWER than one landing
   with a bad one -- backwards. Fixed by inverting and rescaling to a 0-100
   "landing quality" value before applying the spec's own weights.

2. Missing-input neutral defaults. college_dominator_final_year/career and
   breakout_age have no real data source in this repo yet (the spec's own
   Section 1 calls for a manually-curated CSV; per explicit user direction
   this build ships with those three fields blank for every rookie until
   real data is available -- see draftkit/scripts/build_rookie_inputs.py's
   module docstring). dominator_score()'s literal `dominator_rating * 2.2`
   would crash on a missing value -- given an explicit None-input path
   returning a documented neutral 50.0 (NOT a fabricated rating).
   breakout_age_score() already had a spec-defined None/"none" path (20.0),
   kept as-is. athletic_score_rb() gets the same neutral-default treatment
   for the ~9 real rookies (of 74) with no Combine measurement on file.

roster_competition_tier is NOT automatable in v1 either (no real per-team
depth-chart-crowding data source exists in this repo) -- every rookie gets
the spec's own flagged manual default; see build_rookie_inputs.py.
"""

from __future__ import annotations

from draftkit.qb_archetypes import classify_primary as qb_classify_primary

NEUTRAL_DOMINATOR_SCORE = 50.0  # "no real college production data yet" -- not a rating
NEUTRAL_ATHLETIC_SCORE = 50.0   # "no real Combine measurement yet" -- not a fabricated time

# Real range of build_risk_variables.py's offense_environment_score()/
# schedule_weather_venue_score() -- see module docstring for the confirmed
# 1.0-5.0/risk-direction finding this corrects.
TEAM_RISK_SCORE_BEST = 1.0
TEAM_RISK_SCORE_WORST = 5.0


# Real, position-specific decay constants -- calibrate_draft_capital.py
# (draftkit/scripts/calibrate_draft_capital.py), fit via least-squares
# against each position's real empirical "percentile of lateness" curve
# (raw_draft_picks.csv, 2016-2025, 10 real completed classes). Replaces a
# single shared k=15 that was never fit against anything: checked directly,
# k=15 scores AGAINST the real empirical curve at r_squared=-0.15 to -0.95
# per position (worse than predicting the flat mean) -- the old decay was
# far steeper than real draft capital actually falls off. Fitted values:
#   RB: k=116.08 (n=215 picks, r_squared=0.72)
#   WR: k=98.86  (n=322 picks, r_squared=0.83)
#   QB: k=74.32  (n=120 picks, r_squared=0.84) -- computed, unused until a
#   TE: k=101.84 (n=141 picks, r_squared=0.76) -- QB/TE model exists.
# See data/processed/draft_capital_calibration.csv for the full real report.
DRAFT_CAPITAL_K_BY_POSITION = {"RB": 116.08, "WR": 98.86, "QB": 74.32, "TE": 101.84}
DRAFT_CAPITAL_K_DEFAULT = 100.0  # simple average of the four real fitted k's, for any other position


def draft_capital_score(pick_number: int | None, position: str | None = None) -> float:
    if pick_number is None:  # UDFA -- no real pick to calibrate against, flat low floor
        return 15.0
    k = DRAFT_CAPITAL_K_BY_POSITION.get(position, DRAFT_CAPITAL_K_DEFAULT)
    return round(100 * (1 / (1 + pick_number / k)), 1)


def dominator_score(dominator_rating: float | None) -> float:
    if dominator_rating is None:
        return NEUTRAL_DOMINATOR_SCORE
    return round(min(100, dominator_rating * 2.2), 1)


def breakout_age_score(breakout_age) -> float:
    if breakout_age is None or breakout_age == "none":
        return 20.0
    return max(0.0, round(100 - (breakout_age - 18) * 15, 1))


def athletic_score_rb(weight: float | None, forty_time: float | None) -> float:
    if weight is None or forty_time is None or forty_time <= 0:
        return NEUTRAL_ATHLETIC_SCORE
    speed_score = (weight * 200) / (forty_time ** 4)
    return round(min(100, speed_score / 1.2), 1)


def competition_score(roster_competition_tier: int) -> float:
    """1 (clear path to volume) to 5 (crowded room) -- inverted so a clear
    path scores high."""
    return (6 - roster_competition_tier) * 20


def team_context_quality(risk_score: float) -> float:
    """Inverts+rescales a real 1.0(best)-5.0(worst) risk-direction team
    score to a 0-100 landing-quality value, higher=better. See module
    docstring for the confirmed bug this corrects."""
    span = TEAM_RISK_SCORE_WORST - TEAM_RISK_SCORE_BEST
    quality = (TEAM_RISK_SCORE_WORST - risk_score) / span * 100
    return round(max(0.0, min(100.0, quality)), 1)


def team_context_scores(landing_team: str, risk_engine_lookup: dict) -> dict:
    """Direct reuse of existing, already-computed risk engine outputs (real
    per-team offense_environment_score/schedule_weather_venue_score from
    build_risk_variables.py), inverted+rescaled via team_context_quality()
    -- no new landing-spot formula invented, just the correction above."""
    team_data = risk_engine_lookup[landing_team]
    return {
        "offense_environment": team_context_quality(team_data["offense_environment_score"]),
        "schedule": team_context_quality(team_data["schedule_weather_venue_score"]),
    }


def wr_rookie_projection(inputs: dict, risk_engine_lookup: dict) -> dict:
    dc = draft_capital_score(inputs["draft_pick"], "WR")
    dom = dominator_score(inputs["college_dominator_final_year"])
    boa = breakout_age_score(inputs["breakout_age"])
    comp = competition_score(inputs["roster_competition_tier"])
    team_ctx = team_context_scores(inputs["landing_team"], risk_engine_lookup)

    composite = round(
        dc * 0.35 + dom * 0.25 + boa * 0.10 + comp * 0.15
        + team_ctx["offense_environment"] * 0.10 + team_ctx["schedule"] * 0.05, 1
    )

    # "possession" removed (explicit user correction, 2026-08-15: rookies
    # can't have it either, matching the confirmed real system's
    # Boom-Bust/High-Floor split -- claude_code_plan_possession_split.pdf).
    # Not replaced with a projected Boom-Bust/High-Floor read: that split
    # is driven by real aDOT and catch_rate, and no college-level
    # equivalent of either exists anywhere in rookie_inputs.csv or
    # backtest_rookie_inputs.csv (checked directly, not assumed -- college
    # PBP-derived target-depth data was never sourced for this pipeline).
    # Fabricating a boom-bust/high-floor guess from data that can't
    # support the distinction would be worse than not making it. The old
    # 50-74 composite middle band collapses into Complementary instead --
    # a real, honest "not (yet) a clear Alpha" read, not a fake split.
    tier = "alpha" if composite >= 75 else "complementary"

    return {
        "composite": composite, "projected_tier": tier,
        "components": {
            "draft_capital": dc, "dominator": dom, "breakout_age": boa,
            "roster_competition": comp, **team_ctx,
        },
    }


def rb_rookie_projection(inputs: dict, risk_engine_lookup: dict) -> dict:
    dc = draft_capital_score(inputs["draft_pick"], "RB")
    dom = dominator_score(inputs["college_dominator_career"])
    ath = athletic_score_rb(inputs["weight"], inputs["forty_time"])
    comp = competition_score(inputs["roster_competition_tier"])
    team_ctx = team_context_scores(inputs["landing_team"], risk_engine_lookup)

    composite = round(
        dc * 0.30 + dom * 0.15 + ath * 0.15 + comp * 0.20
        + team_ctx["offense_environment"] * 0.15 + team_ctx["schedule"] * 0.05, 1
    )

    # "committee" here is a coarser projected bucket than rb_archetypes.py's
    # real 6-way CONFIRMED "committee_back" -- the pre-season model can't
    # see redzone/third-down/route-participation splits yet, so it only
    # projects the two-way volume split the spec asks for. blend_rookie_tag
    # doesn't try to reconcile the two vocabularies; it just swaps the whole
    # tag over to the real CONFIRMED value once sample_weight is high enough.
    tier = "bellcow" if composite >= 70 else "committee"

    return {
        "composite": composite, "projected_tier": tier,
        "components": {
            "draft_capital": dc, "dominator": dom, "athletic": ath,
            "roster_competition": comp, **team_ctx,
        },
    }


def ncaa_passer_rating_score(cmp: float, att: float, yds: float, td: float, interceptions: float) -> float:
    """Real, standard, externally-validated NCAA passer rating formula --
    not an invented weighted blend of completion%%/Y-A/TD-INT ratio
    separately (claude_code_plan_qb_rookie_projection.pdf). Rescaled to
    0-100 for the composite: computed directly against the full real
    2021-2025 draft class before choosing /2.5 (real observed range
    ~130-210), not tuned to hit a preconceived number -- Jayden Daniels/Mac
    Jones/Zach Wilson land near the top, Anthony Richardson at the bottom,
    consistent with real scouting reputations."""
    if not att:
        return 0.0
    rating = ((8.4 * yds) + (330 * td) + (100 * cmp) - (200 * interceptions)) / att
    return round(max(0.0, min(100.0, rating / 2.5)), 1)


def qb_rookie_projection(inputs: dict, risk_engine_lookup: dict) -> dict:
    """No combine term (confirmed absent for QB anywhere in this repo --
    only rb_/wr_combine_measurables_*.csv exist) -- weight redistributed
    across the three real available inputs, same honesty principle as
    dropping a fabricated Boom-Bust/High-Floor split for rookie WRs.

    Archetype lean is NOT part of the composite above and NOT a nearest-fit
    distance calc -- direct reuse of qb_archetypes.classify_primary(), the
    same veteran function, fed a college-scale rushing_fantasy_pct (real,
    manually overridden, or assumed-zero -- see
    build_rookie_projections.py's QB_ROOKIE_RUSHING_OVERRIDES). Cross-
    validated against real anchors before locking this reuse in: Milroe/
    Lance/Willis/Richardson/Daniels all clear Dual-Threat comfortably on
    college data; Maye/Williams/Nix land the same tier college vs. NFL."""
    dc = draft_capital_score(inputs["draft_pick"], "QB")
    passing = ncaa_passer_rating_score(
        inputs["cmp"], inputs["att"], inputs["yds"], inputs["td"], inputs["interceptions"],
    )
    team_ctx = team_context_scores(inputs["landing_team"], risk_engine_lookup)

    composite = round(
        dc * 0.45 + passing * 0.40
        + team_ctx["offense_environment"] * 0.10 + team_ctx["schedule"] * 0.05, 1
    )

    tier = qb_classify_primary(inputs["rushing_fantasy_pct"], inputs["att"], inputs["games"]).value

    return {
        "composite": composite, "projected_tier": tier,
        "components": {
            "draft_capital": dc, "ncaa_passer_rating": passing, **team_ctx,
        },
    }


def blend_rookie_tag(projected: dict, confirmed_scores, sample_weight: float) -> dict:
    """Unchanged from spec. sample_weight in [0, 1] -- how much real
    in-season sample has accumulated (see build_rookie_projections.py for
    how this is computed via each position's own sample_confidence()
    floor).

    A years-since-draft-based display cap was tried and reverted
    (claude_code_plan_ui_and_blend_fix.pdf): it let a stale pre-draft
    composite keep partially overriding an ALREADY-CONFIRMED real
    archetype (Ashton Jeanty showing "Committee" despite a real, unambiguous
    2025 bellcow season) just because too few calendar years had passed.
    sample_weight here is real usage-sample confidence, not elapsed time --
    once a real confirmed_scores exists AND sample_weight clears 1.0, that
    reads as fully confirmed immediately, regardless of draft year."""
    if sample_weight >= 1.0:
        return {"status": "confirmed", "tag": confirmed_scores}
    if sample_weight <= 0.0:
        return {"status": "projected", "tag": projected["projected_tier"], "model_score": projected["composite"]}
    return {
        "status": "blended",
        "tag": confirmed_scores if sample_weight >= 0.5 else projected["projected_tier"],
        "confirmed_pct": round(sample_weight * 100),
        "note": f"Blended read -- {round(sample_weight * 100)}% weighted toward in-season data.",
    }


__all__ = [
    "DRAFT_CAPITAL_K_BY_POSITION",
    "draft_capital_score",
    "dominator_score",
    "breakout_age_score",
    "athletic_score_rb",
    "competition_score",
    "team_context_quality",
    "team_context_scores",
    "wr_rookie_projection",
    "rb_rookie_projection",
    "ncaa_passer_rating_score",
    "qb_rookie_projection",
    "blend_rookie_tag",
]
