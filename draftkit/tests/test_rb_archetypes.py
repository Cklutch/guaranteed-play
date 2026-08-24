"""RB archetype taxonomy tests (rb_archetype_implementation_spec.pdf).

Plain assert-based, no pytest -- matches this repo's established
convention. Calls the real draftkit.scripts.build_rb_archetypes pipeline
once against real 2025 season data and checks named real players in the
output -- not a frozen snapshot, not a reimplementation of its logic.

rb_archetype_taxonomy_log.md (referenced by the spec for the "expected"
tags on these named players) isn't in this repo. Per explicit user
direction, every assertion below is derived from each player's real,
directly-verified 2025 inputs against the spec's own given thresholds --
not copied from an unseen ground truth. Where a case surfaced a real
implementation bug during that verification, the bug and fix are noted
inline (see build_rb_archetypes.py's own comments for the full detail).

Runnable directly:
    python -m draftkit.tests.test_rb_archetypes
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_rb_archetypes import build_rb_archetypes  # noqa: E402

_BOARD = None


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = build_rb_archetypes()
    return _BOARD


def _row(player_name: str):
    board = _board()
    matches = board[board["player_name"] == player_name]
    assert not matches.empty, f"expected {player_name} in the current RB pool, found none"
    return matches.iloc[0]


def test_gibbs_bellcow():
    """Real 2025: 337 total touches (243 carries + 94 targets) over 17
    games, a real 63.6% share of Detroit's backfield work. Below the
    spec's literal 0.65 bellcow floor but above the corrected 0.55 floor
    (see rb_archetypes.py's BELLCOW_OPPORTUNITY_SHARE_FLOOR docstring for
    why 0.65 was unreachable for real bellcows once opportunity_share was
    correctly redefined as backfield share)."""
    row = _row("Jahmyr Gibbs")
    assert 0.60 <= row["opportunity_share"] <= 0.68, f"expected ~0.636, got {row['opportunity_share']}"
    assert row["archetype_primary"] == "bellcow", f"expected bellcow, got {row['archetype_primary']}"
    print(f"PASS -- Gibbs: opp_share={row['opportunity_share']:.3f}, archetype={row['archetype_primary']}, "
          f"down_split={row['down_split']}, lean={row['lean']}")


def test_henry_bellcow_two_down():
    """Real 2025: 307 carries + 21 targets, 70.5% of BAL's backfield --
    a clear, dominant bellcow. Real down_split expected 2-down (Henry's
    well-known real profile: minimal receiving-down usage relative to his
    rushing workload)."""
    row = _row("Derrick Henry")
    assert row["opportunity_share"] >= 0.65, f"expected clear bellcow-range share, got {row['opportunity_share']}"
    assert row["archetype_primary"] == "bellcow", f"expected bellcow, got {row['archetype_primary']}"
    assert row["down_split"] == "2-down", f"expected 2-down (real low third-down usage), got {row['down_split']}"
    print(f"PASS -- Henry: opp_share={row['opportunity_share']:.3f}, archetype={row['archetype_primary']}, "
          f"down_split={row['down_split']}")


def test_kyren_williams_bellcow_three_down():
    """Real 2025: LAR's true 3-down back -- 64.8% backfield share with a
    real high third_down_snap_share, matching his known real role."""
    row = _row("Kyren Williams")
    assert row["archetype_primary"] == "bellcow", f"expected bellcow, got {row['archetype_primary']}"
    assert row["down_split"] == "3-down", f"expected 3-down, got {row['down_split']}"
    print(f"PASS -- Kyren Williams: opp_share={row['opportunity_share']:.3f}, down_split={row['down_split']}")


def test_stevenson_henderson_committee():
    """The spec's named 'committee pair' (New England). Real 2025: Henderson
    48.8% backfield share, Stevenson 36.7% (missed 3 games) -- a real,
    active timeshare, gap=12.1 points, both clear real-sample floors.

    Surfaced and fixed a real bug during verification: NE's roster also
    carries zero-real-data names (Jam Miller, Lan Larison) that were being
    included as "teammates" for the min_games confidence check, dragging
    it to 0 and failing what should have been a correct committee_back
    match for Henderson. Fixed by filtering teammates to real sample sizes
    (games_played>=4) before the archetype conditions run -- see
    build_rb_archetypes.py's build_rb_archetypes() for the full comment.

    RB taxonomy migration (explicit user correction, 2026-08-15): primary
    is a usage tier only (Bellcow/Committee/Handcuff/Unconfirmed) -- a
    role/trait like Receiving Back is now a SECONDARY lean tag, never
    primary. Stevenson used to independently qualify for receiving_back as
    PRIMARY (real high third_down_snap_share, 67%) and precedence resolved
    to it over committee_back -- that was the exact taxonomy bug this
    migration fixes. He now shows committee_back as primary (his real
    opportunity_share genuinely sits in Committee's band) with
    receiving_lean as a secondary tag, carrying the same real signal
    without pretending it's a competing usage tier."""
    henderson = _row("TreVeyon Henderson")
    stevenson = _row("Rhamondre Stevenson")
    assert 0.30 <= henderson["opportunity_share"] <= 0.55, f"got {henderson['opportunity_share']}"
    assert henderson["archetype_primary"] == "committee_back", (
        f"expected committee_back, got {henderson['archetype_primary']}"
    )
    assert stevenson["archetype_primary"] == "committee_back", (
        f"expected committee_back (usage-tier primary), got {stevenson['archetype_primary']}"
    )
    assert stevenson["lean"] == "receiving_lean", (
        f"expected receiving_lean as a secondary tag, got {stevenson['lean']}"
    )
    print(f"PASS -- Henderson: {henderson['archetype_primary']} (opp_share={henderson['opportunity_share']:.3f}); "
          f"Stevenson: {stevenson['archetype_primary']} · {stevenson['lean']} "
          f"(opp_share={stevenson['opportunity_share']:.3f})")


def test_jordan_mason_committee():
    """Real 2025: Minnesota's real near-even split -- Mason 42.0% vs Aaron
    Jones' 41.5% backfield share, a 0.5-point gap. Same teammate-filtering
    bug as Stevenson/Henderson, caught independently: Jordan Mims' single
    real game (not zero, so the first fix pass didn't catch it) was still
    dragging min_games to 1. Fixed by the same games_played>=4 teammate
    filter."""
    mason = _row("Jordan Mason")
    jones = _row("Aaron Jones")
    assert abs(mason["opportunity_share"] - jones["opportunity_share"]) < 0.05, (
        "expected a real near-even split between Mason and Jones"
    )
    assert mason["archetype_primary"] == "committee_back", f"expected committee_back, got {mason['archetype_primary']}"
    assert jones["archetype_primary"] == "committee_back", f"expected committee_back, got {jones['archetype_primary']}"
    print(f"PASS -- Jordan Mason: opp_share={mason['opportunity_share']:.3f}, "
          f"Aaron Jones: opp_share={jones['opportunity_share']:.3f}, both committee_back")


def test_dobbins_committee():
    """Real 2025: missed real time (10 of 17 games), 36.9% opportunity
    share on a real Denver committee -- correctly still classifiable
    (games_played=10 clears the >=4 floor even though he missed 7 games)."""
    row = _row("J.K. Dobbins")
    assert row["games_played"] >= 4, "expected enough real games to classify, not UNCONFIRMED on sample size alone"
    assert row["archetype_primary"] == "committee_back", f"expected committee_back, got {row['archetype_primary']}"
    print(f"PASS -- Dobbins: opp_share={row['opportunity_share']:.3f}, games={row['games_played']:.0f}, "
          f"archetype={row['archetype_primary']}")


def test_irving_and_spears_real_data_report():
    """Spec explicitly flags these two as 'likely misclassified under the
    old system' and asks for a direct re-verification, not a specific
    expected tag. Reported against real 2025 data: both come back with
    real, defensible, non-generic usage-tier primaries (Irving in Tampa
    Bay's real committee with Rachaad White; Spears -- real 29.0% share,
    below Handcuff's own <0.20 ceiling but nearest to it by distance
    (0.150) given his real Tennessee teammate Tony Pollard's dominant
    67.2% share and Spears' own real depth_chart_rank=2 -- a genuinely
    ambiguous real case, not a meaningless catch-all) -- neither falls to
    unconfirmed.

    Post RB-taxonomy-migration (2026-08-15): Spears previously landed
    receiving_back as PRIMARY under the old 6-way system (a role/trait
    masquerading as a usage tier, the exact bug the migration fixes) --
    this test intentionally does not assert a specific tier for him since
    the spec itself only asked for "not a meaningless catch-all," and his
    real profile is genuinely a close, defensible call between Handcuff
    and Committee."""
    irving = _row("Bucky Irving")
    spears = _row("Tyjae Spears")
    assert irving["archetype_primary"] not in ("unconfirmed",), (
        f"expected a real, specific archetype for Irving, got {irving['archetype_primary']}"
    )
    assert spears["archetype_primary"] not in ("unconfirmed",), (
        f"expected a real, specific archetype for Spears, got {spears['archetype_primary']}"
    )
    print(f"REPORT -- Bucky Irving: opp_share={irving['opportunity_share']:.3f}, "
          f"archetype={irving['archetype_primary']}")
    print(f"REPORT -- Tyjae Spears: opp_share={spears['opportunity_share']:.3f}, "
          f"archetype={spears['archetype_primary']}")


def test_committee_back_floor_lowered():
    """COMMITTEE_BACK_OPPORTUNITY_SHARE_FLOOR (rb_archetypes.py) was lowered
    from 0.30 to 0.20 to close a real gap that stranded well-established
    committee backs at 'unconfirmed' purely because their real backfield
    share fell between HANDCUFF's <0.20 ceiling and the old 0.30 floor.

    Real 2025 traced effects: Tuten (21.7%), Skattebo (26.9%), and
    Singletary (27.9%) all clear the new 0.20 floor, each with a real
    teammate/gap/games profile that also satisfies the rest of
    _meets_conditions()'s COMMITTEE_BACK branch, so they reclassify from
    unconfirmed to committee_back via a clean condition match.

    Chris Rodriguez (29.0% share, clears the floor) does NOT clear the
    branch's separate best_teammate>=0.25 gate cleanly (his own best
    teammate, Tuten, sits at 21.7%) -- at the time this test was first
    written, that meant he stayed unconfirmed. Since then, nearest-fit
    classification shipped (claude_code_plan_qb_context_risk_taxonomy.pdf
    item 4: "never resolve to unconfirmed" when nothing cleanly clears),
    so Rodriguez now resolves via mean-shortfall distance instead --
    verified directly, not assumed: his real opportunity_share (0.29) sits
    comfortably inside Committee's band (0 shortfall there), and even with
    the teammate-relative conditions falling short, Committee's overall
    mean distance (0.044) is decisively the closest of the 3 real
    usage-tier primaries (next-closest: bellcow at 0.473, handcuff at
    0.519 -- re-verified after the RB taxonomy migration removed Goal-
    Line/Explosive/Receiving from primary competition entirely). This is
    the correct, intended nearest-fit outcome, not a regression -- Chris
    Rodriguez legitimately looks like a real committee back by volume,
    just without a clean qualifying teammate comparison."""
    tuten = _row("Bhayshul Tuten")
    skattebo = _row("Cam Skattebo")
    singletary = _row("Devin Singletary")
    rodriguez = _row("Chris Rodriguez")
    for label, row in (("Tuten", tuten), ("Skattebo", skattebo), ("Singletary", singletary)):
        assert 0.20 <= row["opportunity_share"] < 0.30, f"{label}: got {row['opportunity_share']}"
        assert row["archetype_primary"] == "committee_back", (
            f"{label}: expected committee_back, got {row['archetype_primary']}"
        )
    assert rodriguez["archetype_primary"] == "committee_back", (
        f"Rodriguez: expected committee_back (nearest-fit, real opportunity_share "
        f"squarely in Committee's band), got {rodriguez['archetype_primary']}"
    )
    print(f"PASS -- Tuten: opp_share={tuten['opportunity_share']:.3f} committee_back; "
          f"Skattebo: opp_share={skattebo['opportunity_share']:.3f} committee_back; "
          f"Singletary: opp_share={singletary['opportunity_share']:.3f} committee_back; "
          f"Rodriguez: opp_share={rodriguez['opportunity_share']:.3f} committee_back (nearest-fit)")


def test_primary_is_always_usage_tier():
    """RB taxonomy migration (explicit user correction, 2026-08-15): primary
    must always be a usage tier -- Bellcow / Committee / Handcuff /
    Unconfirmed ("No Role") -- never a role/trait like Goal-Line,
    Explosive, or Receiving. Full-pool guard, not just the two named cases
    (Hubbard/Stevenson) already covered above -- confirms the taxonomy
    fix holds across every real RB in the current build, not just the
    ones this test file happens to name."""
    board = _board()
    allowed = {"bellcow", "committee_back", "handcuff", "unconfirmed"}
    bad = set(board["archetype_primary"].unique()) - allowed
    assert not bad, f"expected primary values only in {allowed}, found: {bad}"
    counts = board["archetype_primary"].value_counts().to_dict()
    print(f"PASS -- full pool ({len(board)} real RBs): primary distribution {counts}, "
          f"no role/trait values leaked into primary")


def test_traded_player_nearest_fit_intrinsic_only():
    """claude_code_plan_qb_context_risk_taxonomy.pdf item 4: Kenneth Walker
    (KC) is the named test case for nearest-fit. His real 2025 usage (51%
    opportunity_share, 257 touches, 17 games -- all real Seattle
    production) never changed, but a real team correction (SEA->KC, March
    2026 free agency) made his COMMITTEE_BACK/HANDCUFF teammate-relative
    conditions compare him against KC's real CURRENT backfield -- players
    he never shared a season with. Naive nearest-fit (verified directly
    before this fix shipped) picked explosive_back (mean distance 0.069)
    over committee_back (0.306, dragged down entirely by the two now-
    meaningless teammate terms) -- the wrong answer for a real reason.

    TEAM_CHANGED_PLAYERS (build_rb_archetypes.py) + intrinsic_only_distance
    (rb_archetypes.py's classify_primary()/_archetype_distance()) fixes
    this: for a small, explicit, dated list of real team-changers, only
    each archetype's own-stat conditions count toward distance --
    Committee's real opportunity_share condition alone gives it a clean
    0.0 distance, decisively correct."""
    row = _row("Kenneth Walker")
    assert row["team"] == "KC", f"expected KC (real March 2026 signing), got {row['team']}"
    assert 0.45 <= row["opportunity_share"] <= 0.55, f"expected real ~51% share, got {row['opportunity_share']}"
    assert row["archetype_primary"] == "committee_back", (
        f"expected committee_back (intrinsic-only nearest-fit), got {row['archetype_primary']}"
    )
    print(f"PASS -- Kenneth Walker: team={row['team']}, opp_share={row['opportunity_share']:.3f}, "
          f"archetype=committee_back (intrinsic-only nearest-fit, real team-changer)")


def main() -> int:
    test_gibbs_bellcow()
    test_henry_bellcow_two_down()
    test_kyren_williams_bellcow_three_down()
    test_stevenson_henderson_committee()
    test_jordan_mason_committee()
    test_dobbins_committee()
    test_irving_and_spears_real_data_report()
    test_committee_back_floor_lowered()
    test_primary_is_always_usage_tier()
    test_traded_player_nearest_fit_intrinsic_only()
    print("\nALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
