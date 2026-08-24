"""TE Archetype System v1 spec tests (plan_te_archetypes.pdf).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_te_archetypes
"""

from __future__ import annotations

from draftkit.te_archetypes import (
    ELITE_REDZONE_TARGET_SHARE_FLOOR,
    ELITE_TARGET_SHARE_FLOOR,
    RECEIVING_TARGET_SHARE_FLOOR,
    SAMPLE_FLOOR_GAMES,
    SAMPLE_FLOOR_SNAP_SHARE,
    TEArchetype,
    TERoleProfile,
    classify_primary,
    classify_role_profile,
)
from draftkit.scripts.build_te_archetypes import build_te_archetypes


def _board():
    return build_te_archetypes()


def _row(board, name):
    hit = board[board["player_name"] == name]
    assert not hit.empty, f"{name} not found in real te_archetypes board"
    return hit.iloc[0]


def test_a_receiving_anchors():
    board = _board()
    for name in ("Trey McBride", "Brock Bowers", "George Kittle", "Travis Kelce"):
        row = _row(board, name)
        assert row["te_archetype_primary"] == "receiving_te", (name, row["te_archetype_primary"])
    print("A PASS -- McBride/Bowers/Kittle/Kelce -> receiving_te")


def test_b_blocking_anchors():
    """Adam Trautman/Drew Sample/John Bates -- real committed blockers,
    corrected snap-share numbers (season=CURRENT_SEASON row's
    prior_snap_share, not RECENT_SEASON's -- see build_te_archetypes.py's
    module docstring for the real lag bug this replaces)."""
    board = _board()
    for name in ("Adam Trautman", "Drew Sample", "John Bates"):
        row = _row(board, name)
        assert row["te_archetype_primary"] == "blocking_te", (name, row["te_archetype_primary"])
    print("B PASS -- Trautman/Sample/Bates -> blocking_te")


def test_c_conklin_and_moreau_are_unconfirmed_not_blocking():
    """Real regression lock for the exact bug caught before this shipped:
    Tyler Conklin (real 2025: 5.0% target share, 14.8% snap share, 6 games)
    and Foster Moreau (4.1%/22.5%/7 games) were originally reported as
    Kittle-level-snap blocking anchors due to a lagged-column misread.
    Corrected, both are genuinely thin-sample/marginal-role players on
    BOTH axes -- SAMPLE_FLOOR_SNAP_SHARE (0.30) must exclude them from
    confident classification entirely, not read them as blocking_te."""
    board = _board()
    conklin = _row(board, "Tyler Conklin")
    moreau = _row(board, "Foster Moreau")
    assert conklin["te_archetype_primary"] == "unconfirmed", conklin["te_archetype_primary"]
    assert moreau["te_archetype_primary"] == "unconfirmed", moreau["te_archetype_primary"]
    assert conklin["snap_share"] < SAMPLE_FLOOR_SNAP_SHARE
    assert moreau["snap_share"] < SAMPLE_FLOOR_SNAP_SHARE
    print(f"C PASS -- Conklin (snap={conklin['snap_share']:.3f}) and Moreau "
          f"(snap={moreau['snap_share']:.3f}) both correctly Unconfirmed, not blocking_te")


def test_d_target_share_is_exhaustive_no_dead_zone():
    """Same structural guarantee QB's classify_primary() has: a single real
    scalar banded into 3 exhaustive, non-overlapping ranges cannot produce
    a dead zone. Spot-check a real mid-range value lands Balanced, not
    Unconfirmed, when sample floors are cleared."""
    assert classify_primary(0.10, 10, 0.50) == TEArchetype.BALANCED
    assert classify_primary(RECEIVING_TARGET_SHARE_FLOOR, 10, 0.50) == TEArchetype.RECEIVING_TE
    assert classify_primary(0.02, 10, 0.50) == TEArchetype.BLOCKING_TE
    assert classify_primary(0.20, SAMPLE_FLOOR_GAMES - 1, 0.50) == TEArchetype.UNCONFIRMED
    assert classify_primary(0.20, 10, SAMPLE_FLOOR_SNAP_SHARE - 0.01) == TEArchetype.UNCONFIRMED
    print("D PASS -- exhaustive banding confirmed: no real value falls through unclassified "
          "once both sample floors clear")


def test_e_full_pool_no_invalid_primary_values():
    board = _board()
    allowed = {a.value for a in TEArchetype}
    bad = set(board["te_archetype_primary"].unique()) - allowed
    assert not bad, f"unexpected te_archetype_primary values: {bad}"
    counts = board["te_archetype_primary"].value_counts()
    print(f"E PASS -- full pool ({len(board)} real TEs): {counts.to_dict()}")


def test_f_target_share_bug_fix_regression():
    """Real regression lock for the target_share denominator bug caught
    verifying Fannin's number against SumerSports/Muffed: the shared
    load_target_share_rate() reconstructs team pass-attempts-per-game from
    each player's SEASON-ENDING recent_team, which silently breaks for any
    player traded mid-season (real 2025 case: Joe Flacco CLE->CIN after
    Burrow's injury). Fixed TE-only via load_te_target_share_corrected()
    in build_te_archetypes.py. Fannin must land near the real, independently
    verified ~20-22% range, not the old buggy 28.6%. Njoku/Fant/Hooper are
    real, explainable reclassifications caused by the same fix (see that
    function's docstring) -- locked in so they don't silently regress."""
    board = _board()
    fannin = _row(board, "Harold Fannin")
    assert 0.19 <= fannin["target_share"] <= 0.24, fannin["target_share"]
    assert fannin["te_archetype_primary"] == "receiving_te", fannin["te_archetype_primary"]

    njoku = _row(board, "David Njoku")
    fant = _row(board, "Noah Fant")
    hooper = _row(board, "Austin Hooper")
    assert njoku["te_archetype_primary"] == "balanced", njoku["te_archetype_primary"]
    assert fant["te_archetype_primary"] == "balanced", fant["te_archetype_primary"]
    assert hooper["te_archetype_primary"] == "balanced", hooper["te_archetype_primary"]
    print(f"F PASS -- Fannin corrected to {fannin['target_share']:.3f} (was buggy 0.286); "
          "Njoku/Fant/Hooper correctly balanced, not receiving_te/blocking_te")


def test_g_role_profile_exhaustive():
    """classify_role_profile() is a 2-branch OR-gate on two real, independent
    scalars -- always resolves, no dead zone possible."""
    assert classify_role_profile(ELITE_TARGET_SHARE_FLOOR, 0.0) == TERoleProfile.ELITE
    assert classify_role_profile(0.0, ELITE_REDZONE_TARGET_SHARE_FLOOR) == TERoleProfile.ELITE
    assert classify_role_profile(ELITE_TARGET_SHARE_FLOOR - 0.001, ELITE_REDZONE_TARGET_SHARE_FLOOR - 0.001) == \
        TERoleProfile.COMPLEMENTARY
    print("G PASS -- classify_role_profile() exhaustive, both paths independently sufficient")


def test_h_role_profile_anchors():
    """Real, corrected anchors (plan_te_role_profile_elite.pdf, corrected
    after the target_share bug fix). McBride/Bowers clear both axes.
    Andrews/Henry/Ferguson are the clean redzone-only regression case --
    none of the three clear the volume floor alone. Fannin/Goedert clear
    both, but only just (grazing the volume floor, not decisive -- real
    numbers after the fix, not the original plan's buggy 28.6% for Fannin).
    LaPorta/Otton are the real complementary contrast case: comfortably
    receiving_te by volume, both near the bottom of the pool on redzone
    share."""
    board = _board()
    for name in ("Trey McBride", "Brock Bowers"):
        row = _row(board, name)
        assert row["te_role_profile"] == "elite", (name, row["te_role_profile"])

    for name in ("Mark Andrews", "Hunter Henry", "Jake Ferguson"):
        row = _row(board, name)
        assert row["target_share"] < ELITE_TARGET_SHARE_FLOOR, (name, row["target_share"])
        assert row["redzone_target_share"] >= ELITE_REDZONE_TARGET_SHARE_FLOOR, (name, row["redzone_target_share"])
        assert row["te_role_profile"] == "elite", (name, row["te_role_profile"])

    for name in ("Harold Fannin", "Dallas Goedert"):
        row = _row(board, name)
        assert row["te_role_profile"] == "elite", (name, row["te_role_profile"])

    for name in ("Sam LaPorta", "Cade Otton"):
        row = _row(board, name)
        assert row["te_role_profile"] == "complementary", (name, row["te_role_profile"])

    print("H PASS -- McBride/Bowers elite (both axes); Andrews/Henry/Ferguson elite "
          "(redzone-only, volume floor not cleared); Fannin/Goedert elite (both, grazing); "
          "LaPorta/Otton complementary")


def main() -> int:
    test_a_receiving_anchors()
    test_b_blocking_anchors()
    test_c_conklin_and_moreau_are_unconfirmed_not_blocking()
    test_d_target_share_is_exhaustive_no_dead_zone()
    test_e_full_pool_no_invalid_primary_values()
    test_f_target_share_bug_fix_regression()
    test_g_role_profile_exhaustive()
    test_h_role_profile_anchors()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
