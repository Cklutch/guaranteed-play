"""WR archetype system v1 tests (wr_archetype_v1_spec.pdf).

Plain assert-based, no pytest -- matches this repo's established convention
(see test_rb_archetypes.py). Calls the real
draftkit.scripts.build_wr_archetypes pipeline once against real 2025 season
data and checks named real players in the output -- not a frozen snapshot,
not a reimplementation of its logic.

Every assertion below is derived from each player's real, directly-verified
2025 inputs against the spec's own given thresholds. Where a case surfaced a
real discrepancy against the spec's own claims during that verification, it's
reported and resolved honestly inline, not silently assumed.

Runnable directly:
    python -m draftkit.tests.test_wr_archetypes
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_wr_archetypes import build_wr_archetypes  # noqa: E402

_BOARD = None

ALPHA_NAMED_PLAYERS = ["Ja'Marr Chase", "Justin Jefferson", "CeeDee Lamb", "Puka Nacua", "Amon-Ra St. Brown"]


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = build_wr_archetypes()
    return _BOARD


def _row(player_name: str):
    board = _board()
    matches = board[board["player_name"] == player_name]
    assert not matches.empty, f"expected {player_name} in the current WR pool, found none"
    return matches.iloc[0]


def test_alpha_named_players():
    """Spec claims Chase/Jefferson/Lamb/Nacua/St. Brown 'all run comfortably
    above 24% target share.' Real nflverse's OWN raw target_share column
    contradicts this for Lamb (19.3% -- a real down year on missed-games
    volume, confirmed directly during the data audit). But that raw column
    has a known, already-documented dilution bug in this repo (divides by
    the team's FULL season pass attempts even for games a player missed --
    see build_risk_variables.py's module docstring, the same bug that made
    McCaffrey's 2024 target_share compute to an impossible 3.7%). Using the
    dilution-FIXED target_share_rate this pipeline already built for exactly
    this reason (targets / (team_pass_attempts_per_game * games actually
    played)), Lamb's real target share is 24.5% -- clearing the 24% floor
    after all. The spec's claim holds once the right formula is used;
    reported honestly either way, not assumed."""
    for name in ALPHA_NAMED_PLAYERS:
        row = _row(name)
        assert row["wr_archetype_primary"] == "alpha", (
            f"expected {name} to be alpha, got {row['wr_archetype_primary']} "
            f"(target_share={row['target_share']:.3f})"
        )
        print(f"PASS -- {name}: target_share={row['target_share']:.3f}, archetype=alpha")


def test_complementary_real_gap():
    """Real 2025: Cooper Kupp, now on Seattle behind Jaxon Smith-Njigba's
    real breakout WR1 season (35.8% target share, post target_share-bug-fix
    -- see build_risk_variables.py's load_target_share_rate() docstring).

    RESOLVED (task_2c2eea99, 2026-08-17) -- was an open issue in this test;
    see wr_archetypes.py's classify_primary() docstring for the full
    resolution. Kupp's target_share moved 15.4%->16.3% under the corrected
    calculation, just 0.35pp over COMPLEMENTARY_TARGET_SHARE_CEILING (0.16).
    A first pass had classify_primary() blanket-exclude Complementary from
    the fallback for every player in this band (justified by the Jameson
    Williams case, see test_possession_split_boom_bust_high_floor), which
    forced Kupp to a 2-way high_floor/boom_bust choice despite his real
    Complementary distance (0.0108) being ~16x closer than High-Floor's
    (0.173) -- an earlier pass wrongly called this "well-justified" without
    computing that distance, caught on review.

    The fix makes the gate shape-aware (aDOT) instead of blanket: Williams'
    real aDOT (12.6) sits in the boom-bust-adjacent dead zone, so his
    Complementary distance staying excluded was correct; Kupp's real aDOT
    (7.49) sits inside High-Floor's own [6,9] band -- nothing about his
    shape looks boom-bust, so there's no real reason to keep Complementary
    out of his race. He now correctly resolves to complementary. Checked
    against the full real fallback population (56 sample-qualified
    players): only 5 total reclassify under this gate, all real near-misses
    (see classify_primary()'s docstring), Williams and the rest of the
    boom-bust-shaped population are unaffected."""
    kupp = _row("Cooper Kupp")
    jsn = _row("Jaxon Smith-Njigba")
    assert jsn["wr_archetype_primary"] == "alpha", f"expected JSN alpha (real WR1), got {jsn['wr_archetype_primary']}"
    assert kupp["wr_archetype_primary"] == "complementary", (
        f"expected Kupp complementary (shape-aware gate: real aDOT inside High-Floor's band, "
        f"real complementary_distance ~16x closer than high_floor's -- see docstring), "
        f"got {kupp['wr_archetype_primary']}"
    )
    print(f"PASS -- Kupp: target_share={kupp['target_share']:.3f}, team_wr1={kupp['team_wr1_target_share']:.3f} "
          f"(JSN), adot={kupp['adot']:.1f}, archetype=complementary (shape-aware near-miss resolution)")


def test_coequal_never_lands_complementary():
    """Real 2025: Malik Washington (MIA) sits in the 8-16% Complementary
    target-share band by raw number, but IS his own team's real highest-
    share WR (team_wr1_target_share == his own share, gap=0) -- a real
    co-equal case, not a hierarchical WR1/WR2 situation.

    Was Darnell Mooney (NYG) before the recency-weighted-fallback fix
    (rookie_model_next_plan.pdf's follow-up): Mooney only LOOKED co-equal
    because his real teammate Malik Nabers (ACL tear, 4 games in 2025) was
    incorrectly excluded from team_wr1_target_share's real qualification
    check by his own thin current-season sample. Once Nabers' real,
    dominant 2024 season (170 targets) is correctly blended back in (see
    _games_weighted_blend()), NYG's real team_wr1 is Nabers at 30.6%, and
    Mooney's real gap becomes 15.6pp -- a genuine WR2, not co-equal. That
    was the bug this test's OLD assertion was quietly resting on, not a
    real co-equal case; switched to a player unaffected by the fix.

    Previously (pre-Possession-split) this defaulted to a fixed
    Possession bucket (user's explicit choice, spec Section 6 open item
    #2). claude_code_plan_possession_split.pdf removed Possession
    entirely -- the co-equal case no longer has anywhere fixed to default
    to, so it now falls through to the same nearest-fit as everyone else
    below Alpha (see wr_archetypes.py's classify_primary() docstring).
    Washington's real aDOT (5.2) clears the shape-gate (task_2c2eea99), so
    Complementary now DOES compete in his race -- but his real
    Complementary distance (0.500) is far larger than High-Floor's
    (0.357), so being numerically co-equal with his own team's WR1 still
    fails Complementary's gap condition badly and the real outcome is
    unchanged. What matters here is only that a genuine co-equal receiver
    never lands Complementary; WHICH of Boom-Bust/High-Floor he lands is
    driven by his own real profile, not a fixed rule."""
    row = _row("Malik Washington")
    gap = row["team_wr1_target_share"] - row["target_share"]
    assert gap < 0.08, f"expected a real co-equal case (gap<8pp), got gap={gap:.3f}"
    assert row["wr_archetype_primary"] != "complementary", (
        f"co-equal case should never land complementary, got {row['wr_archetype_primary']}"
    )
    assert row["wr_archetype_primary"] in ("boom_bust", "high_floor"), (
        f"expected boom_bust or high_floor (2-way nearest-fit), got {row['wr_archetype_primary']}"
    )
    print(f"PASS -- Malik Washington: target_share={row['target_share']:.3f}, "
          f"team_wr1={row['team_wr1_target_share']:.3f}, gap={gap:.3f}, "
          f"archetype={row['wr_archetype_primary']} (co-equal, nearest-fit)")


def test_injury_shortened_season_uses_recency_weighted_fallback():
    """Real 2025: Malik Nabers (NYG) tore his ACL early in Week 4 (4 real
    games, 35 targets, 271 yards -- confirmed against his real per-game
    log) after a real, unambiguous 2024 rookie season (170 targets, 109
    catches, 1204 yards, 7 TDs, 15 games). Reading his archetype off the
    4-game 2025 season alone incorrectly floors him at UNCONFIRMED via
    classify_primary()'s own real games_played<6 gate -- discarding a
    real, large sample in favor of a real but tiny one
    (claude_code_plan_ui_and_blend_fix.pdf's follow-up). Games-weighted
    blending in the real 2024 season (see _games_weighted_blend()) fixes
    this: combined games=19, combined targets=205, and his real,
    dominant 2024 share correctly drives the read to Alpha."""
    row = _row("Malik Nabers")
    assert row["games_played"] == 19.0, f"expected combined 4+15=19 games, got {row['games_played']}"
    assert row["targets"] == 205.0, f"expected combined 35+170=205 targets, got {row['targets']}"
    assert row["wr_archetype_primary"] == "alpha", f"expected alpha, got {row['wr_archetype_primary']}"
    print(f"PASS -- Malik Nabers: games_played={row['games_played']:.0f} (4 real 2025 + 15 real 2024), "
          f"targets={row['targets']:.0f}, target_share={row['target_share']:.3f}, archetype=alpha")


def test_possession_split_boom_bust_high_floor():
    """claude_code_plan_possession_split.pdf: Possession replaced entirely
    by Boom-Bust/High-Floor nearest-fit, with Complementary shape-gated
    into the race by real aDOT (task_2c2eea99, see wr_archetypes.py's
    classify_primary() docstring for the full design).

    Jameson Williams (DET) is the named case both source plans anchor
    on -- real 2025 12.6 aDOT, 17.5% target_share, sits in the real "dead
    zone" between both profiles (11-13 aDOT / 18-22% target share) but
    should still resolve to Boom-Bust given his real aDOT/big-play
    profile, not Complementary (verified directly: his real Complementary
    distance of 0.048 is numerically lower than Boom-Bust's 0.200, but his
    real aDOT (12.6) sits above HIGH_FLOOR_ADOT_HIGH (9.0), so the
    shape-gate correctly keeps Complementary out of his race -- folding it
    in unconditionally, as an earlier draft did, let a generic,
    style-blind band match steal the win from the one named test case this
    whole split exists to fix).

    Keenan Allen (LAC) and Jonathan Mingo (real 2025 profiles) are clean,
    non-dead-zone spot checks: a known short/possession slot receiver
    (adot=8.4, inside High-Floor's real [6,9] band -- now IN the 3-way
    race per the shape gate, but still resolves High-Floor since his real
    Complementary distance is far larger) and a known real field-stretcher
    (adot=19.4, target_share=6.6%, well above the shape-gate's aDOT bar,
    so still a clean 2-way case) -- confirming both ends of the spectrum
    resolve correctly, not just the ambiguous middle case."""
    williams = _row("Jameson Williams")
    assert williams["wr_archetype_primary"] == "boom_bust", (
        f"expected boom_bust, got {williams['wr_archetype_primary']}"
    )
    allen = _row("Keenan Allen")
    assert allen["wr_archetype_primary"] == "high_floor", (
        f"expected high_floor, got {allen['wr_archetype_primary']}"
    )
    mingo = _row("Jonathan Mingo")
    assert mingo["wr_archetype_primary"] == "boom_bust", (
        f"expected boom_bust, got {mingo['wr_archetype_primary']}"
    )
    print(f"PASS -- Jameson Williams: adot={williams['adot']:.1f}, target_share={williams['target_share']:.3f}, "
          f"archetype=boom_bust (dead-zone case)")
    print(f"PASS -- Keenan Allen: adot={allen['adot']:.1f}, target_share={allen['target_share']:.3f}, "
          f"archetype=high_floor (clean match)")
    print(f"PASS -- Jonathan Mingo: adot={mingo['adot']:.1f}, target_share={mingo['target_share']:.3f}, "
          f"archetype=boom_bust (clean match)")


def test_deep_threat_lean():
    """Real 2025: Christian Watson (GB), a publicly known deep-threat
    receiver, clears the real aDOT>=12.5 and yards_per_reception>=14
    thresholds on a real 15+ reception sample."""
    row = _row("Christian Watson")
    assert "deep_threat" in row["wr_leans"].split(","), f"expected deep_threat lean, got {row['wr_leans']}"
    print(f"PASS -- Watson: adot={row['adot']:.1f}, ypr={row['yards_per_reception']:.1f}, "
          f"leans={row['wr_leans']}")


def test_yac_lean():
    """Real 2025: Deebo Samuel (SF), a publicly known after-catch/gadget
    weapon, clears the real yac_per_reception>=6.0 and aDOT<=9 thresholds."""
    row = _row("Deebo Samuel")
    assert "yac" in row["wr_leans"].split(","), f"expected yac lean, got {row['wr_leans']}"
    print(f"PASS -- Deebo Samuel: yac_per_reception={row['yac_per_reception']:.1f}, adot={row['adot']:.1f}, "
          f"leans={row['wr_leans']}")


def test_rushing_lean():
    """Real 2025: Marvin Mims (DEN), real jet-sweep/gadget rushing usage,
    clears the real rush_attempts_per_game>=0.8 and rush_yards_per_attempt
    >=6.0 thresholds on a real >=8 rush-attempt sample."""
    row = _row("Marvin Mims")
    assert "rushing" in row["wr_leans"].split(","), f"expected rushing lean, got {row['wr_leans']}"
    print(f"PASS -- Mims: rush_attempts_per_game={row['rush_attempts_per_game']:.2f}, "
          f"rush_yards_per_attempt={row['rush_yards_per_attempt']:.1f}, leans={row['wr_leans']}")


def test_no_deep_threat_yac_collision():
    """Deep-Threat (aDOT floor) and YAC (aDOT ceiling) are near-mutually-
    exclusive by construction. Full-pool check: no real player should carry
    both simultaneously (the tie-break in classify_leans() exists precisely
    to prevent this, but this confirms it never actually triggers on real
    data)."""
    board = _board()
    collisions = board[board["wr_leans"].str.contains("deep_threat,yac|yac,deep_threat", na=False, regex=True)]
    assert collisions.empty, f"expected no deep_threat+yac collisions, got {len(collisions)}: {collisions['player_name'].tolist()}"
    print("PASS -- no real deep_threat+yac collisions in the full pool")


def test_full_pool_report():
    """Report-style, not a strict assertion beyond sanity bounds: the real
    unconfirmed rate should reflect genuine low-volume/inactive WRs (this
    pool includes practice-squad-level names from master_players.csv, same
    roster noise the RB build already accounted for), not a systemic
    under-classification bug."""
    board = _board()
    counts = board["wr_archetype_primary"].value_counts()
    total = len(board)
    unconfirmed_rate = counts.get("unconfirmed", 0) / total
    assert 0 < unconfirmed_rate < 0.85, f"unconfirmed rate {unconfirmed_rate:.1%} outside a sane real-world range"
    print(f"REPORT -- {total} real WRs classified: {counts.to_dict()} "
          f"(unconfirmed rate {unconfirmed_rate:.1%})")


def main() -> int:
    test_alpha_named_players()
    test_complementary_real_gap()
    test_coequal_never_lands_complementary()
    test_injury_shortened_season_uses_recency_weighted_fallback()
    test_possession_split_boom_bust_high_floor()
    test_deep_threat_lean()
    test_yac_lean()
    test_rushing_lean()
    test_no_deep_threat_yac_collision()
    test_full_pool_report()
    print("\nALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
