"""build_tier_calibration() formula-correction regression tests
(task_4d5e2bb0 follow-up, 2026-08-17).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_tier_calibration
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.draft_analysis import (  # noqa: E402
    _TIER_EDGES,
    TIER_FACTOR_FLOOR,
    build_tier_calibration,
    load_realized_tier_vor,
)


def test_a_premium_tier_always_normalizes_to_one():
    """Tier 1-3 equals exactly 1.0 for every position, by construction.

    Under the bounded normalization (2026-08-21):

        factor = m + (1 - m) * (VOR_tier - VOR_min) / (VOR_1-3 - VOR_min)

    tier 1-3's numerator equals its denominator, so it evaluates to
    m + (1 - m) = 1.0 regardless of the data -- the same invariant the
    previous ratio formula guaranteed, preserved deliberately through the
    formula change. This is the literal bug it protects against: an earlier
    version divided by each position's own CURRENT-SEASON projected_VOR,
    which produced a real cross-position premium-tier discount (QB 0.23,
    RB 0.57) that neither of the two documented formulas can produce.

    Also a live regression guard on replacement-rank resolution: QB
    silently vanished from the calibration when _position_replacement_rank()
    read roster_settings before session state was seeded, resolving every
    rank to 1 and leaving QB's tier 1-3 VOR at or below zero."""
    load_realized_tier_vor.cache_clear()
    cal = build_tier_calibration()
    assert cal, "expected a non-empty real calibration table (positional_tier_curve.csv missing?)"
    positions_seen = set()
    for (position, tier), factor in cal.items():
        if tier == "1-3":
            assert factor == 1.0, f"expected {position} tier 1-3 factor == 1.0 by construction, got {factor}"
            positions_seen.add(position)
    assert positions_seen == {"QB", "RB", "WR", "TE"}, (
        f"expected all four scorable positions to have a real tier 1-3 anchor, got {positions_seen}"
    )
    print(f"A PASS -- premium tier (1-3) factor == 1.0 for all real positions with data: {sorted(positions_seen)}")


def test_b_lower_tiers_still_carry_real_within_position_shape():
    """The fix does not flatten everything to 1.0 -- real, measured
    within-position decay still applies below tier 1-3 (the part
    build_positional_tier_curve_v1.py's docstring says this SHOULD
    capture: e.g. TE1-3 outperforming TE4-6/TE7-9). Confirmed here
    against the real, current 2015-2025 realized-outcomes table rather
    than asserting a hardcoded number, since the underlying CSV can be
    regenerated.

    Asserts the bounded normalization, which replaced the raw ratio
    (2026-08-21). The ratio was unstable wherever a tier's realized VOR
    approached zero -- which is where most non-elite tiers sit -- so a cell
    moving from +1.1 to -2.6, inside sampling noise at ~25 distinct players,
    flipped the factor from 0.02 to a floored 0.05 and annihilated the
    whole TE4-15 range."""
    load_realized_tier_vor.cache_clear()
    cal = build_tier_calibration()
    realized = load_realized_tier_vor()
    checked = 0
    for position in ("QB", "RB", "WR", "TE"):
        base = realized.get((position, "1-3"))
        if base is None or base <= 0:
            continue
        fitted = [
            realized[(position, label)]
            for label, _lo, _hi in _TIER_EDGES
            if (position, label) in realized
        ]
        floor_vor = min(fitted)
        span = base - floor_vor
        assert span > 0, f"{position}: degenerate tier span, no shape to fit"

        for label, _lo, _hi in _TIER_EDGES:
            realized_vor = realized.get((position, label))
            factor = cal.get((position, label))
            if realized_vor is None or factor is None:
                continue
            expected = TIER_FACTOR_FLOOR + (1.0 - TIER_FACTOR_FLOOR) * (
                (realized_vor - floor_vor) / span
            )
            expected = min(max(expected, TIER_FACTOR_FLOOR), 1.0)
            assert abs(factor - expected) < 1e-6, (
                f"expected {position} {label} factor {expected:.4f} (realized {realized_vor:.1f}, "
                f"floor {floor_vor:.1f}, base {base:.1f}), got {factor:.4f}"
            )
            # The whole point of the change: nothing gets switched off.
            assert factor >= TIER_FACTOR_FLOOR - 1e-9, (
                f"{position} {label} factor {factor} fell below the floor {TIER_FACTOR_FLOOR}"
            )
            checked += 1

    assert checked >= 20, f"expected to check a real number of cells, only saw {checked}"
    print(f"B PASS -- {checked} lower-tier factors match the bounded normalization exactly, "
          f"none below the {TIER_FACTOR_FLOOR} floor")


def test_c_no_cross_position_leakage_at_premium_tier():
    """Direct regression for the bug this fix targets: before the fix,
    QB's real tier-1-3 factor (0.2288) and RB's (0.5707) differed sharply
    because the old formula divided by each position's own CURRENT
    projected_VOR, importing QB's much higher raw scoring scale. Since the
    corrected formula takes no current-season PROJECTION input at all (see
    build_tier_calibration()'s signature -- zero-argument), there is no
    mechanism left for a current-season scale difference between positions
    to reach tier 1-3's factor.

    Note the boundary this test does NOT claim (revised 2026-08-21): the
    calibration now does depend on the active league's replacement RANKS,
    which select which historical fit to read. That is a roster-format
    input, not a current-season one -- it picks among fixed 2015-2025
    measurements rather than importing this year's numbers -- so the
    anti-leakage property still holds. Tier 1-3 remains 1.0 for every
    position under any roster configuration, which is what this asserts."""
    import inspect
    sig = inspect.signature(build_tier_calibration)
    assert len(sig.parameters) == 0, (
        f"expected build_tier_calibration() to take no arguments (pure function of the realized-outcomes "
        f"table, no current-season projection/baseline input), got parameters: {list(sig.parameters)}"
    )
    load_realized_tier_vor.cache_clear()
    cal = build_tier_calibration()
    tier13 = {pos: factor for (pos, tier), factor in cal.items() if tier == "1-3"}
    assert len(set(tier13.values())) == 1, (
        f"expected identical tier-1-3 factors across positions (no cross-position leakage possible under "
        f"the corrected formula), got {tier13}"
    )
    print(f"C PASS -- zero-argument signature confirmed, all positions' tier-1-3 factors identical: {tier13}")


def main() -> int:
    test_a_premium_tier_always_normalizes_to_one()
    test_b_lower_tiers_still_carry_real_within_position_shape()
    test_c_no_cross_position_leakage_at_premium_tier()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
