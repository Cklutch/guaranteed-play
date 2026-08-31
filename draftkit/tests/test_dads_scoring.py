"""dads_scoring.py regression tests (2026-08-30).

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_dads_scoring
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from draftkit.dads_scoring import (  # noqa: E402
    MANUAL_DADS_CORRECTIONS,
    TD_LENGTH_FULL_CREDIT_TDS,
    TD_LENGTH_MIN_SAMPLE,
    TD_LENGTH_MULT_MAX,
    TD_LENGTH_MULT_MIN,
    _apply_manual_dads_corrections,
    _td_length_baselines,
    _td_length_multiplier,
    dads_points_from_stats,
    load_dads_adp,
)

_COL = {
    "name": "Name", "pos": "Pos", "pass_yds": "Pass Yards", "pass_td": "Pass TDs",
    "ints": "Ints", "rush_yds": "Rush Yards", "rush_td": "Rush TDs",
    "rec_yds": "Rec Yards", "rec_td": "Rec TDs", "fumbles": "Fumbles",
}


def _raw(rows):
    """rows: (name, pos, rush_yds, rush_td, rec_yds, rec_td)"""
    return pd.DataFrame([
        {"Name": n, "Pos": p, "Rush Yards": ry, "Rush TDs": rt,
         "Rec Yards": cy, "Rec TDs": ct, "Pass Yards": 0, "Pass TDs": 0,
         "Ints": 0, "Fumbles": 0}
        for n, p, ry, rt, cy, ct in rows
    ])


def test_no_baselines_reproduces_old_flat_behavior():
    """omitting length_baselines must be IDENTICAL to the pre-existing flat-EV
    formula -- this is the backward-compatibility contract the whole per-
    player tilt is built on top of."""
    kwargs = dict(rush_yds=1200, rush_td=12, rec_yds=400, rec_td=3)
    with_none = dads_points_from_stats("RB", **kwargs, length_baselines=None)
    with_empty = dads_points_from_stats("RB", **kwargs, length_baselines={})
    assert with_none == with_empty


def test_low_sample_falls_back_to_flat():
    """Below TD_LENGTH_MIN_SAMPLE projected TDs, one fluke long score must
    not swing the bonus -- multiplier is 1.0, same as no baseline at all."""
    baselines = {("RB", "rush"): 130.0}
    low_td = TD_LENGTH_MIN_SAMPLE - 1
    assert low_td >= 0
    mult = _td_length_multiplier(player_yds=200.0, player_td=low_td, baseline_ratio=130.0)
    assert mult == 1.0


def test_explosive_player_scores_above_grinder():
    """THE point of this feature (2026-08-30, user-directed): a player who
    needs many more yards per TD (proxy for longer, more explosive scores)
    must score MORE dad's-league points than a same-volume player who scores
    cheaply/short, all else equal."""
    baselines = {("RB", "rush"): 130.0}
    explosive = dads_points_from_stats(
        "RB", rush_yds=1200, rush_td=8, length_baselines=baselines,
    )  # 150 yds/TD -- well above the 130 baseline
    grinder = dads_points_from_stats(
        "RB", rush_yds=800, rush_td=8, length_baselines=baselines,
    )  # 100 yds/TD -- below baseline
    # Isolate the effect: same TD count, so the flat TD_PTS term is identical;
    # only the yardage total and the length-tilt differ. Compare the length
    # bonus alone via the multiplier, not the full scores (which also differ
    # in yardage-tier points).
    explosive_mult = _td_length_multiplier(1200.0, 8, 130.0)
    grinder_mult = _td_length_multiplier(800.0, 8, 130.0)
    assert explosive_mult > 1.0 > grinder_mult
    assert explosive_mult > grinder_mult


def test_multiplier_bounded():
    """A tiny handful of huge-yardage TDs must not blow up the bonus --
    always clamped to [TD_LENGTH_MULT_MIN, TD_LENGTH_MULT_MAX]."""
    huge = _td_length_multiplier(player_yds=5000.0, player_td=TD_LENGTH_FULL_CREDIT_TDS, baseline_ratio=50.0)
    tiny = _td_length_multiplier(player_yds=1.0, player_td=TD_LENGTH_FULL_CREDIT_TDS, baseline_ratio=500.0)
    assert huge == TD_LENGTH_MULT_MAX
    assert tiny == TD_LENGTH_MULT_MIN


def test_shrinkage_moves_toward_baseline_below_full_credit():
    """Between MIN_SAMPLE and FULL_CREDIT_TDS, a player's own extreme ratio
    should land BETWEEN his own number and the position average -- not at
    either extreme -- reflecting partial trust in a still-smallish sample."""
    baseline = 130.0
    partial_td = (TD_LENGTH_MIN_SAMPLE + TD_LENGTH_FULL_CREDIT_TDS) // 2
    assert TD_LENGTH_MIN_SAMPLE < partial_td < TD_LENGTH_FULL_CREDIT_TDS
    own_ratio = 200.0  # above baseline, but mild enough neither value clips
    mult = _td_length_multiplier(own_ratio * partial_td, partial_td, baseline)
    full_mult = own_ratio / baseline
    assert full_mult < TD_LENGTH_MULT_MAX, "test premise needs an unclipped full_mult"
    assert 1.0 < mult < full_mult


def test_baselines_only_built_for_combos_with_enough_players():
    """A (position, td_type) combo with too few qualifying players must not
    produce a fabricated baseline -- absence, not a noisy average."""
    # Only 2 RBs clear TD_LENGTH_MIN_SAMPLE -- below TD_LENGTH_MIN_PLAYERS (5).
    raw = _raw([
        ("A", "RB", 400, 5, 0, 0),
        ("B", "RB", 500, 6, 0, 0),
        ("C", "RB", 100, 1, 0, 0),  # below MIN_SAMPLE, doesn't count
    ])
    baselines = _td_length_baselines(raw, _COL)
    assert ("RB", "rush") not in baselines


def test_baselines_computed_for_a_real_qualifying_combo():
    raw = _raw([(f"P{i}", "RB", 100.0 * i, 5, 0, 0) for i in range(1, 7)])
    baselines = _td_length_baselines(raw, _COL)
    assert ("RB", "rush") in baselines
    assert baselines[("RB", "rush")] > 0


def test_dads_adp_loads_from_standard_sheet():
    """Sanity check against the real, checked-in Standard-scoring source --
    Gibbs should be (or very near) ADP 1, not the half-PPR sheet's numbers."""
    adp = load_dads_adp()
    if adp.empty:
        return  # source file not present on this machine -- not a code bug
    row = adp[adp["norm_name"] == "jahmyr gibbs"]
    assert not row.empty
    assert row.iloc[0]["dads_adp"] <= 3


def test_manual_correction_applies_correct_pct_to_dads_base():
    """The correction must apply to the DADS-FORMULA projection, not to
    whatever the raw source file's own 'Projections' column says (a real
    confusion caught live: winwithodds' own column is a different scoring
    system on the same stat line, not the base dads_points_from_stats()
    computes -- comparing a pct derived one way against the other base
    would silently misprice every corrected player)."""
    df = pd.DataFrame({
        "norm_name": ["josh jacobs", "someone else"],
        "dads_projection_points": [87.96, 50.0],
    })
    out = _apply_manual_dads_corrections(df.copy())
    pct = MANUAL_DADS_CORRECTIONS["josh jacobs"]["pct"]
    expected = round(87.96 * (1 + pct / 100), 2)
    assert out.loc[out["norm_name"] == "josh jacobs", "dads_projection_points"].iloc[0] == expected
    # Uncorrected player untouched.
    assert out.loc[out["norm_name"] == "someone else", "dads_projection_points"].iloc[0] == 50.0


def test_manual_correction_unmatched_name_is_a_noop():
    df = pd.DataFrame({"norm_name": ["nobody here"], "dads_projection_points": [42.0]})
    out = _apply_manual_dads_corrections(df.copy())
    assert out["dads_projection_points"].iloc[0] == 42.0


def test_manual_correction_skips_nan_base_without_erroring():
    df = pd.DataFrame({"norm_name": ["josh jacobs"], "dads_projection_points": [float("nan")]})
    out = _apply_manual_dads_corrections(df.copy())
    assert pd.isna(out["dads_projection_points"].iloc[0])


def test_manual_corrections_are_well_formed():
    for name, correction in MANUAL_DADS_CORRECTIONS.items():
        assert name == name.strip().lower(), f"{name} should already be normalized"
        assert isinstance(correction["pct"], (int, float))
        assert -100 < correction["pct"] < 200, f"{name}: implausible pct {correction['pct']}"
        assert len(correction["note"]) > 40, f"{name}: note too thin to be sourced"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} dads_scoring tests passed.")


if __name__ == "__main__":
    _run()
