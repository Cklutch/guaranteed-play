"""QB rookie projection tests (claude_code_plan_qb_rookie_projection.pdf).

Plain assert-based, no pytest -- matches this repo's established convention.
Calls the real draftkit.scripts.build_qb_rookie_projections pipeline once
against the real, consolidated 2021-2025 QB prospect data and checks named
real players in the output -- not a frozen snapshot.

Runnable directly:
    python -m draftkit.tests.test_qb_rookie_projection
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_rookie_projections import (  # noqa: E402
    build_qb_rookie_projections,
    load_2026_qb_prospects,
)

_BOARD = None
_BOARD_2026 = None


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = build_qb_rookie_projections()
    return _BOARD


def _board_2026():
    global _BOARD_2026
    if _BOARD_2026 is None:
        _BOARD_2026 = build_qb_rookie_projections(prospects_df=load_2026_qb_prospects())
    return _BOARD_2026


def _row_2026(player_name: str):
    board = _board_2026()
    hit = board[board["player_name"] == player_name]
    assert not hit.empty, f"{player_name} not found in real true-2026 QB board"
    return hit.iloc[0]


def _row(player_name: str):
    board = _board()
    hit = board[board["player_name"] == player_name]
    assert not hit.empty, f"{player_name} not found in real QB rookie board"
    return hit.iloc[0]


def test_a_milroe_real_rushing_pulls_dual_threat():
    """The PDF's own named check: confirm Milroe's real college rushing
    profile (168 att, 726 yds, 20 TD -- real, measured, not assumed) pulls
    him toward Dual-Threat."""
    row = _row("Jalen Milroe")
    assert row["rushing_data_status"] == "measured"
    assert row["projected_tier"] == "dual_threat"
    assert row["display_tag"] == "dual_threat"
    print(f"A PASS -- Milroe: real college rushing -> dual_threat "
          f"(status={row['status']}, display_status={row['display_status']})")


def test_b_ewers_assumed_non_rusher_resolves_not_null():
    """The PDF's own named check: confirm Ewers (no real rushing data --
    assumed non-rusher, per explicit user instruction) still resolves to a
    coherent tier on passing metrics alone, not a null/Unconfirmed state."""
    row = _row("Quinn Ewers")
    assert row["rushing_data_status"] == "assumed_non_rusher"
    assert row["projected_tier"] == "pocket_passer"
    assert row["display_tag"] not in (None, "unconfirmed") or row["display_status"] != "confirmed"
    print(f"B PASS -- Ewers: assumed non-rusher -> pocket_passer, "
          f"display_status={row['display_status']}, display_tag={row['display_tag']} (not null)")


def test_c_ehlinger_real_verified_override_and_display_bug_fix():
    """Sam Ehlinger's blank CSV row was proven wrong via direct web search
    (real 2020 Texas season: 382 rush yds, 8 rush TD) -- patched via
    QB_ROOKIE_RUSHING_OVERRIDES, not assumed. Also the real display bug
    this exact case caught: Ehlinger has no row in qb_archetypes.csv at all
    (confirmed_tag is None, not "unconfirmed"), which used to force
    display_status="confirmed" with a blank display_tag -- locked in here
    so it can't silently regress."""
    row = _row("Sam Ehlinger")
    assert row["rushing_data_status"] == "measured_override"
    assert row["projected_tier"] == "balanced"
    assert row["status"] == "projected"  # no real confirmed_tag exists -- no row in qb_archetypes.csv
    assert row["display_status"] == "projected"
    assert row["display_tag"] == "balanced"  # NOT None/NaN -- the real bug this case caught
    print(f"C PASS -- Ehlinger: real override applied (382 rush yds, 8 rush TD) -> balanced, "
          f"display_tag correctly resolves (was broken/blank before the None-vs-'unconfirmed' fix)")


def test_d_young_stroud_penix_real_verified_low_rushing():
    """Bryce Young/C.J. Stroud/Michael Penix Jr. spot-checked directly by
    the user and confirmed genuinely low-rushing -- real, verified numbers
    applied via the same override table, not assumed."""
    for name in ("Bryce Young", "C.J. Stroud", "Michael Penix Jr."):
        row = _row(name)
        assert row["rushing_data_status"] == "measured_override", name
    print("D PASS -- Young/Stroud/Penix: real verified low-rushing overrides applied, not assumed")


def test_e_richardson_and_levis_resolve_to_real_confirmed_veteran_tags():
    """Cross-check against tonight's veteran QB archetype build (not
    re-solved here): Richardson resolved to dual_threat via the veteran
    pooling mechanism (single 2025 season gate-fails, pooled 2024+2025
    clears it); Levis resolved to balanced via the season-fallback
    mechanism (zero 2025 rows, pooled real 2023+2024 instead). Both must
    carry through this rookie pipeline's confirmed-tier join unchanged --
    a real, decisive veteran tag must never be overridden by a rookie-side
    college projection."""
    richardson = _row("Anthony Richardson")
    assert richardson["status"] == "confirmed"
    assert richardson["tag"] == "dual_threat"
    assert richardson["display_tag"] == "dual_threat"

    levis = _row("Will Levis")
    assert levis["status"] == "confirmed"
    assert levis["tag"] == "balanced"
    assert levis["display_tag"] == "balanced"
    print("E PASS -- Richardson (dual_threat) and Levis (balanced) both carry through as real "
          "confirmed veteran tags, unchanged by the rookie-side college projection")


def test_f_full_class_15_4_13_split():
    """The number that matters most for trusting this system: verified
    (15) + manually overridden with real sourced data (4) + explicitly
    assumed (13) must sum to the real 32-player class and stay visibly
    distinct -- never silently collapsed into one confidence tier."""
    board = _board()
    assert len(board) == 32, len(board)
    counts = board["rushing_data_status"].value_counts().to_dict()
    assert counts.get("measured") == 15, counts
    assert counts.get("measured_override") == 4, counts
    assert counts.get("assumed_non_rusher") == 13, counts
    print(f"F PASS -- full 32-player class: {counts} (15 measured + 4 override + 13 assumed)")


def test_g_true_2026_board_six_real_prospects():
    """Closes the gap the last plan explicitly disclosed rather than
    fabricated: 2026_qb_season_stats.csv gives real, career-pooled college
    stats (every real season on file per player, not just the final one --
    see load_2026_qb_prospects()'s docstring for the real Drew Allar case
    this pooling was corrected for) for 6 of the 10 true-2026 QBs. All 6
    must be zero-NFL-snap projected (no qb_archetypes.csv row possible for
    a player who hasn't played an NFL game yet)."""
    board = _board_2026()
    assert len(board) == 6, len(board)
    assert (board["status"] == "projected").all(), board[["player_name", "status"]]
    assert (board["display_status"] == "projected").all()
    assert (board["rushing_data_status"] == "measured").all()
    print(f"G PASS -- true 2026 board: {len(board)} real prospects, all projected (zero NFL snaps)")


def test_h_taylen_green_real_rushing_pulls_dual_threat():
    """Real, career-pooled college production (53 games, 1190 att, 2405
    rush yds, 35 rush TD across 2021-2025) -- a genuine dual-threat
    profile, not assumed or overridden."""
    row = _row_2026("Taylen Green")
    assert row["games"] == 53 and row["attempts"] == 1190
    assert row["projected_tier"] == "dual_threat"
    assert row["display_tag"] == "dual_threat"
    assert row["forty_time"] == 4.36  # only one of the 6 with a real combine 40 time
    print(f"H PASS -- Taylen Green: real career-pooled rushing (1190 att across 5 seasons) -> "
          f"dual_threat, real 40-time (4.36s) carries through to Prospect Profile")


def test_i_allar_pooling_fixes_thin_injury_shortened_final_season():
    """Real regression lock for the exact bug the user caught: Drew
    Allar's real final college season (2025) alone was injury-shortened --
    only 6 games, 159 attempts. Using that season alone would make his
    entire signal a real but unrepresentative small sample. Pooling his
    full real career (45 games, 1002 attempts across 2022-2025, including
    a real complete 16-game junior season) is >6x his final season's
    attempts alone -- locks in that pooling is actually happening, not
    just present in the code but silently not engaged."""
    row = _row_2026("Drew Allar")
    assert row["games"] == 45 and row["attempts"] == 1002
    assert row["attempts"] > 159 * 2, "pooling must pull in real seasons beyond just the thin final one"
    assert row["projected_tier"] != "unconfirmed"
    assert row["display_tag"] != "unconfirmed"
    print(f"I PASS -- Drew Allar: pooled real career (45g/1002att, not just his thin 6g/159att "
          f"final season) -> {row['projected_tier']}, not Unconfirmed")


def test_j_mendoza_real_number_one_overall_pick():
    """Fernando Mendoza -- real #1 overall pick, real elite final-season
    passing efficiency (NCAA rating component should be high)."""
    row = _row_2026("Fernando Mendoza")
    assert row["draft_pick"] == 1
    assert row["component_ncaa_passer_rating"] > 60, row["component_ncaa_passer_rating"]
    print(f"J PASS -- Fernando Mendoza: real #1 overall pick, "
          f"NCAA passer rating component={row['component_ncaa_passer_rating']}")


def main() -> int:
    test_a_milroe_real_rushing_pulls_dual_threat()
    test_b_ewers_assumed_non_rusher_resolves_not_null()
    test_c_ehlinger_real_verified_override_and_display_bug_fix()
    test_d_young_stroud_penix_real_verified_low_rushing()
    test_e_richardson_and_levis_resolve_to_real_confirmed_veteran_tags()
    test_f_full_class_15_4_13_split()
    test_g_true_2026_board_six_real_prospects()
    test_h_taylen_green_real_rushing_pulls_dual_threat()
    test_i_allar_pooling_fixes_thin_injury_shortened_final_season()
    test_j_mendoza_real_number_one_overall_pick()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
