"""QB Archetype System v1 spec tests (claude_code_plan_qb_archetypes.pdf).

Plain assert-based, no pytest (not installed in this project's venv) --
matches the loose test_*.py convention already used elsewhere in this repo.
Runnable directly:

    python -m draftkit.tests.test_qb_archetypes
"""

from __future__ import annotations

from draftkit.qb_archetypes import (
    DUAL_THREAT_FLOOR,
    POCKET_PASSER_CEILING,
    SAMPLE_FLOOR_ATTEMPTS,
    SAMPLE_FLOOR_GAMES,
    QBArchetype,
    classify_primary,
)
from draftkit.scripts.build_qb_archetypes import (
    _available_season_files,
    _build_id_name_crosswalk,
    _key,
    _load_qb_season,
    build_qb_archetypes,
    pool_qb_seasons,
)


def _board():
    board = build_qb_archetypes()
    board["_key"] = _key(board["player_name"])
    return board


def _row(board, name):
    hit = board[board["player_name"] == name]
    assert not hit.empty, f"{name} not found in real qb_archetypes board"
    return hit.iloc[0]


def test_a_josh_allen_clean_dual_threat():
    board = _board()
    row = _row(board, "Josh Allen")
    assert row["games"] == 32 and row["attempts"] == 943
    assert round(row["rushing_fantasy_pct"], 4) == 0.3445, row["rushing_fantasy_pct"]
    assert row["qb_archetype_primary"] == "dual_threat"
    print(f"A PASS -- Josh Allen: pooled rushing_fantasy_pct={row['rushing_fantasy_pct']:.3f} -> dual_threat")


def test_b_tua_clean_pocket_passer():
    board = _board()
    row = _row(board, "Tua Tagovailoa")
    assert round(row["rushing_fantasy_pct"], 4) == 0.0238, row["rushing_fantasy_pct"]
    assert row["qb_archetype_primary"] == "pocket_passer"
    print(f"B PASS -- Tua Tagovailoa: pooled rushing_fantasy_pct={row['rushing_fantasy_pct']:.3f} -> pocket_passer")


def test_c_hurts_blend_resolves_into_dual_threat():
    """Real, deliberately-checked methodology case: single-season 2025 alone
    (28.2%) sits under DUAL_THREAT_FLOOR; the 2-season pool (36.2%) clears
    it. Locks in that the blend genuinely changes his classification, not
    just his displayed percentage."""
    crosswalk = _build_id_name_crosswalk()
    player_id = crosswalk.loc[crosswalk["player_name"] == "Jalen Hurts", "player_id"].iloc[0]
    seasons = _available_season_files()
    frames = {s: _load_qb_season(s) for s in seasons[:3]}

    single = pool_qb_seasons(player_id, frames, max_seasons=1)
    pooled = pool_qb_seasons(player_id, frames, max_seasons=2)

    assert round(single["rushing_fantasy_pct"], 4) == 0.2824, single
    assert round(pooled["rushing_fantasy_pct"], 4) == 0.3624, pooled
    assert classify_primary(single["rushing_fantasy_pct"], single["attempts"], single["games"]) == QBArchetype.BALANCED
    assert classify_primary(pooled["rushing_fantasy_pct"], pooled["attempts"], pooled["games"]) == QBArchetype.DUAL_THREAT
    print(f"C PASS -- Jalen Hurts: single-season {single['rushing_fantasy_pct']:.3f} (Balanced) "
          f"-> pooled {pooled['rushing_fantasy_pct']:.3f} (Dual-Threat) -- blend genuinely resolves this one")


def test_d_lamar_blend_does_not_resolve():
    """Equally deliberate: the same pooling mechanism does NOT move Lamar
    Jackson into Dual-Threat (20.1% single-season -> 23.9% pooled, both
    under the 30% floor) -- locked in so this doesn't quietly drift, and so
    nobody mistakes the blend for a threshold tuned to force every reputed
    Dual-Threat QB into that tier."""
    crosswalk = _build_id_name_crosswalk()
    player_id = crosswalk.loc[crosswalk["player_name"] == "Lamar Jackson", "player_id"].iloc[0]
    seasons = _available_season_files()
    frames = {s: _load_qb_season(s) for s in seasons[:3]}

    single = pool_qb_seasons(player_id, frames, max_seasons=1)
    pooled = pool_qb_seasons(player_id, frames, max_seasons=2)

    assert round(single["rushing_fantasy_pct"], 4) == 0.2014, single
    assert round(pooled["rushing_fantasy_pct"], 4) == 0.2391, pooled
    assert classify_primary(single["rushing_fantasy_pct"], single["attempts"], single["games"]) == QBArchetype.BALANCED
    assert classify_primary(pooled["rushing_fantasy_pct"], pooled["attempts"], pooled["games"]) == QBArchetype.BALANCED
    print(f"D PASS -- Lamar Jackson: single-season {single['rushing_fantasy_pct']:.3f} and pooled "
          f"{pooled['rushing_fantasy_pct']:.3f} both stay Balanced -- blend does not rescue this one")


def test_e_richardson_pooled_sample_clears_gate_single_season_does_not():
    """The PDF's own named 'decide deliberately' edge case: single 2025
    season alone (2 games, 2 attempts) fails SAMPLE_FLOOR_GAMES/ATTEMPTS.
    Pooling his last 2 real seasons (2024+2025, 13 games/266 attempts)
    gives him a real, complete-season-equivalent sample that clears the
    gate -- resolved by pooling, not a hardcoded special case."""
    crosswalk = _build_id_name_crosswalk()
    player_id = crosswalk.loc[crosswalk["player_name"] == "Anthony Richardson", "player_id"].iloc[0]
    seasons = _available_season_files()
    frames = {s: _load_qb_season(s) for s in seasons[:3]}

    single = pool_qb_seasons(player_id, frames, max_seasons=1)
    pooled = pool_qb_seasons(player_id, frames, max_seasons=2)

    assert single["games"] == 2 and single["attempts"] == 2
    assert classify_primary(single["rushing_fantasy_pct"], single["attempts"], single["games"]) == QBArchetype.UNCONFIRMED
    assert pooled["games"] == 13 and pooled["attempts"] == 266
    assert classify_primary(pooled["rushing_fantasy_pct"], pooled["attempts"], pooled["games"]) == QBArchetype.DUAL_THREAT
    print(f"E PASS -- Anthony Richardson: single-season (2g/2att) hits the gate -> Unconfirmed; "
          f"pooled (13g/266att) clears it -> Dual-Threat")


def test_f_brosmer_gate_fails_even_with_only_one_real_season():
    """Max Brosmer has no ADP/real projection on file (deep-bench rookie
    backup), so he's correctly excluded from the full board -- same
    has_adp|has_real_proj filter RB/WR archetypes already use. Tested
    directly against his real, hand-verified 2025 stats instead: only 1
    real season exists (no 2024 row, true rookie), so pooling can't rescue
    him -- 71 attempts stays under SAMPLE_FLOOR_ATTEMPTS regardless."""
    crosswalk = _build_id_name_crosswalk()
    matches = crosswalk[crosswalk["player_name"] == "Max Brosmer"]
    assert not matches.empty, "Max Brosmer must still resolve via the crosswalk even though he's off the board"
    player_id = matches["player_id"].iloc[0]
    seasons = _available_season_files()
    frames = {s: _load_qb_season(s) for s in seasons[:3]}

    pooled = pool_qb_seasons(player_id, frames, max_seasons=2)
    assert pooled["seasons_used"] == "2025", pooled  # only 1 real season on file
    assert pooled["attempts"] == 71 and pooled["games"] == 7
    assert classify_primary(pooled["rushing_fantasy_pct"], pooled["attempts"], pooled["games"]) == QBArchetype.UNCONFIRMED
    print(f"F PASS -- Max Brosmer: only 1 real season (71 attempts) -- gate fails even pooled, "
          f"and he's correctly absent from the full board (no ADP/real projection)")


def test_g_levis_zero_2025_rows_falls_back_to_2023_2024():
    """Real absence from the league in 2025 (zero rows) -- the pooling walk
    must skip that empty season and reach back to his last 2 real seasons
    (2023, 2024) rather than reading 2025 as a false zero or stopping
    early. Confirms the recency-fallback behavior the user explicitly
    required be built, not deferred."""
    board = _board()
    row = _row(board, "Will Levis")
    assert row["seasons_used"] == "2023,2024", row["seasons_used"]
    assert row["games"] == 21 and row["attempts"] == 556
    assert row["qb_archetype_primary"] == "balanced"
    print(f"G PASS -- Will Levis: zero 2025 rows, pooled real 2023+2024 instead "
          f"(seasons_used={row['seasons_used']}) -> balanced")


def test_h_full_pool_no_invalid_primary_values():
    board = build_qb_archetypes()
    allowed = {a.value for a in QBArchetype}
    bad = set(board["qb_archetype_primary"].unique()) - allowed
    assert not bad, f"unexpected qb_archetype_primary values: {bad}"
    counts = board["qb_archetype_primary"].value_counts()
    print(f"H PASS -- full pool ({len(board)} real QBs): {counts.to_dict()}")


def main() -> int:
    test_a_josh_allen_clean_dual_threat()
    test_b_tua_clean_pocket_passer()
    test_c_hurts_blend_resolves_into_dual_threat()
    test_d_lamar_blend_does_not_resolve()
    test_e_richardson_pooled_sample_clears_gate_single_season_does_not()
    test_f_brosmer_gate_fails_even_with_only_one_real_season()
    test_g_levis_zero_2025_rows_falls_back_to_2023_2024()
    test_h_full_pool_no_invalid_primary_values()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
