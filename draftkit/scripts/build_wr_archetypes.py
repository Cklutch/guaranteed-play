"""One-time build: WR archetype system v1 (wr_archetype_v1_spec.pdf).

Assembles the real inputs draftkit/wr_archetypes.py's classify_primary()/
classify_leans()/qb_context_label() need, applies them per WR, and writes
data/processed/wr_archetypes.csv. Runs standalone -- does not modify
risk_variables.csv, current_season_archetypes.csv, or any RB archetype file.

Reuses build_risk_variables.py's crosswalk/team-alias conventions directly,
same as build_rb_archetypes.py, since this is the exact same join-key
landscape.

Field provenance (see draftkit/wr_archetypes.py's docstring for the full
spec-placeholder -> real-field mapping):
  - target_share, games_played, target_share_data_quality_flag:
    build_risk_variables.load_target_share_rate()'s target_share_rate/
    games_recent/target_share_data_quality_flag, WR rows only -- that
    function sources target_share from nflverse's own real target_share
    column (stint-safe, targets-denominated) rescaled by team_games/
    games_played, and flags "corrected" when that rescaling moved the
    number by more than TARGET_SHARE_FLAG_THRESHOLD (see that function's
    docstring for the real Flacco CLE->CIN trade case this replaced an
    attempts-based team reconstruction to fix).
  - redzone_target_share, redzone_targets (raw), adot: real PBP,
    research/validation_v1/data/redzone_airyards_player_seasons.csv
    (prior_redzone_targets added this build for rz_conf's sample_confidence
    call -- see build_redzone_airyards_features_v1.py).
  - receptions, targets, yards_per_reception, yac_per_reception,
    rush_attempts, rush_attempts_per_game, rush_yards_per_attempt: derived
    from stats_player_reg_by_season, no PBP needed.
  - team_wr1_target_share: derived -- max target_share_rate among a team's
    own real, sample-qualified WRs (targets>=30, games>=6), including the
    player himself. Only relevant to the 8-16% Complementary band's gap
    check; irrelevant to Alpha/Possession/Unconfirmed reads.
  - qb_tier (1-5): PROXY -- team epa_rank (1-32, real, from
    load_offense_environment()) bucketed via ceil(epa_rank/32*5), reusing
    the exact substitution decision offense_environment_score() already
    made for this same missing field (see that function's docstring in
    build_risk_variables.py).

Usage:
    python -m draftkit.scripts.build_wr_archetypes
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.wr_archetypes import (  # noqa: E402
    SAMPLE_FLOOR_GAMES,
    SAMPLE_FLOOR_TARGETS,
    WRPrimary,
    classify_leans,
    classify_primary,
    qb_context_label,
)
from draftkit.scripts.build_risk_variables import (  # noqa: E402
    OFFENSE_CSV,
    RECENT_SEASON,
    TEAM_CODE_ALIASES,
    _build_abbrev_name_crosswalk,
    _build_id_name_crosswalk,
    _key,
    load_offense_environment,
    load_target_share_rate,
)

DATA_DIR = REPO_ROOT / "research" / "validation_v1" / "data"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
REDZONE_CSV = DATA_DIR / "redzone_airyards_player_seasons.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "wr_archetypes.csv"

CURRENT_SEASON = 2026

# Real games threshold below which the CURRENT season alone is too thin to
# trust for archetype classification -- the same real constant
# classify_primary() itself gates on (SAMPLE_FLOOR_GAMES). Concrete real
# case this exists for: Malik Nabers tore his ACL early in the 2025 season
# (4 real games) after a real, unambiguous 2024 rookie season (170 targets,
# 109 catches, 1204 yards, 7 TDs, 15 games) -- reading his archetype off
# the 4-game season alone discarded a real, large sample in favor of a
# real but tiny one, and classify_primary()'s own games_played gate
# correctly-but-misleadingly floored him at UNCONFIRMED. When the current
# season doesn't clear the floor, blend in the most recent PRIOR real
# season, games-weighted (see _games_weighted_blend()) -- the larger real
# sample dominates without fully discarding the current season's own real,
# if partial, signal.
RECENCY_FALLBACK_SEASON = RECENT_SEASON - 1


def _games_weighted_blend(current_val: float, current_games: float, prior_val: float, prior_games: float) -> tuple[float, float]:
    """Only blends when the current season alone doesn't clear
    SAMPLE_FLOOR_GAMES and a real prior-season row exists -- a player with
    a normal full season uses that season's own real read unchanged, never
    diluted by an older, less-current one."""
    if pd.notna(current_games) and current_games >= SAMPLE_FLOOR_GAMES:
        return current_val, current_games
    if prior_games is None or pd.isna(prior_games) or prior_games <= 0:
        return current_val, current_games
    current_games = current_games if pd.notna(current_games) else 0.0
    current_val = current_val if pd.notna(current_val) else 0.0
    combined_games = current_games + prior_games
    combined_val = (current_val * current_games + prior_val * prior_games) / combined_games
    return combined_val, combined_games


# ---------------------------------------------------------------------------
# WR-specific derived fields (no new data source -- see module docstring)
# ---------------------------------------------------------------------------

def load_receiving_rushing_stats(season: int = RECENT_SEASON) -> pd.DataFrame:
    """Real receptions/targets/receiving efficiency + WR rushing (jet
    sweeps/gadget plays, not a RB's rushing role), all raw from
    stats_player_reg_by_season -- WR rows only."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "games",
        "receptions", "targets", "receiving_yards", "receiving_yards_after_catch",
        "carries", "rushing_yards",
    ])
    wr = df[df["position"] == "WR"].copy()
    for col in ["receptions", "targets", "receiving_yards", "receiving_yards_after_catch", "carries", "rushing_yards"]:
        wr[col] = wr[col].fillna(0)
    wr["yards_per_reception"] = np.where(wr["receptions"] > 0, wr["receiving_yards"] / wr["receptions"], np.nan)
    wr["yac_per_reception"] = np.where(wr["receptions"] > 0, wr["receiving_yards_after_catch"] / wr["receptions"], np.nan)
    wr["rush_attempts_per_game"] = np.where(wr["games"] > 0, wr["carries"] / wr["games"], np.nan)
    wr["rush_yards_per_attempt"] = np.where(wr["carries"] > 0, wr["rushing_yards"] / wr["carries"], np.nan)
    wr = wr.rename(columns={"carries": "rush_attempts"})
    return wr[[
        "player_id", "receptions", "targets", "rush_attempts",
        "yards_per_reception", "yac_per_reception",
        "rush_attempts_per_game", "rush_yards_per_attempt",
    ]]


def load_redzone_and_adot(abbrev_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """redzone_target_share, redzone_targets (raw), adot -- all real PBP,
    see build_redzone_airyards_features_v1.py."""
    df = pd.read_csv(REDZONE_CSV)
    df = df[df["season"] == CURRENT_SEASON].copy()
    df = df.rename(columns={"player_name": "abbrev_name"})
    df = df.merge(abbrev_crosswalk, on=["abbrev_name", "team"], how="left")
    df = df.rename(columns={
        "prior_redzone_target_share": "redzone_target_share",
        "prior_redzone_targets": "redzone_targets",
        "prior_adot": "adot",
    })
    return df[["player_name", "redzone_target_share", "redzone_targets", "adot"]].dropna(subset=["player_name"])


QB_STARTER_INJURY_GAMES_MISSED_FLOOR = 6  # same "moderate-or-worse" threshold Home.py's _chronic_injury_badge() uses

RISK_VARIABLES_CSV = REPO_ROOT / "data" / "processed" / "risk_variables.csv"


def _qb_team_in_season(season: int) -> dict:
    """Real {full player_name: recent_team} for every QB row in that
    season's stats_player_reg_by_season file -- used to verify a presumed
    starter was actually ON his current team in the fallback season too
    (see _starter_injury_affected_teams()'s team-changer guard)."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["player_display_name", "position", "recent_team"])
    qbs = df[df["position"] == "QB"].dropna(subset=["player_display_name", "recent_team"])
    return dict(zip(qbs["player_display_name"], qbs["recent_team"]))


def _starter_injury_affected_teams() -> set[str]:
    """Real teams whose presumed 2026 starting QB missed
    >=QB_STARTER_INJURY_GAMES_MISSED_FLOOR real games in CURRENT_SEASON-1
    (the season offense_environment_team_seasons.csv's CURRENT_SEASON row
    is actually built from -- see load_offense_environment()'s module-level
    OFFENSE_CSV comment). "Presumed starter" = the team's real lowest-ADP
    QB (master_players.csv), a defensible proxy from data already on file,
    not a new source. Concrete case: Joe Burrow (CIN), real Grade 3 turf
    toe, missed ~9 games in 2025 -- CIN's real 2025 team-EPA (what
    CURRENT_SEASON=2026's row is built from) is diluted by Jake
    Browning/Joe Flacco's replacement-level starts, structurally
    undersellling Burrow's own real talent level now that he's healthy and
    the presumed 2026 starter again.

    Team-changer guard (real bug caught during verification, not assumed):
    a real games_missed_by_season>=6 flag alone isn't enough -- Kyler
    Murray's real lowest-ADP-QB slot at MIN and Malik Willis's at MIA both
    cleared it, but Murray played 2025 for ARI and Willis for GB (confirmed
    via stats_player_reg_by_season's real recent_team column), and neither
    has any real prior-season presence on their new team at all. Falling
    back to that team's OWN CURRENT_SEASON-2 row in that case would swap in
    a DIFFERENT quarterback's old performance, not "the healthy starter's
    own talent level" -- the opposite of what this override is for. Only
    flags a team when the presumed starter's real recent_team matches the
    current team in BOTH CURRENT_SEASON-1 (the diluted season) AND
    CURRENT_SEASON-2 (the fallback target) -- this also correctly excludes
    true rookies with no prior-season NFL row at all (Tyler Shough/NO,
    Shedeur Sanders/CLE), who separately cleared the games-missed floor but
    have no real earlier season to fall back to either.

    Reads risk_variables.csv directly (must be rebuilt before this runs) --
    real games_missed_by_season already computed there by
    draftkit.injury_history.injury_risk_score_v4()'s pipeline; recomputing
    it here from scratch would duplicate that whole machinery."""
    if not RISK_VARIABLES_CSV.exists() or not MASTER_CSV.exists():
        return set()

    master = pd.read_csv(MASTER_CSV)
    qbs = master[master["position"] == "QB"].dropna(subset=["adp", "team"]).copy()
    if qbs.empty:
        return set()
    starters = qbs.sort_values("adp").drop_duplicates("team", keep="first")

    risk = pd.read_csv(RISK_VARIABLES_CSV, usecols=["player_name", "games_missed_by_season"])
    risk["_key"] = _key(risk["player_name"])
    starters = starters.copy()
    starters["_key"] = _key(starters["player_name"])
    starters = starters.merge(risk[["_key", "games_missed_by_season"]], on="_key", how="left")

    team_prior1 = _qb_team_in_season(CURRENT_SEASON - 1)
    team_prior2 = _qb_team_in_season(CURRENT_SEASON - 2)

    affected = set()
    prior_season_str = str(CURRENT_SEASON - 1)
    for _, row in starters.iterrows():
        raw = row.get("games_missed_by_season")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            games_missed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if games_missed.get(prior_season_str, 0) < QB_STARTER_INJURY_GAMES_MISSED_FLOOR:
            continue
        name, team = row["player_name"], row["team"]
        if team_prior1.get(name) == team and team_prior2.get(name) == team:
            affected.add(team)
    return affected


def load_qb_context_by_team() -> pd.DataFrame:
    """team -> qb_tier proxy, from real epa_rank (1-32, 1=best offense).
    Bucketed via ceil(epa_rank/32*5): rank 1-6 -> tier 1, ..., rank 27-32
    -> tier 5. See module docstring for why epa_rank is the real, already-
    approved substitute for the spec's nonexistent qb_tier field.

    For teams in _starter_injury_affected_teams(), CURRENT_SEASON's real
    per-game inputs (prior_team_epa_pg/plays_pg/tds_pg -- themselves built
    from a real backup-QB-diluted season) are swapped for the PRIOR
    season's real inputs (the presumed starter's own last full healthy
    season) -- anchors qb_tier to the healthy starter's own talent level
    instead of a season-long blend across every QB who played. Splicing
    happens on the RAW per-game values, before ranking, and the whole
    32-team set is ranked exactly once afterward -- ranking each season
    separately and then concatenating the two already-ranked subsets would
    produce two independent, overlapping 1-32 scales stitched together,
    not one real, comparable ranking."""
    raw = pd.read_csv(OFFENSE_CSV)
    current = raw[raw["season"] == CURRENT_SEASON].copy()

    affected = _starter_injury_affected_teams()
    if affected:
        prior = raw[raw["season"] == CURRENT_SEASON - 1].copy()
        prior_affected = prior[prior["team"].isin(affected)]
        current = pd.concat(
            [current[~current["team"].isin(affected)], prior_affected], ignore_index=True
        )

    current["epa_rank"] = current["prior_team_epa_pg"].rank(ascending=False, method="first")
    current["qb_tier"] = current["epa_rank"].apply(
        lambda r: min(5, max(1, math.ceil(r / 32 * 5)))
    )
    return current[["team", "qb_tier"]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_wr_board() -> pd.DataFrame:
    """One row per WR with every real field classify_primary/_leans/
    qb_context_label need, EXCEPT team_wr1_target_share, which needs the
    full team grouping done in build_wr_archetypes() below."""
    master = pd.read_csv(MASTER_CSV)
    master = master[master["position"] == "WR"].copy()
    has_adp = pd.to_numeric(master.get("adp"), errors="coerce").notna()
    has_real_proj = master.get("projection_source", pd.Series(dtype=object)) == "real"
    master = master[has_adp | has_real_proj].copy()
    master["_key"] = _key(master["player_name"])

    crosswalk = _build_id_name_crosswalk()
    abbrev_crosswalk = _build_abbrev_name_crosswalk()

    target_share_df = load_target_share_rate(crosswalk)
    target_share_df = target_share_df[target_share_df["position"] == "WR"].copy()
    target_share_df = target_share_df.rename(columns={
        "target_share_rate": "target_share",
        "games_recent": "games_played",
    })

    id_lookup = crosswalk[["player_id", "player_name"]].copy()
    id_lookup["_key"] = _key(id_lookup["player_name"])
    id_lookup = id_lookup.drop_duplicates("_key", keep="first")

    board = master[["player_name", "team", "age", "adp", "_key"]].copy()
    board = board.merge(id_lookup[["_key", "player_id"]], on="_key", how="left")

    name_keyed_sources = {
        "target_share": target_share_df[
            ["player_name", "target_share", "games_played", "target_share_data_quality_flag"]
        ],
        "redzone": load_redzone_and_adot(abbrev_crosswalk),
    }
    for name, df in name_keyed_sources.items():
        df = df.copy()
        df["_key"] = _key(df["player_name"])
        df = df.drop(columns=["player_name"]).drop_duplicates("_key")
        board = board.merge(df, on="_key", how="left", suffixes=("", f"_{name}"))

    board = board.merge(
        load_receiving_rushing_stats().drop_duplicates("player_id"), on="player_id", how="left",
    )

    # Recency-weighted fallback -- only touches players whose CURRENT season
    # alone doesn't clear SAMPLE_FLOOR_GAMES (see RECENCY_FALLBACK_SEASON's
    # docstring). Real prior-season target_share/games_played (same loader,
    # season-1) and raw targets (classify_primary()'s OTHER real gate,
    # targets<SAMPLE_FLOOR_TARGETS) get pulled and games-weighted-blended in.
    prior_target_share_df = load_target_share_rate(crosswalk, season=RECENCY_FALLBACK_SEASON)
    prior_target_share_df = prior_target_share_df[prior_target_share_df["position"] == "WR"].copy()
    prior_target_share_df = prior_target_share_df.rename(columns={
        "target_share_rate": "prior_target_share", "games_recent": "prior_games_played",
    })
    prior_target_share_df["_key"] = _key(prior_target_share_df["player_name"])
    board = board.merge(
        prior_target_share_df[["_key", "prior_target_share", "prior_games_played"]].drop_duplicates("_key"),
        on="_key", how="left",
    )
    prior_receiving = load_receiving_rushing_stats(season=RECENCY_FALLBACK_SEASON)
    prior_receiving = prior_receiving[["player_id", "targets"]].rename(columns={"targets": "prior_targets"})
    board = board.merge(prior_receiving.drop_duplicates("player_id"), on="player_id", how="left")

    # Real per-row decision: did THIS player's current season clear the
    # games floor on its own? Captured before games_played gets overwritten
    # below, so the raw targets blend (a separate real field, not derived
    # from games_played) uses the exact same season decision as target_share
    # -- both must come from the same real season, never blended independently.
    current_season_was_thin = board["games_played"].fillna(0) < SAMPLE_FLOOR_GAMES

    blended = board.apply(
        lambda r: _games_weighted_blend(r["target_share"], r["games_played"], r["prior_target_share"], r["prior_games_played"]),
        axis=1, result_type="expand",
    )
    board["target_share"], board["games_played"] = blended[0], blended[1]

    used_prior = current_season_was_thin & board["prior_games_played"].notna() & (board["prior_games_played"] > 0)
    board.loc[used_prior, "targets"] = board.loc[used_prior, "targets"].fillna(0) + board.loc[used_prior, "prior_targets"].fillna(0)
    board = board.drop(columns=["prior_target_share", "prior_games_played", "prior_targets"])

    qb_context_df = load_qb_context_by_team()
    board["_join_team"] = board["team"].replace(TEAM_CODE_ALIASES)
    qb_context_df = qb_context_df.copy()
    qb_context_df["_join_team"] = qb_context_df["team"].replace(TEAM_CODE_ALIASES)
    board = board.merge(qb_context_df[["_join_team", "qb_tier"]], on="_join_team", how="left")

    return board.drop(columns=["_key"])


def _player_row_to_dict(row) -> dict:
    return {
        "target_share": _num(row.get("target_share")),
        "games_played": _num(row.get("games_played")),
        "receptions": _num(row.get("receptions")),
        "targets": _num(row.get("targets")),
        "redzone_target_share": _num(row.get("redzone_target_share")),
        "redzone_targets": _num(row.get("redzone_targets")),
        "adot": _num(row.get("adot")),
        "yards_per_reception": _num(row.get("yards_per_reception")),
        "yac_per_reception": _num(row.get("yac_per_reception")),
        "rush_attempts": _num(row.get("rush_attempts")),
        "rush_attempts_per_game": _num(row.get("rush_attempts_per_game")),
        "rush_yards_per_attempt": _num(row.get("rush_yards_per_attempt")),
    }


def _num(value) -> float:
    """Missing real data becomes 0.0, not NaN -- every threshold check in
    classify_primary/_leans is a real-valued comparison (>=, <=, <), and NaN
    comparisons are always False in Python, which would silently make a
    player fail every lean check rather than correctly floor at UNCONFIRMED
    via the explicit sample-floor gate. 0.0 is the neutral "no evidence of
    this" value for every field here (a share, a rate, a raw count)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return float(value)


def build_wr_archetypes() -> pd.DataFrame:
    board = build_wr_board()

    # team_wr1_target_share: max target_share among a team's own real,
    # sample-qualified WRs (targets>=30, games>=6), including the player
    # himself -- self-inclusion is harmless since this value is only
    # consulted for the 8-16% Complementary band, where the player being
    # evaluated is essentially never his own team's real WR1. Teams with no
    # qualified WR (rare) fall back to 0.0 -- no real gap can be computed,
    # so a Complementary-range read there correctly falls through to
    # Possession (co-equal default) rather than crashing on a NaN gap.
    qualified = (
        (board["targets"].fillna(0) >= SAMPLE_FLOOR_TARGETS)
        & (board["games_played"].fillna(0) >= SAMPLE_FLOOR_GAMES)
    )
    board["_qualified_target_share"] = np.where(qualified, board["target_share"], np.nan)
    board["team_wr1_target_share"] = board.groupby("_join_team")["_qualified_target_share"].transform("max")
    board["team_wr1_target_share"] = board["team_wr1_target_share"].fillna(0.0)
    board = board.drop(columns=["_qualified_target_share"])

    primaries, leans_col, qb_contexts, multi_lean_log = [], [], [], []
    for _, row in board.iterrows():
        player = _player_row_to_dict(row)
        team_wr1_share = _num(row.get("team_wr1_target_share"))

        primary = classify_primary(player, team_wr1_share)
        primaries.append(primary.value)

        leans = classify_leans(player) if primary != WRPrimary.UNCONFIRMED else []
        leans_col.append(",".join(l.value for l in leans) if leans else "none")
        if len(leans) > 1:
            multi_lean_log.append({"player_name": row["player_name"], "leans": [l.value for l in leans]})

        qb_tier = row.get("qb_tier")
        qb_contexts.append(qb_context_label({"qb_tier": qb_tier}).value if pd.notna(qb_tier) else "unknown")

    board["wr_archetype_primary"] = primaries
    board["wr_leans"] = leans_col
    board["wr_qb_context"] = qb_contexts
    # Real, computed flag (see load_target_share_rate's docstring) --
    # "corrected" when the games-played rescaling moved this player's
    # target_share by more than TARGET_SHARE_FLAG_THRESHOLD, not a manual
    # per-player list. Missing for rookies/players load_target_share_rate
    # has no row for at all, not just the ones it corrected -- fillna to
    # keep the column a clean flag/no-flag string either way.
    board["target_share_data_quality_flag"] = board["target_share_data_quality_flag"].fillna("")

    if multi_lean_log:
        print(f"[multi-lean] {len(multi_lean_log)} player(s) carry more than one lean:")
        for entry in multi_lean_log:
            print(f"    {entry['player_name']}: {entry['leans']}")

    return board.drop(columns=["_join_team"])


def main() -> int:
    board = build_wr_archetypes()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s)")
    print(board["wr_archetype_primary"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
