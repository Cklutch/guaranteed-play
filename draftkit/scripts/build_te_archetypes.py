"""One-time build: TE archetype system v1 (plan_te_archetypes.pdf).

Assembles the real inputs draftkit/te_archetypes.py's classify_primary()/
classify_leans() need, applies them per TE, and writes
data/processed/te_archetypes.csv. Runs standalone -- does not modify
risk_variables.csv, rb_archetypes.csv, wr_archetypes.csv, qb_archetypes.csv,
or the legacy build_current_season_archetypes.py/archetype_primary system
(deliberately left untouched -- see the plan's Context section).

Field provenance:
  - snap_share: research/validation_v1/data/snap_share_player_seasons.csv's
    prior_snap_share column, read at season == CURRENT_SEASON (2026) --
    NOT season == RECENT_SEASON. This file's prior_snap_share is a LAGGED
    feature (the row labeled season=2025 holds the real 2024 value); the
    season=2026 row's prior_snap_share is the real 2025 value. A first
    pass at this build read season == RECENT_SEASON directly and produced
    real, meaningfully wrong anchor numbers (Kittle 74.9% instead of the
    real 49.4%, Conklin 75.5% instead of the real 14.8%) -- caught via
    direct user-verification against PlayerProfiler before it shipped.
  - redzone_target_share, redzone_targets, targets (raw): real PBP,
    reused directly from build_wr_archetypes.py's load_redzone_and_adot()
    (redzone_airyards_player_seasons.csv has no position column -- the
    existing name/team join already works position-agnostically) and a
    new TE-filtered targets loader mirroring
    build_wr_archetypes.load_receiving_rushing_stats()'s shape.
  - target_share, games_played: NOT sourced from build_risk_variables.
    load_target_share_rate() -- that shared function reconstructs each
    team's pass-attempts-per-game by summing the `attempts` column across
    every player currently on that team, grouped by `recent_team` (a
    player's SEASON-ENDING team). This silently breaks for any player who
    changed teams mid-season, because nflverse folds a traded player's
    whole-season stat line into their final team only. Real, confirmed
    2025 case: Joe Flacco started the season with Cleveland, then moved to
    Cincinnati mid-season after Joe Burrow's injury -- his 416 attempts
    landed entirely in Cincinnati's team total and none in Cleveland's,
    which is why CLE's reconstructed team total (398 attempts) was the
    lowest in the league and CIN's (800) the highest, while every other
    team fell in a tight 422-649 range. Caught chasing down a real
    independent-source discrepancy on Harold Fannin's target_share (28.6%
    computed vs. ~20.5% from SumerSports/Muffed/nflverse's own column).
    There's also a second, separate issue in the shared function: its
    denominator is team total pass ATTEMPTS (includes throwaways/spikes
    with no targeted receiver), while nflverse's own target_share column
    denominates on team total TARGETS -- a real, structural overstatement
    unrelated to the stint bug (shows up even on stint-unaffected teams,
    e.g. Noah Fant/NO, Austin Hooper/ATL). Fixed here, TE-scoped only
    (the shared function stays as-is for WR/RB/risk_variables.csv, tracked
    as a separate urgent follow-up), via load_te_target_share_corrected():
    nflverse's own real target_share column (stint-safe on the team-total
    side) times team_games/player_games, recovering the same
    rate-per-game-played property load_target_share_rate() already
    intends without reintroducing its season-total dilution bug (verified
    directly on Sam LaPorta: raw nflverse column reads 8.9%, diluted by 8
    missed games; corrected reads 16.8%, matching this pipeline's existing
    LaPorta number of 15.9%).

Usage:
    python -m draftkit.scripts.build_te_archetypes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.te_archetypes import (  # noqa: E402
    TEArchetype,
    TELean,
    classify_leans,
    classify_primary,
    classify_role_profile,
)
from draftkit.scripts.build_risk_variables import (  # noqa: E402
    CURRENT_SEASON,
    RECENT_SEASON,
    STATS_DIR,
    _build_abbrev_name_crosswalk,
    _build_id_name_crosswalk,
    _key,
)
from draftkit.scripts.build_wr_archetypes import load_redzone_and_adot  # noqa: E402

DATA_DIR = REPO_ROOT / "research" / "validation_v1" / "data"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
SNAP_SHARE_CSV = DATA_DIR / "snap_share_player_seasons.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "te_archetypes.csv"
FULL_SEASON_GAMES = 17


def load_te_target_share_corrected(crosswalk: pd.DataFrame, season: int = RECENT_SEASON) -> pd.DataFrame:
    """Real, stint-safe, games-corrected target_share for TE only -- see
    module docstring for the real Flacco CLE->CIN 2025 mid-season trade
    this replaces load_target_share_rate()'s reconstruction to work around,
    and the separate attempts-vs-targets denominator issue it also fixes.
    Derived entirely from stats_player_reg_by_season -- no new data source.
    `crosswalk` supplies the real full player_name (the season file's own
    player_name column is abbreviated, e.g. "T.McBride" -- not usable for
    this build's name-key joins), same pattern load_target_share_rate() uses."""
    path = STATS_DIR / f"{season}.csv"
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "recent_team", "games", "targets", "target_share",
    ])
    df["games"] = pd.to_numeric(df["games"], errors="coerce")

    team_games = (
        df.dropna(subset=["player_id"])
        .assign(games_clip=lambda d: d["games"].clip(upper=FULL_SEASON_GAMES))
        .groupby("recent_team")["games_clip"].max()
    )

    te = df[df["position"] == "TE"].copy()
    te["team_games"] = te["recent_team"].map(team_games)
    te["target_share"] = np.where(
        (te["games"] > 0) & te["team_games"].notna(),
        te["target_share"] * te["team_games"] / te["games"],
        np.nan,
    )
    te = te.rename(columns={"recent_team": "team", "games": "games_played"})
    te = te.merge(crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    return te[["player_name", "team", "target_share", "games_played", "targets"]].dropna(subset=["player_name"])


def load_te_snap_share(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Real snap share -- see module docstring for why this reads the
    CURRENT_SEASON (2026) row's prior_snap_share, not RECENT_SEASON's."""
    df = pd.read_csv(SNAP_SHARE_CSV)
    te = df[(df["season"] == season) & (df["position"] == "TE")].copy()
    return te[["player_name", "prior_snap_share"]].rename(columns={"prior_snap_share": "snap_share"})


def load_te_targets(season: int = RECENT_SEASON) -> pd.DataFrame:
    """Real raw targets -- classify_leans()'s LEAN_REDZONE_MIN_TEAM_TARGETS
    gate and rz_conf both need the raw count, not just the share."""
    path = STATS_DIR / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "position", "targets"])
    te = df[df["position"] == "TE"].copy()
    te["targets"] = pd.to_numeric(te["targets"], errors="coerce").fillna(0.0)
    return te[["player_id", "targets"]]


def _num(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return float(value)


def build_te_board() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV)
    master = master[master["position"] == "TE"].copy()
    has_adp = pd.to_numeric(master.get("adp"), errors="coerce").notna()
    has_real_proj = master.get("projection_source", pd.Series(dtype=object)) == "real"
    master = master[has_adp | has_real_proj].copy()
    master["_key"] = _key(master["player_name"])

    crosswalk = _build_id_name_crosswalk()
    abbrev_crosswalk = _build_abbrev_name_crosswalk()

    target_share_df = load_te_target_share_corrected(crosswalk)

    id_lookup = crosswalk[["player_id", "player_name"]].copy()
    id_lookup["_key"] = _key(id_lookup["player_name"])
    id_lookup = id_lookup.drop_duplicates("_key", keep="first")

    board = master[["player_name", "team", "age", "adp", "_key"]].copy()
    board = board.merge(id_lookup[["_key", "player_id"]], on="_key", how="left")

    name_keyed_sources = {
        "target_share": target_share_df[["player_name", "target_share", "games_played"]],
        "snap_share": load_te_snap_share(),
        "redzone": load_redzone_and_adot(abbrev_crosswalk),
    }
    for name, df in name_keyed_sources.items():
        df = df.copy()
        df["_key"] = _key(df["player_name"])
        df = df.drop(columns=["player_name"]).drop_duplicates("_key")
        board = board.merge(df, on="_key", how="left", suffixes=("", f"_{name}"))

    board = board.merge(load_te_targets().drop_duplicates("player_id"), on="player_id", how="left")

    return board.drop(columns=["_key"])


def build_te_archetypes() -> pd.DataFrame:
    board = build_te_board()

    primaries, leans_col, role_profiles = [], [], []
    for _, row in board.iterrows():
        target_share = _num(row.get("target_share"))
        games_recent = _num(row.get("games_played"))
        snap_share = _num(row.get("snap_share"))
        redzone_target_share = _num(row.get("redzone_target_share"))

        primary = classify_primary(target_share, games_recent, snap_share)
        primaries.append(primary.value)

        player = {
            "redzone_target_share": redzone_target_share,
            "redzone_targets": _num(row.get("redzone_targets")),
            "target_share": target_share,
            "targets": _num(row.get("targets")),
            "games_played": games_recent,
        }
        leans = classify_leans(player) if primary.value != "unconfirmed" else []
        leans_col.append(",".join(l.value for l in leans) if leans else "none")

        if primary == TEArchetype.RECEIVING_TE:
            role_profiles.append(classify_role_profile(target_share, redzone_target_share).value)
        else:
            role_profiles.append("")

    board["te_archetype_primary"] = primaries
    board["te_leans"] = leans_col
    board["te_role_profile"] = role_profiles
    return board


def main() -> int:
    board = build_te_archetypes()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s)")
    print(board["te_archetype_primary"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
