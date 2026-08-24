"""One-time build: QB archetype system v1 (claude_code_plan_qb_archetypes.pdf).

Assembles rushing_fantasy_pct -- pooled across the most recent 2 real
seasons with data per QB, not a single season, per the plan's explicit
methodology decision (see draftkit/qb_archetypes.py's module docstring) --
and writes data/processed/qb_archetypes.csv. Runs standalone -- does not
modify risk_variables.csv, rb_archetypes.csv, or wr_archetypes.csv.

Reuses build_risk_variables.py's crosswalk/master-filtering conventions
directly, same join-key landscape as build_rb_archetypes.py/
build_wr_archetypes.py.

Field provenance:
  - games, attempts, passing_yards, passing_tds, carries (rushing
    attempts), rushing_yards, rushing_tds: real,
    research/validation_v1/data/stats_player_reg_by_season/{season}.csv,
    QB rows only -- the same real per-season stat file
    build_risk_variables.py already reads.
  - rushing_fantasy_pct: derived, pool_qb_seasons() below, using this
    project's own FULL_PPR_SCORING weights (projection_engine.py) --
    NOT the stats file's own precomputed fantasy_points column, which may
    use different scoring assumptions.

Pooling is NOT the same mechanism as RB/WR's RECENCY_FALLBACK_SEASON /
_games_weighted_blend() (build_rb_archetypes.py / build_wr_archetypes.py):
that mechanism only blends in a prior season when the CURRENT season alone
is too thin (a conditional rescue). This module always pools up to the 2
most recent real season-rows regardless of whether the current season is
thin, per the explicit real-world reasoning that QB rushing volume is
strategically volatile in a way RB/WR usage isn't (contract-year caution,
scheme-driven single-season dips) -- a different trigger, same underlying
"don't discard a real season's signal" philosophy and the same
games-weighted-sum math.

Usage:
    python -m draftkit.scripts.build_qb_archetypes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.qb_archetypes import QBArchetype, classify_primary  # noqa: E402
from draftkit.projection_engine import FULL_PPR_SCORING  # noqa: E402
from draftkit.scripts.build_risk_variables import (  # noqa: E402
    DATA_DIR,
    MASTER_CSV,
    RECENT_SEASON,
    STATS_DIR,
    _build_id_name_crosswalk,
    _key,
)

OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "qb_archetypes.csv"

BLEND_MAX_SEASONS = 2
PASS_YARD_PTS = FULL_PPR_SCORING["passing_yards"]
PASS_TD_PTS = FULL_PPR_SCORING["passing_tds"]
RUSH_YARD_PTS = FULL_PPR_SCORING["rushing_yards"]
RUSH_TD_PTS = FULL_PPR_SCORING["rushing_tds"]


def _available_season_files() -> list[int]:
    return sorted(
        (int(p.stem) for p in STATS_DIR.glob("*.csv") if p.stem.isdigit()),
        reverse=True,
    )


def _load_qb_season(season: int) -> pd.DataFrame:
    """Real per-QB row for one season: player_id, games, attempts,
    passing_yards, passing_tds, carries, rushing_yards, rushing_tds.
    Empty frame if the season file doesn't exist or has no QB rows."""
    path = STATS_DIR / f"{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "player_id", "games", "attempts", "passing_yards", "passing_tds",
            "carries", "rushing_yards", "rushing_tds",
        ])
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "games", "attempts", "passing_yards", "passing_tds",
        "carries", "rushing_yards", "rushing_tds",
    ])
    qb = df[df["position"] == "QB"].dropna(subset=["player_id"]).copy()
    for col in ("games", "attempts", "passing_yards", "passing_tds", "carries", "rushing_yards", "rushing_tds"):
        qb[col] = pd.to_numeric(qb[col], errors="coerce").fillna(0)
    return qb[[
        "player_id", "games", "attempts", "passing_yards", "passing_tds",
        "carries", "rushing_yards", "rushing_tds",
    ]]


def pool_qb_seasons(player_id: str, season_frames: dict[int, pd.DataFrame], max_seasons: int = BLEND_MAX_SEASONS) -> dict:
    """Walk backward through real season files (most recent first),
    collecting up to max_seasons real rows for this player_id -- skipping
    seasons with no row at all (the real Will Levis case: zero 2025 rows,
    a full real 2024 season). Sums raw volume/yardage/TD counts across
    whatever's collected, then derives rushing_fantasy_pct from the pooled
    totals -- volume-weighted, not an average of per-season percentages."""
    collected = []
    seasons_used = []
    for season in sorted(season_frames.keys(), reverse=True):
        row = season_frames[season]
        row = row[row["player_id"] == player_id]
        if row.empty:
            continue
        collected.append(row.iloc[0])
        seasons_used.append(season)
        if len(collected) >= max_seasons:
            break

    if not collected:
        return {
            "games": 0.0, "attempts": 0.0, "rushing_fantasy_pct": 0.0, "seasons_used": "",
        }

    games = sum(r["games"] for r in collected)
    attempts = sum(r["attempts"] for r in collected)
    passing_yards = sum(r["passing_yards"] for r in collected)
    passing_tds = sum(r["passing_tds"] for r in collected)
    rushing_yards = sum(r["rushing_yards"] for r in collected)
    rushing_tds = sum(r["rushing_tds"] for r in collected)

    rush_fp = rushing_yards * RUSH_YARD_PTS + rushing_tds * RUSH_TD_PTS
    pass_fp = passing_yards * PASS_YARD_PTS + passing_tds * PASS_TD_PTS
    total_fp = rush_fp + pass_fp
    rushing_fantasy_pct = rush_fp / total_fp if total_fp else 0.0

    return {
        "games": games,
        "attempts": attempts,
        "rushing_fantasy_pct": rushing_fantasy_pct,
        "seasons_used": ",".join(str(s) for s in sorted(seasons_used)),
    }


def build_qb_archetypes() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV)
    master = master[master["position"] == "QB"].copy()
    has_adp = pd.to_numeric(master.get("adp"), errors="coerce").notna()
    has_real_proj = master.get("projection_source", pd.Series(dtype=object)) == "real"
    master = master[has_adp | has_real_proj].copy()
    master["_key"] = _key(master["player_name"])

    crosswalk = _build_id_name_crosswalk()
    id_lookup = crosswalk[["player_id", "player_name"]].copy()
    id_lookup["_key"] = _key(id_lookup["player_name"])
    id_lookup = id_lookup.drop_duplicates("_key", keep="first")

    board = master[["player_name", "team", "_key"]].copy()
    board = board.merge(id_lookup[["_key", "player_id"]], on="_key", how="left")
    board = board.dropna(subset=["player_id"]).drop(columns=["_key"])

    seasons_to_check = _available_season_files()[:BLEND_MAX_SEASONS + 2]  # small margin for gap years
    season_frames = {season: _load_qb_season(season) for season in seasons_to_check}

    pooled_rows = []
    for player_id in board["player_id"]:
        pooled_rows.append(pool_qb_seasons(player_id, season_frames))
    pooled_df = pd.DataFrame(pooled_rows)
    board = pd.concat([board.reset_index(drop=True), pooled_df.reset_index(drop=True)], axis=1)

    board["qb_archetype_primary"] = board.apply(
        lambda r: classify_primary(r["rushing_fantasy_pct"], r["attempts"], r["games"]).value,
        axis=1,
    )
    return board


def main() -> int:
    board = build_qb_archetypes()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s)")
    print(board["qb_archetype_primary"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
