"""One-time build: RB archetype taxonomy (rb_archetype_implementation_spec.pdf).

Assembles the real inputs draftkit/rb_archetypes.py's classify_primary()/
classify_down_split()/classify_lean() need, applies them per RB, and writes
data/processed/rb_archetypes.csv. Runs standalone -- does not modify
risk_variables.csv or the existing current_season_archetypes.csv system
(a different, percentile-based, position-agnostic archetype engine
explicitly excluded from scoring; see draftkit/rb_archetypes.py's
docstring).

Reuses build_risk_variables.py's crosswalk/team-alias conventions directly
rather than duplicating them, since this is the exact same join-key
landscape (same real name-collision and team-code-mismatch risks already
solved there).

Field provenance (see draftkit/rb_archetypes.py's docstring for the full
spec-placeholder -> real-field mapping):
  - opportunity_share, games_played, team_target_share: reused directly
    from build_risk_variables.load_target_share_rate() (already RB-aware;
    team_target_share = its pure-receiving target_share_rate).
  - total_touches, targets: raw carries/targets from
    stats_player_reg_by_season, RB rows.
  - redzone_touches, redzone_touch_share, inside_10_carry_share,
    explosive_run_rate: real PBP, research/validation_v1/data/
    redzone_airyards_player_seasons.csv (extended this session).
  - third_down_snap_share: real PBP+participation,
    research/validation_v1/data/third_down_participation_player_seasons.csv
    (new this session, build_third_down_participation_features_v1.py).
  - route_participation_rate: real, research/validation_v1/data/
    route_participation_player_seasons.csv (already existed, gsis_id join).
  - target_share_of_backfield, yards_per_touch: derived from
    stats_player_reg_by_season, no PBP needed.
  - depth_chart_rank: PROXY -- rank among same-team RBs by
    opportunity_share (no real depth-chart feed exists in this repo; same
    proxy pattern as build_risk_variables.compute_depth_chart_competition).

Usage:
    python -m draftkit.scripts.build_rb_archetypes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.rb_archetypes import (  # noqa: E402
    RBArchetype,
    _meets_conditions,
    classify_down_split,
    classify_lean,
    classify_primary,
)
from draftkit.scripts.build_risk_variables import (  # noqa: E402
    RECENT_SEASON,
    TEAM_CODE_ALIASES,
    _build_abbrev_name_crosswalk,
    _build_id_name_crosswalk,
    _key,
    load_target_share_rate,
)

DATA_DIR = REPO_ROOT / "research" / "validation_v1" / "data"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
REDZONE_CSV = DATA_DIR / "redzone_airyards_player_seasons.csv"
THIRD_DOWN_CSV = DATA_DIR / "third_down_participation_player_seasons.csv"
ROUTE_PARTICIPATION_CSV = DATA_DIR / "route_participation_player_seasons.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "rb_archetypes.csv"

CURRENT_SEASON = 2026

# Real players whose CURRENT team (master_players.csv) differs from the
# team their own most-recently-classified real season's stats actually
# belong to -- passed to classify_primary()'s intrinsic_only_distance flag
# (see rb_archetypes.py's _archetype_distance() docstring for the full
# rationale). Kenneth Walker: real 2025 production (51% opportunity_share,
# 257 touches) all happened at SEA; his team field was corrected to KC
# (real March 2026 free-agency signing) partway through this session,
# which is CORRECT for display/current-roster purposes but means
# COMMITTEE_BACK/HANDCUFF's teammate-relative nearest-fit conditions would
# otherwise compare him against KC players he never shared a season with.
#
# A small, explicit, dated list -- same MANUALLY_EXCLUDED_PLAYERS-style
# precedent as Home.py's own manual list, not a heuristic "detect recent
# trades" system. Explicitly a stand-in for the real fix (a separate
# stats_season_team field distinguishing "team the stats were produced
# for" from "current team," which doesn't exist anywhere in this pipeline
# yet) -- once that field exists, this list and the intrinsic_only_distance
# flag it drives both go away.
TEAM_CHANGED_PLAYERS = {"Kenneth Walker"}

# Real games threshold below which the CURRENT season alone is too thin to
# trust for archetype classification -- the same real threshold
# classify_primary()'s own gate uses (games_played<4). Same real bug class
# found and fixed for WR (build_wr_archetypes.py's RECENCY_FALLBACK_SEASON
# docstring has the concrete Malik Nabers ACL-tear case this generalizes
# from): an injury-shortened current season can discard a real, large prior
# season in favor of a real but tiny one. When the current season doesn't
# clear the floor, blend in the most recent PRIOR real season, games-
# weighted (see _games_weighted_blend()) -- scoped to exactly the three
# real fields classify_primary()'s own _any_floor_cleared() gate and its
# BELLCOW fallthrough consult (total_touches, targets, opportunity_share,
# plus games_played itself) -- NOT the other archetypes' own secondary
# fields (redzone_touch_share, third_down_snap_share, etc.), which stay on
# the current season's real (if thin) read.
RECENCY_FALLBACK_SEASON = RECENT_SEASON - 1


def _games_weighted_blend(current_val: float, current_games: float, prior_val: float, prior_games: float) -> tuple[float, float]:
    """Only blends when the current season alone doesn't clear the real
    4-game floor and a real prior-season row exists -- a player with a
    normal season uses that season's own real read unchanged, never
    diluted by an older, less-current one."""
    if pd.notna(current_games) and current_games >= 4:
        return current_val, current_games
    if prior_games is None or pd.isna(prior_games) or prior_games <= 0:
        return current_val, current_games
    current_games = current_games if pd.notna(current_games) else 0.0
    current_val = current_val if pd.notna(current_val) else 0.0
    combined_games = current_games + prior_games
    combined_val = (current_val * current_games + prior_val * prior_games) / combined_games
    return combined_val, combined_games


# ---------------------------------------------------------------------------
# RB-specific derived fields (no new data source -- see module docstring)
# ---------------------------------------------------------------------------

def load_raw_touches(season: int = RECENT_SEASON) -> pd.DataFrame:
    """Real carries/targets counts (RB rows only) -- total_touches and
    targets are raw-count inputs to sample_confidence(), not rates, so
    these can't be derived from opportunity_share_rate alone."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "position", "carries", "targets"])
    rb = df[df["position"] == "RB"].copy()
    rb["carries"] = rb["carries"].fillna(0)
    rb["targets"] = rb["targets"].fillna(0)
    rb["total_touches"] = rb["carries"] + rb["targets"]
    return rb[["player_id", "carries", "targets", "total_touches"]]


def load_backfield_opportunity_share(season: int = RECENT_SEASON) -> pd.DataFrame:
    """opportunity_share = (carries+targets) / (team's TOTAL RB carries+
    targets) -- share of the BACKFIELD's work, not share of all team
    offensive plays.

    Confirmed via real data this is what the spec's own thresholds require
    (Bellcow >=0.65, Committee 0.30-0.55, Handcuff <0.20): reusing
    build_risk_variables.load_target_share_rate()'s opportunity_share_rate
    (share of ALL team offensive snaps, built for a different purpose --
    judging whether a RISK category's elevated TD rate is explained by
    real volume) made 0.65 mathematically unreachable -- the highest real
    opportunity_share_rate in the entire 2025 RB pool was 0.417
    (McCaffrey, the league's highest-workload back). Recomputed against
    the real backfield-share denominator instead: Taylor 0.840, McCaffrey
    0.809, Bijan 0.699 -- all comfortably clear 0.65, as a real bellcow
    should. This is intentionally a DIFFERENT metric from
    build_risk_variables.py's opportunity_share_rate, scoped to this
    archetype module only -- that field is already validated/tuned for
    its own purpose there and isn't changed by this correction."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "position", "recent_team", "carries", "targets"])
    rb = df[df["position"] == "RB"].copy()
    rb["carries"] = rb["carries"].fillna(0)
    rb["targets"] = rb["targets"].fillna(0)
    rb["touches"] = rb["carries"] + rb["targets"]
    team_rb_touches = rb.groupby("recent_team")["touches"].transform("sum")
    rb["opportunity_share"] = np.where(team_rb_touches > 0, rb["touches"] / team_rb_touches, np.nan)
    return rb[["player_id", "opportunity_share"]]


def load_target_share_of_backfield(season: int = RECENT_SEASON) -> pd.DataFrame:
    """Player's real targets as a share of his TEAM's total RB targets that
    season -- a different denominator than target_share_rate (which divides
    by team pass attempts), specifically answering "who's the pass-catching
    back on this team," not "how big is his overall receiving role."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "position", "recent_team", "targets"])
    rb = df[df["position"] == "RB"].copy()
    rb["targets"] = rb["targets"].fillna(0)
    team_rb_targets = rb.groupby("recent_team")["targets"].transform("sum")
    rb["target_share_of_backfield"] = np.where(team_rb_targets > 0, rb["targets"] / team_rb_targets, np.nan)
    return rb[["player_id", "target_share_of_backfield"]]


def load_yards_per_touch(season: int = RECENT_SEASON) -> pd.DataFrame:
    """(rushing_yards+receiving_yards)/(carries+receptions) -- real touches
    (receptions, not targets), matching the spec's literal "yards per
    touch" wording. Distinct from opportunity_share's carries+targets
    convention on purpose: yards-per-touch is an efficiency stat and should
    be judged against realized touches, not opportunities."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "rushing_yards", "receiving_yards", "carries", "receptions",
    ])
    rb = df[df["position"] == "RB"].copy()
    touches = rb["carries"].fillna(0) + rb["receptions"].fillna(0)
    total_yards = rb["rushing_yards"].fillna(0) + rb["receiving_yards"].fillna(0)
    rb["yards_per_touch"] = np.where(touches > 0, total_yards / touches, np.nan)
    return rb[["player_id", "yards_per_touch"]]


def load_redzone_and_explosive(abbrev_crosswalk: pd.DataFrame) -> pd.DataFrame:
    """redzone_touches (raw), redzone_touch_share, inside_10_carry_share,
    explosive_run_rate -- all real PBP, see build_redzone_airyards_features_v1.py."""
    df = pd.read_csv(REDZONE_CSV)
    df = df[df["season"] == CURRENT_SEASON].copy()
    df = df.rename(columns={"player_name": "abbrev_name"})
    df = df.merge(abbrev_crosswalk, on=["abbrev_name", "team"], how="left")
    df = df.rename(columns={
        "prior_redzone_touches": "redzone_touches",
        "prior_redzone_touch_share": "redzone_touch_share",
        "prior_inside10_carry_share": "inside_10_carry_share",
        "prior_explosive_run_rate": "explosive_run_rate",
    })
    return df[[
        "player_name", "redzone_touches", "redzone_touch_share",
        "inside_10_carry_share", "explosive_run_rate",
    ]].dropna(subset=["player_name"])


def load_third_down_snap_share(id_crosswalk: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(THIRD_DOWN_CSV)
    df = df[df["season"] == CURRENT_SEASON].copy()
    df["player_id"] = df["player_id"].astype(str)
    df = df.merge(id_crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    return df[["player_name", "prior_third_down_snap_share"]].rename(
        columns={"prior_third_down_snap_share": "third_down_snap_share"}
    ).dropna(subset=["player_name"])


def load_route_participation(id_crosswalk: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(ROUTE_PARTICIPATION_CSV)
    df = df[df["season"] == CURRENT_SEASON].copy()
    df["player_id"] = df["player_id"].astype(str)
    df = df.merge(id_crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    return df[["player_name", "prior_route_participation_rate"]].rename(
        columns={"prior_route_participation_rate": "route_participation_rate"}
    ).dropna(subset=["player_name"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_rb_board() -> pd.DataFrame:
    """One row per RB with every real field classify_primary/_down_split/
    _lean need, EXCEPT depth_chart_rank and teammate context, which need
    the full team grouping done in build_rb_archetypes() below."""
    master = pd.read_csv(MASTER_CSV)
    master = master[master["position"] == "RB"].copy()
    has_adp = pd.to_numeric(master.get("adp"), errors="coerce").notna()
    has_real_proj = master.get("projection_source", pd.Series(dtype=object)) == "real"
    master = master[has_adp | has_real_proj].copy()
    master["_key"] = _key(master["player_name"])

    crosswalk = _build_id_name_crosswalk()
    abbrev_crosswalk = _build_abbrev_name_crosswalk()

    # team_target_share and games_played reused from build_risk_variables
    # (both correct as-is for archetype purposes -- team_target_share is a
    # share of team real TARGETS, unrelated to the backfield-share issue
    # below). opportunity_share is NOT reused -- see
    # load_backfield_opportunity_share()'s docstring for why.
    # target_share_data_quality_flag carried through too, same disclosed
    # marker WR archetypes surface (see load_target_share_rate's docstring)
    # -- "corrected" whenever a missed-games rescale moved this player's
    # own team_target_share by more than TARGET_SHARE_FLAG_THRESHOLD.
    target_share_df = load_target_share_rate(crosswalk)
    target_share_df = target_share_df[target_share_df["position"] == "RB"].copy()
    target_share_df = target_share_df.rename(columns={
        "target_share_rate": "team_target_share",
        "games_recent": "games_played",
    })

    id_lookup = crosswalk[["player_id", "player_name"]].copy()
    id_lookup["_key"] = _key(id_lookup["player_name"])
    id_lookup = id_lookup.drop_duplicates("_key", keep="first")

    board = master[["player_name", "team", "age", "adp", "_key"]].copy()
    board = board.merge(id_lookup[["_key", "player_id"]], on="_key", how="left")

    name_keyed_sources = {
        "opportunity": target_share_df[
            ["player_name", "team_target_share", "games_played", "target_share_data_quality_flag"]
        ],
        "redzone": load_redzone_and_explosive(abbrev_crosswalk),
        "third_down": load_third_down_snap_share(crosswalk),
        "route": load_route_participation(crosswalk),
    }
    for name, df in name_keyed_sources.items():
        df = df.copy()
        df["_key"] = _key(df["player_name"])
        df = df.drop(columns=["player_name"]).drop_duplicates("_key")
        board = board.merge(df, on="_key", how="left", suffixes=("", f"_{name}"))

    id_keyed_sources = {
        "touches": load_raw_touches(),
        "opp_share": load_backfield_opportunity_share(),
        "backfield_share": load_target_share_of_backfield(),
        "ypt": load_yards_per_touch(),
    }
    for name, df in id_keyed_sources.items():
        board = board.merge(df.drop_duplicates("player_id"), on="player_id", how="left", suffixes=("", f"_{name}"))

    # Recency-weighted fallback -- see RECENCY_FALLBACK_SEASON's docstring.
    # Only touches players whose CURRENT season alone doesn't clear the
    # real 4-game floor; real prior-season total_touches/targets (raw
    # counts, load_raw_touches) and opportunity_share (load_backfield_
    # opportunity_share) get pulled and games-weighted-blended in.
    prior_target_share_df = load_target_share_rate(crosswalk, season=RECENCY_FALLBACK_SEASON)
    prior_target_share_df = prior_target_share_df[prior_target_share_df["position"] == "RB"].copy()
    prior_target_share_df = prior_target_share_df.rename(columns={"games_recent": "prior_games_played"})
    prior_target_share_df["_key"] = _key(prior_target_share_df["player_name"])
    board = board.merge(
        prior_target_share_df[["_key", "prior_games_played"]].drop_duplicates("_key"), on="_key", how="left",
    )
    prior_touches = load_raw_touches(season=RECENCY_FALLBACK_SEASON).rename(
        columns={"total_touches": "prior_total_touches", "targets": "prior_targets"}
    )
    board = board.merge(
        prior_touches[["player_id", "prior_total_touches", "prior_targets"]].drop_duplicates("player_id"),
        on="player_id", how="left",
    )
    prior_opp_share = load_backfield_opportunity_share(season=RECENCY_FALLBACK_SEASON).rename(
        columns={"opportunity_share": "prior_opportunity_share"}
    )
    board = board.merge(prior_opp_share.drop_duplicates("player_id"), on="player_id", how="left")

    # Real per-row decision, captured before games_played gets overwritten
    # below -- total_touches/targets/opportunity_share must all be blended
    # against the SAME real season split, not independently re-derived.
    current_games = board["games_played"].copy()
    needs_blend = (
        current_games.fillna(0).lt(4)
        & board["prior_games_played"].notna()
        & board["prior_games_played"].gt(0)
    )

    # total_touches/targets are RAW ACCUMULATED COUNTS, not rates -- a real
    # season's worth of touches summed with a real prior season's touches
    # is a real combined total (matches how targets was correctly summed
    # for WR); _games_weighted_blend's rate-averaging formula only applies
    # to opportunity_share below, which IS a real rate.
    board.loc[needs_blend, "total_touches"] = (
        board.loc[needs_blend, "total_touches"].fillna(0) + board.loc[needs_blend, "prior_total_touches"].fillna(0)
    )
    board.loc[needs_blend, "targets"] = (
        board.loc[needs_blend, "targets"].fillna(0) + board.loc[needs_blend, "prior_targets"].fillna(0)
    )

    opp_blend = board.apply(
        lambda r: _games_weighted_blend(r["opportunity_share"], current_games[r.name], r["prior_opportunity_share"], r["prior_games_played"]),
        axis=1, result_type="expand",
    )
    board["opportunity_share"], board["games_played"] = opp_blend[0], opp_blend[1]

    board = board.drop(columns=["prior_games_played", "prior_total_touches", "prior_targets", "prior_opportunity_share"])

    return board.drop(columns=["_key"])


def _player_row_to_dict(row) -> dict:
    return {
        "opportunity_share": _num(row.get("opportunity_share")),
        "games_played": _num(row.get("games_played")),
        "total_touches": _num(row.get("total_touches")),
        "targets": _num(row.get("targets")),
        "redzone_touches": _num(row.get("redzone_touches")),
        "redzone_touch_share": _num(row.get("redzone_touch_share")),
        "inside_10_carry_share": _num(row.get("inside_10_carry_share")),
        "yards_per_touch": _num(row.get("yards_per_touch")),
        "explosive_run_rate": _num(row.get("explosive_run_rate")),
        "target_share_of_backfield": _num(row.get("target_share_of_backfield")),
        "third_down_snap_share": _num(row.get("third_down_snap_share")),
        "team_target_share": _num(row.get("team_target_share")),
        "route_participation_rate": _num(row.get("route_participation_rate")),
        "depth_chart_rank": int(row["depth_chart_rank"]) if pd.notna(row.get("depth_chart_rank")) else 99,
    }


def _num(value) -> float:
    """Missing real data becomes 0.0, not NaN -- every threshold check in
    classify_primary/_lean is a real-valued comparison (>=, <), and NaN
    comparisons are always False in Python, which would silently make a
    player fail every archetype check rather than correctly floor at
    UNCONFIRMED via the explicit games_played<4 gate. 0.0 is the neutral
    "no evidence of this" value for every field here (a share, a rate, a
    raw count)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return float(value)


def build_rb_archetypes() -> pd.DataFrame:
    board = build_rb_board()
    board["_join_team"] = board["team"].replace(TEAM_CODE_ALIASES)
    board["depth_chart_rank"] = board.groupby("_join_team")["opportunity_share"].rank(
        ascending=False, method="first"
    )

    primaries, down_splits, leans, tie_break_log = [], [], [], []
    for idx, row in board.iterrows():
        player = _player_row_to_dict(row)
        # Exclude self by index, not identity -- _player_row_to_dict()
        # builds a fresh dict per call, so an `is` check would never match.
        # Real teammates only -- games_played>=4 matches the SAME sample
        # floor COMMITTEE_BACK's own min_games check requires of the
        # player himself, so a token appearance can't silently disqualify
        # a real committee match. Two confirmed real bugs from a looser
        # >0 filter, not theoretical:
        #   - TreVeyon Henderson's real committee_back match with
        #     Rhamondre Stevenson failed min_games>=4 purely because NE's
        #     roster also includes Jam Miller/Lan Larison (0 games) --
        #     min([17,14,0,11,0])=0.
        #   - Jordan Mason's real, near-even split with Aaron Jones at
        #     Minnesota (0.42 vs 0.41 opportunity_share) failed the same
        #     gate because Jordan Mims' single 1-game appearance (real,
        #     nonzero, but not a meaningful sample) alone dragged
        #     min_games to 1.
        teammates = [
            _player_row_to_dict(r) for j, r in board[board["_join_team"] == row["_join_team"]].iterrows()
            if j != idx and pd.notna(r.get("games_played")) and r.get("games_played", 0) >= 4
        ]

        primary = classify_primary(
            player, teammates, intrinsic_only_distance=row["player_name"] in TEAM_CHANGED_PLAYERS
        )
        primaries.append(primary.value)
        down_splits.append(classify_down_split(player, primary).value)
        leans.append(classify_lean(player, primary).value)

        # Log precedence ties for the validation report (spec Section 10d):
        # any case where more than one PRIMARY archetype's raw conditions
        # were independently true, not just the one precedence returned.
        satisfied = [
            a.value for a in RBArchetype if a not in (RBArchetype.BELLCOW, RBArchetype.UNCONFIRMED)
            and _meets_conditions(a, player, teammates)
        ]
        if len(satisfied) > 1:
            tie_break_log.append({"player_name": row["player_name"], "satisfied": satisfied, "chosen": primary.value})

    board["archetype_primary"] = primaries
    board["down_split"] = down_splits
    board["lean"] = leans
    board["target_share_data_quality_flag"] = board["target_share_data_quality_flag"].fillna("")

    if tie_break_log:
        print(f"[precedence] {len(tie_break_log)} player(s) matched multiple primary archetypes:")
        for entry in tie_break_log:
            print(f"    {entry['player_name']}: satisfied={entry['satisfied']} -> chose {entry['chosen']}")

    return board.drop(columns=["_join_team"])


def main() -> int:
    board = build_rb_archetypes()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s)")
    print(board["archetype_primary"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
