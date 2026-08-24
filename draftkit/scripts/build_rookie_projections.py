"""One-time build: rookie projection model v1 (rookie_projection_model_v1_spec.pdf).

Assembles real inputs for draftkit/rookie_projection.py's wr_rookie_projection()/
rb_rookie_projection()/blend_rookie_tag(), applies them per rookie, and writes
data/processed/rookie_projections.csv. Runs standalone -- does not modify
risk_variables.csv or either archetype file.

Field provenance:
  - Rookie inputs: data/raw/rookie_inputs.csv (real, see
    build_rookie_inputs.py -- draft capital/landing team/college/combine
    measurables real; dominator/breakout_age blank for v1).
  - Team context: data/processed/risk_variables.csv's real per-team
    offense_environment_score/schedule_weather_venue_score (duplicated
    across every player on that team -- deduped to one row per team here).
    landing_team in rookie_inputs.csv uses PFR-style 3-letter codes (KAN,
    LVR, NOR, NWE, SFO, TAM) that don't match risk_variables.csv's codes
    (KC, LV, NO, NE, SF, TB) -- confirmed via direct diff of both team-code
    sets (the other 25 codes already match). PFR_TEAM_ALIASES below is the
    real, complete fix (same pattern as build_risk_variables.py's own
    TEAM_CODE_ALIASES for the LAR/LA mismatch).
  - Confirmed in-season tier + sample size: data/processed/rb_archetypes.csv/
    wr_archetypes.csv, joined by player_name -- a rookie with real
    current-season games already gets his real confirmed archetype tier and
    a real sample_weight (see _sample_weight_wr/_sample_weight_rb below,
    reusing each module's own already-established real sample floor). A
    rookie with zero real games this season simply has no row in either
    file -- sample_weight correctly defaults to 0.0 (pure projected).

Usage:
    python -m draftkit.scripts.build_rookie_projections
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.rookie_projection import (  # noqa: E402
    blend_rookie_tag,
    qb_rookie_projection,
    rb_rookie_projection,
    wr_rookie_projection,
)
from draftkit.wr_archetypes import SAMPLE_FLOOR_GAMES as WR_SAMPLE_FLOOR_GAMES  # noqa: E402
from draftkit.wr_archetypes import SAMPLE_FLOOR_TARGETS  # noqa: E402
from draftkit.wr_archetypes import sample_confidence as wr_sample_confidence  # noqa: E402
from draftkit.rb_archetypes import sample_confidence as rb_sample_confidence  # noqa: E402
from draftkit.qb_archetypes import SAMPLE_FLOOR_ATTEMPTS as QB_SAMPLE_FLOOR_ATTEMPTS  # noqa: E402
from draftkit.qb_archetypes import SAMPLE_FLOOR_GAMES as QB_SAMPLE_FLOOR_GAMES  # noqa: E402
# qb_archetypes.py has no sample_confidence() of its own (classify_primary()
# gates directly on raw attempts/games, no separate call needed there) --
# reuses the same generic utility WR already imports (identical signature,
# same generic (actual, floor, games_played, min_games) shape).

ROOKIE_INPUTS_CSV = REPO_ROOT / "data" / "raw" / "rookie_inputs.csv"
# NOT imported from build_backtest_rookie_inputs.py -- that module imports
# PFR_TEAM_ALIASES from THIS one, so importing its OUTPUT_CSV back here
# would create a circular import. Same real path, defined independently.
BACKTEST_INPUTS_CSV = REPO_ROOT / "data" / "raw" / "backtest_rookie_inputs.csv"
# QB backtest-equivalent population (2021-2025 real draft classes,
# claude_code_plan_qb_rookie_projection.pdf) -- structurally the same role
# as BACKTEST_INPUTS_CSV, but QB never had one until this file existed.
# NOT the "2026 board" -- confirmed by direct inspection that
# rookie_inputs.csv's 10 real true-2026 QB rows (Mendoza, Simpson, Beck,
# Allar, Klubnik, Payton, Green, Kaliakmanis, Morton, Nussmeier) have ZERO
# name overlap with this CSV, which stops at the 2025 draft class. True
# 2026 rookie QBs have no real college stats source anywhere in this repo
# yet -- a real, honest gap, not fabricated (see the QB/TE `continue`
# branch below). UPDATE: closed for 6 of the 10 real true-2026 QBs (see
# QB_2026_SEASON_STATS_CSV/load_2026_qb_prospects() below) -- Payton/
# Kaliakmanis/Morton/Nussmeier still have no real source, still honestly
# unprojected.
QB_PROSPECTS_CSV = REPO_ROOT / "data" / "raw" / "qb_prospects_inputs.csv"
# True 2026 board -- real final-college-season stats for 6 of the 10 real
# true-2026 QBs (Simpson/Mendoza/Klubnik/Green/Beck/Allar). Combine
# measurables turned out redundant with what rookie_inputs.csv already
# carries for these players (confirmed directly, values match closely) --
# not re-sourced here.
QB_2026_SEASON_STATS_CSV = REPO_ROOT / "data" / "raw" / "2026_qb_season_stats.csv"
QB_ARCHETYPES_CSV = REPO_ROOT / "data" / "processed" / "qb_archetypes.csv"
RISK_VARIABLES_CSV = REPO_ROOT / "data" / "processed" / "risk_variables.csv"
RB_ARCHETYPES_CSV = REPO_ROOT / "data" / "processed" / "rb_archetypes.csv"
WR_ARCHETYPES_CSV = REPO_ROOT / "data" / "processed" / "wr_archetypes.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "rookie_projections.csv"

# claude_code_plan_qb_rookie_projection.pdf, Context items 1-3 (verified
# directly against qb_prospects_inputs.csv, not assumed from the plan PDF,
# which had two real factual errors -- see this session's chat record):
# actual count is 17 of 32 real QBs missing rushing splits, not 16, and
# Ehlinger was wrongly listed as having real data when his row is blank.
# Sam Ehlinger's real 2020 Texas season (382 rush yds, 8 rush TD) verified
# via web search (Hook 'Em Headlines/StatMuse). Bryce Young/C.J. Stroud/
# Michael Penix Jr. spot-checked directly by the user and confirmed
# genuinely low-rushing -- real, verified numbers, not assumed.
QB_ROOKIE_RUSHING_OVERRIDES = {
    "Sam Ehlinger": {"RushYds": 382, "RushTD": 8, "source": "Hook 'Em Headlines/StatMuse, verified via search", "date": "2026-08-17"},
    "Bryce Young": {"RushYds": 185, "RushTD": 4, "source": "user-verified spot-check", "date": "2026-08-17"},
    "C.J. Stroud": {"RushYds": 108, "RushTD": 0, "source": "user-verified spot-check", "date": "2026-08-17"},
    "Michael Penix Jr.": {"RushYds": 8, "RushTD": 3, "source": "user-verified spot-check", "date": "2026-08-17"},
}

# raw_draft_picks.csv (PFR-style) -> risk_variables.csv (standard) team
# codes. GNB (Green Bay) found via the 2023-2025 backtest (Jayden Reed/
# Matthew Golden were silently skipped) -- the other 6 were caught building
# the 2026 board. Historical-only codes (OAK/PHO/RAI/RAM/SDG/STL -- relocated
# or renamed franchises) are NOT included here: no current risk_variables.csv
# team uses them, and every backtest player's real landing team is an
# active, current franchise, so there's nothing real to map them to.
PFR_TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LVR": "LV", "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
}

# RB's classify_primary() doesn't call sample_confidence() directly (see
# rb_archetypes.py's _any_floor_cleared()/_meets_conditions() -- it gates on
# raw total_touches/targets/opportunity_share thresholds instead), so there's
# no single existing sample_confidence() call to reuse verbatim for RB the
# way there is for WR. Reasoned mapping, using RB's own already-established
# real constants: total_touches>=25 mirrors _any_floor_cleared()'s primary
# volume floor, games_played>=4 mirrors COMMITTEE_BACK's own real teammate-
# sample gate (see rb_archetypes.py's _meets_conditions()).
RB_SAMPLE_FLOOR_TOUCHES = 25
RB_SAMPLE_FLOOR_GAMES = 4


def load_team_context() -> dict:
    risk_df = pd.read_csv(RISK_VARIABLES_CSV)
    team_df = risk_df[["team", "offense_environment_score", "schedule_weather_venue_score"]].dropna()
    team_df = team_df.drop_duplicates("team")
    return team_df.set_index("team").to_dict(orient="index")


def load_confirmed_archetypes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = pd.read_csv(RB_ARCHETYPES_CSV) if RB_ARCHETYPES_CSV.exists() else pd.DataFrame()
    wr = pd.read_csv(WR_ARCHETYPES_CSV) if WR_ARCHETYPES_CSV.exists() else pd.DataFrame()
    return rb, wr


def _sample_weight_wr(row) -> float:
    if pd.isna(row.get("targets")):
        return 0.0
    return wr_sample_confidence(
        row.get("targets", 0.0), SAMPLE_FLOOR_TARGETS, row.get("games_played", 0.0), WR_SAMPLE_FLOOR_GAMES,
    )


def _sample_weight_rb(row) -> float:
    if pd.isna(row.get("total_touches")):
        return 0.0
    return rb_sample_confidence(
        row.get("total_touches", 0.0), RB_SAMPLE_FLOOR_TOUCHES, row.get("games_played", 0.0), RB_SAMPLE_FLOOR_GAMES,
    )


def _sample_weight_qb(row) -> float:
    """Mirrors _sample_weight_wr() -- reuses qb_archetypes.py's own
    already-established real SAMPLE_FLOOR_ATTEMPTS/GAMES and generic
    sample_confidence() utility, same as WR does for its own floor."""
    if pd.isna(row.get("attempts")):
        return 0.0
    return wr_sample_confidence(
        row.get("attempts", 0.0), QB_SAMPLE_FLOOR_ATTEMPTS, row.get("games", 0.0), QB_SAMPLE_FLOOR_GAMES,
    )


def load_qb_prospects(inputs_csv: Path = QB_PROSPECTS_CSV) -> pd.DataFrame:
    """Loads + reshapes qb_prospects_inputs.csv into the fields
    qb_rookie_projection() needs. Applies QB_ROOKIE_RUSHING_OVERRIDES for
    the 4 real, verified cases; the remaining real blanks resolve to
    rushing_data_status="assumed_non_rusher" with RushYds/RushTD treated as
    0 for the rushing_fantasy_pct calc -- NOT presented at the same
    confidence as a real measured value (see rushing_data_status, surfaced
    to Home.py for disclosure)."""
    df = pd.read_csv(inputs_csv)

    rush_yds, rush_td, status = [], [], []
    for _, row in df.iterrows():
        name = row["Player"]
        override = QB_ROOKIE_RUSHING_OVERRIDES.get(name)
        if override is not None:
            rush_yds.append(override["RushYds"])
            rush_td.append(override["RushTD"])
            status.append("measured_override")
        elif pd.notna(row.get("RushAtt")):
            rush_yds.append(row["RushYds"])
            rush_td.append(row["RushTD"])
            status.append("measured")
        else:
            rush_yds.append(0.0)
            rush_td.append(0.0)
            status.append("assumed_non_rusher")

    df["_rush_yds"] = rush_yds
    df["_rush_td"] = rush_td
    df["rushing_data_status"] = status

    rush_fp = df["_rush_yds"] * 0.10 + df["_rush_td"] * 6.0
    pass_fp = df["Yds"] * 0.04 + df["TD"] * 4.0
    total_fp = rush_fp + pass_fp
    df["rushing_fantasy_pct"] = (rush_fp / total_fp).where(total_fp > 0, 0.0)

    out = pd.DataFrame({
        "player_name": df["Player"],
        "draft_season": df["draft_class"],
        "draft_pick": df["draft_pick"],
        "landing_team": df["draft_team"].replace(PFR_TEAM_ALIASES),
        "college": df["Team"],  # real college name -- CSV's own "Team" column, not the NFL landing team
        "cmp": df["Cmp"],
        "att": df["Att"],
        "yds": df["Yds"],
        "td": df["TD"],
        "interceptions": df["Int"],
        "games": df["G"],
        "attempts": df["Att"],  # same real value as classify_primary()'s sample gate uses
        "rushing_fantasy_pct": df["rushing_fantasy_pct"],
        "rushing_data_status": df["rushing_data_status"],
    })
    return out


def load_2026_qb_prospects(
    season_stats_csv: Path = QB_2026_SEASON_STATS_CSV,
    rookie_inputs_csv: Path = ROOKIE_INPUTS_CSV,
) -> pd.DataFrame:
    """True 2026 board -- real, games/attempts-weighted college career
    stats for whichever real true-2026 QBs have a source (currently 6 of
    10, see QB_2026_SEASON_STATS_CSV's module comment).

    Pools EVERY real season row on file per player (not just the final
    one) -- real, explicit correction: an earlier version of this function
    used only each player's max-Season row, which meant Drew Allar's real
    injury-shortened 2025 season (6 games) was his ENTIRE signal, discarding
    a real, complete 2024 junior season (16 games, 394 attempts) that's
    obviously more representative. Same real distortion the veteran QB
    archetype's 2-season pool exists to correct for -- a single season
    shouldn't be trusted alone when more real seasons are on file. Not
    capped at 2 seasons the way the veteran pool is (that cap exists to
    bound NFL-career recency; a college career is already naturally
    bounded to ~4-5 years, and every real season here is part of the same
    developmental arc, not independent strategic noise) -- raw counts
    summed across all real rows, so a near-empty early season (Ty Simpson's
    2022: 4 games, 5 attempts) contributes almost nothing to the pooled
    total on its own, the same way volume-weighting already protects
    pool_qb_seasons() for veterans. draft_pick/landing_team/college/
    combine measurables come from rookie_inputs.csv (already real and
    present there -- confirmed directly, not re-sourced from the separate
    combine CSV, which turned out redundant)."""
    stats = pd.read_csv(season_stats_csv)
    for col in ("G", "Cmp", "Att", "Yds", "TD", "Int", "RushAtt", "RushYds", "RushTD"):
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.0)
    pooled = stats.groupby("Player", as_index=False)[
        ["G", "Cmp", "Att", "Yds", "TD", "Int", "RushYds", "RushTD"]
    ].sum()

    rookie_inputs = pd.read_csv(rookie_inputs_csv)
    qb_2026 = rookie_inputs[rookie_inputs["position"] == "QB"]

    merged = pooled.merge(
        qb_2026[["player_name", "draft_season", "draft_pick", "landing_team", "college", "height_inches", "weight", "forty_time"]],
        left_on="Player", right_on="player_name", how="inner",
    )

    rush_fp = merged["RushYds"] * 0.10 + merged["RushTD"] * 6.0
    pass_fp = merged["Yds"] * 0.04 + merged["TD"] * 4.0
    total_fp = rush_fp + pass_fp
    merged["rushing_fantasy_pct"] = (rush_fp / total_fp).where(total_fp > 0, 0.0)

    out = pd.DataFrame({
        "player_name": merged["Player"],
        "draft_season": merged["draft_season"],
        "draft_pick": merged["draft_pick"],
        "landing_team": merged["landing_team"].replace(PFR_TEAM_ALIASES),
        "college": merged["college"],
        "height_inches": merged["height_inches"],
        "weight": merged["weight"],
        "forty_time": merged["forty_time"],
        "cmp": merged["Cmp"],
        "att": merged["Att"],
        "yds": merged["Yds"],
        "td": merged["TD"],
        "interceptions": merged["Int"],
        "games": merged["G"],
        "attempts": merged["Att"],
        "rushing_fantasy_pct": merged["rushing_fantasy_pct"],
        "rushing_data_status": "measured",  # no blanks in any of the 6 real players' rushing rows -- verified directly
    })
    return out


def _row_to_inputs(row) -> dict:
    return {
        "draft_pick": int(row["draft_pick"]) if pd.notna(row.get("draft_pick")) else None,
        "landing_team": PFR_TEAM_ALIASES.get(row["landing_team"], row["landing_team"]),
        "college_dominator_final_year": row.get("college_dominator_final_year") if pd.notna(row.get("college_dominator_final_year")) else None,
        "college_dominator_career": row.get("college_dominator_career") if pd.notna(row.get("college_dominator_career")) else None,
        "breakout_age": row.get("breakout_age") if pd.notna(row.get("breakout_age")) else None,
        "roster_competition_tier": int(row["roster_competition_tier"]),
        "weight": row.get("weight") if pd.notna(row.get("weight")) else None,
        "forty_time": row.get("forty_time") if pd.notna(row.get("forty_time")) else None,
    }


def build_rookie_projections(inputs_csv: Path = ROOKIE_INPUTS_CSV) -> pd.DataFrame:
    """inputs_csv defaults to the current year's rookie_inputs.csv; pass
    data/raw/backtest_rookie_inputs.csv to project the 2023-2025 backtest
    classes instead (see draftkit/scripts/run_rookie_backtest.py) -- same
    projection formulas, same real confirmed-tier join, just a different
    real input source."""
    rookies = pd.read_csv(inputs_csv)
    team_context = load_team_context()
    rb_confirmed, wr_confirmed = load_confirmed_archetypes()

    records = []
    for _, row in rookies.iterrows():
        inputs = _row_to_inputs(row)
        team = inputs["landing_team"]
        if team not in team_context:
            print(f"[skip] {row['player_name']}: landing_team {team!r} not in risk_variables.csv team context")
            continue

        # Backtest-only: backtest_rookie_inputs.csv carries a real
        # historical_offense_environment_score (see
        # build_backtest_rookie_inputs.load_historical_offense_environment())
        # for the player's own real draft season, instead of today's team
        # context -- the 2026 board's rookie_inputs.csv has no such column,
        # so this is a no-op there. schedule_weather_venue_score has no
        # historical source (team_schedule_risk.csv carries no season
        # column) and stays on the current-season approximation either way.
        row_team_context = team_context
        hist_oe = row.get("historical_offense_environment_score")
        if pd.notna(hist_oe):
            row_team_context = dict(team_context)
            row_team_context[team] = {**team_context[team], "offense_environment_score": hist_oe}

        if row["position"] == "WR":
            projected = wr_rookie_projection(inputs, row_team_context)
            confirmed_rows = wr_confirmed[wr_confirmed["player_name"] == row["player_name"]]
            if not confirmed_rows.empty:
                crow = confirmed_rows.iloc[0]
                sample_weight = _sample_weight_wr(crow)
                confirmed_tag = crow.get("wr_archetype_primary")
            else:
                sample_weight, confirmed_tag = 0.0, None
        elif row["position"] == "RB":
            projected = rb_rookie_projection(inputs, row_team_context)
            confirmed_rows = rb_confirmed[rb_confirmed["player_name"] == row["player_name"]]
            if not confirmed_rows.empty:
                crow = confirmed_rows.iloc[0]
                sample_weight = _sample_weight_rb(crow)
                confirmed_tag = crow.get("archetype_primary")
            else:
                sample_weight, confirmed_tag = 0.0, None
        else:
            # TE: no confirmed-tier taxonomy exists yet, out of scope.
            # QB: real confirmed-tier taxonomy DOES now exist
            # (qb_archetypes.py) and real historical college data DOES
            # exist for 2021-2025 classes (see build_qb_rookie_projections()
            # below, a separate ingestion path -- qb_prospects_inputs.csv's
            # schema doesn't match this unified rookie_inputs.csv, so it
            # can't run through this same loop). This branch DOES still
            # correctly skip QB rows that reach it (the 10 real true-2026
            # QBs in rookie_inputs.csv -- Mendoza, Simpson, Beck, Allar,
            # Klubnik, Payton, Green, Kaliakmanis, Morton, Nussmeier):
            # confirmed via direct name-overlap check that qb_prospects_
            # inputs.csv has zero rows for any of them (it stops at the
            # 2025 draft class) -- a real, honest gap, not fabricated. This
            # comment previously claimed "not silently dropped" while doing
            # exactly that for every QB row that ever reached here -- a
            # stale, actively-misleading comment, corrected here rather
            # than left in place once found.
            continue

        # Real usage-sample confidence, NOT years-since-draft -- a
        # years-since-draft display cap was tried and reverted (see
        # blend_rookie_tag()'s docstring): it let a stale pre-draft
        # composite keep partially overriding an ALREADY-CONFIRMED real
        # archetype (Ashton Jeanty showing "Committee" despite a real,
        # unambiguous 2025 bellcow season) purely because too few calendar
        # years had passed. Once sample_weight clears 1.0 (real usage
        # cleared the sample floor -- see _sample_weight_wr/_sample_weight_rb
        # above), the confirmed read renders immediately and fully,
        # regardless of draft year.
        blend = blend_rookie_tag(projected, confirmed_tag, sample_weight)

        # display_* is a SEPARATE blend from status/tag/sample_weight above --
        # that pair stays real usage-sample confidence, untouched, because
        # run_rookie_backtest.py/validate_component_predictiveness.py depend
        # on "unconfirmed" as real, ranked ground truth (WR_OUTCOME_RANK/
        # RB_COARSE_TIER both rank it at 0, not exclude it -- e.g. it's the
        # real, correct read for Malik Nabers' pre-recency-fix injury year).
        # display_* is for the LIVE Home.py badge only: when the real tag is
        # literally "unconfirmed" (real snaps exist, but no real usage tier
        # was ever crossed), the college-based projection is the more
        # informative signal available, so it's treated exactly like "no
        # confirmed row at all" for display purposes -- badge falls back to
        # the real projected_tier (e.g. "Projected: Committee" for a real
        # committee-share back still short of the sample confidence needed
        # to call the read decisive) instead of showing the bare word
        # "Unconfirmed". A real, DECISIVE tag (bellcow, alpha, etc.) passes
        # through unchanged at full weight -- this must never re-hedge an
        # already-confirmed real archetype (that was the years-since-draft
        # mechanism's mistake, tried and reverted -- see blend_rookie_tag()'s
        # docstring).
        # Real bug, caught by direct user report (Braelon Allen showing
        # "Handcuff -- 84% confirmed"): forcing sample_weight_for_display
        # to 0.0 only for the literal "unconfirmed" case wasn't enough --
        # for any OTHER decisive tag, this used to leave the real generic
        # sample_weight (touches/games-based) in place, so blend_rookie_tag()
        # still took its partial-weight "blended" branch whenever that
        # generic confidence sat below 1.0. Low-volume-by-definition
        # archetypes (Handcuff) structurally can never clear a touches-
        # based confidence floor -- Allen's real classification (Handcuff)
        # was already fully decisive, but his generic sample_weight (0.84,
        # from total_touches/games) hedged it anyway. That's the exact
        # years-since-draft mistake this mechanism was built to prevent,
        # just reached through a different door. Fix: force 1.0 for ANY
        # decisive tag, not just pass the real sample_weight through --
        # blend_rookie_tag()'s own >=1.0 branch is what actually renders
        # solid, unhedged, no-percentage.
        # Real bug found via QB verification (Sam Ehlinger -- no adp/
        # projection_source in master_players.csv at all, a real long-
        # retired backup, so he never gets a row in qb_archetypes.csv and
        # confirmed_tag is literally None, not the string "unconfirmed").
        # The original condition only checked for "unconfirmed" and forced
        # sample_weight_for_display to 1.0 for confirmed_tag=None too --
        # producing a broken display_status="confirmed" with a None tag.
        # Latent in the RB/WR path too (never triggered there: every RB/WR
        # in rookie_inputs.csv/backtest_rookie_inputs.csv is currently
        # ADP-relevant, so always gets SOME row, even "unconfirmed" --
        # QB's historical backtest population is the first to include a
        # real player with no row at all). Both real "no decisive tag"
        # cases -- no row, or a row that says literally unconfirmed --
        # must fall back to the projected display the same way.
        no_decisive_tag = confirmed_tag is None or confirmed_tag == "unconfirmed"
        confirmed_tag_for_display = None if no_decisive_tag else confirmed_tag
        sample_weight_for_display = 0.0 if no_decisive_tag else 1.0
        display_blend = blend_rookie_tag(projected, confirmed_tag_for_display, sample_weight_for_display)

        records.append({
            "player_name": row["player_name"],
            "position": row["position"],
            "draft_season": int(row["draft_season"]),
            "draft_pick": row["draft_pick"],
            "landing_team": team,
            "college": row.get("college"),
            "height_inches": row.get("height_inches"),
            "weight": row.get("weight"),
            "forty_time": row.get("forty_time"),
            "college_dominator_final_year": row.get("college_dominator_final_year"),
            "college_dominator_career": row.get("college_dominator_career"),
            "breakout_age": row.get("breakout_age"),
            "projected_tier": projected["projected_tier"],
            "composite": projected["composite"],
            "sample_weight": round(sample_weight, 3),
            "status": blend["status"],
            "tag": blend.get("tag"),
            "display_weight": round(sample_weight_for_display, 3),
            "display_status": display_blend["status"],
            "display_tag": display_blend.get("tag"),
            **{f"component_{k}": v for k, v in projected["components"].items()},
        })

    return pd.DataFrame.from_records(records)


def build_qb_rookie_projections(inputs_csv: Path = QB_PROSPECTS_CSV, prospects_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """QB's own ingestion path -- see QB_PROSPECTS_CSV's module-level
    comment for why this can't run through the shared RB/WR loop above
    (different real data source/schema; the 2021-2025 QB classes here are
    structurally a backtest population, not a "2026 board"). Same real
    confirmed-tier join + dual status/display blend as RB/WR, reusing
    qb_archetypes.csv/_sample_weight_qb()/blend_rookie_tag() directly.

    prospects_df lets a caller pass an already-loaded population (e.g.
    load_2026_qb_prospects()'s true-2026 board) instead of re-reading
    inputs_csv -- same real formulas/joins apply either way, just a
    different real source."""
    qbs = prospects_df if prospects_df is not None else load_qb_prospects(inputs_csv)
    team_context = load_team_context()
    qb_confirmed = pd.read_csv(QB_ARCHETYPES_CSV) if QB_ARCHETYPES_CSV.exists() else pd.DataFrame()

    records = []
    for _, row in qbs.iterrows():
        team = row["landing_team"]
        if team not in team_context:
            print(f"[skip] {row['player_name']}: landing_team {team!r} not in risk_variables.csv team context")
            continue

        inputs = {
            "draft_pick": int(row["draft_pick"]) if pd.notna(row.get("draft_pick")) else None,
            "landing_team": team,
            "cmp": row["cmp"], "att": row["att"], "yds": row["yds"],
            "td": row["td"], "interceptions": row["interceptions"],
            "games": row["games"], "rushing_fantasy_pct": row["rushing_fantasy_pct"],
        }
        projected = qb_rookie_projection(inputs, team_context)

        confirmed_rows = qb_confirmed[qb_confirmed["player_name"] == row["player_name"]] if not qb_confirmed.empty else qb_confirmed
        if not confirmed_rows.empty:
            crow = confirmed_rows.iloc[0]
            sample_weight = _sample_weight_qb(crow)
            confirmed_tag = crow.get("qb_archetype_primary")
        else:
            sample_weight, confirmed_tag = 0.0, None

        # Same dual status/display blend as RB/WR -- see the long comment
        # block above (~line 282) for why this exists and the real Braelon
        # Allen bug it was built to prevent. Reused verbatim, not
        # reinvented, for QB.
        blend = blend_rookie_tag(projected, confirmed_tag, sample_weight)
        # Real bug found via QB verification (Sam Ehlinger -- no adp/
        # projection_source in master_players.csv at all, a real long-
        # retired backup, so he never gets a row in qb_archetypes.csv and
        # confirmed_tag is literally None, not the string "unconfirmed").
        # The original condition only checked for "unconfirmed" and forced
        # sample_weight_for_display to 1.0 for confirmed_tag=None too --
        # producing a broken display_status="confirmed" with a None tag.
        # Latent in the RB/WR path too (never triggered there: every RB/WR
        # in rookie_inputs.csv/backtest_rookie_inputs.csv is currently
        # ADP-relevant, so always gets SOME row, even "unconfirmed" --
        # QB's historical backtest population is the first to include a
        # real player with no row at all). Both real "no decisive tag"
        # cases -- no row, or a row that says literally unconfirmed --
        # must fall back to the projected display the same way.
        no_decisive_tag = confirmed_tag is None or confirmed_tag == "unconfirmed"
        confirmed_tag_for_display = None if no_decisive_tag else confirmed_tag
        sample_weight_for_display = 0.0 if no_decisive_tag else 1.0
        display_blend = blend_rookie_tag(projected, confirmed_tag_for_display, sample_weight_for_display)

        records.append({
            "player_name": row["player_name"],
            "position": "QB",
            "draft_season": int(row["draft_season"]),
            "draft_pick": row["draft_pick"],
            "landing_team": team,
            "college": row.get("college"),
            "height_inches": row.get("height_inches"),
            "weight": row.get("weight"),
            "forty_time": row.get("forty_time"),
            "games": row.get("games"),
            "attempts": row.get("attempts"),
            "rushing_data_status": row["rushing_data_status"],
            "projected_tier": projected["projected_tier"],
            "composite": projected["composite"],
            "sample_weight": round(sample_weight, 3),
            "status": blend["status"],
            "tag": blend.get("tag"),
            "display_weight": round(sample_weight_for_display, 3),
            "display_status": display_blend["status"],
            "display_tag": display_blend.get("tag"),
            **{f"component_{k}": v for k, v in projected["components"].items()},
        })

    return pd.DataFrame.from_records(records)


def main() -> int:
    # Both real pools, combined -- the 2026 board (true pre-season rookies,
    # draft_season==2026 always) and the 2023-2025 backtest pool (real
    # confirmed veterans with prospect data on file, e.g. Bijan Robinson,
    # Jahmyr Gibbs, Cam Skattebo, Bhayshul Tuten). No real overlap is
    # possible between them (a player has exactly one real draft season),
    # so concatenation needs no dedup logic. This is the one change that
    # makes backtest-pool players reach Home.py's live board at all --
    # previously this file only ever held the 2026 pool, and the backtest
    # computation was run ad-hoc, never written anywhere Home.py reads.
    board_2026 = build_rookie_projections()
    board_backtest = build_rookie_projections(inputs_csv=BACKTEST_INPUTS_CSV)
    # QB backtest-equivalent population (2021-2025 real draft classes) --
    # its own ingestion path, see build_qb_rookie_projections()'s docstring.
    board_qb_backtest = build_qb_rookie_projections()
    # True 2026 QB board -- closes the gap for 6 of the 10 real true-2026
    # QBs (see QB_2026_SEASON_STATS_CSV's module comment). The other 4
    # (Payton/Kaliakmanis/Morton/Nussmeier) still have no real data source
    # -- correctly get no projection rather than a fabricated one.
    board_qb_2026 = build_qb_rookie_projections(prospects_df=load_2026_qb_prospects())
    board = pd.concat([board_2026, board_backtest, board_qb_backtest, board_qb_2026], ignore_index=True)

    # Real landing-spot context (target competition, backfield competition,
    # O-line quality) can shift after this file is generated -- a trade, an
    # injury to a projected starter, a camp battle resolving -- and nothing
    # else in this output signals staleness. Phase 1 of the rookie
    # integration plan.
    board["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)
    board_qb = pd.concat([board_qb_backtest, board_qb_2026], ignore_index=True)
    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s) ({len(board_2026)} 2026 + {len(board_backtest)} backtest + "
          f"{len(board_qb_backtest)} QB backtest + {len(board_qb_2026)} QB 2026 board)")
    print(board["status"].value_counts().to_string())
    print(board.groupby("position")["projected_tier"].value_counts().to_string())
    if not board_qb.empty:
        print()
        print("QB rushing_data_status breakdown (measured / measured_override / assumed_non_rusher):")
        print(board_qb["rushing_data_status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
