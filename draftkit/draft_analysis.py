from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from draftkit.data_access import get_available_players_df, load_players_df, safe_col
from draftkit.construction_pressure import (
    calculate_construction_adjustment,
    calculate_construction_pressure,
)
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_team_positions,
    get_next_pick_distance,
)


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
TEAM_COLS = ["team", "Team", "team_abbr"]
PROJECTION_COLS = ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"]
ADP_COLS = ["adp", "ADP", "rank", "Rank"]
ARCHETYPE_COLS = ["archetype", "Archetype", "player_archetype", "profile", "risk_profile"]
INJURY_RISK_COLS = ["injury_risk", "Injury Risk", "injury_score", "risk_score"]
DURABILITY_COLS = ["durability_grade", "Durability", "durability", "durability_score"]
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
SUPPORTED_ARCHETYPES = ["BOOM", "STEADY", "RISKY", "UPSIDE", "SAFE"]
# Positional multipliers are all 1.00 deliberately: position_value_score
# is already value-over-replacement (projection minus the position's
# replacement baseline), which IS the positional-scarcity adjustment.
# Multiplying it again double-counts scarcity.
#
# These carried TE penalties (value 0.68, urgency 0.70, need cap 1.18)
# until they were backtested against 16 seasons of realized outcomes in
# research/validation_v1/backtest_positional_multipliers_v1.py. The data
# says the opposite of what they assumed -- early-round TEs returned the
# HIGHEST value over replacement of any position, not the lowest:
#
#   ADP 1-48, mean realized VOR:  TE +75.3 | QB +54.9 | RB +46.7 | WR +20.4
#   TE - RB difference: +28.6 pts, bootstrap CI [+4.7, +52.2] (excludes 0)
#   2018+ only:         +40.7 pts, bootstrap CI [+9.8, +71.9] (excludes 0)
#   TE was the top position in EVERY ADP bucket tested.
#
# The penalties were measurably harmful: they removed every tight end from
# the board's top 50 (ADP's top 50 has three), pushing Brock Bowers from
# ADP 19.7 to board rank 147 and Sam LaPorta from 91.7 to 388.
#
# Do not reintroduce a positional multiplier without backtesting it the
# same way. If TE scarcity needs expressing, it belongs in the replacement
# baseline, not as a second multiplier on top of it.
POSITION_VALUE_MULTIPLIERS = {
    "QB": 1.00,
    "RB": 1.00,
    "WR": 1.00,
    "TE": 1.00,
}
# Empty by design. This existed to stop the unfilled-TE need bonus running
# away, but it was treating the symptom: the real defect was that strategy
# components could lift a replacement-level player regardless of position.
# That is now fixed at the source in calculate_final_recommendation_score(),
# which damps strategy terms toward neutral for players with no value over
# replacement. Capping the need weight on top of that would just re-apply a
# TE penalty the VOR backtest already contradicted.
POSITION_NEED_WEIGHT_CAPS = {}
POSITION_URGENCY_MULTIPLIERS = {
    "QB": 1.00,
    "RB": 1.00,
    "WR": 1.00,
    "TE": 1.00,
}
RB_WR_BALANCE_MANDATE_THRESHOLD = 3
CONSTRUCTION_MANDATE_BOOST = 15.0
CONSTRUCTION_MANDATE_NON_TARGET_MULTIPLIER = 0.35
# Smooth multipliers replacing the old flat SINGLE_QB_SCORE_CAP -- see
# apply_single_qb_value_adjustments() for why a hard cap was removed.
SINGLE_QB_BASE_MULTIPLIER = 0.72
SINGLE_QB_HIGH_PRESSURE_MULTIPLIER = 0.85
SINGLE_QB_SEVERE_PRESSURE_MULTIPLIER = 0.92
DEFAULT_MASTER_COMPONENT_WEIGHTS = {
    "projection": 0.30,
    "position_need": 0.25,
    "adp_value": 0.15,
    "tier_urgency": 0.15,
    "team_fit": 0.15,
}
MASTER_COMPONENT_COLUMNS = {
    "projection": ("position_value_score", "projection_component_score"),
    "position_need": ("need_bonus", "position_need_component_score"),
    "adp_value": ("value_score", "adp_value_component_score"),
    "tier_urgency": ("urgency_score", "tier_urgency_component_score"),
    "team_fit": ("team_fit_bonus", "team_fit_component_score"),
}


def _get_analysis_columns(df):
    return {
        "player_col": safe_col(df, PLAYER_COLS),
        "pos_col": safe_col(df, POSITION_COLS),
        "team_col": safe_col(df, TEAM_COLS),
        "proj_col": safe_col(df, PROJECTION_COLS),
        "adp_col": safe_col(df, ADP_COLS),
        "archetype_col": safe_col(df, ARCHETYPE_COLS),
        "injury_risk_col": safe_col(df, INJURY_RISK_COLS),
        "durability_col": safe_col(df, DURABILITY_COLS),
    }


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_score(value, default):
    score = _safe_float(value, default)
    return max(0.0, min(100.0, score))


def _normalize_durability(value, default=70.0):
    if isinstance(value, str):
        grade = value.strip().upper()
        grade_map = {
            "A+": 98.0,
            "A": 94.0,
            "A-": 90.0,
            "B+": 86.0,
            "B": 82.0,
            "B-": 78.0,
            "C+": 74.0,
            "C": 70.0,
            "C-": 66.0,
            "D": 58.0,
            "F": 45.0,
        }
        if grade in grade_map:
            return grade_map[grade]

    return _normalize_score(value, default)


def _get_row_value(row, column, default=None):
    if column is None:
        return default
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return value


def _build_scoring_context(df, columns):
    proj_col = columns.get("proj_col")
    adp_col = columns.get("adp_col")

    context = {
        "median_projection": 0.0,
        "high_projection": 0.0,
        "median_adp": 100.0,
        "late_adp": 100.0,
    }

    if df.empty:
        return context

    if proj_col is not None and proj_col in df.columns:
        projections = pd.to_numeric(df[proj_col], errors="coerce").dropna()
        if not projections.empty:
            context["median_projection"] = float(projections.median())
            context["high_projection"] = float(projections.quantile(0.75))

    if adp_col is not None and adp_col in df.columns:
        adps = pd.to_numeric(df[adp_col], errors="coerce").dropna()
        if not adps.empty:
            context["median_adp"] = float(adps.median())
            context["late_adp"] = float(adps.quantile(0.75))

    return context


def get_player_injury_risk(player_row, columns=None):
    columns = columns or {}
    return _normalize_score(
        _get_row_value(player_row, columns.get("injury_risk_col"), 50.0),
        50.0,
    )


def get_player_durability(player_row, columns=None):
    columns = columns or {}
    return _normalize_durability(
        _get_row_value(player_row, columns.get("durability_col"), 70.0),
        70.0,
    )


def derive_player_archetype(player_row, columns=None, context=None):
    columns = columns or {}
    context = context or {}

    archetype_col = columns.get("archetype_col")
    if archetype_col is not None:
        raw_archetype = str(_get_row_value(player_row, archetype_col, "")).strip().upper()
        if raw_archetype in SUPPORTED_ARCHETYPES:
            return raw_archetype

    projection = _safe_float(
        _get_row_value(player_row, columns.get("proj_col"), 0.0),
        0.0,
    )
    adp = _safe_float(
        _get_row_value(player_row, columns.get("adp_col"), context.get("median_adp", 100.0)),
        context.get("median_adp", 100.0),
    )
    injury_risk = get_player_injury_risk(player_row, columns)
    durability = get_player_durability(player_row, columns)

    median_projection = context.get("median_projection", 0.0)
    high_projection = context.get("high_projection", median_projection)
    median_adp = context.get("median_adp", 100.0)
    late_adp = context.get("late_adp", median_adp)

    if injury_risk >= 70 or durability <= 45:
        return "RISKY"
    if durability >= 85 and injury_risk <= 35:
        return "SAFE"
    if projection >= high_projection and adp > median_adp:
        return "UPSIDE"
    if adp >= late_adp and projection >= median_projection:
        return "BOOM"
    if projection >= median_projection:
        return "STEADY"

    return "SAFE"


def fill_missing_projection_points(df, pos_col, proj_col):
    """
    Fill missing projection_points with a per-position replacement-level
    estimate instead of dropping the player from every ranking/tier table.

    A player with no real projection but real ADP/tier data (e.g. a rookie
    a full projections export hasn't caught up to yet) should still show up
    in the board -- ranked conservatively -- rather than vanish silently.
    Uses 90% of the *worst* real projection at that position as the fallback
    anchor, so unprojected players can never accidentally outrank a real,
    worse-projected player. (An earlier version used the 25th percentile,
    which broke down when the only "real" data available was an elite-only
    sample -- the 25th percentile of an elite-only sample is still elite,
    not replacement level.)

    Adds a 'projection_source' column: 'real' or 'replacement_fallback'.
    """
    df = df.copy()
    df[proj_col] = pd.to_numeric(df[proj_col], errors="coerce")
    df["projection_source"] = df[proj_col].apply(
        lambda value: "real" if pd.notna(value) else "replacement_fallback"
    )

    position_fallback = {}
    for position in SCORABLE_POSITIONS:
        real_values = df.loc[
            (df[pos_col].astype(str).str.upper() == position) & df[proj_col].notna(),
            proj_col,
        ]
        if not real_values.empty:
            position_fallback[position] = float(real_values.min()) * 0.9

    def _fallback_value(row):
        if pd.notna(row[proj_col]):
            return row[proj_col]
        position = str(row[pos_col]).upper()
        return position_fallback.get(position, 0.0)

    df[proj_col] = df.apply(_fallback_value, axis=1)
    return df


def _prep_ranked_df():
    df = get_available_players_df().copy()
    if df.empty:
        return pd.DataFrame(), None, None, None, None, None

    player_col = safe_col(df, ["player_name", "Player", "player", "name", "full_name"])
    pos_col = safe_col(df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(df, ["team", "Team", "team_abbr"])
    proj_col = safe_col(df, ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"])
    adp_col = safe_col(df, ["adp", "ADP", "rank", "Rank"])

    if player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame(), None, None, None, None, None

    df = df[df[pos_col].astype(str).isin(["QB", "RB", "WR", "TE"])].copy()
    df = fill_missing_projection_points(df, pos_col, proj_col)
    df = df.sort_values([pos_col, proj_col], ascending=[True, False]).reset_index(drop=True)

    return df, player_col, pos_col, team_col, proj_col, adp_col


def get_tier_build_debug_info():
    raw_df = get_available_players_df().copy()

    info = {
        "raw_shape": raw_df.shape,
        "raw_columns": list(raw_df.columns),
        "player_col": None,
        "pos_col": None,
        "team_col": None,
        "proj_col": None,
        "adp_col": None,
        "filtered_shape": (0, 0),
        "position_values": [],
        "projection_dtype": None,
        "projection_non_null_count": 0,
        "message": ""
    }

    if raw_df.empty:
        info["message"] = "Available player DataFrame is empty."
        return info

    player_col = safe_col(raw_df, ["player_name", "Player", "player", "name", "full_name"])
    pos_col = safe_col(raw_df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(raw_df, ["team", "Team", "team_abbr"])
    proj_col = safe_col(raw_df, ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"])
    adp_col = safe_col(raw_df, ["adp", "ADP", "rank", "Rank"])

    info["player_col"] = player_col
    info["pos_col"] = pos_col
    info["team_col"] = team_col
    info["proj_col"] = proj_col
    info["adp_col"] = adp_col

    if player_col is None or pos_col is None or proj_col is None:
        info["message"] = "Missing one or more required columns: player, position, or projection."
        return info

    debug_df = raw_df.copy()
    info["position_values"] = sorted(debug_df[pos_col].astype(str).dropna().unique().tolist())
    info["projection_dtype"] = str(debug_df[proj_col].dtype)

    debug_df[proj_col] = pd.to_numeric(debug_df[proj_col], errors="coerce")
    info["projection_non_null_count"] = int(debug_df[proj_col].notna().sum())

    debug_df = debug_df[debug_df[pos_col].astype(str).isin(["QB", "RB", "WR", "TE"])].copy()
    debug_df = debug_df.dropna(subset=[proj_col]).copy()

    info["filtered_shape"] = debug_df.shape

    if debug_df.empty:
        info["message"] = "Rows disappeared after position filtering or numeric projection cleanup."
    else:
        info["message"] = "Tier input data looks usable."

    return info


def get_position_need_weights():
    roster = st.session_state.get("roster_settings", {})
    raw_df = load_players_df().copy()
    position_counts = get_my_team_positions(raw_df)
    weights = {}

    for pos in ["QB", "RB", "WR", "TE"]:
        target = int(roster.get(pos, 0))
        have = position_counts.get(pos, 0)

        if target <= 0:
            weights[pos] = 0.2
        elif have <= 0:
            weights[pos] = 1.5
        elif have < target:
            gap = target - have
            weights[pos] = min(1.45, 1.15 + (0.10 * gap))
        elif have == target:
            weights[pos] = 0.8
        else:
            weights[pos] = 0.6

        if pos in POSITION_NEED_WEIGHT_CAPS:
            weights[pos] = min(weights[pos], POSITION_NEED_WEIGHT_CAPS[pos])

    return weights


def get_construction_mandate(position_counts=None, roster_settings=None):
    """
    Return hard roster-construction mandates that should override soft ranking inputs.

    Example: if the roster starts WR/WR/WR while RB is still empty, the next pick
    should be an RB. The inverse is also true for RB/RB/RB with no WRs.
    """
    roster_settings = roster_settings or st.session_state.get("roster_settings", {})
    if position_counts is None:
        raw_df = load_players_df().copy()
        position_counts = get_my_team_positions(raw_df)

    rb_target = int(roster_settings.get("RB", 0))
    wr_target = int(roster_settings.get("WR", 0))
    rb_count = int(position_counts.get("RB", 0))
    wr_count = int(position_counts.get("WR", 0))

    if (
        rb_target > 0
        and rb_count <= 0
        and wr_count >= RB_WR_BALANCE_MANDATE_THRESHOLD
    ):
        severity = "severe" if wr_count >= RB_WR_BALANCE_MANDATE_THRESHOLD + 2 else "required"
        return {
            "target_position": "RB",
            "severity": severity,
            "reason": (
                f"Roster has {wr_count} WRs and 0 RBs. "
                "RB is a mandatory construction pick."
            ),
            "wr_count": wr_count,
            "rb_count": rb_count,
            "rb_target": rb_target,
        }

    if (
        wr_target > 0
        and wr_count <= 0
        and rb_count >= RB_WR_BALANCE_MANDATE_THRESHOLD
    ):
        severity = "severe" if rb_count >= RB_WR_BALANCE_MANDATE_THRESHOLD + 2 else "required"
        return {
            "target_position": "WR",
            "severity": severity,
            "reason": (
                f"Roster has {rb_count} RBs and 0 WRs. "
                "WR is a mandatory construction pick."
            ),
            "wr_count": wr_count,
            "rb_count": rb_count,
            "wr_target": wr_target,
        }

    return None


def calculate_context_score(player_row, projection, position, need_weight):
    """
    Calculate the roster-aware score for a player.

    Version 1 intentionally stays simple so future ADP, tier, scarcity, or ML
    weighting can replace this without changing callers.
    """
    try:
        projection_points = float(projection)
    except (TypeError, ValueError):
        projection_points = 0.0

    try:
        weight = float(need_weight)
    except (TypeError, ValueError):
        weight = 1.0

    return round(projection_points * weight, 2)


def build_context_rankings_df():
    """
    Rank available QB/RB/WR/TE players by roster-aware context score.
    """
    df = get_available_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    player_col = safe_col(df, ["player_name", "Player", "player", "name", "full_name"])
    pos_col = safe_col(df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(df, ["team", "Team", "team_abbr"])
    proj_col = safe_col(df, ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"])
    adp_col = safe_col(df, ["adp", "ADP", "rank", "Rank"])

    if player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame()

    df = df[df[pos_col].astype(str).str.upper().isin(["QB", "RB", "WR", "TE"])].copy()
    if df.empty:
        return pd.DataFrame()

    df[proj_col] = pd.to_numeric(df[proj_col], errors="coerce")
    df = df.dropna(subset=[proj_col]).copy()
    if df.empty:
        return pd.DataFrame()

    need_weights = get_position_need_weights()
    df["position"] = df[pos_col].astype(str).str.upper()
    df["need_weight"] = df["position"].map(need_weights).fillna(1.0)
    df["context_score"] = df.apply(
        lambda row: calculate_context_score(
            row,
            row[proj_col],
            row["position"],
            row["need_weight"],
        ),
        axis=1,
    )

    output_cols = [player_col, "position", proj_col, "need_weight", "context_score"]
    rename_map = {
        player_col: "player_name",
        proj_col: "projection_points",
    }

    if team_col is not None:
        output_cols.insert(2, team_col)
        rename_map[team_col] = "team"

    if adp_col is not None:
        output_cols.append(adp_col)
        rename_map[adp_col] = "adp"

    rankings_df = df[output_cols].rename(columns=rename_map)
    rankings_df = rankings_df.sort_values(
        ["context_score", "projection_points"],
        ascending=False,
    ).reset_index(drop=True)

    return rankings_df


def get_best_context_target():
    """
    Return the highest ranked available player by roster-aware context score.
    """
    rankings_df = build_context_rankings_df()
    if rankings_df.empty:
        return None

    return rankings_df.iloc[0].to_dict()


def get_context_scoring_debug_info():
    """
    Return debug information for roster-aware scoring validation.
    """
    raw_df = load_players_df().copy()
    available_df = get_available_players_df().copy()
    rankings_df = build_context_rankings_df()

    return {
        "raw_shape": raw_df.shape,
        "available_shape": available_df.shape,
        "current_roster_positions": get_my_team_positions(raw_df),
        "need_weights": get_position_need_weights(),
        "top_context_targets": rankings_df.head(10).to_dict("records")
        if not rankings_df.empty
        else [],
        "player_col": safe_col(available_df, ["player_name", "Player", "player", "name", "full_name"])
        if not available_df.empty
        else None,
        "pos_col": safe_col(available_df, ["position", "pos", "Pos", "Position"])
        if not available_df.empty
        else None,
        "proj_col": safe_col(available_df, ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"])
        if not available_df.empty
        else None,
        "adp_col": safe_col(available_df, ["adp", "ADP", "rank", "Rank"])
        if not available_df.empty
        else None,
    }


def get_team_profile():
    """
    Analyze the current roster's construction, risk, and archetype mix.
    """
    raw_df = load_players_df().copy()
    columns = _get_analysis_columns(raw_df)
    player_col = columns.get("player_col")
    position_counts = get_my_team_positions(raw_df)
    my_team = st.session_state.get("my_team", [])

    empty_profile = {
        "profile": "BALANCED",
        "position_counts": position_counts,
        "average_injury_risk": 50.0,
        "average_durability": 70.0,
        "archetype_distribution": {archetype: 0 for archetype in SUPPORTED_ARCHETYPES},
        "volatility": "Moderate",
        "volatility_score": 50.0,
        "boom_players": 0,
        "steady_players": 0,
        "risky_players": 0,
        "upside_players": 0,
        "safe_players": 0,
        "roster_size": len(my_team),
    }

    if raw_df.empty or player_col is None or not my_team:
        return empty_profile

    roster_names = {str(name).strip().lower() for name in my_team}
    roster_df = raw_df[
        raw_df[player_col].astype(str).str.strip().str.lower().isin(roster_names)
    ].copy()

    if roster_df.empty:
        return empty_profile

    context = _build_scoring_context(raw_df, columns)
    archetype_counts = {archetype: 0 for archetype in SUPPORTED_ARCHETYPES}
    injury_risks = []
    durability_scores = []

    for _, row in roster_df.iterrows():
        archetype = derive_player_archetype(row, columns, context)
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        injury_risks.append(get_player_injury_risk(row, columns))
        durability_scores.append(get_player_durability(row, columns))

    roster_size = len(roster_df)
    average_injury_risk = round(sum(injury_risks) / roster_size, 1)
    average_durability = round(sum(durability_scores) / roster_size, 1)
    volatile_players = (
        archetype_counts.get("BOOM", 0)
        + archetype_counts.get("RISKY", 0)
        + archetype_counts.get("UPSIDE", 0)
    )
    volatile_share = volatile_players / roster_size if roster_size else 0.0
    volatility_score = round((volatile_share * 70.0) + (average_injury_risk * 0.30), 1)

    if volatility_score >= 65:
        volatility = "High"
    elif volatility_score >= 40:
        volatility = "Moderate"
    else:
        volatility = "Low"

    boom_share = archetype_counts.get("BOOM", 0) / roster_size
    upside_share = archetype_counts.get("UPSIDE", 0) / roster_size
    risky_share = archetype_counts.get("RISKY", 0) / roster_size
    safe_steady_share = (
        archetype_counts.get("SAFE", 0) + archetype_counts.get("STEADY", 0)
    ) / roster_size

    if average_injury_risk >= 70 or risky_share >= 0.35:
        profile = "HIGH RISK"
    elif boom_share + upside_share >= 0.50:
        profile = "HIGH UPSIDE"
    elif safe_steady_share >= 0.65 and average_injury_risk <= 45:
        profile = "SAFE"
    elif boom_share + upside_share >= 0.35:
        profile = "AGGRESSIVE"
    else:
        profile = "BALANCED"

    return {
        "profile": profile,
        "position_counts": position_counts,
        "average_injury_risk": average_injury_risk,
        "average_durability": average_durability,
        "archetype_distribution": archetype_counts,
        "volatility": volatility,
        "volatility_score": volatility_score,
        "boom_players": archetype_counts.get("BOOM", 0),
        "steady_players": archetype_counts.get("STEADY", 0),
        "risky_players": archetype_counts.get("RISKY", 0),
        "upside_players": archetype_counts.get("UPSIDE", 0),
        "safe_players": archetype_counts.get("SAFE", 0),
        "roster_size": roster_size,
    }


def calculate_team_fit_score(player_row, team_profile=None, columns=None, context=None):
    """
    Return a transparent multiplier describing roster-construction fit.
    """
    team_profile = team_profile or get_team_profile()
    columns = columns or {}
    context = context or {}

    archetype = derive_player_archetype(player_row, columns, context)
    injury_risk = get_player_injury_risk(player_row, columns)
    durability = get_player_durability(player_row, columns)

    fit_bonus = 1.0

    if team_profile.get("average_injury_risk", 50.0) >= 60:
        if injury_risk <= 40 or durability >= 80 or archetype in ["SAFE", "STEADY"]:
            fit_bonus += 0.08
        if injury_risk >= 70 or archetype == "RISKY":
            fit_bonus -= 0.10

    if team_profile.get("volatility") == "High":
        if archetype in ["SAFE", "STEADY"]:
            fit_bonus += 0.08
        if archetype in ["BOOM", "RISKY"]:
            fit_bonus -= 0.08
    elif team_profile.get("volatility") == "Low":
        if archetype in ["BOOM", "UPSIDE"]:
            fit_bonus += 0.05

    if team_profile.get("profile") in ["HIGH RISK", "AGGRESSIVE"]:
        if archetype in ["SAFE", "STEADY"] and durability >= 70:
            fit_bonus += 0.05
    elif team_profile.get("profile") == "SAFE":
        if archetype in ["UPSIDE", "BOOM"] and injury_risk <= 60:
            fit_bonus += 0.04

    if durability <= 45:
        fit_bonus -= 0.05
    elif durability >= 85:
        fit_bonus += 0.03

    return round(max(0.80, min(1.20, fit_bonus)), 3)


def calculate_adp_bonus(projection_rank, adp_value):
    if adp_value is None or pd.isna(adp_value):
        return 1.0

    adp = _safe_float(adp_value, projection_rank)
    value_delta = adp - float(projection_rank)
    bonus = 1.0 + max(-0.10, min(0.15, value_delta / 100.0))
    return round(bonus, 3)


def calculate_adp_delta(projection_rank, adp_rank):
    """
    Positive delta means a player is cheaper than their projection rank suggests.
    Negative delta means a player is more expensive than their projection rank suggests.
    """
    projection_rank = _safe_float(projection_rank, 0.0)
    adp_rank = _safe_float(adp_rank, projection_rank)

    return round(adp_rank - projection_rank, 2)


def calculate_value_score(adp_delta, projection_points, current_pick, adp_rank):
    """
    Score draft value by combining projection strength, ADP discount, and current cost.
    """
    delta = _safe_float(adp_delta, 0.0)
    projection = max(_safe_float(projection_points, 0.0), 0.0)
    pick = max(_safe_float(current_pick, 1.0), 1.0)
    adp = max(_safe_float(adp_rank, pick), 1.0)

    projection_score = min(projection / 6.0, 60.0)
    discount_score = max(-30.0, min(delta * 1.5, 40.0))
    current_discount_score = max(0.0, min((pick - adp) * 0.75, 15.0))

    return round(projection_score + discount_score + current_discount_score, 2)


def classify_player_value(adp_delta, value_score):
    delta = _safe_float(adp_delta, 0.0)
    score = _safe_float(value_score, 0.0)

    if delta >= 20 and score >= 70:
        return "STEAL"
    if delta >= 8 or score >= 60:
        return "VALUE"
    if delta > -8:
        return "FAIR"
    if delta >= -20:
        return "REACH"

    return "AVOID"


def calculate_fall_risk(adp_rank, current_pick=None, next_pick_distance=None):
    """
    Estimate whether a player is likely to survive until the user's next pick.
    """
    if current_pick is None:
        current_pick = get_current_pick_number()
    if next_pick_distance is None:
        next_pick_distance = get_next_pick_distance()

    pick = max(_safe_float(current_pick, 1.0), 1.0)
    wait = next_pick_distance if next_pick_distance is not None else 3
    wait = max(_safe_float(wait, 3.0), 0.0)
    next_pick = pick + wait
    adp = _safe_float(adp_rank, next_pick + 99.0)

    if adp <= pick:
        return "HIGH"
    if adp <= next_pick:
        return "MEDIUM"
    if adp <= next_pick + max(wait, 3.0):
        return "LOW"

    return "VERY LOW"


def build_adp_value_rankings_df():
    """
    Rank available QB/RB/WR/TE players by projection value versus ADP cost.
    """
    df = get_available_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    columns = _get_analysis_columns(df)
    player_col = columns.get("player_col")
    pos_col = columns.get("pos_col")
    team_col = columns.get("team_col")
    proj_col = columns.get("proj_col")
    adp_col = columns.get("adp_col")

    if player_col is None or pos_col is None or proj_col is None or adp_col is None:
        return pd.DataFrame()

    df = df[df[pos_col].astype(str).str.upper().isin(SCORABLE_POSITIONS)].copy()
    if df.empty:
        return pd.DataFrame()

    # A player genuinely needs real ADP to get an ADP-value score, so that
    # drop stays -- but missing projection_points gets a replacement-level
    # fallback instead of dropping the player (see fill_missing_projection_points).
    df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")
    df = df.dropna(subset=[adp_col]).copy()
    if df.empty:
        return pd.DataFrame()
    df = fill_missing_projection_points(df, pos_col, proj_col)

    current_pick = get_current_pick_number()
    next_pick_distance = get_next_pick_distance()

    df = df.sort_values(proj_col, ascending=False).reset_index(drop=True)
    df["projection_rank"] = df.index + 1
    df["adp_rank"] = df[adp_col]
    df["adp_delta"] = df.apply(
        lambda row: calculate_adp_delta(row["projection_rank"], row["adp_rank"]),
        axis=1,
    )
    df["value_score"] = df.apply(
        lambda row: calculate_value_score(
            row["adp_delta"],
            row[proj_col],
            current_pick,
            row["adp_rank"],
        ),
        axis=1,
    )
    df["value_tier"] = df.apply(
        lambda row: classify_player_value(row["adp_delta"], row["value_score"]),
        axis=1,
    )
    df["fall_risk"] = df["adp_rank"].apply(
        lambda adp: calculate_fall_risk(
            adp,
            current_pick=current_pick,
            next_pick_distance=next_pick_distance,
        )
    )

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "player_name": str(row[player_col]),
            "position": str(row[pos_col]).upper(),
            "team": str(row[team_col]) if team_col is not None else "",
            "projection_points": _safe_float(row[proj_col], 0.0),
            "projection_rank": int(row["projection_rank"]),
            "adp_rank": round(_safe_float(row["adp_rank"], 0.0), 2),
            "adp_delta": row["adp_delta"],
            "value_score": row["value_score"],
            "value_tier": row["value_tier"],
            "fall_risk": row["fall_risk"],
            "projection_source": row.get("projection_source", "real"),
        })

    rankings_df = pd.DataFrame(rows)
    if rankings_df.empty:
        return rankings_df

    return rankings_df.sort_values(
        ["value_score", "adp_delta", "projection_points"],
        ascending=False,
    ).reset_index(drop=True)


def get_best_value_pick():
    value_df = build_adp_value_rankings_df()
    if value_df.empty:
        return None

    return value_df.iloc[0].to_dict()


def get_adp_value_debug_info():
    available_df = get_available_players_df().copy()
    columns = _get_analysis_columns(available_df) if not available_df.empty else {}
    value_df = build_adp_value_rankings_df()
    warnings = []

    if available_df.empty:
        warnings.append("Available player DataFrame is empty.")
    if not columns.get("proj_col"):
        warnings.append("Projection column is missing.")
    if not columns.get("adp_col"):
        warnings.append("ADP column is missing.")

    return {
        "current_pick": get_current_pick_number(),
        "next_pick_distance": get_next_pick_distance(),
        "available_shape": available_df.shape,
        "detected_columns": columns,
        "warnings": warnings,
        "top_value_targets": value_df.head(10).to_dict("records")
        if not value_df.empty
        else [],
        "best_value_pick": get_best_value_pick(),
    }


def calculate_tier_bonus(player_name, position, tiers_df=None, tier_summary_df=None):
    if tiers_df is None:
        tiers_df = build_position_tiers_df()
    if tier_summary_df is None:
        tier_summary_df = build_tier_summary_df()

    if tiers_df.empty or tier_summary_df.empty:
        return 1.0

    player_match = tiers_df[
        tiers_df["Player"].astype(str).str.lower() == str(player_name).lower()
    ]
    summary_match = tier_summary_df[
        tier_summary_df["Position"].astype(str).str.upper() == str(position).upper()
    ]

    if player_match.empty or summary_match.empty:
        return 1.0

    player_tier = int(player_match.iloc[0]["Tier"])
    current_tier = int(summary_match.iloc[0]["Tier"])
    tier_status = str(summary_match.iloc[0]["Tier Status"])

    if player_tier != current_tier:
        return 1.0
    if tier_status == "Thin":
        return 1.10
    if tier_status == "Shrinking":
        return 1.06

    return 1.0


# NOTE: this file used to also define calculate_recommendation_score(), a
# second multiplicative scoring formula computed inside
# _build_base_recommendation_rankings_df() below. Its output was immediately
# and unconditionally overwritten by calculate_final_recommendation_score()
# in build_master_recommendations_df() before ever reaching a caller --
# _build_base_recommendation_rankings_df() has exactly one caller, and that
# caller always re-scores every row. It was removed rather than fixed
# because keeping two silently-conflicting scoring formulas in the same file
# is itself the bug: calculate_final_recommendation_score() (see
# DEFAULT_MASTER_COMPONENT_WEIGHTS) is the one formula that actually drives
# every ranking the app displays.


def build_position_replacement_baselines(df, pos_col, proj_col):
    """
    Estimate position-specific replacement baselines for cross-position scoring.
    """
    roster = st.session_state.get("roster_settings", {})
    league_size = get_league_size()
    flex_slots = int(roster.get("FLEX", 0))
    flex_shares = {"RB": 0.4, "WR": 0.4, "TE": 0.2}
    baselines = {}

    for position in SCORABLE_POSITIONS:
        pos_df = df[df[pos_col].astype(str).str.upper() == position].sort_values(
            proj_col,
            ascending=False,
        )

        if pos_df.empty:
            baselines[position] = 0.0
            continue

        starter_slots = int(roster.get(position, 0))
        flex_slot_share = flex_slots * flex_shares.get(position, 0.0)
        replacement_rank = max(int(round(league_size * (starter_slots + flex_slot_share))), 1)
        replacement_idx = min(replacement_rank - 1, len(pos_df) - 1)

        baselines[position] = float(pos_df.iloc[replacement_idx][proj_col])

    return baselines


_TIER_CURVE_PATH = Path("research/validation_v1/data/positional_tier_curve.csv")
_TIER_EDGES = [("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9),
               ("10-12", 10, 12), ("13-18", 13, 18), ("19-30", 19, 30)]
# Beyond the fitted range every position is already at the floor.
_TIER_BEYOND_FACTOR = 0.05


_TIER_FACTOR_MIN = 0.05
_TIER_FACTOR_MAX = 1.50
# Below this projected VOR the ratio is numerically unstable (dividing by
# ~0), and the player scores near nothing anyway, so leave him alone.
_MIN_PROJECTED_VOR = 5.0


@lru_cache(maxsize=1)
def load_realized_tier_vor():
    """
    Realized value-over-replacement by (position, positional-ADP tier),
    measured 2015-2025 by build_positional_tier_curve_v1.py.

    Player-weighted median, not a season mean: a single career could
    otherwise carry a cell (Travis Kelce was 24% of the TE1-3 sample at
    +128 VOR, inflating premium TE well past what it is worth -- and he is
    not in the 2026 premium pool at all).

    Returns {} if missing, which leaves scoring unchanged.
    """
    if not _TIER_CURVE_PATH.exists():
        return {}
    try:
        curve = pd.read_csv(_TIER_CURVE_PATH)
    except Exception:
        return {}
    if "tier_vor" not in curve.columns:
        return {}
    return {
        (str(r["position"]).upper(), str(r["tier"])): float(r["tier_vor"])
        for _, r in curve.iterrows()
    }


def tier_label_for_rank(positional_adp_rank):
    rank = _safe_float(positional_adp_rank, None)
    if rank is None or rank <= 0:
        return None
    for label, lo, hi in _TIER_EDGES:
        if lo <= rank <= hi:
            return label
    return None


def build_tier_calibration(df, pos_col, replacement_baselines, proj_col):
    """
    Calibrate projected value-over-replacement against what each
    (position, draft-tier) actually returned historically.

        factor = realized_VOR(pos, tier) / projected_VOR(pos, tier)

    This corrects a real, measured bias: projections are miscalibrated
    ACROSS positions, not just smoothed within them. On the current pool
    against 2015-2025 outcomes, tier 1-3 calibration comes out at
    QB 0.36, RB 0.48, WR 0.61, TE 1.10 -- i.e. projections overrate the
    premium QB tier by roughly 3x while pricing premium TE about right.
    Left uncorrected, that is why the board floated six QBs into a top 50
    that ADP gives one.

    This is NOT the double-count the old TE 0.68 multiplier was.
    Replacement baselines correct scarcity in *projection* space; this
    corrects the projections themselves against outcomes. Different error,
    measured separately.

    Tiers whose realized value is at or below replacement are floored
    rather than allowed to go negative -- a negative factor would invert a
    player's value and rank bad players above good ones.
    """
    realized = load_realized_tier_vor()
    if not realized:
        return {}

    work = df[[pos_col, proj_col, "positional_adp_rank"]].copy()
    work["position"] = work[pos_col].astype(str).str.upper()
    work["tier"] = work["positional_adp_rank"].apply(tier_label_for_rank)
    work = work[work["tier"].notna()]
    if work.empty:
        return {}

    work["projected_vor"] = [
        max(_safe_float(p, 0.0) - _safe_float(replacement_baselines.get(pos), 0.0), 0.0)
        for p, pos in zip(work[proj_col], work["position"])
    ]
    projected = work.groupby(["position", "tier"])["projected_vor"].mean()

    calibration = {}
    for key, projected_vor in projected.items():
        realized_vor = realized.get(key)
        if realized_vor is None:
            continue
        if realized_vor <= 0:
            calibration[key] = _TIER_FACTOR_MIN
        elif projected_vor < _MIN_PROJECTED_VOR:
            calibration[key] = 1.0
        else:
            calibration[key] = float(
                min(max(realized_vor / projected_vor, _TIER_FACTOR_MIN), _TIER_FACTOR_MAX)
            )
    return calibration


def positional_tier_factor(position, positional_adp_rank, calibration):
    """Calibration factor for one player, given the fitted tier table."""
    if not calibration:
        return 1.0
    tier = tier_label_for_rank(positional_adp_rank)
    # No ADP means the market never priced him, so he is not a premium
    # asset -- treat him as beyond the fitted range rather than defaulting
    # to tier 1-3 and handing him full value.
    if tier is None:
        return _TIER_FACTOR_MIN
    return calibration.get((str(position).upper(), tier), 1.0)


def calculate_position_value_score(projection_points, position, replacement_baselines):
    """
    Convert raw projection into position-adjusted value over replacement.
    """
    position = str(position).upper()
    projection = _safe_float(projection_points, 0.0)
    baseline = _safe_float(replacement_baselines.get(position), 0.0)
    value_over_replacement = projection - baseline
    positional_multiplier = _safe_float(POSITION_VALUE_MULTIPLIERS.get(position), 1.0)
    adjusted_value = (max(value_over_replacement, 0.0) + 1.0) * positional_multiplier

    return round(max(adjusted_value, 0.01), 2)


def _build_base_recommendation_rankings_df():
    """
    Build player-value plus team-fit recommendation rankings.
    """
    get_current_pick_number()
    df = get_available_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    columns = _get_analysis_columns(df)
    player_col = columns.get("player_col")
    pos_col = columns.get("pos_col")
    team_col = columns.get("team_col")
    proj_col = columns.get("proj_col")
    adp_col = columns.get("adp_col")

    if player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame()

    df = df[df[pos_col].astype(str).str.upper().isin(SCORABLE_POSITIONS)].copy()
    if df.empty:
        return pd.DataFrame()

    df = fill_missing_projection_points(df, pos_col, proj_col)

    if adp_col is not None:
        df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")

    df = df.sort_values(proj_col, ascending=False).reset_index(drop=True)
    df["projection_rank"] = df.index + 1

    context = _build_scoring_context(df, columns)
    replacement_baselines = build_position_replacement_baselines(df, pos_col, proj_col)

    # Positional ADP rank (TE1, TE2, ...) keys the value-cliff curve. Ranked
    # over the FULL pool including already-drafted players would be wrong,
    # but df here is the available pool, which is what we want: as premium
    # players come off the board the remaining ones move up their tier,
    # which is the real behaviour -- the best TE left genuinely is TE1 now.
    if adp_col is not None:
        df["positional_adp_rank"] = (
            pd.to_numeric(df[adp_col], errors="coerce")
            .groupby(df[pos_col].astype(str).str.upper())
            .rank(method="first")
        )
    else:
        df["positional_adp_rank"] = None

    tier_calibration = build_tier_calibration(df, pos_col, replacement_baselines, proj_col)
    need_weights = get_position_need_weights()
    team_profile = get_team_profile()
    tiers_df = build_position_tiers_df()
    tier_summary_df = build_tier_summary_df()

    rows = []
    for _, row in df.iterrows():
        player_name = str(row[player_col])
        position = str(row[pos_col]).upper()
        projection = _safe_float(row[proj_col], 0.0)
        position_value_score = calculate_position_value_score(
            projection,
            position,
            replacement_baselines,
        )
        # Calibrate against what this position/tier actually returned.
        # Projections rate TE5 nearly as highly as TE2, and rate QBs far
        # above what they deliver; outcomes disagree with both.
        tier_factor = positional_tier_factor(
            position, row.get("positional_adp_rank"), tier_calibration
        )
        position_value_score = round(max(position_value_score * tier_factor, 0.01), 2)
        adp = row[adp_col] if adp_col is not None else None
        need_bonus = float(need_weights.get(position, 1.0))
        adp_bonus = calculate_adp_bonus(row["projection_rank"], adp)
        tier_bonus = calculate_tier_bonus(
            player_name,
            position,
            tiers_df=tiers_df,
            tier_summary_df=tier_summary_df,
        )
        team_fit_bonus = calculate_team_fit_score(
            row,
            team_profile=team_profile,
            columns=columns,
            context=context,
        )
        archetype = derive_player_archetype(row, columns, context)

        rows.append({
            "player_name": player_name,
            "position": position,
            "team": str(row[team_col]) if team_col is not None else "",
            "projection_points": projection,
            "position_value_score": position_value_score,
            "position_value_multiplier": POSITION_VALUE_MULTIPLIERS.get(position, 1.0),
            "adp": adp,
            "archetype": archetype,
            "injury_risk": get_player_injury_risk(row, columns),
            "durability_grade": get_player_durability(row, columns),
            "need_bonus": round(need_bonus, 3),
            "adp_bonus": adp_bonus,
            "tier_bonus": tier_bonus,
            "team_fit_bonus": team_fit_bonus,
            "projection_source": row.get("projection_source", "real"),
        })

    rankings_df = pd.DataFrame(rows)
    if rankings_df.empty:
        return rankings_df

    # This is an intermediate table: build_master_recommendations_df() (the
    # only caller) computes the real final_score via
    # calculate_final_recommendation_score() and re-sorts fully afterward, so
    # this ordering only affects tie-breaking before that happens.
    return rankings_df.sort_values(
        ["position_value_score", "projection_points"],
        ascending=False,
    ).reset_index(drop=True)


def _normalize_series_to_100(series, neutral_score=50.0, lower_bound=None, upper_bound=None):
    values = pd.to_numeric(series, errors="coerce")

    if lower_bound is None:
        lower_bound = values.min(skipna=True)
    if upper_bound is None:
        upper_bound = values.max(skipna=True)

    if pd.isna(lower_bound) or pd.isna(upper_bound):
        return pd.Series([neutral_score] * len(series), index=series.index)

    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)

    if upper_bound <= lower_bound:
        return pd.Series([neutral_score] * len(series), index=series.index)

    normalized = ((values - lower_bound) / (upper_bound - lower_bound)) * 100.0
    return normalized.clip(lower=0.0, upper=100.0).fillna(neutral_score).round(2)


def _resolve_component_weights(component_weights=None):
    weights = DEFAULT_MASTER_COMPONENT_WEIGHTS.copy()
    if component_weights:
        for key, value in component_weights.items():
            if key in weights:
                weights[key] = max(_safe_float(value, weights[key]), 0.0)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return DEFAULT_MASTER_COMPONENT_WEIGHTS.copy()

    return weights


def normalize_component_scores(rankings_df):
    """
    Add normalized 0-100 component scores used by the master recommendation engine.
    """
    if rankings_df.empty:
        return rankings_df.copy()

    normalized_df = rankings_df.copy()

    component_defaults = {
        "projection": {"neutral": 50.0, "lower": None, "upper": None},
        "position_need": {"neutral": 50.0, "lower": 0.6, "upper": 1.5},
        "adp_value": {"neutral": 50.0, "lower": None, "upper": None},
        "tier_urgency": {"neutral": 0.0, "lower": 0.0, "upper": 100.0},
        "team_fit": {"neutral": 50.0, "lower": 0.80, "upper": 1.20},
    }

    for component, (source_col, output_col) in MASTER_COMPONENT_COLUMNS.items():
        defaults = component_defaults[component]
        if source_col not in normalized_df.columns:
            normalized_df[output_col] = defaults["neutral"]
            continue

        normalized_df[output_col] = _normalize_series_to_100(
            normalized_df[source_col],
            neutral_score=defaults["neutral"],
            lower_bound=defaults["lower"],
            upper_bound=defaults["upper"],
        )

    return normalized_df


# Components that answer "who should I draft given my roster?" rather than
# "who is the better player." They are legitimate -- needing a TE really
# should move a TE up your board -- but they must MODULATE player value,
# never manufacture it.
STRATEGY_COMPONENTS = ("position_need", "tier_urgency", "team_fit")

# A player at or below his position's replacement baseline scores exactly
# 1.0 from calculate_position_value_score (the `max(vor, 0) + 1.0` floor).
REPLACEMENT_LEVEL_VALUE = 1.0


def calculate_final_recommendation_score(row, component_weights=None):
    """
    Combine normalized component scores into one transparent recommendation score.

    Strategy components are damped for players with no value over
    replacement. Without this, the position-constant strategy terms (40% of
    the weight between position_need and tier_urgency) can lift a
    zero-value player into the top of the board purely because his position
    is unfilled -- measured directly: Evan Engram (ADP 212, projected 90.0
    against a TE replacement baseline of 131.2, i.e. value at the floor)
    landed at board rank 46, and Mike Gesicki (ADP 207, projected 91.2) at
    47. Needing a tight end is a real reason to move a startable tight end
    up; it is not a reason to draft a replacement-level one in the fourth
    round.
    """
    weights = _resolve_component_weights(component_weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    # At/below replacement -> strategy terms fall away entirely. Just above
    # it they scale back in, so there is no cliff at the boundary.
    raw_value = _safe_float(row.get("position_value_score"), None)
    if raw_value is None:
        strategy_scale = 1.0
    else:
        surplus = raw_value - REPLACEMENT_LEVEL_VALUE
        strategy_scale = min(max(surplus / 10.0, 0.0), 1.0)

    weighted_score = 0.0
    for component, weight in weights.items():
        _, normalized_col = MASTER_COMPONENT_COLUMNS[component]
        value = _safe_float(row.get(normalized_col), 50.0)
        if component in STRATEGY_COMPONENTS:
            # Damp toward the neutral 50 rather than toward 0, so damping
            # removes the strategy *tilt* without also imposing a penalty.
            value = 50.0 + (value - 50.0) * strategy_scale
        weighted_score += value * weight

    return round(weighted_score / total_weight, 2)


def apply_signal_trust_adjustments(recommendations_df):
    """
    Damp recommendation scores when upstream data quality is low.

    This is intentionally a post-score guardrail: the core recommendation model
    can still expose what it would have liked, while low-trust data cannot make a
    player the top recommendation.
    """
    if recommendations_df.empty:
        return recommendations_df.copy()

    adjusted_df = recommendations_df.copy()
    try:
        from draftkit.signal_trust import build_signal_trust_report

        trust_df = build_signal_trust_report()
    except Exception:
        adjusted_df["signal_trust_score"] = pd.NA
        adjusted_df["adp_trust_score"] = pd.NA
        adjusted_df["trust_adjustment_applied"] = False
        adjusted_df["trust_adjustment_reason"] = ""
        return adjusted_df

    if trust_df.empty or "player_name" not in trust_df.columns:
        adjusted_df["signal_trust_score"] = pd.NA
        adjusted_df["adp_trust_score"] = pd.NA
        adjusted_df["trust_adjustment_applied"] = False
        adjusted_df["trust_adjustment_reason"] = ""
        return adjusted_df

    trust_lookup = {
        str(row["player_name"]).strip().lower(): row.to_dict()
        for _, row in trust_df.iterrows()
    }

    signal_trust_scores = []
    adp_trust_scores = []
    anomaly_flags = []
    adjusted_scores = []
    adjustment_applied = []
    adjustment_reasons = []

    severe_flags = {
        "elite_projection_late_adp_conflict",
        "extreme_adp_projection_gap",
        "projection_rank_conflict",
        "adp_rank_conflict",
    }

    for _, row in adjusted_df.iterrows():
        player_key = str(row.get("player_name", "")).strip().lower()
        trust_row = trust_lookup.get(player_key, {})
        signal_trust = _safe_float(trust_row.get("signal_trust_score"), 100.0)
        adp_trust = _safe_float(trust_row.get("adp_trust_score"), 100.0)
        flags = trust_row.get("anomaly_flags", [])
        if not isinstance(flags, list):
            flags = []

        original_score = _safe_float(row.get("final_score"), 0.0)
        adjusted_score = original_score
        reasons = []

        if severe_flags.intersection(set(flags)):
            adjusted_score = min(adjusted_score, 58.0)
            reasons.append("severe data conflict")

        if signal_trust < 50:
            adjusted_score = min(adjusted_score, 60.0)
            adjusted_score *= 0.88
            reasons.append("low signal trust")
        elif signal_trust < 70:
            adjusted_score = min(adjusted_score, 68.0)
            adjusted_score *= 0.94
            reasons.append("medium-low signal trust")

        if adp_trust < 40:
            adjusted_score = min(adjusted_score, 62.0)
            adjusted_score *= 0.90
            reasons.append("low ADP trust")

        signal_trust_scores.append(round(signal_trust, 2))
        adp_trust_scores.append(round(adp_trust, 2))
        anomaly_flags.append(flags)
        adjusted_scores.append(round(adjusted_score, 2))
        adjustment_applied.append(round(adjusted_score, 2) < round(original_score, 2))
        adjustment_reasons.append(", ".join(reasons))

    adjusted_df["raw_final_score"] = adjusted_df["final_score"]
    adjusted_df["signal_trust_score"] = signal_trust_scores
    adjusted_df["adp_trust_score"] = adp_trust_scores
    adjusted_df["signal_trust_anomaly_flags"] = anomaly_flags
    adjusted_df["final_score"] = adjusted_scores
    adjusted_df["trust_adjustment_applied"] = adjustment_applied
    adjusted_df["trust_adjustment_reason"] = adjustment_reasons

    return adjusted_df


def apply_single_qb_value_adjustments(recommendations_df):
    """
    Guard against raw QB fantasy points overpowering RB/WR economics in 1-QB
    leagues, via a smooth multiplier rather than a hard score cap.

    A prior version clamped every non-elite QB's final_score to a flat
    constant (SINGLE_QB_SCORE_CAP = 50.0), which meant any two QBs without
    SEVERE/HIGH position pressure scored identically regardless of how far
    apart their ADP or projection actually was -- e.g. a QB going 26th
    overall and one going 139th both landed on exactly 50.00. That flattening
    was itself dragging down correlation to ADP for the whole QB position.
    Applying a multiplier instead discounts QBs as a group (the actual
    intent) while preserving their relative order.
    """
    if recommendations_df.empty:
        return recommendations_df.copy()

    roster = st.session_state.get("roster_settings", {})
    qb_slots = int(_safe_float(roster.get("QB"), 1))
    superflex_slots = int(_safe_float(roster.get("SUPERFLEX", roster.get("SF")), 0))

    adjusted_df = recommendations_df.copy()
    adjusted_df["single_qb_adjustment_applied"] = False
    adjusted_df["single_qb_adjustment_reason"] = ""

    if qb_slots > 1 or superflex_slots > 0:
        return adjusted_df

    qb_mask = adjusted_df["position"].astype(str).str.upper() == "QB"
    if not qb_mask.any():
        return adjusted_df

    qb_multipliers = adjusted_df.loc[qb_mask, "position_pressure"].map(
        lambda level: SINGLE_QB_SEVERE_PRESSURE_MULTIPLIER
        if str(level).upper() == "SEVERE"
        else SINGLE_QB_HIGH_PRESSURE_MULTIPLIER
        if str(level).upper() == "HIGH"
        else SINGLE_QB_BASE_MULTIPLIER
    )
    adjusted_df.loc[qb_mask, "final_score"] = (
        adjusted_df.loc[qb_mask, "final_score"] * qb_multipliers
    ).round(2)
    adjusted_df.loc[qb_mask, "single_qb_adjustment_applied"] = True
    adjusted_df.loc[qb_mask, "single_qb_adjustment_reason"] = (
        "single-QB positional value discount"
    )

    return adjusted_df


def apply_construction_pressure_adjustments(recommendations_df):
    """
    Apply draft-phase roster construction pressure as a score modifier.
    """
    if recommendations_df.empty:
        return recommendations_df.copy()

    adjusted_df = recommendations_df.copy()
    pressure = calculate_construction_pressure()
    position_pressure = pressure.get("position_pressure", {})

    construction_scores = []
    pressure_levels = []
    critical_needs = []
    roster_completeness_values = []
    adjustments = []
    adjusted_scores = []

    for _, row in adjusted_df.iterrows():
        position = str(row.get("position", "")).upper()
        pressure_info = position_pressure.get(position, {})
        modifier = calculate_construction_adjustment(position, pressure)
        original_score = _safe_float(row.get("final_score"), 0.0)

        construction_scores.append(_safe_float(pressure_info.get("pressure_score"), 0.0))
        pressure_levels.append(pressure_info.get("pressure_level", "NONE"))
        critical_needs.append(pressure.get("critical_needs", []))
        roster_completeness_values.append(pressure.get("roster_completeness", 0.0))
        adjustments.append(modifier)
        adjusted_scores.append(round(original_score * modifier, 2))

    adjusted_df["pre_construction_score"] = adjusted_df["final_score"]
    adjusted_df["construction_pressure_score"] = construction_scores
    adjusted_df["position_pressure"] = pressure_levels
    adjusted_df["critical_needs"] = critical_needs
    adjusted_df["roster_completeness"] = roster_completeness_values
    adjusted_df["construction_adjustment"] = adjustments
    adjusted_df["final_score"] = adjusted_scores

    return adjusted_df


def generate_recommendation_reasons(row, max_reasons=4):
    """
    Explain why a player ranks highly in the master recommendation engine.
    """
    reasons = []
    position = str(row.get("position", "")).upper()

    if row.get("construction_mandate_active") and row.get("construction_mandate"):
        if position == "RB":
            reasons.append(str(row.get("construction_mandate")))

    if row.get("trust_adjustment_applied"):
        reason = row.get("trust_adjustment_reason") or "low input trust"
        reasons.append(f"Recommendation dampened due to {reason}")

    if row.get("single_qb_adjustment_applied"):
        reason = row.get("single_qb_adjustment_reason") or "single-QB positional value"
        reasons.append(f"QB score capped due to {reason}")

    pressure_level = str(row.get("position_pressure", "NONE"))
    if pressure_level in ["HIGH", "SEVERE"]:
        reasons.append(f"{position} construction pressure is {pressure_level.lower()}")

    if _safe_float(row.get("projection_component_score"), 0.0) >= 75:
        reasons.append("Strong position-adjusted projection value")
    if _safe_float(row.get("position_need_component_score"), 0.0) >= 70:
        reasons.append(f"Fills current {position} roster need")

    value_tier = str(row.get("value_tier", "")).upper()
    adp_delta = _safe_float(row.get("adp_delta"), 0.0)
    if value_tier in ["STEAL", "VALUE"] or adp_delta >= 8:
        reasons.append(f"{value_tier.title() if value_tier else 'Positive'} ADP value")

    urgency_label = str(row.get("urgency_label", "")).title()
    if urgency_label in ["Critical", "High"]:
        reasons.append(f"{position} tier urgency is {urgency_label.lower()}")

    if _safe_float(row.get("team_fit_component_score"), 0.0) >= 65:
        reasons.append("Fits current team construction")

    fall_risk = str(row.get("fall_risk", "")).upper()
    if fall_risk == "HIGH":
        reasons.append("Unlikely to survive until your next pick")
    elif fall_risk == "MEDIUM":
        reasons.append("May not make it back to your next pick")

    archetype = str(row.get("archetype", "")).upper()
    if archetype in ["SAFE", "STEADY"]:
        reasons.append(f"{archetype.title()} profile stabilizes the roster")
    elif archetype in ["BOOM", "UPSIDE"]:
        reasons.append(f"{archetype.title()} profile adds ceiling")

    if not reasons:
        reasons.append("Best normalized blend of projection, need, value, urgency, and fit")

    return reasons[:max_reasons]


# How far the board is allowed to drift from consensus. Step 5c swept this
# directly (research/validation_v1/optimize_blend_weights_v1.py): mean AUC
# gain peaks at lambda ~0.10-0.25 and degrades monotonically after, hitting
# -0.075 at QB / -0.072 at RB by lambda=1.0. 0.20 sits inside that band and
# is the midpoint of the per-position optima (QB 0.10, RB 0.20, WR 0.25).
#
# The board was previously running at effectively lambda=1.0 -- deep in the
# region measured as harmful -- which is what pushed Trey McBride to #4
# against an ADP of 29 and stripped WRs out of the top 50.
#
# Honest framing: Step 5c also found the optimum is NOT statistically
# distinguishable from pure ADP (bootstrap CIs include zero) and that
# walk-forward lambda selection was negative. 0.20 is the least-bad
# deviation, not an edge. The claim is "tracks consensus with small
# evidence-backed tilts," never "beats consensus."
ADP_ANCHOR_LAMBDA = 0.20


def apply_adp_anchor(recommendations_df, lam=ADP_ANCHOR_LAMBDA):
    """
    Blend the model score toward ADP in percentile space.

        anchored = (1 - lam) * adp_percentile + lam * model_percentile

    Percentiles rather than raw values so the two scales are comparable and
    a handful of extreme scores can't dominate the blend.

    Players with no ADP were never priced by the market, so there is no
    consensus to anchor to -- they keep their model percentile scaled into
    the range below the last ADP-having player rather than being dropped or
    treated as ADP-worst.
    """
    if recommendations_df.empty or "final_score" not in recommendations_df.columns:
        return recommendations_df
    if "adp" not in recommendations_df.columns:
        return recommendations_df

    out = recommendations_df.copy()
    adp = pd.to_numeric(out["adp"], errors="coerce")
    model = pd.to_numeric(out["final_score"], errors="coerce")
    has_adp = adp.notna() & model.notna()
    if has_adp.sum() < 10:
        return recommendations_df

    out["pre_anchor_score"] = out["final_score"]

    # Lower ADP is better, so negate before ranking to percentile.
    adp_pct = (-adp[has_adp]).rank(pct=True)
    model_pct = model[has_adp].rank(pct=True)
    blended = (1.0 - lam) * adp_pct + lam * model_pct

    anchored = pd.Series(np.nan, index=out.index, dtype=float)
    # Rescale to the original score range so downstream display/thresholds
    # keep working on a familiar 0-100-ish scale.
    lo, hi = float(model[has_adp].min()), float(model[has_adp].max())
    span = hi - lo
    if span <= 0:
        return recommendations_df
    anchored.loc[has_adp] = lo + blended.rank(pct=True) * span

    # Unpriced players sit below every priced one, ordered among themselves
    # by model score -- they are speculative depth, not consensus values.
    no_adp = ~has_adp & model.notna()
    if no_adp.any():
        anchored.loc[no_adp] = lo - 1.0 + model[no_adp].rank(pct=True)

    out["final_score"] = anchored.round(2).fillna(out["final_score"])
    return out


def build_master_recommendations_df(component_weights=None, drop_threshold=12.0):
    """
    Build transparent master recommendations from all existing scoring systems.
    """
    get_current_pick_number()
    base_df = _build_base_recommendation_rankings_df()
    if base_df.empty:
        return pd.DataFrame()

    master_df = base_df.copy()

    adp_value_df = build_adp_value_rankings_df()
    if not adp_value_df.empty:
        adp_cols = [
            "player_name",
            "position",
            "projection_rank",
            "adp_rank",
            "adp_delta",
            "value_score",
            "value_tier",
            "fall_risk",
        ]
        master_df = master_df.merge(
            adp_value_df[[col for col in adp_cols if col in adp_value_df.columns]],
            on=["player_name", "position"],
            how="left",
            suffixes=("", "_adp_value"),
        )

    if "projection_rank" not in master_df.columns:
        master_df = master_df.sort_values(
            "position_value_score",
            ascending=False,
        ).reset_index(drop=True)
        master_df["projection_rank"] = master_df.index + 1

    if "adp_rank" not in master_df.columns:
        master_df["adp_rank"] = pd.to_numeric(master_df.get("adp"), errors="coerce")

    if "adp_delta" not in master_df.columns:
        master_df["adp_delta"] = master_df.apply(
            lambda row: calculate_adp_delta(row.get("projection_rank"), row.get("adp_rank")),
            axis=1,
        )

    if "value_score" not in master_df.columns:
        current_pick = get_current_pick_number()
        master_df["value_score"] = master_df.apply(
            lambda row: calculate_value_score(
                row.get("adp_delta"),
                row.get("projection_points"),
                current_pick,
                row.get("adp_rank"),
            ),
            axis=1,
        )

    if "value_tier" not in master_df.columns:
        master_df["value_tier"] = master_df.apply(
            lambda row: classify_player_value(row.get("adp_delta"), row.get("value_score")),
            axis=1,
        )

    if "fall_risk" not in master_df.columns:
        master_df["fall_risk"] = master_df["adp_rank"].apply(calculate_fall_risk)

    urgency_df = build_position_urgency_df(drop_threshold=drop_threshold)
    if not urgency_df.empty:
        urgency_lookup = {
            str(row["Position"]).upper(): row.to_dict()
            for _, row in urgency_df.iterrows()
        }
    else:
        urgency_lookup = {}

    master_df["raw_urgency_score"] = master_df["position"].map(
        lambda pos: _safe_float(
            urgency_lookup.get(str(pos).upper(), {}).get("Urgency Score"),
            0.0,
        )
    )
    master_df["position_urgency_multiplier"] = master_df["position"].map(
        lambda pos: _safe_float(
            POSITION_URGENCY_MULTIPLIERS.get(str(pos).upper()),
            1.0,
        )
    )
    master_df["urgency_score"] = (
        master_df["raw_urgency_score"] * master_df["position_urgency_multiplier"]
    ).round(2)
    master_df["urgency_bonus"] = master_df["position"].map(
        lambda pos: _safe_float(
            urgency_lookup.get(str(pos).upper(), {}).get("Urgency Bonus"),
            1.0,
        )
    )
    master_df["urgency_label"] = master_df["position"].map(
        lambda pos: urgency_lookup.get(str(pos).upper(), {}).get("Urgency Label", "Low")
    )
    master_df["tier_dropoff"] = master_df["position"].map(
        lambda pos: _safe_float(
            urgency_lookup.get(str(pos).upper(), {}).get("Tier Dropoff"),
            0.0,
        )
    )
    master_df["players_left_in_tier"] = master_df["position"].map(
        lambda pos: int(
            _safe_float(
                urgency_lookup.get(str(pos).upper(), {}).get("Players Left In Tier"),
                0.0,
            )
        )
    )

    master_df = normalize_component_scores(master_df)
    weights = _resolve_component_weights(component_weights)
    master_df["final_score"] = master_df.apply(
        lambda row: calculate_final_recommendation_score(row, weights),
        axis=1,
    )
    master_df = apply_construction_pressure_adjustments(master_df)
    master_df = apply_signal_trust_adjustments(master_df)
    master_df = apply_single_qb_value_adjustments(master_df)
    master_df = apply_adp_anchor(master_df)
    master_df["recommendation_reasons"] = master_df.apply(
        lambda row: generate_recommendation_reasons(row),
        axis=1,
    )

    sort_cols = [
        "final_score",
        "projection_component_score",
        "position_need_component_score",
        "adp_value_component_score",
        "tier_urgency_component_score",
    ]

    return master_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def build_recommendation_rankings_df():
    """
    Backward-compatible public recommendation rankings powered by the master engine.
    """
    return build_master_recommendations_df()


def get_best_pick_recommendation(component_weights=None, drop_threshold=12.0):
    rankings_df = build_master_recommendations_df(
        component_weights=component_weights,
        drop_threshold=drop_threshold,
    )
    if rankings_df.empty:
        return None

    best = rankings_df.iloc[0].to_dict()
    reasons = best.get("recommendation_reasons")
    if not isinstance(reasons, list):
        reasons = generate_recommendation_reasons(best)

    return {
        "player": best["player_name"],
        "position": best["position"],
        "team": best.get("team", ""),
        "projection_points": best["projection_points"],
        "final_score": best["final_score"],
        "reasons": reasons,
        "score_breakdown": {
            "projection_score": best.get("position_value_score"),
            "raw_projection": best.get("projection_points"),
            "need_bonus": best.get("need_bonus"),
            "adp_bonus": best.get("adp_bonus"),
            "tier_bonus": best.get("tier_bonus"),
            "raw_urgency_score": best.get("raw_urgency_score"),
            "urgency_bonus": best.get("urgency_bonus"),
            "urgency_score": best.get("urgency_score"),
            "team_fit_bonus": best.get("team_fit_bonus"),
            "position_value_multiplier": best.get("position_value_multiplier"),
            "position_urgency_multiplier": best.get("position_urgency_multiplier"),
            "normalized_components": {
                "projection": best.get("projection_component_score"),
                "position_need": best.get("position_need_component_score"),
                "adp_value": best.get("adp_value_component_score"),
                "tier_urgency": best.get("tier_urgency_component_score"),
                "team_fit": best.get("team_fit_component_score"),
            },
            "component_weights": _resolve_component_weights(component_weights),
        },
        "player_profile": {
            "archetype": best.get("archetype"),
            "injury_risk": best.get("injury_risk"),
            "durability_grade": best.get("durability_grade"),
            "value_tier": best.get("value_tier"),
            "fall_risk": best.get("fall_risk"),
            "urgency_label": best.get("urgency_label"),
        },
    }


def get_top_recommendations(limit=3, component_weights=None, drop_threshold=12.0):
    rankings_df = build_master_recommendations_df(
        component_weights=component_weights,
        drop_threshold=drop_threshold,
    )
    if rankings_df.empty:
        return []

    return rankings_df.head(limit).to_dict("records")


def get_master_recommendation_debug_info(component_weights=None, drop_threshold=12.0):
    raw_df = load_players_df().copy()
    available_df = get_available_players_df().copy()
    columns = _get_analysis_columns(available_df) if not available_df.empty else {}
    rankings_df = build_master_recommendations_df(
        component_weights=component_weights,
        drop_threshold=drop_threshold,
    )
    weights = _resolve_component_weights(component_weights)

    missing_optional_fields = []
    if not columns.get("archetype_col"):
        missing_optional_fields.append("archetype")
    if not columns.get("injury_risk_col"):
        missing_optional_fields.append("injury_risk")
    if not columns.get("durability_col"):
        missing_optional_fields.append("durability_grade")

    component_ranges = {}
    if not rankings_df.empty:
        for component, (source_col, normalized_col) in MASTER_COMPONENT_COLUMNS.items():
            component_ranges[component] = {
                "source_column": source_col,
                "normalized_column": normalized_col,
                "raw_min": _safe_float(rankings_df[source_col].min(), 0.0)
                if source_col in rankings_df.columns
                else None,
                "raw_max": _safe_float(rankings_df[source_col].max(), 0.0)
                if source_col in rankings_df.columns
                else None,
                "normalized_min": _safe_float(rankings_df[normalized_col].min(), 0.0)
                if normalized_col in rankings_df.columns
                else None,
                "normalized_max": _safe_float(rankings_df[normalized_col].max(), 0.0)
                if normalized_col in rankings_df.columns
                else None,
            }

    score_columns = [
        "player_name",
        "position",
        "projection_points",
        "position_value_score",
        "position_value_multiplier",
        "need_bonus",
        "value_score",
        "value_tier",
        "fall_risk",
        "raw_urgency_score",
        "urgency_score",
        "position_urgency_multiplier",
        "urgency_label",
        "team_fit_bonus",
        "projection_component_score",
        "position_need_component_score",
        "adp_value_component_score",
        "tier_urgency_component_score",
        "team_fit_component_score",
        "final_score",
        "recommendation_reasons",
    ]

    return {
        "current_roster": st.session_state.get("my_team", []),
        "position_needs": get_my_team_positions(raw_df),
        "team_profile": get_team_profile(),
        "need_weights": get_position_need_weights(),
        "component_weights": weights,
        "component_ranges": component_ranges,
        "top_recommendations": rankings_df.head(10).to_dict("records")
        if not rankings_df.empty
        else [],
        "score_breakdowns": rankings_df.head(10)[
            [col for col in score_columns if col in rankings_df.columns]
        ].to_dict("records")
        if not rankings_df.empty
        else [],
        "best_pick": get_best_pick_recommendation(
            component_weights=component_weights,
            drop_threshold=drop_threshold,
        ),
        "adp_value_debug": get_adp_value_debug_info(),
        "urgency_debug": get_urgency_debug_info(drop_threshold=drop_threshold),
        "detected_columns": columns,
        "missing_optional_fields": missing_optional_fields,
        "raw_shape": raw_df.shape,
        "available_shape": available_df.shape,
    }


def get_recommendation_debug_info():
    """
    Backward-compatible debug wrapper for the master recommendation engine.
    """
    return get_master_recommendation_debug_info()


def build_position_scarcity_df(top_n=10):
    df, player_col, pos_col, _, proj_col, _ = _prep_ranked_df()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df[pos_col] == pos].head(top_n).copy()
        if len(pos_df) < 2:
            continue

        top_proj = float(pos_df.iloc[0][proj_col])
        nth_proj = float(pos_df.iloc[-1][proj_col])
        drop = round(top_proj - nth_proj, 2)

        rows.append({
            "Position": pos,
            "Top Players Window": len(pos_df),
            "Top Projection": top_proj,
            "Nth Projection": nth_proj,
            "Scarcity Drop": drop
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Scarcity Drop", ascending=False).reset_index(drop=True)


def build_position_cliff_df(window_size=3):
    df, _, pos_col, _, proj_col, _ = _prep_ranked_df()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df[pos_col] == pos].reset_index(drop=True)
        if pos_df.empty:
            continue

        current_proj = float(pos_df.iloc[0][proj_col])
        future_idx = min(window_size, len(pos_df) - 1)
        future_proj = float(pos_df.iloc[future_idx][proj_col])
        drop_if_wait = round(current_proj - future_proj, 2)

        rows.append({
            "Position": pos,
            "Current Best Projection": current_proj,
            "Projection After Wait": future_proj,
            "Drop If Wait": drop_if_wait
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Urgency Score"] = out["Drop If Wait"].rank(ascending=False, method="dense")
    out = out.sort_values("Drop If Wait", ascending=False).reset_index(drop=True)
    return out


def build_turn_aware_cliff_df():
    wait = get_next_pick_distance()
    if wait is None:
        wait = 3
    return build_position_cliff_df(window_size=wait)


def get_falloff_recommendation(window_size=3):
    cliff_df = build_position_cliff_df(window_size=window_size)
    if cliff_df.empty:
        return {"headline": "No recommendation", "message": "Not enough data."}

    row = cliff_df.iloc[0]
    return {
        "headline": f"{row['Position']} has the steepest fall-off",
        "message": f"If you wait {window_size} picks, expected drop is {row['Drop If Wait']} projection points."
    }


def get_turn_aware_falloff_recommendation():
    wait = get_next_pick_distance()
    cliff_df = build_turn_aware_cliff_df()
    if cliff_df.empty:
        return {"headline": "No recommendation", "message": "Not enough data."}

    row = cliff_df.iloc[0]
    return {
        "headline": f"{row['Position']} is most at risk by your next turn",
        "message": f"Waiting about {wait} picks may cost {row['Drop If Wait']} projection points."
    }


def build_position_tiers_df(drop_threshold=12.0):
    df, player_col, pos_col, team_col, proj_col, adp_col = _prep_ranked_df()
    if df.empty:
        return pd.DataFrame()

    rows = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = df[df[pos_col] == pos].copy().reset_index(drop=True)
        if pos_df.empty:
            continue

        current_tier = 1
        last_proj = None

        for _, row in pos_df.iterrows():
            projection = float(row[proj_col])

            if last_proj is not None and (last_proj - projection) >= drop_threshold:
                current_tier += 1

            rows.append({
                "Position": pos,
                "Tier": current_tier,
                "Player": str(row[player_col]),
                "Team": str(row[team_col]) if team_col else "",
                "Projection": round(projection, 2),
                "ADP": row[adp_col] if adp_col else None
            })

            last_proj = projection

    return pd.DataFrame(rows)


def build_tier_summary_df(drop_threshold=12.0):
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    if tiers_df.empty:
        return pd.DataFrame()

    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = tiers_df[tiers_df["Position"] == pos].copy()
        if pos_df.empty:
            continue

        first_tier = int(pos_df["Tier"].min())
        current_tier_df = pos_df[pos_df["Tier"] == first_tier].copy()
        players_left = len(current_tier_df)

        if players_left <= 2:
            tier_status = "Thin"
        elif players_left <= 4:
            tier_status = "Shrinking"
        else:
            tier_status = "Healthy"

        rows.append({
            "Position": pos,
            "Tier": first_tier,
            "Players Left In Tier": players_left,
            "Tier Status": tier_status
        })

    return pd.DataFrame(rows).sort_values(
        ["Tier Status", "Players Left In Tier"],
        ascending=[True, True]
    ).reset_index(drop=True)


def calculate_tier_dropoff(position, tier, tiers_df=None, drop_threshold=12.0):
    """
    Calculate projection drop from a position's current tier to the next tier.
    """
    if tiers_df is None:
        tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)

    if tiers_df.empty:
        return 0.0

    position = str(position).upper()
    tier = int(tier)

    current_tier_df = tiers_df[
        (tiers_df["Position"].astype(str).str.upper() == position)
        & (tiers_df["Tier"].astype(int) == tier)
    ].copy()
    next_tier_df = tiers_df[
        (tiers_df["Position"].astype(str).str.upper() == position)
        & (tiers_df["Tier"].astype(int) == tier + 1)
    ].copy()

    if current_tier_df.empty or next_tier_df.empty:
        return 0.0

    current_floor = float(current_tier_df["Projection"].min())
    next_tier_top = float(next_tier_df["Projection"].max())

    return round(max(current_floor - next_tier_top, 0.0), 2)


def calculate_position_urgency(
    position,
    players_left,
    tier_dropoff,
    next_pick_distance,
    need_weight,
):
    """
    Score how urgently a position should be addressed before the user's next pick.
    """
    try:
        players_left = int(players_left)
    except (TypeError, ValueError):
        players_left = 0

    wait = next_pick_distance if next_pick_distance is not None else 3
    try:
        wait = max(int(wait), 0)
    except (TypeError, ValueError):
        wait = 3

    dropoff = max(_safe_float(tier_dropoff, 0.0), 0.0)
    need = max(_safe_float(need_weight, 1.0), 0.0)

    if players_left <= 0:
        availability_score = 35.0
    elif players_left == 1:
        availability_score = 35.0
    elif players_left == 2:
        availability_score = 27.5
    elif players_left <= 4:
        availability_score = 17.5
    else:
        availability_score = 7.5

    if players_left <= 0:
        turn_pressure_score = 20.0
    else:
        turn_pressure_ratio = max(wait - players_left + 1, 0) / max(wait, 1)
        turn_pressure_score = min(turn_pressure_ratio * 25.0, 25.0)

    dropoff_score = min(dropoff * 1.5, 25.0)
    need_score = min(max(need - 0.6, 0.0) * 20.0, 20.0)

    urgency_score = round(
        availability_score + turn_pressure_score + dropoff_score + need_score,
        2,
    )

    if urgency_score >= 75:
        urgency_label = "Critical"
        urgency_bonus = 1.15
    elif urgency_score >= 55:
        urgency_label = "High"
        urgency_bonus = 1.10
    elif urgency_score >= 35:
        urgency_label = "Moderate"
        urgency_bonus = 1.05
    else:
        urgency_label = "Low"
        urgency_bonus = 1.00

    return {
        "Position": str(position).upper(),
        "Players Left In Tier": players_left,
        "Tier Dropoff": round(dropoff, 2),
        "Picks Until Next Pick": wait,
        "Need Weight": round(need, 2),
        "Urgency Score": urgency_score,
        "Urgency Bonus": urgency_bonus,
        "Urgency Label": urgency_label,
    }


def build_position_urgency_df(drop_threshold=12.0):
    """
    Build position-level urgency rankings for QB/RB/WR/TE tier cliffs.
    """
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    tier_summary_df = build_tier_summary_df(drop_threshold=drop_threshold)

    if tiers_df.empty or tier_summary_df.empty:
        return pd.DataFrame()

    next_pick_distance = get_next_pick_distance()
    if next_pick_distance is None:
        next_pick_distance = 3

    need_weights = get_position_need_weights()
    rows = []

    for _, row in tier_summary_df.iterrows():
        position = str(row["Position"]).upper()
        if position not in SCORABLE_POSITIONS:
            continue

        tier = int(row["Tier"])
        players_left = int(row["Players Left In Tier"])
        tier_dropoff = calculate_tier_dropoff(
            position,
            tier,
            tiers_df=tiers_df,
            drop_threshold=drop_threshold,
        )
        urgency = calculate_position_urgency(
            position=position,
            players_left=players_left,
            tier_dropoff=tier_dropoff,
            next_pick_distance=next_pick_distance,
            need_weight=need_weights.get(position, 1.0),
        )
        urgency["Tier"] = tier
        urgency["Tier Status"] = row["Tier Status"]
        rows.append(urgency)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Urgency Score", "Tier Dropoff", "Need Weight"],
        ascending=False,
    ).reset_index(drop=True)


def get_most_urgent_position(drop_threshold=12.0):
    urgency_df = build_position_urgency_df(drop_threshold=drop_threshold)
    if urgency_df.empty:
        return None

    return urgency_df.iloc[0].to_dict()


def get_tier_cliff_recommendation(drop_threshold=12.0):
    urgent_position = get_most_urgent_position(drop_threshold=drop_threshold)
    if urgent_position is None:
        return {
            "headline": "No tier cliff recommendation",
            "message": "Not enough tier data is available.",
            "position": None,
            "urgency_score": 0.0,
            "urgency_bonus": 1.0,
            "reasons": [],
        }

    position = urgent_position["Position"]
    label = urgent_position["Urgency Label"]
    players_left = int(urgent_position["Players Left In Tier"])
    dropoff = urgent_position["Tier Dropoff"]
    wait = int(urgent_position["Picks Until Next Pick"])
    need = urgent_position["Need Weight"]

    reasons = [
        f"{players_left} player(s) remain in the current {position} tier",
        f"Projected drop to next tier is {dropoff}",
        f"Next pick is about {wait} pick(s) away",
    ]

    if need > 1.0:
        reasons.append(f"{position} is a roster need")
    elif need < 1.0:
        reasons.append(f"{position} is less urgent based on roster construction")

    return {
        "headline": f"{position} tier cliff is {label.lower()} urgency",
        "message": (
            f"{position} has {players_left} current-tier player(s) left "
            f"with a {dropoff} point dropoff."
        ),
        "position": position,
        "urgency_score": urgent_position["Urgency Score"],
        "urgency_bonus": urgent_position["Urgency Bonus"],
        "reasons": reasons,
        "details": urgent_position,
    }


def get_urgency_debug_info(drop_threshold=12.0):
    raw_df = load_players_df().copy()
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    tier_summary_df = build_tier_summary_df(drop_threshold=drop_threshold)
    urgency_df = build_position_urgency_df(drop_threshold=drop_threshold)

    return {
        "next_pick_distance": get_next_pick_distance(),
        "roster_position_counts": get_my_team_positions(raw_df),
        "need_weights": get_position_need_weights(),
        "tier_summary": tier_summary_df.to_dict("records")
        if not tier_summary_df.empty
        else [],
        "tiers_shape": tiers_df.shape,
        "urgency_rows": urgency_df.to_dict("records")
        if not urgency_df.empty
        else [],
        "most_urgent_position": get_most_urgent_position(
            drop_threshold=drop_threshold
        ),
        "tier_cliff_recommendation": get_tier_cliff_recommendation(
            drop_threshold=drop_threshold
        ),
    }


def get_tier_warning(drop_threshold=12.0):
    summary_df = build_tier_summary_df(drop_threshold=drop_threshold)
    if summary_df.empty:
        return {"headline": "No tier warning", "message": "Not enough data."}

    thin_df = summary_df[summary_df["Tier Status"] == "Thin"]
    if not thin_df.empty:
        row = thin_df.iloc[0]
        return {
            "headline": f"{row['Position']} tier is thin",
            "message": f"Only {row['Players Left In Tier']} player(s) remain in the current tier."
        }

    shrinking_df = summary_df[summary_df["Tier Status"] == "Shrinking"]
    if not shrinking_df.empty:
        row = shrinking_df.iloc[0]
        return {
            "headline": f"{row['Position']} tier is shrinking",
            "message": f"{row['Players Left In Tier']} players remain before the next tier drop."
        }

    row = summary_df.iloc[0]
    return {
        "headline": f"{row['Position']} tier is healthy",
        "message": f"{row['Players Left In Tier']} players remain in the top current tier."
    }


def build_desperation_targets_df(drop_threshold=12.0):
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    tier_summary_df = build_tier_summary_df(drop_threshold=drop_threshold)
    cliff_df = build_turn_aware_cliff_df()

    if tiers_df.empty or tier_summary_df.empty:
        return pd.DataFrame()

    roster_needs = get_position_need_weights()

    cliff_lookup = {}
    if not cliff_df.empty:
        for _, row in cliff_df.iterrows():
            drop_if_wait = float(row["Drop If Wait"])
            if drop_if_wait >= 20:
                cliff_level = "Extreme"
                draft_signal = "Draft now"
            elif drop_if_wait >= 12:
                cliff_level = "High"
                draft_signal = "Strong priority"
            elif drop_if_wait >= 6:
                cliff_level = "Moderate"
                draft_signal = "Consider soon"
            else:
                cliff_level = "Low"
                draft_signal = "Monitor"

            cliff_lookup[str(row["Position"])] = {
                "Drop If Wait": drop_if_wait,
                "Urgency Score": float(row["Urgency Score"]),
                "Cliff Level": cliff_level,
                "Draft Signal": draft_signal,
            }

    rows = []

    priority_summary = tier_summary_df[
        tier_summary_df["Tier Status"].isin(["Thin", "Shrinking"])
    ].copy()

    for _, summary_row in priority_summary.iterrows():
        position = str(summary_row["Position"])
        tier = int(summary_row["Tier"])
        tier_status = str(summary_row["Tier Status"])
        players_left = int(summary_row["Players Left In Tier"])

        tier_players = tiers_df[
            (tiers_df["Position"] == position) &
            (tiers_df["Tier"] == tier)
        ].copy().reset_index(drop=True)

        if tier_players.empty:
            continue

        target_row = tier_players.iloc[-1]
        player_name = str(target_row["Player"])
        team = str(target_row["Team"])
        projection = float(target_row["Projection"])
        adp = target_row["ADP"]

        next_tier_players = tiers_df[
            (tiers_df["Position"] == position) &
            (tiers_df["Tier"] == tier + 1)
        ].copy().reset_index(drop=True)

        next_tier_player = None
        next_tier_projection = None
        tier_drop = 0.0

        if not next_tier_players.empty:
            next_tier_player = str(next_tier_players.iloc[0]["Player"])
            next_tier_projection = float(next_tier_players.iloc[0]["Projection"])
            tier_drop = round(projection - next_tier_projection, 2)

        need_weight = float(roster_needs.get(position, 0.0))

        thinness_score = 1.0 if tier_status == "Thin" else 0.7
        if players_left <= 1:
            thinness_score += 0.25
        elif players_left == 2:
            thinness_score += 0.10

        cliff_info = cliff_lookup.get(position, {})
        drop_if_wait = float(cliff_info.get("Drop If Wait", 0.0))
        urgency_score = float(cliff_info.get("Urgency Score", 0.0))
        cliff_level = str(cliff_info.get("Cliff Level", "Unknown"))
        draft_signal = str(cliff_info.get("Draft Signal", "Monitor"))

        desperation_score = round(
            (thinness_score * 35)
            + (need_weight * 25)
            + min(tier_drop, 25.0)
            + min(drop_if_wait, 25.0),
            2
        )

        rows.append({
            "Position": position,
            "Tier": tier,
            "Tier Status": tier_status,
            "Players Left In Tier": players_left,
            "Player": player_name,
            "Team": team,
            "Projection": projection,
            "ADP": adp,
            "Next Tier Player": next_tier_player,
            "Next Tier Projection": next_tier_projection,
            "Tier Drop": tier_drop,
            "Need Weight": round(need_weight, 2),
            "Drop If Wait": round(drop_if_wait, 2),
            "Urgency Score": round(urgency_score, 2),
            "Cliff Level": cliff_level,
            "Draft Signal": draft_signal,
            "Desperation Score": desperation_score
        })

    desperation_df = pd.DataFrame(rows)
    if desperation_df.empty:
        return desperation_df

    return desperation_df.sort_values(
        ["Desperation Score", "Tier Drop", "Drop If Wait"],
        ascending=False
    ).reset_index(drop=True)


def get_best_desperation_target(drop_threshold=12.0):
    desperation_df = build_desperation_targets_df(drop_threshold=drop_threshold)
    if desperation_df.empty:
        return None
    return desperation_df.iloc[0].to_dict()
