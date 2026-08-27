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
    # Same override as _build_base_recommendation_rankings_df() -- without
    # this, adp_delta/value_score (the "adp_value" component) would score
    # off the stale, un-researched number while position_value_score (the
    # "projection" component) used this project's corrected one, silently
    # disagreeing on the same player. Legacy PROJECTION_MANUAL_ADJUSTMENTS
    # was never applied on this path either (checked directly) -- a
    # pre-existing gap, not introduced here.
    df = apply_model_projection_override(df, player_col, proj_col)

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


# Perf fix (2026-08-27, load-time investigation): calculate_tier_bonus()'s
# only caller runs it once per player in a 3900+-row loop, passing the SAME
# tiers_df/tier_summary_df object every time -- but the function used to
# re-filter both DataFrames from scratch on every call (a fresh
# .astype(str).str.lower() pass over tiers_df included), profiled at 14.4s
# of a 40s cold board build (38% of total time) for what should be an O(1)
# lookup. This single-slot, id()-keyed cache holds the dict-ified lookup
# for whichever (tiers_df, tier_summary_df) pair was built most recently --
# safe because the hot loop only ever has one such pair live at a time
# within a single build_recommendation_rankings_df() call, and a fresh
# call always constructs fresh DataFrame objects (fresh ids) anyway.
_TIER_BONUS_LOOKUP_CACHE: dict = {}


def _tier_bonus_lookups(tiers_df, tier_summary_df):
    key = (id(tiers_df), id(tier_summary_df))
    cached = _TIER_BONUS_LOOKUP_CACHE.get(key)
    if cached is not None:
        return cached

    tp = tiers_df.assign(_k=tiers_df["Player"].astype(str).str.lower())
    tp = tp.drop_duplicates(subset="_k", keep="first")  # first match wins, matching the old .iloc[0]
    player_tier = dict(zip(tp["_k"], tp["Tier"].astype(int)))

    ts = tier_summary_df.assign(_k=tier_summary_df["Position"].astype(str).str.upper())
    ts = ts.drop_duplicates(subset="_k", keep="first")
    summary_lookup = {
        r["_k"]: (int(r["Tier"]), str(r["Tier Status"])) for _, r in ts.iterrows()
    }

    _TIER_BONUS_LOOKUP_CACHE.clear()  # only ever need the most recent pair
    _TIER_BONUS_LOOKUP_CACHE[key] = (player_tier, summary_lookup)
    return player_tier, summary_lookup


def calculate_tier_bonus(player_name, position, tiers_df=None, tier_summary_df=None):
    if tiers_df is None:
        tiers_df = build_position_tiers_df()
    if tier_summary_df is None:
        tier_summary_df = build_tier_summary_df()

    if tiers_df.empty or tier_summary_df.empty:
        return 1.0

    player_tier_lookup, summary_lookup = _tier_bonus_lookups(tiers_df, tier_summary_df)

    player_tier = player_tier_lookup.get(str(player_name).lower())
    summary = summary_lookup.get(str(position).upper())
    if player_tier is None or summary is None:
        return 1.0

    current_tier, tier_status = summary

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


# Strategy-scale calibration (2026-08-21). See compute_strategy_scale()
# and calculate_final_recommendation_score() for the full rationale.
#
# STRATEGY_SCALE_MIN is a real floor, not a formality: it is what stops a
# below-replacement player's strategy terms from being switched off
# entirely, which is what the previous binary rule did. Bounded strictly
# above zero so every component keeps differentiating players WITHIN a
# position no matter how far below replacement the position's calibration
# curve pushes them.
STRATEGY_SCALE_MIN = 0.15
STRATEGY_SCALE_AT_REPLACEMENT = 0.5
STRATEGY_FULL_CREDIT_POSITION_RANK = 3


# Share of each FLEX slot that realistically goes to a given position.
#
# Single source of truth, deliberately: this used to be a local literal
# inside build_position_replacement_baselines(), and a 2026-08-21
# experiment that set TE to 0.0 (chasing a real "TEs rank too high vs
# point-matched WRs" finding) moved TE's replacement baseline 119.4 ->
# 133.4 and halved the VOR of the entire TE4-TE17 band in one step --
# Travis Kelce's position_value_score went 0.83 -> 0.13 and Tucker Kraft
# fell to board rank 203. Reverted, and hoisted here so the value can't
# drift between the baseline calc and the VOR-span calc below, which must
# agree or strategy_scale is normalized against a different replacement
# level than the one actually used for scoring.
POSITION_FLEX_SHARES = {"RB": 0.4, "WR": 0.4, "TE": 0.2}


def _position_replacement_rank(position):
    """
    Real replacement rank for a position under the CURRENT roster settings.
    """
    # get_league_size() seeds session state via init_session_state(). Read it
    # FIRST: reading roster_settings before that seeding returns {} on the very
    # first call of a fresh process, which made every starter_slots 0 and every
    # replacement rank 1. In the app that never bit (Home.py inits at import),
    # but it silently dropped QB from the tier calibration under a bare
    # `python -m draftkit.tests....` -- caught by test_tier_calibration test A.
    league_size = get_league_size()
    roster = st.session_state.get("roster_settings", {})
    flex_slots = int(roster.get("FLEX", 0))
    starter_slots = int(roster.get(str(position).upper(), 0))
    flex_slot_share = flex_slots * POSITION_FLEX_SHARES.get(str(position).upper(), 0.0)

    return max(int(round(league_size * (starter_slots + flex_slot_share))), 1)


# VOR-baseline-only override of _position_replacement_rank, for positions
# where the roster-formula rank and the position's real depth diverge.
#
# TE (investigated 2026-08-26, user-reported "TEs rank too high overall" +
# "mid/late TE crowd"): the roster formula gives TE rank 17 (12 teams x
# (1 starter + 2 flex x 0.2 TE-flex-share) = 16.8), which _position_
# replacement_rank still returns unchanged for its OTHER two callers
# (build_tier_calibration's real-outcomes lookup, build_position_vor_
# spans' down_span for strategy_scale) -- both are a separate, legacy
# scoring path (position_value_score/strategy_scale) that never reaches
# final_score/Our Score (see calculate_base_value_score: VOR + projection
# percentile + market + risk, no strategy_scale) and isn't surfaced on
# either active page (Home.py, draft_center.py). Rank 17 is correct THERE
# -- it's where the 2015-2025 realized-outcomes curve goes flat instead of
# collapsing every non-elite TE to the old 0.05 floor (see build_tier_
# calibration's docstring, 2026-08-21 fix).
#
# But real TE PROJECTED production is itself nearly flat from TE7 through
# TE20+ (measured: TE10 112.0 pts vs TE17 104.9 -- a 7-point spread across
# 7 ranks), while WR/RB production keeps falling steeply over that same
# range. So a replacement baseline drawn from TE17 sits IN that flat zone,
# giving every mid/late TE a VOR near zero -- rather than the deeply
# negative VOR a same-ADP WR/RB gets (measured, ADP 145-175: WR -7 to -62,
# RB -47 to -92, TE 0 to +7) -- which is what inflated the whole tier onto
# the board 25-70 spots ahead of where the real market drafts them
# (measured mean/median ADP-rank-minus-our-rank: TE +25.4/+23.5, vs RB
# -14.4/-2.0, WR -2.2/0.0, QB -15.4/-14.0 -- TE alone is a huge outlier).
#
# The real market's own signal for where TE stops being meaningfully
# scarce is TE9: real ADP jumps from ~92 (TE9) to ~153 (TE10), a 60-pick
# cliff dwarfing every other position's rank-to-rank gap at that depth.
# Anchoring the VOR baseline there (rather than TE17) keeps the elite
# tier's fair, market-beating edge (Bowers/McBride still +7/+8, unaffected
# -- they're well above either baseline) while giving the flat mid/late
# crowd a real, negative VOR instead of a false near-zero one.
VOR_REPLACEMENT_RANK_OVERRIDES = {"TE": 9}


def build_position_replacement_baselines(df, pos_col, proj_col):
    """
    Estimate position-specific replacement baselines for cross-position scoring.
    """
    baselines = {}

    for position in SCORABLE_POSITIONS:
        pos_df = df[df[pos_col].astype(str).str.upper() == position].sort_values(
            proj_col,
            ascending=False,
        )

        if pos_df.empty:
            baselines[position] = 0.0
            continue

        rank = VOR_REPLACEMENT_RANK_OVERRIDES.get(position, _position_replacement_rank(position))
        replacement_idx = min(rank - 1, len(pos_df) - 1)

        baselines[position] = float(pos_df.iloc[replacement_idx][proj_col])

    return baselines


def build_position_vor_spans(df, pos_col, proj_col, replacement_baselines):
    """
    Real per-position VOR spans that normalize strategy_scale.

    Two spans per position, both measured off that position's OWN real
    projection distribution rather than any shared constant, so a point of
    VOR means the same thing (proportionally) at every position:

    up_span:
        replacement -> the position's elite tier. Anchored on position rank
        STRATEGY_FULL_CREDIT_POSITION_RANK (3), deliberately matching the
        "1-3" premium tier that positional_tier_curve.csv itself already
        defines as the reference tier (factor 1.0 by construction). A
        top-3 player at his position therefore earns full strategy credit,
        which is what the previous thresholded rule effectively gave every
        player with position_value_score >= 11.0.

    down_span:
        replacement -> one full replacement-depth deeper (2x the
        replacement rank). Uses a real rank rather than the position's
        minimum projection because the raw pool carries a long tail of
        camp-body rows (830 TEs, 1746 WRs) whose ~0 projections would make
        the span meaningless.
    """
    spans = {}

    for position in SCORABLE_POSITIONS:
        pos_df = df[df[pos_col].astype(str).str.upper() == position].sort_values(
            proj_col,
            ascending=False,
        )

        if pos_df.empty:
            spans[position] = (1.0, 1.0)
            continue

        baseline = _safe_float(replacement_baselines.get(position), 0.0)

        elite_idx = min(STRATEGY_FULL_CREDIT_POSITION_RANK - 1, len(pos_df) - 1)
        up_span = max(_safe_float(pos_df.iloc[elite_idx][proj_col], baseline) - baseline, 1.0)

        deep_idx = min(_position_replacement_rank(position) * 2 - 1, len(pos_df) - 1)
        down_span = max(baseline - _safe_float(pos_df.iloc[deep_idx][proj_col], 0.0), 1.0)

        spans[position] = (up_span, down_span)

    return spans


_TIER_CURVE_PATH = Path("research/validation_v1/data/positional_tier_curve.csv")
_TIER_EDGES = [("1-3", 1, 3), ("4-6", 4, 6), ("7-9", 7, 9),
               ("10-12", 10, 12), ("13-18", 13, 18), ("19-30", 19, 30)]
# Beyond the fitted range every position is already at the floor.
_TIER_BEYOND_FACTOR = 0.05


_TIER_FACTOR_MIN = 0.05
_TIER_FACTOR_MAX = 1.50

# Floor for the bounded tier normalization (see build_tier_calibration()).
# Deliberately NOT _TIER_FACTOR_MIN's 0.05: that value existed to express
# "this tier returned nothing over replacement", but applied as a
# MULTIPLIER it instead expressed "this player carries no information",
# which is a different and much stronger claim. 0.25 chosen by sensitivity
# test over {0.20, 0.25, 0.30} against real within-position ordering.
#
# _TIER_FACTOR_MIN stays 0.05 and still applies to players with NO ADP at
# all (positional_tier_factor below). That is not the same case: an unpriced
# player was never drafted by anyone, which is a real signal, where a
# fitted-but-mediocre tier is a priced player the market simply likes less.
TIER_FACTOR_FLOOR = 0.25


@lru_cache(maxsize=1)
def _load_tier_curve_table():
    """
    The full (position, replacement_rank, tier) -> realized VOR grid.

    Returns (table, ranks_by_position) or ({}, {}) if the file is missing
    or malformed, which leaves scoring unchanged.
    """
    if not _TIER_CURVE_PATH.exists():
        return {}, {}
    try:
        curve = pd.read_csv(_TIER_CURVE_PATH)
    except Exception:
        return {}, {}
    if "tier_vor" not in curve.columns or "replacement_rank" not in curve.columns:
        return {}, {}

    table = {}
    ranks = {}
    for _, r in curve.iterrows():
        position = str(r["position"]).upper()
        rank = int(r["replacement_rank"])
        table[(position, rank, str(r["tier"]))] = float(r["tier_vor"])
        ranks.setdefault(position, set()).add(rank)

    return table, {pos: sorted(rs) for pos, rs in ranks.items()}


@lru_cache(maxsize=32)
def _tier_vor_for_ranks(ranks_key):
    """Select the fitted rows matching one specific replacement-rank config."""
    table, available = _load_tier_curve_table()
    if not table:
        return {}

    realized = {}
    for position, rank in ranks_key:
        options = available.get(position)
        if not options:
            continue
        # Clamp to the fitted grid rather than dropping the position: a
        # league far outside the grid still deserves the nearest real fit.
        nearest = min(options, key=lambda r: abs(r - rank))
        for label, _lo, _hi in _TIER_EDGES:
            value = table.get((position, nearest, label))
            if value is not None:
                realized[(position, label)] = value
    return realized


def load_realized_tier_vor(replacement_ranks=None):
    """
    Realized value-over-replacement by (position, positional-ADP tier),
    measured 2015-2025 by build_positional_tier_curve_v1.py.

    Player-weighted median, not a season mean: a single career could
    otherwise carry a cell (Travis Kelce was 24% of the TE1-3 sample at
    +128 VOR, inflating premium TE well past what it is worth -- and he is
    not in the 2026 premium pool at all).

    Selected for the ACTIVE league configuration (2026-08-21). The curve is
    now fit across a grid of replacement ranks rather than at one hardcoded
    set, because replacement rank determines the baseline VOR is measured
    against and therefore the SIGN of the result: the old fixed
    {"QB":12,"RB":29,"WR":29,"TE":14} described a roster format nobody here
    plays, and its negative TE mid-tiers -- the entire basis for the 0.05
    floor that collapsed every non-elite TE -- turn roughly flat at this
    league's real TE replacement rank of 17. WR was mis-specified further
    still (29 vs 46). Passing None derives the ranks from the live roster
    settings, so changing the lineup selects a different real fit instead
    of silently reusing one built for another format.
    """
    if replacement_ranks is None:
        replacement_ranks = {
            position: _position_replacement_rank(position)
            for position in SCORABLE_POSITIONS
        }
    return _tier_vor_for_ranks(tuple(sorted(replacement_ranks.items())))


def _clear_tier_curve_caches():
    _load_tier_curve_table.cache_clear()
    _tier_vor_for_ranks.cache_clear()


# load_realized_tier_vor() is a thin selector over two lru_cached layers
# rather than being cached itself (its argument is a dict). Callers -- the
# tests especially -- still expect the .cache_clear() handle the old
# lru_cached version exposed, so keep that contract.
load_realized_tier_vor.cache_clear = _clear_tier_curve_caches


def tier_label_for_rank(positional_adp_rank):
    rank = _safe_float(positional_adp_rank, None)
    if rank is None or rank <= 0:
        return None
    for label, lo, hi in _TIER_EDGES:
        if lo <= rank <= hi:
            return label
    return None


def build_tier_calibration():
    """
    Within-position tier-shape correction, matching
    build_positional_tier_curve_v1.py's own documented formula exactly:

        tier_factor(pos, tier) = realized_VOR(pos, tier) / realized_VOR(pos, "1-3")

    Corrected 2026-08-17 (task_4d5e2bb0 follow-up) -- the prior version of
    this function computed realized_VOR(pos, tier) / projected_VOR(pos,
    tier) instead, using THIS SEASON's own projections as the denominator.
    That is a different formula than the one the source script's own
    docstring specifies and explicitly warns against building: "It is NOT
    a cross-position multiplier... scaling it by a factor derived from
    cross-position VOR levels would double-count scarcity, which is
    exactly the mistake the old TE 0.68 multiplier made." Dividing by each
    position's own current projected_VOR imports exactly that cross-
    position scale difference (QB's raw point totals run far higher than
    RB's, so an equally-real absolute realized-VOR gap reads as a far
    smaller fraction of a QB's projected_VOR than of an RB's) -- verified
    directly: it produced QB 0.23 vs RB 0.57 at tier 1-3, a cross-position
    premium-tier discount the documented formula cannot produce at all
    (realized_VOR(pos,"1-3") divided by itself is 1.0 for every position,
    by construction -- there is no data-dependent way for tier 1-3 to come
    out as anything else once the formula matches the docstring).

    What this DOES still correct for, real and measured: projections are
    smooth within a position but real outcomes cliff -- TE1-3 real median
    VOR (35.7) vastly outperforms TE4-6 (-4.3) and TE7-9 (-15.1) even
    though FantasyPros projects TE4-6 close to TE1-3. That within-position
    shape is what this function now captures, uncontaminated by any
    cross-position rescaling. It takes no current-season projection or
    replacement-baseline input at all -- it is a fixed function of the
    2015-2025 realized-outcomes table alone, recomputed only when that
    table changes.

    BOUNDED NORMALIZATION, NOT A RAW RATIO (2026-08-21)
    ---------------------------------------------------
    The ratio above is unstable by construction wherever a tier's realized
    VOR approaches zero, which is exactly where most non-elite tiers sit.
    A cell moving from +1.1 to -2.6 -- well inside sampling noise at ~25
    distinct players per cell -- flips the factor from +0.02 to a floored
    0.05, converting noise into a scoring decision that annihilates the
    whole TE4-15 range. Measured on the real re-fit table, TE7-9 (-2.6) and
    TE10-12 (+11.2) differ by 14 VOR points and produced factors of -0.05
    and +0.22.

    Replaced with a bounded, sign-stable normalization over each position's
    own fitted tiers:

        factor(tier) = m + (1 - m) * (VOR_tier - VOR_min) / (VOR_1-3 - VOR_min)

    m = TIER_FACTOR_FLOOR. Tier 1-3 still evaluates to exactly 1.0 for
    every position by construction (numerator equals denominator), so the
    cross-position anti-leakage invariant this function was corrected to
    guarantee is preserved unchanged -- there is still no data-dependent
    way for a position's premium tier to come out as anything else. The
    difference is only in how the tiers BELOW it are spaced: proportionally
    within the position's own real range, instead of as a fraction of a
    premium-tier number they are nowhere near.

    A tier above the premium tier's own VOR clamps to 1.0 rather than
    exceeding it; the floor keeps the worst tier meaningfully scored rather
    than switched off.
    """
    realized = load_realized_tier_vor()
    if not realized:
        return {}

    positions = {pos for pos, _tier in realized}
    calibration = {}
    for position in positions:
        base = realized.get((position, "1-3"))
        if base is None or base <= 0:
            continue  # no stable premium-tier anchor to normalize against

        fitted = [
            realized[(position, label)]
            for label, _lo, _hi in _TIER_EDGES
            if (position, label) in realized
        ]
        if not fitted:
            continue

        floor_vor = min(fitted)
        span = base - floor_vor
        if span <= 0:
            continue  # degenerate: every tier identical, no shape to fit

        for label, _lo, _hi in _TIER_EDGES:
            realized_vor = realized.get((position, label))
            if realized_vor is None:
                continue
            normalized = (realized_vor - floor_vor) / span
            factor = TIER_FACTOR_FLOOR + (1.0 - TIER_FACTOR_FLOOR) * normalized
            calibration[(position, label)] = float(
                min(max(factor, TIER_FACTOR_FLOOR), 1.0)
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


# Manual, dated projection adjustments from real, current information not yet
# reflected in the sourced projection -- role/opportunity signals (coach
# comments, beat-reporter depth-chart notes) or injury signals too acute for
# the historical risk model to see yet. Not derivable from stats, same class
# of override as TEAM_CHANGED_PLAYERS (build_rb_archetypes.py)/
# INJURY_MANUAL_OVERRIDES (build_risk_variables.py).
#
# Real, load-bearing distinction from risk_index (Home.py's RISK column):
# risk_index never feeds this scoring pipeline at all (confirmed via a
# full-file grep of this module -- zero references). Only projection_points
# does, through calculate_position_value_score() (the "projection" component)
# and calculate_value_score() (the "adp_value" component) below. So this is
# the one real integration point where a manual override actually moves
# OUR SCORE/rank, not just a descriptive badge -- adjusting projection_points
# here flows into both automatically, the same way a real projection update
# would, with no new scoring-component wiring needed.
#
# Jordyn Tyson is a retrofit: his existing INJURY_MANUAL_OVERRIDES entry only
# ever set injury_score (RISK column only) -- the exact real, current,
# decisive hamstring news it was built for never moved his rank at all until
# this entry. Every percentage below is an editorial magnitude estimate, not
# a verified fact -- the underlying situations (trades, coach quotes) are
# independently verified real news; the specific number is a judgment call.
# Sizing/mechanism/confidence rules for every entry below: see
# research/news_override_policy.md (written 2026-08-26 by extracting the
# rules already applied consistently across these entries -- most load-
# bearing rule: only add an entry HERE (a season-points cut) when there's a
# real, DECISIVE detail (a confirmed return week, a confirmed role split);
# an uncertain/pending situation belongs in INJURY_MANUAL_OVERRIDES
# instead, see build_risk_variables.py).
PROJECTION_MANUAL_ADJUSTMENTS = {
    "DeVonta Smith": {
        "pct": 12.0,
        "note": "AJ Brown traded to NE -- Smith is Philly's clear WR1 now",
        "source": "ESPN/Yahoo, real 2026 offseason reporting",
        "date": "2026-08-16",
        # requires build_risk_variables.py re-run to take effect, unlike pct
        # (which is live on the next Streamlit rerun) -- see role_usage_td_score
        # override wiring in build_risk_variables.py.
        "usage_risk_score": 1.5,
    },
    "Jordyn Tyson": {
        # Updated 2026-08-17: superseded the prior -15.0% "uncertain for
        # Week 1" entry -- news has firmed up to a real, decisive out-until-
        # Week-9 timeline (user-reported; my own web check the same day
        # corroborated the underlying hamstring re-injury and prior Week-1-
        # uncertain reporting but had not yet indexed a source confirming
        # the specific Week 9 date -- treated as current per the user's
        # direct report, not independently re-verified beyond that). Missing
        # weeks 1-8 is ~8 of 17 games (~47%) -- -50.0% is a rough, disclosed
        # games-missed-proportional estimate (round number, not a precision
        # ramp-up/target-redistribution model), same editorial-estimate
        # standard as every other entry here.
        "pct": -50.0,
        "note": "Hamstring re-injury -- real, decisive timeline: out until Week 9",
        "source": "User-reported 2026-08-17; underlying injury independently corroborated via web search "
                  "(ESPN/NBC Sports/ProFootballRumors)",
        "date": "2026-08-17",
    },
    "Luther Burden": {  # real player_name in master_players.csv omits the "III" suffix
        "pct": 10.0,
        "note": "DJ Moore traded to Buffalo; real 2026 camp buzz (HC Ben Johnson), but no "
                "established real target share yet reflecting the new opportunity",
        "source": "Multiple outlets (RotoBaller, CBS Sports, BearsTalk)",
        "date": "2026-08-16",
    },
    "Blake Corum": {
        "pct": 8.0,
        "note": "Real McVay 'big factor' comments, but Kyren Williams remains the more-trusted "
                "back per the same sources -- real but incremental, not a lead-role change",
        "source": "SI.com/NBC Sports",
        "date": "2026-08-16",
    },
    # Jahmyr Gibbs deliberately excluded from this dict (reverted 2026-08-19,
    # model_proj_staleness_fix_plan.pdf, Step 1): the underlying issue --
    # stale team-context/competitor-volume features -- is a diagnosed defect
    # inside the model's own feature pipeline (build_live_projections_v1.py),
    # not a subjective situational read layered on an already-good external
    # number. It belongs in model_projection_points_adjusted (see
    # research/validation_v1/build_live_projections_v1.py), not here --
    # PROJECTION_MANUAL_ADJUSTMENTS is reserved for narrative/editorial
    # overrides on the FantasyPros-sourced projection_points (Smith, Tyson,
    # Burden, Corum), where the external number itself has no known defect.
    #
    # Jaylen Warren deliberately excluded: real, but genuinely conflicting
    # signal (a defined passing-down role vs. a real report his early-down
    # work is shrinking toward Rico Dowdle) -- a split-the-difference number
    # wouldn't accurately reflect either real direction. Add once the
    # picture clarifies, consistent with "forward, seeded opportunistically."

    # Broncos backfield (checked 2026-08-27, no entry needed): a "set to
    # market" nudge was considered and REJECTED on inspection -- the board
    # already has Harvey ahead of Dobbins (161.6 vs 141.5) in market ADP
    # order, via their existing model_projections_v1.csv companion entries.
    # The raw FantasyPros projection_points in master_players.csv still read
    # the other way (Harvey 154.6, Dobbins 161.8), which is misleading if
    # read on its own -- the model-correction layer is what the board
    # actually scores off.
}


def apply_projection_adjustments(df, player_col, proj_col):
    """Applies PROJECTION_MANUAL_ADJUSTMENTS to projection_points in place.
    Stamps projection_adjustment_pct/projection_adjustment_note (NaN/None for
    everyone else) so the adjustment is never silent -- Home.py surfaces it
    as a visible marker on PROJECTED PTS, same transparency principle as
    Home.py's _injury_override_badge()."""
    df["projection_adjustment_pct"] = np.nan
    df["projection_adjustment_note"] = None
    for name, override in PROJECTION_MANUAL_ADJUSTMENTS.items():
        mask = df[player_col] == name
        if not mask.any():
            continue
        df.loc[mask, proj_col] = df.loc[mask, proj_col] * (1 + override["pct"] / 100)
        df.loc[mask, "projection_adjustment_pct"] = override["pct"]
        df.loc[mask, "projection_adjustment_note"] = (
            f"{override['note']} ({override['source']}, {override['date']})"
        )
    return df


MODEL_PROJECTIONS_PATH = Path("data/processed/model_projections_v1.csv")

# Real, structural gap found and fixed (2026-08-21, "every player needs to
# be running on the Our Score model-adjusted number" audit): master_players
# .csv (FantasyPros-sourced, the real board's own player universe) and
# model_projections_v1.csv (this model's own crosswalk, built from
# stats_player_reg_by_season/roster_2026.csv) can carry a REAL, different
# canonical name for the same real person -- confirmed directly for Marquise
# "Hollywood" Brown: master_players.csv's live, scored, ranked row uses
# "Hollywood Brown" (with real ADP/market data), while a SEPARATE,
# structurally empty "Marquise Brown" row also exists there (no ADP, no
# projection_points at all -- a real duplicate/phantom entry, not the one
# anyone actually drafts). This model's own correction was written keyed to
# "Marquise Brown" (see build_live_projections_v1.py) since that's the name
# stats_player_reg_by_season/roster_2026.csv use for him -- a real, correctly
# -verified correction that was silently landing on the WRONG, invisible
# duplicate row instead of the real, live-scored one. Different failure mode
# from NAME_KEYED_FALLBACK_OVERRIDES (that one covers a player ABSENT from
# this model's own crosswalk; this covers a name mismatch AT THE BOUNDARY
# between this model's crosswalk and master_players.csv's separate one).
# Keyed on the name model_projections_v1.csv actually uses -> the real name
# master_players.csv uses for the SAME live, scored player. Add an entry
# here ONLY when directly confirmed (a phantom master_players.csv row with
# no real ADP/projection data under the model's own name), not speculatively.
MODEL_TO_MASTER_NAME_ALIASES = {
    "Marquise Brown": "Hollywood Brown",
}


def apply_model_projection_override(df, player_col, proj_col):
    """Replaces proj_col with this project's own researched value, for every
    player who actually has one -- the "replace where corrected" decision
    (2026-08-21) for wiring the whole team-by-team WR/TE/RB pass plus the
    QB Book-anchored pass into OUR SCORE/Rank, which until now only the
    small, older PROJECTION_MANUAL_ADJUSTMENTS dict above could reach.

    Runs strictly AFTER apply_projection_adjustments (same real integration
    point that dict uses -- see the comment above PROJECTION_MANUAL_
    ADJUSTMENTS: proj_col is the one column that actually reaches OUR
    SCORE/rank). Last-write-wins is deliberate: for any player covered by
    BOTH mechanisms (DeVonta Smith, Jordyn Tyson, Luther Burden all
    overlap), this session's researched value -- individually verified,
    dated, pool-conservation-checked -- supersedes the older, thinner
    editorial guess rather than stacking on top of it. The legacy
    projection_adjustment_pct/note badge is cleared for those players too,
    so Home.py doesn't keep showing a stale rationale for a number that
    no longer reflects it.

    Only counts as "corrected" if a REAL research signal exists:
    model_projection_points_adjusted where model_adjustment_pct is notna
    (a real MODEL_PROJECTION_CORRECTIONS entry, even a disclosed 0% "
    confirmed, no change") OR model_projection_points_fallback where notna
    (a rookie/name-keyed placeholder, or a QB Book-anchored value).
    Deliberately does NOT fall back to the bare, uncorrected
    model_projection_points -- that's the model output this whole model's
    own docstring says validated WORSE than ADP for QB and is only a
    "newer, less-proven" signal even at RB/WR/TE (see build_live_
    projections_v1.py); an untouched raw number has no human research
    behind it and shouldn't silently replace projection_points."""
    if not MODEL_PROJECTIONS_PATH.exists():
        df["model_override_applied"] = False
        df["model_override_note"] = None
        return df

    model_df = pd.read_csv(MODEL_PROJECTIONS_PATH)
    needed = ["player_name", "model_projection_points_adjusted", "model_adjustment_pct",
              "model_adjustment_note", "model_projection_points_fallback", "model_projection_fallback_note"]
    model_df = model_df[[c for c in needed if c in model_df.columns]].drop_duplicates("player_name")

    corrected = model_df["model_adjustment_pct"].notna() if "model_adjustment_pct" in model_df.columns else pd.Series(False, index=model_df.index)
    override_value = model_df["model_projection_points_adjusted"].where(corrected) if "model_projection_points_adjusted" in model_df.columns else pd.Series(np.nan, index=model_df.index)
    # Real gap found and fixed (2026-08-21, "what have you done to the
    # middle rounds" audit): compute_rookie_fallback()'s own generic
    # percentile-mapped estimate (pre-draft composite percentile -> same
    # percentile of validated veteran model_projection_points, build_live_
    # projections_v1.py) was flowing through this override exactly like a
    # real, dated, human-researched correction -- fully replacing proj_col
    # AND fully exempted from ADP-anchoring (apply_adp_anchor()'s lam=1.0
    # for model_override_applied rows). That mechanism's own note discloses
    # it plainly: "Rookie-score fallback (not a real outcome-trained
    # prediction)" -- a rough statistical placeholder for an otherwise-
    # blank cell, never meant to carry the same confidence as this
    # project's real research (Carnell Tate's independent build, Chris
    # Bell's, the QB Book-anchored pass, etc. -- all real, dated, sourced,
    # and rightly still exempted). Confirmed live: Seth McGowan's real
    # FantasyPros projection is 21.6 (sensible for an unproven rookie);
    # this fallback replaced it with 182.3 (a 90th-percentile veteran-
    # mapped number), pushed him above the RB replacement baseline,
    # undamped his strategy components, and combined with an unrelated
    # ADP-value bug (see build_adp_value_rankings_df()) to land him at
    # rank 40 overall. Excluded from proj_col entirely here -- it still
    # shows in the separate MODEL PROJ comparison column (Home.py's own
    # direct merge of model_projections_v1.csv), just doesn't drive Our
    # Score with false confidence. Identified by the literal,
    # programmatically-generated note text (not a per-player judgment
    # call) so this is precise, not a broad rookie penalty -- a REAL
    # rookie correction with actual dated research (placeholders above)
    # is untouched by this filter.
    is_composite_fallback = pd.Series(False, index=model_df.index)
    if "model_projection_fallback_note" in model_df.columns:
        is_composite_fallback = model_df["model_projection_fallback_note"].fillna("").str.contains(
            "Rookie-score fallback", regex=False
        )
    if "model_projection_points_fallback" in model_df.columns:
        fallback_value = model_df["model_projection_points_fallback"].where(~is_composite_fallback)
        override_value = override_value.fillna(fallback_value)
    override_note = model_df["model_adjustment_note"].where(corrected) if "model_adjustment_note" in model_df.columns else pd.Series(None, index=model_df.index)
    if "model_projection_fallback_note" in model_df.columns:
        override_note = override_note.fillna(model_df["model_projection_fallback_note"])

    lookup = model_df[["player_name"]].copy()
    lookup["_override_value"] = override_value
    lookup["_override_note"] = override_note
    lookup = lookup[lookup["_override_value"].notna()]
    # Remap to master_players.csv's real name BEFORE merging, so a real
    # correction attaches to the live, scored row rather than a phantom
    # duplicate under this model's own crosswalk name (see
    # MODEL_TO_MASTER_NAME_ALIASES above).
    lookup["player_name"] = lookup["player_name"].replace(MODEL_TO_MASTER_NAME_ALIASES)

    df = df.merge(lookup, left_on=player_col, right_on="player_name", how="left", suffixes=("", "_mdl"))
    has_override = df["_override_value"].notna()
    df.loc[has_override, proj_col] = df.loc[has_override, "_override_value"]
    # Legacy PROJECTION_MANUAL_ADJUSTMENTS badge cleared for overridden
    # players -- it described a different, now-superseded number.
    df.loc[has_override, "projection_adjustment_pct"] = np.nan
    df.loc[has_override, "projection_adjustment_note"] = None
    df["model_override_applied"] = has_override
    df["model_override_note"] = df["_override_note"].where(has_override)
    df = df.drop(columns=[c for c in ("player_name_mdl", "_override_value", "_override_note") if c in df.columns])
    if "player_name" in df.columns and player_col != "player_name":
        df = df.drop(columns=["player_name"])
    return df


# Real, position-specific decay scale for calculate_position_value_score()'s
# below-replacement soft floor (plan_below_replacement_floor_collapse.pdf,
# 2026-08-17). Each value is the real median |value_over_replacement| among
# that position's current below-replacement pool, EXCLUDING players with a
# literal zero projection (24-109 per position -- up to a quarter of the
# below-replacement pool at some positions -- carry no real point signal to
# preserve at all, so including them would anchor the scale to noise, not
# real spread). At value_over_replacement == -scale, the transform below
# returns exp(-1) ~= 37% of the at-replacement value -- a real,
# data-anchored "typical below-replacement depth" reference, not a picked
# number. Recompute if replacement baselines or the projection source
# change meaningfully enough to shift this real distribution.
BELOW_REPLACEMENT_DECAY_SCALE = {
    "QB": 288.7,
    "RB": 129.6,
    "WR": 104.2,
    "TE": 101.1,
}
_DEFAULT_DECAY_SCALE = 100.0  # any position missing from the table above


def calculate_position_value_score(projection_points, position, replacement_baselines):
    """
    Convert raw projection into position-adjusted value over replacement.

    Above replacement: unchanged, linear VOR (see
    plan_below_replacement_floor_collapse.pdf's explicit requirement not to
    touch this side -- confirmed via before/after check that the
    above-replacement tier is untouched).

    Below replacement (fixed 2026-08-17,
    plan_below_replacement_floor_collapse.pdf): previously
    `max(value_over_replacement, 0.0)` -- a literal hard clamp. Confirmed
    directly on Jordyn Tyson's real case that this was discarding real,
    correctly-computed information: his projection genuinely updated
    (119.77 -> 70.45 points on a real, decisive injury-timeline update),
    but both values sat below the WR replacement baseline (170.2), so both
    collapsed to the exact same adjusted_value and his rank never moved --
    not a caching bug, confirmed by decomposing every component. Real,
    measured scope: 646 of 730 players on the current board (88.5%) sit
    below their position's replacement baseline right now, so this wasn't
    a rare edge case.

    Replaced with a smooth exponential decay, continuous with the linear
    side at value_over_replacement == 0 (exp(0) == 1, matching the "+1.0"
    boundary exactly), monotonic (real magnitude differences among
    below-replacement players keep registering -- someone 5 points below
    replacement now clearly outranks someone 150 points below), and always
    strictly positive (a far-below-replacement player can never invert
    rankings with a wild negative score, satisfying the same "replacement
    is the reference point" intent the old hard clamp was protecting,
    without discarding real information to do it).
    """
    position = str(position).upper()
    projection = _safe_float(projection_points, 0.0)
    baseline = _safe_float(replacement_baselines.get(position), 0.0)
    value_over_replacement = projection - baseline
    positional_multiplier = _safe_float(POSITION_VALUE_MULTIPLIERS.get(position), 1.0)

    if value_over_replacement >= 0:
        adjusted_value = (value_over_replacement + 1.0) * positional_multiplier
    else:
        scale = BELOW_REPLACEMENT_DECAY_SCALE.get(position, _DEFAULT_DECAY_SCALE)
        adjusted_value = np.exp(value_over_replacement / scale) * positional_multiplier

    # Precision raised 2 -> 6 decimals (plan_below_replacement_floor_collapse.pdf
    # follow-up, 2026-08-17): 2 decimals was a legacy display-oriented
    # convention -- fine for the old hard-clamp output (which only ever took
    # a handful of distinct values anyway) but not for the soft floor above,
    # whose whole point is to preserve real, continuous below-replacement
    # differentiation. Rounding this early to 2 decimals, then multiplying
    # by a per-player tier_factor as small as 0.05 in the row loop below,
    # collapsed the entire below-replacement population (646 real players)
    # into just 5 distinct values -- confirmed directly. No other component
    # in this pipeline rounds a value this early AND THEN multiplies it by a
    # per-player factor this small (checked calculate_adp_bonus,
    # calculate_value_score, calculate_team_fit_score -- none combine
    # premature rounding with a compressing multiplier the way this did).
    return round(max(adjusted_value, 0.01), 6)


def compute_strategy_scale(projection_points, position, replacement_baselines, vor_spans):
    """
    How much of the strategy tilt (position need, tier urgency, team fit) a
    player earns -- continuous in real VOR points, never zero.

    Replaces (2026-08-21) a binary cutoff in
    calculate_final_recommendation_score():

        surplus = position_value_score - 1.0
        strategy_scale = clamp(surplus / 10.0, 0.0, 1.0)

    That rule read position_value_score, which is POST-tier_factor. TE's
    calibration curve floors tier_factor at 0.05 for every tier outside
    "1-3" (realized tier_vor is negative or zero for all of them), so every
    non-elite TE was driven below 1.0, strategy_scale collapsed to exactly
    0, and 55% of the total weight (position_need 0.25 + tier_urgency 0.15
    + team_fit 0.15) became the constant 50 for the entire position at
    once. With projection also compressed by the same 0.05 factor,
    adp_value's 0.15 was left effectively deciding the order -- measured
    live: Mike Gesicki (91.2 projected, ADP 207) and Cade Otton (101.7,
    ADP 189) both outranked Travis Kelce (135.0, ADP 121), a 33-44 point
    projection gap inverted purely by being cheap.

    The historical finding underneath that curve ("mid-round TEs
    disappoint") is real and is NOT discarded here -- it belongs in the
    tier curve, lowering the position's overall appeal. What it must not
    do is destroy the model's ability to tell TE7 from TE17, which is a
    separate question from how much that tier is worth in the abstract.

    So: scale on real VOR *points* (pre-tier_factor), normalized by the
    position's own real spans (build_position_vor_spans), piecewise-linear
    and continuous at replacement, bounded to
    [STRATEGY_SCALE_MIN, 1.0]:

        vor >= 0:  0.5 -> 1.0   across replacement -> position rank 3
        vor <  0:  0.5 -> 0.15  across replacement -> 2x replacement rank

    Continuous at vor == 0 (both branches give
    STRATEGY_SCALE_AT_REPLACEMENT), monotonic in vor, and identical in
    form at every position, so a middling TE can legitimately earn less
    strategy credit than an elite WR while Kelce still earns materially
    more than a 100-point TE -- and that difference survives into
    final_score instead of being flattened away before it gets there.
    """
    position = str(position).upper()
    projection = _safe_float(projection_points, 0.0)
    baseline = _safe_float(replacement_baselines.get(position), 0.0)
    value_over_replacement = projection - baseline

    up_span, down_span = vor_spans.get(position, (1.0, 1.0))

    if value_over_replacement >= 0:
        ratio = min(value_over_replacement / up_span, 1.0) if up_span > 0 else 1.0
        scale = STRATEGY_SCALE_AT_REPLACEMENT + (
            1.0 - STRATEGY_SCALE_AT_REPLACEMENT
        ) * ratio
    else:
        ratio = min(-value_over_replacement / down_span, 1.0) if down_span > 0 else 1.0
        scale = STRATEGY_SCALE_MIN + (
            STRATEGY_SCALE_AT_REPLACEMENT - STRATEGY_SCALE_MIN
        ) * (1.0 - ratio)

    return round(min(max(scale, STRATEGY_SCALE_MIN), 1.0), 6)


# === Base Value (2026-08-21) ==============================================
#
# Replaces final_score's dependency on the multi-layer blend below (tier
# calibration, strategy_scale/position_value_score's exp-decay transform,
# tier urgency, position need, team fit, signal-trust damping, single-QB
# multiplier, ADP-anchor blend) for the player-VALUE question. That chain
# accumulated one fragile layer at a time, each able to move a player's
# rank independently, and it stopped being auditable: Dalton Kincaid (TE3,
# overall 26) and Kyle Pitts (TE4, overall 35) outranked Kyren Williams
# (216.4 projected RB points) despite trailing him by 60-80 points, and no
# single row explained why -- the answer was scattered across five
# functions. Measured real-board consequence: 30 TEs in the top 150 against
# 61 WRs, in a format that starts 36 WR slots (3 WR + 3-WR-share of 2 FLEX)
# before a single true TE flex exists.
#
# Base Value asks a narrower, more stable question -- "how good is this
# player" -- and answers it from four real, already-computed quantities,
# combined with fixed documented weights and NO per-position transform:
#
#     Base Value = a*VOR + b*projection + market_swing - risk_penalty
#
# VOR and projection are collinear within a position (VOR is just
# projection minus that position's own real replacement baseline, a
# constant), so within a position this is monotonic in projection by
# construction -- there is no arithmetic path for a 100-point TE to
# outscore a 135-point TE unless market_swing/risk_penalty (both capped,
# both far smaller than a real tier gap) supply a specific, visible reason.
# VOR alone carries the cross-position scarcity signal (via each position's
# baseline), so a raw 350-point QB does not automatically outrank a
# 300-point RB the way pure projection would.
#
# Position need, team fit, and tier urgency are deliberately NOT inputs
# here -- they are roster-construction questions, not player-value
# questions, and belong in a separate live "Pick Recommendation" layer
# (Roster Need + VONA/next-pick risk + construction fit) added on top of
# Base Value only once a draft is actually in progress. That layer is not
# built yet; Base Value is the entire final_score for now, at every roster
# state, so their contribution is exactly zero rather than damped toward
# neutral -- see calculate_final_recommendation_score()'s legacy
# position_need_component_score / team_fit_component_score, which still
# compute for display/backward-compatibility but no longer reach
# final_score at all.
BASE_VALUE_VOR_WEIGHT = 1.0

# Projection's OWN term is a bounded, WITHIN-POSITION percentile bonus (0 to
# +20), not raw points added on top of VOR (fixed 2026-08-21, same session:
# the first version used BASE_VALUE_PROJECTION_WEIGHT=0.5 on raw
# projection_points directly, and it reintroduced exactly the cross-
# position scale problem VOR exists to remove -- measured live, it put 10
# QBs in the top 30 and 28 in the top 150. Real cause: QB's genuine VOR
# spread (QB1 60.1 points over the replacement baseline) is legitimately
# the SHALLOWEST of any position -- the top 12 QBs cluster tightly, which
# is real and correct -- but raw projection_points ignores that entirely
# and rewards QB1's 355.9 raw points on the same absolute scale as an RB's
# 306, importing the passing-game scoring-rule advantage QB carries over
# every other position regardless of real scarcity.
#
# VOR is exactly monotonic in projection within a position already (same
# real points, position-constant baseline subtracted), so this bonus is not
# load-bearing for the "higher projection can't rank below a position-mate"
# guarantee -- VOR alone already provides that, exactly, in every case.
# What it adds is a small extra credit for being the best in your own
# position group, bounded to the same order of magnitude as the market and
# risk terms so it stays a reinforcement, never a second cross-position
# scale.
BASE_VALUE_PROJECTION_BONUS_MAX = 20.0

# Bounded, zero-centered swing: a player with the best possible market
# treatment gets +6, the worst gets -6, relative to a neutral 0 for a
# player with no ADP at all. Deliberately small next to real inter-tier VOR
# gaps (which run 20-170+ points) so ADP can settle a close call but cannot
# invert a material one -- the single-QB multiplier, signal-trust damping,
# and ADP-anchor blend this replaces had no such cap and each could move a
# score by double digits or more on their own.
BASE_VALUE_MARKET_MAX_SWING = 12.0

# Discount only -- never a bonus. injury_risk is 0-100 (higher = riskier,
# 50.0 the neutral default for missing data, see get_player_injury_risk()).
# Scaled so the worst real risk score costs 15 points, comparable in
# magnitude to the market swing above and, again, well short of what a real
# tier-defining projection gap should require to overturn.
BASE_VALUE_RISK_MAX_PENALTY = 15.0


def calculate_base_value_score(df):
    """
    Vectorized Base Value for every row in df at once (not a per-row apply,
    since the market-swing term needs the whole pool's ADP distribution to
    resolve a percentile).

    Requires df to already carry projection_points, value_over_replacement_points,
    adp, and injury_risk (all present on the frame _build_base_recommendation_rankings_df()
    produces). Returns df with four new component columns plus base_value_score,
    the exact, un-adjusted sum of those four -- every rank difference this
    function produces is readable off one row with no later step able to
    change it.
    """
    out = df.copy()
    projection = pd.to_numeric(out["projection_points"], errors="coerce").fillna(0.0)
    vor = pd.to_numeric(out.get("value_over_replacement_points"), errors="coerce").fillna(
        -projection  # no baseline resolved for this row -> treat as fully below replacement
    )
    adp = pd.to_numeric(out.get("adp"), errors="coerce")
    risk = pd.to_numeric(out.get("injury_risk"), errors="coerce").fillna(50.0).clip(0.0, 100.0)

    out["base_value_vor_component"] = (BASE_VALUE_VOR_WEIGHT * vor).round(3)

    positions = out["position"].astype(str).str.upper()
    proj_pct_in_position = projection.groupby(positions).rank(pct=True, method="average")
    out["base_value_projection_component"] = (
        proj_pct_in_position.fillna(0.0) * BASE_VALUE_PROJECTION_BONUS_MAX
    ).round(3)

    has_adp = adp.notna()
    market = pd.Series(0.0, index=out.index)
    if has_adp.sum() >= 2:
        # Lower ADP is better; percentile-rank so one extreme value can't
        # skew the whole scale, then re-center on 0 and cap the swing.
        adp_pct = (-adp[has_adp]).rank(pct=True)
        market.loc[has_adp] = (adp_pct - 0.5) * BASE_VALUE_MARKET_MAX_SWING
    out["base_value_market_component"] = market.round(3)

    out["base_value_risk_component"] = (-(risk / 100.0) * BASE_VALUE_RISK_MAX_PENALTY).round(3)

    out["base_value_score"] = (
        out["base_value_vor_component"]
        + out["base_value_projection_component"]
        + out["base_value_market_component"]
        + out["base_value_risk_component"]
    ).round(3)

    return out


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
    df = apply_projection_adjustments(df, player_col, proj_col)
    df = apply_model_projection_override(df, player_col, proj_col)

    if adp_col is not None:
        df[adp_col] = pd.to_numeric(df[adp_col], errors="coerce")

    df = df.sort_values(proj_col, ascending=False).reset_index(drop=True)
    df["projection_rank"] = df.index + 1

    context = _build_scoring_context(df, columns)
    replacement_baselines = build_position_replacement_baselines(df, pos_col, proj_col)
    vor_spans = build_position_vor_spans(df, pos_col, proj_col, replacement_baselines)

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

    tier_calibration = build_tier_calibration()
    need_weights = get_position_need_weights()
    team_profile = get_team_profile()
    tiers_df = build_position_tiers_df()
    tier_summary_df = build_tier_summary_df(tiers_df=tiers_df)

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
        # Precision raised 2 -> 6 (see calculate_position_value_score()'s
        # own docstring/comment for why): this is where a small tier_factor
        # (as low as 0.05) previously compounded with premature 2-decimal
        # rounding to collapse the below-replacement population's real
        # differentiation into a handful of shared buckets.
        position_value_score = round(max(position_value_score * tier_factor, 0.01), 6)
        # Computed from the PRE-tier_factor projection deliberately (see
        # compute_strategy_scale): tier_factor answers "how much is this
        # tier worth", strategy_scale answers "how far above/below
        # replacement is this specific player". Reading the post-factor
        # score conflated the two and let a floored tier curve switch the
        # strategy terms off for an entire position at once.
        strategy_scale = compute_strategy_scale(
            projection,
            position,
            replacement_baselines,
            vor_spans,
        )
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

        # This player's OWN real tier within his position (same lookup
        # calculate_tier_bonus already does above) -- carried through so
        # build_master_recommendations_df() can look up HIS tier's real
        # urgency (see build_full_tier_urgency_df()) instead of broadcasting
        # the position's single current-tier value to every player at that
        # position regardless of which tier they're actually in. Named
        # "position_tier", not "tier" -- Home.py separately assigns a
        # whole-board "tier" display column (automatic score-gap bands,
        # _assign_automatic_tiers()), an unrelated concept this would be
        # confusable with under the same name.
        tier_match = tiers_df[tiers_df["Player"].astype(str).str.lower() == player_name.lower()] if not tiers_df.empty else tiers_df
        player_position_tier = int(tier_match.iloc[0]["Tier"]) if not tier_match.empty else None

        rows.append({
            "player_name": player_name,
            "position": position,
            "position_tier": player_position_tier,
            "team": str(row[team_col]) if team_col is not None else "",
            "projection_points": projection,
            "position_value_score": position_value_score,
            "value_over_replacement_points": round(
                projection - _safe_float(replacement_baselines.get(position), 0.0), 3
            ),
            "strategy_scale": strategy_scale,
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
            "projection_adjustment_pct": row.get("projection_adjustment_pct"),
            "projection_adjustment_note": row.get("projection_adjustment_note"),
            "model_override_applied": bool(row.get("model_override_applied", False)),
            "model_override_note": row.get("model_override_note"),
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


# log(x + epsilon) before min-max, opt-in per component
# (plan_below_replacement_floor_collapse.pdf follow-up, 2026-08-17): fixing
# calculate_position_value_score()'s hard clamp restored real magnitude
# differentiation among below-replacement players, but linear min-max
# against position_value_score's real range (0.01 to 166.5, a handful of
# elite-RB outliers at the top) still compressed that differentiation into
# a slice of the range smaller than 2-decimal rounding can register --
# confirmed directly: Jordyn Tyson's real, updated position_value_score
# (0.03 -> 0.02, genuinely different) both normalized to 0.01. Population
# check confirmed this isn't Tyson-specific: the 25th/50th/75th percentiles
# of the FULL pool's position_value_score are all exactly 0.01, i.e. the
# median sits at the literal floor of the range. Log compresses large
# values proportionally more than small ones, so real relative
# differentiation near the floor survives without an equivalent flattening
# at the top -- checked directly against every other component this
# function normalizes (adp_value, tier_urgency, position_need, team_fit)
# before applying: only `projection` (source: position_value_score) shows
# this median-at-the-floor pattern. adp_value has some skew too (median at
# 2.9% of its range) but its own median (-26.23) sits clearly apart from
# its floor (-30), i.e. real differentiation already survives there --
# applying the same transform to a component that isn't actually broken
# risks distorting a distribution shape that's fine as-is, so it stays on
# the existing linear path.
_LOG_TRANSFORM_EPSILON = 0.001


def _normalize_series_to_100(series, neutral_score=50.0, lower_bound=None, upper_bound=None, log_transform=False):
    values = pd.to_numeric(series, errors="coerce")

    if log_transform:
        values = np.log(values.clip(lower=0.0) + _LOG_TRANSFORM_EPSILON)
        if lower_bound is not None:
            lower_bound = np.log(max(lower_bound, 0.0) + _LOG_TRANSFORM_EPSILON)
        if upper_bound is not None:
            upper_bound = np.log(max(upper_bound, 0.0) + _LOG_TRANSFORM_EPSILON)

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
        "projection": {"neutral": 50.0, "lower": None, "upper": None, "log_transform": True},
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
            log_transform=defaults.get("log_transform", False),
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

    Revised 2026-08-21: that damping is still here and still does the above
    job, but it is no longer a binary switch. It previously keyed on
    position_value_score (POST-tier_factor) against a hard 1.0 cutoff,
    which meant a position whose calibration curve floors out -- TE, whose
    realized tier_vor is negative or zero for every tier outside "1-3" --
    had 55% of its weight collapse to a flat 50 for every player at once,
    leaving adp_value to decide the order and inverting real 33-44 point
    projection gaps. It now keys on strategy_scale, which is continuous in
    real VOR points and bounded strictly above zero. See
    compute_strategy_scale().
    """
    weights = _resolve_component_weights(component_weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    # Continuous in real VOR points and bounded strictly above zero -- see
    # compute_strategy_scale(), which precomputes this per row where the
    # replacement baselines and per-position spans are actually available.
    scale_value = _safe_float(row.get("strategy_scale"), None)
    if scale_value is not None:
        strategy_scale = min(max(scale_value, STRATEGY_SCALE_MIN), 1.0)
    else:
        # Callers that hand-build a row (tests, ad-hoc scoring) won't carry
        # strategy_scale. Fall back to the old position_value_score rule,
        # but floored at STRATEGY_SCALE_MIN rather than 0.0: the binary
        # switch-off is the exact behaviour this replaced, and it should
        # not survive in a fallback path either.
        raw_value = _safe_float(row.get("position_value_score"), None)
        if raw_value is None:
            strategy_scale = 1.0
        else:
            surplus = raw_value - REPLACEMENT_LEVEL_VALUE
            strategy_scale = min(max(surplus / 10.0, STRATEGY_SCALE_MIN), 1.0)

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
    Damp recommendation scores when upstream data quality is genuinely low.

    This is intentionally a post-score guardrail: the core recommendation model
    can still expose what it would have liked, while low-trust DATA cannot make a
    player the top recommendation.

    plan_deanchor_scoring_from_adp.pdf, Phase 1: real, confirmed circularity
    fixed at the source (draftkit/signal_trust.py) -- signal_trust_score and
    adp_trust_score now measure ONLY genuine data-quality issues (missing/
    inconsistent/out-of-range ADP, missing/invalid projections, sportsbook
    provider health). A real model-vs-ADP or model-vs-market disagreement no
    longer lowers either score -- disagreeing with the market is the whole
    point of running a proprietary model, not evidence it's broken.

    The old `severe_flags` tier (elite_projection_late_adp_conflict,
    extreme_adp_projection_gap, projection_rank_conflict, adp_rank_conflict)
    is REMOVED, not reworked: every one of those four flags was itself a
    model-vs-ADP rank-gap check, now emitted as `market_divergence_flags`
    (disclosed, never dampened) instead of `anomaly_flags`. Measured on the
    real live board before this fix: trust_adjustment_applied fired for 617
    of 730 players (84.5% of the entire pool), and the severe_flags tier
    alone accounted for real elite players (e.g. most startable QBs)
    getting hard-capped at 58.0 purely for the model legitimately valuing
    them differently than ADP.

    Real, necessary companion fix, caught by re-measuring rather than
    assumed safe: the dampening THRESHOLDS below (previously 50/70/40) were
    calibrated against a distribution that included the now-removed
    circular deductions. Left unchanged, they become mathematically
    unreachable -- measured directly on the full real 3941-player pool
    post-fix, signal_trust_score's real floor is 70.0 and adp_trust_score's
    is 50.0, so `< 50`/`< 70`/`< 40` would NEVER fire for anyone, silently
    deleting the legitimate-anchoring safeguard Phase 2 explicitly requires
    (a genuine thin-sample/data-conflict case must still get dampened).
    Recalibrated against the real, corrected, discretized distribution:
    signal_trust_score clusters at exactly {70.0, 76.0, 79.0, 80.5, ...},
    with a clean real gap between 70.0 (198 of 3941 real players, 5.0%) and
    76.0 (the next tier up, 3228 players) -- SIGNAL_TRUST_DAMPEN_THRESHOLD=76
    isolates exactly that real floor tier. adp_trust_score has its own real
    floor plateau at 50.0 (328 of 3941, 8.3%) -- ADP_TRUST_DAMPEN_THRESHOLD=55
    isolates it. Both single-tier (the old two-severity split had no real
    population left to distinguish between once the circular deductions
    were removed -- inventing a second tier would fit noise, not data).
    Dampening magnitudes (caps/multipliers) softened from the old 60/0.88
    and 62/0.90, since what remains behind these thresholds is now
    genuinely mild real data friction (e.g. one inconsistent ADP field), not
    compounded severe-plus-disagreement cases. Reasoned defaults pending
    their own Phase 2 validation, not final-calibrated -- same standard as
    every other composite this session.
    """
    SIGNAL_TRUST_DAMPEN_THRESHOLD = 76.0
    ADP_TRUST_DAMPEN_THRESHOLD = 55.0
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
    divergence_flags_col = []
    adjusted_scores = []
    adjustment_applied = []
    adjustment_reasons = []

    for _, row in adjusted_df.iterrows():
        player_key = str(row.get("player_name", "")).strip().lower()
        trust_row = trust_lookup.get(player_key, {})
        signal_trust = _safe_float(trust_row.get("signal_trust_score"), 100.0)
        adp_trust = _safe_float(trust_row.get("adp_trust_score"), 100.0)
        flags = trust_row.get("anomaly_flags", [])
        if not isinstance(flags, list):
            flags = []
        divergence = trust_row.get("market_divergence_flags", [])
        if not isinstance(divergence, list):
            divergence = []

        original_score = _safe_float(row.get("final_score"), 0.0)
        adjusted_score = original_score
        reasons = []

        if signal_trust < SIGNAL_TRUST_DAMPEN_THRESHOLD:
            adjusted_score = min(adjusted_score, 65.0)
            adjusted_score *= 0.92
            reasons.append("low signal trust")

        if adp_trust < ADP_TRUST_DAMPEN_THRESHOLD:
            adjusted_score = min(adjusted_score, 70.0)
            adjusted_score *= 0.95
            reasons.append("low ADP trust")

        signal_trust_scores.append(round(signal_trust, 2))
        adp_trust_scores.append(round(adp_trust, 2))
        anomaly_flags.append(flags)
        divergence_flags_col.append(divergence)
        adjusted_scores.append(round(adjusted_score, 2))
        adjustment_applied.append(round(adjusted_score, 2) < round(original_score, 2))
        adjustment_reasons.append(", ".join(reasons))

    adjusted_df["raw_final_score"] = adjusted_df["final_score"]
    adjusted_df["signal_trust_score"] = signal_trust_scores
    adjusted_df["adp_trust_score"] = adp_trust_scores
    adjusted_df["signal_trust_anomaly_flags"] = anomaly_flags
    adjusted_df["market_divergence_flags"] = divergence_flags_col
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
    Explain why a player ranks where he does under Base Value.

    Rewritten 2026-08-21 alongside calculate_base_value_score(): the
    previous version cited trust_adjustment_applied, single_qb_adjustment_
    applied, position_pressure, position_need_component_score,
    urgency_label, and team_fit_component_score -- every one of them from
    the legacy blend chain that final_score no longer reads at all. Left
    as-is, this function would still tell a user "QB score capped due to
    single-QB positional value" or "Fills current TE roster need" on a
    score that structurally cannot reflect either -- the same "unexplained
    rank movement" problem the scoring rewrite fixed, just relocated into
    the explanation text instead of removed. Every reason below now traces
    to one of Base Value's own four components, so the text and the score
    can never disagree.
    """
    reasons = []
    position = str(row.get("position", "")).upper()

    vor_component = _safe_float(row.get("base_value_vor_component"), 0.0)
    proj_component = _safe_float(row.get("base_value_projection_component"), 0.0)
    market_component = _safe_float(row.get("base_value_market_component"), 0.0)
    risk_component = _safe_float(row.get("base_value_risk_component"), 0.0)

    if vor_component >= 40:
        reasons.append(f"Well above {position} replacement level ({vor_component:+.0f} VOR)")
    elif vor_component <= -20:
        reasons.append(f"Below {position} replacement level ({vor_component:+.0f} VOR)")

    if proj_component >= BASE_VALUE_PROJECTION_BONUS_MAX * 0.8:
        reasons.append(f"Top-tier projection within {position}")

    if market_component >= BASE_VALUE_MARKET_MAX_SWING * 0.4:
        reasons.append("Market (ADP) values him higher than the field")
    elif market_component <= -BASE_VALUE_MARKET_MAX_SWING * 0.4:
        reasons.append("Market (ADP) values him lower than the field")

    if risk_component <= -BASE_VALUE_RISK_MAX_PENALTY * 0.6:
        reasons.append("Discounted for real injury/durability risk")

    fall_risk = str(row.get("fall_risk", "")).upper()
    if fall_risk == "HIGH":
        reasons.append("Unlikely to survive until your next pick")
    elif fall_risk == "MEDIUM":
        reasons.append("May not make it back to your next pick")

    if not reasons:
        reasons.append("Base Value: VOR + within-position projection, ADP and risk as small adjustments")

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

# Phase 1b (plan_deanchor_scoring_from_adp.pdf): confidence-conditional
# anchor lambda -- a genuinely different experiment from the flat lambda
# sweep optimize_blend_weights_v1.py already ran and found near-optimal at
# a UNIFORM 0.20 (see that constant's own docstring). This only flexes lam
# for players the CORRECTED (post Phase 1 circularity fix) signal_trust_score
# says are genuinely well-supported or genuinely thin -- everyone else stays
# at the already-validated 0.20 baseline.
#
# Real, reachable thresholds (learned from Phase 1's own near-miss: a first
# draft of that fix's dampening thresholds turned out unreachable against
# the corrected score distribution, caught only by re-measuring). Checked
# directly against the real, post-Phase-1 signal_trust_score distribution
# (3941 real players): the floor tier (<76) holds 198 players (5.0%,
# reuses the exact same real cut SIGNAL_TRUST_DAMPEN_THRESHOLD already
# validates -- these are the SAME genuinely-thin-data players, now also
# anchored tighter, not just dampened); the real elite/clean tier (>=85)
# holds 342 players (8.7%), comfortably including the real elite-QB cluster
# (Burrow/Lawrence/Mahomes/Love, all at the real 89.5 tier) with margin to
# spare -- not razor-fit to catch them specifically.
#
# Reasoned defaults pending Phase 2's own validation, NOT final-calibrated
# -- explicit exit clause: if Phase 2 doesn't show a real improvement for
# the players this is meant to help, this phase gets dropped and Phase 1
# alone ships (see draftkit/tests/test_adp_deanchor.py for the real
# validation this claim is checked against, not just asserted).
#
# EXIT CLAUSE EXERCISED, THEN RE-ENABLED (2026-08-17). Original hold: Phase
# 2 validation found the near-zero QB `projection_component_score` pushed
# real, meaningfully-drafted elite QBs (Burrow, Herbert, C.Williams,
# Prescott) 13-15 rank spots WORSE under a higher lam (0.35), and it was
# unclear whether that near-zero component was a genuine 1-QB-scarcity
# signal or a scaling bug -- amplifying an unverified number is worse than
# leaving it alone, so this was held off.
#
# That question is now resolved (task_4d5e2bb0 follow-up investigation):
# the near-zero QB component was a real, confirmed formula bug in
# build_tier_calibration(), NOT a genuine scarcity signal and NOT the
# replacement-baseline math (that part was independently verified sound
# against a real external QB12/13 benchmark, ~289.7 pts, and the engine's
# own 300.2 baseline sits above it). build_tier_calibration() divided
# realized_VOR by each position's own CURRENT projected_VOR, importing a
# cross-position VOR-scale difference the source backtest script's own
# docstring explicitly warns against as "exactly the mistake the old TE
# 0.68 multiplier made." Fixed to the documented formula (realized_VOR(pos,
# tier) / realized_VOR(pos, "1-3"), a pure within-position shape ratio) --
# confirmed this collapses every position's premium-tier factor to exactly
# 1.0 by construction, moving Josh Allen from a QB discounted to 23% of his
# raw VOR to his correct, undiscounted value (position_value_score
# 16.70 -> 73.00, live-verified: overall rank moves to #30). With that fixed,
# the ground Phase 1b's amplification stands on is solid, so
# use_confidence_conditional now defaults to True.
#
# This does NOT mean the 0.35/0.20/0.10 lambda values themselves are
# proven correct -- they remain a reasoned default carrying their own,
# still-live exit clause: if a fresh Phase 2 validation pass (now run
# against real, non-discounted QB scores) doesn't show a real improvement
# for the players this is meant to help, this phase gets dropped again and
# Phase 1 alone ships. Resolving "was the ground solid" is a different
# question from "are these specific numbers right," and only the first one
# is settled here.
PHASE_1B_LOW_TRUST_THRESHOLD = 76.0
PHASE_1B_LOW_TRUST_LAMBDA = 0.10
PHASE_1B_HIGH_TRUST_THRESHOLD = 85.0
PHASE_1B_HIGH_TRUST_LAMBDA = 0.35


def compute_confidence_conditional_lambda(signal_trust_scores, base_lam=ADP_ANCHOR_LAMBDA):
    """Per-row anchor lambda, real signal_trust_score-conditional (Phase 1b).
    See PHASE_1B_LOW_TRUST_THRESHOLD's docstring for the real thresholds and
    why they're calibrated the way they are. Missing signal_trust_score
    (e.g. a player signal_trust.py couldn't resolve) falls back to base_lam
    -- absence of a trust read is not evidence of either high or low trust."""
    scores = pd.to_numeric(signal_trust_scores, errors="coerce")
    lam = pd.Series(base_lam, index=scores.index, dtype=float)
    lam.loc[scores < PHASE_1B_LOW_TRUST_THRESHOLD] = PHASE_1B_LOW_TRUST_LAMBDA
    lam.loc[scores >= PHASE_1B_HIGH_TRUST_THRESHOLD] = PHASE_1B_HIGH_TRUST_LAMBDA
    return lam


def apply_adp_anchor(recommendations_df, lam=ADP_ANCHOR_LAMBDA, use_confidence_conditional=True):
    """
    Blend the model score toward ADP on a common 0-1 scale.

        blended  = (1 - lam) * adp_percentile + lam * model_normalized
        anchored = lo + minmax(blended) * (hi - lo)

    ADP contributes as a percentile (rank-based, deliberate -- raw ADP is
    not a meaningful interval scale). The MODEL side contributes as a
    magnitude-preserving log-min-max normalization of pre_anchor_score, not
    a percentile -- see the inline notes below for the real measured
    distributions behind both choices, and for the compression bug this
    replaced (both the model input and the final output were previously
    rank-transformed, which forced perfectly uniform score spacing across
    the entire board and discarded every real quality difference the
    scoring pipeline had computed).

    Players with no ADP were never priced by the market, so there is no
    consensus to anchor to -- they keep their model percentile scaled into
    the range below the last ADP-having player rather than being dropped or
    treated as ADP-worst.

    `lam` becomes PER-ROW when `use_confidence_conditional=True` (the
    default) and a real `signal_trust_score` column is present -- see
    compute_confidence_conditional_lambda() / PHASE_1B_LOW_TRUST_THRESHOLD's
    docstring for the real history: this was held off after Phase 2
    validation found amplifying a near-zero QB projection component pushed
    real elite QBs worse, then re-enabled once that near-zero component was
    traced to a confirmed formula bug in build_tier_calibration() (fixed
    separately) rather than a genuine signal. Still carries its own,
    separate exit clause on the 0.35/0.20/0.10 lambda values themselves --
    see that docstring.
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

    if use_confidence_conditional and "signal_trust_score" in out.columns:
        lam_series = compute_confidence_conditional_lambda(out.loc[has_adp, "signal_trust_score"], base_lam=lam)
    else:
        lam_series = pd.Series(lam, index=out.index[has_adp], dtype=float)

    # Real gap found and fixed (2026-08-21): signal_trust_score is a GENERIC
    # data-quality read (messy/thin/inconsistent inputs) -- it has no way to
    # tell that apart from a player who simply has a real, dated, human-
    # verified correction the market hasn't priced in yet (model_
    # override_applied, see apply_model_projection_override() above).
    # First attempt floored these rows' lambda at PHASE_1B_HIGH_TRUST_LAMBDA
    # (0.35) -- measured directly on Jordyn Tyson and found to be a no-op:
    # his signal_trust_score (95.5) already sits in the automated top tier,
    # already giving lam=0.35, and even that "most-trusted" tier still
    # blends in 65% ADP weight -- enough to take his real, correctly-
    # damped pre-anchor score (31.21, appropriately reflecting a real
    # ~2-month injury) most of the way back up to 59.68, nearly doubling
    # it. A partial discount isn't enough here. These rows are human-
    # verified, dated, sourced corrections -- a stronger signal than the
    # automated trust metric that already maxed out, and the entire point
    # of doing that research is defeated if it gets diluted back toward a
    # stale market number. Fully exempted from the ADP blend (lam=1.0,
    # pure model score) rather than partially discounted.
    if "model_override_applied" in out.columns:
        overridden = out.loc[has_adp, "model_override_applied"].fillna(False).astype(bool)
        lam_series = lam_series.where(~overridden, 1.0)

    # Lower ADP is better, so negate before ranking to percentile. ADP stays
    # RANK-based deliberately: raw ADP values are not a meaningful interval
    # scale (the real distance between ADP 1.1 and 2.0 is not comparable to
    # the distance between 150 and 151), so ordinal treatment is correct
    # for this input specifically.
    adp_pct = (-adp[has_adp]).rank(pct=True)

    # Magnitude-preserving model normalization (final_score compression fix,
    # 2026-08-17). This was `model[has_adp].rank(pct=True)` -- which threw
    # away every real magnitude difference the scoring pipeline had just
    # computed, BEFORE the blend even happened. Unlike ADP, pre_anchor_score
    # IS a real interval scale (it is a weighted composite of normalized
    # 0-100 components), so collapsing it to pure rank discarded genuine
    # signal: measured on the real board, pre_anchor_score spans 18.71-84.53
    # with std 11.9 and real, differentiated gaps, while the rank transform
    # reduced every one of those to an identical 1/N step.
    #
    # Log-scale rather than plain min-max, chosen from the real measured
    # distribution rather than by default: pre_anchor_score's real skew is
    # +2.37 (p25=25.25, p50=28.89, p75=34.55, p99=77.09) -- a long right
    # tail with the population bunched near the bottom. Plain min-max
    # against a range dominated by a handful of elite outliers would
    # compress that bunched majority into a sliver, which is exactly the
    # bug already found and fixed in position_value_score's own
    # normalization earlier today. Same epsilon-guarded treatment reused.
    log_model = np.log(model[has_adp].clip(lower=0.0) + _LOG_TRANSFORM_EPSILON)
    log_lo, log_hi = float(log_model.min()), float(log_model.max())
    if log_hi > log_lo:
        model_pct = (log_model - log_lo) / (log_hi - log_lo)
    else:
        model_pct = pd.Series(0.5, index=log_model.index, dtype=float)

    blended = (1.0 - lam_series) * adp_pct + lam_series * model_pct

    anchored = pd.Series(np.nan, index=out.index, dtype=float)
    # Rescale to the original score range so downstream display/thresholds
    # keep working on a familiar 0-100-ish scale.
    lo, hi = float(model[has_adp].min()), float(model[has_adp].max())
    span = hi - lo
    if span <= 0:
        return recommendations_df

    # Direct min-max rescale of blended's OWN values (final_score
    # compression fix, 2026-08-17). This was `blended.rank(pct=True) * span`
    # -- and rank(pct=True) on N distinct values always returns exactly
    # {1/N, 2/N, ... 1}, so no matter what blended actually looked like
    # going in, the output was forced into perfectly uniform steps. That one
    # line is what produced the measured 0.0975-points-per-rank spacing
    # across the entire real board: rank 1 (84.53) to rank 250 (60.36) is a
    # 24.17-point span over 249 steps = 0.0971/rank, matching a pure
    # rank-linear sequence almost exactly. Every real quality difference
    # between adjacent players was erased at the final step.
    #
    # blended carries real structure worth preserving -- measured on the
    # real board: std 0.2603, skew 0.464, and consecutive-gap std/mean of
    # 2.91 (a perfectly uniform sequence would be 0.0). Its mild skew is
    # why a straight min-max is right HERE, where a log transform was right
    # for the strongly-skewed model input above -- each transform picked
    # from that input's own real measured distribution, not applied blanket.
    blended_lo, blended_hi = float(blended.min()), float(blended.max())
    if blended_hi > blended_lo:
        anchored.loc[has_adp] = lo + ((blended - blended_lo) / (blended_hi - blended_lo)) * span
    else:
        anchored.loc[has_adp] = lo + 0.5 * span

    # Unpriced players sit below every priced one, ordered among themselves
    # by model score -- they are speculative depth, not consensus values.
    no_adp = ~has_adp & model.notna()
    if no_adp.any():
        anchored.loc[no_adp] = lo - 1.0 + model[no_adp].rank(pct=True)

    out["final_score"] = anchored.round(2).fillna(out["final_score"])
    return out


def _attach_sleeper_value_gap(master_df):
    """_build_base_recommendation_rankings_df() builds its return frame from
    an explicit dict of named fields (see its `rows.append({...})`), so any
    master_players.csv column not in that fixed list -- including
    sleeper_value_gap, added by apply_sleeper_adp_overlay.py -- never
    survives into base_df even though it's sitting right there in the CSV.
    `adp` itself is explicitly listed there so it DOES flow through; only
    this derived column needs pulling back in, same pattern as
    _attach_breakout_tags() above."""
    if "player_name" not in master_df.columns:
        return master_df
    if not MASTER_PLAYERS_PATH.exists():
        return master_df
    master = pd.read_csv(MASTER_PLAYERS_PATH, low_memory=False)
    if "sleeper_value_gap" not in master.columns:
        return master_df
    gap = master[["player_name", "sleeper_value_gap"]].drop_duplicates("player_name")
    return master_df.merge(gap, on="player_name", how="left")


MASTER_PLAYERS_PATH = Path("data/processed/master_players.csv")
BREAKOUT_TAGS_PATH = Path("data/processed/breakout_tags_2026.csv")


def _attach_breakout_tags(master_df):
    """Merge in the WR breakout_score_v1 tag (research/MODEL_REGISTRY.md --
    RESEARCH_ONLY, not a scoring input) for the display badge only. Written
    by research/validation_v1/score_current_wr_pool_v1.py: top 15-20 WRs by
    a validated, backtested (not yet season-proven) breakout-probability
    model, cut at a natural gap rather than a fixed count. Deliberately does
    NOT touch final_score, tier, or any ranking -- badge-only, same as the
    rookie_display_tag pattern elsewhere in this pipeline."""
    master_df["is_breakout_v1"] = False
    master_df["breakout_probability_v1"] = pd.NA
    if not BREAKOUT_TAGS_PATH.exists():
        return master_df
    tags = pd.read_csv(BREAKOUT_TAGS_PATH)
    if tags.empty or "player_name" not in tags.columns:
        return master_df
    tags = tags[["player_name", "breakout_probability_v1", "breakout_rank"]].drop_duplicates("player_name")
    master_df = master_df.merge(
        tags, on="player_name", how="left", suffixes=("", "_tag"),
    )
    if "breakout_probability_v1_tag" in master_df.columns:
        master_df["breakout_probability_v1"] = master_df["breakout_probability_v1_tag"]
        master_df = master_df.drop(columns=["breakout_probability_v1_tag"])
    master_df["is_breakout_v1"] = master_df["breakout_rank"].notna()
    return master_df


def build_master_recommendations_df(component_weights=None, drop_threshold=12.0):
    """
    Build transparent master recommendations from all existing scoring systems.
    """
    get_current_pick_number()
    base_df = _build_base_recommendation_rankings_df()
    if base_df.empty:
        return pd.DataFrame()

    master_df = base_df.copy()
    master_df = _attach_breakout_tags(master_df)
    master_df = _attach_sleeper_value_gap(master_df)

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

    # Real gap found and fixed (2026-08-21): the position-only lookup below
    # used to key on `str(row["Position"]).upper()` alone, but urgency_df
    # (build_position_urgency_df()) only ever carried each position's
    # CURRENT tier -- so every player at a position, regardless of his own
    # real tier, got that single tier's urgency broadcast to him (verified
    # live: every RB row read raw_urgency_score=70.5, every WR row read
    # 60.5, uniformly). calculate_position_urgency() itself is a real,
    # working formula (verified by hand -- it reproduces 70.5/60.5 exactly
    # from players_left/tier_dropoff/need_weight), so the fix is reach, not
    # the formula: build_full_tier_urgency_df() computes it for EVERY
    # tier of every position, and the lookup below is now keyed on
    # (Position, Tier) using each player's own real position_tier (see
    # _build_base_recommendation_rankings_df()) rather than position alone.
    full_urgency_df = build_full_tier_urgency_df(drop_threshold=drop_threshold)
    if not full_urgency_df.empty:
        urgency_lookup = {
            (str(row["Position"]).upper(), int(row["Tier"])): row.to_dict()
            for _, row in full_urgency_df.iterrows()
        }
        # Fallback for a player whose own tier didn't resolve (position_tier
        # is None) or whose specific tier has no urgency row for some other
        # reason -- his position's OWN lowest/current tier, the same value
        # every player at that position used to get, rather than a silent
        # zero.
        position_fallback = {
            str(row["Position"]).upper(): row.to_dict()
            for _, row in full_urgency_df.sort_values("Tier").groupby("Position").first().reset_index().iterrows()
        }
    else:
        urgency_lookup = {}
        position_fallback = {}

    def _urgency_row(row):
        pos = str(row.get("position") or "").upper()
        tier = row.get("position_tier")
        key = (pos, int(tier)) if tier is not None and pd.notna(tier) else None
        return (urgency_lookup.get(key) if key is not None else None) or position_fallback.get(pos, {})

    master_df["raw_urgency_score"] = master_df.apply(
        lambda row: _safe_float(_urgency_row(row).get("Urgency Score"), 0.0), axis=1
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
    master_df["urgency_bonus"] = master_df.apply(
        lambda row: _safe_float(_urgency_row(row).get("Urgency Bonus"), 1.0), axis=1
    )
    master_df["urgency_label"] = master_df.apply(
        lambda row: _urgency_row(row).get("Urgency Label", "Low"), axis=1
    )
    master_df["tier_dropoff"] = master_df.apply(
        lambda row: _safe_float(_urgency_row(row).get("Tier Dropoff"), 0.0), axis=1
    )
    master_df["players_left_in_tier"] = master_df.apply(
        lambda row: int(_safe_float(_urgency_row(row).get("Players Left In Tier"), 0.0)), axis=1
    )

    master_df = normalize_component_scores(master_df)
    weights = _resolve_component_weights(component_weights)
    # Legacy per-row blend -- kept computing so its component-score columns
    # (projection_component_score, position_need_component_score, etc.)
    # keep existing behind-the-scenes readers (recommendation_explainer.py,
    # other pages) working, and so this session's git history stays legible.
    # No longer what final_score is set to: see the Base Value block above
    # calculate_base_value_score() for why. Stored under a distinct name so
    # nothing downstream can mistake it for the real score.
    master_df["legacy_blend_score"] = master_df.apply(
        lambda row: calculate_final_recommendation_score(row, weights),
        axis=1,
    )
    # The four calls below each read/write a column literally named
    # final_score internally (construction pressure logs
    # pre_construction_score off it, the ADP anchor blends against it,
    # etc.) -- give them the legacy blend to operate on so they don't
    # raise, fully aware every one of their writes gets discarded by the
    # unconditional overwrite a few lines down.
    master_df["final_score"] = master_df["legacy_blend_score"]
    master_df = apply_construction_pressure_adjustments(master_df)
    master_df = apply_signal_trust_adjustments(master_df)
    master_df = apply_single_qb_value_adjustments(master_df)
    master_df = apply_adp_anchor(master_df)

    # Base Value (2026-08-21) is the entire player-value score now, at every
    # roster state -- see calculate_base_value_score()'s own docstring.
    # Applied LAST and unconditionally so nothing above (construction
    # pressure, signal-trust damping, the single-QB multiplier, the
    # ADP-anchor blend) can leave a trace in what actually gets sorted:
    # those calls above still run only to populate their own display/
    # backward-compat columns (pre_anchor_score, signal_trust_score,
    # single_qb_adjustment_applied, construction pressure fields), each of
    # which independently touched final_score before today and is exactly
    # the "each layer can alter the ranking, layers aren't measuring the
    # same thing" problem this replaces.
    master_df = calculate_base_value_score(master_df)
    master_df["final_score"] = master_df["base_value_score"]

    master_df["recommendation_reasons"] = master_df.apply(
        lambda row: generate_recommendation_reasons(row),
        axis=1,
    )

    # Tiebreakers changed to Base Value's own components (2026-08-21): the
    # legacy component scores below no longer feed final_score at all, so
    # using them to break ties would let them influence rank through the
    # back door -- exactly the hidden-layer problem this rewrite removes.
    sort_cols = [
        "final_score",
        "base_value_vor_component",
        "base_value_projection_component",
        "base_value_market_component",
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


def build_tier_summary_df(drop_threshold=12.0, tiers_df=None):
    """`tiers_df`: pass an already-built frame (same drop_threshold) to skip
    rebuilding it -- perf fix (2026-08-27), same load-time investigation as
    calculate_tier_bonus's lookup cache. The hot loop in
    _build_base_recommendation_rankings_df() was calling
    build_position_tiers_df() once directly AND again implicitly through
    this function's own default, profiled at ~2.5s per real rebuild.
    Every other caller is unaffected -- the default (None) still rebuilds
    exactly as before."""
    if tiers_df is None:
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


def build_full_tier_urgency_df(drop_threshold=12.0):
    """Real per-(position, tier) urgency, for every tier -- not just the
    current one build_position_urgency_df()/build_tier_summary_df() are
    scoped to (2026-08-21 audit finding: real, verified -- calculate_
    position_urgency() itself is a genuine dynamic formula, reproduced by
    hand from players_left/tier_dropoff/need_weight, NOT a hardcoded
    per-position constant. The real gap is narrower: build_tier_summary_df()
    only ever computes `first_tier = pos_df["Tier"].min()`, so that real
    formula only ever ran on each position's CURRENT tier, and every player
    at that position -- tier 1 or tier 8 -- got broadcast the identical
    result via a position-only lookup in _build_base_recommendation_
    rankings_df(). Confirmed live: every RB row read raw_urgency_score=70.5,
    every WR row read 60.5, regardless of the player's own real tier.

    Deliberately a NEW function, not a modification of build_position_
    urgency_df()/build_tier_summary_df() -- those have real other callers
    (opponent_model.py, recommendation_explainer.py, pages_archive/
    8_Tier_Desperation.py) that want "the single current tier cliff right
    now," a genuinely different, valid question from "what tier is THIS
    specific player actually in." Consumed by _build_base_recommendation_
    rankings_df() below, keyed on (Position, Tier) so each player's row can
    look up its OWN tier's real urgency rather than its position's current
    one."""
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    if tiers_df.empty:
        return pd.DataFrame()

    next_pick_distance = get_next_pick_distance()
    if next_pick_distance is None:
        next_pick_distance = 3

    need_weights = get_position_need_weights()
    rows = []

    for position in ["QB", "RB", "WR", "TE"]:
        pos_df = tiers_df[tiers_df["Position"] == position]
        if pos_df.empty:
            continue
        for tier in sorted(pos_df["Tier"].unique()):
            players_left = int((pos_df["Tier"] == tier).sum())
            if players_left <= 2:
                tier_status = "Thin"
            elif players_left <= 4:
                tier_status = "Shrinking"
            else:
                tier_status = "Healthy"
            tier_dropoff = calculate_tier_dropoff(
                position, tier, tiers_df=tiers_df, drop_threshold=drop_threshold
            )
            urgency = calculate_position_urgency(
                position=position,
                players_left=players_left,
                tier_dropoff=tier_dropoff,
                next_pick_distance=next_pick_distance,
                need_weight=need_weights.get(position, 1.0),
            )
            urgency["Tier"] = int(tier)
            urgency["Tier Status"] = tier_status
            rows.append(urgency)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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
