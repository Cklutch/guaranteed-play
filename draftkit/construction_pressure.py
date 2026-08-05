import streamlit as st

from draftkit.common import safe_float as _safe_float
from draftkit.common import safe_int as _safe_int
from draftkit.data_access import load_players_df
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_team_positions,
    init_session_state,
)


POSITIONS = ["QB", "RB", "WR", "TE"]
PRESSURE_LEVELS = ["NONE", "LOW", "MODERATE", "HIGH", "SEVERE"]


def _current_round(current_pick=None, league_size=None):
    pick = max(_safe_int(current_pick, get_current_pick_number()), 1)
    size = max(_safe_int(league_size, get_league_size()), 1)
    return ((pick - 1) // size) + 1


def _round_pressure_multiplier(round_number):
    if round_number <= 2:
        return 0.35
    if round_number <= 5:
        return 0.75
    if round_number <= 8:
        return 1.00
    return 1.15


def _pressure_level(score):
    score = _safe_float(score, 0.0)
    if score < 10:
        return "NONE"
    if score < 30:
        return "LOW"
    if score < 55:
        return "MODERATE"
    if score < 80:
        return "HIGH"
    return "SEVERE"


def _roster_size(position_counts):
    return sum(_safe_int(position_counts.get(position), 0) for position in POSITIONS)


def _starter_targets(roster_settings):
    return {
        position: _safe_int(roster_settings.get(position), 0)
        for position in POSITIONS
    }


def _roster_completeness(position_counts, roster_settings):
    targets = _starter_targets(roster_settings)
    total_required = sum(targets.values())
    if total_required <= 0:
        return 100.0

    filled = sum(
        min(_safe_int(position_counts.get(position), 0), target)
        for position, target in targets.items()
    )
    return round((filled / total_required) * 100.0, 2)


def calculate_position_pressure(
    position,
    position_counts=None,
    roster_settings=None,
    current_pick=None,
    league_size=None,
):
    init_session_state()
    position = str(position).upper()
    roster_settings = roster_settings or st.session_state.get("roster_settings", {})
    if position_counts is None:
        position_counts = get_my_team_positions(load_players_df())

    round_number = _current_round(current_pick, league_size)
    round_multiplier = _round_pressure_multiplier(round_number)
    target = _safe_int(roster_settings.get(position), 0)
    current = _safe_int(position_counts.get(position), 0)
    missing = max(target - current, 0)
    overfilled = max(current - target, 0)

    raw_score = 0.0
    if target > 0 and missing > 0:
        raw_score = 35.0 + (missing * 13.0)
        if current == 0:
            raw_score += 12.0

    rb_count = _safe_int(position_counts.get("RB"), 0)
    wr_count = _safe_int(position_counts.get("WR"), 0)
    rb_target = _safe_int(roster_settings.get("RB"), 0)
    wr_target = _safe_int(roster_settings.get("WR"), 0)

    if position == "RB" and rb_target > 0 and rb_count == 0:
        if wr_count >= 5:
            raw_score += 45.0
        elif wr_count >= 4:
            raw_score += 25.0
        elif wr_count >= 3:
            raw_score += 20.0

    if position == "WR" and wr_target > 0 and wr_count == 0:
        if rb_count >= 5:
            raw_score += 45.0
        elif rb_count >= 4:
            raw_score += 25.0
        elif rb_count >= 3:
            raw_score += 20.0

    if position in ["RB", "WR"] and missing > 0:
        flex_slots = _safe_int(roster_settings.get("FLEX"), 0)
        raw_score += min(12.0, flex_slots * 6.0)

    pressure_score = round(max(0.0, min(100.0, raw_score * round_multiplier)), 2)
    level = _pressure_level(pressure_score)

    return {
        "position": position,
        "pressure_score": pressure_score,
        "pressure_level": level,
        "current_count": current,
        "target_count": target,
        "missing_count": missing,
        "overfilled_count": overfilled,
        "round": round_number,
        "round_multiplier": round_multiplier,
    }


def calculate_construction_pressure(
    position_counts=None,
    roster_settings=None,
    current_pick=None,
    league_size=None,
):
    init_session_state()
    roster_settings = roster_settings or st.session_state.get("roster_settings", {})
    if position_counts is None:
        position_counts = get_my_team_positions(load_players_df())

    position_pressure = {
        position: calculate_position_pressure(
            position,
            position_counts=position_counts,
            roster_settings=roster_settings,
            current_pick=current_pick,
            league_size=league_size,
        )
        for position in POSITIONS
    }
    critical_needs = [
        position
        for position, pressure in position_pressure.items()
        if pressure["pressure_level"] in ["HIGH", "SEVERE"]
    ]
    total_players = _roster_size(position_counts)
    bench_target = _safe_int(roster_settings.get("BENCH"), 0)
    starter_targets = _starter_targets(roster_settings)
    total_starter_targets = sum(starter_targets.values())
    total_roster_target = total_starter_targets + bench_target

    return {
        "position_pressure": position_pressure,
        "critical_needs": critical_needs,
        "roster_completeness": _roster_completeness(position_counts, roster_settings),
        "total_players": total_players,
        "bench_pressure": max(total_roster_target - total_players, 0),
        "flex_slots": _safe_int(roster_settings.get("FLEX"), 0),
        "round": _current_round(current_pick, league_size),
    }


def calculate_construction_adjustment(position, construction_pressure=None):
    position = str(position).upper()
    construction_pressure = construction_pressure or calculate_construction_pressure()
    pressure = construction_pressure.get("position_pressure", {}).get(position, {})
    level = pressure.get("pressure_level", "NONE")

    modifiers = {
        "NONE": 1.00,
        "LOW": 1.02,
        "MODERATE": 1.06,
        "HIGH": 1.12,
        "SEVERE": 1.18,
    }
    return round(modifiers.get(level, 1.00), 3)


def get_construction_pressure_debug_info(
    position_counts=None,
    roster_settings=None,
    current_pick=None,
    league_size=None,
):
    pressure = calculate_construction_pressure(
        position_counts=position_counts,
        roster_settings=roster_settings,
        current_pick=current_pick,
        league_size=league_size,
    )
    return {
        "round": pressure["round"],
        "position_pressure": pressure["position_pressure"],
        "critical_needs": pressure["critical_needs"],
        "roster_completeness": pressure["roster_completeness"],
        "adjustments": {
            position: calculate_construction_adjustment(position, pressure)
            for position in POSITIONS
        },
    }
