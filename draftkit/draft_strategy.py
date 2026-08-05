from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from draftkit.common import name_key as _name_key
from draftkit.common import normalize_player_name as _normalize_name
from draftkit.common import safe_float as _safe_float
from draftkit.common import safe_int as _safe_int
from draftkit.construction_pressure import calculate_construction_pressure
from draftkit.data_access import load_players_df
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_team_positions,
    init_session_state,
)


SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
STRATEGY_TYPES = [
    "Balanced",
    "Hero RB",
    "Hero WR",
    "Double RB",
    "Double WR",
    "Elite QB",
    "Late QB",
    "Elite TE",
    "Late TE",
    "Best Player Available",
]


def _current_round(current_pick: Optional[int] = None, league_size: Optional[int] = None) -> int:
    pick = max(_safe_int(current_pick, get_current_pick_number()), 1)
    size = max(_safe_int(league_size, get_league_size()), 1)
    return ((pick - 1) // size) + 1


def _position_counts_from_state() -> Dict[str, int]:
    players_df = load_players_df()
    counts = get_my_team_positions(players_df)
    return {position: _safe_int(counts.get(position), 0) for position in SCORABLE_POSITIONS}


def _roster_size(position_counts: Dict[str, int]) -> int:
    return sum(_safe_int(position_counts.get(position), 0) for position in SCORABLE_POSITIONS)


def _recommendation_records(recommendations_df: Optional[pd.DataFrame], limit: int = 3) -> List[Dict[str, Any]]:
    if recommendations_df is None:
        recommendations_df = build_recommendation_rankings_df()

    if recommendations_df is None or recommendations_df.empty:
        return []

    rows = []
    for _, row in recommendations_df.head(limit).iterrows():
        rows.append({
            "player_name": _normalize_name(row.get("player_name")),
            "position": _normalize_name(row.get("position")),
            "team": _normalize_name(row.get("team")),
            "recommendation_score": round(_safe_float(row.get("final_score")), 2),
            "position_pressure": row.get("position_pressure"),
            "construction_pressure_score": row.get("construction_pressure_score"),
            "roster_completeness": row.get("roster_completeness"),
        })
    return rows


def identify_current_strategy(
    position_counts: Optional[Dict[str, int]] = None,
    current_pick: Optional[int] = None,
    league_size: Optional[int] = None,
) -> str:
    """
    Classify the user's current roster direction without changing recommendations.
    """
    init_session_state()
    position_counts = position_counts or _position_counts_from_state()

    roster_size = _roster_size(position_counts)
    round_number = _current_round(current_pick, league_size)
    qb_count = _safe_int(position_counts.get("QB"))
    rb_count = _safe_int(position_counts.get("RB"))
    wr_count = _safe_int(position_counts.get("WR"))
    te_count = _safe_int(position_counts.get("TE"))

    if roster_size <= 0:
        return "Best Player Available"

    if qb_count >= 1 and roster_size <= 3:
        return "Elite QB"
    if te_count >= 1 and roster_size <= 3:
        return "Elite TE"
    if qb_count == 0 and round_number >= 8:
        return "Late QB"
    if te_count == 0 and round_number >= 8:
        return "Late TE"
    if rb_count == 1 and wr_count >= 2:
        return "Hero RB"
    if wr_count == 1 and rb_count >= 2:
        return "Hero WR"
    if rb_count >= 2 and wr_count == 0:
        return "Double RB"
    if wr_count >= 2 and rb_count == 0:
        return "Double WR"
    if rb_count >= 1 and wr_count >= 1:
        return "Balanced"

    return "Best Player Available"


def identify_roster_path(
    position_counts: Optional[Dict[str, int]] = None,
    construction_pressure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Explain the roster build path and the positions that most affect the next turn.
    """
    init_session_state()
    position_counts = position_counts or _position_counts_from_state()
    construction_pressure = construction_pressure or calculate_construction_pressure(
        position_counts=position_counts,
    )

    pressure_by_position = construction_pressure.get("position_pressure", {})
    critical_needs = list(construction_pressure.get("critical_needs", []))
    next_positions = sorted(
        SCORABLE_POSITIONS,
        key=lambda position: _safe_float(
            pressure_by_position.get(position, {}).get("pressure_score"),
            0.0,
        ),
        reverse=True,
    )
    next_positions = [
        position
        for position in next_positions
        if pressure_by_position.get(position, {}).get("pressure_level") != "NONE"
    ]

    strategy = identify_current_strategy(position_counts)
    if critical_needs:
        summary = "Prioritize " + ", ".join(critical_needs) + " to stabilize construction."
    elif next_positions:
        summary = "Stay flexible while monitoring " + ", ".join(next_positions[:2]) + "."
    elif strategy == "Best Player Available":
        summary = "No roster direction has been established yet."
    else:
        summary = f"{strategy} path is structurally on track."

    return {
        "path_name": strategy,
        "summary": summary,
        "position_counts": position_counts,
        "critical_needs": critical_needs,
        "next_positions": next_positions[:3],
        "roster_completeness": construction_pressure.get("roster_completeness", 0.0),
        "round": construction_pressure.get("round"),
    }


def track_passed_targets(
    recommendation_history: Optional[List[Dict[str, Any]]] = None,
    drafted_players: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Track recommendations the user passed on and whether those targets are still alive.
    """
    init_session_state()
    history = recommendation_history
    if history is None:
        history = st.session_state.get("recommendation_history", [])
    drafted = {_name_key(player) for player in (drafted_players or st.session_state.get("drafted_players", []))}

    passed_targets = []
    for entry in history:
        recommended_player = _normalize_name(entry.get("recommended_player"))
        selected_player = _normalize_name(entry.get("selected_player"))
        if not recommended_player or not selected_player:
            continue
        if _name_key(recommended_player) == _name_key(selected_player):
            continue

        target_lost = _name_key(recommended_player) in drafted
        passed_targets.append({
            "pick_number": entry.get("pick_number"),
            "recommended_player": recommended_player,
            "player_drafted_instead": selected_player,
            "recommendation_score": entry.get("recommendation_score"),
            "target_survived": not target_lost,
            "target_lost": target_lost,
        })

    return passed_targets


def track_surviving_targets(
    recommendation_history: Optional[List[Dict[str, Any]]] = None,
    drafted_players: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return previously passed recommendation targets that are still available.
    """
    surviving = [
        target
        for target in track_passed_targets(recommendation_history, drafted_players)
        if target.get("target_survived")
    ]
    return surviving


def build_next_pick_plan(
    recommendations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Use the current recommendation output as the target board for the next pick plan.
    """
    recommendations = _recommendation_records(recommendations_df, limit=3)

    primary = recommendations[0] if len(recommendations) >= 1 else None
    secondary = recommendations[1] if len(recommendations) >= 2 else None
    contingency = recommendations[2] if len(recommendations) >= 3 else None

    return {
        "primary_target": primary,
        "secondary_target": secondary,
        "contingency_target": contingency,
        "target_count": len(recommendations),
        "plan_summary": _build_plan_summary(primary, secondary, contingency),
    }


def _build_plan_summary(
    primary: Optional[Dict[str, Any]],
    secondary: Optional[Dict[str, Any]],
    contingency: Optional[Dict[str, Any]],
) -> str:
    if not primary:
        return "No next-pick target is available yet."

    if secondary and contingency:
        return (
            f"Target {primary['player_name']}; pivot to {secondary['player_name']} "
            f"or {contingency['player_name']} if the board breaks against you."
        )
    if secondary:
        return f"Target {primary['player_name']}; backup plan is {secondary['player_name']}."
    return f"Target {primary['player_name']}."


def calculate_strategy_confidence(
    current_strategy: Optional[str] = None,
    roster_path: Optional[Dict[str, Any]] = None,
    passed_targets: Optional[List[Dict[str, Any]]] = None,
    construction_pressure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate confidence in the current draft path.
    """
    init_session_state()
    construction_pressure = construction_pressure or calculate_construction_pressure()
    roster_path = roster_path or identify_roster_path(construction_pressure=construction_pressure)
    current_strategy = current_strategy or roster_path.get("path_name", "Best Player Available")
    passed_targets = passed_targets if passed_targets is not None else track_passed_targets()

    score = 55.0
    roster_completeness = _safe_float(roster_path.get("roster_completeness"), 0.0)
    critical_needs = roster_path.get("critical_needs", [])
    lost_targets = [target for target in passed_targets if target.get("target_lost")]
    surviving_targets = [target for target in passed_targets if target.get("target_survived")]

    if current_strategy in ["Balanced", "Double RB", "Double WR", "Hero RB", "Hero WR"]:
        score += 10.0
    if current_strategy in ["Best Player Available"]:
        score -= 5.0
    if roster_completeness >= 70:
        score += 12.0
    elif roster_completeness >= 45:
        score += 5.0
    elif roster_completeness < 25:
        score -= 8.0

    score -= min(len(critical_needs) * 10.0, 25.0)
    score -= min(len(lost_targets) * 8.0, 20.0)
    score += min(len(surviving_targets) * 3.0, 9.0)

    score = round(max(0.0, min(100.0, score)), 2)
    if score >= 75:
        level = "HIGH"
    elif score >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "strategy_confidence": level,
        "strategy_confidence_score": score,
        "drivers": {
            "roster_completeness": roster_completeness,
            "critical_need_count": len(critical_needs),
            "lost_target_count": len(lost_targets),
            "surviving_target_count": len(surviving_targets),
        },
    }


def build_draft_strategy(
    recommendations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build the current draft strategy plan from roster state and recommendation output.
    """
    init_session_state()
    position_counts = _position_counts_from_state()
    construction_pressure = calculate_construction_pressure(position_counts=position_counts)
    current_strategy = identify_current_strategy(position_counts)
    roster_path = identify_roster_path(position_counts, construction_pressure)
    passed_targets = track_passed_targets()
    surviving_targets = track_surviving_targets()
    next_pick_plan = build_next_pick_plan(recommendations_df)
    confidence = calculate_strategy_confidence(
        current_strategy=current_strategy,
        roster_path=roster_path,
        passed_targets=passed_targets,
        construction_pressure=construction_pressure,
    )

    return {
        "current_strategy": current_strategy,
        "strategy_confidence": confidence["strategy_confidence"],
        "strategy_confidence_score": confidence["strategy_confidence_score"],
        "primary_target": next_pick_plan.get("primary_target"),
        "secondary_target": next_pick_plan.get("secondary_target"),
        "contingency_target": next_pick_plan.get("contingency_target"),
        "roster_path": roster_path,
        "next_pick_plan": next_pick_plan,
        "passed_targets": passed_targets,
        "surviving_targets": surviving_targets,
        "construction_pressure": construction_pressure,
        "confidence_drivers": confidence.get("drivers", {}),
    }


def get_strategy_debug_info(
    recommendations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    strategy = build_draft_strategy(recommendations_df)
    return {
        "current_pick": get_current_pick_number(),
        "league_size": get_league_size(),
        "my_team": st.session_state.get("my_team", []),
        "position_counts": strategy["roster_path"].get("position_counts", {}),
        "recommendation_history_count": len(st.session_state.get("recommendation_history", [])),
        "passed_target_count": len(strategy.get("passed_targets", [])),
        "surviving_target_count": len(strategy.get("surviving_targets", [])),
        "lost_target_count": len(
            [target for target in strategy.get("passed_targets", []) if target.get("target_lost")]
        ),
        "strategy": {
            "current_strategy": strategy.get("current_strategy"),
            "strategy_confidence": strategy.get("strategy_confidence"),
            "strategy_confidence_score": strategy.get("strategy_confidence_score"),
            "primary_target": strategy.get("primary_target"),
            "secondary_target": strategy.get("secondary_target"),
            "contingency_target": strategy.get("contingency_target"),
            "roster_path": strategy.get("roster_path"),
        },
    }
