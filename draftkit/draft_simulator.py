from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from draftkit.championship_equity import build_championship_equity_df
from draftkit.common import name_key as _name_key
from draftkit.common import normalize_player_name as _normalize_name
from draftkit.common import safe_float as _safe_float
from draftkit.common import safe_int as _safe_int
from draftkit.construction_pressure import calculate_construction_pressure
from draftkit.data_access import get_available_players_df, load_players_df, safe_col
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_draft_slot,
    get_my_team_positions,
    init_session_state,
)
from draftkit.opponent_model import build_position_demand_forecast


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
TEAM_COLS = ["team", "Team", "team_abbr"]
PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "sportsbook_projection",
    "projection_points",
    "Projection",
    "projected_points",
    "FPTS",
    "proj_points",
]
ADP_COLS = ["consensus_adp", "adp", "ADP", "rank", "Rank"]
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
SIMULATION_BOARD_LIMIT = 260


def _get_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "player_col": safe_col(df, PLAYER_COLS),
        "position_col": safe_col(df, POSITION_COLS),
        "team_col": safe_col(df, TEAM_COLS),
        "projection_col": safe_col(df, PROJECTION_COLS),
        "adp_col": safe_col(df, ADP_COLS),
    }


def _snake_pick_slot(pick_number: int, league_size: int) -> Optional[int]:
    if pick_number <= 0 or league_size <= 0:
        return None

    round_number = ((pick_number - 1) // league_size) + 1
    pick_in_round = ((pick_number - 1) % league_size) + 1
    if round_number % 2 == 1:
        return pick_in_round
    return league_size - pick_in_round + 1


def _future_pick_numbers_after_selection() -> List[int]:
    current_pick = get_current_pick_number()
    league_size = get_league_size()
    my_slot = get_my_draft_slot()

    if league_size <= 0 or my_slot <= 0:
        return []

    max_search = league_size * 3
    future_picks = []
    for pick_number in range(current_pick + 1, current_pick + max_search + 1):
        slot = _snake_pick_slot(pick_number, league_size)
        if slot == my_slot:
            break
        future_picks.append(pick_number)

    return future_picks


def _prepare_available_players() -> tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    available_df = get_available_players_df().copy()
    columns = _get_columns(available_df) if not available_df.empty else {}

    player_col = columns.get("player_col")
    position_col = columns.get("position_col")
    projection_col = columns.get("projection_col")
    adp_col = columns.get("adp_col")

    if available_df.empty or player_col is None:
        return pd.DataFrame(), columns

    working_df = available_df.copy()
    working_df["sim_player_name"] = working_df[player_col].astype(str)

    if position_col is not None:
        working_df["sim_position"] = working_df[position_col].astype(str).str.upper()
        working_df = working_df[working_df["sim_position"].isin(SCORABLE_POSITIONS)].copy()
    else:
        working_df["sim_position"] = ""

    if projection_col is not None:
        working_df["sim_projection"] = pd.to_numeric(
            working_df[projection_col],
            errors="coerce",
        )
    else:
        working_df["sim_projection"] = 0.0

    if adp_col is not None:
        working_df["sim_adp"] = pd.to_numeric(working_df[adp_col], errors="coerce")
    else:
        working_df["sim_adp"] = None

    sort_cols = []
    ascending = []
    if adp_col is not None:
        sort_cols.append("sim_adp")
        ascending.append(True)
    if projection_col is not None:
        sort_cols.append("sim_projection")
        ascending.append(False)
    if sort_cols:
        working_df = working_df.sort_values(
            sort_cols,
            ascending=ascending,
            na_position="last",
        )

    if len(working_df) > SIMULATION_BOARD_LIMIT:
        working_df = working_df.head(SIMULATION_BOARD_LIMIT).copy()

    return working_df.reset_index(drop=True), columns


def _candidate_row(candidate: Any, recommendations_df: pd.DataFrame) -> Dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, pd.Series):
        return candidate.to_dict()

    name = _normalize_name(candidate)
    if recommendations_df is not None and not recommendations_df.empty:
        match = recommendations_df[
            recommendations_df["player_name"].astype(str).str.lower() == name.lower()
        ]
        if not match.empty:
            return match.iloc[0].to_dict()

    return {"player_name": name}


def _build_recommendation_lookup(recommendations_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if recommendations_df is None or recommendations_df.empty:
        return {}
    return {
        _name_key(row.get("player_name")): row.to_dict()
        for _, row in recommendations_df.iterrows()
    }


def _build_equity_lookup() -> Dict[str, float]:
    try:
        equity_df = build_championship_equity_df(load_players_df())
    except Exception:
        return {}

    if equity_df.empty:
        return {}

    return {
        _name_key(row.get("player_name")): _safe_float(
            row.get("championship_equity_score"),
            50.0,
        )
        for _, row in equity_df.iterrows()
    }


def _build_position_demand_lookup() -> Dict[str, float]:
    try:
        demand_df = build_position_demand_forecast()
    except Exception:
        return {}

    if demand_df.empty:
        return {}

    return {
        str(row.get("position")).upper(): _safe_float(
            row.get("expected_picks_before_next_turn"),
            0.0,
        )
        for _, row in demand_df.iterrows()
    }


def _pick_weights(
    players_df: pd.DataFrame,
    pick_number: int,
    position_demand: Dict[str, float],
    rng: random.Random,
) -> List[float]:
    row_count = len(players_df)
    if row_count <= 0:
        return []

    fallback_adp = np.arange(pick_number + 1, pick_number + row_count + 1, dtype=float)
    adp_values = pd.to_numeric(players_df["sim_adp"], errors="coerce").to_numpy(dtype=float)
    adp_values = np.where(np.isnan(adp_values), fallback_adp, adp_values)

    projection_values = pd.to_numeric(
        players_df["sim_projection"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    positions = players_df["sim_position"].astype(str).str.upper()
    demand_values = positions.map(
        lambda position: min(position_demand.get(position, 0.0) * 0.18, 0.55)
    ).to_numpy(dtype=float)

    adp_weight = np.exp(-np.abs(adp_values - pick_number) / 9.0)
    projection_weight = 1.0 + np.minimum(projection_values / 300.0, 0.45)
    demand_weight = 1.0 + demand_values
    noise = np.array([0.85 + (rng.random() * 0.30) for _ in range(row_count)])
    weights = adp_weight * projection_weight * demand_weight * noise
    return np.maximum(weights, 0.0001).tolist()


def simulate_player_selection(
    candidate: Any,
    available_df: Optional[pd.DataFrame] = None,
    recommendations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Return the board and roster context after drafting a candidate now.
    """
    init_session_state()
    if recommendations_df is None:
        recommendations_df = build_recommendation_rankings_df()
    candidate_row = _candidate_row(candidate, recommendations_df)
    candidate_name = _normalize_name(candidate_row.get("player_name"))
    candidate_position = _normalize_name(candidate_row.get("position")).upper()

    if available_df is None:
        available_df, columns = _prepare_available_players()
    else:
        columns = _get_columns(available_df)

    if not available_df.empty and candidate_name:
        available_df = available_df[
            available_df["sim_player_name"].astype(str).str.lower() != candidate_name.lower()
        ].copy()

    position_counts = get_my_team_positions(load_players_df())
    position_counts = {
        position: _safe_int(position_counts.get(position), 0)
        for position in SCORABLE_POSITIONS
    }
    if candidate_position in SCORABLE_POSITIONS:
        position_counts[candidate_position] = position_counts.get(candidate_position, 0) + 1

    construction = calculate_construction_pressure(position_counts=position_counts)

    return {
        "candidate": candidate_row,
        "candidate_name": candidate_name,
        "candidate_position": candidate_position,
        "remaining_board": available_df.reset_index(drop=True),
        "columns": columns,
        "position_counts_after_selection": position_counts,
        "construction_after_selection": construction,
    }


def estimate_future_board_state(
    selection_state: Dict[str, Any],
    rng: Optional[random.Random] = None,
    position_demand: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Simulate the board after opponent picks before the user's next turn.
    """
    rng = rng or random.Random()
    position_demand = position_demand or {}
    players_df = selection_state.get("remaining_board", pd.DataFrame()).copy()
    drafted_rows = []

    for pick_number in _future_pick_numbers_after_selection():
        if players_df.empty:
            break

        weights = _pick_weights(players_df, pick_number, position_demand, rng)
        selected_position = rng.choices(range(len(players_df)), weights=weights, k=1)[0]
        selected_index = players_df.index[selected_position]
        drafted_rows.append(players_df.iloc[selected_position].to_dict())
        players_df = players_df.drop(index=selected_index).reset_index(drop=True)

    return {
        "remaining_board": players_df.reset_index(drop=True),
        "lost_players": drafted_rows,
        "picks_simulated": len(drafted_rows),
    }


def estimate_target_survival(
    candidate: Any,
    target_names: List[str],
    num_simulations: int = 150,
    seed: Optional[int] = None,
    recommendations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Estimate which named targets survive after drafting the candidate.
    """
    report = simulate_draft_paths(
        candidates=[candidate],
        recommendations_df=recommendations_df,
        num_simulations=num_simulations,
        seed=seed,
    )
    rows = report.get("candidate_reports", [])
    if not rows:
        return {}

    row = rows[0]
    target_set = {_name_key(name) for name in target_names}
    return {
        "player_name": row.get("player_name"),
        "target_survival": {
            target.get("player_name"): target.get("survival_probability")
            for target in row.get("target_survival_details", [])
            if _name_key(target.get("player_name")) in target_set
        },
    }


def estimate_expected_roster_value(
    candidate_row: Dict[str, Any],
    surviving_targets: List[Dict[str, Any]],
) -> float:
    candidate_score = _safe_float(candidate_row.get("final_score"), 0.0)
    best_future_score = max(
        [_safe_float(target.get("recommendation_score"), 0.0) for target in surviving_targets]
        or [0.0]
    )
    return round((candidate_score * 0.70) + (best_future_score * 0.30), 2)


def estimate_expected_championship_equity(
    candidate_row: Dict[str, Any],
    surviving_targets: List[Dict[str, Any]],
    equity_lookup: Optional[Dict[str, float]] = None,
) -> float:
    equity_lookup = equity_lookup or _build_equity_lookup()
    candidate_equity = equity_lookup.get(_name_key(candidate_row.get("player_name")), 50.0)
    future_equities = [
        equity_lookup.get(_name_key(target.get("player_name")), 50.0)
        for target in surviving_targets
    ]
    best_future_equity = max(future_equities or [50.0])
    return round((candidate_equity * 0.72) + (best_future_equity * 0.28), 2)


def _construction_score(construction: Dict[str, Any]) -> float:
    completeness = _safe_float(construction.get("roster_completeness"), 0.0)
    critical_need_count = len(construction.get("critical_needs", []))
    return round(max(0.0, min(100.0, completeness - (critical_need_count * 8.0))), 2)


def simulate_draft_paths(
    candidates: Optional[List[Any]] = None,
    recommendations_df: Optional[pd.DataFrame] = None,
    num_simulations: int = 150,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo draft paths after selecting each candidate.
    """
    init_session_state()
    if recommendations_df is None:
        recommendations_df = build_recommendation_rankings_df()
    if recommendations_df is None:
        recommendations_df = pd.DataFrame()

    if candidates is None:
        candidates = recommendations_df.head(5).to_dict("records") if not recommendations_df.empty else []

    available_df, columns = _prepare_available_players()
    recommendation_lookup = _build_recommendation_lookup(recommendations_df)
    equity_lookup = _build_equity_lookup()
    position_demand = _build_position_demand_lookup()
    target_pool = recommendations_df.head(12).to_dict("records") if not recommendations_df.empty else []
    target_names = [_name_key(row.get("player_name")) for row in target_pool]

    num_simulations = max(_safe_int(num_simulations, 150), 1)
    rng = random.Random(seed)
    candidate_reports = []

    for candidate in candidates:
        selection_state = simulate_player_selection(
            candidate,
            available_df=available_df,
            recommendations_df=recommendations_df,
        )
        candidate_row = selection_state["candidate"]
        candidate_name = selection_state["candidate_name"]
        candidate_key = _name_key(candidate_name)

        survival_counts = {name: 0 for name in target_names if name != candidate_key}
        lost_counts = {name: 0 for name in target_names if name != candidate_key}
        board_quality_scores = []

        for _ in range(num_simulations):
            path_rng = random.Random(rng.random())
            future_state = estimate_future_board_state(
                selection_state,
                rng=path_rng,
                position_demand=position_demand,
            )
            remaining_names = {
                _name_key(name)
                for name in future_state["remaining_board"]["sim_player_name"].astype(str).tolist()
            } if not future_state["remaining_board"].empty else set()

            surviving_scores = []
            for target_key in survival_counts:
                if target_key in remaining_names:
                    survival_counts[target_key] += 1
                    surviving_scores.append(
                        _safe_float(
                            recommendation_lookup.get(target_key, {}).get("final_score"),
                            0.0,
                        )
                    )
                else:
                    lost_counts[target_key] += 1

            board_quality_scores.append(max(surviving_scores or [0.0]))

        survival_details = []
        for target_key, survival_count in survival_counts.items():
            target = recommendation_lookup.get(target_key, {})
            probability = round((survival_count / num_simulations) * 100.0, 2)
            survival_details.append({
                "player_name": target.get("player_name"),
                "position": target.get("position"),
                "recommendation_score": _safe_float(target.get("final_score"), 0.0),
                "survival_probability": probability,
                "lost_probability": round(100.0 - probability, 2),
            })

        likely_surviving = sorted(
            survival_details,
            key=lambda row: (
                row.get("survival_probability", 0.0),
                row.get("recommendation_score", 0.0),
            ),
            reverse=True,
        )[:5]
        likely_lost = sorted(
            survival_details,
            key=lambda row: (
                row.get("lost_probability", 0.0),
                row.get("recommendation_score", 0.0),
            ),
            reverse=True,
        )[:5]

        candidate_survival_baseline = 0.0
        if columns.get("adp_col") and candidate_row:
            current_pick = get_current_pick_number()
            next_pick_numbers = _future_pick_numbers_after_selection()
            next_pick = next_pick_numbers[-1] + 1 if next_pick_numbers else current_pick
            adp = _safe_float(candidate_row.get("adp", candidate_row.get("consensus_adp")), None)
            if adp is not None:
                candidate_survival_baseline = round(
                    max(0.0, min(100.0, math.exp(-max(next_pick - adp, 0.0) / 12.0) * 100.0)),
                    2,
                )

        expected_roster_value = estimate_expected_roster_value(
            candidate_row,
            likely_surviving,
        )
        expected_championship_equity = estimate_expected_championship_equity(
            candidate_row,
            likely_surviving,
            equity_lookup=equity_lookup,
        )
        expected_construction_score = _construction_score(
            selection_state.get("construction_after_selection", {}),
        )
        future_pick_quality = round(
            sum(board_quality_scores) / len(board_quality_scores),
            2,
        ) if board_quality_scores else 0.0

        candidate_reports.append({
            "player_name": candidate_name,
            "position": candidate_row.get("position"),
            "team": candidate_row.get("team"),
            "survival_probability": candidate_survival_baseline,
            "expected_roster_value": expected_roster_value,
            "expected_championship_equity": expected_championship_equity,
            "expected_construction_score": expected_construction_score,
            "future_pick_quality": future_pick_quality,
            "likely_surviving_targets": [row["player_name"] for row in likely_surviving if row.get("player_name")],
            "likely_lost_targets": [row["player_name"] for row in likely_lost if row.get("player_name")],
            "target_survival_details": survival_details,
            "simulations_run": num_simulations,
        })

    return {
        "candidate_reports": candidate_reports,
        "simulation_count": num_simulations,
        "future_pick_count": len(_future_pick_numbers_after_selection()),
        "current_pick": get_current_pick_number(),
        "columns": columns,
        "position_demand": position_demand,
    }


def build_simulation_report(
    recommendations_df: Optional[pd.DataFrame] = None,
    limit: int = 5,
    num_simulations: int = 150,
    seed: Optional[int] = 41,
) -> pd.DataFrame:
    """
    Return future-impact metrics for the top recommendation candidates.
    """
    if recommendations_df is None:
        recommendations_df = build_recommendation_rankings_df()
    if recommendations_df is None or recommendations_df.empty:
        return pd.DataFrame()

    candidates = recommendations_df.head(limit).to_dict("records")
    report = simulate_draft_paths(
        candidates=candidates,
        recommendations_df=recommendations_df,
        num_simulations=num_simulations,
        seed=seed,
    )
    rows = report.get("candidate_reports", [])
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_simulator_debug_info(
    recommendations_df: Optional[pd.DataFrame] = None,
    num_simulations: int = 100,
    seed: int = 41,
) -> Dict[str, Any]:
    if recommendations_df is None:
        recommendations_df = build_recommendation_rankings_df()

    simulation_df = build_simulation_report(
        recommendations_df=recommendations_df,
        limit=5,
        num_simulations=num_simulations,
        seed=seed,
    )
    available_df, columns = _prepare_available_players()

    return {
        "current_pick": get_current_pick_number(),
        "league_size": get_league_size(),
        "draft_slot": get_my_draft_slot(),
        "future_pick_numbers": _future_pick_numbers_after_selection(),
        "available_player_count": int(len(available_df)),
        "detected_columns": columns,
        "simulation_count": num_simulations,
        "candidate_count": int(len(simulation_df)),
        "position_demand": _build_position_demand_lookup(),
        "sample_reports": simulation_df.head(5).to_dict("records")
        if not simulation_df.empty
        else [],
    }
