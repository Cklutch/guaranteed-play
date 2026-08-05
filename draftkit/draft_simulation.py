import math
import random

import pandas as pd

from draftkit.data_access import get_available_players_df, safe_col
from draftkit.draft_state import get_current_pick_number, get_next_pick_distance


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
TEAM_COLS = ["team", "Team", "team_abbr"]
PROJECTION_COLS = ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"]
ADP_COLS = ["consensus_adp", "adp", "ADP", "rank", "Rank"]
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_simulation_columns(df):
    return {
        "player_col": safe_col(df, PLAYER_COLS),
        "position_col": safe_col(df, POSITION_COLS),
        "team_col": safe_col(df, TEAM_COLS),
        "projection_col": safe_col(df, PROJECTION_COLS),
        "adp_col": safe_col(df, ADP_COLS),
    }


def _prepare_available_players(df, columns):
    player_col = columns.get("player_col")
    position_col = columns.get("position_col")
    projection_col = columns.get("projection_col")
    adp_col = columns.get("adp_col")

    if df.empty or player_col is None:
        return pd.DataFrame()

    working_df = df.copy()

    if position_col is not None:
        working_df = working_df[
            working_df[position_col].astype(str).str.upper().isin(SCORABLE_POSITIONS)
        ].copy()

    if projection_col is not None:
        working_df[projection_col] = pd.to_numeric(working_df[projection_col], errors="coerce")

    if adp_col is not None:
        working_df[adp_col] = pd.to_numeric(working_df[adp_col], errors="coerce")
        working_df = working_df.sort_values(
            [adp_col, projection_col] if projection_col is not None else [adp_col],
            ascending=[True, False] if projection_col is not None else [True],
        ).reset_index(drop=True)
    elif projection_col is not None:
        working_df = working_df.sort_values(projection_col, ascending=False).reset_index(drop=True)
    else:
        working_df = working_df.reset_index(drop=True)

    working_df["simulation_player_key"] = working_df[player_col].astype(str)

    return working_df


def build_simulation_state():
    """
    Build a clean state object for Monte Carlo draft simulations.
    """
    current_pick = get_current_pick_number()
    next_pick_distance = get_next_pick_distance()
    if next_pick_distance is None:
        next_pick_distance = 0

    raw_df = get_available_players_df().copy()
    columns = _get_simulation_columns(raw_df) if not raw_df.empty else {}
    available_df = _prepare_available_players(raw_df, columns)

    warnings = []
    if raw_df.empty:
        warnings.append("Available players dataframe is empty.")
    if not columns.get("player_col"):
        warnings.append("Player-name column is missing.")
    if not columns.get("adp_col"):
        warnings.append("ADP column is missing; simulation will fall back to projections/order.")
    if next_pick_distance <= 0:
        warnings.append("Next pick distance is zero or unavailable.")

    return {
        "current_pick": current_pick,
        "next_pick_distance": int(max(next_pick_distance, 0)),
        "target_pick": int(current_pick + max(next_pick_distance, 0)),
        "available_players": available_df,
        "columns": columns,
        "warnings": warnings,
    }


def _calculate_pick_weights(players_df, columns, simulated_pick_number, randomness=8.0):
    adp_col = columns.get("adp_col")
    projection_col = columns.get("projection_col")

    weights = []
    for idx, row in players_df.iterrows():
        if adp_col is not None and adp_col in players_df.columns:
            adp = _safe_float(row.get(adp_col), simulated_pick_number + idx + 1)
            distance = abs(adp - simulated_pick_number)
            adp_weight = math.exp(-distance / max(randomness, 1.0))
        else:
            adp_weight = 1.0 / (idx + 1)

        projection_weight = 1.0
        if projection_col is not None and projection_col in players_df.columns:
            projection = max(_safe_float(row.get(projection_col), 0.0), 0.0)
            projection_weight += min(projection / 350.0, 1.0) * 0.30

        weights.append(max(adp_weight * projection_weight, 0.0001))

    return weights


def simulate_pick(players_df, columns, simulated_pick_number, rng=None, randomness=8.0):
    """
    Simulate one draft pick using ADP-weighted randomness.
    """
    if rng is None:
        rng = random.Random()

    if players_df.empty:
        return None, players_df

    weights = _calculate_pick_weights(
        players_df,
        columns,
        simulated_pick_number,
        randomness=randomness,
    )
    selected_index = rng.choices(list(players_df.index), weights=weights, k=1)[0]
    selected_row = players_df.loc[selected_index].to_dict()
    remaining_df = players_df.drop(index=selected_index).reset_index(drop=True)

    return selected_row, remaining_df


def simulate_until_next_pick(state=None, rng=None, randomness=8.0):
    """
    Simulate all picks between now and the user's next pick.
    """
    if state is None:
        state = build_simulation_state()
    if rng is None:
        rng = random.Random()

    players_df = state["available_players"].copy()
    columns = state.get("columns", {})
    current_pick = int(state.get("current_pick", 1))
    next_pick_distance = int(state.get("next_pick_distance", 0))
    drafted_players = []

    for offset in range(next_pick_distance):
        simulated_pick_number = current_pick + offset
        selected_row, players_df = simulate_pick(
            players_df,
            columns,
            simulated_pick_number,
            rng=rng,
            randomness=randomness,
        )

        if selected_row is None:
            break

        drafted_players.append(selected_row)

    return {
        "drafted_players": drafted_players,
        "remaining_players": players_df,
        "picks_simulated": len(drafted_players),
        "current_pick": current_pick,
        "target_pick": state.get("target_pick"),
    }


def run_draft_simulations(num_simulations=500, seed=None, randomness=8.0):
    """
    Run repeated draft simulations and track player survival.
    """
    state = build_simulation_state()
    players_df = state["available_players"]
    columns = state.get("columns", {})
    player_col = columns.get("player_col")

    if players_df.empty or player_col is None:
        return {
            "state": state,
            "num_simulations": 0,
            "survival_counts": {},
            "drafted_counts": {},
            "sample_drafted_players": [],
        }

    num_simulations = max(int(num_simulations), 1)
    rng = random.Random(seed)
    player_names = players_df[player_col].astype(str).tolist()
    survival_counts = {player_name: 0 for player_name in player_names}
    drafted_counts = {player_name: 0 for player_name in player_names}
    sample_drafted_players = []

    for simulation_index in range(num_simulations):
        simulation_rng = random.Random(rng.random())
        result = simulate_until_next_pick(
            state=state,
            rng=simulation_rng,
            randomness=randomness,
        )
        remaining_names = set(result["remaining_players"][player_col].astype(str).tolist())
        drafted_names = [
            str(row.get(player_col))
            for row in result["drafted_players"]
            if row.get(player_col) is not None
        ]

        for player_name in player_names:
            if player_name in remaining_names:
                survival_counts[player_name] += 1
            else:
                drafted_counts[player_name] += 1

        if simulation_index == 0:
            sample_drafted_players = drafted_names

    return {
        "state": state,
        "num_simulations": num_simulations,
        "survival_counts": survival_counts,
        "drafted_counts": drafted_counts,
        "sample_drafted_players": sample_drafted_players,
    }


def calculate_availability_probability(num_simulations=500, seed=None, randomness=8.0):
    """
    Estimate each player's probability of surviving until the user's next pick.
    """
    results = run_draft_simulations(
        num_simulations=num_simulations,
        seed=seed,
        randomness=randomness,
    )
    state = results["state"]
    players_df = state["available_players"]
    columns = state.get("columns", {})
    player_col = columns.get("player_col")
    position_col = columns.get("position_col")
    team_col = columns.get("team_col")
    projection_col = columns.get("projection_col")
    adp_col = columns.get("adp_col")
    simulation_count = results.get("num_simulations", 0)

    if players_df.empty or player_col is None or simulation_count <= 0:
        return pd.DataFrame()

    rows = []
    for _, row in players_df.iterrows():
        player_name = str(row[player_col])
        survival_count = results["survival_counts"].get(player_name, 0)
        availability_probability = survival_count / simulation_count

        rows.append({
            "player_name": player_name,
            "position": str(row[position_col]).upper() if position_col else "",
            "team": str(row[team_col]) if team_col else "",
            "projection_points": _safe_float(row.get(projection_col), None)
            if projection_col
            else None,
            "adp": _safe_float(row.get(adp_col), None) if adp_col else None,
            "availability_probability": round(availability_probability * 100.0, 2),
            "drafted_before_next_pick_probability": round(
                (1.0 - availability_probability) * 100.0,
                2,
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["drafted_before_next_pick_probability", "adp", "projection_points"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)


def get_wait_vs_take_recommendation(player_name, num_simulations=500, seed=None):
    """
    Recommend whether to draft a player now or wait based on survival probability.
    """
    if not player_name:
        return {
            "player_name": player_name,
            "recommendation": "UNKNOWN",
            "availability_probability": None,
            "reason": "No player name was provided.",
        }

    availability_df = calculate_availability_probability(
        num_simulations=num_simulations,
        seed=seed,
    )
    if availability_df.empty:
        return {
            "player_name": player_name,
            "recommendation": "UNKNOWN",
            "availability_probability": None,
            "reason": "Simulation did not produce availability probabilities.",
        }

    match = availability_df[
        availability_df["player_name"].astype(str).str.lower() == str(player_name).lower()
    ]
    if match.empty:
        return {
            "player_name": player_name,
            "recommendation": "UNKNOWN",
            "availability_probability": None,
            "reason": "Player is not currently available in the simulation pool.",
        }

    row = match.iloc[0].to_dict()
    availability = _safe_float(row.get("availability_probability"), 0.0)

    if availability <= 25.0:
        recommendation = "TAKE_NOW"
        reason = "Low chance this player survives to your next pick."
    elif availability <= 55.0:
        recommendation = "CONSIDER"
        reason = "Player may not make it back, so draft cost depends on roster fit."
    else:
        recommendation = "WAIT"
        reason = "Player is reasonably likely to survive to your next pick."

    return {
        "player_name": row["player_name"],
        "position": row.get("position", ""),
        "team": row.get("team", ""),
        "adp": row.get("adp"),
        "projection_points": row.get("projection_points"),
        "recommendation": recommendation,
        "availability_probability": availability,
        "drafted_before_next_pick_probability": row.get(
            "drafted_before_next_pick_probability"
        ),
        "reason": reason,
    }


def get_simulation_debug_info(num_simulations=100, seed=42):
    """
    Return diagnostics for simulation validation.
    """
    state = build_simulation_state()
    availability_df = calculate_availability_probability(
        num_simulations=num_simulations,
        seed=seed,
    )
    results = run_draft_simulations(num_simulations=1, seed=seed)

    return {
        "current_pick": state.get("current_pick"),
        "next_pick_distance": state.get("next_pick_distance"),
        "target_pick": state.get("target_pick"),
        "simulation_count": num_simulations,
        "detected_columns": state.get("columns", {}),
        "available_player_count": int(len(state.get("available_players", []))),
        "warnings": state.get("warnings", []),
        "sample_simulated_picks": results.get("sample_drafted_players", []),
        "top_low_survival_players": availability_df.head(10).to_dict("records")
        if not availability_df.empty
        else [],
        "top_likely_to_survive_players": availability_df.sort_values(
            "availability_probability",
            ascending=False,
        ).head(10).to_dict("records")
        if not availability_df.empty
        else [],
    }
