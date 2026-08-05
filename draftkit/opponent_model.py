import math

import pandas as pd
import streamlit as st

from draftkit.data_access import get_available_players_df, load_players_df, safe_col
from draftkit.draft_analysis import build_position_urgency_df
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_draft_slot,
    get_next_pick_distance,
    get_player_position_map,
)


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
TEAM_COLS = ["team", "Team", "team_abbr"]
PROJECTION_COLS = ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"]
ADP_COLS = ["consensus_adp", "adp", "ADP", "rank", "Rank"]
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
FLEX_POSITIONS = ["RB", "WR", "TE"]


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_player_columns(df):
    return {
        "player_col": safe_col(df, PLAYER_COLS),
        "position_col": safe_col(df, POSITION_COLS),
        "team_col": safe_col(df, TEAM_COLS),
        "projection_col": safe_col(df, PROJECTION_COLS),
        "adp_col": safe_col(df, ADP_COLS),
    }


def _snake_pick_slot(pick_number, league_size):
    if league_size <= 0 or pick_number <= 0:
        return None

    round_number = ((pick_number - 1) // league_size) + 1
    pick_in_round = ((pick_number - 1) % league_size) + 1

    if round_number % 2 == 1:
        return pick_in_round

    return league_size - pick_in_round + 1


def _future_pick_slots_until_next_pick():
    current_pick = get_current_pick_number()
    next_pick_distance = get_next_pick_distance()
    league_size = get_league_size()
    my_slot = get_my_draft_slot()

    if next_pick_distance is None:
        next_pick_distance = 0

    slots = []
    for offset in range(max(int(next_pick_distance), 0)):
        pick_number = current_pick + offset
        slot = _snake_pick_slot(pick_number, league_size)
        if slot is not None and slot != my_slot:
            slots.append(slot)

    return slots


def _get_position_for_player(player_name, position_map):
    if player_name in position_map:
        return position_map[player_name]

    normalized = str(player_name).strip().lower()
    for mapped_name, position in position_map.items():
        if str(mapped_name).strip().lower() == normalized:
            return position

    return None


def _infer_pick_slot_from_log_entry(entry, index, league_size):
    for key in ["draft_slot", "slot", "team_id", "roster_id"]:
        if key in entry and entry.get(key) is not None:
            try:
                return int(entry.get(key))
            except (TypeError, ValueError):
                pass

    pick_number = entry.get("pick_number") or entry.get("pick_no") or entry.get("pick")
    try:
        pick_number = int(pick_number)
    except (TypeError, ValueError):
        pick_number = index + 1

    return _snake_pick_slot(pick_number, league_size)


def build_opponent_roster_profiles():
    """
    Estimate roster construction for each draft slot.
    """
    league_size = get_league_size()
    my_slot = get_my_draft_slot()
    roster_settings = st.session_state.get("roster_settings", {})
    raw_df = load_players_df().copy()
    position_map = get_player_position_map(raw_df)
    draft_log = st.session_state.get("draft_log", [])

    profile_lookup = {}
    for slot in range(1, league_size + 1):
        profile_lookup[slot] = {
            "team_id": slot,
            "is_user_team": slot == my_slot,
            "roster_size": 0,
            "QB": 0,
            "RB": 0,
            "WR": 0,
            "TE": 0,
        }

    for index, entry in enumerate(draft_log):
        if not isinstance(entry, dict):
            continue

        player_name = (
            entry.get("player_name")
            or entry.get("player")
            or entry.get("name")
        )
        if not player_name:
            continue

        action_type = entry.get("action_type")
        if action_type == "my_pick":
            slot = my_slot
        else:
            slot = _infer_pick_slot_from_log_entry(entry, index, league_size)

        if slot not in profile_lookup:
            continue

        position = (
            entry.get("position")
            or entry.get("pos")
            or _get_position_for_player(player_name, position_map)
        )
        position = str(position).upper() if position else None
        if position not in SCORABLE_POSITIONS:
            continue

        profile_lookup[slot][position] += 1
        profile_lookup[slot]["roster_size"] += 1

    profiles = []
    for slot, profile in profile_lookup.items():
        output = profile.copy()
        for position in SCORABLE_POSITIONS:
            target = int(roster_settings.get(position, 0))
            output[f"{position}_starter_need"] = max(target - output[position], 0)

        flex_target = int(roster_settings.get("FLEX", 0))
        flex_have = sum(output[position] for position in FLEX_POSITIONS)
        flex_base_targets = sum(int(roster_settings.get(position, 0)) for position in FLEX_POSITIONS)
        output["FLEX_need"] = max((flex_base_targets + flex_target) - flex_have, 0)
        profiles.append(output)

    return pd.DataFrame(profiles)


def calculate_opponent_position_needs(roster_profiles_df=None):
    """
    Convert opponent roster profiles into position need scores.
    """
    if roster_profiles_df is None:
        roster_profiles_df = build_opponent_roster_profiles()

    if roster_profiles_df.empty:
        return pd.DataFrame()

    rows = []
    future_slots = _future_pick_slots_until_next_pick()
    future_slot_counts = {
        slot: future_slots.count(slot)
        for slot in set(future_slots)
    }

    for _, profile in roster_profiles_df.iterrows():
        if bool(profile.get("is_user_team", False)):
            continue

        team_id = int(profile["team_id"])
        picks_before_next_turn = future_slot_counts.get(team_id, 0)

        for position in SCORABLE_POSITIONS:
            starter_need = _safe_float(profile.get(f"{position}_starter_need"), 0.0)
            flex_need = _safe_float(profile.get("FLEX_need"), 0.0) if position in FLEX_POSITIONS else 0.0
            have = _safe_float(profile.get(position), 0.0)
            need_score = starter_need + (flex_need * 0.35)

            if have <= 0 and starter_need > 0:
                need_score += 0.75

            rows.append({
                "team_id": team_id,
                "position": position,
                "have": int(have),
                "starter_need": int(starter_need),
                "flex_need": int(flex_need),
                "need_score": round(need_score, 3),
                "picks_before_next_turn": picks_before_next_turn,
            })

    return pd.DataFrame(rows)


def calculate_position_selection_probability(team_need_row, position_market=None):
    """
    Estimate how likely one opponent pick is to target a position.
    """
    position_market = position_market or {}
    need_score = _safe_float(team_need_row.get("need_score"), 0.0)
    market_score = _safe_float(position_market.get(team_need_row.get("position")), 1.0)

    raw_score = max(need_score, 0.15) * max(market_score, 0.25)
    return raw_score


def _build_position_market_scores(available_df, columns):
    position_col = columns.get("position_col")
    adp_col = columns.get("adp_col")
    current_pick = get_current_pick_number()
    next_pick_distance = get_next_pick_distance() or 0
    target_pick = current_pick + next_pick_distance

    if available_df.empty or position_col is None:
        return {position: 1.0 for position in SCORABLE_POSITIONS}

    market_scores = {}
    for position in SCORABLE_POSITIONS:
        pos_df = available_df[
            available_df[position_col].astype(str).str.upper() == position
        ].copy()

        if pos_df.empty:
            market_scores[position] = 0.25
            continue

        if adp_col is not None and adp_col in pos_df.columns:
            adps = pd.to_numeric(pos_df[adp_col], errors="coerce").dropna()
            near_pick_count = int(
                ((adps >= current_pick - 3) & (adps <= target_pick + 6)).sum()
            )
            market_scores[position] = 0.75 + min(near_pick_count * 0.20, 1.25)
        else:
            market_scores[position] = 1.0

    urgency_df = build_position_urgency_df()
    if not urgency_df.empty:
        for _, row in urgency_df.iterrows():
            position = str(row["Position"]).upper()
            if position in market_scores:
                urgency_score = _safe_float(row.get("Urgency Score"), 0.0)
                market_scores[position] += min(urgency_score / 100.0, 0.75)

    return market_scores


def _normalize_probabilities(rows):
    total = sum(max(_safe_float(row["raw_probability"], 0.0), 0.0) for row in rows)
    if total <= 0:
        equal_probability = 1.0 / len(rows) if rows else 0.0
        for row in rows:
            row["selection_probability"] = round(equal_probability, 4)
        return rows

    for row in rows:
        row["selection_probability"] = round(
            max(_safe_float(row["raw_probability"], 0.0), 0.0) / total,
            4,
        )

    return rows


def build_position_demand_forecast():
    """
    Forecast expected position demand before the user's next pick.
    """
    available_df = get_available_players_df().copy()
    columns = _get_player_columns(available_df) if not available_df.empty else {}
    needs_df = calculate_opponent_position_needs()
    future_slots = _future_pick_slots_until_next_pick()
    picks_before_next_turn = len(future_slots)

    if needs_df.empty:
        return pd.DataFrame()

    position_market = _build_position_market_scores(available_df, columns)
    position_expected_picks = {position: 0.0 for position in SCORABLE_POSITIONS}
    position_demand = {position: 0.0 for position in SCORABLE_POSITIONS}

    for slot in future_slots:
        team_needs = needs_df[needs_df["team_id"] == slot].copy()
        if team_needs.empty:
            continue

        probability_rows = []
        for _, need_row in team_needs.iterrows():
            position = str(need_row["position"]).upper()
            raw_probability = calculate_position_selection_probability(
                need_row,
                position_market=position_market,
            )
            probability_rows.append({
                "position": position,
                "raw_probability": raw_probability,
            })
            position_demand[position] += _safe_float(need_row.get("need_score"), 0.0)

        for probability_row in _normalize_probabilities(probability_rows):
            position_expected_picks[probability_row["position"]] += probability_row[
                "selection_probability"
            ]

    rows = []
    position_col = columns.get("position_col")
    player_col = columns.get("player_col")
    adp_col = columns.get("adp_col")
    projection_col = columns.get("projection_col")

    for position in SCORABLE_POSITIONS:
        expected_picks = round(position_expected_picks.get(position, 0.0), 2)
        demand = round(position_demand.get(position, 0.0), 2)
        pos_df = pd.DataFrame()

        if not available_df.empty and position_col is not None:
            pos_df = available_df[
                available_df[position_col].astype(str).str.upper() == position
            ].copy()
            if adp_col is not None:
                pos_df[adp_col] = pd.to_numeric(pos_df[adp_col], errors="coerce")
                pos_df = pos_df.sort_values(adp_col, ascending=True)
            elif projection_col is not None:
                pos_df[projection_col] = pd.to_numeric(pos_df[projection_col], errors="coerce")
                pos_df = pos_df.sort_values(projection_col, ascending=False)

        available_top_players = (
            pos_df[player_col].astype(str).head(5).tolist()
            if not pos_df.empty and player_col is not None
            else []
        )

        if expected_picks >= 2.5:
            threat_level = "CRITICAL"
        elif expected_picks >= 1.5:
            threat_level = "HIGH"
        elif expected_picks >= 0.75:
            threat_level = "MODERATE"
        else:
            threat_level = "LOW"

        rows.append({
            "position": position,
            "position_demand": demand,
            "expected_picks_before_next_turn": expected_picks,
            "available_top_players": available_top_players,
            "tier_pressure": round(position_market.get(position, 1.0), 2),
            "threat_level": threat_level,
            "picks_before_next_turn": picks_before_next_turn,
        })

    return pd.DataFrame(rows).sort_values(
        ["expected_picks_before_next_turn", "position_demand", "tier_pressure"],
        ascending=False,
    ).reset_index(drop=True)


def estimate_player_draft_probability(player_row, demand_forecast_df=None):
    """
    Estimate chance a player is drafted before the user's next pick.
    """
    if demand_forecast_df is None:
        demand_forecast_df = build_position_demand_forecast()

    if demand_forecast_df.empty:
        return 0.0

    position = str(player_row.get("position") or player_row.get("Position") or "").upper()
    if position not in SCORABLE_POSITIONS:
        return 0.0

    forecast_match = demand_forecast_df[demand_forecast_df["position"] == position]
    if forecast_match.empty:
        return 0.0

    forecast = forecast_match.iloc[0]
    expected_position_picks = _safe_float(
        forecast.get("expected_picks_before_next_turn"),
        0.0,
    )

    adp = _safe_float(
        player_row.get("adp", player_row.get("ADP", player_row.get("consensus_adp"))),
        None,
    )
    current_pick = get_current_pick_number()
    next_pick_distance = get_next_pick_distance() or 0
    target_pick = current_pick + next_pick_distance

    if adp is None:
        adp_pressure = 0.45
    else:
        adp_pressure = math.exp(-max(adp - target_pick, 0.0) / 10.0)
        if adp <= current_pick:
            adp_pressure = 1.0

    probability = 1.0 - math.exp(-expected_position_picks * max(adp_pressure, 0.10))
    return round(max(0.0, min(probability * 100.0, 100.0)), 2)


def get_position_threat_report():
    """
    Return the most threatened positions and likely player losses.
    """
    forecast_df = build_position_demand_forecast()
    available_df = get_available_players_df().copy()
    columns = _get_player_columns(available_df) if not available_df.empty else {}
    player_col = columns.get("player_col")
    position_col = columns.get("position_col")
    adp_col = columns.get("adp_col")
    projection_col = columns.get("projection_col")

    if forecast_df.empty:
        return []

    reports = []
    for _, forecast in forecast_df.iterrows():
        position = forecast["position"]
        pos_df = pd.DataFrame()

        if not available_df.empty and position_col is not None:
            pos_df = available_df[
                available_df[position_col].astype(str).str.upper() == position
            ].copy()
            if adp_col is not None:
                pos_df[adp_col] = pd.to_numeric(pos_df[adp_col], errors="coerce")
                pos_df = pos_df.sort_values(adp_col, ascending=True)
            elif projection_col is not None:
                pos_df[projection_col] = pd.to_numeric(pos_df[projection_col], errors="coerce")
                pos_df = pos_df.sort_values(projection_col, ascending=False)

        likely_player_losses = []
        if not pos_df.empty and player_col is not None:
            loss_count = max(
                1,
                int(math.ceil(_safe_float(forecast["expected_picks_before_next_turn"], 0.0))),
            )
            likely_player_losses = pos_df[player_col].astype(str).head(loss_count).tolist()

        reasons = [
            f"Expected {forecast['expected_picks_before_next_turn']} {position} pick(s) before your next turn",
            f"{position} demand score is {forecast['position_demand']}",
            f"Tier/market pressure score is {forecast['tier_pressure']}",
        ]

        reports.append({
            "position": position,
            "threat_level": forecast["threat_level"],
            "expected_picks_before_next_turn": forecast["expected_picks_before_next_turn"],
            "position_demand": forecast["position_demand"],
            "likely_player_losses": likely_player_losses,
            "reasons": reasons,
        })

    return reports


def get_opponent_model_debug_info():
    """
    Return diagnostics for opponent demand and threat modeling.
    """
    available_df = get_available_players_df().copy()
    columns = _get_player_columns(available_df) if not available_df.empty else {}
    roster_profiles_df = build_opponent_roster_profiles()
    position_needs_df = calculate_opponent_position_needs(roster_profiles_df)
    forecast_df = build_position_demand_forecast()
    future_slots = _future_pick_slots_until_next_pick()

    warnings = []
    if not st.session_state.get("draft_log"):
        warnings.append("No detailed opponent draft log found; opponent rosters are inferred as mostly empty.")
    if get_next_pick_distance() in [None, 0]:
        warnings.append("Next pick distance is unavailable or zero, so demand forecast may be flat.")
    if not columns.get("adp_col"):
        warnings.append("ADP column is missing; player-loss ordering falls back to projection/order.")

    return {
        "current_pick": get_current_pick_number(),
        "next_pick_distance": get_next_pick_distance(),
        "league_size": get_league_size(),
        "draft_slot": get_my_draft_slot(),
        "future_pick_slots_before_next_turn": future_slots,
        "roster_settings": st.session_state.get("roster_settings", {}),
        "detected_player_columns": columns,
        "opponent_roster_profiles": roster_profiles_df.to_dict("records")
        if not roster_profiles_df.empty
        else [],
        "position_needs": position_needs_df.to_dict("records")
        if not position_needs_df.empty
        else [],
        "position_demand_forecast": forecast_df.to_dict("records")
        if not forecast_df.empty
        else [],
        "position_threat_report": get_position_threat_report(),
        "warnings": warnings,
    }
