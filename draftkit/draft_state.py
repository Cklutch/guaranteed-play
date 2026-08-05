# draftkit/draft_state.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


DEFAULT_LEAGUE_SIZE = 12
DEFAULT_DRAFT_SLOT = 1
DEFAULT_CURRENT_PICK = 1

DEFAULT_ROSTER_SETTINGS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "DST": 1,
    "K": 1,
    "BENCH": 6,
}


def _normalize_player_name(player_name: Any) -> str:
    """
    Normalize player names for consistent session-state comparisons.
    """
    if player_name is None:
        return ""

    return str(player_name).strip()


def _dedupe_preserve_order(values: List[Any]) -> List[str]:
    """
    Deduplicate a list while preserving order.
    """
    seen = set()
    cleaned = []

    for value in values:
        normalized = _normalize_player_name(value)

        if not normalized:
            continue

        key = normalized.lower()

        if key not in seen:
            cleaned.append(normalized)
            seen.add(key)

    return cleaned


def _remove_name_from_list(values: List[Any], player_name: str) -> List[str]:
    """
    Remove a player from a list using case-insensitive matching.
    """
    target = _normalize_player_name(player_name).lower()

    return [
        _normalize_player_name(value)
        for value in values
        if _normalize_player_name(value).lower() != target
    ]


def _get_player_name_from_row(player_row: Any) -> str:
    """
    Extract player name from a row, dict, Series, or raw string.
    """
    if isinstance(player_row, str):
        return _normalize_player_name(player_row)

    if isinstance(player_row, dict):
        for col in ["player_name", "player", "name", "Player", "PLAYER"]:
            if col in player_row:
                return _normalize_player_name(player_row[col])

    if isinstance(player_row, pd.Series):
        for col in ["player_name", "player", "name", "Player", "PLAYER"]:
            if col in player_row.index:
                return _normalize_player_name(player_row[col])

    return ""


def _get_position_from_row(player_row: Any) -> Optional[str]:
    """
    Extract position from a row, dict, or Series.
    """
    if isinstance(player_row, dict):
        for col in ["position", "pos", "Position", "POS"]:
            if col in player_row:
                value = player_row[col]
                return None if pd.isna(value) else str(value).strip().upper()

    if isinstance(player_row, pd.Series):
        for col in ["position", "pos", "Position", "POS"]:
            if col in player_row.index:
                value = player_row[col]
                return None if pd.isna(value) else str(value).strip().upper()

    return None


def init_session_state() -> None:
    """
    Initialize all Streamlit session-state values used across the multipage app.
    """
    defaults = {
        "drafted_players": [],
        "my_team": [],
        "draft_log": [],
        "league_size": DEFAULT_LEAGUE_SIZE,
        "my_draft_slot": DEFAULT_DRAFT_SLOT,
        "current_pick_number": DEFAULT_CURRENT_PICK,
        "roster_settings": DEFAULT_ROSTER_SETTINGS.copy(),
        "player_position_map": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Defensive cleanup in case older versions stored duplicates.
    st.session_state["drafted_players"] = _dedupe_preserve_order(
        st.session_state.get("drafted_players", [])
    )
    st.session_state["my_team"] = _dedupe_preserve_order(
        st.session_state.get("my_team", [])
    )


def keep_session_state_alive() -> None:
    """
    Call this near the top of each page to ensure state exists and stays normalized.
    """
    init_session_state()


def get_league_size() -> int:
    init_session_state()
    return int(st.session_state.get("league_size", DEFAULT_LEAGUE_SIZE))


def get_my_draft_slot() -> int:
    init_session_state()
    return int(st.session_state.get("my_draft_slot", DEFAULT_DRAFT_SLOT))


def get_current_pick_number() -> int:
    init_session_state()
    return int(st.session_state.get("current_pick_number", DEFAULT_CURRENT_PICK))


def set_current_pick_number(pick_number: int) -> None:
    init_session_state()

    try:
        pick_number = int(pick_number)
    except (TypeError, ValueError):
        pick_number = DEFAULT_CURRENT_PICK

    st.session_state["current_pick_number"] = max(1, pick_number)


def advance_pick(amount: int = 1) -> None:
    init_session_state()
    st.session_state["current_pick_number"] = get_current_pick_number() + int(amount)


def undo_pick() -> None:
    """
    Undo the most recent logged draft action.

    This reverts:
    - drafted player state
    - my team state, if the pick was mine
    - current pick number moves back by one
    """
    init_session_state()

    draft_log = st.session_state.get("draft_log", [])

    if not draft_log:
        return

    last_action = draft_log.pop()
    player_name = _normalize_player_name(last_action.get("player_name"))
    action_type = last_action.get("action_type")

    if player_name:
        st.session_state["drafted_players"] = _remove_name_from_list(
            st.session_state.get("drafted_players", []),
            player_name,
        )

        if action_type == "my_pick":
            st.session_state["my_team"] = _remove_name_from_list(
                st.session_state.get("my_team", []),
                player_name,
            )

    st.session_state["draft_log"] = draft_log
    st.session_state["current_pick_number"] = max(1, get_current_pick_number() - 1)


def get_next_pick_distance() -> Optional[int]:
    """
    Estimate distance from current pick to user's next pick in a snake draft.

    Returns:
        int distance to next user pick, or None if settings are invalid.
    """
    init_session_state()

    league_size = get_league_size()
    draft_slot = get_my_draft_slot()
    current_pick = get_current_pick_number()

    if league_size <= 0 or draft_slot <= 0 or draft_slot > league_size:
        return None

    if current_pick <= draft_slot:
        return draft_slot - current_pick

    max_search_picks = league_size * 30

    for pick in range(current_pick, current_pick + max_search_picks + 1):
        round_number = ((pick - 1) // league_size) + 1
        pick_in_round = ((pick - 1) % league_size) + 1

        if round_number % 2 == 1:
            user_pick_in_round = draft_slot
        else:
            user_pick_in_round = league_size - draft_slot + 1

        if pick_in_round == user_pick_in_round:
            return pick - current_pick

    return None


def get_player_position_map(players_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Return a player-name-to-position map.

    If players_df is provided, update session state with any discovered positions.
    """
    init_session_state()

    position_map = dict(st.session_state.get("player_position_map", {}))

    if players_df is not None and not players_df.empty:
        name_col = None
        pos_col = None

        for col in ["player_name", "player", "name", "Player", "PLAYER"]:
            if col in players_df.columns:
                name_col = col
                break

        for col in ["position", "pos", "Position", "POS"]:
            if col in players_df.columns:
                pos_col = col
                break

        if name_col and pos_col:
            for _, row in players_df.iterrows():
                player_name = _normalize_player_name(row.get(name_col))
                position = row.get(pos_col)

                if player_name and not pd.isna(position):
                    position_map[player_name] = str(position).strip().upper()

    st.session_state["player_position_map"] = position_map

    return position_map


def get_my_team_positions(players_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """
    Count positions on the user's current team.

    This is the function draft_analysis.py should rely on for roster need logic.
    """
    init_session_state()

    position_map = get_player_position_map(players_df)
    counts: Dict[str, int] = {}

    for player_name in st.session_state.get("my_team", []):
        position = position_map.get(player_name)

        if not position:
            continue

        counts[position] = counts.get(position, 0) + 1

    return counts


def handle_draft_action(player_row: Any) -> None:
    """
    Mark a player as drafted by another team.
    """
    init_session_state()

    player_name = _get_player_name_from_row(player_row)

    if not player_name:
        return

    st.session_state["drafted_players"] = _dedupe_preserve_order(
        st.session_state.get("drafted_players", []) + [player_name]
    )

    position = _get_position_from_row(player_row)

    if position:
        st.session_state["player_position_map"][player_name] = position

    st.session_state["draft_log"].append(
        {
            "pick_number": get_current_pick_number(),
            "player_name": player_name,
            "position": position,
            "action_type": "drafted",
        }
    )

    advance_pick()


def handle_my_pick(player_row: Any) -> None:
    """
    Draft a player to the user's team.
    """
    init_session_state()

    player_name = _get_player_name_from_row(player_row)

    if not player_name:
        return

    st.session_state["drafted_players"] = _dedupe_preserve_order(
        st.session_state.get("drafted_players", []) + [player_name]
    )

    st.session_state["my_team"] = _dedupe_preserve_order(
        st.session_state.get("my_team", []) + [player_name]
    )

    position = _get_position_from_row(player_row)

    if position:
        st.session_state["player_position_map"][player_name] = position

    st.session_state["draft_log"].append(
        {
            "pick_number": get_current_pick_number(),
            "player_name": player_name,
            "position": position,
            "action_type": "my_pick",
        }
    )

    advance_pick()


def remove_from_my_team(
    player_name: str,
    make_available: bool = True,
) -> None:
    """
    Remove a player from the user's team.

    Args:
        player_name:
            Player to remove from my_team.

        make_available:
            If True, also remove the player from drafted_players so they reappear
            on the Draft Board. This is usually what you want when correcting
            an accidental My Team click.

            If False, the player is removed from my roster but remains drafted.
            This is useful if the player was drafted by someone else instead.
    """
    init_session_state()

    normalized_name = _normalize_player_name(player_name)

    if not normalized_name:
        return

    st.session_state["my_team"] = _remove_name_from_list(
        st.session_state.get("my_team", []),
        normalized_name,
    )

    if make_available:
        st.session_state["drafted_players"] = _remove_name_from_list(
            st.session_state.get("drafted_players", []),
            normalized_name,
        )

    # Remove related my_pick entries from draft log if the player is being made available.
    # This keeps the draft log aligned with the corrected state.
    if make_available:
        st.session_state["draft_log"] = [
            entry
            for entry in st.session_state.get("draft_log", [])
            if not (
                _normalize_player_name(entry.get("player_name")).lower()
                == normalized_name.lower()
                and entry.get("action_type") == "my_pick"
            )
        ]


def reset_draft_state() -> None:
    """
    Reset draft-related state while keeping league settings.
    """
    init_session_state()

    st.session_state["drafted_players"] = []
    st.session_state["my_team"] = []
    st.session_state["draft_log"] = []
    st.session_state["current_pick_number"] = DEFAULT_CURRENT_PICK
    st.session_state["player_position_map"] = {}


def get_draft_state_debug_info() -> Dict[str, Any]:
    """
    Optional helper for debug pages or expanders.
    """
    init_session_state()

    return {
        "league_size": st.session_state.get("league_size"),
        "my_draft_slot": st.session_state.get("my_draft_slot"),
        "current_pick_number": st.session_state.get("current_pick_number"),
        "drafted_players_count": len(st.session_state.get("drafted_players", [])),
        "my_team_count": len(st.session_state.get("my_team", [])),
        "draft_log_count": len(st.session_state.get("draft_log", [])),
        "drafted_players": st.session_state.get("drafted_players", []),
        "my_team": st.session_state.get("my_team", []),
    }
