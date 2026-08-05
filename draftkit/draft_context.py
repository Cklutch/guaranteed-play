from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from draftkit.common import safe_int
from draftkit.data_access import get_available_players_df, load_players_df
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_draft_slot,
    get_my_team_positions,
    init_session_state,
)


@dataclass(frozen=True)
class RosterProfile:
    my_team: List[str]
    position_counts: Dict[str, int]
    roster_settings: Dict[str, Any]

    @property
    def total_players(self) -> int:
        return len(self.my_team)

    def count(self, position: str) -> int:
        return safe_int(self.position_counts.get(str(position).upper()), 0)


@dataclass(frozen=True)
class DraftContext:
    raw_players_df: pd.DataFrame
    available_players_df: pd.DataFrame
    roster: RosterProfile
    current_pick: int
    league_size: int
    draft_slot: int
    drafted_players: List[str]
    draft_log: List[Dict[str, Any]]
    recommendation_history: List[Dict[str, Any]]
    recommendations_df: Optional[pd.DataFrame] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def round_number(self) -> int:
        if self.league_size <= 0:
            return 1
        return ((self.current_pick - 1) // self.league_size) + 1


def build_roster_profile(players_df: Optional[pd.DataFrame] = None) -> RosterProfile:
    init_session_state()
    players_df = players_df if players_df is not None else load_players_df()
    return RosterProfile(
        my_team=list(st.session_state.get("my_team", [])),
        position_counts=get_my_team_positions(players_df),
        roster_settings=dict(st.session_state.get("roster_settings", {})),
    )


def get_draft_context(
    include_recommendations: bool = False,
    recommendations_df: Optional[pd.DataFrame] = None,
) -> DraftContext:
    """
    Build a read-only snapshot of draft state and player data for engines/pages.
    """
    init_session_state()
    raw_players_df = load_players_df()
    available_players_df = get_available_players_df()

    if include_recommendations and recommendations_df is None:
        from draftkit.draft_analysis import build_recommendation_rankings_df

        recommendations_df = build_recommendation_rankings_df()

    return DraftContext(
        raw_players_df=raw_players_df,
        available_players_df=available_players_df,
        roster=build_roster_profile(raw_players_df),
        current_pick=get_current_pick_number(),
        league_size=get_league_size(),
        draft_slot=get_my_draft_slot(),
        drafted_players=list(st.session_state.get("drafted_players", [])),
        draft_log=list(st.session_state.get("draft_log", [])),
        recommendation_history=list(st.session_state.get("recommendation_history", [])),
        recommendations_df=recommendations_df,
    )


def get_draft_context_debug_info(context: Optional[DraftContext] = None) -> Dict[str, Any]:
    context = context or get_draft_context()
    return {
        "raw_player_rows": int(len(context.raw_players_df)),
        "available_player_rows": int(len(context.available_players_df)),
        "current_pick": context.current_pick,
        "round_number": context.round_number,
        "league_size": context.league_size,
        "draft_slot": context.draft_slot,
        "my_team_count": context.roster.total_players,
        "position_counts": context.roster.position_counts,
        "drafted_players_count": len(context.drafted_players),
        "draft_log_count": len(context.draft_log),
        "recommendation_history_count": len(context.recommendation_history),
        "recommendation_rows": (
            int(len(context.recommendations_df))
            if context.recommendations_df is not None
            else None
        ),
    }
