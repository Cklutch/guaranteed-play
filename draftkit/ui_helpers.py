import streamlit as st

from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_draft_slot,
    set_current_pick_number,
    advance_pick,
    undo_pick,
    reset_draft_state,
    get_my_team_positions,
    get_player_position_map,
)


def render_league_settings_sidebar():
    with st.sidebar:
        st.header("Draft Controls")

        st.number_input(
            "League Size",
            min_value=2,
            max_value=20,
            key="league_size",
            help="Total number of teams in the league.",
        )

        st.number_input(
            "My Draft Slot",
            min_value=1,
            max_value=max(2, int(st.session_state.get("league_size", 12))),
            key="my_draft_slot",
            help="Your position in the draft order.",
        )

        if "draft_pick_number" not in st.session_state:
            st.session_state["draft_pick_number"] = 1

        st.number_input(
            "Current Pick",
            min_value=1,
            step=1,
            key="draft_pick_number",
            help="Overall draft pick currently on the clock.",
        )

        st.divider()
        st.subheader("Roster Settings")

        roster = st.session_state.get("roster_settings", {})

        qb = st.number_input("QB starters", min_value=0, max_value=4, value=int(roster.get("QB", 1)), step=1)
        rb = st.number_input("RB starters", min_value=0, max_value=6, value=int(roster.get("RB", 2)), step=1)
        wr = st.number_input("WR starters", min_value=0, max_value=6, value=int(roster.get("WR", 2)), step=1)
        te = st.number_input("TE starters", min_value=0, max_value=3, value=int(roster.get("TE", 1)), step=1)
        flex = st.number_input("FLEX starters", min_value=0, max_value=4, value=int(roster.get("FLEX", 1)), step=1)
        bench = st.number_input("Bench spots", min_value=0, max_value=12, value=int(roster.get("BENCH", 6)), step=1)

        st.session_state["roster_settings"] = {
            "QB": int(qb),
            "RB": int(rb),
            "WR": int(wr),
            "TE": int(te),
            "FLEX": int(flex),
            "BENCH": int(bench),
        }

        st.divider()
        st.subheader("Draft Navigation")

        nav1, nav2 = st.columns(2)

        with nav1:
            if st.button("Undo Pick", use_container_width=True):
                undo_pick()

        with nav2:
            if st.button("Advance Pick", use_container_width=True):
                advance_pick()

        if st.button("Reset Draft", use_container_width=True, type="secondary"):
            reset_draft_state()

        st.divider()
        st.subheader("League Snapshot")

        st.caption(f"League size: {get_league_size()}")
        st.caption(f"My slot: {get_my_draft_slot()}")
        st.caption(f"Current pick: {get_current_pick_number()}")
        st.caption(f"Drafted players tracked: {len(st.session_state.get('drafted_players', []))}")
        st.caption(f"My team count: {len(st.session_state.get('my_team', []))}")

        st.divider()
        st.subheader("My Team")

        my_team = st.session_state.get("my_team", [])
        player_positions = get_player_position_map()

        if not my_team:
            st.caption("No players on My Team yet.")
        else:
            for i, player in enumerate(my_team):
                pos = player_positions.get(player, "Unknown")
                st.write(f"{i + 1}. {player} ({pos})")

