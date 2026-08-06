import streamlit as st

from draftkit.data_access import get_available_players_df
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import (
    get_current_pick_number,
    handle_my_pick,
    init_session_state,
    keep_session_state_alive,
)
from draftkit.ui_helpers import render_league_settings_sidebar

# Migrated off the legacy utils.py layer. Previously this page called
# utils.build_live_rankings_df() (Best Fit / Best Value / Best Available --
# simple min-max blends of projection and ADP) against utils.py's own sqlite
# read of guaranteed_play.db, which is a *different and stale* dataset from
# the one the rest of the app uses. That's why this page showed numbers that
# disagreed with Draft Mode. It now reads the same real
# data/processed/master_players.csv and runs the same scoring engine.
#
# It also used utils.py's session-state schema (draft_pick_number,
# draft_history, user_draft_slot), which is NOT the schema draftkit uses
# (current_pick_number, draft_log, my_draft_slot) -- so a pick made here
# never showed up in Draft Mode. Now on draftkit.draft_state throughout.

st.set_page_config(page_title="Live Rankings", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Live Rankings")
st.caption(
    "Same engine and same player data as Draft Mode. Picks made here are "
    "reflected everywhere in the app."
)

top1, top2 = st.columns(2)
top1.metric("Current Pick", get_current_pick_number())
top2.metric("Players Drafted", len(st.session_state.get("drafted_players", [])))

SORT_OPTIONS = {
    "Model Score": "final_score",
    "Position Value": "position_value_score",
    "ADP Value": "value_score",
    "Tier Urgency": "urgency_score",
    "Projection": "projection_points",
    "ADP": "adp",
}

filter_cols = st.columns(3)

with filter_cols[0]:
    position_filter = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE"])

with filter_cols[1]:
    sort_by = st.selectbox("Sort by", list(SORT_OPTIONS.keys()))

with filter_cols[2]:
    rows_to_show = st.selectbox("Rows", [25, 50, 100, 150], index=1)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_rankings(drafted_tuple):
    # Keyed on the drafted-player tuple so the board refreshes after a pick
    # but doesn't recompute on every unrelated widget change.
    return build_recommendation_rankings_df()


rankings_df = _cached_rankings(tuple(st.session_state.get("drafted_players", [])))

if rankings_df.empty:
    st.info("No rankings available. Check the Component Audit page for data diagnostics.")
    st.stop()

if position_filter != "All":
    rankings_df = rankings_df[rankings_df["position"].astype(str).str.upper() == position_filter]

if rankings_df.empty:
    st.info(f"No available {position_filter} players.")
    st.stop()

sort_col = SORT_OPTIONS[sort_by]
if sort_col not in rankings_df.columns:
    st.warning(f"'{sort_by}' is unavailable for the current data; falling back to Model Score.")
    sort_col = "final_score"

# ADP is "lower is better"; every other sort option is "higher is better".
ascending = sort_col == "adp"
rankings_df = rankings_df.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

# Flag rows whose projection is a replacement-level fallback rather than a
# real projection, so a player without real FantasyPros data isn't silently
# presented as if he had one (see Phase 0's projection_source column).
if "projection_source" in rankings_df.columns:
    fallback_count = int((rankings_df["projection_source"] == "replacement_fallback").sum())
    if fallback_count:
        st.caption(
            f"{fallback_count} of {len(rankings_df)} players have no real projection "
            "and are scored at replacement level (shown in the Proj Source column)."
        )

DISPLAY_COLUMNS = [
    "player_name", "position", "team", "projection_points", "adp",
    "final_score", "position_value_score", "value_score", "urgency_score",
    "value_tier", "projection_source",
]
display_df = rankings_df[[c for c in DISPLAY_COLUMNS if c in rankings_df.columns]]

st.subheader("Rankings Table")
st.dataframe(display_df.head(rows_to_show), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Top 10 Actions")

for idx, row in rankings_df.head(10).iterrows():
    player_name = str(row["player_name"])

    with st.container(border=True):
        c1, c2 = st.columns([4, 2])

        with c1:
            st.write(f"**{player_name}**")
            projection = row.get("projection_points")
            adp = row.get("adp")
            caption = (
                f"{row.get('position', '?')} - {row.get('team', '?')} - "
                f"Proj: {projection:.1f}" if projection is not None and str(projection) != "nan"
                else f"{row.get('position', '?')} - {row.get('team', '?')} - Proj: n/a"
            )
            caption += f" - ADP: {adp:.1f}" if adp is not None and str(adp) != "nan" else " - ADP: n/a"
            caption += f" - Model Score: {row.get('final_score', float('nan')):.1f}"
            if row.get("projection_source") == "replacement_fallback":
                caption += " - (replacement-level projection)"
            st.caption(caption)

        with c2:
            if st.button("Draft", key=f"live_draft_{idx}_{player_name}", use_container_width=True):
                handle_my_pick(row)
                st.rerun()
