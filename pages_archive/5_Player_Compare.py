import streamlit as st
from utils import (
    init_session_state,
    keep_session_state_alive,
    render_league_settings_sidebar,
    get_available_players_df,
    safe_col,
    build_player_comparison_df,
    draft_player
)

st.set_page_config(page_title="Player Compare", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Player Compare")

available_df = get_available_players_df()

player_col = safe_col(available_df, ["player_name", "Player", "player"])
pos_col = safe_col(available_df, ["position", "pos", "Pos", "Position"])

filter_cols = st.columns(2)

with filter_cols[0]:
    position_filter = st.selectbox(
        "Filter by position",
        ["All", "QB", "RB", "WR", "TE"]
    )

filtered_df = available_df.copy()
if pos_col is not None and position_filter != "All":
    filtered_df = filtered_df[filtered_df[pos_col].astype(str) == position_filter]

player_options = filtered_df[player_col].astype(str).dropna().tolist() if player_col else []

selected_players = st.multiselect(
    "Choose 2 to 4 players",
    options=player_options,
    max_selections=4
)

if len(selected_players) == 0:
    st.info("Select players to compare.")
    st.stop()

compare_df = build_player_comparison_df(selected_players)

st.subheader("Comparison Table")
st.dataframe(compare_df, use_container_width=True, hide_index=True)

if not compare_df.empty:
    st.subheader("Quick View")

    cards = st.columns(min(len(compare_df), 4))

    for i, (_, row) in enumerate(compare_df.iterrows()):
        with cards[i]:
            with st.container(border=True):
                st.metric("Player", row["Player"])
                st.write(f"Position: {row['Position']}")
                st.write(f"Team: {row['Team']}")
                st.write(f"Projection: {row['Projection']}")
                st.write(f"ADP: {row['ADP']}")
                st.write(f"Fit: {row['Best Fit Score']}")
                st.write(f"Value: {row['Best Value Score']}")

                if st.button("Draft", key=f"compare_draft_{row['Player']}", use_container_width=True):
                    drafted = draft_player(row["Player"])
                    if drafted:
                        st.rerun()
