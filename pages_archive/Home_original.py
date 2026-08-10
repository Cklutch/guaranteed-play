import streamlit as st
from draftkit.draft_state import init_session_state, keep_session_state_alive
from draftkit.ui_helpers import render_league_settings_sidebar

st.set_page_config(page_title="Fantasy Draft App", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Fantasy Draft App")
st.write("Use the sidebar to configure league settings and open a page from the Pages menu.")
