"""Draft Command Center -- live in-draft advisor.

Syncs to a Sleeper draft (real or mock) and, after every pick, recomputes
the best available recommendations, highest tier remaining, roster needs,
survival odds, and a round-by-round plan -- rendered as the 3-column
command center from the design handoff (draftkit/draft_center.py), in the
same server-rendered-HTML style as the rankings board.

Sleeper-driven: paste the draft link and (optionally) your slot; auto-
refresh re-polls the live pick feed. The heavy computation lives in
draftkit/{live_draft,sleeper,draft_center}.py; this page is just wiring.
"""

import streamlit as st

from draftkit.draft_state import init_session_state
from draftkit.live_draft import build_candidate_pool
from draftkit.sleeper import parse_draft_id, fetch_picks, summarize_picks, draft_info
from draftkit.draft_center import render_command_center

st.set_page_config(
    page_title="Draft Command Center", layout="wide", initial_sidebar_state="collapsed"
)

DCC_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  .stApp, header[data-testid="stHeader"] { background: #0d0f10 !important; }
  /* Sleeper-driven page -- no sidebar; hide it and the default nav list. */
  section[data-testid="stSidebar"], [data-testid="stSidebarNav"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stExpandSidebarButton"] { display: none !important; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
  body, .stApp p, .stApp div { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
  /* Only non-icon spans -- the broad override otherwise forces Streamlit's
     Material icon spans (expander caret, etc.) off their ligature font,
     leaking raw text like "keyboard_arrow_right". */
  .stApp span:not([data-testid="stIconMaterial"]) { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
  .dcc-scroll::-webkit-scrollbar { height: 8px; width: 8px; }
  .dcc-scroll::-webkit-scrollbar-thumb { background: #2a3033; border-radius: 4px; }
  div[data-testid="stTextInput"] input { background: #141718; border: 1px solid #23282b; color: #eef1ec; }
  div[data-testid="stSelectbox"] { max-width: 260px; }
  /* Clear, sage back-to-rankings link -- styled as a visible button. */
  [data-testid="stPageLink"] { margin-bottom: 6px; }
  [data-testid="stPageLink"] a, a[data-testid="stPageLink-NavLink"] {
    color: #a8c686 !important; font-weight: 700; font-size: 13px;
    background: rgba(168,198,134,.08); padding: 8px 16px; border-radius: 8px;
    border: 1px solid rgba(168,198,134,.2); display: inline-block;
  }
  [data-testid="stPageLink"] a:hover { color: #c3dba6 !important; background: rgba(168,198,134,.15); }
</style>
"""
st.markdown(DCC_CSS, unsafe_allow_html=True)

# Explicit, always-visible way back to the rankings board.
st.page_link("Home.py", label="← Rankings Board")
init_session_state()


@st.cache_data(show_spinner="Scoring draft board...", max_entries=4)
def _pool(score_col):
    return build_candidate_pool(score_col=score_col)


@st.cache_data(ttl=20, show_spinner=False)
def _draft_info(draft_id):
    return draft_info(draft_id)


# --- controls: paste the link, everything else follows -------------------
st.text_input(
    "Sleeper draft link", key="sleeper_draft_id",
    placeholder="Paste your Sleeper draft URL or ID to sync a live draft",
)
_c1, _c2, _c3 = st.columns([1, 1, 2])
_c1.number_input(
    "My draft slot", min_value=0, max_value=32, value=0, step=1, key="sleeper_my_slot",
    help="Your seat (1 = first overall). 0 = don't track my team / needs / byes.",
)
_c2.checkbox("Auto-refresh", value=True, key="sleeper_auto", help="Re-poll Sleeper every 5s.")
_c3.radio(
    "Scoring model", ["Standard (Half-PPR)", "Dad's League"],
    horizontal=True, key="live_score_mode",
)
st.text_input(
    "Compare a player", key="compare_player",
    placeholder="Type a player name to compare against the model's recommendations",
)

# Poll only when there's actually a live draft to poll.
_poll = bool(
    st.session_state.get("sleeper_auto")
    and parse_draft_id(st.session_state.get("sleeper_draft_id", ""))
)


@st.fragment(run_every=(5 if _poll else None))
def _command_center():
    score_col = (
        "dads_final_score"
        if str(st.session_state.get("live_score_mode", "")).startswith("Dad")
        else "final_score"
    )
    scoring_label = "Dad's League" if score_col == "dads_final_score" else "Half-PPR"
    pool = _pool(score_col)
    if pool.empty:
        st.info("Draft board unavailable for the selected scoring model.")
        return

    draft_id = parse_draft_id(st.session_state.get("sleeper_draft_id", ""))
    my_slot = int(st.session_state.get("sleeper_my_slot") or 0) or None

    drafted, my_team, drafted_by, num_picks, teams, code, complete = [], [], {}, 0, 12, None, False
    if draft_id:
        try:
            picks = fetch_picks(draft_id)
            summ = summarize_picks(picks, pool, my_slot=my_slot)
            drafted, my_team = summ["drafted"], summ["my_team"]
            drafted_by, num_picks = summ["drafted_by"], summ["num_picks"]
            info = _draft_info(draft_id)
            teams = info["teams"]
            complete = info["status"] == "complete"
            code = draft_id[-6:]
        except Exception as exc:  # keep the board usable if Sleeper hiccups
            st.warning(f"Couldn't reach Sleeper draft {draft_id}: {exc}")

    compare_query = st.session_state.get("compare_player", "").strip() or None
    html = render_command_center(
        pool, drafted, my_team, drafted_by, num_picks, teams, my_slot, scoring_label, code,
        complete=complete, compare_player=compare_query,
    )
    st.markdown(html, unsafe_allow_html=True)


_command_center()
