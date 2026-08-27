"""News override resolve queue -- light in-app replacement for
hand-editing research/pending_news_adjustments.md. The scheduled
nfl-camp-news-watch sweep drops Lane B (human-review) findings into
research/pending_news_adjustments.json; this page renders each as a
card with one button. See research/news_override_policy.md for the
judgment-call rules the sweep and this page both follow.

Only `injury_score` entries can be applied directly here -- see
draftkit/news_queue.py's docstring for why a `projection_pct` entry is
flagged for a manual code edit instead.
"""

import streamlit as st

from draftkit.draft_state import init_session_state
from draftkit.news_queue import apply_injury_override, dismiss_entry, load_queue, save_queue
from draftkit.ui_helpers import render_tool_nav

st.set_page_config(page_title="News Queue", layout="wide", initial_sidebar_state="collapsed")

# Same dark ground as the rest of the app -- kept minimal, this is a
# utility page, not another full design pass.
st.markdown(
    """
    <style>
      .stApp, header[data-testid="stHeader"] { background: #0d0f10 !important; }
      section[data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
      .block-container { padding-top: 1.5rem; max-width: 760px; }
      body, .stApp p, .stApp div, .stApp span { color: #eef1ec; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: #141718; border-color: #23282b !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session_state()
render_tool_nav("News Queue")

st.title("News override queue")
st.caption("Lane B findings awaiting a decision. Full policy: research/news_override_policy.md")

queue = load_queue()

if not queue:
    st.success("Queue is clear — nothing waiting on you.")
else:
    for entry in list(queue):
        with st.container(border=True):
            top = st.columns([3, 1])
            top[0].subheader(entry.get("player", "Unknown player"))
            mechanism = entry.get("mechanism", "injury_score")
            top[1].caption(f"{entry.get('date', '')}")

            st.caption(
                f"**{mechanism}** = {entry.get('value', '—')}  ·  "
                f"confidence: {entry.get('confidence', '—')}"
            )
            st.write(entry.get("reason", ""))
            if entry.get("source"):
                st.caption(f"Source: {entry['source']}")

            c1, c2 = st.columns(2)
            if mechanism == "injury_score":
                if c1.button("Apply", key=f"apply_{entry['player']}", type="primary"):
                    before, after = apply_injury_override(entry)
                    queue.remove(entry)
                    save_queue(queue)
                    st.success(f"Applied. risk_index {before} → {after}.")
                    st.rerun()
            elif mechanism == "projection_pct":
                c1.warning("Projection cut — needs a code edit. Ask Claude to apply this one.")
            else:
                c1.caption("Informational — nothing to apply.")

            if c2.button("Dismiss", key=f"dismiss_{entry['player']}"):
                queue.remove(entry)
                save_queue(queue)
                dismiss_entry(entry)
                st.info("Dismissed and logged.")
                st.rerun()
