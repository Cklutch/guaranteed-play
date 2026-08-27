"""News override resolve queue -- light in-app replacement for
hand-editing research/pending_news_adjustments.md. The scheduled
nfl-camp-news-watch sweep drops Lane B (human-review) findings into
research/pending_news_adjustments.json; this page renders each as a
card with one button. See research/news_override_policy.md for the
judgment-call rules the sweep and this page both follow.

Both `injury_score` and `projection_pct` entries can be applied directly
here (2026-08-27) -- the value shown is editable before applying, so a
proposal can be approved as-is or revised right on the card, not just
accepted or rejected. See draftkit/news_queue.py's docstring for which
data file a projection_pct edit actually lands in.

A `projection_pct` card also shows a live positional-rank preview (e.g.
"RB #50 -> RB #48") as the number is edited, using
preview_projection_rank() -- the exact same base-points computation
Apply itself uses, so the preview can't show a different outcome than
clicking Apply actually produces.
"""

import streamlit as st

from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import init_session_state
from draftkit.news_queue import (
    apply_injury_override, apply_projection_override, dismiss_entry,
    load_queue, preview_projection_rank, save_queue,
)
from draftkit.ui_helpers import render_tool_nav


@st.cache_data(show_spinner=False, ttl=300)
def _board():
    return build_recommendation_rankings_df()

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

            st.caption(f"confidence: {entry.get('confidence', '—')}")
            st.write(entry.get("reason", ""))
            if entry.get("source"):
                st.caption(f"Source: {entry['source']}")

            if mechanism in ("injury_score", "projection_pct"):
                label = "injury_score" if mechanism == "injury_score" else "projection change (%)"
                step = 0.1 if mechanism == "injury_score" else 0.5
                revised = st.number_input(
                    label, value=float(entry.get("value", 0.0)), step=step,
                    key=f"value_{entry['player']}",
                    help="Edit before applying to revise the proposal, or leave as-is to approve it.",
                )

            if mechanism == "projection_pct":
                preview = preview_projection_rank(entry, revised, _board())
                if preview:
                    pos = preview["position"]
                    cur, new, total = preview["current_rank"], preview["new_rank"], preview["total"]
                    if new < cur:
                        arrow, color = "↑", "#7fc98a"
                    elif new > cur:
                        arrow, color = "↓", "#e08a7f"
                    else:
                        arrow, color = "→", "#8c948f"
                    st.markdown(
                        f'<div style="font:600 13px \'IBM Plex Mono\',monospace;color:{color};margin:4px 0 2px">'
                        f'{pos} #{cur} {arrow} {pos} #{new} <span style="color:#6f776f;font-weight:400">'
                        f'(of {total} · {preview["current_pts"]:.1f} → {preview["new_pts"]:.1f} pts)</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No rank preview available (player not found on the current board).")

            c1, c2 = st.columns(2)
            if mechanism == "injury_score":
                if c1.button("Apply", key=f"apply_{entry['player']}", type="primary"):
                    before, after = apply_injury_override(entry, score=revised)
                    queue.remove(entry)
                    save_queue(queue)
                    st.success(f"Applied. risk_index {before} → {after}.")
                    st.rerun()
            elif mechanism == "projection_pct":
                if c1.button("Apply", key=f"apply_{entry['player']}", type="primary"):
                    before, after, layer = apply_projection_override(entry, pct=revised)
                    queue.remove(entry)
                    save_queue(queue)
                    st.success(f"Applied via {layer}. {before} → {after} pts.")
                    st.rerun()
            else:
                c1.caption("Informational — nothing to apply.")

            if c2.button("Dismiss", key=f"dismiss_{entry['player']}"):
                queue.remove(entry)
                save_queue(queue)
                dismiss_entry(entry)
                st.info("Dismissed and logged.")
                st.rerun()
