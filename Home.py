"""
Top 250 rankings -- the static preseason board.

This page itself has no navigation, no draft-state controls, and no draft
actions. Draft Mode and News Queue have both been restored to pages/, so
Streamlit's multipage nav is active again; Team Outlook is the one page
still parked in pages_archive/ rather than deleted, and moving it into
pages/ brings it back the same way. (Player Cards, Draft Lab, Tier
Desperation, Player Compare, Live Rankings, and Component Audit were
removed outright -- see CHANGELOG.md.)

Scoring is unchanged -- this reads the same draftkit engine and the same
data/processed/master_players.csv that Draft Mode uses. Because there is no
live draft state here, the board is evaluated at its draft-start position
(empty roster, pick 1), which is what a static preseason ranking list
should show.

Visual design: "Rankings Lab" redesign (2026-08-24), recreated from a
design-canvas handoff (design_handoff_rankings_board/) in this app's own
stack rather than shipped as the handoff's raw HTML/JS. Still 100%
server-rendered HTML via st.markdown(unsafe_allow_html=True) -- no JS. The
one new interaction primitive is native <details>/<summary> for the
click-to-expand player card and its sub-collapsibles: zero JS, and expand
state resetting on every filter change (a full-page rerun) matches the
handoff's own stated behavior rather than fighting it.
"""
import html
import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from draftkit.age_context import age_note
from draftkit.archetypes import archetype_label, risk_profile
from draftkit.data_access import load_players_df
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.news_queue import (
    apply_projection_override, preview_projection_rank, resolve_projection_base,
)
from draftkit.draft_state import init_session_state
from draftkit.ui_helpers import render_tool_nav

TOP_N = 250

POSITION_CHIPS = ["Overall", "QB", "RB", "WR", "TE"]
VIEW_TABS = [("median", "Median"), ("ceiling", "Ceiling"), ("riskAdj", "Risk-adjusted")]
RISK_TABS = [("ALL", "Any risk"), ("low", "Low"), ("mid", "Medium"), ("high", "High")]

# Design tokens -- "Rankings Lab" redesign (design_handoff_rankings_board/
# README's Design Tokens section). Kept as Python constants (not just CSS)
# wherever a color needs to be picked dynamically per row/value.
POSITION_COLORS = {"QB": "#dcc06a", "RB": "#e0947f", "WR": "#7fa8d9", "TE": "#b79ada"}
RISK_COLORS = {"low": "#7fc98a", "mid": "#e0b45e", "high": "#e08a7f"}
RISK_LABELS = {"low": "Low", "mid": "Medium", "high": "High"}
GRADE_COLORS = {
    "Elite": "#7fc98a", "Great": "#7fa8d9", "Strong": "#dcc06a",
    "Solid": "#b79ada", "Speculative": "#b79ada",
}
# Tier band colors, cycled -- recolored to the new sage-led palette above.
# Still cycled rather than one-per-tier: TARGET_TIER_COUNT=10 tiers, more
# than there are genuinely distinct semantic colors to spend on them.
TIER_BAND_COLORS = ["#a8c686", "#7fa8d9", "#dcc06a", "#b79ada", "#e0947f", "#7fc98a"]

# Display-only re-sort penalty for the "Risk-adjusted" view. Uses the richer
# 4-category risk_index (risk_variables.csv) rather than re-deriving from
# the single injury_risk aggregate base_value_risk_component already nets
# out of final_score, so this view is a genuinely different resort, not a
# restatement of Median. Never touches final_score/tiering.
RISK_ADJ_VIEW_PENALTY = 15.0

# Display-only upside bonus for the "Ceiling" view, symmetric with
# RISK_ADJ_VIEW_PENALTY above. There is no floor/ceiling *points*
# projection anywhere in the pipeline (master_players.csv carries the
# columns but they're 100% null), and sorting the cross-position board by
# raw ceiling points would just float every QB to the top anyway. So the
# Ceiling view re-sorts final_score (already VOR/position-fair) plus a
# variance-driven bonus: TD-dependent, high-volatility players -- the ones
# whose 90th-percentile outcome most exceeds their median -- rise. Never
# touches final_score/tiering.
CEILING_VIEW_BONUS = 15.0


def _position_rank_number(series):
    """
    "RB7" -> 7, as a real number so the column sorts numerically.

    These arrive from the comparison CSV as text labels ("RB7"), and
    Streamlit's column-header sort compares text lexicographically -- RB10
    lands before RB2. Zero/space padding fixes the sort but corrupts the
    display, and an ordered Categorical is ignored by the client-side sort.
    Dropping the prefix loses nothing: the Pos column already shows RB.
    """
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    ).astype("Int64")


RANKINGS_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
  .stApp, .block-container { background: #0d0f10 !important; }
  /* Dark-match Streamlit's own chrome so the page reads as one surface.
     Without a dark theme config these default to the light theme (white
     top toolbar, light-gray sidebar) and frame the dark board with
     mismatched edges. Belt-and-suspenders with .streamlit/config.toml,
     which is the canonical fix but only applies on server restart. */
  header[data-testid="stHeader"] { background: #0d0f10 !important; }
  [data-testid="stToolbar"] { background: transparent !important; }
  /* Rankings has no sidebar content -- navigation is the top-of-page tool
     dropdown (render_tool_nav) -- so hide Streamlit's sidebar and its
     collapsed-state expander entirely rather than showing an empty rail. */
  section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
  /* Tighten the dropdown so it reads as a compact tool switcher. */
  div[data-testid="stSelectbox"] { max-width: 260px; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 100%; }
  /* :not([data-testid="stIconMaterial"]) on the span rule (2026-08-27, found
     while diagnosing the new "Adjust a player" popover's glitchy overlapping
     label): this used to apply to EVERY span including Streamlit's own
     Material Symbols icon glyphs (expand_more, keyboard_double_arrow_right,
     etc.) -- IBM Plex Sans has no glyph for those ligature names, so the
     browser fell back to rendering the literal text "expand_more" instead
     of the chevron icon, wrapping awkwardly inside a button sized for a
     single glyph. Pre-existing app-wide (the sidebar's collapse arrow had
     the same issue), just more visible on the new popover's small button. */
  body, .stApp, .stApp p, .stApp div,
  .stApp span:not([data-testid="stIconMaterial"]) { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

  /* ---- Header bar ---- */
  .gp-header {
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    flex-wrap: wrap; padding: 4px 4px 14px; border-bottom: 1px solid #23282b; margin-bottom: 18px;
  }
  .gp-header-left { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .gp-logo {
    width: 26px; height: 26px; border-radius: 6px; background: #123d22; border: 1px solid #1f5c33;
    display: inline-flex; align-items: center; justify-content: center;
    font: 800 12px Archivo, sans-serif; color: #a8c686;
  }
  .gp-wordmark { font: 700 13px Archivo, sans-serif; letter-spacing: -.01em; color: #eef1ec; }
  .gp-nav { display: flex; gap: 20px; margin-left: 6px; font: 500 12.5px 'IBM Plex Sans', sans-serif; }
  .gp-nav a, .gp-nav span { color: #8c948f; text-decoration: none; }
  .gp-nav a:hover { color: #eef1ec; }
  .gp-nav .active { color: #eef1ec; padding-bottom: 2px; border-bottom: 2px solid #a8c686; }
  .gp-chips { display: flex; align-items: center; gap: 10px; font: 500 11.5px 'IBM Plex Sans', sans-serif; color: #8c948f; }
  .gp-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border: 1px solid #23282b; border-radius: 99px; }
  .gp-chip-dot { width: 5px; height: 5px; border-radius: 99px; background: #a8c686; }

  /* ---- Title block ---- */
  .gp-eyebrow {
    font: 600 10px 'IBM Plex Mono', monospace; letter-spacing: .2em; color: #a8c686; text-transform: uppercase;
  }
  h1.gp-title {
    color: #f4f6f2 !important; font-family: Archivo, sans-serif !important; font-size: 32px !important;
    font-weight: 800 !important; letter-spacing: -.03em !important; line-height: 1.05 !important;
    margin: 8px 0 0 !important; padding: 0 !important;
  }
  .gp-subhead { margin: 9px 0 0; font: 400 13px/1.6 'IBM Plex Sans', sans-serif; color: #8c948f; max-width: 600px; }
  .gp-stamps {
    display: flex; gap: 18px; flex-wrap: wrap; padding: 14px 0 0;
    font: 400 11px 'IBM Plex Mono', monospace; color: #6f776f; letter-spacing: .03em;
  }
  .gp-stamps b { color: #b9c1ba; font-weight: 500; }
  .gp-title-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap; padding-bottom: 18px; border-bottom: 1px solid #23282b; }
  .gp-export-caption { font: 400 11px 'IBM Plex Sans', sans-serif; color: #6f776f; text-align: right; margin-top: 6px; }

  /* ---- Streamlit widgets restyled as design-token pills/tabs ---- */
  div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
    border-radius: 8px; font: 600 12.5px 'IBM Plex Sans', sans-serif;
    padding: 8px 14px; border: 1px solid #23282b; background: #141718; color: #8c948f;
  }
  div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {
    border-color: #3d4549; color: #eef1ec;
  }
  div[data-testid="stButton"] button[kind="primary"], div[data-testid="stDownloadButton"] button[kind="primary"] {
    background: #123d22; border-color: #1f5c33; color: #cfe4b4;
  }
  div[data-testid="stButton"] button[kind="primary"]:hover { background: #17512d; }
  div[data-testid="stTextInput"] input {
    background: #141718; border: 1px solid #23282b; border-radius: 8px; color: #eef1ec;
    font: 400 12.5px 'IBM Plex Sans', sans-serif;
  }
  .gp-view-note { font: 400 11.5px 'IBM Plex Sans', sans-serif; color: #6f776f; padding-top: 8px; }
  .gp-count { color: #6f776f; font: 500 11.5px 'IBM Plex Mono', monospace; text-align: right; padding-top: 10px; }

  /* ---- Table ---- */
  .gp-table-wrap {
    background: #101314; border: 1px solid #23282b; border-radius: 12px;
    overflow-x: auto; margin-top: 16px;
  }
  .gp-table-inner { min-width: 1220px; }
  .gp-row-grid {
    display: grid;
    grid-template-columns: 56px minmax(220px,1.2fr) 74px 84px 92px 128px 108px 104px 104px 44px;
    align-items: center; gap: 12px;
  }
  .gp-thead {
    height: 38px; padding: 0 20px; background: #131718; border-bottom: 1px solid #23282b;
    font: 600 9.5px 'IBM Plex Mono', monospace; letter-spacing: .13em; color: #7c847e; text-transform: uppercase;
  }
  .gp-thead > div:not(:first-child):not(:nth-child(2)):not(:nth-child(3)) { text-align: right; }

  .tier-band {
    display: flex; align-items: baseline; gap: 12px; padding: 10px 20px;
    background: #121516; border-top: 1px solid #1c2022; border-bottom: 1px solid #1c2022;
  }
  .tier-band-bar { width: 3px; height: 12px; border-radius: 2px; align-self: center; }
  .tier-band-label { font: 700 10.5px 'IBM Plex Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
  .tier-band-desc { font: 400 11.5px 'IBM Plex Sans', sans-serif; color: #78807a; }

  details.gp-row { border-bottom: 1px solid #1a1e1f; }
  details.gp-row > summary {
    list-style: none; cursor: pointer; padding: 11px 20px;
  }
  details.gp-row > summary::-webkit-details-marker { display: none; }
  details.gp-row > summary:hover { background: rgba(168,198,134,.04); }
  details.gp-row[open] > summary { background: rgba(168,198,134,.035); }
  .gp-cell-rank { font: 600 15px 'IBM Plex Mono', monospace; color: #8c948f; }
  .gp-cell-player { display: flex; align-items: center; gap: 11px; min-width: 0; }
  .gp-avatar {
    position: relative; overflow: hidden;
    width: 34px; height: 34px; border-radius: 99px; background: #1d2124; border: 1px solid #2a3033;
    flex-shrink: 0; display: flex; align-items: center; justify-content: center;
    font: 600 11px 'IBM Plex Mono', monospace; color: #6f776f;
  }
  /* Initials sit underneath; the headshot overlay paints on top of them
     only when the Sleeper CDN returns a real image (a 403 leaves the
     overlay transparent, so the initials show through). */
  .gp-avatar-ini {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  }
  .gp-avatar-img {
    position: absolute; inset: 0; border-radius: 99px;
    background-size: cover; background-position: center top; background-repeat: no-repeat;
  }
  .gp-avatar-lg { width: 52px; height: 52px; font-size: 15px; }
  .gp-cell-name { font: 600 14.5px Archivo, sans-serif; letter-spacing: -.01em; color: #f4f6f2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .gp-cell-role { font: 400 11px 'IBM Plex Sans', sans-serif; color: #7c847e; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .gp-cell-pos { display: flex; align-items: center; gap: 6px; }
  .gp-cell-team { font: 500 10.5px 'IBM Plex Mono', monospace; color: #6f776f; }
  .gp-cell-num { text-align: right; font: 500 13px 'IBM Plex Mono', monospace; color: #c9d0c9; }
  .gp-cell-num-strong { color: #eef1ec; }
  .gp-score-wrap { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; }
  .gp-score-num { font: 700 16px Archivo, sans-serif; letter-spacing: -.02em; color: #f4f6f2; }
  .gp-score-of100 { font: 500 10px 'IBM Plex Mono', monospace; color: #6f776f; }
  .gp-score-bar-track { display: block; width: 96px; height: 3px; background: #23282b; border-radius: 2px; }
  .gp-score-bar-fill { display: block; height: 3px; border-radius: 2px; background: #a8c686; }
  .gp-risk-cell { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
  .gp-risk-bars { display: flex; align-items: flex-end; gap: 2px; height: 16px; }
  .gp-risk-bars span { width: 5px; border-radius: 1px; }
  .gp-risk-num { font: 600 12px 'IBM Plex Mono', monospace; width: 22px; text-align: right; }
  .gp-cell-delta { text-align: right; font: 400 11px 'IBM Plex Sans', sans-serif; }
  .gp-caret { display: flex; justify-content: flex-end; color: #7c847e; font: 400 12px 'IBM Plex Mono', monospace; }
  details.gp-row[open] .gp-caret-closed, details.gp-row:not([open]) .gp-caret-open { display: none; }

  /* ---- Player card ---- */
  .gp-card { margin: 4px 20px 26px; background: #141718; border: 1px solid #262b2e; border-radius: 16px; overflow: hidden; }
  .gp-ticker { background: #0b0d0e; border-bottom: 1px solid #23282b; display: flex; flex-wrap: wrap; }
  .gp-ticker-cell {
    display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px;
    font: 600 10.5px 'IBM Plex Mono', monospace; letter-spacing: .06em; color: #6f776f;
    white-space: nowrap; border-right: 1px solid #171b1c;
  }
  .gp-card-head { padding: 16px 20px 14px; display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
  .gp-card-head-left { display: flex; gap: 13px; align-items: center; }
  .gp-card-rank { font: 800 24px Archivo, sans-serif; color: #6f776f; line-height: 1; }
  .gp-card-name { font: 800 19px Archivo, sans-serif; letter-spacing: -.01em; color: #f4f6f2; }
  .gp-card-postteam { font: 400 12.5px 'IBM Plex Sans', sans-serif; color: #8c948f; margin-top: 2px; }
  .gp-tier-pill {
    display: inline-block; font: 700 10px 'IBM Plex Mono', monospace; letter-spacing: .05em;
    padding: 2px 8px; border-radius: 6px; margin-top: 6px; background: rgba(168,198,134,.12);
  }
  .gp-breakout-pill {
    display: inline-block; font: 700 10px 'IBM Plex Mono', monospace; letter-spacing: .05em;
    padding: 2px 8px; border-radius: 6px; margin-top: 6px; margin-left: 6px;
    background: rgba(224,180,94,.16); color: #e0b45e;
  }
  .gp-card-score { text-align: right; }
  .gp-card-score-num { font: 800 32px Archivo, sans-serif; color: #a8c686; line-height: 1; letter-spacing: -.02em; }
  .gp-card-score-lbl { font: 600 9px 'IBM Plex Mono', monospace; color: #6f776f; text-transform: uppercase; letter-spacing: .1em; margin-top: 3px; }
  .gp-card-grade { font: 700 11px 'IBM Plex Sans', sans-serif; margin-top: 4px; }

  .gp-stat3 { display: grid; grid-template-columns: repeat(3,1fr); border-top: 1px solid #23282b; border-bottom: 1px solid #23282b; }
  .gp-stat3-cell { padding: 11px 8px; text-align: center; border-right: 1px solid #23282b; }
  .gp-stat3-cell:last-child { border-right: none; }
  .gp-stat3-num { font: 700 16px 'IBM Plex Mono', monospace; color: #eef1ec; }
  .gp-stat3-lbl { font: 600 9px 'IBM Plex Mono', monospace; color: #6f776f; text-transform: uppercase; letter-spacing: .08em; margin-top: 3px; }

  .gp-callout { margin: 14px 20px 0; padding: 13px 16px; border-radius: 12px; display: flex; align-items: center; gap: 14px; }
  .gp-callout-arrow { font: 800 20px Archivo, sans-serif; flex-shrink: 0; }
  .gp-callout-text { font: 600 13px 'IBM Plex Sans', sans-serif; color: #eef1ec; }
  .gp-callout-sub { font: 400 10.5px 'IBM Plex Sans', sans-serif; color: #8c948f; margin-top: 2px; }

  .gp-verdict-lbl { font: 600 9px 'IBM Plex Mono', monospace; letter-spacing: .16em; color: #a8c686; text-transform: uppercase; }
  .gp-verdict { font: 600 16px/1.4 Archivo, sans-serif; color: #f4f6f2; margin-top: 7px; }
  .gp-why-block { padding: 8px 20px 16px; border-bottom: 1px solid #23282b; }
  .gp-why { font: 400 12.5px/1.55 'IBM Plex Sans', sans-serif; color: #8c948f; }
  .gp-chips-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 11px; }
  .gp-chip-tag {
    font: 400 11px 'IBM Plex Sans', sans-serif; color: #8c948f; background: #1a1e20;
    border: 1px solid #262b2e; border-radius: 7px; padding: 4px 10px;
  }
  .gp-chip-tag b { color: #e6eae4; font-weight: 600; }

  .gp-range-block { padding: 14px 20px; border-bottom: 1px solid #23282b; }
  .gp-range-track { position: relative; height: 6px; background: #1c2022; border-radius: 3px; margin: 12px 0 6px; }
  .gp-range-fill { position: absolute; inset: 0; background: rgba(168,198,134,.3); border-radius: 3px; }
  .gp-range-dot { position: absolute; top: -3px; width: 12px; height: 12px; border-radius: 99px; background: #a8c686; border: 2px solid #141718; }
  .gp-range-labels { display: flex; justify-content: space-between; font: 500 10.5px 'IBM Plex Mono', monospace; color: #7c847e; }

  details.gp-section { border-bottom: 1px solid #23282b; }
  details.gp-section:last-child { border-bottom: none; }
  details.gp-section > summary {
    list-style: none; cursor: pointer; display: flex; justify-content: space-between;
    align-items: center; padding: 12px 20px; font: 700 12px 'IBM Plex Sans', sans-serif; color: #b9c1ba;
  }
  details.gp-section > summary::-webkit-details-marker { display: none; }
  details.gp-section > summary:hover { background: rgba(168,198,134,.04); }
  details.gp-section[open] .sec-caret-closed, details.gp-section:not([open]) .sec-caret-open { display: none; }
  .sec-caret { font: 400 11px 'IBM Plex Mono', monospace; color: #7c847e; }
  .gp-section-body { padding: 14px 20px 16px; background: #111516; }

  .bar-row-grid { display: grid; grid-template-columns: 150px 1fr 46px; align-items: center; gap: 9px; }
  .bar-row-grid.nested { grid-template-columns: 136px 1fr 44px; }
  .bar-row-lbl { font: 400 11.5px 'IBM Plex Sans', sans-serif; color: #8c948f; }
  .bar-track { display: block; height: 7px; background: #1c2022; border-radius: 4px; }
  .bar-track.thin { height: 6px; border-radius: 3px; }
  .bar-fill { display: block; height: 7px; border-radius: 4px; }
  .bar-fill.thin { height: 6px; border-radius: 3px; }
  .bar-val { text-align: right; font: 600 11px 'IBM Plex Mono', monospace; color: #c9d0c9; }
  .nest-group { border-left: 2px solid rgba(168,198,134,.3); padding-left: 13px; display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
  .gp-risk-note {
    margin-top: 12px; padding: 10px 13px; border-radius: 9px; background: rgba(224,180,94,.07);
    border: 1px solid rgba(224,180,94,.22); font: 400 11.5px/1.55 'IBM Plex Sans', sans-serif; color: #c9d0c9;
  }
  .gp-draft-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
  .gp-draft-field { border-left: 2px solid rgba(168,198,134,.5); padding-left: 10px; }
  .gp-draft-field-lbl { font: 600 9px 'IBM Plex Mono', monospace; letter-spacing: .1em; color: #6f776f; text-transform: uppercase; }
  .gp-draft-field-val { font: 400 12px 'IBM Plex Sans', sans-serif; color: #e6eae4; margin-top: 3px; }
  .gp-model-row { font: 400 11px/1.5 'IBM Plex Sans', sans-serif; color: #8c948f; padding: 7px 0; border-bottom: 1px solid #22272a; }
  .gp-model-row b { color: #e6eae4; font-weight: 600; }
  .gp-pending-chip { opacity: .55; font-style: italic; }

  .gp-footer-note { text-align: center; font: 400 11px 'IBM Plex Sans', sans-serif; color: #5f665f; margin-top: 20px; }
</style>
"""


def _fmt(value, decimals=1):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    number = float(value)
    if decimals == 0 or number.is_integer():
        return f"{number:.0f}"
    return f"{number:.{decimals}f}"


RISK_CATEGORY_META = [
    ("injury_score", "Injury"),
    ("role_usage_td_score", "Role/Usage/TD"),
    ("offense_environment_score", "Offense Env"),
    ("schedule_weather_venue_score", "Schedule/Wx"),
]

# Same >66/>34 split as the overall risk_index band (see _risk_band below),
# applied to each category's own 0-100 conversion, so a bar's color always
# means the same thing as the row's risk pill color.
def _risk_bar_color(score100):
    if score100 > 66:
        return RISK_COLORS["high"]
    if score100 > 34:
        return RISK_COLORS["mid"]
    return RISK_COLORS["low"]


def _risk_band(risk_index):
    """low/mid/high classification for a 0-100 risk_index -- same >66/>34
    thresholds _risk_bar_color uses per-category, applied to the overall
    composite. Single source of truth for the row pill, the risk filter
    tabs, and the card's risk label."""
    if risk_index is None or pd.isna(risk_index):
        return None
    v = float(risk_index)
    if v > 66:
        return "high"
    if v > 34:
        return "mid"
    return "low"


def _risk_value(row):
    """Display risk (0-100): the board-wide min-max-scaled riskScaled column
    when present, else the raw risk_index. Mirrors how scoreScaled is the
    display form of final_score for OUR SCORE -- the riskiest player on the
    board reads 100, the safest 0. Every risk *display* (the number, bars,
    band pill/color, and the risk filter) reads through this so they stay
    coherent; the raw risk_index still drives the Risk-adjusted view's sort
    and the per-category bars."""
    scaled = row.get("riskScaled")
    if scaled is not None and pd.notna(scaled):
        return scaled
    return row.get("risk_index")


def _breakout_pill_html(row):
    """WR breakout_score_v1 tag (research/MODEL_REGISTRY.md -- RESEARCH_ONLY,
    backtested and stress-tested but not yet season-proven; see
    score_current_wr_pool_v1.py). Display-only, same as the tier pill next
    to it -- never touches final_score or rank. Title attribute states the
    caveat inline rather than only in a popover, since this pill is visible
    without any hover."""
    if not row.get("is_breakout_v1"):
        return ""
    prob = row.get("breakout_probability_v1")
    prob_str = f"{float(prob) * 100:.0f}%" if pd.notna(prob) else ""
    title = (
        "Backtested breakout-probability model (WR only, research-stage -- "
        f"not yet checked against a real season). Est. {prob_str} chance of beating own ADP by 12+."
        if prob_str else "Backtested breakout-probability model (WR only, research-stage)."
    )
    return f'<span class="gp-breakout-pill" title="{html.escape(title)}">🚀 BREAKOUT{f" {prob_str}" if prob_str else ""}</span>'


def _sleeper_value_badge(sleeper_value_gap):
    """sleeper_value_gap = consensus Average ADP - Sleeper Half ADP (from
    apply_sleeper_adp_overlay.py). Positive = Sleeper drafts him EARLIER
    than the wider market -- a Sleeper-specific reach. Negative = Sleeper
    drafts him LATER -- a Sleeper-specific bargain. Distinct from
    _market_badge above: that one compares the model's own signal (sportsbook
    projection) against ADP; this compares one platform's ADP against a
    cross-platform consensus. Threshold matches _market_badge's +/-3 for
    the same reason (roughly the size of a real, noticeable gap in this
    data, not a formal cutoff)."""
    if sleeper_value_gap is None or pd.isna(sleeper_value_gap):
        return ""
    v = float(sleeper_value_gap)
    if v >= 3:
        return '<span class="sc-badge sc-badge-avoid">Sleeper: Reach vs. consensus</span>'
    if v <= -3:
        return '<span class="sc-badge sc-badge-value">Sleeper: Value vs. consensus</span>'
    return ""


def _market_badge(value_signal_market):
    """Thresholds are a new design decision made for this popover (the
    hover prototype hardcoded badges per example rather than defining a
    rule) -- roughly matching the magnitude of real gaps seen this session
    (e.g. KC Concepcion's -4.0). Sign convention matches the VEGAS vs ADP
    column: positive = market likes him more than ADP (bargain)."""
    if value_signal_market is None or pd.isna(value_signal_market):
        return ""
    v = float(value_signal_market)
    if v <= -3:
        return '<span class="sc-badge sc-badge-avoid">Market: Avoid (overpriced)</span>'
    if v >= 3:
        return '<span class="sc-badge sc-badge-value">Market: Value (underpriced)</span>'
    return '<span class="sc-badge sc-badge-fair">Market: Fair price</span>'


def _chronic_injury_badge(games_missed_json):
    """New rule for this popover, grounded in the v4 season-range data:
    flag 2+ seasons with games_missed >= 6 (moderate-or-worse severity)."""
    if not games_missed_json or pd.isna(games_missed_json):
        return ""
    try:
        games_missed = json.loads(games_missed_json)
    except (TypeError, ValueError):
        return ""
    moderate_or_worse = sum(1 for m in games_missed.values() if m >= 6)
    if moderate_or_worse >= 2:
        return '<span class="sc-badge sc-badge-chronic">⚠ Multiple significant injuries</span>'
    return ""


def _manual_override_badge(note):
    """Distinguishes any manually-forced risk category score (real, dated
    news the computed pipeline can't see yet -- e.g. a hamstring re-injury,
    or a role-certainty change from a real trade) from a normal model
    output. Same badge pattern as _chronic_injury_badge() above, deliberately
    worded "Manual override" so it never reads as just another computed risk
    driver. Shared across every RiskCategory's override note (injury_score,
    role_usage_td_score, ...) rather than one badge function per category."""
    if not note or pd.isna(note):
        return ""
    return f'<span class="sc-badge sc-badge-chronic">⚠ Manual override: {html.escape(str(note))}</span>'


def _risk_note_text(row):
    """Plain-language driver note for the risk profile card section --
    names the primary category and, when both real sub-scores are present,
    the acute/chronic injury split. Same underlying reads as the old hover
    popover, just returned as plain text for the new card body."""
    best_label, best_score100 = None, -1
    for col, label in RISK_CATEGORY_META:
        raw = row.get(col)
        if raw is None or pd.isna(raw):
            continue
        score100 = max(0, min(100, round((float(raw) - 1) / 4 * 100)))
        if score100 > best_score100:
            best_label, best_score100 = label, score100

    note = f"Driven mainly by <b>{html.escape(best_label)}</b> ({best_score100}/100)." if best_label else ""

    acute = row.get("injury_acute_score")
    chronic = row.get("injury_chronic_score")
    if pd.notna(acute) and pd.notna(chronic):
        driver = "recent/acute" if float(acute) >= float(chronic) else "career/chronic pattern"
        note += (
            f" Injury reads as {round((max(float(acute), float(chronic)) - 1) / 4 * 100)}/100, "
            f"driven by the <b>{driver}</b> component -- acute {round((float(acute) - 1) / 4 * 100)}/100, "
            f"chronic {round((float(chronic) - 1) / 4 * 100)}/100."
        )
    return note.strip()


ARCHETYPE_LABELS = {
    "bellcow": "Bellcow",
    "committee_back": "Committee",
    # rookie_projection.py's rb_rookie_projection() emits the coarser
    # pre-draft tier "committee" (not "committee_back") -- a separate
    # vocabulary that was never reconciled with the confirmed 6-way system
    # (see build_rookie_projections.py's docstring). Without this key,
    # ARCHETYPE_LABELS.get() fell through to the raw string, rendering the
    # badge as "Projected: committee" uncapitalized.
    "committee": "Committee",
    # goal_line_specialist/explosive_back/receiving_back removed -- RB
    # primary is a usage tier only (Bellcow/Committee/Handcuff/Unconfirmed),
    # never a role/trait (explicit user correction 2026-08-15). Those real
    # traits still surface as secondary lean tags -- see LEAN_LABELS below,
    # unaffected by this change.
    "handcuff": "Handcuff",
    # WR usage tiers (draftkit/wr_archetypes.py) -- disjoint key set from the
    # RB slugs above, share this same dict since both feed the same
    # ARCHETYPE_LABELS.get(primary, ...) lookup pattern. "unconfirmed" below
    # is genuinely shared -- same meaning, same label, in both systems.
    "alpha": "Alpha",
    # "possession" removed entirely (2026-08-15) -- neither the confirmed
    # real archetype system (claude_code_plan_possession_split.pdf) nor
    # rookie_projection.py's separate pre-draft composite model
    # (explicit user correction, same day: rookies can't show it either)
    # emits it anymore. Nothing reads this key now.
    "boom_bust": "Boom-Bust",
    "high_floor": "High-Floor",
    "complementary": "Complementary",
    "unconfirmed": "Unconfirmed",
    # QB rushing-style tiers (draftkit/qb_archetypes.py) -- disjoint key set,
    # shares this dict. "unconfirmed" above is genuinely shared here too
    # (insufficient real sample -- same meaning as WR's, not RB's "No Role"
    # usage-tier meaning, which is a Home.py-level override, not a slug).
    # Hyphenated (not "Pocket Passer") -- badge_class feeds this label
    # straight into a CSS class name (f"arch-{label}"), same convention
    # Boom-Bust/High-Floor already established; a raw space would silently
    # split into two CSS classes instead of one.
    "pocket_passer": "Pocket-Passer",
    "balanced": "Balanced",
    "dual_threat": "Dual-Threat",
    # TE usage tiers (draftkit/te_archetypes.py) -- "balanced" above is
    # genuinely shared with QB (same meaning: real, in-between usage, not
    # a clean fit for either extreme). "unconfirmed" (shared across every
    # position system) already exists.
    "receiving_te": "Receiving-TE",
    "blocking_te": "Blocking-TE",
}
# TE role_profile (plan_te_role_profile_elite.pdf) -- second-pass split
# within receiving_te only, deliberately NOT merged into ARCHETYPE_LABELS
# above: those values feed straight into a CSS class name (f"arch-{label}"),
# and the user's confirmed wording for this pair ("Elite TE") has a real
# space, unlike every other hyphenated badge label. Kept as its own dict so
# the te_primary render branch can build a hyphen-safe class separately.
TE_ROLE_PROFILE_LABELS = {
    "elite": "Elite TE",
    "complementary": "Complementary TE",
}
DOWN_SPLIT_LABELS = {"2-down": "2-Down", "3-down": "3-Down", "mixed-down": "Mixed-Down"}
LEAN_LABELS = {
    "goal_line_lean": "Goal-Line lean",
    "explosive_lean": "Explosive lean",
    "receiving_lean": "Receiving lean",
    # WR leans (draftkit/wr_archetypes.py) -- disjoint keys, same dict.
    "deep_threat": "Deep-Threat lean",
    "yac": "YAC lean",
    "red_zone": "Red-Zone lean",
    "rushing": "Rushing lean",
}
QB_CONTEXT_LABELS = {
    "elite_qb_context": "Elite QB context",
    "average_qb_context": "Average QB context",
    "below_average_qb_context": "Below-average QB context",
}

# Which real inputs actually drove each archetype's classification --
# shown in the hover popover instead of a fabricated per-archetype fit
# score (this system only computes real pass/fail thresholds, not a
# continuous 0-100 comparative score across all six archetypes).
ARCHETYPE_METRICS = {
    "bellcow": [
        ("opportunity_share", "Backfield Share", "pct"),
        ("third_down_snap_share", "3rd-Down Snaps", "pct"),
        ("games_played", "Games Played", "int"),
    ],
    "committee_back": [
        ("opportunity_share", "Backfield Share", "pct"),
        ("third_down_snap_share", "3rd-Down Snaps", "pct"),
        ("games_played", "Games Played", "int"),
    ],
    # goal_line_specialist/explosive_back/receiving_back removed -- never
    # produced as primary anymore (see ARCHETYPE_LABELS' matching comment).
    "handcuff": [
        ("depth_chart_rank", "Depth Chart Rank", "rank"),
        ("opportunity_share", "Backfield Share", "pct"),
        ("games_played", "Games Played", "int"),
    ],
    "unconfirmed": [
        ("games_played", "Games Played", "int"),
        ("total_touches", "Total Touches", "int"),
        ("opportunity_share", "Backfield Share", "pct"),
    ],
}

# Same idea as ARCHETYPE_METRICS above, for WR usage tiers (real inputs
# behind classify_primary(), see draftkit/wr_archetypes.py). Columns are
# wr_-prefixed on the way into df (see _rankings()'s merge) to avoid
# colliding with rb_archetypes.csv's own generically-named games_played --
# the exact collision class already found and fixed once this session.
WR_ARCHETYPE_METRICS = {
    "alpha": [
        ("wr_target_share", "Target Share", "pct"),
        ("wr_targets", "Targets", "int"),
        ("wr_games_played", "Games Played", "int"),
    ],
    # "possession" removed -- real archetype system no longer produces it
    # (replaced by boom_bust/high_floor below). Unreachable here even for
    # the rookie-projection path, which selects this dict only when a real
    # wr_archetype_primary is on file -- a true rookie never has one.
    "boom_bust": [
        ("wr_target_share", "Target Share", "pct"),
        ("wr_adot", "aDOT", "dec1"),
        ("wr_games_played", "Games Played", "int"),
    ],
    "high_floor": [
        ("wr_target_share", "Target Share", "pct"),
        ("wr_adot", "aDOT", "dec1"),
        ("wr_games_played", "Games Played", "int"),
    ],
    "complementary": [
        ("wr_target_share", "Target Share", "pct"),
        ("wr_team_wr1_target_share", "Team WR1 Share", "pct"),
        ("wr_games_played", "Games Played", "int"),
    ],
    "unconfirmed": [
        ("wr_games_played", "Games Played", "int"),
        ("wr_targets", "Targets", "int"),
        ("wr_target_share", "Target Share", "pct"),
    ],
}

# QB rushing-style tiers (draftkit/qb_archetypes.py). Single-axis taxonomy --
# every tier shows the same 3 real metrics (rushing_fantasy_pct is pooled
# across up to 2 real seasons, see build_qb_archetypes.py, not read from a
# single season).
QB_ARCHETYPE_METRICS = {
    "pocket_passer": [
        ("qb_rushing_fantasy_pct", "Rushing % of Fantasy Pts", "pct"),
        ("qb_attempts", "Pass Attempts (pooled)", "int"),
        ("qb_games", "Games (pooled)", "int"),
    ],
    "balanced": [
        ("qb_rushing_fantasy_pct", "Rushing % of Fantasy Pts", "pct"),
        ("qb_attempts", "Pass Attempts (pooled)", "int"),
        ("qb_games", "Games (pooled)", "int"),
    ],
    "dual_threat": [
        ("qb_rushing_fantasy_pct", "Rushing % of Fantasy Pts", "pct"),
        ("qb_attempts", "Pass Attempts (pooled)", "int"),
        ("qb_games", "Games (pooled)", "int"),
    ],
    "unconfirmed": [
        ("qb_games", "Games (pooled)", "int"),
        ("qb_attempts", "Pass Attempts (pooled)", "int"),
    ],
}

# TE receiving/blocking tiers (plan_te_archetypes.pdf) -- target_share is
# the real primary discriminator, snap_share a real-involvement gate (see
# draftkit/te_archetypes.py's module docstring for the corrected-anchor
# reasoning behind that split).
TE_ARCHETYPE_METRICS = {
    # redzone_target_share included here (only for receiving_te) so the
    # popover discloses the real number driving an Elite TE tag's redzone
    # path, not just the overall target_share -- see
    # draftkit/te_archetypes.py's classify_role_profile().
    "receiving_te": [
        ("te_target_share", "Target Share", "pct"),
        ("te_redzone_target_share", "Redzone Target Share", "pct"),
        ("te_snap_share", "Snap Share", "pct"),
        ("te_games_played", "Games Played", "int"),
    ],
    "balanced": [
        ("te_target_share", "Target Share", "pct"),
        ("te_snap_share", "Snap Share", "pct"),
        ("te_games_played", "Games Played", "int"),
    ],
    "blocking_te": [
        ("te_target_share", "Target Share", "pct"),
        ("te_snap_share", "Snap Share", "pct"),
        ("te_games_played", "Games Played", "int"),
    ],
    "unconfirmed": [
        ("te_games_played", "Games Played", "int"),
        ("te_snap_share", "Snap Share", "pct"),
        ("te_target_share", "Target Share", "pct"),
    ],
}

# Rookie projection component breakdown (draftkit/rookie_projection.py's
# wr_rookie_projection()/rb_rookie_projection()/qb_rookie_projection() --
# each returns a different component subset, e.g. RB has "athletic", WR has
# "breakout_age" instead, QB has "ncaa_passer_rating" instead (no combine
# term -- confirmed absent for QB anywhere in this repo, not fabricated);
# missing components come through as NaN after the merge and are skipped by
# _archetype_metric_value(), same pattern as the RB/WR metric dicts above).
ROOKIE_COMPONENT_METRICS = [
    ("rookie_component_draft_capital", "Draft Capital", "dec1"),
    ("rookie_component_dominator", "College Dominator", "dec1"),
    ("rookie_component_athletic", "Athletic (Speed Score)", "dec1"),
    ("rookie_component_breakout_age", "Breakout Age", "dec1"),
    ("rookie_component_roster_competition", "Roster Competition", "dec1"),
    ("rookie_component_ncaa_passer_rating", "NCAA Passer Rating", "dec1"),
    ("rookie_component_offense_environment", "Offense Environment", "dec1"),
    ("rookie_component_schedule", "Schedule", "dec1"),
]


def _archetype_metric_value(raw, kind: str) -> str | None:
    if raw is None or pd.isna(raw):
        return None
    if kind == "pct":
        return f"{float(raw) * 100:.0f}%"
    if kind == "dec1":
        return f"{float(raw):.1f}"
    if kind == "rank":
        return f"#{int(raw)}"
    return f"{int(raw)}"


def _primary_archetype(row):
    """Whichever position-specific archetype system applies to this row --
    rookie fallback first, then RB/WR/QB/TE's own real primary (same
    precedence the old hover-badge dispatch used) -- returned as data
    (kind, primary_key, label, metrics_dict) rather than badge HTML, so
    both the row's role subtitle and the card's sections can share one
    lookup instead of re-deriving it per section."""
    rookie_display_status = row.get("rookie_display_status")
    if isinstance(rookie_display_status, str) and rookie_display_status in ("projected", "blended"):
        tier = row.get("rookie_display_tag")
        label = ARCHETYPE_LABELS.get(tier, tier or "Unconfirmed")
        if rookie_display_status == "projected":
            label = f"Projected: {label}"
        else:
            weight = row.get("rookie_display_weight")
            pct = round(float(weight) * 100) if pd.notna(weight) else None
            if pct is not None:
                label = f"{label} -- {pct}% confirmed"
        return ("rookie", tier, label, None)

    rb_primary = row.get("rb_archetype_primary")
    if isinstance(rb_primary, str) and rb_primary:
        label = "No Role" if rb_primary == "unconfirmed" else ARCHETYPE_LABELS.get(rb_primary, rb_primary)
        down_split = row.get("rb_down_split")
        if rb_primary in ("bellcow", "committee_back") and isinstance(down_split, str) and down_split in DOWN_SPLIT_LABELS:
            label = f"{label} ({DOWN_SPLIT_LABELS[down_split]})"
        return ("rb", rb_primary, label, ARCHETYPE_METRICS)

    wr_primary = row.get("wr_archetype_primary")
    if isinstance(wr_primary, str) and wr_primary:
        return ("wr", wr_primary, ARCHETYPE_LABELS.get(wr_primary, wr_primary), WR_ARCHETYPE_METRICS)

    qb_primary = row.get("qb_archetype_primary")
    if isinstance(qb_primary, str) and qb_primary:
        return ("qb", qb_primary, ARCHETYPE_LABELS.get(qb_primary, qb_primary), QB_ARCHETYPE_METRICS)

    te_primary = row.get("te_archetype_primary")
    if isinstance(te_primary, str) and te_primary:
        role_profile = row.get("te_role_profile")
        if te_primary == "receiving_te" and isinstance(role_profile, str) and role_profile in TE_ROLE_PROFILE_LABELS:
            label = TE_ROLE_PROFILE_LABELS[role_profile]
        else:
            label = ARCHETYPE_LABELS.get(te_primary, te_primary)
        return ("te", te_primary, label, TE_ARCHETYPE_METRICS)

    return (None, None, None, None)


def _offense_context_label(row):
    """WR carries a real, direct QB-context read (wr_qb_context) -- used
    verbatim when present. Every other position falls back to a generic
    band off the real offense_environment_score risk category (same 1-5 ->
    0-100 conversion _risk_scorecard used), since no position-specific
    "context" field exists for RB/QB/TE. Real data either way, never
    fabricated."""
    qb_label = QB_CONTEXT_LABELS.get(row.get("wr_qb_context"))
    if qb_label:
        return qb_label
    raw = row.get("offense_environment_score")
    if raw is None or pd.isna(raw):
        return None
    score100 = max(0, min(100, round((float(raw) - 1) / 4 * 100)))
    if score100 <= 34:
        return "Strong offense context"
    if score100 <= 66:
        return "Average offense context"
    return "Weak offense context"


def _format_height_inches(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    total = float(value)
    feet, inches = divmod(round(total), 12)
    return f'{feet}\'{inches}"'


@st.cache_data(show_spinner=False)
def _composite_rank_lookup():
    """Per-position sorted composite distributions from the rookie
    projection model (data/processed/rookie_projections.csv), so the draft
    profile can show a player's composite AND where it ranks within its
    position. Returns {POSITION: [composite, ...] sorted high->low}. Cached;
    keyed on nothing since the CSV is a manual-refresh artifact."""
    path = Path("data/processed/rookie_projections.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "composite" not in df.columns or "position" not in df.columns:
        return {}
    df = df.dropna(subset=["composite"])
    lookup = {}
    for pos, grp in df.groupby(df["position"].astype(str).str.upper()):
        lookup[pos] = sorted(pd.to_numeric(grp["composite"], errors="coerce").dropna().tolist(), reverse=True)
    return lookup


def _composite_position_rank(position, composite):
    """(rank, total) for a composite within its position, or None. Rank is
    1-based, ties share the better rank (count of strictly-higher + 1)."""
    if composite is None or pd.isna(composite):
        return None
    pos = str(position or "").upper()
    values = _composite_rank_lookup().get(pos)
    if not values:
        return None
    c = float(composite)
    rank = sum(1 for v in values if v > c) + 1
    return rank, len(values)


def _draft_profile_section_html(row) -> str:
    """Real-only, read-only college/draft context for the card's Draft
    profile collapsible -- draft pick, school, combine measurables,
    college dominator, breakout age, for anyone with real data in
    rookie_inputs.csv (2026 board) or backtest_rookie_inputs.csv
    (2023-2025 real veterans), merged in with a "rookie_" prefix
    regardless of the player's current rookie_status (see _rankings()).
    Narrative context only -- does NOT feed OUR SCORE/risk_index. Returns
    "" if no real draft_pick is on file -- no fabricated placeholders,
    which is also what gates the collapsible's header off in the caller."""
    draft_pick = row.get("rookie_draft_pick")
    if draft_pick is None or pd.isna(draft_pick):
        return ""

    college = row.get("rookie_college")
    height = _format_height_inches(row.get("rookie_height_inches"))
    weight = _fmt(row.get("rookie_weight"), 0)
    forty = _fmt(row.get("rookie_forty_time"), 2)
    dominator = row.get("rookie_college_dominator_final_year")
    if dominator is None or pd.isna(dominator):
        dominator = row.get("rookie_college_dominator_career")
    dominator_str = f"{float(dominator):.1f}%" if dominator is not None and pd.notna(dominator) else None
    breakout_age = _fmt(row.get("rookie_breakout_age"), 1)
    draft_season = row.get("rookie_draft_season")
    pick_label = f"#{int(draft_pick)}" + (f" ({int(draft_season)})" if pd.notna(draft_season) else "")

    # Composite score, shown prominently as the profile's lead field with
    # its within-position rank (x/total) from the rookie projection model.
    fields = []
    composite_val = row.get("rookie_composite")
    if composite_val is not None and pd.notna(composite_val):
        rank_info = _composite_position_rank(row.get("position"), composite_val)
        rank_suffix = f" ({rank_info[0]}/{rank_info[1]})" if rank_info else ""
        fields.append((
            "Composite Score",
            f'<span style="color:#eef1ec;font-weight:700;">{float(composite_val):.1f}</span>'
            f'<span style="color:#8c948f;">{rank_suffix}</span>',
        ))
    fields.append(("Draft capital", pick_label))
    if isinstance(college, str) and college:
        fields.append(("College", html.escape(college)))
    combine_bits = [b for b in (height, f"{weight} lbs" if weight else None, f"{forty}s 40yd" if forty else None) if b]
    if combine_bits:
        fields.append(("Testing", " / ".join(combine_bits)))
    if dominator_str:
        fields.append(("College production", dominator_str))
    if breakout_age:
        fields.append(("Archetype", f"Breakout age {breakout_age}"))

    fields_html = "".join(
        f'<div class="gp-draft-field"><div class="gp-draft-field-lbl">{html.escape(lbl)}</div>'
        f'<div class="gp-draft-field-val">{val}</div></div>'
        for lbl, val in fields
    )
    # Composite now surfaces as the lead field above; the footnote just
    # frames the whole section as reference context.
    note = "Real college/draft context, for reference -- not additional weight beyond what's shown above."
    tag = "Rookie" if row.get("rookie_draft_season") is not None and pd.notna(row.get("rookie_draft_season")) else "Veteran"

    # Composite component breakdown -- only meaningful while the pre-draft
    # projection is still live (rookie_display_status projected/blended),
    # since that's the only time this composite actually drives anything.
    component_rows = ""
    rookie_display_status = row.get("rookie_display_status")
    if isinstance(rookie_display_status, str) and rookie_display_status in ("projected", "blended"):
        rows = []
        for col, label, kind in ROOKIE_COMPONENT_METRICS:
            formatted = _archetype_metric_value(row.get(col), kind)
            if formatted is None:
                continue
            rows.append(
                '<div style="display:flex;justify-content:space-between;font:400 11.5px \'IBM Plex Sans\',sans-serif;color:#8c948f;">'
                f'<span>{html.escape(label)}</span>'
                f'<span style="color:#c9d0c9;font-weight:600;font-family:\'IBM Plex Mono\',monospace;">{formatted}</span></div>'
            )
        if rows:
            component_rows = (
                '<div style="font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.1em;color:#6f776f;'
                'text-transform:uppercase;margin:12px 0 4px;">Composite components</div>'
                f'<div class="nest-group" style="border-left:none;padding-left:0;">{"".join(rows)}</div>'
            )

    return (
        f'<details class="gp-section"><summary><span>Draft profile '
        f'<span style="font:700 9.5px \'IBM Plex Mono\',monospace;letter-spacing:.05em;'
        f'padding:1px 7px;border-radius:5px;color:{"#b79ada" if tag == "Rookie" else "#7fa8d9"};'
        f'background:rgba(255,255,255,.05);margin-left:8px;">{tag}</span></span>'
        f'<span class="sec-caret sec-caret-open">▾</span><span class="sec-caret sec-caret-closed">▸</span></summary>'
        f'<div class="gp-section-body"><div class="gp-draft-grid">{fields_html}</div>'
        f'{component_rows}'
        f'<div style="font:400 10.5px/1.5 \'IBM Plex Sans\',sans-serif;color:#6f776f;font-style:italic;margin-top:11px;">{note}</div>'
        f'</div></details>'
    )


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _initials(name):
    parts = [p for p in str(name or "").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _sleeper_player_id(player_id):
    """Normalize the merged Sleeper player_id (arrives as a float like
    9221.0 after the pandas merge) into the bare integer string Sleeper's
    CDN path expects, or None if it isn't a usable id."""
    if player_id is None or (isinstance(player_id, float) and pd.isna(player_id)):
        return None
    try:
        return str(int(float(player_id)))
    except (TypeError, ValueError):
        text = str(player_id).strip()
        return text or None


def _avatar_html(name, player_id=None, large=False):
    """Circular avatar: a Sleeper CDN headshot layered over an initials
    fallback. player_id is the Sleeper id (see _rankings()); Sleeper serves
    a real head-crop JPEG for valid ids and 403s otherwise. The headshot is
    a `background-image` overlay so a 403 simply doesn't paint (no broken-
    image icon, and no JS/onerror needed -- Streamlit's HTML sanitizer
    strips event handlers), revealing the initials underneath."""
    cls = "gp-avatar gp-avatar-lg" if large else "gp-avatar"
    initials = html.escape(_initials(name))
    sid = _sleeper_player_id(player_id)
    if sid is None:
        return f'<span class="{cls}"><span class="gp-avatar-ini">{initials}</span></span>'
    # thumb crop for the table row, full-res for the larger card avatar
    variant = "" if large else "thumb/"
    url = f"https://sleepercdn.com/content/nfl/players/{variant}{sid}.jpg"
    return (
        f'<span class="{cls}"><span class="gp-avatar-ini">{initials}</span>'
        f'<span class="gp-avatar-img" style="background-image:url(\'{url}\');"></span></span>'
    )


def _rank_delta(row):
    """ADP rank minus display rank -- positive means we rank him EARLIER
    than the market. Computed fresh off ADP Rk/display_rank (not the fixed
    "Engine vs ADP" column, which is pinned to the Median view's rank) so
    it stays correct under the Ceiling/Risk-adjusted views too."""
    adp_rank = row.get("ADP Rk")
    display_rank = row.get("display_rank")
    if adp_rank is None or pd.isna(adp_rank) or display_rank is None or pd.isna(display_rank):
        return None
    return float(adp_rank) - float(display_rank)


def _delta_cell_html(delta, suffix):
    if delta is None or pd.isna(delta):
        return '<span style="color:#8c948f;">--</span>'
    d = float(delta)
    if d > 0.5:
        return f'<span style="color:#7fc98a;">▲ {d:.1f}{suffix}</span>'
    if d < -0.5:
        return f'<span style="color:#e08a7f;">▼ {abs(d):.1f}{suffix}</span>'
    return '<span style="color:#8c948f;">● in line</span>'


def _market_callout(rank_delta):
    if rank_delta is None:
        return {"arrow": "●", "color": "#8c948f", "bg": "rgba(168,198,134,.06)",
                "bd": "#262b2e", "text": "No market comparison available."}
    d = float(rank_delta)
    if d > 0.5:
        return {"arrow": "▲", "color": "#7fc98a", "bg": "rgba(127,201,138,.09)", "bd": "rgba(127,201,138,.3)",
                "text": f"Model likes him {d:.1f} spots more than the market"}
    if d < -0.5:
        return {"arrow": "▼", "color": "#e08a7f", "bg": "rgba(224,138,127,.09)", "bd": "rgba(224,138,127,.3)",
                "text": f"Model is {abs(d):.1f} spots more cautious than the market"}
    return {"arrow": "●", "color": "#8c948f", "bg": "rgba(168,198,134,.06)", "bd": "#262b2e",
            "text": "Model agrees with the market on this one"}


def _risk_minibars_html(risk_index):
    band = _risk_band(risk_index)
    color = RISK_COLORS.get(band, "#8c948f")
    filled = 0 if risk_index is None or pd.isna(risk_index) else min(4, max(1, math.ceil(float(risk_index) / 25)))
    return "".join(
        f'<span style="height:{h}px;background:{color if i < filled else "#2a3033"}"></span>'
        for i, h in enumerate((7, 10, 13, 16))
    )


def _metric_detail_rows_html(primary_key, metrics_dict, row):
    """Plain label/value rows (no bar -- most of these metrics, like games
    played or depth-chart rank, don't share a common 0-100 domain) for
    whichever real archetype metrics drove this player's role read.
    Continues the exact reads the old hover popover used."""
    if not primary_key or not metrics_dict:
        return ""
    rows = []
    for col, label, kind in metrics_dict.get(primary_key, []):
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        rows.append(
            '<div style="display:flex;justify-content:space-between;font:400 11.5px \'IBM Plex Sans\',sans-serif;color:#8c948f;">'
            f'<span>{html.escape(label)}</span>'
            f'<span style="color:#c9d0c9;font-weight:600;font-family:\'IBM Plex Mono\',monospace;">{formatted}</span></div>'
        )
    return "".join(rows)


def _bar_row_html(label, value, pct, color, nested=False):
    sign = "+" if value > 0 else ""
    cls = "bar-row-grid nested" if nested else "bar-row-grid"
    track_cls = "bar-track thin" if nested else "bar-track"
    fill_cls = "bar-fill thin" if nested else "bar-fill"
    return (
        f'<div class="{cls}"><span class="bar-row-lbl">{html.escape(label)}</span>'
        f'<span class="{track_cls}"><span class="{fill_cls}" style="width:{pct:.1f}%;background:{color}"></span></span>'
        f'<span class="bar-val">{sign}{value:.1f}</span></div>'
    )


def _scoring_breakdown_section_html(row):
    """Real four-component Base Value breakdown (draft_analysis.py's
    calculate_base_value_score()) -- shown as-is rather than force-fit into
    the design handoff's invented example categories (base/VOR/ADP
    adjustment/scenario adjustment), since this is the honest shape of
    what the engine actually computes. Nested group below it is whichever
    real per-position archetype usage metrics drove the role read, when
    any exist."""
    vor = _safe_float(row.get("base_value_vor_component"))
    proj = _safe_float(row.get("base_value_projection_component"))
    market = _safe_float(row.get("base_value_market_component"))
    risk = _safe_float(row.get("base_value_risk_component"))
    top_max = max(abs(vor), abs(proj), abs(market), abs(risk), 1.0)

    rows_html = (
        _bar_row_html("Value over replacement", vor, min(100, abs(vor) / top_max * 100), "#5b87c4")
        + _bar_row_html("Within-position projection", proj, min(100, abs(proj) / top_max * 100), "#7fc98a")
        + _bar_row_html("Market (ADP) swing", market, min(100, abs(market) / top_max * 100), "#7fc98a" if market >= 0 else "#e08a7f")
        + _bar_row_html("Risk penalty", risk, min(100, abs(risk) / top_max * 100), "#7fc98a" if risk >= 0 else "#e08a7f")
    )

    _, primary_key, _, metrics_dict = _primary_archetype(row)
    nested_rows = _metric_detail_rows_html(primary_key, metrics_dict, row)
    nested_html = (
        '<div style="font:400 11px \'IBM Plex Sans\',sans-serif;color:#7c847e;margin:11px 0 7px;'
        'padding-top:10px;border-top:1px dashed #262b2e;">Real usage metrics behind the role read:</div>'
        f'<div class="nest-group">{nested_rows}</div>'
    ) if nested_rows else ""

    total = _safe_float(row.get("final_score"))
    return (
        '<details class="gp-section"><summary><span>Scoring breakdown</span>'
        '<span class="sec-caret sec-caret-open">▾</span><span class="sec-caret sec-caret-closed">▸</span></summary>'
        '<div class="gp-section-body">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;padding-bottom:8px;'
        'border-bottom:1px dashed #262b2e;margin-bottom:10px;">'
        '<span style="font:700 12.5px \'IBM Plex Sans\',sans-serif;color:#e6eae4;">Our score</span>'
        f'<span style="font:800 16px Archivo,sans-serif;color:#a8c686;">{total:.1f}</span></div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;">{rows_html}</div>'
        f'{nested_html}'
        '</div></details>'
    )


def _risk_profile_section_html(row):
    risk_index = _risk_value(row)
    band = _risk_band(risk_index)
    label = RISK_LABELS.get(band, "--")

    cat_rows = []
    for col, cat_label in RISK_CATEGORY_META:
        raw = row.get(col)
        if raw is None or pd.isna(raw):
            continue
        score100 = max(0, min(100, round((float(raw) - 1) / 4 * 100)))
        # Fixed 0-60 scale across every player (not this row's own max) so
        # bars are comparable player-to-player.
        pct = min(100, score100 / 60 * 100)
        cat_rows.append(_bar_row_html(cat_label, float(score100), pct, _risk_bar_color(score100)))

    note = _risk_note_text(row)
    badges = (
        _market_badge(row.get("value_signal_market"))
        + _sleeper_value_badge(row.get("sleeper_value_gap"))
        + _chronic_injury_badge(row.get("games_missed_by_season"))
        + _manual_override_badge(row.get("injury_override_note"))
        + _manual_override_badge(row.get("role_usage_td_override_note"))
    )
    badges_html = f'<div class="sc-badges" style="margin-top:10px;">{badges}</div>' if badges else ""
    risk_num = _safe_float(risk_index)

    return (
        '<details class="gp-section"><summary>'
        f'<span>Risk profile · {risk_num:.0f}/100 · {html.escape(label)}</span>'
        '<span class="sec-caret sec-caret-open">▾</span><span class="sec-caret sec-caret-closed">▸</span></summary>'
        '<div class="gp-section-body">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;padding-bottom:8px;'
        'border-bottom:1px dashed #262b2e;margin-bottom:10px;">'
        '<span style="font:700 12.5px \'IBM Plex Sans\',sans-serif;color:#e6eae4;">Risk score</span>'
        f'<span style="font:800 16px Archivo,sans-serif;color:#e0b45e;">{risk_num:.0f} / 100</span></div>'
        f'<div style="display:flex;flex-direction:column;gap:7px;">{"".join(cat_rows)}</div>'
        '<div style="font:400 10.5px \'IBM Plex Sans\',sans-serif;color:#6f776f;margin-top:9px;">'
        'Bars share a fixed 0-60 scale across every player.</div>'
        f'{badges_html}'
        + (f'<div class="gp-risk-note"><span style="color:#e0b45e;font-weight:600;">In plain terms -- </span>{note}</div>' if note else "")
        + '</div></details>'
    )


def _model_details_section_html(row):
    confidence = row.get("archetype_confidence")
    if confidence is None or pd.isna(confidence):
        confidence = "High" if row.get("projection_source") == "real" else "Low"
    dq = "Real" if row.get("projection_source") == "real" else "Replacement fallback"

    notes = []
    if bool(row.get("model_override_applied")) and pd.notna(row.get("model_override_note")):
        notes.append(("This team's own research", row.get("model_override_note")))
    if pd.notna(row.get("projection_adjustment_pct")) and pd.notna(row.get("projection_adjustment_note")):
        notes.append(("Manual projection adjustment", row.get("projection_adjustment_note")))
    if pd.notna(row.get("model_adjustment_pct")) and pd.notna(row.get("model_adjustment_note")):
        notes.append(("Model correction", row.get("model_adjustment_note")))

    notes_html = "".join(
        f'<div class="gp-model-row"><b>{html.escape(str(src))}</b> -- {html.escape(str(note))}.</div>'
        for src, note in notes
    ) or '<div class="gp-model-row">No sourced adjustments on file for this player.</div>'

    return (
        '<details class="gp-section"><summary><span>Model details</span>'
        '<span class="sec-caret sec-caret-open">▾</span><span class="sec-caret sec-caret-closed">▸</span></summary>'
        '<div class="gp-section-body">'
        '<div style="display:flex;gap:16px;flex-wrap:wrap;font:400 11.5px \'IBM Plex Sans\',sans-serif;color:#8c948f;">'
        f'<span>Confidence: <b style="color:#e6eae4;">{html.escape(str(confidence))}</b></span>'
        f'<span>Data quality: <b style="color:#e6eae4;">{html.escape(str(dq))}</b></span></div>'
        '<div style="font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.1em;color:#6f776f;'
        'text-transform:uppercase;margin:12px 0 4px;">Sourced adjustments</div>'
        f'{notes_html}'
        '</div></details>'
    )


def _ticker_html(row, scoreScaled, grade):
    display_rank = row.get("display_rank")
    market_rank = row.get("ADP Rk")
    delta = _rank_delta(row)
    delta_txt, delta_color = "● even w/ market", "#8c948f"
    if delta is not None:
        if delta > 0.5:
            delta_txt, delta_color = f"▲ {delta:.0f} vs market", "#7fc98a"
        elif delta < -0.5:
            delta_txt, delta_color = f"▼ {abs(delta):.0f} vs market", "#e08a7f"

    band = _risk_band(_risk_value(row))
    risk_color = RISK_COLORS.get(band, "#8c948f")
    grade_color = GRADE_COLORS.get(grade, "#e6eae4")
    proj_ppg = _safe_float(row.get("projection_points")) / 17 if pd.notna(row.get("projection_points")) else None
    risk_index = _risk_value(row)

    cells = [
        ("MKT RANK", f"#{int(market_rank)}" if pd.notna(market_rank) else "--", "#f4f6f2"),
        ("VS MARKET", delta_txt, delta_color),
        ("OUR RANK", f"#{int(display_rank)}" if pd.notna(display_rank) else "--", "#a8c686"),
        ("ADP", _fmt(row.get("adp"), 1) or "--", "#c9d0c9"),
        ("OUR SCORE", f"{scoreScaled:.1f}", "#a8c686"),
        ("PROJ PPG", f"{proj_ppg:.1f}" if proj_ppg is not None else "--", "#c9d0c9"),
        ("RISK", f"{risk_index:.0f} · {RISK_LABELS.get(band, '--').upper()}" if pd.notna(risk_index) else "--", risk_color),
        ("GRADE", (grade or "--").upper(), grade_color),
    ]
    return "".join(
        f'<span class="gp-ticker-cell"><span>{lbl}</span><span style="color:{color};">{html.escape(str(val))}</span></span>'
        for lbl, val, color in cells
    )


def _render_player_card_html(row, scoreScaled, grade):
    name = row.get("player_name") or ""
    position = str(row.get("position") or "").upper()
    team = row.get("team") or ""
    tier = row.get("tier")
    tier_id = None if tier is None or pd.isna(tier) else int(tier)
    tier_color = TIER_BAND_COLORS[(max(tier_id, 1) - 1) % len(TIER_BAND_COLORS)] if tier_id else "#8c948f"
    grade_color = GRADE_COLORS.get(grade, "#e6eae4")
    display_rank = row.get("display_rank")

    proj_ppg = _safe_float(row.get("projection_points")) / 17 if pd.notna(row.get("projection_points")) else None
    floor_ppg = _safe_float(row.get("floor_projection")) / 17 if pd.notna(row.get("floor_projection")) else None
    ceiling_ppg = _safe_float(row.get("ceiling_projection")) / 17 if pd.notna(row.get("ceiling_projection")) else None

    band = _risk_band(_risk_value(row))
    risk_color = RISK_COLORS.get(band, "#8c948f")
    risk_label = RISK_LABELS.get(band, "--")

    callout = _market_callout(_rank_delta(row))

    _, _, role_label, _ = _primary_archetype(row)
    role_short = role_label or position
    context_label = _offense_context_label(row)
    reasons = row.get("recommendation_reasons")
    why = (
        ". ".join(reasons) + "."
        if isinstance(reasons, list) and reasons
        else "Base Value: VOR + within-position projection, ADP and risk as small adjustments."
    )
    verdict = f"{grade or 'Unrated'} {position} as a {role_short.lower()}"
    if risk_label != "--":
        verdict += f", carrying {risk_label.lower()} risk."
    else:
        verdict += "."

    chips = [f'Role: <b>{html.escape(role_short)}</b>']
    if context_label:
        chips.append(f'Context: <b>{html.escape(context_label)}</b>')
    bye = row.get("bye_week")
    if pd.notna(bye):
        chips.append(f'Bye <b>Wk {int(bye)}</b>')
    sos_rank = row.get("season_sos_rank")
    if pd.notna(sos_rank):
        sos_band = "Favorable" if sos_rank <= 11 else "Neutral" if sos_rank <= 21 else "Tough"
        sos_color = {"Favorable": "#7fc98a", "Neutral": "#e0b45e", "Tough": "#e08a7f"}[sos_band]
        chips.append(f'Season SOS <b style="color:{sos_color};">{sos_band}</b>')
    chips_html = "".join(f'<span class="gp-chip-tag">{c}</span>' for c in chips)
    # Playoff SOS / week 15-17 opponent+defense-rank: no weekly matchup
    # dataset exists in the pipeline yet -- placeholder, not fabricated.
    pending_chip = '<span class="gp-chip-tag gp-pending-chip">Playoff SOS -- pending</span>'

    proj_ppg_str = f"{proj_ppg:.1f}" if proj_ppg is not None else "--"

    range_html = ""
    if proj_ppg is not None and floor_ppg is not None and ceiling_ppg is not None:
        dot_pct = (
            max(0.0, min(100.0, (proj_ppg - floor_ppg) / (ceiling_ppg - floor_ppg) * 100))
            if ceiling_ppg > floor_ppg else 50.0
        )
        range_html = (
            '<div class="gp-range-block">'
            f'<div style="font:700 18px \'IBM Plex Mono\',monospace;color:#eef1ec;">{proj_ppg:.1f} '
            '<span style="font:400 11px \'IBM Plex Sans\',sans-serif;color:#8c948f;">projected PPG (17-game)</span></div>'
            '<div class="gp-range-track"><span class="gp-range-fill"></span>'
            f'<span class="gp-range-dot" style="left:calc({dot_pct:.1f}% - 6px);"></span></div>'
            f'<div class="gp-range-labels"><span>FLOOR {floor_ppg:.1f}</span><span>CEILING {ceiling_ppg:.1f}</span></div>'
            '</div>'
        )

    return (
        '<div class="gp-card">'
        f'<div class="gp-ticker">{_ticker_html(row, scoreScaled, grade)}</div>'
        '<div class="gp-card-head"><div class="gp-card-head-left">'
        f'<div class="gp-card-rank">#{int(display_rank) if pd.notna(display_rank) else "--"}</div>'
        f'{_avatar_html(name, row.get("player_id"), large=True)}'
        '<div>'
        f'<div class="gp-card-name">{html.escape(str(name))}</div>'
        f'<div class="gp-card-postteam">{html.escape(position)} · {html.escape(str(team))}</div>'
        + (f'<span class="gp-tier-pill" style="color:{tier_color};">TIER {tier_id}</span>' if tier_id else "")
        + (_breakout_pill_html(row))
        + '</div></div>'
        '<div class="gp-card-score">'
        f'<div class="gp-card-score-num">{scoreScaled:.1f}</div>'
        '<div class="gp-card-score-lbl">Our score</div>'
        f'<div class="gp-card-grade" style="color:{grade_color};">{html.escape(grade or "--")}</div>'
        '</div></div>'
        '<div class="gp-stat3">'
        f'<div class="gp-stat3-cell"><div class="gp-stat3-num">{proj_ppg_str}</div>'
        '<div class="gp-stat3-lbl">Proj PPG</div></div>'
        f'<div class="gp-stat3-cell"><div class="gp-stat3-num">{_fmt(row.get("adp"), 1) or "--"}</div>'
        '<div class="gp-stat3-lbl">ADP</div></div>'
        '<div class="gp-stat3-cell">'
        f'<div style="display:flex;align-items:center;justify-content:center;gap:6px;">'
        f'<span style="width:8px;height:8px;border-radius:99px;background:{risk_color};"></span>'
        f'<span style="font:700 14px \'IBM Plex Sans\',sans-serif;color:#eef1ec;">{risk_label}</span></div>'
        f'<div class="gp-stat3-lbl">Risk {_fmt(_risk_value(row), 0) or "--"}</div></div>'
        '</div>'
        f'<div class="gp-callout" style="background:{callout["bg"]};border:1px solid {callout["bd"]};">'
        f'<div class="gp-callout-arrow" style="color:{callout["color"]};">{callout["arrow"]}</div>'
        f'<div><div class="gp-callout-text">{html.escape(callout["text"])}</div>'
        f'<div class="gp-callout-sub">Ranked #{int(display_rank) if pd.notna(display_rank) else "--"} · '
        f'Market rank #{int(row.get("ADP Rk")) if pd.notna(row.get("ADP Rk")) else "--"} · '
        f'ADP {_fmt(row.get("adp"), 1) or "--"}</div></div></div>'
        '<div style="padding:16px 20px 6px;">'
        '<div class="gp-verdict-lbl">Verdict</div>'
        f'<div class="gp-verdict">{html.escape(verdict)}</div></div>'
        '<div class="gp-why-block">'
        f'<div class="gp-why">{html.escape(why)}</div>'
        f'<div class="gp-chips-row">{chips_html}{pending_chip}</div>'
        '</div>'
        f'{range_html}'
        f'{_scoring_breakdown_section_html(row)}'
        f'{_risk_profile_section_html(row)}'
        f'{_draft_profile_section_html(row)}'
        f'{_model_details_section_html(row)}'
        '</div>'
    )


def _render_row_html(row, view):
    display_rank = row.get("display_rank")
    scoreScaled = _safe_float(row.get("scoreScaled"))
    grade = row.get("grade")
    name = row.get("player_name") or ""
    position = str(row.get("position") or "").upper()
    team = row.get("team") or ""
    pos_color = POSITION_COLORS.get(position, "#8c948f")

    _, _, role_label, _ = _primary_archetype(row)
    role_short = role_label or position

    adp_txt = _fmt(row.get("adp"), 1) or "--"
    proj_source = row.get("ceiling_projection") if view == "ceiling" and pd.notna(row.get("ceiling_projection")) else row.get("projection_points")
    proj_txt = _fmt(proj_source, 0) or "--"

    risk_index = _risk_value(row)
    band = _risk_band(risk_index)
    risk_color = RISK_COLORS.get(band, "#8c948f")
    risk_txt = _fmt(risk_index, 0) or "--"

    vs_adp_html = _delta_cell_html(_rank_delta(row), " vs ADP")
    vs_vegas_html = _delta_cell_html(row.get("position_rank_gap"), " spots")

    summary_grid = (
        '<div class="gp-row-grid">'
        f'<div class="gp-cell-rank">{int(display_rank) if pd.notna(display_rank) else "--"}</div>'
        '<div class="gp-cell-player">'
        f'{_avatar_html(name, row.get("player_id"))}'
        '<div style="min-width:0;">'
        f'<div class="gp-cell-name">{html.escape(str(name))}</div>'
        f'<div class="gp-cell-role">{html.escape(role_short)}</div>'
        '</div></div>'
        '<div class="gp-cell-pos">'
        f'<span style="font:600 11.5px \'IBM Plex Mono\',monospace;color:{pos_color};">{html.escape(position)}</span>'
        f'<span class="gp-cell-team">{html.escape(str(team))}</span></div>'
        f'<div class="gp-cell-num">{adp_txt}</div>'
        f'<div class="gp-cell-num gp-cell-num-strong">{proj_txt}</div>'
        '<div class="gp-score-wrap">'
        f'<div style="display:flex;align-items:baseline;gap:6px;">'
        f'<span class="gp-score-num">{scoreScaled:.1f}</span><span class="gp-score-of100">/100</span></div>'
        f'<span class="gp-score-bar-track"><span class="gp-score-bar-fill" style="width:{scoreScaled:.1f}%;"></span></span>'
        '</div>'
        '<div class="gp-risk-cell">'
        f'<span class="gp-risk-bars">{_risk_minibars_html(risk_index)}</span>'
        f'<span class="gp-risk-num" style="color:{risk_color};">{risk_txt}</span></div>'
        f'<div class="gp-cell-delta">{vs_adp_html}</div>'
        f'<div class="gp-cell-delta">{vs_vegas_html}</div>'
        '<div class="gp-caret"><span class="gp-caret-open">▾</span><span class="gp-caret-closed">▸</span></div>'
        '</div>'
    )
    card = _render_player_card_html(row, scoreScaled, grade)
    return f'<details class="gp-row"><summary>{summary_grid}</summary>{card}</details>'


TIER_BAND_DESC = "Grouped at the largest real score gaps on today's board -- not a fixed scale."


def render_rankings_table(board, view, show_tier_bands):
    header = (
        '<div class="gp-row-grid gp-thead">'
        '<div>Rank</div><div>Player</div><div>Pos</div><div>ADP</div><div>Projection</div>'
        '<div>Our score</div><div>Risk</div><div>vs ADP</div><div>vs Vegas</div><div></div>'
        '</div>'
    )

    body = []
    last_tier = object()
    for _, row in board.iterrows():
        if show_tier_bands:
            tier = row.get("tier")
            tier_id = None if pd.isna(tier) else int(tier)
            if tier_id != last_tier:
                last_tier = tier_id
                label = "Unranked" if tier_id is None else f"Tier {tier_id}"
                color = "#8c948f" if tier_id is None else TIER_BAND_COLORS[(max(tier_id, 1) - 1) % len(TIER_BAND_COLORS)]
                body.append(
                    f'<div class="tier-band"><span class="tier-band-bar" style="background:{color};"></span>'
                    f'<span class="tier-band-label" style="color:{color};">{label}</span>'
                    f'<span class="tier-band-desc">{TIER_BAND_DESC}</span></div>'
                )
        body.append(_render_row_html(row, view))

    return f'<div class="gp-table-wrap"><div class="gp-table-inner">{header}{"".join(body)}</div></div>'


st.set_page_config(page_title="Top 250 Rankings", layout="wide", initial_sidebar_state="collapsed")

# Still needed: the scoring engine reads roster_settings / drafted_players /
# current_pick_number from session state. init_session_state() seeds the
# defaults (empty roster, pick 1) that define the draft-start board.
init_session_state()


def _safe_int(value, default=0):
    """
    Coerce a session-state roster/league value to int for cache keying.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scoring_version():
    """
    Cache key covering everything the board depends on.

    st.cache_data hashes only the decorated function's own body, so edits to
    draft_analysis.py (where the scoring actually lives) or to the data files
    do NOT invalidate it -- the app kept serving a pre-fix board through
    repeated reloads. Keying on the mtimes of the real inputs fixes that.

    Roster settings and league size are part of this key for the same
    reason, and it is a correctness issue rather than a freshness one
    (added 2026-08-21): they feed build_position_replacement_baselines()
    and build_position_vor_spans() directly, so changing WR/TE/FLEX counts
    changes every replacement baseline and therefore every score on the
    board. Keyed only on file mtimes, the Rankings page would keep serving
    a board computed for a different league format -- silently, with no
    file having changed.
    """
    roster = st.session_state.get("roster_settings", {}) or {}
    roster_key = tuple(sorted((str(k), _safe_int(v)) for k, v in roster.items()))
    league_key = _safe_int(st.session_state.get("league_size"))

    watched = [
        Path("draftkit/draft_analysis.py"),
        Path("draftkit/draft_state.py"),
        Path("data/processed/master_players.csv"),
        Path("research/validation_v1/data/positional_tier_curve.csv"),
        Path("data/processed/sportsbook_vs_adp_comparison.csv"),
        Path("data/processed/model_projections_v1.csv"),
        Path("data/processed/risk_variables.csv"),
        Path("data/processed/team_schedule_risk.csv"),
        Path("data/processed/rb_archetypes.csv"),
        Path("data/processed/wr_archetypes.csv"),
        Path("data/processed/qb_archetypes.csv"),
        Path("data/processed/te_archetypes.csv"),
        Path("data/processed/rookie_projections.csv"),
    ]
    mtimes = tuple(p.stat().st_mtime if p.exists() else 0 for p in watched)

    return (mtimes, roster_key, league_key)


@st.cache_data(show_spinner="Building rankings...", max_entries=4)
def _rankings(_version):
    df = build_recommendation_rankings_df()
    if df.empty:
        return df

    # The scoring pipeline doesn't carry archetype_primary through, so join
    # it back from the source frame. (draft_analysis does emit a column
    # literally named `archetype`, but that's the legacy STEADY/RISKY bucket
    # -- empty for ~87% of players and derived from scores that are no
    # longer populated. Not the usage-derived archetypes.)
    source = load_players_df()
    if "archetype_primary" in source.columns and "player_name" in source.columns:
        arch = source[["player_name", "archetype_primary"]].drop_duplicates("player_name")
        df = df.merge(arch, on="player_name", how="left")
        df["Archetype"] = df["archetype_primary"].apply(archetype_label)
        df["Scoring Type"] = df["archetype_primary"].apply(
            lambda a: {"event": "TD / big play", "volume": "Volume", "mixed": "Mixed"}.get(
                risk_profile(a), ""
            )
        )

    # Age context -- display only. Age failed as a predictor (see
    # draftkit/age_context.py), so it never touches final_score.
    if "age" in source.columns and "player_name" in source.columns:
        ages = source[["player_name", "age"]].drop_duplicates("player_name")
        df = df.merge(ages, on="player_name", how="left", suffixes=("", "_src"))
        age_col = "age" if "age" in df.columns else "age_src"
        df["Age"] = pd.to_numeric(df[age_col], errors="coerce")
        df["Age Note"] = [age_note(p, a) for p, a in zip(df["position"], df["Age"])]

    # expert_rank is the FantasyPros consensus, carried through purely for
    # display. Not carried through the scoring pipeline, so join it back
    # from the source frame the same way Age was.
    #
    # "tier" is NOT joined from here anymore (2026-08-17), and "position_rank"
    # joined from here (2026-08-21) -- both used to be FantasyPros' own
    # static, externally-curated labels, completely disconnected from this
    # app's real scoring pipeline (confirmed: zero references anywhere in
    # draft_analysis.py). Both are now automatically computed from real
    # final_score once the board is sorted -- see the "automatic tier
    # assignment"/"automatic position-rank assignment" blocks near
    # board.insert(0, "Rank", ...). Joining position_rank here would just
    # be immediately overwritten there, so it's dropped rather than left
    # as dead code.
    # floor_projection/ceiling_projection/bye_week/archetype_confidence
    # (Rankings Lab redesign, 2026-08-24) -- real columns already on
    # master_players.csv, just not carried through the scoring pipeline.
    # Same join pattern as expert_rank above.
    # player_id here is the Sleeper player ID (source frame comes from
    # draftkit/data_sources/sleeper_source.py, api.sleeper.app/v1/players/nfl).
    # Carried through purely so the avatar can pull a Sleeper CDN headshot
    # (https://sleepercdn.com/content/nfl/players/thumb/<id>.jpg) -- display
    # only, never scored.
    for column in ("expert_rank", "floor_projection", "ceiling_projection", "bye_week", "archetype_confidence", "player_id"):
        if column in source.columns and "player_name" in source.columns:
            extra = source[["player_name", column]].drop_duplicates("player_name")
            df = df.merge(extra, on="player_name", how="left", suffixes=("", "_src"))

    # Team schedule risk -- optional, manually refreshed (see
    # draftkit/scripts/build_schedule_data.py). season_sos_rank feeds the
    # card's Season SOS chip. Playoff-week-specific SOS/opponent data isn't
    # produced by this script (see its docstring -- a weeks-15-17-only
    # variant was tried and deliberately widened to whole-season), so the
    # card's Playoff SOS renders as a placeholder rather than this.
    schedule_path = Path("data/processed/team_schedule_risk.csv")
    if schedule_path.exists():
        schedule_df = pd.read_csv(schedule_path)
        keep_cols = [c for c in ("team", "season_sos_rank") if c in schedule_df.columns]
        if "team" in keep_cols and "team" in df.columns:
            df = df.merge(schedule_df[keep_cols], on="team", how="left")

    # Sportsbook vs. ADP comparison -- optional, manually refreshed (see
    # draftkit/scripts/compare_sportsbook_vs_adp.py). Skip silently if it
    # hasn't been generated yet; this data source is manual-pull-only by
    # design, so the board must not break without it.
    sportsbook_path = Path("data/processed/sportsbook_vs_adp_comparison.csv")
    if sportsbook_path.exists():
        sportsbook_df = pd.read_csv(sportsbook_path)
        keep_cols = [
            c for c in ("player_name", "sportsbook_half_ppr_points", "adp_position_rank",
                        "sportsbook_position_rank", "position_rank_gap")
            if c in sportsbook_df.columns
        ]
        if "player_name" in keep_cols:
            df = df.merge(sportsbook_df[keep_cols], on="player_name", how="left")
            for column in ("adp_position_rank", "sportsbook_position_rank"):
                if column in df.columns:
                    df[column] = _position_rank_number(df[column])

    # Model-driven projection (RB/WR/TE only) -- optional, manually
    # refreshed (see research/validation_v1/build_live_projections_v1.py,
    # Iteration 8 of projection_model_iteration_plan.pdf). Same
    # skip-silently convention as the sportsbook merge above. Deliberately
    # a SEPARATE column from projection_points, not a replacement: QB has
    # no model here (its version validated worse than ADP, see that
    # script's docstring), and even at RB/WR/TE this is a new, less-proven
    # signal meant to sit alongside the existing FantasyPros projection,
    # not overwrite it.
    model_projection_path = Path("data/processed/model_projections_v1.csv")
    if model_projection_path.exists():
        model_projection_df = pd.read_csv(model_projection_path)
        keep_cols = [
            c for c in (
                "player_name", "model_projection_points", "model_projection_status",
                "team_changed", "competitor_departed", "competitor_arrived", "continuity_note",
                "model_projection_points_adjusted", "model_adjustment_pct", "model_adjustment_note",
                "model_projection_points_fallback", "model_projection_fallback_note",
            )
            if c in model_projection_df.columns
        ]
        if "player_name" in keep_cols:
            df = df.merge(model_projection_df[keep_cols], on="player_name", how="left")

        # Effective MODEL PROJ value, sort-only -- mirrors the exact display
        # priority in the MODEL PROJ cell renderer below (adjusted, then
        # rookie-fallback, then raw): without this, clicking the MODEL PROJ
        # header sorted on model_projection_points (raw), which silently
        # ignored every diagnosed-defect correction this whole model-
        # correction project built (model_projection_points_adjusted) --
        # the column visibly SHOWED the corrected number but sorted on the
        # uncorrected one underneath it.
        for col in ("model_projection_points_adjusted", "model_projection_points_fallback", "model_projection_points"):
            if col not in df.columns:
                df[col] = float("nan")
        df["model_projection_effective"] = (
            df["model_projection_points_adjusted"]
            .fillna(df["model_projection_points_fallback"])
            .fillna(df["model_projection_points"])
        )

    # Risk scorecard -- optional, manually refreshed (see
    # draftkit/scripts/build_risk_variables.py). Same skip-silently
    # convention as the sportsbook merge above: this data source is
    # manual-pull-only, so the board must not break without it.
    #
    # v2 columns: the four weighted categories are injury_score /
    # role_usage_td_score / offense_environment_score /
    # schedule_weather_venue_score (not risk_injury/risk_role/... -- those
    # were v1's 5-category names). value_signal_market and
    # volatility_diagnostic are NOT part of risk_index -- surfaced in the
    # tooltip for context, same as risk_cli.py shows them as separate lines.
    risk_path = Path("data/processed/risk_variables.csv")
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        keep_cols = [
            c for c in ("player_name", "risk_index", "injury_score", "injury_acute_score",
                        "injury_chronic_score", "role_usage_td_score",
                        "offense_environment_score", "schedule_weather_venue_score",
                        "value_signal_market", "volatility_diagnostic", "games_missed_by_season",
                        "injury_override_note", "role_usage_td_override_note")
            if c in risk_df.columns
        ]
        if "player_name" in keep_cols:
            df = df.merge(risk_df[keep_cols], on="player_name", how="left")

    # RB archetype taxonomy -- optional, manually refreshed (see
    # draftkit/scripts/build_rb_archetypes.py). RB-only; other positions
    # get NaN here and the archetype cell renders blank for them.
    #
    # Renamed with an "rb_" prefix on the way in -- the OLDER, existing
    # archetype system (draftkit/archetypes.py, current_season_archetypes.csv,
    # merged earlier above for the "Archetype"/"Scoring Type" columns) ALREADY
    # has a column literally named archetype_primary in df at this point.
    # Confirmed via a real failed merge, not assumed: without renaming,
    # pandas' default merge silently suffixed both to archetype_primary_x/_y,
    # and _render_cell's row.get("archetype_primary") returned None for
    # every player -- the whole column rendered blank.
    archetype_path = Path("data/processed/rb_archetypes.csv")
    if archetype_path.exists():
        archetype_df = pd.read_csv(archetype_path)
        rb_archetype_cols = {
            "archetype_primary": "rb_archetype_primary",
            "down_split": "rb_down_split",
            "lean": "rb_lean",
        }
        archetype_df = archetype_df.rename(columns=rb_archetype_cols)
        keep_cols = [
            c for c in (
                "player_name", "rb_archetype_primary", "rb_down_split", "rb_lean",
                "opportunity_share", "games_played", "redzone_touch_share",
                "inside_10_carry_share", "third_down_snap_share",
                "target_share_of_backfield", "route_participation_rate",
                "depth_chart_rank", "total_touches", "explosive_run_rate",
                "yards_per_touch",
            )
            if c in archetype_df.columns
        ]
        if "player_name" in keep_cols:
            df = df.merge(archetype_df[keep_cols], on="player_name", how="left")

    # WR archetype system v1 -- optional, manually refreshed (see
    # draftkit/scripts/build_wr_archetypes.py). WR-only; other positions get
    # NaN here and the archetype cell falls through to the RB branch or
    # blank (see _render_cell's "archetype" style).
    #
    # ALL real metric columns renamed with a "wr_" prefix on the way in --
    # not just the three classification outputs. wr_archetypes.csv's own
    # games_played column would otherwise silently collide with the
    # rb_archetypes.csv games_played already merged above (pandas would
    # suffix both to games_played_x/_y) -- the exact same collision class
    # already found and fixed once this session for archetype_primary,
    # avoided here proactively rather than waiting to hit it again.
    wr_archetype_path = Path("data/processed/wr_archetypes.csv")
    if wr_archetype_path.exists():
        wr_archetype_df = pd.read_csv(wr_archetype_path)
        wr_rename_cols = {
            "target_share": "wr_target_share",
            "games_played": "wr_games_played",
            "redzone_target_share": "wr_redzone_target_share",
            "redzone_targets": "wr_redzone_targets",
            "adot": "wr_adot",
            "receptions": "wr_receptions",
            "targets": "wr_targets",
            "rush_attempts": "wr_rush_attempts",
            "yards_per_reception": "wr_yards_per_reception",
            "yac_per_reception": "wr_yac_per_reception",
            "rush_attempts_per_game": "wr_rush_attempts_per_game",
            "rush_yards_per_attempt": "wr_rush_yards_per_attempt",
            "qb_tier": "wr_qb_tier",
            "team_wr1_target_share": "wr_team_wr1_target_share",
        }
        wr_archetype_df = wr_archetype_df.rename(columns=wr_rename_cols)
        wr_keep_cols = [
            c for c in (
                "player_name", "wr_archetype_primary", "wr_leans", "wr_qb_context",
                *wr_rename_cols.values(),
            )
            if c in wr_archetype_df.columns
        ]
        if "player_name" in wr_keep_cols:
            df = df.merge(wr_archetype_df[wr_keep_cols], on="player_name", how="left")

    # QB archetype system v1 (claude_code_plan_qb_archetypes.pdf) --
    # optional, manually refreshed (see draftkit/scripts/build_qb_archetypes.py).
    # QB-only; other positions get NaN here and the archetype cell falls
    # through the RB/WR branches to blank. Real metric columns prefixed
    # "qb_" on the way in, same proactive collision-avoidance convention as
    # the WR merge above -- games/attempts would otherwise collide with
    # rb_archetypes.csv's/wr_archetypes.csv's own generically-named columns.
    qb_archetype_path = Path("data/processed/qb_archetypes.csv")
    if qb_archetype_path.exists():
        qb_archetype_df = pd.read_csv(qb_archetype_path)
        qb_rename_cols = {
            "games": "qb_games",
            "attempts": "qb_attempts",
            "rushing_fantasy_pct": "qb_rushing_fantasy_pct",
            "seasons_used": "qb_seasons_used",
        }
        qb_archetype_df = qb_archetype_df.rename(columns=qb_rename_cols)
        qb_keep_cols = [
            c for c in ("player_name", "qb_archetype_primary", *qb_rename_cols.values())
            if c in qb_archetype_df.columns
        ]
        if "player_name" in qb_keep_cols:
            df = df.merge(qb_archetype_df[qb_keep_cols], on="player_name", how="left")

    # TE archetype system v1 (plan_te_archetypes.pdf) -- optional, manually
    # refreshed (see draftkit/scripts/build_te_archetypes.py). TE-only;
    # other positions get NaN here and the archetype cell falls through to
    # blank. Real metric columns prefixed "te_" on the way in, same
    # proactive collision-avoidance convention as every other position merge.
    te_archetype_path = Path("data/processed/te_archetypes.csv")
    if te_archetype_path.exists():
        te_archetype_df = pd.read_csv(te_archetype_path)
        te_rename_cols = {
            "target_share": "te_target_share",
            "snap_share": "te_snap_share",
            "games_played": "te_games_played",
            "redzone_target_share": "te_redzone_target_share",
        }
        te_archetype_df = te_archetype_df.rename(columns=te_rename_cols)
        te_keep_cols = [
            c for c in ("player_name", "te_archetype_primary", "te_leans", "te_role_profile",
                        *te_rename_cols.values())
            if c in te_archetype_df.columns
        ]
        if "player_name" in te_keep_cols:
            df = df.merge(te_archetype_df[te_keep_cols], on="player_name", how="left")

    # Rookie projection model -- optional, manually refreshed (see
    # draftkit/scripts/build_rookie_projections.py). Real pre-season
    # signal for this year's incoming class, gated in _render_cell by
    # rookie_status ("projected"/"blended") -- a player who's accumulated
    # enough real sample for blend_rookie_tag() to report "confirmed" is
    # already covered by the RB/WR archetype merges above instead.
    #
    # ALL columns renamed with a "rookie_" prefix on the way in, same
    # proactive convention as the WR merge above -- rookie_projections.csv's
    # own "position"/"composite"/"status" column names risk colliding with
    # existing df columns (df already has a real "position" column from
    # earlier in this function) rather than waiting to hit that collision.
    rookie_path = Path("data/processed/rookie_projections.csv")
    if rookie_path.exists():
        rookie_df = pd.read_csv(rookie_path)
        rookie_rename_cols = {
            "draft_pick": "rookie_draft_pick",
            "landing_team": "rookie_landing_team",
            "projected_tier": "rookie_projected_tier",
            "composite": "rookie_composite",
            "sample_weight": "rookie_sample_weight",
            "status": "rookie_status",
            "tag": "rookie_tag",
            # display_* is the "unconfirmed"-aware fallback used for the
            # LIVE badge -- status/tag/sample_weight above stay real
            # usage-sample confidence, untouched (validation ground truth
            # for run_rookie_backtest.py/validate_component_
            # predictiveness.py, which rank "unconfirmed" as real,
            # meaningful outcome data). See build_rookie_projections.py.
            "display_status": "rookie_display_status",
            "display_tag": "rookie_display_tag",
            "display_weight": "rookie_display_weight",
            "draft_season": "rookie_draft_season",
            # Raw prospect fields (not derived component scores) -- for the
            # real, read-only Prospect Profile section, shown for anyone
            # with real college/combine data on file regardless of how
            # confirmed their live badge currently is.
            "college": "rookie_college",
            "height_inches": "rookie_height_inches",
            "weight": "rookie_weight",
            "forty_time": "rookie_forty_time",
            "college_dominator_final_year": "rookie_college_dominator_final_year",
            "college_dominator_career": "rookie_college_dominator_career",
            "breakout_age": "rookie_breakout_age",
            "component_draft_capital": "rookie_component_draft_capital",
            "component_dominator": "rookie_component_dominator",
            "component_athletic": "rookie_component_athletic",
            "component_roster_competition": "rookie_component_roster_competition",
            "component_offense_environment": "rookie_component_offense_environment",
            "component_schedule": "rookie_component_schedule",
            "component_breakout_age": "rookie_component_breakout_age",
            "component_ncaa_passer_rating": "rookie_component_ncaa_passer_rating",
            # QB-only disclosure flag (claude_code_plan_qb_rookie_projection.pdf)
            # -- distinguishes real measured/overridden rushing data from
            # the 13 explicitly assumed-non-rusher cases, never silently
            # merged at the same confidence.
            "rushing_data_status": "rookie_rushing_data_status",
            "generated_at": "rookie_generated_at",
        }
        rookie_df = rookie_df.rename(columns=rookie_rename_cols)
        rookie_keep_cols = [
            c for c in ("player_name", *rookie_rename_cols.values())
            if c in rookie_df.columns
        ]
        if "player_name" in rookie_keep_cols:
            df = df.merge(rookie_df[rookie_keep_cols], on="player_name", how="left")

    # Dad's-league scoring -- an alternate projection that reworks the model
    # score under a no-PPR, TD-and-tiered-yardage format (see
    # draftkit/dads_scoring.py). Adds dads_projection_points / dads_vor /
    # dads_final_score, all NaN for players without a stat projection. The
    # standard final_score is untouched; the main flow swaps these in only
    # when the Dad's League scoring model is selected. Skip silently if the
    # stat-projection source is missing, same convention as the merges above.
    try:
        from draftkit.dads_scoring import add_dads_scores
        df = add_dads_scores(df)
    except Exception:
        pass

    return df


rankings_df = _rankings(_scoring_version())

if rankings_df.empty:
    st.error("No rankings available -- check that data/processed/master_players.csv exists.")
    st.stop()

# Manual exclusions -- players confirmed out for the season but not yet
# reflected in injury_status/the data pipeline. Update this set as needed.
MANUALLY_EXCLUDED_PLAYERS = {
    "Ricky Pearsall",  # out for the year
}
rankings_df = rankings_df[~rankings_df["player_name"].isin(MANUALLY_EXCLUDED_PLAYERS)]

# Restrict to players who are actually draftable BEFORE ranking.
#
# The source pool is Sleeper's full database (~3,941 rows), which includes
# practice-squad, retired, and deceased players. Only ~678 carry a real ADP
# and ~513 a real projection. Without this filter every player with neither
# lands on the same replacement-level fallback score (an exact tie at
# 35.70), so the board sorted them ALPHABETICALLY -- everything past rank 83
# came back as Aaron Dykes, Aaron Green, Aaron Hernandez, Aaron Peck, Abram
# Smith... A board that is mostly alphabetized non-players is worse than
# useless.
#
# The market pricing a player, or a projection service publishing a real
# number for him, is the available evidence that he is draftable at all.
has_adp = pd.to_numeric(rankings_df.get("adp"), errors="coerce").notna()
has_real_projection = rankings_df.get("projection_source", pd.Series(dtype=object)) == "real"
rankings_df = rankings_df[has_adp | has_real_projection].copy()

if rankings_df.empty:
    st.error("No draftable players found (none have a real ADP or projection).")
    st.stop()

def _freshness_stamp(path):
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d, %I:%M %p")


st.markdown(RANKINGS_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="gp-header"><div class="gp-header-left">'
    '<span class="gp-logo">FF</span><span class="gp-wordmark">Rankings Lab</span>'
    '<div class="gp-nav"><span class="active">Board</span>'
    '<span>Projections</span><span>Methodology</span>'
    '<span>Draft assistant</span></div></div>'
    '<div class="gp-chips">'
    '<span class="gp-chip"><span class="gp-chip-dot"></span>Snapshot · not live</span>'
    '<span class="gp-chip">12-team · Half-PPR</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
# Tool switcher -- the single-page dropdown that replaces Streamlit's
# default left sidebar nav. Selecting another tool navigates via
# st.switch_page (see draftkit/ui_helpers.render_tool_nav).
render_tool_nav("Rankings Board")

# Scoring model selector -- the standard board vs a second score setting
# built from Dad's League scoring (draftkit/dads_scoring.py). Only offered
# when dad's scores actually computed (stat-projection source present); the
# board swap itself happens just before ranking, below.
SCORE_MODES = {
    "Standard (12-team Half-PPR)": "standard",
    "Dad's League": "dads",
}
_dads_available = "dads_final_score" in rankings_df.columns and rankings_df["dads_final_score"].notna().any()
if _dads_available:
    _score_label = st.selectbox(
        "Scoring model", list(SCORE_MODES.keys()), key="score_mode_select",
    )
    score_mode = SCORE_MODES[_score_label]
    if score_mode == "dads":
        st.caption(
            "Dad's League: no PPR, no per-yard points -- scored on TDs (rush 5, "
            "pass/rec 4), interceptions/fumbles, and per-game tiered yardage "
            "bonuses. Projection re-derived from stat totals, with yardage "
            "modeled game-by-game (per-game variance around the bucket cliffs, "
            "not the season average); TD-length bonuses use an expected value."
        )
else:
    score_mode = "standard"

_stamps = [
    ("VEGAS", _freshness_stamp("data/processed/sportsbook_vs_adp_comparison.csv")),
    ("EXPERT CONSENSUS", _freshness_stamp("data/processed/master_players.csv")),
    ("RISK DATA", _freshness_stamp("data/processed/risk_variables.csv")),
    ("ADP", _freshness_stamp("data/processed/master_players.csv")),
]
_stamps_html = "".join(
    f'<span>{lbl} <b>{html.escape(val)}</b></span>' for lbl, val in _stamps if val
)

st.markdown(
    '<div class="gp-title-row">'
    '<div><div class="gp-eyebrow">2026 Redraft · Preseason board</div>'
    '<h1 class="gp-title">Consensus Rankings</h1>'
    '<p class="gp-subhead">Curated analyst sheet with model-assisted scoring. '
    'Click any player to open their full card.</p></div></div>'
    f'<div class="gp-stamps">{_stamps_html}</div>',
    unsafe_allow_html=True,
)

# Filter/view state -- three independent st.session_state keys (position,
# view, risk), each an ordinary Streamlit widget rather than HTML links, so
# they never fight the query-param sort mechanism the old column-header
# links used (that mechanism is gone now: the new design has no per-column
# sort, only the view toggle below).
if "pos_filter" not in st.session_state:
    st.session_state["pos_filter"] = "Overall"
if "view" not in st.session_state:
    st.session_state["view"] = "median"
if "risk_filter" not in st.session_state:
    st.session_state["risk_filter"] = "ALL"

_reset_col, _export_col = st.columns([1, 1])
with _reset_col:
    if st.button("Reset filters", key="reset_filters"):
        st.session_state["pos_filter"] = "Overall"
        st.session_state["view"] = "median"
        st.session_state["risk_filter"] = "ALL"
        if "search_query" in st.session_state:
            del st.session_state["search_query"]
        st.rerun()

_chip_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 4])
for _col, _label in zip(_chip_cols[:5], POSITION_CHIPS):
    with _col:
        if st.button(
            _label, key=f"chip_{_label}", width="stretch",
            type="primary" if st.session_state["pos_filter"] == _label else "secondary",
        ):
            st.session_state["pos_filter"] = _label
            st.rerun()
for _col, (_key, _label) in zip(_chip_cols[5:8], VIEW_TABS):
    with _col:
        if st.button(
            _label, key=f"view_{_key}", width="stretch",
            type="primary" if st.session_state["view"] == _key else "secondary",
        ):
            st.session_state["view"] = _key
            st.rerun()

position_filter = st.session_state["pos_filter"]
view = st.session_state["view"]
VIEW_NOTES = {
    "median": "Sorted by our score (median projection).",
    "ceiling": "Re-sorted by ceiling outcome -- high-upside players rise.",
    "riskAdj": "Re-sorted by score minus a risk penalty.",
}
st.markdown(f'<div class="gp-view-note">{VIEW_NOTES[view]}</div>', unsafe_allow_html=True)

_search_col, _risk_cols_wrap, _count_col = st.columns([3, 4, 2])
with _search_col:
    search_query = st.text_input(
        "Search", placeholder="Search player or team", label_visibility="collapsed",
        key="search_query",
    )
_risk_cols = _risk_cols_wrap.columns(4)
for _col, (_key, _label) in zip(_risk_cols, RISK_TABS):
    with _col:
        if st.button(
            _label, key=f"risk_{_key}", width="stretch",
            type="primary" if st.session_state["risk_filter"] == _key else "secondary",
        ):
            st.session_state["risk_filter"] = _key
            st.rerun()
risk_filter = st.session_state["risk_filter"]

# Manual projection adjustment (2026-08-27) -- a single control rather than
# one per row: the table below is one big HTML string rendered through
# st.markdown(unsafe_allow_html=True) (see this file's own architecture
# note at the top), so a real Streamlit widget can't live inside any one
# card without either breaking the table's grid layout or rendering 250+
# separate elements. This reuses the exact same functions the News Queue
# page's Apply button uses (draftkit/news_queue.py), so a change made here
# is identical in effect to one applied there -- same audit log, same
# "which data layer actually wins" resolution, same live rank preview.
with st.popover("🎚️ Adjust a player"):
    st.caption(
        "Same mechanism as the News Queue -- shows where this lands before you commit. "
        "Logged to research/applied_news_overrides_log.md either way."
    )
    _all_names = sorted(rankings_df["player_name"].dropna().unique().tolist())
    _adj_player = st.selectbox(
        "Player", _all_names, index=None, placeholder="Type a name...", key="adjust_player_select",
    )
    if _adj_player:
        _board_row = rankings_df[rankings_df["player_name"] == _adj_player]
        _current_board_pts = _safe_float(_board_row.iloc[0].get("projection_points")) if not _board_row.empty else None
        try:
            _raw_pts, _, _layer = resolve_projection_base(_adj_player)
        except ValueError as _exc:
            st.error(str(_exc))
        else:
            st.caption(f"Current: {_current_board_pts:.1f} pts (source: {_layer})")
            _pct_from_current = st.number_input(
                "Change from current (%)", value=0.0, step=0.5, key="adjust_player_pct",
                help="Relative to the CURRENT board value, not the raw model output -- "
                     "positive raises it, negative lowers it.",
            )
            _target_pts = round(_current_board_pts * (1 + _pct_from_current / 100), 1)
            # `if _raw_pts` (not `!= 0`) was the exact shape of the real bug
            # fixed in resolve_projection_base() (2026-08-27, Trevor Lawrence)
            # -- NaN is truthy in Python, so that guard silently let a NaN
            # raw value through into a NaN pct instead of hitting the 0.0
            # fallback. resolve_projection_base() no longer returns NaN (it
            # raises instead, caught above), but guarding on the actual
            # condition intended -- a genuine zero projection -- rather than
            # bare truthiness avoids reintroducing the same class of bug.
            _equiv_pct_vs_raw = round((_target_pts / _raw_pts - 1) * 100, 1) if _raw_pts != 0 else 0.0
            _fake_entry = {
                "player": _adj_player,
                "reason": f"Manual adjustment via Rankings page ({_pct_from_current:+.1f}% from current), "
                          f"applied {date.today().isoformat()}.",
            }
            _preview = preview_projection_rank(_fake_entry, _equiv_pct_vs_raw, rankings_df)
            if _preview:
                _pos, _cur, _new, _tot = (
                    _preview["position"], _preview["current_rank"], _preview["new_rank"], _preview["total"],
                )
                if _new < _cur:
                    _arrow, _color = "↑", "#7fc98a"
                elif _new > _cur:
                    _arrow, _color = "↓", "#e08a7f"
                else:
                    _arrow, _color = "→", "#8c948f"
                st.markdown(
                    f'<div style="font:600 13px \'IBM Plex Mono\',monospace;color:{_color};margin:4px 0 8px">'
                    f'{_pos} #{_cur} {_arrow} {_pos} #{_new} <span style="color:#6f776f;font-weight:400">'
                    f'(of {_tot} · {_current_board_pts:.1f} → {_target_pts:.1f} pts)</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Apply", key="adjust_player_apply", type="primary"):
                _before, _after, _applied_layer = apply_projection_override(_fake_entry, pct=_equiv_pct_vs_raw)
                st.success(f"Applied via {_applied_layer}. {_before} → {_after} pts.")
                st.rerun()

board = rankings_df.copy()

# Second score setting: when the Dad's League scoring model is active, swap
# dad's reworked projection/VOR/score in for the standard ones and re-sort,
# BEFORE Rank is assigned. Everything downstream (Rank, tiers, scoreScaled,
# grades, the risk-adjusted/ceiling re-sorts, VOR text) reads these columns,
# so the whole board re-derives under dad's scoring with no further changes.
# Players with no dad's projection (NaN) sort last, same as the market/proj
# fallbacks elsewhere.
if score_mode == "dads" and "dads_final_score" in board.columns:
    board["final_score"] = board["dads_final_score"]
    board["projection_points"] = board["dads_projection_points"]
    if "dads_vor" in board.columns:
        board["value_over_replacement_points"] = board["dads_vor"]
    board = board.sort_values(
        "final_score", ascending=False, na_position="last", kind="stable"
    ).reset_index(drop=True)

# Rank is assigned BEFORE filtering so a player keeps his true overall rank
# when you narrow to one position -- filtering to RB should show RB1 as
# overall #1, not renumber him.
board.insert(0, "Rank", range(1, len(board) + 1))

# Where the market has this player, and how far we differ.
#
# ADP rank is computed over the SAME draftable pool as Rank, not from raw ADP
# values -- otherwise the two would be on different scales (raw ADP counts
# kickers and defenses this board excludes) and the delta would be junk.
#
# Sign: POSITIVE = we rank him EARLIER than the market (we're higher on him).
# Negative = we're lower. Named "Engine vs ADP" (not just "vs ADP") to make
# clear this is OUR scoring engine's rank compared to ADP -- distinct from
# the sportsbook comparison further right, which is a different "X vs ADP".
_adp_rank = pd.to_numeric(board["adp"], errors="coerce").rank(method="first")
board["ADP Rk"] = _adp_rank
board["Engine vs ADP"] = _adp_rank - board["Rank"]

board = board.head(TOP_N)

# Automatic tier assignment (2026-08-17, replaces FantasyPros' static,
# externally-curated tier label -- confirmed zero references in the real
# scoring pipeline). Computed HERE -- after Rank, after the TOP_N cut,
# BEFORE the position filter below -- so a player's tier number is fixed
# once, on the full Overall board, and stays identical when the user
# narrows to a single position; recomputing per-position would let the
# same player land in a different tier number depending on which tab is
# open, which is confusing and not what a real tier sheet does.
#
# Signal: final_score (fixed 2026-08-21 -- was reading
# projection_component_score, a pre-Base-Value-rewrite leftover column).
# That choice was made 2026-08-17 because final_score's own gaps were
# "almost perfectly uniform" back then -- an artifact of the old
# percentile-based ADP anchor (apply_adp_anchor()), which no longer feeds
# final_score at all: final_score IS base_value_score now (VOR + within-
# position projection + market swing +/- risk, see
# calculate_base_value_score()), with real, decomposable, non-uniform
# gaps (checked directly: median gap 0.49, p90 1.97, p99 7.25, max 21.19
# across the real top 250) -- exactly the natural-break signal this
# function needs, and the reason the old workaround column existed at all
# is gone. Using the stale column meant every manual projection
# correction this session moved Our Score but never the visible Tier
# label -- a real, confirmed bug, not just a staleness risk.
#
# TARGET_TIER_COUNT=10 (redesigned 2026-08-21 -- replaces a fixed
# GAP_TIER_THRESHOLD=2.5 on final_score's raw point scale). The fixed-
# threshold approach produced 19 tiers on today's real board, most of
# them 1-3-player singletons near the top -- technically real gaps, but
# too fragmented to read as a cheat sheet. The deeper problem: a single
# global point-gap threshold can't work well against this score shape,
# because it has one dominant real elbow (VOR goes to/below zero around
# rank ~65, so gaps there are naturally tiny for the rest of the board)
# sitting next to real, meaningful, LARGER gaps concentrated in the top
# ~30 picks -- no single constant threshold is simultaneously fine-
# grained enough at the top and coarse enough past the elbow. Fixed by
# picking the tier COUNT directly instead of a gap size: take the
# (TARGET_TIER_COUNT - 1) single largest real gaps anywhere in the whole
# sorted board and use those as breakpoints. This self-calibrates to
# whatever the real score distribution looks like on any given day
# (unlike a hardcoded point threshold, which would need re-tuning every
# time the underlying scoring changes) and naturally produces exactly
# TARGET_TIER_COUNT tiers every time: a handful of small, real elite
# tiers at the top (where the biggest gaps actually are) and larger,
# honest groupings further down.
TARGET_TIER_COUNT = 10


def _assign_automatic_tiers(rank_sorted_board, score_col="final_score", n_tiers=TARGET_TIER_COUNT):
    """rank_sorted_board must already be sorted best-to-worst (Rank order).
    Tier breaks are placed at the (n_tiers - 1) single largest real gaps
    in `score_col` anywhere across the whole board, so the result always
    has exactly n_tiers groups (fewer only if there are too few rows with
    a real score to support that many breaks). Returns a tier-number
    Series aligned to the input's index."""
    scores = pd.to_numeric(rank_sorted_board[score_col], errors="coerce")
    valid_idx = [i for i, s in enumerate(scores) if pd.notna(s)]
    gaps = [
        (scores.iloc[valid_idx[j]] - scores.iloc[valid_idx[j + 1]], valid_idx[j])
        for j in range(len(valid_idx) - 1)
    ]
    gaps.sort(key=lambda g: g[0], reverse=True)
    breakpoints = set(pos for _, pos in gaps[: max(n_tiers - 1, 0)])

    tiers = []
    current_tier = 1
    for i in range(len(scores)):
        tiers.append(current_tier)
        if i in breakpoints:
            current_tier += 1
    return pd.Series(tiers, index=rank_sorted_board.index)


# Automatic position-rank assignment (2026-08-21, same real gap as the
# 2026-08-17 tier fix above and the same fix pattern): position_rank
# arrived from master_players.csv as FantasyPros' own static, externally-
# curated rank -- confirmed zero references anywhere in draft_analysis.py,
# completely disconnected from this app's real scoring pipeline. Replaced
# with a real rank-within-position computed off final_score, over the
# same TOP_N-cut board Rank/tier already use, so "WR1" on this board
# means "the WR we actually have #1 by Our Score," not FantasyPros'
# separate call. Written as a bare int (not "WR7" text) -- the "pos"
# pill renderer already reconstructs the position-prefixed label from
# row["position"] + this number, same as it did for the old column.
if "final_score" in board.columns and "position" in board.columns:
    board["position_rank"] = (
        pd.to_numeric(board["final_score"], errors="coerce")
        .groupby(board["position"].astype(str).str.upper())
        .rank(ascending=False, method="first")
    )

if "final_score" in board.columns:
    board["tier"] = _assign_automatic_tiers(board)

# scoreScaled (cosmetic 0-100 display value for "Our score", the row's
# progress bar, and the card header) and grade -- both computed HERE, once,
# on the full TOP_N Overall board, same rationale as tier above: stays
# fixed when narrowing to one position rather than shifting under the
# filter. Neither touches final_score/sort order -- display-only.
if "final_score" in board.columns:
    _final_numeric = pd.to_numeric(board["final_score"], errors="coerce")
    _lo, _hi = _final_numeric.min(), _final_numeric.max()
    if pd.notna(_lo) and pd.notna(_hi) and _hi > _lo:
        board["scoreScaled"] = ((_final_numeric - _lo) / (_hi - _lo) * 100).round(1)
    else:
        board["scoreScaled"] = 50.0

    # riskScaled -- the same cosmetic 0-100 min-max treatment as scoreScaled
    # above, applied to risk_index so the displayed risk uses the full
    # 0-100 range (board-riskiest = 100, safest = 0) and reads on the same
    # scale as OUR SCORE. Computed here on the full TOP_N Overall board so
    # it stays fixed when narrowing to one position. Display only: the raw
    # risk_index column is left untouched for the Risk-adjusted sort.
    if "risk_index" in board.columns:
        _risk_numeric = pd.to_numeric(board["risk_index"], errors="coerce")
        _rlo, _rhi = _risk_numeric.min(), _risk_numeric.max()
        if pd.notna(_rlo) and pd.notna(_rhi) and _rhi > _rlo:
            board["riskScaled"] = ((_risk_numeric - _rlo) / (_rhi - _rlo) * 100).round(1)
        else:
            board["riskScaled"] = _risk_numeric.round(1)

    _pct_rank = _final_numeric.rank(pct=True) * 100

    def _grade_for_pct(p):
        if pd.isna(p):
            return "Speculative"
        if p >= 95:
            return "Elite"
        if p >= 80:
            return "Great"
        if p >= 50:
            return "Strong"
        if p >= 20:
            return "Solid"
        return "Speculative"

    board["grade"] = _pct_rank.apply(_grade_for_pct)

    # Risk-adjusted view's re-sort score -- uses the richer 4-category
    # risk_index (risk_variables.csv) rather than re-deriving from the
    # single injury_risk aggregate base_value_risk_component already nets
    # out of final_score, so this is a genuinely different resort. Display
    # heuristic only; never feeds final_score/tiering.
    _risk_for_adj = pd.to_numeric(board.get("risk_index"), errors="coerce").fillna(0.0)
    board["riskAdjScore"] = (_final_numeric.fillna(0.0) - (_risk_for_adj / 100.0) * RISK_ADJ_VIEW_PENALTY).round(3)

    # Ceiling view's re-sort score: final_score + a variance-driven upside
    # bonus (see CEILING_VIEW_BONUS). Upside (0..1) blends TD-dependence
    # (role_usage_td_score, 1-5) and the volatility diagnostic
    # (volatility_diagnostic, ~0-1.5) from risk_variables.csv, plus a lift
    # for event/big-play archetypes. Missing signals -> no bonus, so those
    # players just hold their median position. Display heuristic only.
    _td = pd.to_numeric(board.get("role_usage_td_score"), errors="coerce")
    _td_norm = ((_td - 1.0) / 4.0).clip(0.0, 1.0)
    _vol = pd.to_numeric(board.get("volatility_diagnostic"), errors="coerce")
    _vol_norm = (_vol / 1.0).clip(0.0, 1.0)
    # Blend whichever signals a player has (mean of present components).
    _upside = pd.concat([_td_norm, _vol_norm], axis=1).mean(axis=1, skipna=True)
    if "Scoring Type" in board.columns:
        _event_lift = board["Scoring Type"].astype(str).eq("TD / big play").map({True: 0.15, False: 0.0})
        _upside = (_upside.fillna(0.0) + _event_lift).clip(0.0, 1.0)
    _upside = _upside.fillna(0.0)
    # Weight the bonus by within-board relevance (percentile of final_score).
    # final_score is VOR-based and mostly negative (replacement baseline),
    # densely packed at the bottom -- a flat bonus there would catapult
    # irrelevant deep-bench TD vultures over startable players. The
    # percentile weight concentrates the reshuffle among genuinely
    # draftable players, where upside actually matters.
    _relevance = _final_numeric.rank(pct=True).fillna(0.0)
    board["ceilingScore"] = (_final_numeric.fillna(0.0) + _upside * CEILING_VIEW_BONUS * _relevance).round(3)

if position_filter != "Overall":
    board = board[board["position"].astype(str).str.upper() == position_filter]

if risk_filter != "ALL":
    # Filter on the same scaled value the row displays (riskScaled when
    # present, else raw risk_index) so the Low/Medium/High tabs match the
    # band pills shown on the board.
    _risk_for_filter = board["riskScaled"] if "riskScaled" in board.columns else board.get("risk_index")
    board = board[
        pd.to_numeric(_risk_for_filter, errors="coerce").apply(_risk_band) == risk_filter
    ]

if search_query:
    board = board[
        board["player_name"].astype(str).str.contains(search_query, case=False, na=False)
    ]

# View controls sort + rank renumbering + whether tier bands show, entirely
# replacing the old column-header sort-link/query-param mechanism -- the
# new design has no per-column sort, only this 3-way toggle.
if view == "ceiling" and "ceilingScore" in board.columns:
    board = board.sort_values("ceilingScore", ascending=False, na_position="last", kind="stable")
    board["display_rank"] = range(1, len(board) + 1)
    show_tier_bands = False
elif view == "riskAdj" and "riskAdjScore" in board.columns:
    board = board.sort_values("riskAdjScore", ascending=False, na_position="last", kind="stable")
    board["display_rank"] = range(1, len(board) + 1)
    show_tier_bands = False
else:
    # Median: tier/Rank order (tiers are already contiguous in Rank order
    # by construction -- see _assign_automatic_tiers()). Tier bands show
    # only here, and only with no active search (search disables them,
    # same as the old sort-link behavior did).
    if "tier" in board.columns:
        board = board.sort_values(["tier", "Rank"], na_position="last", kind="stable")
    board["display_rank"] = board["Rank"]
    show_tier_bands = not bool(search_query)

with _export_col:
    # Clean, spreadsheet-friendly export that mirrors the on-screen board
    # rather than dumping the full internal engine frame: the visible table
    # columns plus the key card summary fields (Tier / Grade / Archetype /
    # Risk band), in the current sort order, already narrowed by the active
    # position/risk/search filters, and under the active scoring model. The
    # same display values the UI renders (display_rank, scoreScaled,
    # riskScaled, grade, the vs-ADP delta) so the file matches what's shown.
    _vs_adp = pd.to_numeric(board.apply(_rank_delta, axis=1), errors="coerce").round(1)
    _risk_scaled = pd.to_numeric(board.get("riskScaled"), errors="coerce")
    export_df = pd.DataFrame(
        {
            "Rank": board.get("display_rank"),
            "Player": board.get("player_name"),
            "Pos": board.get("position"),
            "Team": board.get("team"),
            "Tier": board.get("tier"),
            "Grade": board.get("grade"),
            "Archetype": board.get("Archetype"),
            "ADP": pd.to_numeric(board.get("adp"), errors="coerce").round(1),
            "Projection": pd.to_numeric(board.get("projection_points"), errors="coerce").round(1),
            "Our Score": pd.to_numeric(board.get("scoreScaled"), errors="coerce"),
            "Risk": _risk_scaled,
            "Risk Band": _risk_scaled.apply(lambda v: RISK_LABELS.get(_risk_band(v), "")),
            "vs ADP": _vs_adp,
            "vs Vegas (spots)": pd.to_numeric(board.get("position_rank_gap"), errors="coerce").round(1),
        }
    )
    _score_tag = "dads_league" if score_mode == "dads" else "standard"
    st.download_button(
        "Export CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"guaranteed_play_rankings_{_score_tag}.csv",
        mime="text/csv",
        width="stretch",
        type="primary",
    )
    _score_note = "Dad's League scoring" if score_mode == "dads" else "Standard (Half-PPR) scoring"
    st.markdown(
        f'<div class="gp-export-caption">Exports the {len(board)} rows shown &middot; {_score_note}</div>',
        unsafe_allow_html=True,
    )
with _count_col:
    st.markdown(
        f'<div class="gp-count">{len(board)} / {len(rankings_df.head(TOP_N))} PLAYERS</div>',
        unsafe_allow_html=True,
    )

st.markdown(render_rankings_table(board, view, show_tier_bands), unsafe_allow_html=True)

st.markdown(
    '<div class="gp-footer-note">'
    'Base Value weights are analyst-set and sensitivity-tested, not backtested against '
    'real outcomes. Floor/ceiling are the projection model\'s own real range, not a '
    'simulation.</div>',
    unsafe_allow_html=True,
)

# Replacement-level rows are scored off a fallback floor rather than a real
# FantasyPros projection; surfacing the count keeps that from reading as a
# real forecast.
if "projection_source" in board.columns:
    fallback = int((board["projection_source"] == "replacement_fallback").sum())
    if fallback:
        st.caption(
            f"{fallback} of these {len(board)} players have no real projection and are "
            "scored at replacement level."
        )
