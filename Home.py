"""
Top 250 rankings -- the static preseason board.

This page itself has no navigation, no draft-state controls, and no draft
actions. Draft Mode has been restored to pages/, so Streamlit's multipage
nav is active again; the remaining pages (Player Cards, Team Outlook, Draft
Lab, Player Compare, Tier Desperation, Component Audit) are still parked in
pages_archive/ rather than deleted, and moving any of them into pages/
brings them back the same way.

Scoring is unchanged -- this reads the same draftkit engine and the same
data/processed/master_players.csv that Draft Mode uses. Because there is no
live draft state here, the board is evaluated at its draft-start position
(empty roster, pick 1), which is what a static preseason ranking list
should show.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from draftkit.age_context import age_note
from draftkit.archetypes import archetype_label, risk_profile
from draftkit.data_access import load_players_df
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import init_session_state

TOP_N = 250

POSITION_CHIPS = ["Overall", "QB", "RB", "WR", "TE"]

POSITION_COLORS = {
    "QB": "#22c55e",
    "RB": "#ef4444",
    "WR": "#3b82f6",
    "TE": "#a855f7",
}

# Tier band colors, cycled. The reference used S/A/B/C letter grades, but this
# data carries 16 numeric tiers across the top 250 with uneven sizes, so the
# bands are labeled "Tier N" rather than inventing a grade scale.
TIER_BAND_COLORS = ["#ec4899", "#a855f7", "#3b82f6", "#22c55e", "#f59e0b", "#14b8a6"]

# Table columns: (query-param key, header label, dataframe column, pill style,
# hover tooltip). Labels are written for someone new to fantasy -- no bare
# jargon or internal abbreviations. The query-param key is what appears in
# ?sort=... and is kept short and stable so a bookmarked sort URL survives
# header-label changes.
TABLE_COLUMNS = [
    ("rank", "RANK", "Rank", "rank",
     "Our overall ranking, 1 = best. This is the order we'd draft in."),
    ("player", "PLAYER", "player_name", "player", ""),
    ("posrk", "POSITION RANK", "position_rank", "pos",
     "Rank within his own position. RB1 = the best running back."),
    ("team", "TEAM", "team", "muted", "NFL team."),
    ("age", "AGE", "Age", "blue", "Player age this season."),
    ("adp", "AVG DRAFT PICK", "adp", "amber",
     "ADP -- the average pick number where this player actually gets drafted. "
     "Lower means he goes earlier."),
    ("fp", "EXPERT RANK", "expert_rank", "muted",
     "Where fantasy experts (FantasyPros consensus) rank him overall."),
    ("book", "VEGAS RANK", "sportsbook_position_rank", "purple",
     "His rank within his position based on sportsbook betting lines -- what "
     "Vegas expects him to produce."),
    ("mkt", "VEGAS vs ADP", "position_rank_gap", "signed",
     "How much Vegas disagrees with where he's being drafted. Green/positive = "
     "Vegas likes him more than his draft spot suggests (possible bargain). "
     "Red/negative = Vegas likes him less (possible reach)."),
    # PROJECTED PTS (key "proj", column projection_points) hidden 2026-08-21
    # (user: "its the same as the column beside") -- since
    # apply_model_projection_override() started writing corrections
    # directly into projection_points itself, this cell shows the exact
    # same number as MODEL PROJ for every corrected player, which by this
    # point in the season is most of the relevant board. MODEL PROJ already
    # carries its own adjustment-percent badge (see the model_projection_points
    # render branch below), so nothing is lost by dropping this one -- the
    # underlying column and its render branch are left in place, just no
    # longer in TABLE_COLUMNS, in case a future page wants the raw value
    # without MODEL PROJ's framing.
    ("mdl", "MODEL PROJ", "model_projection_points", "green",
     "Alternate full-season projection from a separate, backtested model "
     "(research/validation_v1, projection_model_iteration_plan.pdf) -- RB/WR/TE "
     "only. QB is deliberately excluded: that position's version of this model "
     "validated WORSE than a simple ADP-based baseline in real backtesting, so "
     "it isn't shown rather than displaying a signal known to mislead. Even at "
     "RB/WR/TE this is a newer, less-proven number than PROJECTED PTS -- shown "
     "alongside it for comparison, not as a replacement."),
    ("score", "OUR SCORE", "final_score", "amber",
     "Our engine's overall 0-100 rating, combining projection, position "
     "scarcity, value vs ADP, and tier urgency."),
    ("arch", "ARCHETYPE", "rb_archetype_primary", "archetype",
     "Role taxonomy, real 2025 usage-based. RBs: a usage tier -- Bellcow, "
     "Committee, Handcuff, or No Role -- by real backfield touch share; "
     "'(2-Down)'/'(3-Down)' shows real third-down snap involvement for "
     "Bellcow/Committee backs. WRs: Alpha, Boom-Bust, High-Floor, or "
     "Complementary usage tier by real target share/aDOT/catch rate. "
     "Both positions also carry an independent secondary trait tag "
     "(RB: Goal-Line, Explosive, Receiving; WR: Deep-Threat, YAC, "
     "Red-Zone, Rushing) plus, for WR, a real QB-context flag. A small "
     "tag flags a real secondary trait that doesn't rise to its own "
     "archetype. Hover for the real numbers behind the call. "
     "Unconfirmed means real data doesn't yet support a confident read "
     "(new/limited-usage player), not that he's unimportant. Incoming "
     "rookies with no NFL snaps yet show a dashed 'Projected: X' badge "
     "(pre-season model, v1/unvalidated -- see hover) that shifts to a "
     "'X -- Y% confirmed' hybrid badge as real usage accumulates, then "
     "becomes a standard badge once confirmed."),
    ("risk", "RISK", "risk_index", "risk",
     "Composite risk score, 0-100, HIGHER = riskier. Combines four "
     "categories -- Injury (recency/severity-weighted), Role/Usage/TD "
     "(target share, snap share, depth-chart competition, TD-rate "
     "dependence), Offense Environment (team EPA/play, pace, scoring), and "
     "Schedule/Weather/Venue (dome/cold games, playoff-week SOS, bye, "
     "primetime/short-week load) -- weighted differently by position (e.g. "
     "RBs weight injury/role heaviest, QBs/WRs weight offense environment "
     "heaviest). Market mispricing and week-to-week volatility are "
     "reported but deliberately NOT part of this score -- one's a value "
     "signal, the other a symptom, not independent risk. Hover a value for "
     "the full breakdown. Live draft-pick risk-stacking warnings are in "
     "the standalone scorecard: python -m draftkit.risk_cli."),
]

SORTABLE_COLUMNS = {key: column for key, _, column, _, _ in TABLE_COLUMNS}
# MODEL PROJ (key "mdl") sorts on the effective displayed value (adjusted /
# fallback / raw, see model_projection_effective above), not the raw
# model_projection_points column the cell renderer only falls back to when
# nothing else is present. Overridden here, after the dict comprehension,
# so the render dispatch (which switches on column == "model_projection_points")
# is untouched -- only the sort key changes.
SORTABLE_COLUMNS["mdl"] = "model_projection_effective"


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
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 100%; }
  /* Streamlit's own h1 rule wins on specificity, hence the explicit
     element+class selector and !important -- without them this renders as
     44px white instead of the intended amber. */
  h1.gp-title {
    color: #f59e0b !important; font-size: 28px !important; font-weight: 800 !important;
    margin: 0 0 12px !important; padding: 0 !important;
  }
  .gp-count { color: #7d8ea3; font-size: 12px; text-align: right; padding-top: 8px; }

  /* Position chips -- Streamlit buttons restyled. Active chip uses the native
     primary type so the state survives without extra CSS plumbing. */
  div[data-testid="stButton"] button {
    border-radius: 8px; font-size: 13px; font-weight: 700; padding: 4px 0;
    border: 1px solid #24303f; background: #131c29; color: #9fb0c3;
  }
  div[data-testid="stButton"] button[kind="primary"] {
    background: #f59e0b; color: #10151d; border-color: #f59e0b;
  }

  .gp-table-wrap { overflow-x: auto; margin-top: 10px; }
  table.gp-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.gp-table th {
    text-align: center; padding: 9px 8px; color: #7d8ea3;
    font-size: 11px; font-weight: 700; letter-spacing: .05em;
    border-bottom: 1px solid #1e2836; white-space: nowrap;
  }
  table.gp-table th a { color: #7d8ea3; text-decoration: none; }
  table.gp-table th a:hover { color: #e6edf3; }
  table.gp-table th.sorted a { color: #f59e0b; }
  table.gp-table td {
    text-align: center; padding: 5px 8px; border-bottom: 1px solid #151d29;
  }
  table.gp-table td.col-rank { color: #7d8ea3; font-size: 11px; width: 34px; }
  table.gp-table td.col-player {
    text-align: left; font-weight: 600; color: #e6edf3; white-space: nowrap;
    background: #101825; border-radius: 5px;
  }
  table.gp-table tr:hover td { background: #131c29; }

  .tier-band td { padding: 0 !important; border: none !important; }
  .tier-band-inner {
    display: block; width: 100%; margin: 10px 0 4px;
    padding: 5px 12px; border-radius: 6px;
    font-size: 12px; font-weight: 800; color: #0b0f16;
  }

  .pill {
    position: relative;
    display: inline-block; min-width: 42px; padding: 2px 8px; border-radius: 5px;
    font-size: 11.5px; font-weight: 700; line-height: 1.5;
  }
  .pill-muted  { background: #1b2534; color: #9fb0c3; }
  .pill-blue   { background: #16273f; color: #60a5fa; }
  .pill-amber  { background: #33260f; color: #fbbf24; }
  .pill-purple { background: #291a3d; color: #c084fc; }
  .pill-green  { background: #10291c; color: #4ade80; }
  .pill-red    { background: #331515; color: #f87171; }
  .pill-empty  { color: #3d4a5c; }

  /* PROJECTED PTS manual-adjustment marker (draft_analysis.py's
     PROJECTION_MANUAL_ADJUSTMENTS) -- small, visibly distinct from the pill
     itself, hover reveals the real note/source/date. Reuses the existing
     pill red/green palette rather than inventing new colors. */
  .proj-adj {
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 4px; cursor: default; vertical-align: middle;
  }
  .proj-adj-pos { background: #10291c; color: #4ade80; }
  .proj-adj-neg { background: #331515; color: #f87171; }

  /* MODEL PROJ diagnosed-defect correction marker
     (model_proj_staleness_fix_plan.pdf, Step 4) -- deliberately a DIFFERENT
     color from both proj-adj-pos/neg (narrative override, PROJECTED PTS)
     and continuity-flag's bare "!" (unresolved, verify-before-trusting).
     This one means "known bug, already fixed for this player" -- reads as
     resolved/corrected, not uncertain. */
  .model-corrected { background: #0f2a3d; color: #38bdf8; }

  /* MODEL PROJ rookie-score fallback marker
     (rb_model_fix_plan.pdf, Phase 1) -- a THIRD distinct color, separate
     from model-corrected (blue, "known bug already fixed") and
     continuity-flag (bare "!", "verify this"). This one means "not a real
     outcome-trained prediction at all -- substituted from the pre-draft
     rookie composite score." */
  .rookie-fallback { background: #2a1f3d; color: #c4b5fd; }

  /* Risk scorecard hover popover -- CSS-only, no JS, matching how this
     table is already 100% server-rendered HTML. Colors reuse the existing
     dark-theme palette above (pill-red/-amber/-green, table bg/border
     tones) rather than introducing a second theme. */
  .pill-risk { cursor: default; }
  .scorecard {
    position: absolute; top: 130%; right: -10px; width: 300px;
    background: #101825; border: 1px solid #1e2836; border-radius: 10px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.5); padding: 14px;
    z-index: 50; opacity: 0; transform: translateY(-6px);
    pointer-events: none; transition: opacity .12s ease, transform .12s ease;
    text-align: left; font-weight: 400; white-space: normal;
  }
  .pill-risk:hover .scorecard { opacity: 1; transform: translateY(0); pointer-events: auto; }

  .sc-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #1e2836;
  }
  .sc-name { font-size: 13px; font-weight: 700; color: #e6edf3; }
  .sc-meta { font-size: 11px; color: #7d8ea3; margin-top: 2px; }
  .sc-overall { text-align: right; }
  .sc-overall .num { font-size: 20px; font-weight: 800; line-height: 1; }
  .sc-overall .of100 { font-size: 11px; font-weight: 500; color: #7d8ea3; }
  .sc-overall .lbl { font-size: 10px; color: #7d8ea3; text-transform: uppercase; letter-spacing: .04em; }

  .sc-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .sc-row:last-of-type { margin-bottom: 0; }
  .sc-cat-label { width: 104px; font-size: 11px; color: #7d8ea3; flex-shrink: 0; }
  .sc-bar-track { flex: 1; height: 6px; background: #1b2534; border-radius: 3px; overflow: hidden; }
  .sc-bar-fill { height: 100%; border-radius: 3px; }
  .sc-cat-score { width: 32px; text-align: right; font-size: 11px; font-weight: 700; color: #e6edf3; flex-shrink: 0; }

  .sc-divider { margin: 12px 0 10px; border-top: 1px dashed #1e2836; }
  .sc-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .sc-badge { font-size: 10px; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
  .sc-badge-avoid   { background: #331515; color: #f87171; }
  .sc-badge-value   { background: #10291c; color: #4ade80; }
  .sc-badge-fair    { background: #1b2534; color: #9fb0c3; }
  .sc-badge-chronic { background: #33260f; color: #fbbf24; }

  .sc-note { font-size: 10.5px; color: #7d8ea3; line-height: 1.5; }
  .sc-note b { color: #e6edf3; }

  /* RB archetype badge + popover -- reuses the .scorecard/.sc-* popover
     scaffold above (same CSS-only hover mechanism), new badge colors only.
     Kept close to existing pill tokens where they already overlap
     (blue/purple/green/red already exist above); orange/gray are new. */
  .arch-wrap { position: relative; display: inline-block; }
  .arch-badge {
    display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px;
    border-radius: 12px; font-size: 11px; font-weight: 700; cursor: default;
    white-space: nowrap;
  }
  .arch-Bellcow      { background: #16273f; color: #60a5fa; }
  .arch-Committee    { background: #291a3d; color: #c084fc; }
  .arch-Handcuff     { background: #232833; color: #8b95a5; }
  /* Class name stays "Unconfirmed" (shared with WR's own unconfirmed
     styling below) even though RB's displayed badge TEXT is "No Role" --
     see Home.py's _render_cell()/_archetype_scorecard_html() RB-specific
     label override. Goal-Line/Explosive/Receiving badge colors removed --
     RB primary is a usage tier only now, those are secondary lean tags
     (see .arch-lean-tag styling), never a badge background anymore. */
  .arch-Unconfirmed  { background: #33260f; color: #fbbf24; }
  /* WR archetype badge colors -- shares the same .arch-wrap/.arch-badge/
     .scorecard scaffold above; Unconfirmed is shared with RB (same CSS
     class, different displayed text for RB). New tier names only. */
  .arch-Alpha         { background: #3a1a2e; color: #f472b6; }
  .arch-Complementary { background: #232833; color: #8b95a5; }
  /* Boom-Bust/High-Floor (claude_code_plan_possession_split.pdf) -- replace
     Possession everywhere, including rookie_projection.py's own pre-draft
     tier (2026-08-15) -- .arch-Possession removed, nothing emits it. */
  .arch-Boom-Bust     { background: #331515; color: #f87171; }
  .arch-High-Floor    { background: #10291c; color: #4ade80; }
  /* QB rushing-style tiers (claude_code_plan_qb_archetypes.pdf). */
  .arch-Pocket-Passer { background: #16273f; color: #60a5fa; }
  .arch-Balanced      { background: #232833; color: #8b95a5; }
  .arch-Dual-Threat   { background: #331515; color: #f87171; }
  /* TE receiving/blocking tiers (plan_te_archetypes.pdf). Balanced reuses
     the class above (same "Balanced" label/color). */
  .arch-Receiving-TE  { background: #3a1a2e; color: #f472b6; }
  .arch-Blocking-TE   { background: #232833; color: #8b95a5; }
  /* TE role_profile (plan_te_role_profile_elite.pdf) -- second-pass split
     within receiving_te only, replaces the flat Receiving-TE badge for
     those rows. Display label has a real space ("Elite TE") per the
     user's confirmed wording, unlike every other hyphenated badge above --
     badge_class strips the space (see _render_cell()'s te_primary branch)
     so the CSS class itself stays a single token. */
  .arch-Elite-TE         { background: #3a1a2e; color: #f472b6; }
  .arch-Complementary-TE { background: #232833; color: #8b95a5; }
  /* Rookie projection badge states (draftkit/rookie_projection.py) --
     layered AFTER the tier-color classes above so they win the cascade
     (same specificity, later source order) regardless of which tier
     they're paired with, matching the user-supplied mockup's own
     .pill-projected/.pill-blended layering over .pill-bellcow/.pill-alpha.
     Values taken directly from that mockup, already visually approved. */
  .arch-projected {
    border: 1px dashed #6b8afd; background: transparent; color: #a9bdff; font-style: italic;
  }
  .arch-blended {
    background: linear-gradient(90deg, rgba(217,164,65,0.28) 0%, rgba(23,50,79,0.9) 60%);
    color: #f0c878;
  }
  .beta-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #e0b23a; margin-left: 5px; vertical-align: middle;
  }
  .arch-lean-tag { font-weight: 500; opacity: 0.8; }
  .arch-wrap:hover .scorecard { opacity: 1; transform: translateY(0); pointer-events: auto; }
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


def _pill(text, style, title=None):
    title_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
    if text is None:
        return f'<span class="pill pill-empty"{title_attr}>--</span>'
    return f'<span class="pill pill-{style}"{title_attr}>{html.escape(str(text))}</span>'


RISK_CATEGORY_META = [
    ("injury_score", "Injury"),
    ("role_usage_td_score", "Role/Usage/TD"),
    ("offense_environment_score", "Offense Env"),
    ("schedule_weather_venue_score", "Schedule/Wx"),
]

# Same >66/>34 split as the overall risk_index band, applied to each
# category's own 0-100 conversion, so a bar's color always means the same
# thing as the pill's color.
def _risk_bar_color(score100):
    if score100 > 66:
        return "#f87171"
    if score100 > 34:
        return "#fbbf24"
    return "#4ade80"


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


def _risk_scorecard_html(row, overall_number, band):
    """Builds the nested <div class="scorecard"> shown by the pure-CSS
    `.pill-risk:hover .scorecard { opacity: 1 }` rule (see RANKINGS_CSS)."""
    name = html.escape(str(row.get("player_name") or ""))
    position = html.escape(str(row.get("position") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"

    cat_rows = []
    best_label, best_score100 = None, -1
    for col, label in RISK_CATEGORY_META:
        raw = row.get(col)
        if raw is None or pd.isna(raw):
            continue
        score100 = round((float(raw) - 1) / 4 * 100)
        score100 = max(0, min(100, score100))
        if score100 > best_score100:
            best_label, best_score100 = label, score100
        cat_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(label)}</div>'
            f'<div class="sc-bar-track"><div class="sc-bar-fill" '
            f'style="width:{score100}%;background:{_risk_bar_color(score100)}"></div></div>'
            f'<div class="sc-cat-score">{score100}</div></div>'
        )

    badges = (
        _market_badge(row.get("value_signal_market"))
        + _chronic_injury_badge(row.get("games_missed_by_season"))
        + _manual_override_badge(row.get("injury_override_note"))
        + _manual_override_badge(row.get("role_usage_td_override_note"))
    )
    badges_html = f'<div class="sc-badges">{badges}</div>' if badges else ""

    note = (
        f"Driven mainly by <b>{html.escape(best_label)}</b> ({best_score100}/100)."
        if best_label else ""
    )

    # Acute/chronic injury split (plan_risk_index_reweight_dynamic.pdf
    # Phase 2) -- disclosed so a hover shows WHY the Injury bar reads what
    # it does, not just the blended number. Only rendered when both real
    # sub-scores are present (older risk_variables.csv builds won't have
    # them).
    acute = row.get("injury_acute_score")
    chronic = row.get("injury_chronic_score")
    injury_split_note = ""
    if pd.notna(acute) and pd.notna(chronic):
        driver = "recent/acute" if float(acute) >= float(chronic) else "career/chronic pattern"
        injury_split_note = (
            f"Injury reads as {round((max(float(acute), float(chronic)) - 1) / 4 * 100)}/100, "
            f"driven by the <b>{driver}</b> component -- acute {round((float(acute) - 1) / 4 * 100)}/100, "
            f"chronic {round((float(chronic) - 1) / 4 * 100)}/100."
        )

    band_color = {"red": "#f87171", "amber": "#fbbf24", "green": "#4ade80"}[band]
    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">{position} · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="num" style="color:{band_color}">{round(overall_number)}<span class="of100">/100</span></div>'
        '<div class="lbl">Risk Index</div></div>'
        "</div>"
        f'{"".join(cat_rows)}'
        '<div class="sc-divider"></div>'
        f"{badges_html}"
        f'<div class="sc-note">{note}</div>'
        f'{f"<div class=\"sc-note\">{injury_split_note}</div>" if injury_split_note else ""}'
        "</div>"
    )


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


def _archetype_scorecard_html(row, primary: str) -> str:
    """Builds the nested <div class="scorecard"> shown by the pure-CSS
    `.arch-wrap:hover .scorecard { opacity: 1 }` rule (see RANKINGS_CSS).
    Shows the real metrics that drove this specific classification, not a
    fit score for every archetype -- this taxonomy only computes real
    pass/fail thresholds per archetype, not a continuous comparative
    score across all six (see draftkit/rb_archetypes.py)."""
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"
    # RB-specific "No Role" override -- see _render_cell's matching comment.
    label = "No Role" if primary == "unconfirmed" else ARCHETYPE_LABELS.get(primary, primary or "Unconfirmed")

    metric_rows = []
    for col, metric_label, kind in ARCHETYPE_METRICS.get(primary, []):
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        metric_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
        )

    lean = row.get("rb_lean")
    lean_note = ""
    if isinstance(lean, str) and lean in LEAN_LABELS:
        lean_note = f"Also shows a real <b>{LEAN_LABELS[lean]}</b> -- a secondary trait that doesn't rise to its own archetype."

    unconfirmed_note = (
        "Real data doesn't yet clear the sample floor for a confident archetype read "
        "(new player or limited usage) -- not a statement that he's unimportant."
        if primary == "unconfirmed" else ""
    )

    note = lean_note or unconfirmed_note or ""
    note_html = f'<div class="sc-note">{note}</div>' if note else ""

    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">RB · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="lbl" style="font-size:12px;color:#e6edf3;font-weight:700;">{html.escape(label)}</div>'
        '</div>'
        "</div>"
        f'{"".join(metric_rows)}'
        f'{"<div class=\"sc-divider\"></div>" if note else ""}'
        f"{note_html}"
        f'{_prospect_profile_html(row)}'
        "</div>"
    )


def _wr_archetype_scorecard_html(row, primary: str) -> str:
    """WR counterpart to _archetype_scorecard_html() -- same popover
    scaffold, WR-shaped inputs (usage tier + independent leans + QB-context
    flag instead of RB's primary/down_split/lean). Shows the real metrics
    behind classify_primary(), not a fit score for every tier (see
    draftkit/wr_archetypes.py -- this taxonomy only computes real pass/fail
    band checks, not a continuous comparative score)."""
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"
    label = ARCHETYPE_LABELS.get(primary, primary or "Unconfirmed")

    metric_rows = []
    for col, metric_label, kind in WR_ARCHETYPE_METRICS.get(primary, []):
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        metric_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
        )

    leans = [l for l in str(row.get("wr_leans") or "none").split(",") if l in LEAN_LABELS]
    lean_note = ""
    if leans:
        lean_text = ", ".join(f"<b>{LEAN_LABELS[l]}</b>" for l in leans)
        lean_note = f"Also carries a real {lean_text} -- an independent boom-trait tag, not tied to usage tier."

    qb_label = QB_CONTEXT_LABELS.get(row.get("wr_qb_context"))
    qb_note = (
        f"Plays in a real <b>{qb_label}</b>, based on his team's real season offensive EPA/play rank."
        if qb_label else ""
    )

    unconfirmed_note = (
        "Real data doesn't yet clear the sample floor (30 targets, 6 games) for a confident usage-tier "
        "read -- not a statement that he's unimportant."
        if primary == "unconfirmed" else ""
    )

    notes = [n for n in (lean_note, qb_note, unconfirmed_note) if n]
    note_html = "".join(f'<div class="sc-note">{n}</div>' for n in notes)

    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">WR · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="lbl" style="font-size:12px;color:#e6edf3;font-weight:700;">{html.escape(label)}</div>'
        '</div>'
        "</div>"
        f'{"".join(metric_rows)}'
        f'{"<div class=\"sc-divider\"></div>" if notes else ""}'
        f"{note_html}"
        f'{_prospect_profile_html(row)}'
        "</div>"
    )


def _qb_archetype_scorecard_html(row, primary: str) -> str:
    """QB counterpart to _archetype_scorecard_html()/_wr_archetype_scorecard_html().
    Single-axis taxonomy (rushing_fantasy_pct only) -- no leans, no QB-context
    flag (that's WR's own tag for the QB it plays behind, not relevant to a
    QB's own row). seasons_used isn't a numeric metric row (no "raw string"
    kind exists in _archetype_metric_value()) -- surfaced in the note text
    instead, same slot _wr_archetype_scorecard_html() uses for qb_note/etc."""
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"
    label = ARCHETYPE_LABELS.get(primary, primary or "Unconfirmed")

    metric_rows = []
    for col, metric_label, kind in QB_ARCHETYPE_METRICS.get(primary, []):
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        metric_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
        )

    seasons_used = row.get("qb_seasons_used")
    seasons_note = (
        f"Pooled across real {html.escape(str(seasons_used).replace(',', ', '))} season(s) -- "
        "QB rushing volume swings more year-to-year than RB/WR usage (contract situations, scheme "
        "changes), so a single season isn't used alone."
        if isinstance(seasons_used, str) and seasons_used else ""
    )
    unconfirmed_note = (
        "Real data doesn't yet clear the sample floor (100 pass attempts, 4 games, pooled across up "
        "to 2 real seasons) for a confident read -- not a statement that he's unimportant."
        if primary == "unconfirmed" else ""
    )

    notes = [n for n in (seasons_note, unconfirmed_note) if n]
    note_html = "".join(f'<div class="sc-note">{n}</div>' for n in notes)

    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">QB · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="lbl" style="font-size:12px;color:#e6edf3;font-weight:700;">{html.escape(label)}</div>'
        '</div>'
        "</div>"
        f'{"".join(metric_rows)}'
        f'{"<div class=\"sc-divider\"></div>" if notes else ""}'
        f"{note_html}"
        "</div>"
    )


def _te_archetype_scorecard_html(row, primary: str) -> str:
    """TE counterpart to _qb_archetype_scorecard_html(). target_share is
    the real primary discriminator; snap_share is a real-involvement
    confidence gate, not a second axis (see draftkit/te_archetypes.py's
    module docstring for the corrected-anchor reasoning)."""
    name = html.escape(str(row.get("player_name") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"
    role_profile = row.get("te_role_profile")
    if primary == "receiving_te" and isinstance(role_profile, str) and role_profile in TE_ROLE_PROFILE_LABELS:
        label = TE_ROLE_PROFILE_LABELS[role_profile]
    else:
        label = ARCHETYPE_LABELS.get(primary, primary or "Unconfirmed")

    metric_rows = []
    for col, metric_label, kind in TE_ARCHETYPE_METRICS.get(primary, []):
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        metric_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
        )

    leans = [l for l in str(row.get("te_leans") or "none").split(",") if l in LEAN_LABELS]
    lean_note = ""
    if leans:
        lean_text = ", ".join(f"<b>{LEAN_LABELS[l]}</b>" for l in leans)
        lean_note = f"Also carries a real {lean_text} -- an independent boom-trait tag, not tied to usage tier."

    unconfirmed_note = (
        "Real data doesn't yet clear the sample floor (4 games, 30% snap share) for a confident "
        "read -- not a statement that he's unimportant."
        if primary == "unconfirmed" else ""
    )

    notes = [n for n in (lean_note, unconfirmed_note) if n]
    note_html = "".join(f'<div class="sc-note">{n}</div>' for n in notes)

    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">TE · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="lbl" style="font-size:12px;color:#e6edf3;font-weight:700;">{html.escape(label)}</div>'
        '</div>'
        "</div>"
        f'{"".join(metric_rows)}'
        f'{"<div class=\"sc-divider\"></div>" if notes else ""}'
        f"{note_html}"
        "</div>"
    )


def _format_height_inches(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    total = float(value)
    feet, inches = divmod(round(total), 12)
    return f'{feet}\'{inches}"'


def _prospect_profile_html(row) -> str:
    """Real-only, read-only college/draft context -- draft pick, school,
    combine measurables, college dominator, breakout age -- for anyone with
    real data in rookie_inputs.csv (2026 board) or backtest_rookie_inputs.csv
    (2023-2025 real veterans), merged in with a "rookie_" prefix regardless
    of the player's current rookie_status (see _rankings()).

    Narrative context only -- does NOT feed OUR SCORE/risk_index (traced
    calculate_final_recommendation_score directly: it never reads any
    archetype field, confirmed via full-file grep before this was built)
    and does NOT reintroduce college signal beyond what rookie_sample_weight
    already governs for the badge/tag above it. Renders nothing if no real
    draft_pick is on file -- no fabricated placeholders."""
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

    rows_html = [
        f'<div class="sc-row"><div class="sc-cat-label">Draft Pick</div>'
        f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{pick_label}</div></div>'
    ]
    if isinstance(college, str) and college:
        rows_html.append(
            f'<div class="sc-row"><div class="sc-cat-label">College</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{html.escape(college)}</div></div>'
        )
    combine_bits = [b for b in (height, f"{weight} lbs" if weight else None, f"{forty}s 40yd" if forty else None) if b]
    if combine_bits:
        rows_html.append(
            f'<div class="sc-row"><div class="sc-cat-label">Combine</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{" / ".join(combine_bits)}</div></div>'
        )
    if dominator_str:
        rows_html.append(
            f'<div class="sc-row"><div class="sc-cat-label">College Dominator</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{dominator_str}</div></div>'
        )
    if breakout_age:
        rows_html.append(
            f'<div class="sc-row"><div class="sc-cat-label">Breakout Age</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{breakout_age}</div></div>'
        )
    composite = _fmt(row.get("rookie_composite"), 1)
    if composite:
        rows_html.append(
            f'<div class="sc-row"><div class="sc-cat-label" style="font-weight:700;color:#e6edf3;">Composite</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;font-weight:700;color:#e6edf3;">'
            f'{composite} <span style="font-weight:400;color:#8b949e;font-size:11px;">(pre-draft estimate, unvalidated)</span>'
            f'</div></div>'
        )

    return (
        '<div class="sc-divider"></div>'
        '<div class="sc-note" style="font-weight:700;color:#e6edf3;margin-bottom:4px;">PROSPECT PROFILE</div>'
        f'{"".join(rows_html)}'
        '<div class="sc-note" style="margin-top:4px;">Real college/draft context, for reference -- '
        "not additional weight beyond what's shown above.</div>"
    )


def _rookie_archetype_scorecard_html(row, status: str) -> str:
    """Popover for a rookie_display_status of 'projected' or 'blended' --
    draftkit/rookie_projection.py's real composite + component breakdown.
    'status' here is the "unconfirmed"-aware display status: it matches
    the real validation status/tag for anyone with a real, DECISIVE
    confirmed archetype, and only diverges when the real tag is literally
    "unconfirmed" (real snaps exist, but no real usage tier was ever
    crossed) -- in which case it falls back to the college-based
    projection instead of showing the bare word "Unconfirmed" (see
    build_rookie_projections.py's dual-blend docstring; a
    years-since-draft version of this same double-computation pattern was
    tried and reverted for hedging ALREADY-decisive tags, which this does
    not do).

    Shows real usage metrics whenever a real rb_archetype_primary/
    wr_archetype_primary exists at all -- including the literal value
    "unconfirmed" -- not just when status=="blended", since a player can
    have real (if inconclusive) NFL snaps driving this popover's fallback.
    Reuses the exact same ARCHETYPE_METRICS/WR_ARCHETYPE_METRICS dicts and
    _archetype_metric_value() the plain confirmed popover uses (including
    each dict's own "unconfirmed" entry, which is real metric rows only --
    games/touches/opportunity_share -- never a label), since
    rb_archetype_primary/wr_archetype_primary are already merged onto every
    row regardless of rookie status.

    Explicitly labeled as an unvalidated v1 model (rookie_integration_plan
    Phase 0 finding, verified against the real codebase before wiring this
    in): the backtest ran and was reported honestly
    (draftkit/scripts/run_rookie_backtest.py), but threshold/weight
    retuning and validating draft_capital_score() against a real NFL
    draft-trade-value chart haven't happened -- no such chart data exists
    anywhere in this repo. Showing the composite with no caveat would imply
    more calibration than a v1 model has earned."""
    name = html.escape(str(row.get("player_name") or ""))
    position = html.escape(str(row.get("position") or ""))
    team = html.escape(str(row.get("team") or ""))
    adp = _fmt(row.get("adp"), 1) or "--"
    tier = row.get("rookie_display_tag")
    label = ARCHETYPE_LABELS.get(tier, tier or "Unconfirmed")

    # NaN (missing, the common case for whichever of the two columns
    # doesn't apply to this player's position) is truthy in Python, so
    # `row.get(...) or row.get(...)` would silently pick the NaN over a
    # real string -- explicit isinstance/str checks instead, same pattern
    # _render_cell's archetype branch already uses.
    rb_primary_val = row.get("rb_archetype_primary")
    wr_primary_val = row.get("wr_archetype_primary")
    qb_primary_val = row.get("qb_archetype_primary")
    real_primary, metrics_dict = (
        (rb_primary_val, ARCHETYPE_METRICS) if isinstance(rb_primary_val, str) and rb_primary_val
        else (wr_primary_val, WR_ARCHETYPE_METRICS) if isinstance(wr_primary_val, str) and wr_primary_val
        else (qb_primary_val, QB_ARCHETYPE_METRICS) if isinstance(qb_primary_val, str) and qb_primary_val
        else (None, None)
    )
    real_metric_rows = []
    if real_primary:
        for col, metric_label, kind in metrics_dict.get(real_primary, []):
            formatted = _archetype_metric_value(row.get(col), kind)
            if formatted is None:
                continue
            real_metric_rows.append(
                f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
                f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
            )

    component_rows = []
    for col, metric_label, kind in ROOKIE_COMPONENT_METRICS:
        formatted = _archetype_metric_value(row.get(col), kind)
        if formatted is None:
            continue
        component_rows.append(
            f'<div class="sc-row"><div class="sc-cat-label">{html.escape(metric_label)}</div>'
            f'<div class="sc-cat-score" style="width:auto;margin-left:auto;">{formatted}</div></div>'
        )

    composite = _fmt(row.get("rookie_composite"), 1)
    draft_season = row.get("rookie_draft_season")
    season_label = f"real {int(draft_season)}" if pd.notna(draft_season) else "real"
    if status == "blended":
        weight = row.get("rookie_display_weight")
        pct = round(float(weight) * 100) if pd.notna(weight) else None
        status_note = (
            f"Blending the pre-draft projection with {season_label} usage ({pct}% confirmed "
            f"weight, based on real sample size). Will shift fully to the confirmed read once "
            f"real usage clears the sample floor."
            if pct is not None else ""
        )
    elif real_primary:
        # Real snaps exist (real_primary is on file), but the real read
        # resolved to "unconfirmed" -- not "no data yet". Different from
        # a true zero-snap rookie; say so honestly.
        status_note = (
            f"Real {season_label} usage exists but hasn't resolved into a clear archetype yet "
            f"-- leaning on the college-based projection in the meantime. Composite: {composite}."
            if composite else
            f"Real {season_label} usage exists but hasn't resolved into a clear archetype yet "
            f"-- leaning on the college-based projection in the meantime."
        )
    else:
        status_note = (
            f"Pre-season projection -- no real NFL snaps yet. Composite: {composite}."
            if composite else "Pre-season projection -- no real NFL snaps yet."
        )

    disclaimer = (
        "This is a v1, unvalidated model -- the composite score and tier thresholds "
        "haven't been retuned against the real backtest yet. Treat as directional, not precise."
    )

    # QB-only disclosure (claude_code_plan_qb_rookie_projection.pdf) -- the
    # rushing-lean tier for 13 of 32 real QBs in the consolidated dataset is
    # an explicit assumption (no real college rushing data captured), never
    # presented at the same confidence as a real measured/verified number.
    rushing_status = row.get("rookie_rushing_data_status")
    rushing_note = (
        '<div class="sc-note" style="margin-top:6px;">Assumed non-rushing QB -- no real college '
        "rushing data captured for this player, not independently verified.</div>"
        if rushing_status == "assumed_non_rusher" else ""
    )

    real_section = ""
    if real_metric_rows:
        real_section = (
            '<div class="sc-note" style="font-weight:700;color:#e6edf3;margin-bottom:4px;">REAL USAGE</div>'
            f'{"".join(real_metric_rows)}'
            '<div class="sc-divider"></div>'
            '<div class="sc-note" style="font-weight:700;color:#e6edf3;margin-bottom:4px;">PROSPECT COMPOSITE</div>'
        )

    return (
        '<div class="scorecard">'
        '<div class="sc-header">'
        f'<div><div class="sc-name">{name}</div>'
        f'<div class="sc-meta">{position} · {team} · ADP {adp}</div></div>'
        '<div class="sc-overall">'
        f'<div class="lbl" style="font-size:12px;color:#e6edf3;font-weight:700;">{html.escape(label)}</div>'
        '</div>'
        "</div>"
        f'{real_section}'
        f'{"".join(component_rows)}'
        '<div class="sc-divider"></div>'
        f'<div class="sc-note">{status_note}</div>'
        f'<div class="sc-note" style="margin-top:6px;">{disclaimer}</div>'
        f'{rushing_note}'
        f'{_prospect_profile_html(row)}'
        "</div>"
    )


def _render_rookie_archetype_badge(row, status: str) -> str:
    """Pre-season/still-inconclusive rookie badge
    (draftkit/rookie_projection.py). Shown while rookie_display_status is
    'projected' or 'blended' -- the "unconfirmed"-aware display status,
    which matches real usage-sample confidence for any player with a
    real, DECISIVE tag and only falls back to the college projection when
    the real tag is literally "unconfirmed" (real snaps exist, but no
    real usage tier was ever crossed). A years-since-draft display cap was
    tried and reverted for a different reason -- see blend_rookie_tag()'s
    docstring: it kept a stale pre-draft composite partially overriding an
    ALREADY-CONFIRMED real archetype, which this does not do. Once a
    player's real read is either decisive OR the sample clears the floor,
    rookie_display_status becomes 'confirmed' and _render_cell falls
    through to the ordinary RB/WR confirmed-archetype rendering
    automatically -- no hedge, no percentage, and never the word
    "Unconfirmed"."""
    tier = row.get("rookie_display_tag")
    label = ARCHETYPE_LABELS.get(tier, tier or "Unconfirmed")
    tier_badge_class = f"arch-{ARCHETYPE_LABELS.get(tier, 'Unconfirmed')}"

    if status == "projected":
        badge_text = f"Projected: {label}"
        status_class = "arch-projected"
        beta_html = '<span class="beta-dot"></span>'
    else:  # blended
        weight = row.get("rookie_display_weight")
        pct = round(float(weight) * 100) if pd.notna(weight) else None
        badge_text = f"{label} — {pct}% confirmed" if pct is not None else label
        status_class = "arch-blended"
        beta_html = ""

    return (
        f'<span class="arch-wrap"><span class="arch-badge {tier_badge_class} {status_class}">'
        f"{html.escape(badge_text)}{beta_html}</span>"
        f"{_rookie_archetype_scorecard_html(row, status)}</span>"
    )


def _render_cell(row, column, style):
    value = row.get(column)

    if style == "rank":
        return _fmt(value, 0) or ""
    if style == "player":
        return html.escape(str(value or ""))

    if style == "pos":
        # Position rank shown as the reference does it -- RB1/WR3 -- colored by
        # position, which is why it isn't a plain number pill.
        position = str(row.get("position") or "").upper()
        number = _fmt(value, 0)
        if number is None:
            return '<span class="pill pill-empty">--</span>'
        color = POSITION_COLORS.get(position, "#64748b")
        return (
            f'<span class="pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55">{html.escape(position)}{number}</span>'
        )

    if style == "archetype":
        # Rookie projections gate FIRST, on rookie_status -- not on whether
        # rb_archetype_primary/wr_archetype_primary is present. Verified
        # this matters: a rookie can already have a real (if "unconfirmed")
        # row in rb_archetypes.csv/wr_archetypes.csv while
        # blend_rookie_tag() still correctly reports "blended" (once
        # sample_weight clears 0) -- checking rb_archetype_primary first
        # would silently swallow real "blended" cases into the plain
        # veteran "Unconfirmed" badge instead. Once a rookie's real sample
        # is large enough for blend_rookie_tag() to report "confirmed",
        # rookie_status no longer matches either branch below and this
        # falls through to the ordinary RB/WR rendering automatically --
        # no extra code needed for that transition. Gates on
        # rookie_display_status (the "unconfirmed"-aware fallback), NOT
        # rookie_status (real usage-sample confidence, reserved for
        # backtest/validation ground truth). The two agree for any player
        # with a real, DECISIVE tag (bellcow, alpha, committee_back...) --
        # a years-since-draft display cap was tried and reverted for
        # exactly this reason (see blend_rookie_tag()'s docstring): it kept
        # a real, already-confirmed archetype (e.g. Ashton Jeanty's real
        # 2025 bellcow season) partially hedged just because too few
        # calendar years had passed. They diverge only when the real tag is
        # literally "unconfirmed" -- real snaps exist, but no real usage
        # tier was ever crossed -- in which case rookie_display_status
        # falls back to the college-based projection instead of showing
        # the bare word "Unconfirmed".
        rookie_display_status = row.get("rookie_display_status")
        if isinstance(rookie_display_status, str) and rookie_display_status in ("projected", "blended"):
            return _render_rookie_archetype_badge(row, rookie_display_status)

        # RB and WR each have their own real taxonomy (rb_archetype_primary
        # / wr_archetype_primary), never both populated for the same row --
        # position-exclusive by construction (build_rb_archetypes.py only
        # emits RB rows, build_wr_archetypes.py only WR rows). Every other
        # position has neither merged in, so the cell renders blank rather
        # than a misleading "--" pill.
        rb_primary = row.get("rb_archetype_primary")
        if isinstance(rb_primary, str) and rb_primary:
            # RB-specific override: "No Role" reads better than the shared
            # "Unconfirmed" label for a real usage-tier taxonomy (explicit
            # user correction 2026-08-15) -- WR's own "unconfirmed" (a
            # different, insufficient-sample meaning) keeps the shared
            # label via ARCHETYPE_LABELS below, unaffected.
            label = "No Role" if rb_primary == "unconfirmed" else ARCHETYPE_LABELS.get(rb_primary, rb_primary)
            down_split = row.get("rb_down_split")
            if rb_primary in ("bellcow", "committee_back") and isinstance(down_split, str) and down_split in DOWN_SPLIT_LABELS:
                label = f"{label} ({DOWN_SPLIT_LABELS[down_split]})"
            lean = row.get("rb_lean")
            lean_html = ""
            if isinstance(lean, str) and lean in LEAN_LABELS:
                lean_html = f'<span class="arch-lean-tag"> · {html.escape(LEAN_LABELS[lean])}</span>'
            badge_class = f"arch-{ARCHETYPE_LABELS.get(rb_primary, 'Unconfirmed')}"
            return (
                f'<span class="arch-wrap"><span class="arch-badge {badge_class}">'
                f"{html.escape(label)}{lean_html}</span>"
                f"{_archetype_scorecard_html(row, rb_primary)}</span>"
            )

        wr_primary = row.get("wr_archetype_primary")
        if isinstance(wr_primary, str) and wr_primary:
            label = ARCHETYPE_LABELS.get(wr_primary, wr_primary)
            leans = [l for l in str(row.get("wr_leans") or "none").split(",") if l in LEAN_LABELS]
            qb_label = QB_CONTEXT_LABELS.get(row.get("wr_qb_context"))
            tags = [LEAN_LABELS[l] for l in leans] + ([qb_label] if qb_label else [])
            tag_html = "".join(f'<span class="arch-lean-tag"> · {html.escape(t)}</span>' for t in tags)
            badge_class = f"arch-{ARCHETYPE_LABELS.get(wr_primary, 'Unconfirmed')}"
            return (
                f'<span class="arch-wrap"><span class="arch-badge {badge_class}">'
                f"{html.escape(label)}{tag_html}</span>"
                f"{_wr_archetype_scorecard_html(row, wr_primary)}</span>"
            )

        qb_primary = row.get("qb_archetype_primary")
        if isinstance(qb_primary, str) and qb_primary:
            label = ARCHETYPE_LABELS.get(qb_primary, qb_primary)
            badge_class = f"arch-{ARCHETYPE_LABELS.get(qb_primary, 'Unconfirmed')}"
            return (
                f'<span class="arch-wrap"><span class="arch-badge {badge_class}">'
                f"{html.escape(label)}</span>"
                f"{_qb_archetype_scorecard_html(row, qb_primary)}</span>"
            )

        te_primary = row.get("te_archetype_primary")
        if isinstance(te_primary, str) and te_primary:
            role_profile = row.get("te_role_profile")
            if te_primary == "receiving_te" and isinstance(role_profile, str) and role_profile in TE_ROLE_PROFILE_LABELS:
                label = TE_ROLE_PROFILE_LABELS[role_profile]
                badge_class = f"arch-{label.replace(' ', '-')}"
            else:
                label = ARCHETYPE_LABELS.get(te_primary, te_primary)
                badge_class = f"arch-{ARCHETYPE_LABELS.get(te_primary, 'Unconfirmed')}"
            leans = [l for l in str(row.get("te_leans") or "none").split(",") if l in LEAN_LABELS]
            tag_html = "".join(f'<span class="arch-lean-tag"> · {html.escape(LEAN_LABELS[l])}</span>' for l in leans)
            return (
                f'<span class="arch-wrap"><span class="arch-badge {badge_class}">'
                f"{html.escape(label)}{tag_html}</span>"
                f"{_te_archetype_scorecard_html(row, te_primary)}</span>"
            )

        return ""

    if style == "risk":
        # Composite 0-100 risk_index (v2: four weighted categories), red/
        # amber/green banded, with a hover POPOVER (not a plain title=
        # tooltip -- see _risk_scorecard_html) showing the four category
        # bars, a market badge, an optional chronic-injury badge, and a
        # templated note naming the primary driver.
        if value is None or pd.isna(value):
            return '<span class="pill pill-empty">--</span>'
        number = float(value)
        band = "red" if number > 66 else "amber" if number > 34 else "green"
        return (
            f'<span class="pill pill-{band} pill-risk">{html.escape(_fmt(number, 0) or "")}'
            f"{_risk_scorecard_html(row, number, band)}</span>"
        )

    if style == "signed":
        # vs MKT: positive means the sportsbook ranks him better within his
        # position than ADP does. Same sign convention as the old tooltip.
        if value is None or pd.isna(value):
            return '<span class="pill pill-empty">--</span>'
        number = float(value)
        if number == 0:
            return _pill("0", "muted")
        return _pill(f"{number:+.0f}", "green" if number > 0 else "red")

    if column == "projection_points":
        # PROJECTED PTS gets a visible marker when a manual
        # PROJECTION_MANUAL_ADJUSTMENTS override (draft_analysis.py) has
        # been applied -- the displayed number already includes the
        # adjustment (same effective-value pattern as the injury-override
        # badge: the number IS the real, effective one used for scoring,
        # never silently hidden behind an unadjusted display).
        text = _fmt(value, 1)
        # This project's own researched value (draft_analysis.py's
        # apply_model_projection_override, 2026-08-21) takes priority --
        # when applied, it already REPLACED this cell's number (and
        # cleared projection_adjustment_pct below, so the two markers
        # never both fire for the same row). Distinct marker/color from
        # the legacy PROJECTION_MANUAL_ADJUSTMENTS one on purpose: this
        # means "this team's own dated research," not an older editorial
        # guess.
        if bool(row.get("model_override_applied")):
            note = row.get("model_override_note") or ""
            marker = (
                f' <span class="proj-adj model-corrected" title="{html.escape(str(note), quote=True)}">'
                f'✓</span>'
            )
            return _pill(text, "green") + marker
        pct = row.get("projection_adjustment_pct")
        if pd.notna(pct):
            note = row.get("projection_adjustment_note") or ""
            direction = "proj-adj-pos" if float(pct) > 0 else "proj-adj-neg"
            marker = (
                f' <span class="proj-adj {direction}" title="{html.escape(str(note), quote=True)}">'
                f'{"+" if float(pct) > 0 else ""}{float(pct):.0f}%</span>'
            )
            return _pill(text, "green") + marker
        return _pill(text, "green")

    if column == "model_projection_points":
        # A blank MODEL PROJ cell can mean three different real things
        # (position excluded, player not identifiable in the model's
        # historical data, or identified but with too little NFL history to
        # project from) -- per review, none of those should render as an
        # identical unexplained "--". Every blank gets a tooltip naming the
        # real reason, matching the QB column-header disclosure but at the
        # per-cell level so it doesn't require a user to already know to
        # hover the header.
        text = _fmt(value, 1)
        if text is not None:
            # Diagnosed-defect correction (model_proj_staleness_fix_plan.pdf,
            # Step 4) takes priority over the bare continuity flag below: a
            # player in MODEL_PROJECTION_CORRECTIONS (build_live_projections_v1.py)
            # has already had this exact staleness bug individually verified
            # and fixed, via a SEPARATE column (model_projection_points_adjusted)
            # -- model_projection_points itself (`value` here) is never
            # touched. Rendered with a visibly different marker/color from
            # continuity-flag's "!" on purpose: that one means "verify this,
            # unresolved," this one means "known bug, already corrected."
            adj_pct = row.get("model_adjustment_pct")
            if pd.notna(adj_pct):
                adjusted_value = row.get("model_projection_points_adjusted")
                adj_text = _fmt(adjusted_value, 1) if pd.notna(adjusted_value) else text
                note = row.get("model_adjustment_note")
                title = "Diagnosed model staleness bug, individually verified and corrected -- not an unresolved flag."
                if pd.notna(note) and str(note).strip():
                    title += f" ({note})"
                marker = (
                    f' <span class="proj-adj model-corrected" title="{html.escape(title, quote=True)}">'
                    f'{"+" if float(adj_pct) > 0 else ""}{float(adj_pct):.0f}%</span>'
                )
                return _pill(adj_text, "green") + marker
            # Roster continuity guardrail (roster_continuity_fix_plan.pdf,
            # Step 5): this model was fit on historical features that
            # assume roster continuity (same team, same teammates). A
            # player who changed teams or whose direct same-position
            # competitor departed/arrived this offseason breaks that
            # assumption -- the number above is real and correctly
            # computed, but rests on a premise (last season's team
            # context, or a committee split) that no longer holds. Flagged
            # rather than silently trusted, same principle as the
            # manual-adjustment marker just above.
            flags = []
            if bool(row.get("team_changed")):
                flags.append("changed teams this offseason")
            if bool(row.get("competitor_departed")):
                flags.append("a real same-position competitor departed")
            if bool(row.get("competitor_arrived")):
                flags.append("a real same-position competitor arrived")
            if flags:
                # pd.notna(), not `or ""` -- a missing continuity_note comes
                # back from the CSV as float NaN, which is truthy in Python
                # (only 0/None/""/etc are falsy), so `note or ""` and
                # `if note:` both silently let a real NaN through and
                # rendered the literal text "(nan)" in the tooltip.
                note = row.get("continuity_note")
                title = "Situational change this offseason -- verify before trusting this number. " + "; ".join(flags).capitalize() + "."
                if pd.notna(note) and str(note).strip():
                    title += f" ({note})"
                marker = f' <span class="proj-adj continuity-flag" title="{html.escape(title, quote=True)}">⚠</span>'
                return _pill(text, "green") + marker
            return _pill(text, "green")
        position = str(row.get("position") or "").upper()
        status = row.get("model_projection_status")
        # Rookie-score fallback (rb_model_fix_plan.pdf, Phase 1): a real
        # rookie with model_projection_status=="insufficient_history" (no
        # NFL outcome history to predict from) but a real, already-computed
        # pre-draft composite (rookie_projection.py) gets THIS substitute
        # value instead of a blank cell -- visibly marked as a substitute,
        # never conflated with the real outcome-trained model's own output.
        fallback_value = row.get("model_projection_points_fallback")
        if pd.notna(fallback_value):
            fallback_note = row.get("model_projection_fallback_note")
            title = "Rookie-score fallback -- NOT a real outcome-trained model prediction."
            if pd.notna(fallback_note) and str(fallback_note).strip():
                title += f" ({fallback_note})"
            marker = f' <span class="proj-adj rookie-fallback" title="{html.escape(title, quote=True)}">≈</span>'
            return _pill(_fmt(fallback_value, 1), "amber") + marker
        if position not in ("RB", "WR", "TE"):
            reason = "Model projection not built for this position -- see MODEL PROJ column header for why."
        elif status == "no_crosswalk_match":
            reason = "No model projection -- couldn't confidently match this player to historical performance data."
        elif status == "insufficient_history":
            reason = "No model projection -- identified, but not enough real NFL game history yet to build one (rookie or very limited prior usage)."
        else:
            reason = "No model projection available for this player."
        return _pill(None, style, title=reason)

    decimals = 1 if style in {"green", "amber"} else 0
    return _pill(_fmt(value, decimals), style)


def _sort_link(key, label, tooltip, current_key, current_dir):
    """Header link toggling ?sort=&dir=. target=_self keeps it in-tab."""
    is_sorted = key == current_key
    next_dir = "desc" if (is_sorted and current_dir == "asc") else "asc"
    arrow = ""
    if is_sorted:
        arrow = " ▲" if current_dir == "asc" else " ▼"
    classes = ' class="sorted"' if is_sorted else ""
    # title= gives beginners a plain-English explanation on hover; there is no
    # column_config help bubble once we render the table ourselves.
    title = f' title="{html.escape(tooltip, quote=True)}"' if tooltip else ""
    return (
        f'<th{classes}{title}><a target="_self" href="?sort={key}&dir={next_dir}">'
        f"{html.escape(label)}{arrow}</a></th>"
    )


def render_rankings_table(board, sort_key="", sort_dir="asc", show_tier_bands=True):
    header = "".join(
        _sort_link(key, label, tooltip, sort_key, sort_dir)
        for key, label, _, _, tooltip in TABLE_COLUMNS
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
                if tier_id is None:
                    color = "#64748b"
                else:
                    # -1 so Tier 1 gets the first palette color, not the second.
                    color = TIER_BAND_COLORS[
                        (max(tier_id, 1) - 1) % len(TIER_BAND_COLORS)
                    ]
                body.append(
                    f'<tr class="tier-band"><td colspan="{len(TABLE_COLUMNS)}">'
                    f'<div class="tier-band-inner" style="background:{color}">'
                    f"{label}</div></td></tr>"
                )

        cells = "".join(
            f'<td class="col-{style if style in {"rank", "player"} else "val"}">'
            f"{_render_cell(row, column, style)}</td>"
            for _, _, column, style, _ in TABLE_COLUMNS
        )
        body.append(f"<tr>{cells}</tr>")

    return (
        '<div class="gp-table-wrap"><table class="gp-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


st.set_page_config(page_title="Top 250 Rankings", layout="wide")

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
    for column in ("expert_rank",):
        if column in source.columns and "player_name" in source.columns:
            extra = source[["player_name", column]].drop_duplicates("player_name")
            df = df.merge(extra, on="player_name", how="left", suffixes=("", "_src"))

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

st.markdown(RANKINGS_CSS, unsafe_allow_html=True)
st.markdown('<h1 class="gp-title">Rankings</h1>', unsafe_allow_html=True)

# Position chips. st.button rather than HTML links so this stays independent
# of the sort query-param state below -- mixing the two makes widget state and
# URL state fight each other.
if "pos_filter" not in st.session_state:
    st.session_state["pos_filter"] = "Overall"

_chip_cols = st.columns([1, 1, 1, 1, 1, 6])
for _col, _label in zip(_chip_cols, POSITION_CHIPS):
    with _col:
        if st.button(
            _label,
            key=f"chip_{_label}",
            width="stretch",
            type="primary" if st.session_state["pos_filter"] == _label else "secondary",
        ):
            st.session_state["pos_filter"] = _label
            st.rerun()

position_filter = st.session_state["pos_filter"]

_search_col, _export_col, _count_col = st.columns([3, 1, 1])
with _search_col:
    search_query = st.text_input(
        "Search", placeholder="Search players...", label_visibility="collapsed"
    )

board = rankings_df.copy()
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

if position_filter != "Overall":
    board = board[board["position"].astype(str).str.upper() == position_filter]

if search_query:
    board = board[
        board["player_name"].astype(str).str.contains(search_query, case=False, na=False)
    ]

# Sort state lives in the URL so the table's <th> links can drive it -- there
# is no Streamlit widget behind a custom HTML table. Unknown column names are
# ignored rather than raising, so a hand-edited URL can't break the page.
_params = st.query_params
_sort_key = _params.get("sort", "")
_sort_dir = _params.get("dir", "asc")
_sort_column = SORTABLE_COLUMNS.get(_sort_key)

if _sort_column and _sort_column in board.columns:
    board = board.sort_values(
        _sort_column, ascending=(_sort_dir != "desc"), na_position="last", kind="stable"
    )
    show_tier_bands = False
else:
    # Default view groups into contiguous tier blocks. Automatic tiers
    # (2026-08-17) are assigned sequentially over the already Rank-sorted
    # board (see _assign_automatic_tiers()), so they are contiguous by
    # construction -- this sort is a defensive no-op against Rank order,
    # not a real reordering, unlike the old FantasyPros-sourced tier this
    # replaced (which genuinely could interleave with engine Rank). The
    # "#" column still carries each player's true engine rank.
    if "tier" in board.columns:
        board = board.sort_values(
            ["tier", "Rank"], na_position="last", kind="stable"
        )
    show_tier_bands = True

with _export_col:
    st.download_button(
        "Export",
        data=board.to_csv(index=False).encode("utf-8"),
        file_name="guaranteed_play_rankings.csv",
        mime="text/csv",
        width="stretch",
    )
with _count_col:
    st.markdown(
        f'<div class="gp-count">{len(board)} players</div>', unsafe_allow_html=True
    )

st.markdown(
    render_rankings_table(board, sort_key=_sort_key, sort_dir=_sort_dir,
                          show_tier_bands=show_tier_bands),
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
