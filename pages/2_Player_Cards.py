import pandas as pd
import streamlit as st
from html import escape

from draftkit.data_access import get_available_players_df, load_players_df, safe_col
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import (
    handle_draft_action,
    handle_my_pick,
    init_session_state,
    keep_session_state_alive,
)
from draftkit.recommendation_explainer import build_player_decision_card
from draftkit.ui_helpers import render_league_settings_sidebar


st.set_page_config(page_title="Player Cards", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Player Cards")
st.caption("Scout a player quickly, understand the draft case, and act with context.")

st.markdown(
    """
    <style>
    .scout-hero {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        padding: 22px 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(31, 41, 55, 0.92));
        color: #f8fafc;
        margin: 14px 0 18px;
    }
    .scout-eyebrow {
        color: #cbd5e1;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 4px;
    }
    .scout-player {
        font-size: 2.25rem;
        font-weight: 750;
        line-height: 1.05;
        margin: 0;
    }
    .scout-meta {
        color: #cbd5e1;
        font-size: 1rem;
        margin-top: 6px;
    }
    .scout-grade {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1;
        margin-top: 18px;
    }
    .scout-card {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        padding: 16px 18px;
        background: rgba(248, 250, 252, 0.98);
        min-height: 128px;
    }
    .scout-card-dark {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        padding: 16px 18px;
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
        min-height: 128px;
    }
    .scout-label {
        color: #64748b;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 8px;
    }
    .scout-value {
        color: #0f172a;
        font-size: 1.32rem;
        font-weight: 750;
        line-height: 1.15;
    }
    .scout-value-light {
        color: #f8fafc;
        font-size: 1.32rem;
        font-weight: 750;
        line-height: 1.15;
    }
    .scout-detail {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .scout-detail-light {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .scout-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        background: #e2e8f0;
        color: #0f172a;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .scout-badge-accent {
        background: #dcfce7;
        color: #14532d;
    }
    .scout-badge-warning {
        background: #fef3c7;
        color: #78350f;
    }
    .scout-badge-risk {
        background: #fee2e2;
        color: #7f1d1d;
    }
    .bar-label {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: #334155;
        font-size: 0.9rem;
        font-weight: 650;
        margin: 10px 0 4px;
    }
    .scouting-panel {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 18px;
        background: #ffffff;
        min-height: 230px;
    }
    .insight-row {
        display: grid;
        grid-template-columns: 132px 1fr;
        gap: 14px;
        align-items: center;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        padding: 10px 0;
    }
    .insight-row:last-child {
        border-bottom: 0;
    }
    .insight-label {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .insight-value {
        color: #f8fafc;
        font-size: 0.96rem;
        font-weight: 650;
    }
    .mini-stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .mini-stat {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 12px;
        background: rgba(15, 23, 42, 0.04);
    }
    .mini-stat-label {
        color: #94a3b8;
        font-size: 0.74rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .mini-stat-value {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 750;
        margin-top: 4px;
    }
    .report-list {
        margin: 0;
        padding-left: 18px;
    }
    .report-list li {
        margin: 0 0 12px;
        padding-left: 4px;
    }
    .simple-hero {
        border: 1px solid rgba(148, 163, 184, 0.34);
        border-radius: 8px;
        padding: 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(31, 41, 55, 0.94));
        color: #f8fafc;
        margin: 14px 0 18px;
        display: grid;
        grid-template-columns: 116px 1fr 260px;
        gap: 22px;
        align-items: center;
    }
    .headshot-placeholder {
        width: 104px;
        height: 104px;
        border-radius: 8px;
        background: #e2e8f0;
        color: #0f172a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 850;
    }
    .headshot-image {
        width: 104px;
        height: 104px;
        border-radius: 8px;
        object-fit: cover;
        background: #e2e8f0;
    }
    .hero-name {
        font-size: 2.35rem;
        font-weight: 850;
        line-height: 1;
    }
    .hero-grade {
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 10px;
    }
    .hero-impact {
        border-left: 1px solid rgba(226, 232, 240, 0.25);
        padding-left: 22px;
        font-size: 1.55rem;
        font-weight: 800;
        color: #bbf7d0;
    }
    .section-card {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 8px;
        padding: 18px 20px;
        background: rgba(15, 23, 42, 0.18);
        min-height: 176px;
    }
    .profile-row {
        display: grid;
        grid-template-columns: 1fr 52px;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        align-items: center;
    }
    .profile-row:last-child {
        border-bottom: 0;
    }
    .profile-label {
        font-weight: 750;
    }
    .profile-grade {
        text-align: right;
        font-size: 1.05rem;
        font-weight: 850;
        color: #bbf7d0;
    }
    @media (max-width: 900px) {
        .simple-hero {
            grid-template-columns: 1fr;
        }
        .hero-impact {
            border-left: 0;
            padding-left: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_score(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_model_score(value):
    formatted = _format_score(value)
    return "N/A" if formatted == "N/A" else f"{formatted} / 100"


def _recommendation_grade(score):
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return "D"

    if score_value >= 97:
        return "A+"
    if score_value >= 93:
        return "A"
    if score_value >= 90:
        return "A-"
    if score_value >= 87:
        return "B+"
    if score_value >= 83:
        return "B"
    if score_value >= 80:
        return "B-"
    if score_value >= 75:
        return "C+"
    if score_value >= 70:
        return "C"
    return "D"


def _format_grade(score):
    return f"{_recommendation_grade(score)} Recommendation"


def _grade_from_score(score):
    return _recommendation_grade(score)


def _grade_from_bonus(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "C"
    if numeric >= 20:
        return "A+"
    if numeric >= 15:
        return "A"
    if numeric >= 10:
        return "A-"
    if numeric >= 5:
        return "B+"
    if numeric >= 0:
        return "B"
    if numeric >= -5:
        return "C"
    return "D"


def _grade_from_multiplier(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "C"
    if numeric >= 1.12:
        return "A+"
    if numeric >= 1.08:
        return "A"
    if numeric >= 1.04:
        return "A-"
    if numeric >= 1.0:
        return "B+"
    if numeric >= 0.96:
        return "B"
    if numeric >= 0.92:
        return "C"
    return "D"


def _grade_from_inverse_risk(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "C"
    return _recommendation_grade(100.0 - numeric)


def _grade_from_risk_label(label):
    label = str(label).upper()
    if label == "LOW":
        return "A"
    if label == "MODERATE":
        return "B"
    if label == "ELEVATED":
        return "C"
    return "C"


def _player_initials(name):
    parts = [part for part in str(name).split() if part]
    if not parts:
        return "GP"
    return "".join(part[0].upper() for part in parts[:2])


def _first_available(mapping, keys, default="N/A"):
    for key in keys:
        value = _safe_get(mapping, key, None)
        if value is not None:
            return value
    return default


def _lookup_latest_championship_impact(player_name):
    equity_df = st.session_state.get("latest_championship_equity_df")
    if equity_df is None or getattr(equity_df, "empty", True):
        return None
    if "player_name" not in equity_df.columns:
        return None

    match = equity_df[
        equity_df["player_name"].astype(str).str.lower() == str(player_name).lower()
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    for column in ["equity_delta", "championship_impact", "championship_equity_delta"]:
        if column in row and pd.notna(row[column]):
            return row[column]
    return None


def _safe_get(row, key, default="N/A"):
    if row is None or key not in row:
        return default
    value = row.get(key)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _risk_label(value):
    try:
        risk = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if risk >= 70:
        return "ELEVATED"
    if risk >= 45:
        return "MODERATE"
    return "LOW"


def _fit_label(value):
    try:
        fit = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if fit >= 15:
        return "STRONG FIT"
    if fit >= 5:
        return "USEFUL FIT"
    return "NEUTRAL"


def _value_label(value):
    try:
        value_score = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN VALUE"
    if value_score >= 15:
        return "ELITE VALUE"
    if value_score >= 5:
        return "POSITIVE VALUE"
    if value_score <= -5:
        return "PRICEY"
    return "FAIR PRICE"


def _score_to_progress(value, default=50.0):
    try:
        if pd.isna(value):
            return default / 100.0
        numeric = float(value)
    except (TypeError, ValueError):
        return default / 100.0
    return max(0.0, min(numeric / 100.0, 1.0))


def _bonus_to_progress(value):
    try:
        if pd.isna(value):
            return 0.5
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min((numeric + 20.0) / 40.0, 1.0))


def _inverse_risk_progress(value):
    try:
        if pd.isna(value):
            return 0.5
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min((100.0 - numeric) / 100.0, 1.0))


def _card(label, value, detail="", dark=False):
    card_class = "scout-card-dark" if dark else "scout-card"
    value_class = "scout-value-light" if dark else "scout-value"
    detail_class = "scout-detail-light" if dark else "scout-detail"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="scout-label">{label}</div>
            <div class="{value_class}">{value}</div>
            <div class="{detail_class}">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _bar(label, value, progress_value):
    st.markdown(
        f"""
        <div class="bar-label">
            <span>{label}</span>
            <span>{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(progress_value)

raw_df = load_players_df()
available_df = get_available_players_df()
recommendation_df = build_recommendation_rankings_df()

if raw_df.empty:
    st.error("No player data loaded.")
    st.stop()

player_col = safe_col(raw_df, ["player_name", "Player", "player", "name", "full_name"])
pos_col = safe_col(raw_df, ["position", "pos", "Pos", "Position", "POS"])
team_col = safe_col(raw_df, ["team", "Team", "team_abbr", "TEAM"])
adp_col = safe_col(raw_df, ["adp", "ADP", "rank", "Rank"])
proj_col = safe_col(
    raw_df,
    ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"],
)
bye_col = safe_col(raw_df, ["bye_week", "Bye", "bye"])
headshot_col = safe_col(
    raw_df,
    ["headshot", "headshot_url", "image_url", "photo_url", "player_image", "player_photo"],
)

if player_col is None:
    st.error("No player name column found in the loaded data.")
    st.stop()

card_df = available_df.copy()
if card_df.empty:
    st.info("No available players found.")
    st.stop()

if proj_col is not None and proj_col in card_df.columns:
    card_df[proj_col] = pd.to_numeric(card_df[proj_col], errors="coerce")

if adp_col is not None and adp_col in card_df.columns:
    card_df[adp_col] = pd.to_numeric(card_df[adp_col], errors="coerce")

top_controls = st.columns([2, 1])

with top_controls[0]:
    search_text = st.text_input("Search player", placeholder="Type a player name")

with top_controls[1]:
    if pos_col is not None:
        position_options = ["All"] + sorted(
            card_df[pos_col].dropna().astype(str).unique().tolist()
        )
        selected_position = st.selectbox("Position", position_options)
    else:
        selected_position = "All"
        st.selectbox("Position", ["All"], disabled=True)

if search_text:
    card_df = card_df[
        card_df[player_col].astype(str).str.contains(search_text, case=False, na=False)
    ].copy()

if pos_col is not None and selected_position != "All":
    card_df = card_df[card_df[pos_col].astype(str) == selected_position].copy()

if card_df.empty:
    st.info("No players match the current filters.")
    st.stop()

if proj_col is not None:
    card_df = card_df.sort_values(proj_col, ascending=False, na_position="last")
elif adp_col is not None:
    card_df = card_df.sort_values(adp_col, ascending=True, na_position="last")
else:
    card_df = card_df.sort_values(player_col, ascending=True)

player_options = card_df[player_col].dropna().astype(str).tolist()
selected_player = st.selectbox("Choose a player", player_options)
player_row = card_df[card_df[player_col].astype(str) == selected_player].iloc[0]
recommendation_row = None
decision_card = {}

player_name = str(player_row[player_col])
if not recommendation_df.empty:
    recommendation_match = recommendation_df[
        recommendation_df["player_name"].astype(str).str.lower() == player_name.lower()
    ]
    if not recommendation_match.empty:
        recommendation_row = recommendation_match.iloc[0]
        try:
            decision_card = build_player_decision_card(
                player_name,
                recommendations_df=recommendation_df,
            )
        except Exception:
            decision_card = {}

position = str(player_row[pos_col]) if pos_col else "Unknown"
team = str(player_row[team_col]) if team_col else "Unknown"
projection = player_row[proj_col] if proj_col else "N/A"
adp = player_row[adp_col] if adp_col else "N/A"
bye = player_row[bye_col] if bye_col else "N/A"
headshot_url = str(player_row[headshot_col]).strip() if headshot_col else ""

st.divider()

model_score = _safe_get(recommendation_row, "final_score", None)
archetype = _safe_get(recommendation_row, "archetype")
injury_risk = _safe_get(recommendation_row, "injury_risk")
durability = _safe_get(recommendation_row, "durability_grade")
adp_bonus = _safe_get(recommendation_row, "adp_bonus", None)
need_bonus = _safe_get(recommendation_row, "need_bonus", None)
tier_bonus = _safe_get(recommendation_row, "tier_bonus", None)
team_fit_bonus = _safe_get(recommendation_row, "team_fit_bonus", None)
projection_component_score = _safe_get(recommendation_row, "projection_component_score", None)
adp_value_component_score = _safe_get(recommendation_row, "adp_value_component_score", None)
team_fit_component_score = _safe_get(recommendation_row, "team_fit_component_score", None)
tier_urgency_component_score = _safe_get(recommendation_row, "tier_urgency_component_score", None)
position_need_component_score = _safe_get(recommendation_row, "position_need_component_score", None)
value_tier = _safe_get(recommendation_row, "value_tier", None)
reasons = decision_card.get("reasons", [])
risks = decision_card.get("risks", [])
score_breakdown = decision_card.get("score_breakdown", {})
position_need = decision_card.get("position_need", {})
tier_pressure = decision_card.get("tier_pressure", {})
availability = decision_card.get("availability", {})
projection_advantage = decision_card.get("projection_advantage", {})
risk_impact = decision_card.get("risk_impact", {})

grade_text = _format_grade(model_score) if recommendation_row is not None else "Unrated"
archetype_text = str(archetype).replace("_", " ").title()
risk_text = _risk_label(injury_risk)
value_text = str(value_tier).replace("_", " ") if value_tier else _value_label(adp_bonus)
fit_text = _fit_label(team_fit_bonus)
tier_label = tier_pressure.get("tier_label") or _format_score(tier_bonus)
championship_impact = _lookup_latest_championship_impact(player_name)
if championship_impact is None:
    championship_impact = _first_available(
        recommendation_row,
        ["equity_delta", "championship_impact", "championship_equity_delta"],
        default=None,
    )
try:
    championship_impact_text = f"{float(championship_impact):+.1f}% Championship Odds"
except (TypeError, ValueError):
    championship_impact_text = f"Model Score {_format_model_score(model_score)}"

upside_score = _first_available(
    score_breakdown,
    ["ceiling_score", "upside_score", "projection_score"],
    default=None,
)
if upside_score is None:
    upside_score = _first_available(
        recommendation_row,
        ["projection_component_score", "position_value_score", "projection_points"],
        default=model_score,
    )
upside_grade = _grade_from_score(upside_score)
risk_grade = _grade_from_risk_label(risk_text)
value_grade = (
    _grade_from_score(adp_value_component_score)
    if adp_value_component_score is not None
    else _grade_from_score(_first_available(recommendation_row, ["value_score"], default=None))
    if _first_available(recommendation_row, ["value_score"], default=None) is not None
    else _grade_from_multiplier(adp_bonus)
)
durability_grade = _grade_from_score(_first_available(recommendation_row, ["durability_grade"], default=70.0))

fit_reasons = list(reasons[:4]) if reasons else []
if recommendation_row is not None and len(fit_reasons) < 4:
    fallback_fit_reasons = [
        f"{value_text.title()} relative to current draft cost",
        f"{fit_text.title()} for roster construction",
        f"{archetype_text} player profile",
        f"Tier signal: {_format_score(tier_urgency_component_score)}",
    ]
    for fallback_reason in fallback_fit_reasons:
        if len(fit_reasons) >= 4:
            break
        if fallback_reason not in fit_reasons:
            fit_reasons.append(fallback_reason)

best_fits = [
    f"Rosters needing {position} production",
    "Builds prioritizing weekly ceiling" if upside_grade in {"A+", "A", "A-"} else "Balanced roster builds",
    "Draft rooms where ADP value holds" if value_grade in {"A+", "A", "A-", "B+"} else "Managers comfortable paying market price",
]
avoid_ifs = [
    "You need maximum injury insulation" if risk_text in {"ELEVATED", "MODERATE"} else "You are avoiding positional duplication",
    "The next tier at this position is still deep" if str(tier_label) in {"N/A", "0.0"} else "A bigger roster need is available at similar grade",
]
headshot_html = (
    f'<img class="headshot-image" src="{escape(headshot_url)}" alt="{escape(player_name)} headshot">'
    if headshot_url and headshot_url.lower() not in {"nan", "none", "n/a"}
    else f'<div class="headshot-placeholder">{escape(_player_initials(player_name))}</div>'
)

st.markdown(
    f"""
    <div class="simple-hero">
        {headshot_html}
        <div>
            <div class="scout-eyebrow">Player Hero</div>
            <div class="hero-name">{escape(player_name)}</div>
            <div class="scout-meta">{escape(position)} - {escape(team)}</div>
            <div class="hero-grade">{escape(grade_text)}</div>
        </div>
        <div class="hero-impact">
            {escape(championship_impact_text)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Why He Fits")
with st.container(border=True):
    if fit_reasons:
        reason_items = "".join(f"<li>{escape(str(reason))}</li>" for reason in fit_reasons[:4])
        st.markdown(f'<ul class="report-list">{reason_items}</ul>', unsafe_allow_html=True)
    else:
        st.caption("No recommendation reasons are available.")

st.markdown("#### Player Profile")
with st.container(border=True):
    st.markdown(
        f"""
        <div class="profile-row">
            <div class="profile-label">Upside</div>
            <div class="profile-grade">{escape(upside_grade)}</div>
        </div>
        <div class="profile-row">
            <div class="profile-label">Risk</div>
            <div class="profile-grade">{escape(risk_grade)}</div>
        </div>
        <div class="profile-row">
            <div class="profile-label">Value</div>
            <div class="profile-grade">{escape(value_grade)}</div>
        </div>
        <div class="profile-row">
            <div class="profile-label">Durability</div>
            <div class="profile-grade">{escape(durability_grade)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("#### Draft Fit")
fit_col, avoid_col, action_col = st.columns([1, 1, 0.8])
with fit_col:
    with st.container(border=True):
        st.markdown("##### Best Fits")
        for item in best_fits:
            st.markdown(f"- {item}")

with avoid_col:
    with st.container(border=True):
        st.markdown("##### Avoid If")
        for item in avoid_ifs:
            st.markdown(f"- {item}")

with action_col:
    with st.container(border=True):
        st.markdown("##### Actions")
        st.button(
            "Mark Drafted",
            use_container_width=True,
            on_click=handle_draft_action,
            args=(player_row,),
        )
        st.button(
            "Draft To My Team",
            use_container_width=True,
            on_click=handle_my_pick,
            args=(player_row,),
        )

with st.expander("Advanced Analytics", expanded=False):
    st.markdown("#### Model Score Breakdown")
    if score_breakdown:
        breakdown_df = pd.DataFrame([score_breakdown])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    elif recommendation_row is not None:
        st.dataframe(
            recommendation_row.to_frame().T,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No model score breakdown is available.")

    st.markdown("#### Recommendation Components")
    if recommendation_row is not None:
        component_fields = [
            "final_score",
            "archetype",
            "need_bonus",
            "adp_bonus",
            "tier_bonus",
            "team_fit_bonus",
            "injury_risk",
            "durability_grade",
            "position_value_score",
            "construction_pressure_score",
        ]
        component_data = {
            field: _safe_get(recommendation_row, field)
            for field in component_fields
            if field in recommendation_row
        }
        st.dataframe(pd.DataFrame([component_data]), use_container_width=True, hide_index=True)
    else:
        st.caption("No recommendation component row is available.")

    st.markdown("#### Position And Tier Context")
    context_rows = []
    for label, payload in [
        ("Position Need", position_need),
        ("Tier Pressure", tier_pressure),
        ("Availability", availability),
        ("Projection Advantage", projection_advantage),
        ("Risk Impact", risk_impact),
    ]:
        if payload:
            context_rows.append({"Context": label, "Details": payload})
    if context_rows:
        st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No contextual analytics are available.")

    st.markdown("#### Full Recommendation Row")
    if recommendation_row is not None:
        st.dataframe(recommendation_row.to_frame().T, use_container_width=True, hide_index=True)
    else:
        st.caption("No full recommendation row is available.")

    st.markdown("#### Full Player Record")
    st.dataframe(player_row.to_frame().T, use_container_width=True, hide_index=True)
