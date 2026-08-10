import json
import pandas as pd
import streamlit as st
from datetime import datetime

from draftkit.archetypes import (
    archetype_label,
    archetype_note,
    build_roster_risk_report,
    risk_profile,
)
from draftkit.data_access import get_available_players_df, load_players_df, safe_col
from draftkit.conviction import build_conviction_report
from draftkit.championship_equity_v2 import build_championship_equity_v2_df
from draftkit.draft_analysis import (
    build_recommendation_rankings_df,
    get_tier_warning,
    get_turn_aware_falloff_recommendation,
)
from draftkit.draft_state import (
    get_current_pick_number,
    get_league_size,
    get_my_draft_slot,
    get_my_team_positions,
    get_next_pick_distance,
    handle_draft_action,
    handle_my_pick,
    init_session_state,
    keep_session_state_alive,
)
from draftkit.draft_simulation import calculate_availability_probability
from draftkit.draft_simulator import build_simulation_report
from draftkit.draft_strategy import build_draft_strategy
from draftkit.performance_profiler import create_profiler, track_optional
from draftkit.recommendation_consensus import build_consensus_recommendations
from draftkit.recommendation_explainer import (
    build_player_decision_card,
    prime_recommendation_explainer_cache,
)
from draftkit.ui_helpers import render_league_settings_sidebar


st.set_page_config(page_title="Draft Command Center", layout="wide")

init_session_state()
keep_session_state_alive()

# This previously blanked player_position_map to {} before rendering the
# sidebar and restored it afterward. The sidebar's My Team list reads that
# map (ui_helpers.py: player_positions.get(player, "Unknown")), so blanking
# it meant every rostered player rendered as "Unknown" here -- while the
# same player showed his real position on other pages. The blank/restore
# also had no upside: render_league_settings_sidebar() calls
# get_player_position_map() with no dataframe, which only reads and writes
# back the same map, so there was nothing to protect against.
render_league_settings_sidebar()

st.title("Draft Command Center")
st.caption("Track your roster, draft status, and next pick from one room.")
show_perf_metrics = st.sidebar.checkbox("Developer performance metrics", value=False)
perf_profiler = create_profiler(show_perf_metrics)

FAST_MODE_CHAMPIONSHIP_SIMULATIONS = 35
FAST_MODE_AVAILABILITY_SIMULATIONS = 35
FAST_MODE_FUTURE_IMPACT_SIMULATIONS = 50
ADVANCED_CANDIDATE_LIMIT = 5


@st.cache_resource(show_spinner=False)
def _draft_mode_runtime_config():
    return {
        "championship_simulations": FAST_MODE_CHAMPIONSHIP_SIMULATIONS,
        "availability_simulations": FAST_MODE_AVAILABILITY_SIMULATIONS,
        "future_impact_simulations": FAST_MODE_FUTURE_IMPACT_SIMULATIONS,
        "advanced_candidate_limit": ADVANCED_CANDIDATE_LIMIT,
    }


runtime_config = _draft_mode_runtime_config()


def _ensure_recommendation_history():
    if "recommendation_history" not in st.session_state:
        st.session_state["recommendation_history"] = []


def _record_recommendation_decision(selected_player, recommendation_df):
    _ensure_recommendation_history()

    top_recommendation = (
        recommendation_df.iloc[0].to_dict()
        if recommendation_df is not None and not recommendation_df.empty
        else {}
    )
    recommended_player = top_recommendation.get("player_name")
    recommendation_score = top_recommendation.get("final_score")

    st.session_state["recommendation_history"].append({
        "pick_number": get_current_pick_number(),
        "recommended_player": recommended_player,
        "selected_player": selected_player,
        "recommendation_score": recommendation_score,
        "recommendation_followed": (
            str(selected_player).strip().lower()
            == str(recommended_player).strip().lower()
            if recommended_player
            else False
        ),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def _format_score(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "N/A"


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


def _format_model_score(score):
    formatted_score = _format_score(score)
    return "N/A" if formatted_score == "N/A" else f"{formatted_score} / 100"


def _format_recommendation_grade(score):
    return f"{_recommendation_grade(score)} Recommendation"


def _format_confidence_percent(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_equity_delta(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _availability_recommendation(player):
    survival_probability = player.get("survival_probability")
    try:
        survival_value = float(survival_probability)
    except (TypeError, ValueError):
        return "TAKE NOW"

    if survival_value >= 70:
        return "CAN WAIT"
    if survival_value >= 45:
        return "HIGH RISK WAIT"
    return "TAKE NOW"


def _championship_impact_level(player):
    delta = player.get("equity_delta")
    try:
        delta_value = float(delta)
    except (TypeError, ValueError):
        delta_value = 0.0

    if delta_value >= 4:
        return "HIGH"
    if delta_value >= 1.5:
        return "MEDIUM"
    return "LOW"


def _get_availability_lookup():
    availability_df = _cached_availability_df(
        _get_draft_compute_key(),
        num_simulations=75,
        seed=17,
    )
    if availability_df.empty:
        return {}, availability_df

    return {
        str(row["player_name"]).lower(): row.to_dict()
        for _, row in availability_df.iterrows()
    }, availability_df


def _get_draft_compute_key():
    roster_settings = st.session_state.get("roster_settings", {})
    return (
        get_current_pick_number(),
        get_league_size(),
        get_my_draft_slot(),
        tuple(st.session_state.get("my_team", [])),
        tuple(st.session_state.get("drafted_players", [])),
        tuple(sorted(roster_settings.items())),
    )


def _recommendation_records_json(recommendation_df, limit=12):
    if recommendation_df is None or recommendation_df.empty:
        return "[]"
    return json.dumps(
        recommendation_df.head(limit).to_dict("records"),
        default=str,
        sort_keys=True,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_recommendation_df(draft_compute_key):
    return build_recommendation_rankings_df()


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_conviction_df(draft_compute_key, recommendation_records_json, limit):
    recommendation_records = json.loads(recommendation_records_json)
    if not recommendation_records:
        return pd.DataFrame()
    return build_conviction_report(pd.DataFrame(recommendation_records), limit=limit)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_availability_df(draft_compute_key, num_simulations, seed):
    try:
        return calculate_availability_probability(
            num_simulations=num_simulations,
            seed=seed,
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_future_impact_report(
    draft_compute_key,
    recommendation_records_json,
    limit,
    num_simulations,
    seed,
):
    recommendation_records = json.loads(recommendation_records_json)
    if not recommendation_records:
        return pd.DataFrame()

    recommendation_subset_df = pd.DataFrame(recommendation_records)
    return build_simulation_report(
        recommendations_df=recommendation_subset_df,
        limit=limit,
        num_simulations=num_simulations,
        seed=seed,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_championship_impact_report(
    draft_compute_key,
    players_df,
    recommendation_records_json,
    current_roster,
    draft_round,
    roster_settings,
    league_size,
    limit,
    num_simulations,
    seed,
):
    recommendation_records = json.loads(recommendation_records_json)
    if not recommendation_records:
        return pd.DataFrame()

    recommendation_subset_df = pd.DataFrame(recommendation_records)
    return build_championship_equity_v2_df(
        players_df=players_df,
        candidates_df=recommendation_subset_df,
        current_roster=list(current_roster),
        draft_round=draft_round,
        projected_roster_construction=dict(roster_settings),
        league_size=league_size,
        num_simulations=num_simulations,
        seed=seed,
        limit=limit,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_consensus_report(
    draft_compute_key,
    recommendation_records_json,
    availability_records_json,
    future_impact_records_json,
    championship_equity_records_json,
    limit,
):
    recommendation_records = json.loads(recommendation_records_json)
    if not recommendation_records:
        return {
            "top_recommendation": None,
            "categories": {},
            "consensus_scores": [],
        }

    return build_consensus_recommendations(
        recommendations_df=pd.DataFrame(recommendation_records),
        availability_df=pd.DataFrame(json.loads(availability_records_json)),
        future_impact_df=pd.DataFrame(json.loads(future_impact_records_json)),
        championship_equity_df=pd.DataFrame(json.loads(championship_equity_records_json)),
        limit=limit,
    )


def _records_json(df, limit=None):
    if df is None or df.empty:
        return "[]"
    out = df.head(limit).copy() if limit is not None else df.copy()
    return json.dumps(out.to_dict("records"), default=str, sort_keys=True)

raw_df = load_players_df()
avail_df = get_available_players_df()

if raw_df.empty:
    st.error("No player database loaded. Check your SQLite file and table selection.")
    st.stop()

player_col = safe_col(raw_df, ["player_name", "Player", "player", "name", "full_name"])
pos_col = safe_col(raw_df, ["position", "pos", "Pos", "Position", "POS"])
team_col = safe_col(raw_df, ["team", "Team", "team_abbr", "TEAM"])
proj_col = safe_col(
    raw_df,
    ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"],
)
adp_col = safe_col(raw_df, ["adp", "ADP", "rank", "Rank"])
bye_col = safe_col(raw_df, ["bye_week", "Bye", "bye"])

if player_col is None:
    st.error("Could not find a player-name column in the loaded data.")
    st.stop()

my_team = st.session_state.get("my_team", [])
position_counts = get_my_team_positions(raw_df)
position_order = ["QB", "RB", "WR", "TE", "DST", "K"]

if not avail_df.empty:
    board_df = avail_df.copy()
else:
    board_df = pd.DataFrame()

with track_optional(perf_profiler, "recommendation_generation"):
    recommendation_df = _cached_recommendation_df(_get_draft_compute_key())
st.session_state["latest_recommendation_df"] = recommendation_df
recommendation_lookup = {}
if not recommendation_df.empty:
    recommendation_lookup = {
        str(row["player_name"]).lower(): row
        for _, row in recommendation_df.iterrows()
    }

for numeric_col in [proj_col, adp_col]:
    if numeric_col is not None and numeric_col in board_df.columns:
        board_df[numeric_col] = pd.to_numeric(board_df[numeric_col], errors="coerce")

if player_col is not None and not board_df.empty and recommendation_lookup:
    board_df["recommendation_score"] = board_df[player_col].astype(str).str.lower().map(
        lambda name: recommendation_lookup.get(name, {}).get("final_score")
    )
    board_df["archetype"] = board_df[player_col].astype(str).str.lower().map(
        lambda name: recommendation_lookup.get(name, {}).get("archetype")
    )
    board_df["team_fit_bonus"] = board_df[player_col].astype(str).str.lower().map(
        lambda name: recommendation_lookup.get(name, {}).get("team_fit_bonus")
    )
    board_df["position_value_score"] = board_df[player_col].astype(str).str.lower().map(
        lambda name: recommendation_lookup.get(name, {}).get("position_value_score")
    )

with track_optional(perf_profiler, "tier_warning"):
    tier_warning = get_tier_warning()
if tier_warning and any(
    status in tier_warning["headline"].lower()
    for status in ["thin", "shrinking"]
):
    st.warning(
        f"Tier Warning: {tier_warning['headline']} - {tier_warning['message']}",
    )

if recommendation_df.empty:
    st.info("No recommendations are available yet.")
else:
    recommendation_center_df = recommendation_df.head(8).copy().reset_index(drop=True)
    with track_optional(perf_profiler, "conviction_report"):
        conviction_df = _cached_conviction_df(
            _get_draft_compute_key(),
            _recommendation_records_json(recommendation_df, limit=8),
            limit=8,
        )
    conviction_lookup = {
        str(row["player_name"]).lower(): row.to_dict()
        for _, row in conviction_df.iterrows()
    } if not conviction_df.empty else {}
    with track_optional(perf_profiler, "availability_simulation", simulations=runtime_config["availability_simulations"]):
        availability_df = _cached_availability_df(
            _get_draft_compute_key(),
            num_simulations=runtime_config["availability_simulations"],
            seed=17,
        )
        availability_lookup = {
            str(row["player_name"]).lower(): row.to_dict()
            for _, row in availability_df.iterrows()
        } if not availability_df.empty else {}
    st.session_state["latest_availability_df"] = availability_df
    prime_recommendation_explainer_cache(
        recommendations_df=recommendation_df,
        availability_df=availability_df,
    )
    run_future_impact = st.sidebar.toggle(
        "Run Future Impact simulations",
        value=False,
        key="run_future_impact_simulations",
        help="Runs 100+ Monte Carlo paths for the top recommendations. Leave off for faster reruns.",
    )
    if run_future_impact:
        try:
            with track_optional(
                perf_profiler,
                "future_impact_simulation",
                candidates=runtime_config["advanced_candidate_limit"],
                simulations=runtime_config["future_impact_simulations"],
            ):
                future_impact_df = _cached_future_impact_report(
                    _get_draft_compute_key(),
                    _recommendation_records_json(recommendation_df, limit=runtime_config["advanced_candidate_limit"]),
                    limit=runtime_config["advanced_candidate_limit"],
                    num_simulations=runtime_config["future_impact_simulations"],
                    seed=29,
                )
        except Exception:
            future_impact_df = pd.DataFrame()
    else:
        future_impact_df = pd.DataFrame()

    future_impact_lookup = {
        str(row["player_name"]).lower(): row.to_dict()
        for _, row in future_impact_df.iterrows()
    } if not future_impact_df.empty else {}
    st.session_state["latest_future_impact_df"] = future_impact_df

    current_pick_for_equity = get_current_pick_number()
    league_size_for_equity = get_league_size()
    draft_round_for_equity = (
        ((current_pick_for_equity - 1) // league_size_for_equity) + 1
        if league_size_for_equity
        else 1
    )
    roster_settings_for_equity = st.session_state.get("roster_settings", {})
    try:
        with track_optional(
            perf_profiler,
            "championship_equity_v2_fast",
            candidates=runtime_config["advanced_candidate_limit"],
            simulations=runtime_config["championship_simulations"],
        ):
            championship_equity_df = _cached_championship_impact_report(
                _get_draft_compute_key(),
                raw_df,
                _recommendation_records_json(recommendation_df, limit=runtime_config["advanced_candidate_limit"]),
                tuple(my_team),
                draft_round_for_equity,
                tuple(sorted(roster_settings_for_equity.items())),
                league_size_for_equity,
                limit=runtime_config["advanced_candidate_limit"],
                num_simulations=runtime_config["championship_simulations"],
                seed=53,
            )
    except Exception:
        championship_equity_df = pd.DataFrame()

    championship_impact_lookup = {
        str(row["player_name"]).lower(): row.to_dict()
        for _, row in championship_equity_df.iterrows()
    } if not championship_equity_df.empty else {}
    st.session_state["latest_championship_equity_df"] = championship_equity_df

    with track_optional(perf_profiler, "consensus_generation", candidates=12):
        consensus_report = _cached_consensus_report(
            _get_draft_compute_key(),
            _recommendation_records_json(recommendation_df, limit=12),
            _records_json(availability_df),
            _records_json(future_impact_df),
            _records_json(championship_equity_df),
            limit=12,
        )
    consensus_top = consensus_report.get("top_recommendation")
    st.session_state["latest_consensus_report"] = consensus_report

    consensus_scores = consensus_report.get("consensus_scores", [])

    if consensus_top:
        st.markdown("#### Recommended Pick")
        hero = st.container(border=True)
        with hero:
            hero_left, hero_middle, hero_right = st.columns([1.35, 1.15, 1])
            with hero_left:
                st.markdown(f"### {consensus_top.get('player_name')}")
                st.caption(
                    f"{consensus_top.get('position', '')} - {consensus_top.get('team', '')}"
                )
                st.markdown(f"## {_format_recommendation_grade(consensus_top.get('consensus_score'))}")
                st.caption("How strongly Guaranteed Play recommends making this pick right now.")
            with hero_middle:
                st.metric(
                    "Championship Impact",
                    f"{_format_equity_delta(consensus_top.get('equity_delta'))} Championship Odds",
                )
                st.metric("Impact Level", _championship_impact_level(consensus_top))
            with hero_right:
                st.caption("Availability Recommendation")
                st.success(_availability_recommendation(consensus_top))
                st.metric("Confidence %", _format_confidence_percent(consensus_top.get("confidence")))
                st.metric("Model Score", _format_model_score(consensus_top.get("consensus_score")))

        st.markdown("#### Why This Pick")
        why_col, risk_col = st.columns([1.4, 1])
        with why_col:
            reasons = consensus_top.get("why_models_agree", [])[:3]
            if reasons:
                for reason in reasons:
                    st.markdown(f"Reason: {reason}")
            else:
                st.caption("No model agreement reasons available.")
        with risk_col:
            st.caption("Primary Risk")
            risks = consensus_top.get("key_risks", [])
            if risks:
                st.warning(f"Risk: {risks[0]}")
            else:
                st.success("No major consensus-level risk flags.")

        st.markdown("#### Top Alternatives")
        alternatives = consensus_scores[1:4]
        if alternatives:
            alt_cols = st.columns(len(alternatives))
            for alt_col, alt in zip(alt_cols, alternatives):
                with alt_col:
                    with st.container(border=True):
                        st.write(f"**{alt.get('player_name')}**")
                        st.caption(
                            f"{alt.get('position', '')} - {alt.get('team', '')}"
                        )
                        st.metric("Recommendation Grade", _recommendation_grade(alt.get("consensus_score")))
                        st.metric("Model Score", _format_model_score(alt.get("consensus_score")))
                        st.metric("Confidence %", _format_confidence_percent(alt.get("confidence")))
                        st.metric(
                            "Championship Impact",
                            f"{_format_equity_delta(alt.get('equity_delta'))} Odds",
                        )
                        st.caption(_availability_recommendation(alt))
        else:
            st.caption("No alternatives are available yet.")

        st.markdown("#### Championship Impact")
        impact_cols = st.columns(4)
        impact_cols[0].metric("Impact Level", _championship_impact_level(consensus_top))
        impact_cols[1].metric(
            "Championship Odds Added",
            _format_equity_delta(consensus_top.get("equity_delta")),
        )
        impact_cols[2].metric(
            "Portfolio Classification",
            str(consensus_top.get("portfolio_classification") or "N/A").replace("_", " "),
        )
        impact_cols[3].metric(
            "League-Winning Upside",
            _format_model_score(consensus_top.get("league_winning_upside")),
        )
    else:
        st.info("No recommendation is available yet.")

    summary_rows = []
    recommendation_player_names = recommendation_center_df["player_name"].astype(str).tolist()
    for idx, rec_row in recommendation_center_df.iterrows():
        player_name = rec_row["player_name"]
        conviction = conviction_lookup.get(str(player_name).lower(), {})
        summary_rows.append({
            "Rank": idx + 1,
            "Player": player_name,
            "Position": rec_row.get("position"),
            "Team": rec_row.get("team"),
            "Conviction": conviction.get("conviction_level"),
            "Conviction Score": conviction.get("conviction_score"),
            "Model Score": round(float(rec_row.get("final_score", 0.0)), 2),
        })

    summary_df = pd.DataFrame(summary_rows)

    draft_strategy = build_draft_strategy(recommendation_df)
    roster_path = draft_strategy.get("roster_path", {})
    next_pick_plan = draft_strategy.get("next_pick_plan", {})
    primary_target = draft_strategy.get("primary_target") or {}
    secondary_target = draft_strategy.get("secondary_target") or {}
    contingency_target = draft_strategy.get("contingency_target") or {}

    with track_optional(perf_profiler, "turn_aware_falloff"):
        falloff = get_turn_aware_falloff_recommendation()

    with st.expander("Advanced Analytics", expanded=False):
        st.markdown("#### Model Score Breakdown")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if consensus_scores:
            st.markdown("#### Consensus Breakdown")
            consensus_display_df = pd.DataFrame(consensus_scores)
            consensus_display_df = consensus_display_df.rename(columns={
                "consensus_rank": "Model Rank",
                "player_name": "Player",
                "position": "Position",
                "team": "Team",
                "consensus_score": "Model Score",
                "confidence": "Confidence %",
                "base_score": "Base Score",
                "value_score": "Value Score",
                "roster_fit_score": "Roster Fit Score",
                "risk_score": "Risk Score",
                "future_pick_impact_score": "Future Pick Impact Score",
                "championship_equity_score": "Championship Equity Score",
                "survival_probability": "Survival Probability",
                "models_top_5_count": "Models Top 5 Count",
                "average_model_rank": "Average Model Rank",
                "disagreement_score": "Disagreement Score",
                "model_ranks": "Model Ranks",
                "why_models_agree": "Why Models Agree",
                "key_risks": "Key Risks",
            })
            consensus_columns = [
                "Model Rank",
                "Player",
                "Position",
                "Team",
                "Model Score",
                "Confidence %",
                "Base Score",
                "Value Score",
                "Roster Fit Score",
                "Risk Score",
                "Future Pick Impact Score",
                "Championship Equity Score",
                "Survival Probability",
            ]
            available_consensus_columns = [
                column for column in consensus_columns
                if column in consensus_display_df.columns
            ]
            st.dataframe(
                consensus_display_df[available_consensus_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Model Agreement Metrics")
            agreement_columns = [
                "Model Rank",
                "Player",
                "Models Top 5 Count",
                "Average Model Rank",
                "Disagreement Score",
                "Model Ranks",
            ]
            available_agreement_columns = [
                column for column in agreement_columns
                if column in consensus_display_df.columns
            ]
            st.dataframe(
                consensus_display_df[available_agreement_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Confidence Details")
            confidence_columns = [
                "Model Rank",
                "Player",
                "Confidence %",
                "Why Models Agree",
                "Key Risks",
            ]
            available_confidence_columns = [
                column for column in confidence_columns
                if column in consensus_display_df.columns
            ]
            st.dataframe(
                consensus_display_df[available_confidence_columns],
                use_container_width=True,
                hide_index=True,
            )

        if not championship_equity_df.empty:
            st.markdown("#### Championship Equity Details")
            st.dataframe(championship_equity_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Championship Equity Details")
            st.caption("No championship equity detail rows are available.")

        if not future_impact_df.empty:
            st.markdown("#### Future Impact Analysis")
            st.dataframe(future_impact_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Future Impact Analysis")
            st.caption("Future impact simulations are off or unavailable.")

        if not availability_df.empty:
            st.markdown("#### Monte Carlo Outputs")
            st.dataframe(availability_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Monte Carlo Outputs")
            st.caption("No availability simulation output is available.")

        st.markdown("#### Simulation Metrics")
        sim_metric_cols = st.columns(4)
        sim_metric_cols[0].metric("Availability Sims", runtime_config["availability_simulations"])
        sim_metric_cols[1].metric("Championship Sims", runtime_config["championship_simulations"])
        sim_metric_cols[2].metric("Future Impact Sims", runtime_config["future_impact_simulations"])
        sim_metric_cols[3].metric("Advanced Candidates", runtime_config["advanced_candidate_limit"])

        for idx, rec_row in recommendation_center_df.iterrows():
            rank = idx + 1
            player_name = rec_row["player_name"]
            conviction = conviction_lookup.get(str(player_name).lower(), {})
            conviction_level = conviction.get("conviction_level", "LOW")
            conviction_score = conviction.get("conviction_score")

            expander_label = (
                f"#{rank} {player_name} | {rec_row.get('position')} | "
                f"{conviction_level} CONVICTION | "
                f"Conviction {_format_score(conviction_score)}"
            )
            with st.expander(expander_label, expanded=False):
                load_detail_key = f"recommendation_detail_loaded_{rank}_{player_name}"
                load_button_key = f"load_recommendation_detail_button_{rank}_{player_name}"
                load_details = st.button(
                    "Load detailed analysis",
                    key=load_button_key,
                    use_container_width=True,
                )
                if load_details:
                    st.session_state[load_detail_key] = True

                if not st.session_state.get(load_detail_key, False):
                    st.caption("Detailed model explanation is loaded on demand.")
                    continue

                with track_optional(perf_profiler, "decision_card_detail", player=player_name):
                    card = build_player_decision_card(
                        player_name,
                        recommendations_df=recommendation_df,
                    )

                recommended_player = card.get("recommended_player", {})
                score_breakdown = card.get("score_breakdown", {})
                detail_cols = st.columns([1.2, 1, 1])

                with detail_cols[0]:
                    st.markdown("#### Recommended Pick")
                    st.write(f"**{recommended_player.get('player_name', player_name)}**")
                    st.caption(
                        f"{recommended_player.get('position', rec_row.get('position'))} - "
                        f"{recommended_player.get('team', rec_row.get('team', ''))}"
                    )
                    st.success(conviction.get("conviction_message", "Multiple players are similarly valued."))

                with detail_cols[1]:
                    st.metric("Conviction Level", conviction_level.replace("_", " "))
                    st.metric("Conviction Score", _format_score(conviction_score))

                with detail_cols[2]:
                    st.metric("Model Score", _format_model_score(score_breakdown.get("overall_score")))
                    st.metric("Confidence %", _format_confidence_percent(conviction.get("signal_confidence")))

                reason_col, risk_col = st.columns(2)

                with reason_col:
                    st.markdown("#### Top Reasons")
                    reasons = conviction.get("primary_reasons") or card.get("reasons", [])
                    if reasons:
                        for reason in reasons[:3]:
                            st.markdown(f"- {reason}")
                    else:
                        st.caption("No supported reasons available.")

                with risk_col:
                    st.markdown("#### Watch Points")
                    secondary = conviction.get("secondary_reasons", [])
                    risks = card.get("risks", [])
                    watch_points = secondary or risks
                    if watch_points:
                        for risk in watch_points[:3]:
                            st.warning(risk)
                    else:
                        st.success("No major risk flags.")

                st.markdown("#### Score Breakdown")
                breakdown_fields = [
                    "overall_score",
                    "need_score",
                    "tier_score",
                    "projection_score",
                    "opponent_score",
                    "risk_score",
                    "team_fit_score",
                    "availability_score",
                ]
                breakdown_cols = st.columns(4)
                for metric_idx, field in enumerate(breakdown_fields):
                    with breakdown_cols[metric_idx % 4]:
                        st.metric(
                            field.replace("_", " ").title(),
                            _format_score(score_breakdown.get(field)),
                        )

                championship_impact = championship_impact_lookup.get(str(player_name).lower(), {})
                if championship_impact:
                    st.markdown("#### Championship Impact")
                    championship_cols = st.columns(4)
                    championship_cols[0].metric(
                        "Championship Equity",
                        _format_score(championship_impact.get("championship_equity")),
                    )
                    championship_cols[1].metric(
                        "Equity Delta",
                        _format_equity_delta(championship_impact.get("equity_delta")),
                    )
                    championship_cols[2].metric(
                        "League-Winning Upside",
                        _format_score(championship_impact.get("league_winning_upside")),
                    )
                    championship_cols[3].metric(
                        "Portfolio",
                        str(championship_impact.get("portfolio_classification") or "N/A").replace("_", " "),
                    )

                alternatives = card.get("alternatives", [])
                if alternatives:
                    st.markdown("#### Alternatives")
                    alt_rows = []
                    for alt_rank, alt in enumerate(alternatives, start=2):
                        alt_availability = availability_lookup.get(
                            str(alt.get("player_name")).lower(),
                            {},
                        )
                        alt_rows.append({
                            "Rank": f"#{alt_rank}",
                            "Player": alt.get("player_name"),
                            "Model Score": alt.get("overall_score"),
                            "Key Reason": alt.get("key_reason"),
                            "Survival Probability": alt_availability.get("availability_probability"),
                            "Drafted Before Next Pick": alt_availability.get(
                                "drafted_before_next_pick_probability"
                            ),
                        })

                    alt_df = pd.DataFrame(alt_rows)
                    display_alt_cols = [
                        "Rank",
                        "Player",
                        "Model Score",
                        "Key Reason",
                    ]
                    if alt_df["Survival Probability"].notna().any():
                        display_alt_cols.extend([
                            "Survival Probability",
                            "Drafted Before Next Pick",
                        ])

                    st.dataframe(
                        alt_df[display_alt_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                future_impact = future_impact_lookup.get(str(player_name).lower(), {})
                if future_impact:
                    st.markdown("#### Future Impact")
                    future_cols = st.columns(5)
                    future_cols[0].metric(
                        "Survival If Passed",
                        _format_score(future_impact.get("survival_probability")),
                    )
                    future_cols[1].metric(
                        "Expected Roster Value",
                        _format_score(future_impact.get("expected_roster_value")),
                    )
                    future_cols[2].metric(
                        "Expected Champ Equity",
                        _format_score(future_impact.get("expected_championship_equity")),
                    )
                    future_cols[3].metric(
                        "Construction Score",
                        _format_score(future_impact.get("expected_construction_score")),
                    )
                    future_cols[4].metric(
                        "Future Pick Quality",
                        _format_score(future_impact.get("future_pick_quality")),
                    )

                    sim_left, sim_right = st.columns(2)
                    surviving = future_impact.get("likely_surviving_targets", [])
                    lost = future_impact.get("likely_lost_targets", [])

                    with sim_left:
                        st.caption("Likely Surviving Targets")
                        if surviving:
                            for target in surviving[:5]:
                                st.markdown(f"- {target}")
                        else:
                            st.caption("No clear surviving targets.")

                    with sim_right:
                        st.caption("Likely Lost Targets")
                        if lost:
                            for target in lost[:5]:
                                st.markdown(f"- {target}")
                        else:
                            st.caption("No clear lost targets.")

        st.markdown("#### Draft Plan")
        plan_cols = st.columns([1, 1, 1, 1])
        plan_cols[0].metric("Current Strategy", draft_strategy.get("current_strategy", "N/A"))
        plan_cols[1].metric("Confidence %", _format_confidence_percent(draft_strategy.get("strategy_confidence_score")))
        plan_cols[2].metric("Primary Target", primary_target.get("player_name", "N/A"))
        plan_cols[3].metric("Backup Plan", secondary_target.get("player_name", "N/A"))

        st.caption(roster_path.get("summary", "No roster path available."))
        if contingency_target:
            st.caption(
                "Contingency: "
                f"{contingency_target.get('player_name')} "
                f"({contingency_target.get('position', 'N/A')})"
            )
        elif next_pick_plan.get("plan_summary"):
            st.caption(next_pick_plan["plan_summary"])

        if falloff and not falloff["headline"].startswith("No recommendation"):
            st.markdown("#### Availability Detail")
            st.caption(f"{falloff['headline']}: {falloff['message']}")

        surviving_targets = draft_strategy.get("surviving_targets", [])
        passed_targets = draft_strategy.get("passed_targets", [])
        if surviving_targets or passed_targets:
            st.markdown("#### Target Tracking")
            if surviving_targets:
                st.markdown("**Surviving Passed Targets**")
                st.dataframe(
                    pd.DataFrame(surviving_targets),
                    use_container_width=True,
                    hide_index=True,
                )
            lost_targets = [
                target
                for target in passed_targets
                if target.get("target_lost")
            ]
            if lost_targets:
                st.markdown("**Lost Targets**")
                st.dataframe(
                    pd.DataFrame(lost_targets),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("#### Recommendation History")
        _ensure_recommendation_history()
        history = st.session_state.get("recommendation_history", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        else:
            st.caption("No recommendation decisions recorded yet.")

st.divider()

top_cols = st.columns(2)

with top_cols[0]:
    st.subheader("My Team Snapshot")
    st.metric("Total Roster Size", len(my_team))

    count_text = " | ".join(
        f"{position}: {position_counts.get(position, 0)}"
        for position in ["QB", "RB", "WR", "TE"]
    )
    st.caption(count_text)

    roster_settings = st.session_state.get("roster_settings", {})
    needs = []
    for position in ["QB", "RB", "WR", "TE"]:
        target = int(roster_settings.get(position, 0))
        have = position_counts.get(position, 0)
        if have < target:
            needs.append(f"{position} need {target - have}")

    if needs:
        st.write("Needs: " + ", ".join(needs))
    else:
        st.write("Core starters filled")

    if my_team:
        st.caption("Roster: " + ", ".join(my_team[-6:]))
    else:
        st.caption("No players drafted yet.")

    # Roster composition risk. Points from touchdowns and explosive plays are
    # far less repeatable week to week than points from targets and carries,
    # so a roster can project well while having a fragile floor. Nothing else
    # on this page surfaces that.
    if my_team:
        # raw_df, not avail_df -- drafted players are removed from the
        # available pool, so the roster would never match against it.
        risk = build_roster_risk_report(my_team, raw_df, player_col=player_col)
        if risk["classified_count"]:
            st.markdown("**Roster Risk Mix**")
            renderer = {
                "warn": st.warning,
                "caution": st.info,
                "ok": st.success,
            }.get(risk["status"], st.caption)
            renderer(risk["message"])

            by_risk = risk["counts_by_risk"]
            st.caption(
                f"Volume-based: {by_risk['volume']} | TD/big-play: {by_risk['event']} | "
                f"Mixed: {by_risk['mixed']} | Unclassified: {risk['unclassified_count']}"
            )

            with st.expander("Archetype breakdown", expanded=False):
                for entry in risk["players"]:
                    if entry["label"]:
                        st.write(f"**{entry['player_name']}** — {entry['label']}")
                        st.caption(entry["note"])
                    else:
                        st.write(f"**{entry['player_name']}** — unclassified")
                        st.caption(entry["note"])
                st.caption(
                    "Thresholds are a rule of thumb (caution at 35% event-dependent, "
                    "warning at 50%), not a fitted model. Archetypes are descriptive "
                    "only and are deliberately not part of the ranking score."
                )

with top_cols[1]:
    st.subheader("Draft Status")

    current_pick = get_current_pick_number()
    league_size = get_league_size()
    draft_slot = get_my_draft_slot()
    round_number = ((current_pick - 1) // league_size) + 1 if league_size else 1
    next_pick_distance = get_next_pick_distance()

    status_cols = st.columns(2)
    status_cols[0].metric("Current Pick", current_pick)
    status_cols[1].metric("Round", round_number)
    status_cols[0].metric("League Size", league_size)
    status_cols[1].metric("Draft Slot", draft_slot)
    st.metric("Picks Until Next Selection", next_pick_distance)

st.divider()

with st.container():
    st.subheader("Best Available Draft Board")

    filter_cols = st.columns([2, 1, 1, 1])

    with filter_cols[0]:
        search_text = st.text_input("Search player", placeholder="Type a player name")

    with filter_cols[1]:
        if pos_col is not None and not board_df.empty:
            pos_options = ["All"] + sorted(
                board_df[pos_col].dropna().astype(str).unique().tolist()
            )
            selected_pos = st.selectbox("Position", pos_options, index=0)
        else:
            selected_pos = "All"
            st.selectbox("Position", ["All"], index=0, disabled=True)

    with filter_cols[2]:
        sort_options = ["Model Score", "Projection", "ADP", "Player"]
        sort_by = st.selectbox("Sort by", sort_options, index=0)

    with filter_cols[3]:
        show_top = st.selectbox("Rows", [10, 25, 50, 100], index=1)

    if board_df.empty:
        st.warning("No available players found.")
        st.stop()

    if search_text:
        board_df = board_df[
            board_df[player_col].astype(str).str.contains(
                search_text,
                case=False,
                na=False,
            )
        ].copy()

    if pos_col is not None and selected_pos != "All":
        board_df = board_df[board_df[pos_col].astype(str) == selected_pos].copy()

    if sort_by == "Model Score" and "recommendation_score" in board_df.columns:
        board_df = board_df.sort_values(
            by="recommendation_score",
            ascending=False,
            na_position="last",
        )
    elif sort_by == "Projection" and proj_col is not None:
        board_df = board_df.sort_values(
            by=proj_col,
            ascending=False,
            na_position="last",
        )
    elif sort_by == "ADP" and adp_col is not None:
        board_df = board_df.sort_values(
            by=adp_col,
            ascending=True,
            na_position="last",
        )
    else:
        board_df = board_df.sort_values(by=player_col, ascending=True)

    board_df = board_df.reset_index(drop=True)
    board_df["board_rank"] = board_df.index + 1
    top_df = board_df.head(show_top).copy()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Available", len(avail_df))
    metric_cols[1].metric("Filtered", len(board_df))
    metric_cols[2].metric("Showing", len(top_df))

    display_cols = ["board_rank", player_col]
    rename_map = {
        "board_rank": "Rank",
        player_col: "Player",
    }

    if pos_col:
        display_cols.append(pos_col)
        rename_map[pos_col] = "Position"
    if team_col:
        display_cols.append(team_col)
        rename_map[team_col] = "Team"
    if proj_col:
        display_cols.append(proj_col)
        rename_map[proj_col] = "Projection"
    if adp_col:
        display_cols.append(adp_col)
        rename_map[adp_col] = "ADP"
    if bye_col:
        display_cols.append(bye_col)
        rename_map[bye_col] = "Bye"
    if "recommendation_score" in top_df.columns:
        display_cols.append("recommendation_score")
        rename_map["recommendation_score"] = "Model Score"
    if "position_value_score" in top_df.columns:
        display_cols.append("position_value_score")
        rename_map["position_value_score"] = "Pos Value"
    # Real usage-derived archetype (bellcow RB, deep threat WR, ...). Shown
    # instead of the legacy STEADY/RISKY bucket, which is empty for 87% of
    # players and derived from scores that are no longer populated.
    if "archetype_primary" in top_df.columns:
        top_df["archetype_display"] = top_df["archetype_primary"].apply(archetype_label)
        top_df["risk_display"] = top_df["archetype_primary"].apply(
            lambda a: {"event": "TD / big-play", "volume": "Volume", "mixed": "Mixed"}.get(
                risk_profile(a), ""
            )
        )
        display_cols += ["archetype_display", "risk_display"]
        rename_map["archetype_display"] = "Archetype"
        rename_map["risk_display"] = "Scoring Type"

    if top_df.empty:
        st.info("No players match the current filters.")
    else:
        board_display_df = top_df[display_cols].rename(columns=rename_map)
        st.dataframe(
            board_display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Draft Player Search")
    st.caption("Search available players, confirm, and draft directly to My Team.")

    draft_search_text = st.text_input(
        "Search available player database",
        placeholder="Type a player name",
        key="draft_player_search",
    )

    draft_search_df = avail_df.copy()
    if draft_search_text:
        draft_search_df = draft_search_df[
            draft_search_df[player_col].astype(str).str.contains(
                draft_search_text,
                case=False,
                na=False,
            )
        ].copy()

    if draft_search_df.empty:
        st.info("No available players match that search.")
    else:
        if proj_col is not None and proj_col in draft_search_df.columns:
            draft_search_df[proj_col] = pd.to_numeric(
                draft_search_df[proj_col],
                errors="coerce",
            )
            draft_search_df = draft_search_df.sort_values(
                proj_col,
                ascending=False,
                na_position="last",
            )
        else:
            draft_search_df = draft_search_df.sort_values(player_col)

        draft_options = draft_search_df[player_col].dropna().astype(str).tolist()
        selected_draft_player = st.selectbox(
            "Player to draft",
            draft_options,
            key="draft_player_select",
        )
        selected_draft_row = draft_search_df[
            draft_search_df[player_col].astype(str) == selected_draft_player
        ].iloc[0]

        selected_position = str(selected_draft_row[pos_col]) if pos_col else "Unknown"
        selected_team = str(selected_draft_row[team_col]) if team_col else ""
        selected_projection = selected_draft_row[proj_col] if proj_col else "N/A"

        st.write(
            f"**{selected_draft_player}** "
            f"({selected_position}{', ' + selected_team if selected_team else ''})"
        )
        st.caption(f"Projection: {selected_projection}")

        # What kind of player this is, and what kind of risk comes with him.
        selected_archetype = selected_draft_row.get("archetype_primary")
        if archetype_label(selected_archetype):
            profile = risk_profile(selected_archetype)
            profile_text = {
                "event": "TD / big-play dependent",
                "volume": "volume-based",
                "mixed": "mixed",
            }.get(profile, profile)
            st.caption(
                f"Archetype: **{archetype_label(selected_archetype)}** ({profile_text}) — "
                f"{archetype_note(selected_archetype)}"
            )

        with st.popover("Draft to My Team", use_container_width=True):
            st.write(f"Are you sure you want to draft **{selected_draft_player}**?")
            st.caption("This will add the player to My Team and remove them from available players.")

            if st.button(
                "Confirm Draft",
                key="confirm_draft_selected_player_to_my_team",
                use_container_width=True,
            ):
                _record_recommendation_decision(
                    selected_draft_player,
                    recommendation_df,
                )
                handle_my_pick(selected_draft_row)
                st.rerun()

st.divider()

with st.expander("Data debug", expanded=False):
    st.write("Raw shape:", raw_df.shape)
    st.write("Available shape:", avail_df.shape)
    st.write("Player column:", player_col)
    st.write("Position column:", pos_col)
    st.write("Team column:", team_col)
    st.write("Projection column:", proj_col)
    st.write("ADP column:", adp_col)
    st.write("Bye column:", bye_col)

if perf_profiler is not None:
    with st.expander("Developer Performance Metrics", expanded=False):
        perf_summary = perf_profiler.summary()
        st.metric("Tracked Runtime (ms)", perf_summary["total_tracked_ms"])
        st.dataframe(
            pd.DataFrame(perf_summary["slowest_events"]),
            use_container_width=True,
            hide_index=True,
        )
