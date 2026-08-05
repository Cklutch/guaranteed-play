import pandas as pd
import streamlit as st

from draftkit.data_access import load_players_df
from draftkit.draft_lab import (
    OPPONENT_ARCHETYPES,
    SUPPORTED_LEAGUE_SIZES,
    run_simulation_batch,
    simulate_complete_draft,
)
from draftkit.draft_state import init_session_state, keep_session_state_alive
from draftkit.model_evaluator import evaluate_draft_lab_results
from draftkit.ui_helpers import render_league_settings_sidebar


st.set_page_config(page_title="Draft Lab", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Draft Lab")
st.caption("Run full mock drafts against AI opponents and evaluate strategy performance.")


@st.cache_resource(show_spinner=False)
def _draft_lab_options():
    return {
        "league_sizes": list(SUPPORTED_LEAGUE_SIZES),
        "strategies": list(OPPONENT_ARCHETYPES),
        "batch_sizes": [10, 50, 100, 500],
    }


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_single_draft(players_df, draft_slot, league_size, strategy, roster_settings, seed):
    return simulate_complete_draft(
        players_df=players_df,
        draft_slot=draft_slot,
        league_size=league_size,
        strategy=strategy,
        roster_settings=dict(roster_settings),
        seed=seed,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_batch(players_df, draft_slot, league_size, strategy, batch_size, roster_settings, seed):
    return run_simulation_batch(
        players_df=players_df,
        draft_slot=draft_slot,
        league_size=league_size,
        strategy=strategy,
        batch_size=batch_size,
        roster_settings=dict(roster_settings),
        seed=seed,
    )


players_df = load_players_df()
if players_df.empty:
    st.error("No player database loaded. Draft Lab needs a player pool.")
    st.stop()

lab_options = _draft_lab_options()

with st.form("draft_lab_controls"):
    controls = st.columns([1, 1, 1.2, 1, 1])
    with controls[0]:
        league_size = st.selectbox("League Size", lab_options["league_sizes"], index=2)
    with controls[1]:
        draft_slot = st.selectbox("Draft Slot", list(range(1, league_size + 1)), index=0)
    with controls[2]:
        strategy = st.selectbox("Strategy", lab_options["strategies"], index=lab_options["strategies"].index("Balanced"))
    with controls[3]:
        batch_size = st.selectbox("Batch", lab_options["batch_sizes"], index=0)
    with controls[4]:
        seed = st.number_input("Seed", min_value=1, value=101, step=1)

    run_lab = st.form_submit_button("Run Simulation", use_container_width=True)

roster_settings = tuple(sorted(st.session_state.get("roster_settings", {}).items()))

if run_lab:
    with st.spinner("Running Draft Lab simulations..."):
        st.session_state["draft_lab_single_result"] = _cached_single_draft(
            players_df,
            draft_slot,
            league_size,
            strategy,
            roster_settings,
            int(seed),
        )
        st.session_state["draft_lab_batch_result"] = _cached_batch(
            players_df,
            draft_slot,
            league_size,
            strategy,
            batch_size,
            roster_settings,
            int(seed),
        )

single_result = st.session_state.get("draft_lab_single_result")
batch_result = st.session_state.get("draft_lab_batch_result")

if not single_result or not batch_result:
    st.info("Choose your settings and click Run Simulation.")
    st.stop()

evaluation = evaluate_draft_lab_results(batch_result)

summary_cols = st.columns(4)
summary_cols[0].metric(
    "Average Draft Grade",
    f"{batch_result.get('average_draft_grade', 0.0):.1f}",
)
summary_cols[1].metric(
    "Average Championship Equity",
    f"{batch_result.get('average_championship_equity', 0.0):.2f}",
)
summary_cols[2].metric(
    "Simulations",
    batch_result.get("simulations_run", 0),
)
recommendation_performance = batch_result.get("recommendation_performance", {})
summary_cols[3].metric(
    "Recommendation Follow Rate",
    f"{recommendation_performance.get('follow_rate', 0.0):.1f}%",
)

st.subheader("Single Mock Draft")
grade = single_result.get("draft_grade", {})
grade_cols = st.columns(6)
grade_cols[0].metric("Grade", grade.get("grade", "N/A"))
grade_cols[1].metric("Score", f"{grade.get('grade_score', 0.0):.1f}")
grade_cols[2].metric("Roster Strength", f"{grade.get('roster_strength', 0.0):.1f}")
grade_cols[3].metric("Balance", f"{grade.get('positional_balance', 0.0):.1f}")
grade_cols[4].metric("Value Gained", f"{grade.get('value_gained', 0.0):.1f}")
grade_cols[5].metric("Risk Profile", f"{grade.get('risk_profile', 0.0):.1f}")

roster_df = pd.DataFrame(single_result.get("final_roster", []))
if roster_df.empty:
    st.info("No roster was drafted. The player pool may be incomplete.")
else:
    display_cols = [
        column
        for column in ["player_name", "position", "team", "projection_points", "adp", "injury_risk"]
        if column in roster_df.columns
    ]
    st.dataframe(roster_df[display_cols], use_container_width=True, hide_index=True)

st.subheader("Batch Results")
batch_cols = st.columns(3)
with batch_cols[0]:
    st.markdown("#### Most Common Roster Builds")
    builds_df = pd.DataFrame(batch_result.get("most_common_roster_builds", []))
    if builds_df.empty:
        st.caption("No roster builds available.")
    else:
        st.dataframe(builds_df, use_container_width=True, hide_index=True)

with batch_cols[1]:
    st.markdown("#### Recommendation Performance")
    performance_rows = [
        {"Metric": key.replace("_", " ").title(), "Value": value}
        for key, value in recommendation_performance.items()
    ]
    st.dataframe(pd.DataFrame(performance_rows), use_container_width=True, hide_index=True)

with batch_cols[2]:
    st.markdown("#### Recommendation Failures")
    failure_counts = batch_result.get("failure_counts", {})
    if failure_counts:
        failure_df = pd.DataFrame(
            [{"Failure": key, "Count": value} for key, value in failure_counts.items()]
        )
        st.dataframe(failure_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No recurring recommendation failures flagged.")

st.subheader("Evaluation Summary")
calibration = evaluation.get("calibration", {})
calibration_cols = st.columns(3)
for idx, key in enumerate([
    "consensus_vs_final_grade",
    "confidence_vs_actual_outcome",
    "championship_equity_vs_simulated_wins",
]):
    metric = calibration.get(key, {})
    with calibration_cols[idx]:
        st.markdown(f"#### {key.replace('_', ' ').title()}")
        st.metric("Correlation", f"{metric.get('correlation', 0.0):.3f}")
        st.metric("Calibration Error", f"{metric.get('calibration_error', 0.0):.2f}")
        st.metric("Ranking Accuracy", f"{metric.get('ranking_accuracy', 0.0):.1f}%")

eval_cols = st.columns(4)
with eval_cols[0]:
    st.markdown("#### Positional Bias")
    bias = evaluation.get("positional_bias", {})
    for finding in bias.get("findings", []):
        st.caption(finding)
    rates = bias.get("position_rates", {})
    if rates:
        st.dataframe(
            pd.DataFrame([{"Position": key, "Rate": value} for key, value in rates.items()]),
            use_container_width=True,
            hide_index=True,
        )

with eval_cols[1]:
    st.markdown("#### Feature Importance")
    importance = evaluation.get("feature_importance", {})
    if importance:
        st.dataframe(
            pd.DataFrame([
                {"Feature": key.replace("_", " ").title(), "Weight": value}
                for key, value in importance.items()
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No feature importance available.")

with eval_cols[2]:
    st.markdown("#### Strategy Rankings")
    strategy_df = evaluation.get("strategy_rankings", pd.DataFrame())
    if strategy_df.empty:
        st.caption("Run multiple strategies to populate strategy comparison.")
    else:
        st.dataframe(strategy_df, use_container_width=True, hide_index=True)

with eval_cols[3]:
    st.markdown("#### Draft Slot Rankings")
    slot_df = evaluation.get("draft_slot_rankings", pd.DataFrame())
    if slot_df.empty:
        st.caption("No draft slot rankings available.")
    else:
        st.dataframe(slot_df, use_container_width=True, hide_index=True)

with st.expander("Ranked Failure Report", expanded=False):
    failure_report = evaluation.get("failure_report", pd.DataFrame())
    if failure_report.empty:
        st.caption("No repeated recommendation failures found.")
    else:
        st.dataframe(failure_report, use_container_width=True, hide_index=True)

with st.expander("Draft History", expanded=False):
    history_df = pd.DataFrame(single_result.get("draft_history", []))
    if history_df.empty:
        st.caption("No draft history available.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)

with st.expander("Recommendation History", expanded=False):
    rec_df = pd.DataFrame(single_result.get("recommendation_history", []))
    if rec_df.empty:
        st.caption("No recommendation history available.")
    else:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
