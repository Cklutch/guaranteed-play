import pandas as pd
import streamlit as st

from draftkit.championship_equity import build_championship_equity_df
from draftkit.data_pipeline import validate_player_data
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_simulation import calculate_availability_probability
from draftkit.draft_state import init_session_state, keep_session_state_alive
from draftkit.ranking_component_audit import build_top_100_component_audit
from draftkit.ui_helpers import render_league_settings_sidebar


st.set_page_config(page_title="Component Audit", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Top 100 Component Audit")
st.caption(
    "Ranks current scoring features by average influence and flags components "
    "responsible for more than 20% of final-score variance."
)


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_data_validation():
    return validate_player_data()


_validation = _cached_data_validation()
if not _validation.get("projection_coverage_ok", True):
    st.error(
        "Data quality: only {0}% of {1} ranked players (real ADP/tier/expert "
        "rank present) have a real projection ({2}% required). Recommendations "
        "below are not trustworthy until a real projections export is added to "
        "data/raw/.".format(
            _validation.get("ranked_projection_coverage", 0.0),
            _validation.get("ranked_player_count", 0),
            _validation.get("projection_coverage_min_pct", 90.0),
        )
    )
if _validation.get("missing_data_counts", {}).get("adp", 0) and _validation.get("ranked_player_count", 0):
    ranked = _validation.get("ranked_player_count", 0)
    missing_adp = _validation["missing_data_counts"]["adp"]
    if missing_adp >= ranked:
        st.warning(
            "Data quality: no real ADP data is loaded (0 of {0} ranked players "
            "have ADP). ADP-value scoring and ADP-based validation are not "
            "meaningful until a real ADP export is added to data/raw/.".format(ranked)
        )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_recommendations():
    return build_recommendation_rankings_df()


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_championship_equity():
    try:
        return build_championship_equity_df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_availability():
    try:
        return calculate_availability_probability(num_simulations=35, seed=17)
    except Exception:
        return pd.DataFrame()


recommendations_df = _cached_recommendations()

if recommendations_df.empty:
    st.info("No recommendation rankings are available.")
    st.stop()

with st.spinner("Building top-100 component audit..."):
    audit = build_top_100_component_audit(
        recommendations_df=recommendations_df,
        championship_equity_df=_cached_championship_equity(),
        availability_df=_cached_availability(),
        top_n=100,
    )

influence_df = audit["feature_influence"]
player_df = audit["player_contributions"]
variance_flags = audit["variance_flags"]
unsupported = audit["unsupported_features"]

summary_cols = st.columns(4)
summary_cols[0].metric("Players Audited", len(player_df))
summary_cols[1].metric("Features Audited", len(influence_df))
summary_cols[2].metric("Variance Flags", len(variance_flags))
summary_cols[3].metric("Unsupported Features", len(unsupported))

st.markdown("#### Feature Influence Ranking")
if influence_df.empty:
    st.caption("No feature influence data is available.")
else:
    display_influence = influence_df.rename(columns={
        "feature_label": "Feature",
        "source_column": "Source Column",
        "supported": "Supported",
        "average_feature_score": "Average Feature Score",
        "average_absolute_influence": "Average Absolute Influence",
        "average_influence_pct": "Average Influence %",
        "correlation_to_final_score": "Correlation To Final Score",
        "feature_variance": "Feature Variance",
        "score_variance_share_pct": "Score Variance Share %",
        "contributes_over_20_pct_variance": ">20% Variance",
    })
    ordered_cols = [
        "Feature",
        "Source Column",
        "Supported",
        "Average Feature Score",
        "Average Influence %",
        "Score Variance Share %",
        "Correlation To Final Score",
        "Feature Variance",
        ">20% Variance",
    ]
    st.dataframe(
        display_influence[[col for col in ordered_cols if col in display_influence.columns]],
        use_container_width=True,
        hide_index=True,
    )

if variance_flags:
    st.markdown("#### Components Above 20% Score Variance")
    st.dataframe(pd.DataFrame(variance_flags), use_container_width=True, hide_index=True)
else:
    st.success("No single component contributes more than 20% of observed score variance.")

if unsupported:
    st.markdown("#### Unsupported Or Missing Feature Inputs")
    st.caption(
        "These requested features are shown in the audit, but the current rankings "
        "do not expose numeric inputs for them."
    )
    st.write(", ".join(unsupported))

st.markdown("#### Top 100 Player Contribution Table")
if player_df.empty:
    st.caption("No player-level contribution data is available.")
else:
    st.dataframe(player_df, use_container_width=True, hide_index=True)

with st.expander("Method", expanded=False):
    st.markdown(
        """
        Feature influence is estimated from the current top-100 recommendation pool.
        Each requested feature is mapped to the best available numeric source column.
        Average influence combines each feature's correlation to final recommendation
        score with its observed spread. Variance share is based on relative squared
        correlation to final recommendation score.
        """
    )
