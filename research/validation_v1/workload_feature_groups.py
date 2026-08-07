"""
Step 3a workload/opportunity feature groups, shared by the four
evaluate_{qb,rb,wr,te}_models.py scripts so the incremental test is defined
identically across positions.

Each position gets only the features that are actually meaningful for it
(e.g. air yards / WOPR are receiving-role metrics, red-zone carries are a
rushing-role metric) rather than throwing every column at every model.

The groups are deliberately INCREMENTAL -- workload alone, ADP+workload,
and ADP+prior-production+workload -- so Step 4 can read off how much the
new Step 3a data adds on top of the ADP baseline, rather than blending it
into one undifferentiated feature dump.
"""
from __future__ import annotations

ADP_BASELINE = ["overall_adp", "positional_adp", "preseason_adp", "estimated_draft_round"]

DRAFT_CAPITAL = ["draft_round", "draft_pick_overall", "years_since_drafted"]

# Team-level context (same value for every player on a team-season).
VACATED = ["prior_team_vacated_target_share", "prior_team_vacated_carry_share"]

RECEIVING_WORKLOAD = [
    "prior_snap_share",
    "prior_redzone_target_share",
    "prior_air_yards",
    "prior_air_yards_share",
    "prior_wopr",
]

RUSHING_WORKLOAD = [
    "prior_snap_share",
    "prior_redzone_carry_share",
    "prior_redzone_target_share",
]

QB_WORKLOAD = [
    "prior_snap_share",
    "prior_redzone_carry_share",  # QB rushing at the goal line is real fantasy signal
]

POSITION_WORKLOAD_FEATURES = {
    "QB": QB_WORKLOAD + DRAFT_CAPITAL + VACATED,
    "RB": RUSHING_WORKLOAD + DRAFT_CAPITAL + VACATED,
    "WR": RECEIVING_WORKLOAD + DRAFT_CAPITAL + VACATED,
    "TE": RECEIVING_WORKLOAD + DRAFT_CAPITAL + VACATED,
}


# Step 4b divergence features -- explicit mispricing measures (one percentile
# rank minus another, within season+position) rather than quality levels.
# Same three for every position; see build_divergence_features_v1.py.
DIVERGENCE = [
    "div_opportunity_minus_efficiency",
    "div_wopr_minus_adp",
    "div_redzone_minus_overall_share",
]


# Step 4c: archetype membership scores. Only the archetypes belonging to the
# position are passed -- an RB's arch_deep_threat_wr is always NaN and would
# just be median-imputed noise.
POSITION_ARCHETYPE_FEATURES = {
    "QB": ["arch_dual_threat_qb", "arch_pocket_passer_qb"],
    "RB": [
        "arch_bellcow_rb", "arch_receiving_back_rb", "arch_early_down_rb",
        "arch_long_run_rb", "arch_goal_line_rb",
    ],
    "WR": ["arch_deep_threat_wr", "arch_possession_wr", "arch_alpha_wr"],
    "TE": ["arch_red_zone_te", "arch_receiving_te", "arch_blocking_te"],
}


def workload_feature_groups(position: str, prior_production: list[str]) -> dict[str, list[str]]:
    workload = POSITION_WORKLOAD_FEATURES[position]
    archetypes = POSITION_ARCHETYPE_FEATURES[position]
    return {
        "workload_v2_only": workload,
        "adp_workload_v2": ADP_BASELINE + workload,
        "adp_prior_production_workload_v2": ADP_BASELINE + prior_production + workload,
        # Divergence groups, incremental so each layer's contribution is
        # readable rather than blended into one undifferentiated dump.
        "divergence_v1_only": DIVERGENCE,
        "adp_divergence_v1": ADP_BASELINE + DIVERGENCE,
        "adp_workload_divergence_v1": ADP_BASELINE + workload + DIVERGENCE,
        "adp_prior_production_divergence_v1": ADP_BASELINE + prior_production + DIVERGENCE,
        # Step 4c archetype groups, again incremental so the archetype
        # scores' own contribution is readable.
        "archetype_v1_only": archetypes,
        "adp_archetype_v1": ADP_BASELINE + archetypes,
        "adp_workload_archetype_v1": ADP_BASELINE + workload + archetypes,
        "adp_all_v1": ADP_BASELINE + prior_production + workload + DIVERGENCE + archetypes,
    }
