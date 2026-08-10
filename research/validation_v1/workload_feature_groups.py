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


# Step 3b Item 1: missed-games risk / durability. Not position-gated -- all
# four positions carry injury/absence risk the same way. See
# build_injury_features_v1.py.
DURABILITY = ["prior_durability_score"]


# Step 11: garbage-time-adjusted production. Raw prior-production column ->
# its non-garbage counterpart. Columns with no counterpart (fantasy points,
# games, passing stats) are left untouched, so the head-to-head swaps ONLY
# what the adjustment can actually clean and holds everything else fixed.
GARBAGE_ADJUSTED_SWAP = {
    "prior_targets": "prior_non_garbage_targets",
    "prior_receiving_yards": "prior_non_garbage_receiving_yards",
    "prior_receiving_tds": "prior_non_garbage_receiving_tds",
    "prior_carries": "prior_non_garbage_carries",
    "prior_rushing_yards": "prior_non_garbage_rushing_yards",
    "prior_rushing_tds": "prior_non_garbage_rushing_tds",
}


def garbage_adjusted(prior_production: list[str]) -> list[str]:
    """Same feature list with cleanable columns swapped for adjusted ones."""
    return [GARBAGE_ADJUSTED_SWAP.get(col, col) for col in prior_production]


# Step 12: route participation / targets-per-route-run. Receiving metrics,
# so RB/WR/TE only -- QBs are on the field for every pass play, which makes
# their route rate ~1.0 and their TPRR exactly 0.000 (measured). Feeding
# those to a QB model is pure noise, the same reason air yards is excluded
# from QB_WORKLOAD.
ROUTE = [
    "prior_route_participation_rate",
    "prior_targets_per_route_run",
    "prior_yards_per_route_run",
]
POSITION_ROUTE_FEATURES = {"QB": [], "RB": ROUTE, "WR": ROUTE, "TE": ROUTE}


# Step 13: team offensive environment. Two families kept SEPARATE so the
# level claim and the mean-reversion claim cannot be confused for each other.
# Team-level context, so no position gating.
OFFENSE_LEVEL = [
    "prior_team_plays_pg",
    "prior_team_pass_rate",
    "prior_team_tds_pg",
    "prior_team_epa_pg",
]
# The actual hypothesis: how far the offense outran its OWN trailing 3-year
# norm. Positive = regression candidate. Team context is only ~40% persistent
# year-over-year, so if the market extrapolates last season harder than that,
# these players are overpriced.
OFFENSE_REVERSION = [
    "prior_team_plays_pg_vs_baseline",
    "prior_team_pass_rate_vs_baseline",
    "prior_team_tds_pg_vs_baseline",
    "prior_team_epa_pg_vs_baseline",
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
    route = POSITION_ROUTE_FEATURES[position]
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
        # adp_all_v1 is the comprehensive stack -- every candidate block
        # tried so far, including durability. This is the group Step 5
        # (Reopened) fits with lasso_logistic/hist_gradient_boosting to
        # test whether combining features beats ADP, since individually
        # testing each block discards any complementary signal between them.
        "adp_all_v1": ADP_BASELINE + prior_production + workload + DIVERGENCE + archetypes + DURABILITY + route,
        # Step 3b Item 1: missed-games risk / durability.
        "injury_v1_only": DURABILITY,
        "adp_injury_v1": ADP_BASELINE + DURABILITY,
        # Step 11 head-to-head: identical to adp_prior_production_baseline
        # except the cleanable production columns are garbage-adjusted. The
        # ONLY question is whether the cleaned input beats the dirty one, so
        # the two groups must be otherwise identical.
        "adp_prior_production_garbage_adjusted": ADP_BASELINE + garbage_adjusted(prior_production),
        # Step 12: route participation / TPRR. Empty for QB, so model_specs()
        # skips those groups automatically.
        "route_v1_only": route,
        "adp_route_v1": ADP_BASELINE + route,
        # Step 13: level (control, expected to fail) vs reversion (hypothesis).
        "adp_offense_level_v1": ADP_BASELINE + OFFENSE_LEVEL,
        "adp_offense_reversion_v1": ADP_BASELINE + OFFENSE_REVERSION,
    }
