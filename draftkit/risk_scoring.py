"""Fantasy draft risk scorecard -- core data model and scoring.

v2: four weighted risk categories (injury, role_usage_td,
offense_environment, schedule_weather_venue), each scored 1 (low risk) to 5
(high risk). risk_index() rolls those into a single 0-100 score using
position-specific category weights (risk_weights.json), since risk
manifests differently by position (e.g. an RB's role risk matters more
than a QB's).

Two things are deliberately NOT part of the weighted composite, per the v2
restructure:
  - market (ADP mispricing) is a VALUE signal, not a risk signal -- it was
    diluting risk_index in v1. Tracked as ValueSignal.MARKET, reported
    separately (see risk_cli.py / Home.py), never blended into risk_index.
  - volatility (week-to-week variance) is a downstream SYMPTOM of role
    instability, not an independent risk driver. Computed as a display-only
    diagnostic in build_risk_variables.py, not a RiskCategory here at all.

risk_index() and player_from_variable_row() are written generically over
RiskCategory's members rather than naming each category's field by hand --
this is the second time the category set has changed, so hardcoding
per-category field names (v1's approach) was a needless source of drift.

Phase 3 (plan_risk_index_reweight_dynamic.pdf): risk_weights.json's
schedule_weather_venue weight, uniformly 0.10 across every position before
this phase, is now position-specific and real-predictive-value-anchored --
published analysis (ESPN's classic defense-consistency study; a 2026
review) puts preseason SOS at roughly 2.5-7.5% of real predictive value for
RB/WR/TE (genuinely marginal -- cut to 0.04) but 10%+ for QB specifically
(game-script/passing-volume effects are more stable season-long than
weekly matchup variance -- cut more conservatively, to 0.07). Freed weight
redistributed PROPORTIONALLY across each position's other 3 categories
(their relative shares to each other unchanged, just rescaled to fill the
gap) -- a simple, disclosed reallocation rule, not hand-picked per
category. Reasoned defaults pending validation, not final-calibrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

WEIGHTS_PATH = Path(__file__).resolve().parent / "risk_weights.json"


class Position(Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"


class RiskCategory(Enum):
    INJURY = "injury"
    ROLE_USAGE_TD = "role_usage_td"
    OFFENSE_ENVIRONMENT = "offense_environment"
    SCHEDULE_WEATHER_VENUE = "schedule_weather_venue"


class ValueSignal(Enum):
    """Reported alongside risk_index, never blended into it -- see module docstring."""
    MARKET = "market"


class DraftStatus(Enum):
    AVAILABLE = "available"
    MY_TEAM = "my_team"
    TAKEN_BY_OTHER = "taken_by_other"


@dataclass
class RiskVariable:
    name: str
    category: RiskCategory
    score: float  # 1.0-5.0
    weight_within_category: float = 1.0
    notes: str = ""


@dataclass
class Player:
    name: str
    position: Position
    adp: float
    variables: list[RiskVariable] = field(default_factory=list)

    def category_score(self, category: RiskCategory) -> float:
        vars_in_cat = [v for v in self.variables if v.category == category]
        if not vars_in_cat:
            return 0.0
        total_weight = sum(v.weight_within_category for v in vars_in_cat)
        if total_weight == 0:
            return 0.0
        return sum(v.score * v.weight_within_category for v in vars_in_cat) / total_weight


@dataclass
class RiskProfile:
    injury: float
    role_usage_td: float
    offense_environment: float
    schedule_weather_venue: float


def load_weights(path: Path = WEIGHTS_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def player_from_variable_row(row) -> Player | None:
    """Build a Player from one row of data/processed/risk_variables.csv.

    Shared by risk_cli.py (needs full Player/RiskVariable objects for its
    per-category breakdown display) and build_risk_variables.py (needs them
    only to compute and store a summary risk_index) -- extracted here so
    that row-to-Player construction exists in exactly one place. Returns
    None if the row's position doesn't map to a known Position (the CSV is
    already restricted to QB/RB/WR/TE by the build script, so this is a
    defensive fallback, not an expected path).

    v2: each RiskCategory is exactly one formula-computed score (see
    build_risk_variables.py's injury_risk_score() / role_usage_td_score() /
    offense_environment_score() / schedule_weather_venue_score()), read
    directly from "{category}_score" columns -- not an aggregate of several
    independently-bucketed sub-variables the way v1 built each category.
    ValueSignal columns (market) and display-only diagnostics (volatility)
    are read separately by callers, not through this Player/RiskVariable path.
    """
    position_raw = str(row.get("position") or "").upper()
    try:
        position = Position(position_raw)
    except ValueError:
        return None

    variables = []
    for category in RiskCategory:
        score = row.get(f"{category.value}_score")
        if pd.isna(score):
            continue
        variables.append(RiskVariable(name=category.value, category=category, score=float(score)))

    adp = row.get("adp")
    return Player(
        name=str(row["player_name"]),
        position=position,
        adp=float(adp) if pd.notna(adp) else float("inf"),
        variables=variables,
    )


def build_risk_profile(player: Player) -> RiskProfile:
    return RiskProfile(**{
        cat.value: player.category_score(cat) for cat in RiskCategory
    })


# Real severity floor (plan_risk_index_reweight_dynamic.pdf, Phase 1) --
# reasoned default, NOT final-calibrated (see draftkit/tests/test_risk_constraints.py
# for the anchor cases this is validated against). A simple weighted average
# dilutes a genuine outlier toward "average": Joe Burrow's real Injury=75
# (Grade 3 turf toe requiring surgery) blended down to a composite of just
# 49.6, "amber but unremarkable," because QB's offense_environment weight
# (0.45) swamped it. THRESHOLD=4.0 anchors to Burrow's own real (pre-decay)
# injury_score; FLOOR=60.0 lands him clearly "amber" (elevated, real
# concern) without pushing all the way to "red" (>66), since he's a real,
# recovering-not-ongoing case, not a confirmed season-long absence.
#
# Scoped to INJURY and ROLE_USAGE_TD only -- NOT all 4 categories. First
# draft applied it generically ("if ANY category clears the bar"), but the
# real, locked-in James Cook regression case caught why that's wrong:
# James Cook's real schedule_weather_venue=4.0 (a genuinely tough real
# slate, nothing more) tripped the same floor as a season-ending injury
# would, forcing his composite to 60 despite 3 of 4 categories sitting at
# the real safest floor -- exactly the "diluted outlier" problem this
# phase exists to FIX, now inverted into "one merely-elevated context
# category dominates." offense_environment/schedule_weather_venue are
# real, but team/context-level and lower-confidence than a player's own
# injury or role -- schedule specifically is being explicitly DOWN-weighted
# in Phase 3 for genuinely marginal (~2.5-7.5%) real predictive value,
# which would directly contradict letting it single-handedly floor the
# whole composite here. A 4.0 in a player-level category (injury, role/
# usage/TD) represents a real, severe, individual outcome; a 4.0 in a
# team-level category doesn't carry the same weight of evidence.
SEVERITY_FLOOR_THRESHOLD = 4.0
SEVERITY_FLOOR_VALUE = 60.0
SEVERITY_FLOOR_CATEGORIES = {RiskCategory.INJURY, RiskCategory.ROLE_USAGE_TD}


def risk_index(player: Player, weights: dict) -> float:
    """0-100 risk score: higher = riskier. See module docstring.

    Floor-subtracted scaling (real bug, diagnosed against James Cook's real
    numbers earlier this session, fixed here): each category's raw 1-5
    score is rescaled to 0-1 via (raw-1)/4 BEFORE weighting -- the same
    scaling Home.py's per-category display bars already use. The previous
    version weighted the raw 1-5 score directly against a ceiling of
    5*sum(weights), so a raw score of 1 (the safest possible value, meant
    to read as zero risk) still contributed 1/5=20% of its own weight to
    the total. Confirmed exactly against James Cook: raw (1.0, 1.1, 1.0,
    4.0) against RB weights (0.38/0.32/0.20/0.10) used to produce
    risk_index=26.6, driven mostly by that fixed baseline rather than his
    one real elevated category (schedule) -- with 3 of 4 categories
    sitting near-floor, the old formula couldn't get anywhere near 0.

    Only iterates categories the player actually has real data for (real
    bug caught during verification, not in the original fix spec):
    category_score() returns 0.0 -- not the 1.0 floor -- for a category
    with no RiskVariable at all (real, common case: offense_environment/
    schedule_weather_venue only cover 625/730 players, per
    build_risk_variables.py's own coverage report). Iterating the full
    RiskCategory enum unconditionally would floor-subtract that 0.0 too --
    (0-1)/4 = -0.25 -- turning "no data for this category" into a NEGATIVE
    contribution, which produced real risk_index values below zero (e.g.
    -4.4) for players missing 2 of 4 categories. Re-normalizing over just
    the present categories (weighted_sum and max_possible both restricted
    to them) treats missing data as neutral -- excluded, not penalized --
    consistent with how player_from_variable_row() already skips NaN
    categories when building player.variables in the first place."""
    w = weights[player.position.value]
    present_categories = {v.category for v in player.variables}
    if not present_categories:
        return 0.0
    weighted_sum = sum(
        ((player.category_score(cat) - 1) / 4) * w[cat.value] for cat in present_categories
    )
    max_possible = sum(w[cat.value] for cat in present_categories)
    if max_possible == 0:
        return 0.0
    composite = (weighted_sum / max_possible) * 100

    # Severity floor (Phase 1, see SEVERITY_FLOOR_THRESHOLD's docstring) --
    # a real outlier PLAYER-level category isn't allowed to get diluted to
    # "average" by the other three. Scoped to SEVERITY_FLOOR_CATEGORIES
    # (injury, role_usage_td) only -- see that constant's own docstring for
    # why team-level offense_environment/schedule_weather_venue don't
    # trigger this.
    if any(
        player.category_score(cat) >= SEVERITY_FLOOR_THRESHOLD
        for cat in present_categories & SEVERITY_FLOOR_CATEGORIES
    ):
        composite = max(composite, SEVERITY_FLOOR_VALUE)

    return round(composite, 1)


__all__ = [
    "SEVERITY_FLOOR_THRESHOLD",
    "SEVERITY_FLOOR_VALUE",
    "SEVERITY_FLOOR_CATEGORIES",
    "Position",
    "RiskCategory",
    "ValueSignal",
    "DraftStatus",
    "RiskVariable",
    "Player",
    "RiskProfile",
    "load_weights",
    "build_risk_profile",
    "risk_index",
    "player_from_variable_row",
    "WEIGHTS_PATH",
]
