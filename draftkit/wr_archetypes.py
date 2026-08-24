"""WR Archetype System v1 -- 2-axis classification (wr_archetype_v1_spec.pdf).

Ports the spec's sample_confidence()/classify_primary()/classify_leans()/
qb_context_label() close to verbatim. Simpler than draftkit/rb_archetypes.py's
6-way system: primary is a target-share usage TIER only (Alpha/Possession/
Complementary, mutually exclusive bands -- no primary-vs-primary conflict by
construction), lean is an independent boom-trait tag (any tier can carry any
lean), and QB-context is a displayed value-context flag, not a classification
input. Separate, additive system -- does not touch draftkit/rb_archetypes.py
or the existing percentile-based draftkit/archetypes.py.

Real field names used here (mapped from the spec's placeholder names, see
draftkit/scripts/build_wr_archetypes.py for how each is built):
  target_share              -> target_share_rate (build_risk_variables.py's
                                load_target_share_rate(), targets / team pass
                                attempts pg -- same real field the RB system
                                and role/usage risk scoring already use)
  games_played               -> games_recent (same source)
  redzone_target_share        -> prior_redzone_target_share (real PBP,
                                 build_redzone_airyards_features_v1.py)
  redzone_targets (raw)        -> prior_redzone_targets (real PBP, same
                                  script -- added this build for rz_conf)
  adot                        -> prior_adot (real PBP, same script)
  yards_per_reception          -> receiving_yards / receptions
                                 (stats_player_reg_by_season, real)
  yac_per_reception            -> receiving_yards_after_catch / receptions
                                 (same source, real)
  rush_attempts / rush_attempts_per_game -> carries / (carries / games)
                                 (same source, real -- WR jet-sweep/gadget
                                 usage, not a RB's rushing role)
  rush_yards_per_attempt       -> rushing_yards / carries (same source, real)
  team_wr1_target_share         -> derived: max target_share_rate among a
                                 team's own real, sample-qualified WRs (see
                                 build_wr_archetypes.py) -- a reasoned reading
                                 of "the team's WR1" as the WR corps, not the
                                 full offensive roster (a bellcow RB shouldn't
                                 force a real WR2 into a false Complementary
                                 read against him)
  qb_tier (1=elite..5=replacement) -> PROXY: no qb_tier field exists anywhere
                                 in this repo (confirmed -- see
                                 build_risk_variables.offense_environment_score()'s
                                 own docstring, which already substitutes real
                                 team epa_rank for this exact missing field).
                                 Reusing that same precedent: epa_rank (1-32,
                                 1=best team offense) bucketed into a 1-5 tier
                                 via ceil(epa_rank/32*5). See
                                 build_wr_archetypes.py for the exact mapping.

Two resolved-by-the-user judgment calls (both genuinely open in the source
PDF, see Section 6):
  - Co-equal edge case (Complementary band, gap from team WR1 < 8pp):
    defaults to Possession tier, not a distinct flag.
  - QB-context modifier: applies to all three usage tiers (Alpha included),
    not just Complementary.

Deep-Threat/YAC lean tie-break (spec Section 2 prose, not in its literal
code sample): if a real case somehow clears both thresholds at once (they're
"near-mutually-exclusive by construction" per an aDOT floor vs. ceiling),
keep whichever clears its own threshold by the larger proportional margin --
a reasoned interpretation of the spec's described (not coded) rule, flagged
inline in classify_leans().
"""

from __future__ import annotations

from enum import Enum


class WRPrimary(Enum):
    ALPHA = "alpha"
    # POSSESSION removed (claude_code_plan_possession_split.pdf) -- a
    # single target-share-only band lumped boom-bust field-stretchers
    # (Jameson Williams: 12.6 aDOT, big-play-reliant) together with true
    # high-floor possession receivers, which was never a real distinction
    # the taxonomy could see. Replaced entirely by BOOM_BUST/HIGH_FLOOR,
    # both real-profile-shape archetypes (aDOT + catch_rate, not just
    # volume). rookie_projection.py's separate pre-draft composite model
    # still emits the raw string "possession" for its own, unrelated tier
    # vocabulary -- never reconciled with this one (same precedent as RB's
    # "committee"/"committee_back" split, see that module's docstring).
    BOOM_BUST = "boom_bust"
    HIGH_FLOOR = "high_floor"
    COMPLEMENTARY = "complementary"
    UNCONFIRMED = "unconfirmed"


class WRLean(Enum):
    DEEP_THREAT = "deep_threat"
    YAC = "yac"
    RED_ZONE = "red_zone"
    RUSHING = "rushing"


class QBContext(Enum):
    ELITE = "elite_qb_context"
    AVERAGE = "average_qb_context"
    BELOW_AVERAGE = "below_average_qb_context"


# Sample floor (all tiers) -- higher than RB's Receiving Back floor (20
# targets) since WRs see materially more volume; a 30-target floor avoids a
# hot 3-week stretch faking an Alpha read.
SAMPLE_FLOOR_TARGETS = 30
SAMPLE_FLOOR_GAMES = 6

# Primary usage-tier bands, target_share. Alpha and Complementary are real,
# untouched thresholds from the original spec.
ALPHA_TARGET_SHARE_FLOOR = 0.24
COMPLEMENTARY_TARGET_SHARE_FLOOR = 0.08
COMPLEMENTARY_TARGET_SHARE_CEILING = 0.16  # was POSSESSION_TARGET_SHARE_FLOOR before the split
COMPLEMENTARY_WR1_GAP_FLOOR = 0.08

# Boom-Bust / High-Floor profiles (claude_code_plan_possession_split.pdf) --
# replace Possession entirely. Season-level inputs (aDOT, target_share,
# catch_rate) used as the primary classifier, not weekly variance -- more
# stable and available even for players without a full season logged (per
# the plan's own explicit rationale). Real gap between the two profiles
# (e.g. 11-13 aDOT / 18-22% target share) is deliberately left uncovered by
# either band -- nearest-fit (not a third hard bucket) resolves it, same
# mean-shortfall mechanism as the rest of this module.
BOOM_BUST_ADOT_FLOOR = 15.0
BOOM_BUST_TARGET_SHARE_CEILING = 0.15
BOOM_BUST_CATCH_RATE_CEILING = 0.50

HIGH_FLOOR_ADOT_LOW = 6.0
HIGH_FLOOR_ADOT_HIGH = 9.0
HIGH_FLOOR_TARGET_SHARE_FLOOR = 0.25

# Lean thresholds -- exactly as given in the spec.
LEAN_DEEP_THREAT_ADOT = 12.5
LEAN_DEEP_THREAT_YPR = 14.0
LEAN_DEEP_THREAT_MIN_RECEPTIONS = 15

LEAN_YAC_MIN = 6.0
LEAN_YAC_ADOT_CEILING = 9.0
LEAN_YAC_MIN_RECEPTIONS = 15

LEAN_REDZONE_TARGET_SHARE = 0.16
# Was 0.30 -- real-pool check found that floor fired for exactly 1 of 278
# real WRs (Amon-Ra St. Brown), never functioning as an actual signal.
# Lowered after a real coherence check (Tee Higgins/Rashod Bateman case,
# 2026-08-16): 0.16 sits in the middle of a stable plateau (9 real players
# at both 0.15 and 0.16, no new entries down to 0.14) -- not tuned to just
# catch the two named cases. Concentration ratio (below) still does the
# real discriminating work: a WR needs real redzone-CONCENTRATED usage
# relative to his own overall share, not just generic volume.
LEAN_REDZONE_CONCENTRATION_RATIO = 1.3
LEAN_REDZONE_MIN_TARGETS = 6
LEAN_REDZONE_MIN_TEAM_TARGETS = 20  # spec: rz_conf only computed "if targets >= 20 else 0"

LEAN_RUSHING_ATTEMPTS_PER_GAME = 0.8
LEAN_RUSHING_YARDS_PER_ATTEMPT = 6.0
LEAN_RUSHING_MIN_ATTEMPTS = 8

LEAN_CONFIDENCE_FLOOR = 0.6


def sample_confidence(actual, floor, games_played=None, min_games=None) -> float:
    conf = min(1.0, actual / floor) if floor else 1.0
    if games_played is not None and min_games is not None:
        conf = min(conf, min(1.0, games_played / min_games) if min_games else 1.0)
    return conf


# ---------------------------------------------------------------------------
# Nearest-fit (claude_code_plan_qb_context_risk_taxonomy.pdf item 4, extended
# by claude_code_plan_possession_split.pdf to replace Possession).
#
# Alpha keeps its own real, untouched hard floor (target_share>=0.24),
# checked first -- the Possession-split plan only replaces Possession,
# never Alpha. Everyone below that floor now resolves via mean-shortfall
# distance across the 3 remaining real archetypes (Boom-Bust, High-Floor,
# Complementary) uniformly -- not just the old "below 8%" dead zone.
# Necessary once Possession's old band [0.16, 0.24) has nowhere else to go
# (per the split plan: "every player currently tagged Possession needs to
# be re-run through the new two-profile nearest-fit logic -- this is a
# full re-classification... not an additive change"), and it also
# subsumes the OLD "co-equal defaults to Possession" special case for
# free: a co-equal receiver's real Complementary distance already reflects
# its own failed gap condition (mean of a 0-shortfall band match + a
# nonzero gap shortfall), so no separate hardcoded override is needed
# anymore -- he lands wherever his real aDOT/target_share profile actually
# fits closest among the 3 survivors, which is a richer, more accurate
# read than a fixed fallback bucket ever was.
#
# Distance = MEAN per-condition shortfall, not summed -- same fairness
# rationale as rb_archetypes.py (archetypes have different condition
# counts: Boom-Bust 3, High-Floor 2, Complementary 2). Distance functions
# are exported so build_wr_archetypes.py can report a player's full
# distance vector for auditability.
# ---------------------------------------------------------------------------

def _shortfall_at_least(actual: float, floor: float) -> float:
    if actual >= floor:
        return 0.0
    return (floor - actual) / floor if floor else abs(actual)


def _shortfall_at_most(actual: float, ceiling: float) -> float:
    if actual <= ceiling:
        return 0.0
    return (actual - ceiling) / ceiling if ceiling else abs(actual)


def _shortfall_range(actual: float, low: float, high: float) -> float:
    """[low, high] real band -- two-sided: distance to whichever bound was
    actually violated, zero if inside on either side."""
    if actual < low:
        return (low - actual) / low if low else abs(actual)
    if actual > high:
        return (actual - high) / high
    return 0.0


def boom_bust_distance(player: dict) -> float:
    catch_rate = player["receptions"] / player["targets"] if player["targets"] else 0.0
    shortfalls = [
        _shortfall_at_least(player["adot"], BOOM_BUST_ADOT_FLOOR),
        _shortfall_at_most(player["target_share"], BOOM_BUST_TARGET_SHARE_CEILING),
        _shortfall_at_most(catch_rate, BOOM_BUST_CATCH_RATE_CEILING),
    ]
    return sum(shortfalls) / len(shortfalls)


def high_floor_distance(player: dict) -> float:
    shortfalls = [
        _shortfall_range(player["adot"], HIGH_FLOOR_ADOT_LOW, HIGH_FLOOR_ADOT_HIGH),
        _shortfall_at_least(player["target_share"], HIGH_FLOOR_TARGET_SHARE_FLOOR),
    ]
    return sum(shortfalls) / len(shortfalls)


def complementary_distance(player: dict, team_wr1_target_share: float) -> float:
    ts = player["target_share"]
    gap = team_wr1_target_share - ts
    shortfalls = [
        _shortfall_range(ts, COMPLEMENTARY_TARGET_SHARE_FLOOR, COMPLEMENTARY_TARGET_SHARE_CEILING),
        _shortfall_at_least(gap, COMPLEMENTARY_WR1_GAP_FLOOR),
    ]
    return sum(shortfalls) / len(shortfalls)


def classify_primary(player: dict, team_wr1_target_share: float) -> WRPrimary:
    if player["targets"] < SAMPLE_FLOOR_TARGETS or player["games_played"] < SAMPLE_FLOOR_GAMES:
        return WRPrimary.UNCONFIRMED

    ts = player["target_share"]
    if ts >= ALPHA_TARGET_SHARE_FLOOR:
        return WRPrimary.ALPHA

    # Complementary's own real band -- untouched, still a clean hard match
    # (Alpha and Complementary are the only two tiers the split plan
    # doesn't touch). The old "co-equal" case (target_share in this band
    # but gap doesn't clear the floor) no longer defaults to Possession --
    # Possession is gone -- it falls through to the Boom-Bust/High-Floor
    # comparison below instead, the same population that used to land
    # Possession under the old system.
    if COMPLEMENTARY_TARGET_SHARE_FLOOR <= ts < COMPLEMENTARY_TARGET_SHARE_CEILING:
        gap = team_wr1_target_share - ts
        if gap >= COMPLEMENTARY_WR1_GAP_FLOOR:
            return WRPrimary.COMPLEMENTARY
        # else fall through to the 2-way comparison below

    # Everyone else below Alpha (the old Possession band, the co-equal
    # case above, and the old sub-8% dead zone) -- nearest-fit among the
    # real archetype shapes, exactly as claude_code_plan_possession_split.pdf
    # specifies verbatim for the Boom-Bust/High-Floor pair: "Every player
    # gets a computed distance to the Boom-Bust profile and a computed
    # distance to the High-Floor profile, and whichever is closer wins."
    #
    # Complementary is SHAPE-GATED, not blanket-excluded (task_2c2eea99,
    # 2026-08-17 redesign of the module's first-draft blanket exclusion).
    # The original exclusion was written to fix one real case -- Jameson
    # Williams (17.5% target_share, real DET WR1 gap ~12pp, real aDOT 12.6):
    # his Complementary distance (0.048, one condition barely misses) came
    # out LOWER than his Boom-Bust distance (0.200), stealing the win even
    # though his real aDOT/catch-rate profile is exactly the boom-bust
    # shape this split exists to catch. But a later real case (Cooper Kupp,
    # 2025: target_share 0.1635, missing COMPLEMENTARY_TARGET_SHARE_CEILING
    # by just 0.35pp, real aDOT 7.49) showed the blanket exclusion also
    # blocks genuine near-misses: Kupp's real Complementary distance
    # (0.0108) is ~16x closer than his High-Floor distance (0.173), and his
    # aDOT sits inside HIGH_FLOOR's own [6,9] band -- nothing about his
    # shape looks boom-bust. Blanket-excluding Complementary treated both
    # cases the same when they aren't: Williams' low distance is spurious
    # (his real shape contradicts it), Kupp's is real (his shape agrees
    # with it).
    #
    # Resolution: use aDOT -- the same signal that made Williams' case
    # wrong -- to gate entry instead of excluding Complementary outright.
    # HIGH_FLOOR_ADOT_HIGH (9.0, already an existing archetype boundary,
    # not a new anchor-fit constant) marks the edge of "does not look
    # boom-bust." A player whose real aDOT clears that bar has no
    # shape-based reason to keep Complementary out of the race; a player
    # whose aDOT sits at or above it (Williams: 12.6, squarely in the
    # deliberately-uncovered dead zone or beyond) keeps the 2-way
    # comparison exactly as before. Checked against the full real,
    # sample-qualified fallback population (56 players, 2025 data): this
    # produces exactly 5 reclassifications, all real near-misses where the
    # old 2-way choice was between two poor fits and Complementary is a
    # materially closer one (Kupp, Dortch, Turpin, Raymond, Watson) --
    # Williams and every other genuinely boom-bust-shaped player are
    # unaffected.
    bb_dist = boom_bust_distance(player)
    hf_dist = high_floor_distance(player)
    if player["adot"] <= HIGH_FLOOR_ADOT_HIGH:
        comp_dist = complementary_distance(player, team_wr1_target_share)
        best = min(
            (WRPrimary.BOOM_BUST, bb_dist),
            (WRPrimary.HIGH_FLOOR, hf_dist),
            (WRPrimary.COMPLEMENTARY, comp_dist),
            key=lambda pair: pair[1],
        )
        return best[0]
    if bb_dist <= hf_dist:
        return WRPrimary.BOOM_BUST
    return WRPrimary.HIGH_FLOOR


def classify_leans(player: dict) -> list[WRLean]:
    leans: list[WRLean] = []

    deep_threat_conf = sample_confidence(player["receptions"], LEAN_DEEP_THREAT_MIN_RECEPTIONS)
    yac_conf = sample_confidence(player["receptions"], LEAN_YAC_MIN_RECEPTIONS)
    rz_conf = (
        sample_confidence(player["redzone_targets"], LEAN_REDZONE_MIN_TARGETS, player["games_played"])
        if player["targets"] >= LEAN_REDZONE_MIN_TEAM_TARGETS else 0.0
    )
    rushing_conf = sample_confidence(player["rush_attempts"], LEAN_RUSHING_MIN_ATTEMPTS)

    deep_threat_hit = (
        player["adot"] >= LEAN_DEEP_THREAT_ADOT
        and player["yards_per_reception"] >= LEAN_DEEP_THREAT_YPR
        and deep_threat_conf >= LEAN_CONFIDENCE_FLOOR
    )
    yac_hit = (
        player["yac_per_reception"] >= LEAN_YAC_MIN
        and player["adot"] <= LEAN_YAC_ADOT_CEILING
        and yac_conf >= LEAN_CONFIDENCE_FLOOR
    )
    if deep_threat_hit and yac_hit:
        # Real data anomaly (shouldn't happen given the aDOT floor/ceiling
        # split, but not structurally impossible) -- keep whichever clears
        # its own threshold by the larger proportional margin. See module
        # docstring: reasoned interpretation of spec prose, not spec code.
        deep_threat_margin = (player["adot"] - LEAN_DEEP_THREAT_ADOT) / LEAN_DEEP_THREAT_ADOT
        yac_margin = (player["yac_per_reception"] - LEAN_YAC_MIN) / LEAN_YAC_MIN
        leans.append(WRLean.DEEP_THREAT if deep_threat_margin >= yac_margin else WRLean.YAC)
    elif deep_threat_hit:
        leans.append(WRLean.DEEP_THREAT)
    elif yac_hit:
        leans.append(WRLean.YAC)

    if (
        player["redzone_target_share"] >= LEAN_REDZONE_TARGET_SHARE
        and player["redzone_target_share"] / max(player["target_share"], 0.01) >= LEAN_REDZONE_CONCENTRATION_RATIO
        and rz_conf >= LEAN_CONFIDENCE_FLOOR
    ):
        leans.append(WRLean.RED_ZONE)

    # Independent axis -- can coexist with either Deep-Threat or YAC, no
    # single-lean cap (per spec Section 2: "there's no real contradiction
    # between them").
    if (
        player["rush_attempts_per_game"] >= LEAN_RUSHING_ATTEMPTS_PER_GAME
        and player["rush_yards_per_attempt"] >= LEAN_RUSHING_YARDS_PER_ATTEMPT
        and rushing_conf >= LEAN_CONFIDENCE_FLOOR
    ):
        leans.append(WRLean.RUSHING)

    return leans


def qb_context_label(offense_environment_data: dict) -> QBContext:
    qb_tier = offense_environment_data["qb_tier"]  # 1 (elite) to 5 (replacement level) -- proxy, see module docstring
    if qb_tier <= 2:
        return QBContext.ELITE
    if qb_tier == 3:
        return QBContext.AVERAGE
    return QBContext.BELOW_AVERAGE


__all__ = [
    "WRPrimary",
    "WRLean",
    "QBContext",
    "sample_confidence",
    "classify_primary",
    "classify_leans",
    "qb_context_label",
    "boom_bust_distance",
    "high_floor_distance",
    "complementary_distance",
]
