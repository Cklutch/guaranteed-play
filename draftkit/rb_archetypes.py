"""RB Archetype Taxonomy -- usage-tier primary (rb_archetype_implementation_spec.pdf).

Ports the spec's classify_primary()/classify_down_split()/classify_lean()
close to verbatim. A separate, RB-specific, threshold-based system from the
existing draftkit/archetypes.py (percentile-based, position-agnostic,
explicitly excluded from scoring) -- this runs alongside it as new,
additive output, not a replacement.

Primary is a USAGE TIER only -- Bellcow / Committee / Handcuff / "No Role"
(RBArchetype.UNCONFIRMED) -- never a role/trait like Goal-Line, Explosive,
or Receiving (explicit user correction, 2026-08-15: those were originally
primary-eligible too, a real taxonomy bug -- confirmed in the raw data
that Chuba Hubbard and Rhamondre Stevenson both showed receiving_back as
PRIMARY). Those traits are independent secondary tags any usage tier can
carry -- see LeanArchetype/classify_lean(), unaffected by this migration.

Real field names used here (mapped from the spec's placeholder names,
see draftkit/scripts/build_rb_archetypes.py for how each is built):
  opportunity_share            -> opportunity_share_rate (carries+targets / team plays, rate-per-game)
  games_played                 -> games_recent
  total_touches                -> carries + targets (this pipeline's existing "opportunity" convention,
                                   not carries + receptions -- see build_risk_variables.py's
                                   opportunity_share_rate docstring for why targets, not catches)
  redzone_touches               -> redzone_carries + redzone_targets
  redzone_touch_share           -> prior_redzone_touch_share (real PBP, build_redzone_airyards_features_v1.py)
  inside_10_carry_share         -> prior_inside10_carry_share (real PBP, same script)
  explosive_run_rate            -> prior_explosive_run_rate (real PBP, already existed)
  target_share_of_backfield     -> real, derived from stats_player_reg_by_season (RB-only target share)
  third_down_snap_share         -> prior_third_down_snap_share (real PBP+participation,
                                    build_third_down_participation_features_v1.py)
  route_participation_rate      -> prior_route_participation_rate (real, already existed)
  depth_chart_rank              -> PROXY: rank among same-team RBs by opportunity_share_rate
                                    (no real depth-chart feed exists in this repo -- see
                                    build_rb_archetypes.py for the exact proxy computation)

Three thresholds in classify_lean() were genuinely truncated in the source
PDF (confirmed with two different extraction methods, same cutoff points
both times -- a real gap in the source document, not a parsing artifact).
Per explicit user direction, these are reasoned defaults, not
spec-confirmed values, flagged inline at each one:
  - GOAL_LINE lean: redzone concentration ratio >= 1.5 (his redzone share
    is at least 1.5x his overall share -- a genuine concentration signal,
    not just "he gets some redzone touches").
  - EXPLOSIVE lean: explosive_run_rate >= 0.07 (looser than the primary
    explosive_back archetype's 0.10 -- a lean is a softer secondary tag).
  - RECEIVING lean: route_participation_rate >= 0.50, with the same
    min_games=6 confidence floor the primary receiving_back check uses.
"""

from __future__ import annotations

from enum import Enum


class RBArchetype(Enum):
    """Primary is a USAGE TIER only -- Bellcow / Committee / Handcuff /
    Unconfirmed ("No Role") -- never a role/trait like Goal-Line,
    Explosive, or Receiving. Those are real, independent traits an RB of
    ANY usage tier can carry (a bellcow can be explosive; a committee back
    can lead the room in goal-line work) -- LeanArchetype below is the
    secondary-tag mechanism for exactly that, not a competing primary
    axis. GOAL_LINE_SPECIALIST/EXPLOSIVE_BACK/RECEIVING_BACK used to be
    primary-eligible here too (a real taxonomy bug, explicit user
    correction 2026-08-15: confirmed in the raw data that Chuba Hubbard
    and Rhamondre Stevenson both showed receiving_back as their primary
    archetype, exactly the bug being described) -- removed entirely, not
    just deprioritized, since LeanArchetype's GOAL_LINE/EXPLOSIVE/
    RECEIVING already cover the same real signals as independent
    secondary tags (see classify_lean(), unchanged by this migration)."""
    COMMITTEE_BACK = "committee_back"
    HANDCUFF = "handcuff"
    BELLCOW = "bellcow"
    UNCONFIRMED = "unconfirmed"  # rookie/insufficient-data/no-real-role fallback -- displays as "No Role" for RB (see Home.py's ARCHETYPE_LABELS)


class DownSplit(Enum):
    TWO_DOWN = "2-down"
    THREE_DOWN = "3-down"
    MIXED_DOWN = "mixed-down"
    # "n/a" would silently round-trip through CSV as NaN -- pandas treats
    # it as a default missing-value string on read_csv. Found via a real
    # test failure, not assumed.
    NOT_APPLICABLE = "not_applicable"  # for archetypes that don't get this modifier


class LeanArchetype(Enum):
    GOAL_LINE = "goal_line_lean"
    EXPLOSIVE = "explosive_lean"
    RECEIVING = "receiving_lean"
    NONE = "none"


# Precedence among the real usage-tier primaries. Handcuff is the more
# touch-constrained/decisive read, checked first; Committee second;
# Bellcow is the general catch-all, only reached once neither more
# specific tier matches. Bellcow's 55% opportunity-share floor and
# Committee's 20-55% range are mutually exclusive by construction -- a
# conflict between them should never occur; if it does, that's a data bug
# (stale opportunity_share), not a taxonomy flaw.
PRIMARY_PRECEDENCE_ORDER = [
    RBArchetype.HANDCUFF,
    RBArchetype.COMMITTEE_BACK,
]

# Reasoned defaults for the three PDF-truncated thresholds -- see module
# docstring. Named as constants (not inlined) so they're easy to find and
# revise if the real spec values ever surface.
LEAN_GOAL_LINE_CONCENTRATION_RATIO = 1.5
LEAN_EXPLOSIVE_RUN_RATE = 0.07
LEAN_RECEIVING_ROUTE_PARTICIPATION = 0.50
LEAN_RECEIVING_MIN_GAMES = 6

# Lowered from the spec's literal 0.65 (post-build fix, confirmed on real
# 2025 data): with opportunity_share correctly redefined as backfield
# share (see build_rb_archetypes.py's load_backfield_opportunity_share for
# why that redefinition itself was necessary), a real, confirmed gap
# opened between COMMITTEE_BACK's ceiling (<=0.55) and the spec's own
# 0.65 bellcow floor -- 6 real, well-established lead backs (Gibbs 63.6%,
# Kyren Williams 64.8%, Josh Jacobs 59.9%, Rico Dowdle 59.3%, D'Andre
# Swift 56.6%, Quinshon Judkins 55.8%) fell through as UNCONFIRMED despite
# abundant real data and a clearly lead-back workload. 0.55 makes the two
# ranges directly abut with no gap and no overlap.
BELLCOW_OPPORTUNITY_SHARE_FLOOR = 0.55

# Same real gap, same fix, one boundary down: HANDCUFF's ceiling (<0.20)
# and COMMITTEE_BACK's literal spec floor (0.30) left a real, unclassifiable
# 20-30% opportunity_share band. 11 real RBs with full real samples
# (games_played>=4) landed there as of this build -- Chris Rodriguez,
# Tyjae Spears, Tyler Allgeier, Devin Singletary, Cam Skattebo, Austin
# Ekeler, Samaje Perine, Dylan Sampson, Bhayshul Tuten, Devin Neal, D'Onta
# Foreman. 2 escaped via RECEIVING_BACK's alternate path (Spears, Perine);
# the other 9 sat at UNCONFIRMED despite real, meaningful committee-level
# volume. 0.20 makes COMMITTEE_BACK directly abut HANDCUFF, same "no gap,
# no overlap" principle as the bellcow floor above.
#
# Traced real teammate math before shipping this, not just the floor
# crossing (COMMITTEE_BACK also requires best_teammate>=0.25, gap<=0.20,
# min_games>=4): Tuten (JAX, 0.217) -- teammate Chris Rodriguez 0.290
# clears best_teammate, gap=0.073, min_games=5 -- reclassifies. Skattebo
# (NYG, 0.269) -- teammate Tyrone Tracy 0.4525 clears best_teammate,
# gap=0.184, min_games=8 -- reclassifies. Devin Singletary (NYG, 0.279) --
# same Tracy teammate math -- also reclassifies. Chris Rodriguez himself
# (JAX, 0.290) does NOT reclassify -- his own best teammate is Tuten at
# 0.217, which fails best_teammate>=0.25 -- stays UNCONFIRMED. Lowering
# the floor is evidence-based, not a blanket fix for everyone in the gap.
COMMITTEE_BACK_OPPORTUNITY_SHARE_FLOOR = 0.20


def sample_confidence(actual, floor, games_played=None, min_games=None) -> float:
    conf = min(1.0, actual / floor) if floor else 1.0
    if games_played is not None and min_games is not None:
        conf = min(conf, min(1.0, games_played / min_games) if min_games else 1.0)
    return conf


def _any_floor_cleared(p: dict) -> bool:
    return (
        p.get("total_touches", 0) >= 25
        or p.get("targets", 0) >= 20
        or p.get("opportunity_share", 0) >= 0.30
    )


def _meets_conditions(archetype: RBArchetype, p: dict, teammates: list[dict]) -> bool:
    if archetype == RBArchetype.COMMITTEE_BACK:
        best_teammate = max((t["opportunity_share"] for t in teammates), default=0)
        gap = abs(p["opportunity_share"] - best_teammate)
        min_games = min([p["games_played"]] + [t["games_played"] for t in teammates]) if teammates else p["games_played"]
        return (
            COMMITTEE_BACK_OPPORTUNITY_SHARE_FLOOR <= p["opportunity_share"] <= 0.55
            and best_teammate >= 0.25
            and gap <= 0.20
            and min_games >= 4
        )

    if archetype == RBArchetype.HANDCUFF:
        starter = max((t["opportunity_share"] for t in teammates), default=0)
        return (
            p["opportunity_share"] < 0.20
            and starter >= 0.55
            and p["depth_chart_rank"] == 2
        )

    return False


# ---------------------------------------------------------------------------
# Nearest-fit fallback (claude_code_plan_qb_context_risk_taxonomy.pdf, item 4)
#
# Real bug this replaces: classify_primary() used to fall all the way
# through to UNCONFIRMED whenever a player didn't cleanly clear any single
# archetype's hard thresholds -- even when his real profile was obviously
# close to one of them. Concrete case: Kenneth Walker (KC), real 51%
# opportunity_share, real 17 games -- after his team correction, his
# "teammates" for the COMMITTEE_BACK gap check became KC's real backfield
# instead of his actual 2025 Seattle teammates, so best_teammate/gap no
# longer cleared, and he fell to UNCONFIRMED despite a real, well-
# established committee-level workload that never itself changed.
#
# Distance = MEAN per-condition shortfall, not summed -- archetypes have
# different numbers of defining conditions (Bellcow: 1, Committee/Goal-
# Line/Explosive/Handcuff: 3, Receiving: 2), and summing would bias the
# comparison toward whichever archetype happens to have fewer dimensions.
# Confidence/sample-size sub-checks (conf>=0.6, min_games>=4) are excluded
# from distance entirely -- they answer "do we trust this measurement,"
# not "how close is his profile to this shape." The real "not enough data"
# gate stays exactly where it already is, in classify_primary(), unchanged.
# ---------------------------------------------------------------------------

def _shortfall_at_least(actual: float, floor: float) -> float:
    """Real value must be >= floor. 0 if satisfied, else relative gap."""
    if actual >= floor:
        return 0.0
    return (floor - actual) / floor if floor else abs(actual)


def _shortfall_at_most(actual: float, ceiling: float) -> float:
    """Real value must be <= (or <) ceiling. 0 if satisfied, else relative overage."""
    if actual <= ceiling:
        return 0.0
    return (actual - ceiling) / ceiling if ceiling else abs(actual)


def _shortfall_range(actual: float, low: float, high: float) -> float:
    """Real value must fall within [low, high] -- ONE condition, not two.
    Needs explicit two-sided handling: distance to whichever bound was
    actually violated, zero if inside the band on either side. A one-sided
    template (e.g. always measuring from `low`) would silently misjudge a
    player who overshot `high`."""
    if actual < low:
        return (low - actual) / low if low else abs(actual)
    if actual > high:
        return (actual - high) / high
    return 0.0


def _shortfall_equals(actual: float, target: float) -> float:
    denom = max(abs(target), 1)
    return abs(actual - target) / denom


def _archetype_distance(
    archetype: RBArchetype, p: dict, teammates: list[dict], intrinsic_only: bool = False
) -> float:
    """Mean shortfall across exactly the real conditions listed in the
    module docstring / plan -- vacuous always-true conditions are excluded,
    since including them would silently dilute that archetype's mean
    toward 0 for free. Only the 3 real usage-tier primaries compete here
    (GOAL_LINE_SPECIALIST/EXPLOSIVE_BACK/RECEIVING_BACK were removed from
    RBArchetype entirely -- see that enum's docstring) -- a player whose
    profile leans toward one of those real traits still surfaces it via
    classify_lean(), just never as primary.

    intrinsic_only=True drops COMMITTEE_BACK's and HANDCUFF's teammate-
    relative conditions (best_teammate/gap/starter), leaving only each
    archetype's own-stat condition(s). Real, verified case this exists for:
    Kenneth Walker (KC) -- his own opportunity_share (51%) never changed,
    but a team correction made his "teammates" for these conditions KC's
    real CURRENT backfield, not his actual 2025 Seattle teammate (Zach
    Charbonnet) -- comparing him against players he never shared a season
    with isn't a genuine "his profile doesn't fit Committee" signal, it's
    corrupted comparison data. Explicitly a stand-in for the real fix -- a
    separate stats_season_team field distinguishing "team stats were
    produced for" from "current team" -- not a permanent design; see
    TEAM_CHANGED_PLAYERS in build_rb_archetypes.py for the exact, small,
    explicit list this applies to (same MANUALLY_EXCLUDED_PLAYERS-style
    precedent as Home.py's own manual list, not a heuristic trade-detector)."""
    if archetype == RBArchetype.COMMITTEE_BACK:
        shortfalls = [
            _shortfall_range(p["opportunity_share"], COMMITTEE_BACK_OPPORTUNITY_SHARE_FLOOR, 0.55),
        ]
        if not intrinsic_only:
            best_teammate = max((t["opportunity_share"] for t in teammates), default=0)
            gap = abs(p["opportunity_share"] - best_teammate)
            shortfalls.append(_shortfall_at_least(best_teammate, 0.25))
            shortfalls.append(_shortfall_at_most(gap, 0.20))
    elif archetype == RBArchetype.HANDCUFF:
        shortfalls = [_shortfall_at_most(p["opportunity_share"], 0.20)]
        if not intrinsic_only:
            starter = max((t["opportunity_share"] for t in teammates), default=0)
            shortfalls.append(_shortfall_at_least(starter, 0.55))
            shortfalls.append(_shortfall_equals(p["depth_chart_rank"], 2))
    elif archetype == RBArchetype.BELLCOW:
        shortfalls = [_shortfall_at_least(p["opportunity_share"], BELLCOW_OPPORTUNITY_SHARE_FLOOR)]
    else:
        return float("inf")
    return sum(shortfalls) / len(shortfalls)


# All 3 real usage-tier primaries, precedence order preserved -- used both
# for the clean-qualifier loop (existing behavior) and as the nearest-fit
# tie-break order (min() returns the first-seen minimum, so a distance tie
# resolves toward whichever archetype would have won by precedence anyway).
_ALL_PRIMARY_ARCHETYPES = [*PRIMARY_PRECEDENCE_ORDER, RBArchetype.BELLCOW]


def classify_primary(
    player_data: dict, teammate_data: list[dict], intrinsic_only_distance: bool = False
) -> RBArchetype:
    """intrinsic_only_distance is passed through to the nearest-fit fallback
    only -- see _archetype_distance()'s docstring. It never affects the
    clean-qualifier loop above (_meets_conditions() is unchanged either
    way): a player whose real teammates genuinely support a clean
    committee_back/handcuff match still gets it exactly as before."""
    # Insufficient data check first -- before any archetype logic runs.
    # Unaffected by the nearest-fit change below: a genuine data-sufficiency
    # gap is a different question from "which archetype does he look most
    # like," and forcing a nearest-fit archetype onto a near-zero-sample
    # player would fabricate a confident-looking answer from noise.
    if player_data["games_played"] < 4 and not _any_floor_cleared(player_data):
        return RBArchetype.UNCONFIRMED

    for archetype in PRIMARY_PRECEDENCE_ORDER:
        if _meets_conditions(archetype, player_data, teammate_data):
            return archetype
    if player_data["opportunity_share"] >= BELLCOW_OPPORTUNITY_SHARE_FLOOR:
        return RBArchetype.BELLCOW

    # Nothing cleanly cleared -- nearest-fit instead of UNCONFIRMED.
    distances = {
        a: _archetype_distance(a, player_data, teammate_data, intrinsic_only_distance)
        for a in _ALL_PRIMARY_ARCHETYPES
    }
    return min(distances, key=distances.get)


def classify_down_split(player_data: dict, primary: RBArchetype) -> DownSplit:
    if primary not in (RBArchetype.BELLCOW, RBArchetype.COMMITTEE_BACK):
        return DownSplit.NOT_APPLICABLE
    tds = player_data["third_down_snap_share"]
    if tds >= 0.65:
        return DownSplit.THREE_DOWN
    if tds < 0.40:
        return DownSplit.TWO_DOWN
    return DownSplit.MIXED_DOWN


def classify_lean(player_data: dict, primary: RBArchetype) -> LeanArchetype:
    if primary == RBArchetype.HANDCUFF or player_data["opportunity_share"] < 0.30:
        return LeanArchetype.NONE

    # Reasoned default (PDF-truncated, see module docstring): redzone
    # touches concentrated well above his overall share, not just present.
    concentration = player_data["redzone_touch_share"] / max(player_data["opportunity_share"], 0.05)
    if (
        player_data["redzone_touch_share"] >= 0.30
        and concentration >= LEAN_GOAL_LINE_CONCENTRATION_RATIO
        and sample_confidence(player_data["redzone_touches"], 8) >= 0.6
    ):
        return LeanArchetype.GOAL_LINE

    # Reasoned default (PDF-truncated): looser than the primary
    # explosive_back's 0.10 -- a lean is a softer secondary signal.
    if (
        player_data["yards_per_touch"] >= 5.0
        and player_data["explosive_run_rate"] >= LEAN_EXPLOSIVE_RUN_RATE
        and sample_confidence(player_data["total_touches"], 40) >= 0.6
    ):
        return LeanArchetype.EXPLOSIVE

    # Reasoned default (PDF-truncated): same min_games=6 floor the primary
    # receiving_back check uses.
    if (
        player_data["team_target_share"] >= 0.08
        and player_data["route_participation_rate"] >= LEAN_RECEIVING_ROUTE_PARTICIPATION
        and sample_confidence(
            player_data["targets"], 20, player_data["games_played"], LEAN_RECEIVING_MIN_GAMES
        ) >= 0.6
    ):
        return LeanArchetype.RECEIVING

    return LeanArchetype.NONE


__all__ = [
    "RBArchetype",
    "DownSplit",
    "LeanArchetype",
    "PRIMARY_PRECEDENCE_ORDER",
    "sample_confidence",
    "classify_primary",
    "classify_down_split",
    "classify_lean",
    "_archetype_distance",
    "_ALL_PRIMARY_ARCHETYPES",
]
