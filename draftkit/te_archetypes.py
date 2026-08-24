"""TE Archetype System v1 (plan_te_archetypes.pdf).

Single-axis exhaustive banding on target_share -- same shape as
draftkit/qb_archetypes.py, not the RB/WR nearest-fit mechanism. This was a
real, corrected design decision, not the plan's original assumption: a
first pass at this plan used a 2D nearest-fit (target_share + snap_share as
two independent discriminating axes), justified by a real-looking anchor
case (Tyler Conklin: apparently Kittle-level snap share paired with
near-zero targets). That anchor turned out to be built on a real
methodology bug -- snap_share_player_seasons.csv's `prior_snap_share`
column is a LAGGED feature (the row labeled season=2025 actually holds the
season-2024 value, same lagged-feature convention this repo already uses
elsewhere for prior_adot/prior_redzone_target_share). Corrected (season
2026's row's prior_snap_share = the real 2025 value), Conklin turned out to
be a genuinely thin-sample, low-role player on BOTH axes (14.8% snap, 6
games) -- not a real "high snap, blocking role" case at all. Worse for the
2D design: real committed blockers (Adam Trautman 57.1% snap, Drew Sample
53.6% snap) showed snap shares comparable to or higher than real receiving
anchors (Kittle 49.4%, Bowers 60.9%) once corrected -- snap_share does NOT
cleanly separate blocking from receiving. target_share does (receiving
cluster 18.5%-26.0%, blocking-candidate cluster 3.5%-6.8%, no real overlap,
validated directly against the full 134-player real pool). snap_share's
real role here is a sample-CONFIDENCE gate (is this role real, or just a
thin sample -- same job WR's targets/games gate does), not a second
classification axis.

No leans built speculatively beyond the one named in the plan (red_zone) --
same discipline as every other archetype module this session.
"""

from __future__ import annotations

from enum import Enum


class TEArchetype(Enum):
    RECEIVING_TE = "receiving_te"
    BALANCED = "balanced"
    BLOCKING_TE = "blocking_te"
    UNCONFIRMED = "unconfirmed"


class TELean(Enum):
    RED_ZONE = "red_zone"


class TERoleProfile(Enum):
    ELITE = "elite"
    COMPLEMENTARY = "complementary"


# Real confidence gate -- validated directly against the corrected real
# pool: SAMPLE_FLOOR_SNAP_SHARE=0.30 correctly excludes Conklin (14.8%) and
# Foster Moreau (22.5%, both real thin/marginal-role cases) from being
# confidently classified either way, while correctly including every real
# committed blocker (Trautman 57.1%, Sample 53.6%, Bates 45.4%) and every
# real receiving anchor (all well above 45%).
SAMPLE_FLOOR_GAMES = 4
SAMPLE_FLOOR_SNAP_SHARE = 0.30

# target_share bands -- validated against the real, corrected pool before
# locking in: RECEIVING_TARGET_SHARE_FLOOR=0.15 produces a real 23-player
# population (Bowers/Kittle/Kelce/McBride/Goedert/Schultz/Njoku/Andrews/
# Ferguson/Pitts/LaPorta/Hockenson/Ertz/Henry/etc. -- all real, known
# pass-catching TEs, no surprises). BLOCKING_TARGET_SHARE_CEILING=0.07 sits
# cleanly below that with real committed blockers (Trautman 4.9%, Sample
# 3.5%, Bates 5.8%) and real air between the two bands (no anchor lands in
# 0.07-0.15).
RECEIVING_TARGET_SHARE_FLOOR = 0.15
BLOCKING_TARGET_SHARE_CEILING = 0.07

# Lean threshold -- reuses wr_archetypes.py's exact red_zone mechanism
# shape (floor + concentration ratio). Started at WR's literal 0.30 value,
# then validated against the real TE pool the same way WR's own threshold
# was recalibrated earlier tonight: 0.30 fired for exactly 1 of 164 real
# TEs (Hunter Henry), never functioning as a real signal. 0.15 sits in the
# middle of a stable plateau (11 real players at both 0.14 and 0.15, no
# new entries down to 0.12) -- not tuned to catch a specific name.
LEAN_REDZONE_TARGET_SHARE = 0.15
LEAN_REDZONE_CONCENTRATION_RATIO = 1.3
LEAN_REDZONE_MIN_TARGETS = 6
LEAN_REDZONE_MIN_TEAM_TARGETS = 20
LEAN_CONFIDENCE_FLOOR = 0.6


# Second-pass lean, computed only within receiving_te (plan_te_role_profile_
# elite.pdf). Reasoned defaults, NOT validated against an outcome backtest --
# same disclosure as QB rookie composite weights. Validated against the real,
# corrected 22-player receiving_te pool (not just named anchors, and not the
# original 23-player pool -- 1 player, David Njoku, moved out of receiving_te
# entirely once te_archetypes.csv's own target_share bug was fixed -- see
# build_te_archetypes.py's module docstring): produces a real 11-elite/
# 11-complementary split. ELITE_TARGET_SHARE_FLOOR=0.20 captures real
# elite-volume cases (McBride 27.4%, Bowers 24.7%, Pitts 22.7%, Ertz 21.8%,
# Fannin 21.8%, Warren 21.1%, Kraft 20.3%, Goedert 20.0%) sitting above
# Kelce/Kittle's ~19-20% cluster (who stay complementary on volume alone).
# ELITE_REDZONE_TARGET_SHARE_FLOOR=0.22 captures Henry (33.3%), Andrews
# (26.8%), Ferguson (24.0%), Goedert (24.6%) -- real redzone-concentration
# cases, including two (Andrews, Henry) who do NOT clear the volume floor at
# all, proving the redzone path is load-bearing on its own, not redundant
# with the volume path.
ELITE_TARGET_SHARE_FLOOR = 0.20
ELITE_REDZONE_TARGET_SHARE_FLOOR = 0.22


def classify_role_profile(target_share: float, redzone_target_share: float) -> TERoleProfile:
    """Only meaningful for players already tagged receiving_te by
    classify_primary() -- no UNCONFIRMED state here, since classify_primary()
    already gates on sample size before a player can reach this function."""
    if target_share >= ELITE_TARGET_SHARE_FLOOR or redzone_target_share >= ELITE_REDZONE_TARGET_SHARE_FLOOR:
        return TERoleProfile.ELITE
    return TERoleProfile.COMPLEMENTARY


def sample_confidence(actual, floor, games_played=None, min_games=None) -> float:
    conf = min(1.0, actual / floor) if floor else 1.0
    if games_played is not None and min_games is not None:
        conf = min(conf, min(1.0, games_played / min_games) if min_games else 1.0)
    return conf


def classify_primary(target_share: float, games_recent: float, snap_share: float) -> TEArchetype:
    if games_recent < SAMPLE_FLOOR_GAMES or snap_share < SAMPLE_FLOOR_SNAP_SHARE:
        return TEArchetype.UNCONFIRMED
    if target_share >= RECEIVING_TARGET_SHARE_FLOOR:
        return TEArchetype.RECEIVING_TE
    if target_share <= BLOCKING_TARGET_SHARE_CEILING:
        return TEArchetype.BLOCKING_TE
    return TEArchetype.BALANCED


def classify_leans(player: dict) -> list[TELean]:
    leans: list[TELean] = []
    redzone_target_share = player.get("redzone_target_share", 0.0) or 0.0
    redzone_targets = player.get("redzone_targets", 0.0) or 0.0
    target_share = player.get("target_share", 0.0) or 0.0
    targets = player.get("targets", 0.0) or 0.0
    games_played = player.get("games_played", 0.0) or 0.0

    rz_conf = (
        sample_confidence(redzone_targets, LEAN_REDZONE_MIN_TARGETS, games_played)
        if targets >= LEAN_REDZONE_MIN_TEAM_TARGETS else 0.0
    )
    if (
        redzone_target_share >= LEAN_REDZONE_TARGET_SHARE
        and redzone_target_share / max(target_share, 0.01) >= LEAN_REDZONE_CONCENTRATION_RATIO
        and rz_conf >= LEAN_CONFIDENCE_FLOOR
    ):
        leans.append(TELean.RED_ZONE)

    return leans


__all__ = [
    "TEArchetype",
    "TELean",
    "TERoleProfile",
    "SAMPLE_FLOOR_GAMES",
    "SAMPLE_FLOOR_SNAP_SHARE",
    "RECEIVING_TARGET_SHARE_FLOOR",
    "BLOCKING_TARGET_SHARE_CEILING",
    "ELITE_TARGET_SHARE_FLOOR",
    "ELITE_REDZONE_TARGET_SHARE_FLOOR",
    "sample_confidence",
    "classify_primary",
    "classify_leans",
    "classify_role_profile",
]
