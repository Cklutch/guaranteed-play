"""QB Archetype System v1 (claude_code_plan_qb_archetypes.pdf).

Single axis: rushing fantasy points as a share of total fantasy points
(rushing_fantasy_pct), banded into 3 tiers -- the standard convention used
across current fantasy analysis, not an invented taxonomy. Simple threshold
banding is sufficient here (confirmed before building, per the PDF's own
explicit ask to check first): unlike RB/WR, this is a single continuous
scalar with no multi-condition ambiguity, so the RB/WR nearest-fit
mean-distance mechanism isn't needed.

rushing_fantasy_pct's real input (see build_qb_archetypes.py) is pooled
across the most recent 2 real seasons with data, not read from a single
season -- a real, explicit methodology decision (not a threshold tuned to
fit expectations): QB rushing volume is real-world strategically volatile
in a way RB/WR usage isn't (contract-year caution, scheme-driven
single-season dips). Validated directly against real anchor cases before
locking in: the pool resolves Jalen Hurts into Dual-Threat (28.2% single-
season -> 36.2% pooled) but does NOT resolve Lamar Jackson (20.1% -> 23.9%,
still Balanced) -- confirming this is a real, uneven methodology effect,
not a knob turned to force a specific answer.

No leans built speculatively, per the PDF's explicit instruction -- add
only if a real, clear case emerges (e.g. a redzone-rushing-specific lean).

Rookie QB projection is explicitly out of scope (its own future plan) --
this module only classifies QBs with real, logged NFL rushing/passing
splits, however limited that sample is.
"""

from __future__ import annotations

from enum import Enum


class QBArchetype(Enum):
    POCKET_PASSER = "pocket_passer"
    BALANCED = "balanced"
    DUAL_THREAT = "dual_threat"
    UNCONFIRMED = "unconfirmed"


# Sample-size gate, checked against POOLED games/attempts (after the
# 2-season resolution in build_qb_archetypes.py), not a single season --
# validated against the real validation population before locking in: gates
# Max Brosmer (71 attempts, only 1 real season) and Quinn Ewers (83
# attempts); passes every "partial 2025 sample" case (Sanders 212/8,
# Gabriel 185/10, Shough 327/11, McCarthy 243/10) with room to spare; lets
# Anthony Richardson's real pooled sample (266 attempts/13 games) through
# even though his single 2025 season alone (2 games) would fail -- the
# PDF's own named "decide deliberately" edge case, resolved by pooling
# rather than a hardcoded special case. Mirrors WR's targets<30-or-games<6
# shape (SAMPLE_FLOOR_ATTEMPTS is QB's volume-stat analogue to WR's
# targets -- no "starts" field exists anywhere in the real stats source
# this is built from, confirmed during investigation).
SAMPLE_FLOOR_ATTEMPTS = 100
SAMPLE_FLOOR_GAMES = 4

# Real, externally-validated convention (not invented for this tool) --
# see plan PDF.
POCKET_PASSER_CEILING = 0.10
DUAL_THREAT_FLOOR = 0.30


def classify_primary(rushing_fantasy_pct: float, attempts: float, games: float) -> QBArchetype:
    if attempts < SAMPLE_FLOOR_ATTEMPTS or games < SAMPLE_FLOOR_GAMES:
        return QBArchetype.UNCONFIRMED
    if rushing_fantasy_pct >= DUAL_THREAT_FLOOR:
        return QBArchetype.DUAL_THREAT
    if rushing_fantasy_pct < POCKET_PASSER_CEILING:
        return QBArchetype.POCKET_PASSER
    return QBArchetype.BALANCED


__all__ = [
    "QBArchetype",
    "SAMPLE_FLOOR_ATTEMPTS",
    "SAMPLE_FLOOR_GAMES",
    "POCKET_PASSER_CEILING",
    "DUAL_THREAT_FLOOR",
    "classify_primary",
]
