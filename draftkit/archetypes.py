"""
Player archetypes and roster risk profile for the live draft.

Purpose is roster CONSTRUCTION, not prediction. Validation (Step 4c in the
rebuild plan) showed archetype scores do not beat ADP as model features and
they are deliberately NOT wired into scoring. What they are good at is
telling you what *kind* of player you're taking, and -- more usefully --
what kind of risk your roster is accumulating.

The core idea: fantasy points from touchdowns and explosive plays are far
less stable week to week than points from volume (targets, carries,
receptions). A roster of red-zone tight ends, goal-line backs, and deep
threats can project fine on paper while having a fragile floor, because
every one of those players needs a low-frequency event to hit. Volume
archetypes (bellcows, possession receivers, receiving backs) are what
stabilize a lineup. That trade-off is invisible in a normal ranking list,
which is what this module surfaces.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from draftkit.common import safe_float as _safe_float

# How each archetype earns its fantasy points, which is what determines
# week-to-week variance.
#   event  = needs touchdowns or explosive plays (low-frequency, high-variance)
#   volume = earns points on repeatable usage (high-frequency, stable floor)
#   mixed  = meaningful amounts of both
RISK_PROFILES: Dict[str, str] = {
    # Event-dependent -- boom/bust, fragile floor
    "red_zone_te": "event",
    "goal_line_rb": "event",
    "deep_threat_wr": "event",
    "long_run_rb": "event",
    "early_down_rb": "event",
    # Volume-dependent -- stable floor
    "bellcow_rb": "volume",
    "possession_wr": "volume",
    "receiving_back_rb": "volume",
    "receiving_te": "volume",
    "alpha_wr": "volume",
    # Mixed
    "dual_threat_qb": "mixed",
    "pocket_passer_qb": "mixed",
    "blocking_te": "event",
}

ARCHETYPE_LABELS: Dict[str, str] = {
    "dual_threat_qb": "Dual-Threat QB",
    "pocket_passer_qb": "Pocket Passer",
    "bellcow_rb": "Bellcow RB",
    "receiving_back_rb": "Receiving Back",
    "early_down_rb": "Early-Down Back",
    "long_run_rb": "Explosive Back",
    "goal_line_rb": "Goal-Line Back",
    "deep_threat_wr": "Deep Threat",
    "possession_wr": "Possession WR",
    "alpha_wr": "Alpha WR",
    "red_zone_te": "Red-Zone TE",
    "receiving_te": "Receiving TE",
    "blocking_te": "Blocking TE",
}

# Short plain-language note on what taking this player means for a roster.
ARCHETYPE_NOTES: Dict[str, str] = {
    "dual_threat_qb": "Rushing floor makes weekly output steadier than most QBs.",
    "pocket_passer_qb": "Output tracks passing volume; game script matters.",
    "bellcow_rb": "High-volume workhorse. Stable floor, injury is the main risk.",
    "receiving_back_rb": "Reception volume gives one of the steadiest floors in fantasy.",
    "early_down_rb": "Carries without receiving work. Needs TDs to pay off.",
    "long_run_rb": "Points come in bursts. Big ceiling, thin floor.",
    "goal_line_rb": "TD-dependent. Great weeks and near-zero weeks.",
    "deep_threat_wr": "Low catch rate, big plays. Highest weekly variance at WR.",
    "possession_wr": "Reception volume, high floor, limited ceiling.",
    "alpha_wr": "Dominant target share. Volume-backed, reliable.",
    "red_zone_te": "TD-leveraged. Boom weeks, but quiet without a score.",
    "receiving_te": "Real route role, so points come from targets rather than TDs.",
    "blocking_te": "Blocking role, minimal target share. Not a fantasy asset.",
}

# Share of a roster's classified players that can sit in event-dependent
# archetypes before the lineup's floor is genuinely fragile. Not tuned to
# anything -- a deliberate rule of thumb, and labeled as such in the UI.
EVENT_CONCENTRATION_WARN = 0.50
EVENT_CONCENTRATION_CAUTION = 0.35


def archetype_label(archetype: Any) -> str:
    key = str(archetype or "").strip()
    return ARCHETYPE_LABELS.get(key, "")


def archetype_note(archetype: Any) -> str:
    return ARCHETYPE_NOTES.get(str(archetype or "").strip(), "")


def risk_profile(archetype: Any) -> str:
    return RISK_PROFILES.get(str(archetype or "").strip(), "")


def describe_player(row: Any) -> Dict[str, str]:
    """One player's archetype story, for a card or board row."""
    archetype = row.get("archetype_primary") if hasattr(row, "get") else None
    key = str(archetype or "").strip()
    if not key or key == "unclassified":
        return {
            "archetype": "",
            "label": "",
            "risk_profile": "",
            "note": "No prior-season usage to classify (rookie or no snaps).",
        }
    return {
        "archetype": key,
        "label": archetype_label(key),
        "risk_profile": risk_profile(key),
        "note": archetype_note(key),
    }


def build_roster_risk_report(
    my_team: List[str],
    players_df: pd.DataFrame,
    player_col: str = "player_name",
) -> Dict[str, Any]:
    """
    Summarize the archetype mix of the drafted roster and flag
    over-concentration in event-dependent (TD/explosive) archetypes.

    Only classified players count toward the ratio -- unclassified players
    (rookies, no prior snaps) are reported separately rather than being
    silently treated as safe, which would understate real risk.
    """
    empty = {
        "roster_size": 0,
        "classified_count": 0,
        "unclassified_count": 0,
        "counts_by_archetype": {},
        "counts_by_risk": {"event": 0, "volume": 0, "mixed": 0},
        "event_share": 0.0,
        "status": "none",
        "message": "No players drafted yet.",
        "players": [],
    }
    if not my_team or players_df is None or players_df.empty:
        return empty
    if player_col not in players_df.columns or "archetype_primary" not in players_df.columns:
        return empty

    roster_names = {str(name).strip().lower() for name in my_team}
    roster_df = players_df[
        players_df[player_col].astype(str).str.strip().str.lower().isin(roster_names)
    ]
    if roster_df.empty:
        return empty

    counts_by_archetype: Dict[str, int] = {}
    counts_by_risk = {"event": 0, "volume": 0, "mixed": 0}
    players: List[Dict[str, str]] = []
    unclassified = 0

    for _, row in roster_df.iterrows():
        info = describe_player(row)
        players.append({"player_name": str(row.get(player_col)), **info})
        key = info["archetype"]
        if not key:
            unclassified += 1
            continue
        counts_by_archetype[key] = counts_by_archetype.get(key, 0) + 1
        profile = info["risk_profile"]
        if profile in counts_by_risk:
            counts_by_risk[profile] += 1

    classified = sum(counts_by_risk.values())
    event_share = counts_by_risk["event"] / classified if classified else 0.0

    if classified == 0:
        status, message = "none", "No drafted players have prior-season usage to classify yet."
    elif event_share >= EVENT_CONCENTRATION_WARN:
        status = "warn"
        message = (
            f"{counts_by_risk['event']} of {classified} classified players depend on "
            "touchdowns or explosive plays. That is a high-variance roster: strong ceiling, "
            "fragile week-to-week floor. Consider volume-based players (bellcow backs, "
            "possession receivers, pass-catching backs) with your next picks."
        )
    elif event_share >= EVENT_CONCENTRATION_CAUTION:
        status = "caution"
        message = (
            f"{counts_by_risk['event']} of {classified} classified players are TD- or "
            "big-play dependent. Reasonable so far, but adding more raises weekly volatility."
        )
    else:
        status = "ok"
        message = (
            f"Balanced mix: {counts_by_risk['volume']} volume-based vs "
            f"{counts_by_risk['event']} event-dependent."
        )

    return {
        "roster_size": len(roster_df),
        "classified_count": classified,
        "unclassified_count": unclassified,
        "counts_by_archetype": counts_by_archetype,
        "counts_by_risk": counts_by_risk,
        "event_share": round(float(event_share), 3),
        "status": status,
        "message": message,
        "players": players,
    }
