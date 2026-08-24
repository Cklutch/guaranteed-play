"""Fantasy draft risk scorecard -- portfolio constraint engine.

Ports spec Sections 4 and 6: prevents stacking correlated risk at the same
position, and tightens risk tolerance as the roster fills with a skewed
risk profile.

TeamRiskBudget/DraftPool here are the CLI's own in-process state container,
NOT draftkit/draft_state.py -- that module hard-requires a running Streamlit
session (st.session_state with no abstraction layer) and cannot run in a
standalone CLI. Field names below intentionally mirror draft_state.py's
schema (drafted players list, my-team list, current pick number) so a
future sync layer between the two is straightforward, without actually
depending on Streamlit here. See the build plan's Context section for the
full reasoning.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from draftkit.risk_scoring import DraftStatus, Player, RiskCategory


@dataclass
class TeamRiskBudget:
    picks: list[Player] = field(default_factory=list)

    def exposure_by_category(self, category: RiskCategory) -> list[float]:
        return [p.category_score(category) for p in self.picks]

    def category_high_risk_count(self, category: RiskCategory, threshold: float = 4.0) -> int:
        return sum(1 for s in self.exposure_by_category(category) if s >= threshold)

    def position_players(self, position) -> list[Player]:
        return [p for p in self.picks if p.position == position]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(".", "").replace("'", "")


@dataclass
class DraftPool:
    all_players: dict[str, Player]
    status: dict[str, DraftStatus] = field(default_factory=dict)
    pick_order: list[str] = field(default_factory=list)

    def __post_init__(self):
        for name in self.all_players:
            self.status.setdefault(name, DraftStatus.AVAILABLE)

    def resolve_name(self, name: str) -> str:
        """Fuzzy-tolerant lookup: exact normalized match, else closest match."""
        key = normalize_name(name)
        if key in self.all_players:
            return key
        matches = difflib.get_close_matches(key, self.all_players.keys(), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        raise ValueError(f"Unknown player: {name}")

    def available_players(self, position=None) -> list[Player]:
        avail = [
            p for name, p in self.all_players.items()
            if self.status[name] == DraftStatus.AVAILABLE
        ]
        return [p for p in avail if p.position == position] if position else avail

    def mark_taken(self, name: str, by_me: bool) -> Player:
        key = self.resolve_name(name)
        if self.status[key] != DraftStatus.AVAILABLE:
            raise ValueError(f"{name} already marked {self.status[key].value}")
        self.status[key] = DraftStatus.MY_TEAM if by_me else DraftStatus.TAKEN_BY_OTHER
        self.pick_order.append(key)
        return self.all_players[key]

    def undo_last_pick(self) -> str | None:
        if not self.pick_order:
            return None
        last = self.pick_order.pop()
        self.status[last] = DraftStatus.AVAILABLE
        return last


def check_correlated_risk(candidate: Player, budget: TeamRiskBudget, threshold: float = 4.0) -> dict:
    """True per-category if the roster already has a same-position player
    also at/above the risk threshold in that category -- stacking risk."""
    flags = {}
    same_pos = budget.position_players(candidate.position)
    for category in RiskCategory:
        if candidate.category_score(category) >= threshold:
            existing_high = [p for p in same_pos if p.category_score(category) >= threshold]
            flags[category.value] = len(existing_high) > 0
        else:
            flags[category.value] = False
    return flags


def dynamic_risk_ceiling(budget: TeamRiskBudget, base_ceiling: float = 4.0) -> dict:
    """Tightens the acceptable risk ceiling for a category once 2+ existing
    picks are already high-risk in it -- the roster gets less tolerant of
    further risk in a category it's already skewed toward."""
    ceilings = {cat.value: base_ceiling for cat in RiskCategory}
    for category in RiskCategory:
        if budget.category_high_risk_count(category, threshold=base_ceiling) >= 2:
            ceilings[category.value] = base_ceiling - 1.0
    return ceilings


def evaluate_pick_fit(candidate: Player, budget: TeamRiskBudget) -> str:
    """Returns "red" / "yellow" / "green" per spec Section 6."""
    ceilings = dynamic_risk_ceiling(budget)
    if any(check_correlated_risk(candidate, budget).values()):
        return "red"
    for category in RiskCategory:
        if candidate.category_score(category) > ceilings[category.value]:
            return "yellow"
    return "green"


def draft_pick(pool: DraftPool, budget: TeamRiskBudget, name: str, by_me: bool) -> Player:
    player = pool.mark_taken(name, by_me)
    if by_me:
        budget.picks.append(player)
    return player


def undo_pick(pool: DraftPool, budget: TeamRiskBudget) -> str | None:
    """Reverses the single most recent pick in BOTH pool and budget.

    pool.undo_last_pick() alone only resets availability status -- if the
    reversed pick was a "mine" pick, its Player object is still sitting in
    budget.picks, which would keep it counted in every category_high_risk /
    correlated-risk calculation after the undo. Mirrors how draft_pick()
    already coordinates both structures together.
    """
    key = pool.undo_last_pick()
    if key is None:
        return None
    # A taken_by_other pick is never in budget.picks, so this is a no-op for
    # that case -- only a "mine" pick's Player object needs removing here.
    for index in range(len(budget.picks) - 1, -1, -1):
        if normalize_name(budget.picks[index].name) == key:
            del budget.picks[index]
            break
    return key


__all__ = [
    "TeamRiskBudget",
    "DraftPool",
    "normalize_name",
    "check_correlated_risk",
    "dynamic_risk_ceiling",
    "evaluate_pick_fit",
    "draft_pick",
    "undo_pick",
]
