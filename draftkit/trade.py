"""Trade calculator: value both sides of a deal in in-season Base Value.

Preseason draft value is the wrong currency for a Week 3 trade -- a fast
riser or a cold start has already moved off it. So this values every player
by the position-adjusted in-season score from draftkit.in_season
(VOR + role + risk, blended actuals-and-preseason), the same number the live
Week 3 board ranks on. A scarce RB1 is therefore worth more than a WR with
equal projected points, which is what makes a WR-for-RB verdict honest.

Two halves:
  * fetch_league_teams(league_id) -- pull real rosters + owner names from
    Sleeper's public API, so the calculator can offer each team's actual
    players instead of a blank search.
  * evaluate_trade(side_a, side_b, value_index) -- sum each side's value,
    report the gap and a plain-language verdict.

This is the calculator (you propose a deal, it judges it), not the suggestor
(scan every roster for deals) -- that's the deeper follow-on.
"""

import pandas as pd
import requests

from draftkit.in_season import _pid_key

SLEEPER_BASE = "https://api.sleeper.app/v1"
_TIMEOUT = 15
_HEADERS = {"User-Agent": "guaranteed-play-draftkit"}

# A trade within this fraction of the larger side reads as fair; the second
# band is a lean; beyond it, a clear winner.
FAIR_BAND = 0.10
LEAN_BAND = 0.25


def _get(url):
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


def parse_league_id(text):
    """Pull a numeric league id from a raw id or a Sleeper league URL."""
    import re

    if not text:
        return ""
    t = str(text).strip()
    m = re.search(r"leagues?/(\d+)", t)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6,})", t)
    return m.group(1) if m else t


def fetch_league_teams(league_id):
    """Real teams in a league: owner display name + their Sleeper player ids.

    Returns [] on any error (bad id, private league, API down) so the caller
    can fall back to manual player search rather than crash.
    """
    league_id = parse_league_id(league_id)
    if not league_id:
        return []
    try:
        rosters = _get(f"{SLEEPER_BASE}/league/{league_id}/rosters") or []
        users = _get(f"{SLEEPER_BASE}/league/{league_id}/users") or []
    except Exception:
        return []

    name_by_user = {}
    for u in users:
        if not isinstance(u, dict):
            continue
        meta = u.get("metadata") or {}
        name_by_user[u.get("user_id")] = (
            meta.get("team_name") or u.get("display_name") or "Unknown"
        )

    teams = []
    for r in rosters:
        if not isinstance(r, dict):
            continue
        rid = r.get("roster_id")
        teams.append(
            {
                "roster_id": rid,
                "owner": name_by_user.get(r.get("owner_id"), f"Team {rid}"),
                "player_ids": [str(p) for p in (r.get("players") or [])],
            }
        )
    return teams


def build_value_index(in_season_board):
    """{sleeper_player_id: {...}} keyed for trade lookup.

    Values come from build_in_season_board's output, so every score is the
    position-adjusted, blended Week-N number -- not preseason.
    """
    index = {}
    if in_season_board is None or in_season_board.empty:
        return index
    cols = in_season_board.columns
    for _, row in in_season_board.iterrows():
        pid = _pid_key(row.get("player_id")) if "player_id" in cols else None
        if not pid:
            continue
        index[pid] = {
            "player_id": pid,
            "name": row.get("player_name", ""),
            "position": row.get("position", ""),
            "team": row.get("team", ""),
            "score": float(pd.to_numeric(row.get("base_value_score"), errors="coerce") or 0.0),
            "ros_points": float(pd.to_numeric(row.get("ros_points"), errors="coerce") or 0.0),
            "rank": int(pd.to_numeric(row.get("in_season_rank"), errors="coerce") or 0),
        }
    return index


def _side(player_ids, value_index):
    """Resolve a list of player ids to their value rows + running total."""
    players, total = [], 0.0
    for pid in player_ids:
        key = _pid_key(pid) if not isinstance(pid, str) else (pid[:-2] if pid.endswith(".0") else pid)
        rec = value_index.get(key)
        if rec is None:
            players.append({"player_id": str(pid), "name": f"(unknown {pid})",
                            "position": "", "team": "", "score": 0.0,
                            "ros_points": 0.0, "rank": 0})
        else:
            players.append(rec)
            total += rec["score"]
    return players, round(total, 2)


def evaluate_trade(side_a_ids, side_b_ids, value_index,
                   label_a="You give", label_b="You get"):
    """Judge a proposed trade by summed position-adjusted value.

    Returns a dict with per-side player breakdowns, both totals, the gap
    (from side A's perspective: positive = A comes out ahead), a fairness
    verdict, and a flag when the sides swap different numbers of players
    (consolidating bodies into fewer, better players carries roster-spot
    value the raw sum can't see -- surfaced, not silently scored).
    """
    a_players, a_total = _side(side_a_ids, value_index)
    b_players, b_total = _side(side_b_ids, value_index)

    larger = max(a_total, b_total, 1.0)
    gap = round(b_total - a_total, 2)  # >0 means side B is worth more -> A wins
    pct = abs(gap) / larger

    if pct <= FAIR_BAND:
        verdict = "Fair trade"
        winner = None
    elif pct <= LEAN_BAND:
        winner = label_b if gap > 0 else label_a
        verdict = f"Slightly favors: {winner}"
    else:
        winner = label_b if gap > 0 else label_a
        verdict = f"Favors: {winner}"

    return {
        "label_a": label_a,
        "label_b": label_b,
        "a_players": a_players,
        "b_players": b_players,
        "a_total": a_total,
        "b_total": b_total,
        "gap": gap,
        "gap_pct": round(pct * 100, 1),
        "verdict": verdict,
        "winner": winner,
        "uneven_count": len(a_players) != len(b_players),
    }
