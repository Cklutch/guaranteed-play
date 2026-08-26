"""Live Sleeper draft ingestion for the draft engine.

Reads a Sleeper draft (real or mock) through Sleeper's public, read-only
v1 API -- no auth, no account -- and maps its picks onto our candidate pool
so draftkit.live_draft.recommend_picks can react to a draft as it happens.

Endpoints used (https://api.sleeper.app/v1):
  * /draft/<draft_id>          -- draft metadata (settings, slots)
  * /draft/<draft_id>/picks    -- every pick so far, each with a Sleeper
                                  player_id AND a metadata block carrying
                                  first/last name, position, team.

Matching is by Sleeper player_id first (our pool carries the same id from
master_players.csv), falling back to a normalized-name match, so picks line
up with the pool's own name spelling -- which is what recommend_picks
filters on. Names that match nothing in the pool (kickers, defenses, deep
players) simply don't exclude anything, which is correct.
"""

import re

import pandas as pd
import requests

from draftkit.dads_scoring import _norm

SLEEPER_BASE = "https://api.sleeper.app/v1"
_TIMEOUT = 12
_HEADERS = {"User-Agent": "guaranteed-play-draftkit"}


def parse_draft_id(text):
    """Pull a draft id out of a raw id or a Sleeper draft URL
    (e.g. https://sleeper.com/draft/nfl/1234567890123456789)."""
    if not text:
        return ""
    t = str(text).strip()
    m = re.search(r"draft/(?:nfl/)?(\d+)", t)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6,})", t)
    return m.group(1) if m else t


def _get(url):
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


def fetch_draft(draft_id):
    """Draft metadata dict (raises requests.HTTPError on a bad id)."""
    return _get(f"{SLEEPER_BASE}/draft/{draft_id}")


def fetch_picks(draft_id):
    """List of pick dicts, in pick order (empty list before any pick)."""
    picks = _get(f"{SLEEPER_BASE}/draft/{draft_id}/picks")
    return picks or []


def draft_team_count(draft_id, default=12):
    """Number of teams in the draft (from its settings), for snake math.
    Falls back to `default` on any error."""
    return draft_info(draft_id, default_teams=default)["teams"]


def draft_info(draft_id, default_teams=12):
    """Draft metadata we need to render: team count, round count, and status
    ("complete" once the draft is finished). Robust to any fetch error."""
    try:
        meta = fetch_draft(draft_id)
        settings = meta.get("settings") or {}
        return {
            "teams": int(settings.get("teams") or default_teams),
            "rounds": int(settings.get("rounds") or 0),
            "status": str(meta.get("status") or ""),
        }
    except Exception:
        return {"teams": default_teams, "rounds": 0, "status": ""}


def pick_name(pick):
    """Best display name for a pick, from its metadata block."""
    md = pick.get("metadata") or {}
    name = f"{md.get('first_name', '')} {md.get('last_name', '')}".strip()
    return name or str(pick.get("player_id", "")).strip()


def summarize_picks(picks, pool, my_slot=None):
    """Map Sleeper picks onto the candidate pool.

    Returns a dict:
      drafted   -- pool player_names for every matched pick (feeds
                   recommend_picks' `drafted_players`)
      my_team   -- pool player_names for picks at `my_slot` (feeds
                   `my_team`); [] if my_slot is None
      num_picks -- total picks seen
      matched   -- how many mapped onto the pool
      last      -- short label for the most recent pick ("R2.14 Bijan Robinson")
    """
    result = {
        "drafted": [], "my_team": [], "drafted_by": {},
        "num_picks": len(picks), "matched": 0, "last": "",
    }
    if pool is None or pool.empty or not picks:
        return result

    id_to_name = {}
    if "player_id" in pool.columns:
        for pid, name in zip(pool["player_id"], pool["player_name"]):
            if pd.notna(pid):
                # Sleeper ids are strings; pool ids may arrive as floats.
                key = str(pid)
                if key.endswith(".0"):
                    key = key[:-2]
                id_to_name[key] = name
    norm_to_name = {_norm(n): n for n in pool["player_name"]}

    for pick in picks:
        pid = str(pick.get("player_id", "")).strip()
        name = id_to_name.get(pid) or norm_to_name.get(_norm(pick_name(pick)))
        if name is None:
            continue  # not in our pool (K/DST/deep) -> nothing to exclude
        result["matched"] += 1
        result["drafted"].append(name)
        slot = pick.get("draft_slot")
        if slot is not None:
            result["drafted_by"][name] = slot
        if my_slot and slot == my_slot:
            result["my_team"].append(name)

    last = picks[-1]
    result["last"] = f"R{last.get('round', '?')}.{last.get('pick_no', '?')} {pick_name(last)}"
    # De-dup while preserving order (a pick can't be taken twice, but guard).
    result["drafted"] = list(dict.fromkeys(result["drafted"]))
    result["my_team"] = list(dict.fromkeys(result["my_team"]))
    return result
