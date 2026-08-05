from datetime import datetime
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_SPORT = "nfl"


def _empty_result(default_value, error_message=None):
    if isinstance(default_value, list):
        return []
    if isinstance(default_value, dict):
        result = {}
        if error_message:
            result["_error"] = error_message
        return result
    return default_value


def _clean_id(value):
    if value is None:
        return ""
    return str(value).strip()


def _sleeper_get(endpoint, default_value):
    """
    Safely call a Sleeper public API endpoint.
    """
    endpoint = str(endpoint).lstrip("/")
    url = f"{SLEEPER_BASE_URL}/{endpoint}"
    request = Request(url, headers={"User-Agent": "Guaranteed-Play/1.0"})

    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        return _empty_result(default_value, f"HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        return _empty_result(default_value, f"Network error: {exc.reason}")
    except OSError as exc:
        return _empty_result(default_value, f"Network error: {exc}")

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError:
        return _empty_result(default_value, "Sleeper returned a non-JSON response.")

    if data is None:
        return default_value

    return data


def get_draft_info(draft_id):
    """
    Return Sleeper draft metadata for a draft ID.
    """
    draft_id = _clean_id(draft_id)
    if not draft_id:
        return {"_error": "Missing draft_id."}

    data = _sleeper_get(f"draft/{draft_id}", {})
    return data if isinstance(data, dict) else {"_error": "Unexpected draft response."}


def get_draft_picks(draft_id):
    """
    Return all completed picks for a Sleeper draft.
    """
    draft_id = _clean_id(draft_id)
    if not draft_id:
        return []

    data = _sleeper_get(f"draft/{draft_id}/picks", [])
    return data if isinstance(data, list) else []


def get_current_pick(draft_id):
    """
    Infer current pick state from draft info and completed picks.
    """
    draft_info = get_draft_info(draft_id)
    picks = get_draft_picks(draft_id)

    if draft_info.get("_error"):
        return {
            "draft_id": _clean_id(draft_id),
            "status": "unknown",
            "current_pick": None,
            "completed_picks": len(picks),
            "is_complete": False,
            "error": draft_info["_error"],
        }

    settings = draft_info.get("settings") or {}
    status = draft_info.get("status", "unknown")
    completed_picks = len(picks)
    teams = int(settings.get("teams") or 0)
    rounds = int(settings.get("rounds") or 0)
    total_picks = teams * rounds if teams and rounds else None
    is_complete = status == "complete"

    if is_complete:
        current_pick = completed_picks
    elif total_picks is not None:
        current_pick = min(completed_picks + 1, total_picks)
    else:
        current_pick = completed_picks + 1

    return {
        "draft_id": _clean_id(draft_id),
        "status": status,
        "current_pick": current_pick,
        "completed_picks": completed_picks,
        "total_picks": total_picks,
        "is_complete": is_complete,
    }


def get_drafted_players(draft_id):
    """
    Return normalized player pick records from a Sleeper draft.
    """
    picks = get_draft_picks(draft_id)
    drafted_players = []

    for pick in picks:
        if not isinstance(pick, dict):
            continue

        metadata = pick.get("metadata") or {}
        drafted_players.append({
            "player_id": pick.get("player_id"),
            "player_name": metadata.get("first_name") and metadata.get("last_name")
            and f"{metadata.get('first_name')} {metadata.get('last_name')}"
            or metadata.get("player_name")
            or metadata.get("full_name")
            or "",
            "position": metadata.get("position", ""),
            "team": metadata.get("team", ""),
            "picked_by": pick.get("picked_by"),
            "roster_id": pick.get("roster_id"),
            "round": pick.get("round"),
            "pick_no": pick.get("pick_no"),
            "draft_slot": pick.get("draft_slot"),
            "metadata": metadata,
        })

    return drafted_players


def get_rosters(league_id):
    """
    Return Sleeper rosters for a league ID.
    """
    league_id = _clean_id(league_id)
    if not league_id:
        return []

    data = _sleeper_get(f"league/{league_id}/rosters", [])
    return data if isinstance(data, list) else []


def get_user_drafts(user_id, season=None):
    """
    Return a user's Sleeper drafts for an NFL season.
    """
    user_id = _clean_id(user_id)
    if not user_id:
        return []

    if season is None:
        season = datetime.now().year

    data = _sleeper_get(f"user/{user_id}/drafts/{DEFAULT_SPORT}/{season}", [])
    return data if isinstance(data, list) else []


def _get_league_info(league_id):
    league_id = _clean_id(league_id)
    if not league_id:
        return {"_error": "Missing league_id."}

    data = _sleeper_get(f"league/{league_id}", {})
    return data if isinstance(data, dict) else {"_error": "Unexpected league response."}


def _count_roster_players(roster):
    players = roster.get("players") or []
    starters = roster.get("starters") or []
    reserve = roster.get("reserve") or []
    taxi = roster.get("taxi") or []

    return {
        "roster_id": roster.get("roster_id"),
        "owner_id": roster.get("owner_id"),
        "players": len(players),
        "starters": len(starters),
        "reserve": len(reserve),
        "taxi": len(taxi),
    }


def get_sleeper_debug_info(
    draft_id=None,
    league_id=None,
    user_id=None,
    season=None,
):
    """
    Return live Sleeper integration diagnostics without mutating app state.
    """
    errors = []
    warnings = []
    league_info = {}

    if league_id:
        league_info = _get_league_info(league_id)
        if league_info.get("_error"):
            errors.append(league_info["_error"])
        elif not draft_id:
            draft_id = league_info.get("draft_id")
            if not draft_id:
                warnings.append("League did not include a draft_id.")

    draft_info = get_draft_info(draft_id) if draft_id else {}
    if draft_info.get("_error"):
        errors.append(draft_info["_error"])

    picks = get_draft_picks(draft_id) if draft_id else []
    drafted_players = get_drafted_players(draft_id) if draft_id else []
    current_pick = get_current_pick(draft_id) if draft_id else {}
    rosters = get_rosters(league_id) if league_id else []
    user_drafts = get_user_drafts(user_id, season=season) if user_id else []

    roster_counts = [
        _count_roster_players(roster)
        for roster in rosters
        if isinstance(roster, dict)
    ]

    settings = draft_info.get("settings") if isinstance(draft_info, dict) else {}
    metadata = draft_info.get("metadata") if isinstance(draft_info, dict) else {}

    return {
        "league_id": _clean_id(league_id),
        "draft_id": _clean_id(draft_id),
        "user_id": _clean_id(user_id),
        "season": season,
        "draft_status": draft_info.get("status") if isinstance(draft_info, dict) else None,
        "draft_type": draft_info.get("type") if isinstance(draft_info, dict) else None,
        "draft_settings": settings or {},
        "draft_metadata": metadata or {},
        "league_settings": league_info.get("settings", {}) if isinstance(league_info, dict) else {},
        "league_roster_positions": league_info.get("roster_positions", [])
        if isinstance(league_info, dict)
        else [],
        "pick_counts": {
            "completed": len(picks),
            "drafted_players": len(drafted_players),
            "current_pick": current_pick.get("current_pick") if current_pick else None,
            "total_picks": current_pick.get("total_picks") if current_pick else None,
        },
        "roster_counts": {
            "rosters": len(rosters),
            "players_by_roster": roster_counts,
        },
        "user_draft_count": len(user_drafts),
        "errors": errors,
        "warnings": warnings,
    }
