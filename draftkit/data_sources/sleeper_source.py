from datetime import date, datetime
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
DEFAULT_TIMEOUT_SECONDS = 30

CANONICAL_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "team",
    "bye_week",
    "age",
    "projection_points",
    "projection_rank",
    "adp",
    "adp_rank",
    "injury_risk",
    "durability_grade",
    "boom_score",
    "bust_score",
    "stability_score",
    "archetype",
]
SLEEPER_EXTRA_COLUMNS = ["injury_status"]
OUTPUT_COLUMNS = CANONICAL_COLUMNS + SLEEPER_EXTRA_COLUMNS


def _empty_sleeper_df(error=None):
    df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    df.attrs["source_url"] = SLEEPER_PLAYERS_URL
    df.attrs["error"] = error
    df.attrs["validation"] = validate_sleeper_players(df)
    return df


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    number = _safe_float(value, default)
    if number is None:
        return default
    return int(number)


def _fetch_sleeper_players_payload(timeout=DEFAULT_TIMEOUT_SECONDS):
    request = Request(
        SLEEPER_PLAYERS_URL,
        headers={"User-Agent": "Guaranteed-Play/1.0"},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        return None, f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return None, f"Network error: {exc.reason}"
    except OSError as exc:
        return None, f"Network error: {exc}"

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError:
        return None, "Sleeper returned a non-JSON player payload."

    if not isinstance(data, dict):
        return None, "Sleeper player payload was not a dictionary."

    return data, None


def _build_player_name(player):
    full_name = player.get("full_name") or player.get("search_full_name")
    if full_name:
        return str(full_name).strip()

    first_name = player.get("first_name") or ""
    last_name = player.get("last_name") or ""
    name = f"{first_name} {last_name}".strip()
    if name:
        return name

    return str(player.get("player_id") or "").strip()


def _derive_age(player):
    direct_age = _safe_int(player.get("age"))
    if direct_age is not None:
        return direct_age

    birth_date = player.get("birth_date")
    if not birth_date:
        return None

    parsed_birth_date = None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y"]:
        try:
            parsed_birth_date = datetime.strptime(str(birth_date), fmt).date()
            break
        except ValueError:
            continue

    if parsed_birth_date is None:
        return None

    today = date.today()
    age = today.year - parsed_birth_date.year
    if (today.month, today.day) < (parsed_birth_date.month, parsed_birth_date.day):
        age -= 1

    return age


def _normalize_injury_risk(injury_status):
    status = str(injury_status or "").strip().lower()
    if not status:
        return None
    if any(term in status for term in ["out", "ir", "pup", "doubtful", "nfi"]):
        return 85.0
    if any(term in status for term in ["questionable", "limited"]):
        return 65.0
    if any(term in status for term in ["probable", "active", "healthy"]):
        return 30.0
    return 50.0


def _map_sleeper_player(player_id, player):
    injury_status = player.get("injury_status") or player.get("status")

    return {
        "player_id": str(player.get("player_id") or player_id),
        "player_name": _build_player_name(player),
        "position": str(player.get("position") or "").upper(),
        "team": str(player.get("team") or player.get("fantasy_data_id") and "" or ""),
        "bye_week": _safe_int(player.get("bye_week")),
        "age": _derive_age(player),
        "projection_points": None,
        "projection_rank": None,
        "adp": None,
        "adp_rank": None,
        "injury_risk": _normalize_injury_risk(injury_status),
        "durability_grade": None,
        "boom_score": None,
        "bust_score": None,
        "stability_score": None,
        "archetype": None,
        "injury_status": injury_status,
    }


def validate_sleeper_players(players_df):
    """
    Validate Sleeper source output without raising hard failures.
    """
    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in players_df.columns
    ]

    if players_df.empty:
        return {
            "is_valid": False,
            "row_count": 0,
            "missing_columns": missing_columns,
            "duplicate_player_ids": [],
            "missing_data_counts": {},
            "messages": ["Sleeper source returned no players."],
        }

    duplicate_player_ids = (
        players_df.loc[
            players_df["player_id"].notna()
            & players_df["player_id"].duplicated(keep=False),
            "player_id",
        ]
        .astype(str)
        .unique()
        .tolist()
        if "player_id" in players_df.columns
        else []
    )
    missing_data_counts = {
        column: int(players_df[column].isna().sum())
        for column in OUTPUT_COLUMNS
        if column in players_df.columns
    }

    messages = []
    if missing_columns:
        messages.append("One or more Sleeper output columns are missing.")
    if duplicate_player_ids:
        messages.append("Duplicate Sleeper player IDs detected.")
    if missing_data_counts.get("player_name", 0) > 0:
        messages.append("Some Sleeper players are missing names.")
    if missing_data_counts.get("position", 0) > 0:
        messages.append("Some Sleeper players are missing positions.")

    return {
        "is_valid": not missing_columns and not duplicate_player_ids and len(players_df) > 0,
        "row_count": int(len(players_df)),
        "missing_columns": missing_columns,
        "duplicate_player_ids": duplicate_player_ids,
        "missing_data_counts": missing_data_counts,
        "messages": messages,
    }


def load_sleeper_players(timeout=DEFAULT_TIMEOUT_SECONDS):
    """
    Load real Sleeper NFL player metadata into the canonical player schema.
    """
    payload, error = _fetch_sleeper_players_payload(timeout=timeout)
    if error:
        return _empty_sleeper_df(error=error)

    rows = []
    for player_id, player in payload.items():
        if not isinstance(player, dict):
            continue

        mapped_player = _map_sleeper_player(player_id, player)
        if not mapped_player["player_name"]:
            continue

        rows.append(mapped_player)

    players_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    players_df.attrs["source_url"] = SLEEPER_PLAYERS_URL
    players_df.attrs["error"] = None
    players_df.attrs["validation"] = validate_sleeper_players(players_df)

    return players_df


def get_sleeper_source_debug_info(timeout=DEFAULT_TIMEOUT_SECONDS):
    """
    Return diagnostics for the Sleeper source adapter.
    """
    players_df = load_sleeper_players(timeout=timeout)
    validation = players_df.attrs.get("validation", validate_sleeper_players(players_df))

    if players_df.empty:
        position_counts = {}
        team_counts = {}
        active_count = 0
    else:
        position_counts = players_df["position"].value_counts(dropna=False).head(20).to_dict()
        team_counts = players_df["team"].value_counts(dropna=False).head(20).to_dict()
        active_count = int(players_df["team"].astype(str).str.strip().ne("").sum())

    return {
        "source_url": SLEEPER_PLAYERS_URL,
        "row_count": int(len(players_df)),
        "columns": list(players_df.columns),
        "validation": validation,
        "error": players_df.attrs.get("error"),
        "position_counts": position_counts,
        "top_team_counts": team_counts,
        "players_with_team_count": active_count,
        "players_with_age_count": int(players_df["age"].notna().sum())
        if "age" in players_df.columns
        else 0,
        "players_with_injury_status_count": int(players_df["injury_status"].notna().sum())
        if "injury_status" in players_df.columns
        else 0,
        "sample_rows": players_df.head(10).to_dict("records")
        if not players_df.empty
        else [],
    }
