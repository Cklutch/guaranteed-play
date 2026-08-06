from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from draftkit.data_access import safe_col


# Underdog's real "Draft Table" export doesn't use a plain "ADP" header --
# it has one or more dated snapshot columns like "ADP on August  5" (and
# often a stale earlier snapshot alongside it, e.g. "ADP on April 25").
# Always prefer the most recent snapshot over a stale one.
_ADP_ON_PATTERN = re.compile(r"^adp on\s+(.+)$", re.IGNORECASE)
_ADP_ON_DATE_FORMATS = ("%B %d", "%b %d", "%m/%d/%Y", "%m/%d")


def _find_latest_adp_on_column(columns):
    candidates = []
    for column in columns:
        match = _ADP_ON_PATTERN.match(str(column).strip())
        if match:
            candidates.append((column, match.group(1).strip()))

    if not candidates:
        return None

    dated = []
    for column, label in candidates:
        parsed_date = None
        for fmt in _ADP_ON_DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(label, fmt)
                break
            except ValueError:
                continue
        if parsed_date is not None:
            dated.append((column, parsed_date))

    if dated:
        return max(dated, key=lambda item: item[1])[0]

    # Couldn't parse any label as a date -- fall back to the last matching
    # column in file order (Underdog exports place the newest snapshot last).
    return candidates[-1][0]


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

UNDERDOG_SOURCE_CANDIDATES = [
    Path("data/raw/underdog_adp.csv"),
    Path("data/raw/real_adp.csv"),
    Path("data/raw/current_adp.csv"),
    Path("data/underdog_adp.csv"),
    Path("data/adp.csv"),
]
# Intentionally excludes any sample/placeholder file. A prior version fell
# back to "data/players.csv" (a 27-row hand-authored sample) when none of the
# real candidates existed, which silently made that sample file the source of
# "ADP" for the entire app. If no real ADP export is present, this loader
# must return empty rather than fabricate one.


def _empty_canonical_df():
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _read_first_existing_csv():
    for path in UNDERDOG_SOURCE_CANDIDATES:
        if not path.exists():
            continue

        try:
            return pd.read_csv(path), path
        except Exception:
            return pd.DataFrame(), path

    return pd.DataFrame(), None


def load_underdog_adp():
    """
    Load a real ADP export (Underdog, Sleeper mock drafts, FantasyFootballCalculator,
    etc.) into the canonical player schema. This reads whichever manually-provided
    CSV matches UNDERDOG_SOURCE_CANDIDATES -- it does not call any live API despite
    the module name. Returns an empty frame if no real export is present; callers
    must not treat that as "ADP is 0/unknown," it means no ADP data was provided.
    """
    df, source_path = _read_first_existing_csv()
    if df.empty:
        out = _empty_canonical_df()
        out.attrs["source_path"] = str(source_path) if source_path else None
        return out

    player_id_col = safe_col(df, ["player_id", "underdog_id", "id"])
    name_col = safe_col(df, ["player_name", "Player", "player", "name", "full_name"])
    position_col = safe_col(df, ["position", "pos", "Position"])
    team_col = safe_col(df, ["team", "Team", "team_abbr"])
    adp_col = safe_col(df, ["adp", "ADP", "consensus_adp"]) or _find_latest_adp_on_column(df.columns)
    adp_rank_col = safe_col(df, ["adp_rank", "ADP Rank", "rank", "Rank"])

    rows = []
    working_df = df.copy()
    if adp_col:
        working_df[adp_col] = pd.to_numeric(working_df[adp_col], errors="coerce")
        working_df = working_df.sort_values(adp_col, ascending=True).reset_index(drop=True)

    for idx, row in working_df.iterrows():
        player_name = row.get(name_col) if name_col else None
        if not player_name:
            continue

        adp = row.get(adp_col) if adp_col else None
        adp_rank = row.get(adp_rank_col) if adp_rank_col else idx + 1

        rows.append({
            "player_id": row.get(player_id_col) if player_id_col else None,
            "player_name": str(player_name).strip(),
            "position": str(row.get(position_col, "")).upper() if position_col else "",
            "team": str(row.get(team_col, "")) if team_col else "",
            "bye_week": None,
            "age": None,
            "projection_points": None,
            "projection_rank": None,
            "adp": adp,
            "adp_rank": adp_rank,
            "injury_risk": None,
            "durability_grade": None,
            "boom_score": None,
            "bust_score": None,
            "stability_score": None,
            "archetype": None,
        })

    out = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    out.attrs["source_path"] = str(source_path) if source_path else None
    return out
