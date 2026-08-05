from pathlib import Path

import pandas as pd

from draftkit.data_access import safe_col


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
    Path("data/underdog_adp.csv"),
    Path("data/adp.csv"),
    Path("data/players.csv"),
]


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
    Load Underdog ADP data into the canonical player schema.
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
    adp_col = safe_col(df, ["adp", "ADP", "consensus_adp"])
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
