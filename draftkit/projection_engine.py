import pandas as pd

from draftkit.data_sources.draftkings_source import (
    get_source_debug_info as get_draftkings_debug_info,
    load_draftkings_props,
)
from draftkit.data_sources.fanduel_source import (
    get_source_debug_info as get_fanduel_debug_info,
    load_fanduel_props,
)
from draftkit.data_sources.pinnacle_source import (
    get_source_debug_info as get_pinnacle_debug_info,
    load_pinnacle_props,
)


MARKET_TYPES = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
]

PROJECTION_COLUMNS = [
    "player_name",
    "passing_yards_projection",
    "passing_td_projection",
    "rushing_yards_projection",
    "rushing_td_projection",
    "receiving_yards_projection",
    "receiving_td_projection",
    "receptions_projection",
    "fantasy_points_projection",
]

MARKET_TO_OUTPUT_COLUMN = {
    "passing_yards": "passing_yards_projection",
    "passing_tds": "passing_td_projection",
    "rushing_yards": "rushing_yards_projection",
    "rushing_tds": "rushing_td_projection",
    "receiving_yards": "receiving_yards_projection",
    "receiving_tds": "receiving_td_projection",
    "receptions": "receptions_projection",
}

FULL_PPR_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "receptions": 1.0,
}

HALF_PPR_SCORING = {**FULL_PPR_SCORING, "receptions": 0.5}


def _empty_projection_df():
    return pd.DataFrame(columns=PROJECTION_COLUMNS)


def build_player_projection_inputs(
    draftkings_df=None,
    fanduel_df=None,
    pinnacle_df=None,
):
    """
    Load sportsbook props and aggregate identical player-market lines.
    """
    draftkings_df = load_draftkings_props() if draftkings_df is None else draftkings_df
    fanduel_df = load_fanduel_props() if fanduel_df is None else fanduel_df
    pinnacle_df = load_pinnacle_props() if pinnacle_df is None else pinnacle_df

    props_df = pd.concat(
        [draftkings_df, fanduel_df, pinnacle_df],
        ignore_index=True,
    )

    if props_df.empty:
        out = pd.DataFrame(
            columns=[
                "player_name",
                "market_type",
                "consensus_market_line",
                "sportsbook_count",
                "sportsbooks",
            ]
        )
        out.attrs["source_row_counts"] = {
            "DraftKings": len(draftkings_df),
            "FanDuel": len(fanduel_df),
            "Pinnacle": len(pinnacle_df),
        }
        return out

    props_df = props_df[props_df["market_type"].isin(MARKET_TYPES)].copy()
    props_df["line_value"] = pd.to_numeric(props_df["line_value"], errors="coerce")
    props_df = props_df.dropna(subset=["player_name", "market_type", "line_value"]).copy()

    if props_df.empty:
        return pd.DataFrame(
            columns=[
                "player_name",
                "market_type",
                "consensus_market_line",
                "sportsbook_count",
                "sportsbooks",
            ]
        )

    rows = []
    for (player_name, market_type), group in props_df.groupby(["player_name", "market_type"]):
        sportsbook_lines = group.dropna(subset=["line_value"])
        rows.append({
            "player_name": player_name,
            "market_type": market_type,
            "consensus_market_line": round(float(sportsbook_lines["line_value"].mean()), 3),
            "sportsbook_count": int(sportsbook_lines["sportsbook"].nunique()),
            "sportsbooks": sorted(sportsbook_lines["sportsbook"].dropna().unique().tolist()),
        })

    out = pd.DataFrame(rows)
    out.attrs["source_row_counts"] = {
        "DraftKings": len(draftkings_df),
        "FanDuel": len(fanduel_df),
        "Pinnacle": len(pinnacle_df),
    }

    return out.sort_values(["player_name", "market_type"]).reset_index(drop=True)


def calculate_market_projection(consensus_market_line):
    """
    Version 1 uses sportsbook prop lines directly as market projections.
    """
    try:
        if pd.isna(consensus_market_line):
            return None
        return float(consensus_market_line)
    except (TypeError, ValueError):
        return None


def calculate_fantasy_points_projection(player_projection_row, scoring=HALF_PPR_SCORING):
    """
    Convert market projections into simple fantasy-point projections.
    """
    total = 0.0

    for market_type, output_column in MARKET_TO_OUTPUT_COLUMN.items():
        value = player_projection_row.get(output_column)
        try:
            if pd.isna(value):
                continue
            total += float(value) * scoring[market_type]
        except (TypeError, ValueError):
            continue

    return round(total, 2)


def build_player_projection_df(
    draftkings_df=None,
    fanduel_df=None,
    pinnacle_df=None,
    scoring=HALF_PPR_SCORING,
):
    projection_inputs_df = build_player_projection_inputs(
        draftkings_df=draftkings_df,
        fanduel_df=fanduel_df,
        pinnacle_df=pinnacle_df,
    )

    if projection_inputs_df.empty:
        return _empty_projection_df()

    rows = []
    for player_name, group in projection_inputs_df.groupby("player_name"):
        row = {"player_name": player_name}
        for market_type, output_column in MARKET_TO_OUTPUT_COLUMN.items():
            match = group[group["market_type"] == market_type]
            if match.empty:
                row[output_column] = pd.NA
            else:
                row[output_column] = calculate_market_projection(
                    match.iloc[0]["consensus_market_line"]
                )

        row["fantasy_points_projection"] = calculate_fantasy_points_projection(row, scoring=scoring)
        rows.append(row)

    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS).sort_values(
        "fantasy_points_projection",
        ascending=False,
    ).reset_index(drop=True)


def validate_projection_coverage(projections_df=None):
    projections_df = build_player_projection_df() if projections_df is None else projections_df.copy()

    if projections_df.empty:
        return {
            "is_valid": False,
            "player_count": 0,
            "players_with_any_projection": 0,
            "players_with_full_projection_profile": 0,
            "market_coverage_counts": {},
            "missing_markets_by_player": {},
            "messages": ["No sportsbook projections available."],
        }

    market_columns = [column for column in PROJECTION_COLUMNS if column.endswith("_projection")]
    market_columns = [column for column in market_columns if column != "fantasy_points_projection"]

    market_coverage_counts = {
        column: int(projections_df[column].notna().sum())
        for column in market_columns
    }
    players_with_any_projection = int(projections_df[market_columns].notna().any(axis=1).sum())
    players_with_full_projection_profile = int(projections_df[market_columns].notna().all(axis=1).sum())
    missing_markets_by_player = {}

    for _, row in projections_df.iterrows():
        missing = [
            column for column in market_columns
            if pd.isna(row.get(column))
        ]
        if missing:
            missing_markets_by_player[row["player_name"]] = missing

    return {
        "is_valid": players_with_any_projection > 0,
        "player_count": int(len(projections_df)),
        "players_with_any_projection": players_with_any_projection,
        "players_with_full_projection_profile": players_with_full_projection_profile,
        "market_coverage_counts": market_coverage_counts,
        "missing_markets_by_player": missing_markets_by_player,
        "messages": [],
    }


def get_projection_debug_info(
    draftkings_df=None,
    fanduel_df=None,
    pinnacle_df=None,
):
    projection_inputs_df = build_player_projection_inputs(
        draftkings_df=draftkings_df,
        fanduel_df=fanduel_df,
        pinnacle_df=pinnacle_df,
    )
    projections_df = build_player_projection_df(
        draftkings_df=draftkings_df,
        fanduel_df=fanduel_df,
        pinnacle_df=pinnacle_df,
    )
    validation = validate_projection_coverage(projections_df)

    return {
        "source_debug": {
            "DraftKings": get_draftkings_debug_info(),
            "FanDuel": get_fanduel_debug_info(),
            "Pinnacle": get_pinnacle_debug_info(),
        },
        "source_row_counts": projection_inputs_df.attrs.get("source_row_counts", {}),
        "merged_prop_count": int(len(projection_inputs_df)),
        "projection_row_count": int(len(projections_df)),
        "validation": validation,
        "sample_projection_inputs": projection_inputs_df.head(10).to_dict("records")
        if not projection_inputs_df.empty
        else [],
        "sample_projections": projections_df.head(10).to_dict("records")
        if not projections_df.empty
        else [],
    }
