from pathlib import Path
import re

import pandas as pd


SPORTSBOOK = "Pinnacle"
DEFAULT_SOURCE_PATH = Path("data/raw/pinnacle_props.csv")
PROP_COLUMNS = ["player_name", "market_type", "line_value", "sportsbook"]


def _empty_props_df(source_path=None, error=None):
    df = pd.DataFrame(columns=PROP_COLUMNS)
    df.attrs["source_path"] = str(source_path) if source_path else str(DEFAULT_SOURCE_PATH)
    df.attrs["error"] = error
    df.attrs["validation"] = validate_sportsbook_props(df)
    return df


def _safe_col(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _normalize_market_type(value):
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    if "pass" in text and ("yard" in text or "yd" in text):
        return "passing_yards"
    if "pass" in text and ("td" in text or "touchdown" in text):
        return "passing_tds"
    if "rush" in text and ("yard" in text or "yd" in text):
        return "rushing_yards"
    if "rush" in text and ("td" in text or "touchdown" in text):
        return "rushing_tds"
    if "receiv" in text and ("yard" in text or "yd" in text):
        return "receiving_yards"
    if "receiv" in text and ("td" in text or "touchdown" in text):
        return "receiving_tds"
    if "reception" in text or text == "rec" or " receptions" in f" {text}":
        return "receptions"

    return text.replace(" ", "_")


def validate_sportsbook_props(props_df):
    try:
        missing_columns = [column for column in PROP_COLUMNS if column not in props_df.columns]
        if props_df.empty:
            return {
                "is_valid": False,
                "row_count": 0,
                "missing_columns": missing_columns,
                "missing_player_count": 0,
                "missing_market_count": 0,
                "missing_line_count": 0,
                "messages": ["No Pinnacle props loaded."],
            }

        missing_player_count = int(props_df["player_name"].isna().sum())
        missing_market_count = int(props_df["market_type"].isna().sum())
        missing_line_count = int(props_df["line_value"].isna().sum())
        messages = []

        if missing_columns:
            messages.append("One or more canonical prop columns are missing.")
        if missing_player_count:
            messages.append("Some rows are missing player names.")
        if missing_market_count:
            messages.append("Some rows are missing market types.")
        if missing_line_count:
            messages.append("Some rows are missing numeric lines.")

        return {
            "is_valid": not missing_columns and missing_player_count == 0 and missing_market_count == 0,
            "row_count": int(len(props_df)),
            "missing_columns": missing_columns,
            "missing_player_count": missing_player_count,
            "missing_market_count": missing_market_count,
            "missing_line_count": missing_line_count,
            "messages": messages,
        }
    except Exception as exc:
        return {
            "is_valid": False,
            "row_count": 0,
            "missing_columns": PROP_COLUMNS,
            "missing_player_count": 0,
            "missing_market_count": 0,
            "missing_line_count": 0,
            "messages": [f"Pinnacle validation failed: {exc}"],
        }


def load_pinnacle_props(source_path=DEFAULT_SOURCE_PATH):
    source_path = Path(source_path)
    if not source_path.exists():
        return _empty_props_df(source_path=source_path)

    try:
        raw_df = pd.read_csv(source_path)
    except Exception as exc:
        return _empty_props_df(source_path=source_path, error=str(exc))

    if raw_df.empty:
        return _empty_props_df(source_path=source_path)

    player_col = _safe_col(raw_df, ["player_name", "player", "name", "Player", "PLAYER"])
    market_col = _safe_col(raw_df, ["market_type", "market", "prop", "Prop", "stat", "Stat"])
    line_col = _safe_col(raw_df, ["line_value", "line", "Line", "value", "Value"])

    if player_col is None or market_col is None or line_col is None:
        out = _empty_props_df(source_path=source_path)
        out.attrs["error"] = "Missing player, market, or line column."
        return out

    out = pd.DataFrame({
        "player_name": raw_df[player_col].astype(str).str.strip(),
        "market_type": raw_df[market_col].apply(_normalize_market_type),
        "line_value": pd.to_numeric(raw_df[line_col], errors="coerce"),
        "sportsbook": SPORTSBOOK,
    })
    out = out[out["player_name"].astype(str).str.strip().ne("")].copy()
    out.attrs["source_path"] = str(source_path)
    out.attrs["error"] = None
    out.attrs["validation"] = validate_sportsbook_props(out)

    return out[PROP_COLUMNS].reset_index(drop=True)


def get_source_debug_info(source_path=DEFAULT_SOURCE_PATH):
    props_df = load_pinnacle_props(source_path=source_path)
    validation = props_df.attrs.get("validation", validate_sportsbook_props(props_df))

    return {
        "sportsbook": SPORTSBOOK,
        "source_path": props_df.attrs.get("source_path"),
        "row_count": int(len(props_df)),
        "market_counts": props_df["market_type"].value_counts().to_dict()
        if not props_df.empty
        else {},
        "player_count": int(props_df["player_name"].nunique()) if not props_df.empty else 0,
        "validation": validation,
        "error": props_df.attrs.get("error"),
        "sample_rows": props_df.head(10).to_dict("records") if not props_df.empty else [],
    }
