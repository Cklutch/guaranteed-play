from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from validation_utils import PROJECT_ROOT, VALIDATION_DIR, clean_name, initial_last_key

ACCEPTED_PATHS = [
    VALIDATION_DIR / "historical_sportsbook_props.csv",
    VALIDATION_DIR / "historical_player_props.csv",
    PROJECT_ROOT / "data" / "research" / "historical_sportsbook_props.csv",
]

NORMALIZED_OUTPUT = VALIDATION_DIR / "sportsbook_props_normalized.csv"
REQUIRED_COLUMNS = ["season", "player_name", "position", "market", "line", "sportsbook", "odds", "odds_format", "snapshot_date"]
OPTIONAL_COLUMNS = ["team", "opponent", "market_type", "over_odds", "under_odds", "source", "source_url_or_file", "is_season_long", "is_preseason_snapshot"]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + [
    "normalized_market", "player_key", "initial_last_key", "american_odds", "raw_implied_probability",
    "over_implied_probability", "under_implied_probability", "no_vig_over_probability",
    "missing_line_flag", "suspicious_line_flag", "unsupported_market_flag", "preseason_safety_flag",
]

SUPPORTED_MARKETS = {
    "receiving_yards", "receptions", "receiving_tds", "anytime_td", "season_receiving_yards", "season_receptions", "season_receiving_tds",
    "rushing_yards", "rushing_attempts", "rushing_tds", "season_rushing_yards", "season_rushing_tds", "season_receptions",
    "team_win_total", "team_total_points", "game_total", "team_implied_points",
}

MARKET_ALIASES = {
    "rec_yards": "receiving_yards",
    "receiver_yards": "receiving_yards",
    "receiving_yds": "receiving_yards",
    "season_rec_yards": "season_receiving_yards",
    "season_receiving_yds": "season_receiving_yards",
    "rec": "receptions",
    "receiving_receptions": "receptions",
    "season_rec": "season_receptions",
    "receiving_touchdowns": "receiving_tds",
    "rec_tds": "receiving_tds",
    "rush_yards": "rushing_yards",
    "rushing_yds": "rushing_yards",
    "rush_attempts": "rushing_attempts",
    "carries": "rushing_attempts",
    "rush_tds": "rushing_tds",
    "rushing_touchdowns": "rushing_tds",
    "any_time_td": "anytime_td",
    "anytime_touchdown": "anytime_td",
    "win_total": "team_win_total",
    "team_points": "team_total_points",
    "total": "game_total",
    "implied_team_total": "team_implied_points",
}


def first_existing_path() -> Path | None:
    for path in ACCEPTED_PATHS:
        if path.exists():
            return path
    return None


def normalize_market(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    text = MARKET_ALIASES.get(text, text)
    if text.startswith("player_"):
        text = text[len("player_"):]
    return MARKET_ALIASES.get(text, text)


def normalize_position(value: object, market: str) -> str:
    text = str(value or "").upper().strip()
    if "WR" in text:
        return "WR"
    if "RB" in text:
        return "RB"
    if market in {"team_win_total", "team_total_points", "game_total", "team_implied_points"}:
        return "TEAM"
    return text


def american_to_implied_probability(odds: object) -> float:
    value = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(value) or value == 0:
        return np.nan
    value = float(value)
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def no_vig_probability(over_odds: object, under_odds: object) -> float:
    over = american_to_implied_probability(over_odds)
    under = american_to_implied_probability(under_odds)
    total = over + under if pd.notna(over) and pd.notna(under) else np.nan
    if pd.isna(total) or total <= 0:
        return np.nan
    return over / total


def coerce_bool(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).lower().strip(): c for c in df.columns}
    aliases = {
        "season": ["season", "year"],
        "player_name": ["player_name", "player", "name", "player name"],
        "position": ["position", "pos"],
        "market": ["market", "market_type", "prop", "stat"],
        "line": ["line", "line_value", "value"],
        "sportsbook": ["sportsbook", "book", "bookmaker", "provider"],
        "odds": ["odds", "price", "american_odds"],
        "odds_format": ["odds_format", "format"],
        "snapshot_date": ["snapshot_date", "date", "as_of", "captured_at"],
        "team": ["team", "player_team"],
        "opponent": ["opponent", "opp"],
        "over_odds": ["over_odds", "over_price"],
        "under_odds": ["under_odds", "under_price"],
        "source": ["source", "dataset"],
        "source_url_or_file": ["source_url_or_file", "source_file", "source_url", "url"],
        "is_season_long": ["is_season_long", "season_long"],
        "is_preseason_snapshot": ["is_preseason_snapshot", "preseason_snapshot"],
    }
    rename = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower:
                rename[lower[candidate]] = target
                break
    return df.rename(columns=rename)


def load_outcome_keys() -> pd.DataFrame:
    path = VALIDATION_DIR / "predraft_validation_dataset_expanded.csv"
    if not path.exists():
        path = VALIDATION_DIR / "predraft_validation_dataset.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "position", "player_key", "initial_last_key"])
    df = pd.read_csv(path, usecols=lambda c: c in {"season", "position", "player_name"})
    df = df[df["position"].isin(["WR", "RB"])].copy()
    df["player_key"] = df["player_name"].apply(clean_name)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    return df[["season", "position", "player_key", "initial_last_key"]].drop_duplicates()


def normalize_props(path: Path | None) -> tuple[pd.DataFrame, dict[str, object]]:
    if path is None:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, {"sportsbook_data_file_used": "not_found", "sportsbook_source": "not_found", "raw_rows": 0, "normalized_rows": 0}
    raw = pd.read_csv(path)
    df = normalize_columns(raw)
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns or df[col].isna().all()]
    df = df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["normalized_market"] = df["market"].apply(normalize_market)
    df["position"] = [normalize_position(pos, market) for pos, market in zip(df["position"], df["normalized_market"])]
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["american_odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_key"] = df["player_name"].apply(clean_name)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    df["raw_implied_probability"] = df["american_odds"].apply(american_to_implied_probability)
    df["over_implied_probability"] = df["over_odds"].apply(american_to_implied_probability)
    df["under_implied_probability"] = df["under_odds"].apply(american_to_implied_probability)
    df["no_vig_over_probability"] = [no_vig_probability(o, u) for o, u in zip(df["over_odds"], df["under_odds"])]
    df["missing_line_flag"] = df["line"].isna()
    df["unsupported_market_flag"] = ~df["normalized_market"].isin(SUPPORTED_MARKETS)
    df["is_season_long"] = df["is_season_long"].apply(coerce_bool)
    df["is_preseason_snapshot"] = df["is_preseason_snapshot"].apply(coerce_bool)
    df["preseason_safety_flag"] = np.where(df["is_preseason_snapshot"].eq(True), "preseason", "unknown_or_not_preseason")

    suspicious = pd.Series(False, index=df.index)
    suspicious |= df["line"].lt(0)
    suspicious |= df["normalized_market"].isin(["receptions", "season_receptions"]) & df["line"].gt(180)
    suspicious |= df["normalized_market"].str.contains("yard", na=False) & df["line"].gt(2500)
    suspicious |= df["normalized_market"].str.contains("td", na=False) & df["line"].gt(40)
    suspicious |= df["normalized_market"].eq("anytime_td") & df["line"].gt(1)
    df["suspicious_line_flag"] = suspicious

    df = df.dropna(subset=["season", "position", "normalized_market"])
    df = df.sort_values(["season", "position", "player_key", "normalized_market", "snapshot_date", "sportsbook"], na_position="last")
    duplicate_rows = int(df.duplicated(["season", "position", "player_key", "normalized_market", "snapshot_date", "sportsbook"], keep="first").sum())
    df = df.drop_duplicates(["season", "position", "player_key", "normalized_market", "snapshot_date", "sportsbook"], keep="first")
    meta = {
        "sportsbook_data_file_used": str(path),
        "sportsbook_source": str(df["source"].dropna().iloc[0]) if df["source"].notna().any() else str(df["sportsbook"].dropna().iloc[0]) if df["sportsbook"].notna().any() else path.name,
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(df)),
        "missing_required_columns_or_empty": missing_required,
        "duplicate_rows_dropped": duplicate_rows,
        "missing_line_rows": int(df["missing_line_flag"].sum()),
        "suspicious_line_rows": int(df["suspicious_line_flag"].sum()),
        "unsupported_market_rows": int(df["unsupported_market_flag"].sum()),
        "seasons_covered": sorted(df["season"].dropna().astype(int).unique().tolist()),
        "markets_covered": sorted(df["normalized_market"].dropna().unique().tolist()),
    }
    return df[OUTPUT_COLUMNS], meta


def write_diagnostics(df: pd.DataFrame, meta: dict[str, object]) -> None:
    coverage = pd.DataFrame(columns=["season", "position", "rows", "players", "markets"])
    if not df.empty:
        coverage = df.groupby(["season", "position"], dropna=False).agg(
            rows=("player_name", "size"), players=("player_key", "nunique"), markets=("normalized_market", "nunique")
        ).reset_index()
    coverage.to_csv(VALIDATION_DIR / "sportsbook_props_coverage_by_season.csv", index=False)

    unmatched = pd.DataFrame(columns=df.columns)
    if not df.empty:
        keys = load_outcome_keys()
        exact = set(zip(keys["season"].astype(str), keys["position"].astype(str), keys["player_key"].astype(str)))
        initial = set(zip(keys["season"].astype(str), keys["position"].astype(str), keys["initial_last_key"].astype(str)))
        tmp = df[df["position"].isin(["WR", "RB"])].copy()
        tmp["_exact"] = list(zip(tmp["season"].astype(str), tmp["position"].astype(str), tmp["player_key"].astype(str)))
        tmp["_initial"] = list(zip(tmp["season"].astype(str), tmp["position"].astype(str), tmp["initial_last_key"].astype(str)))
        unmatched = tmp[~tmp["_exact"].isin(exact) & ~tmp["_initial"].isin(initial)].drop(columns=["_exact", "_initial"]).head(100)
    unmatched.to_csv(VALIDATION_DIR / "sportsbook_props_unmatched_player_examples.csv", index=False)

    market_rows = df[df["unsupported_market_flag"]].head(100) if not df.empty else pd.DataFrame(columns=df.columns)
    market_rows.to_csv(VALIDATION_DIR / "sportsbook_props_market_rows_not_matched_examples.csv", index=False)

    meta = dict(meta)
    meta["unmatched_player_examples_rows"] = int(len(unmatched))
    meta["unsupported_market_example_rows"] = int(len(market_rows))
    with (VALIDATION_DIR / "sportsbook_props_import_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import historical sportsbook props for fantasy research validation only.")
    parser.add_argument("--input", default=None, help="Optional explicit CSV path; otherwise first accepted path is used.")
    args = parser.parse_args()
    path = Path(args.input) if args.input else first_existing_path()
    df, meta = normalize_props(path)
    df.to_csv(NORMALIZED_OUTPUT, index=False)
    write_diagnostics(df, meta)
    print(f"sportsbook data file used: {meta['sportsbook_data_file_used']}")
    print(f"sportsbook source: {meta['sportsbook_source']}")
    print(f"rows imported: {len(df)}")
    print(f"seasons covered: {meta.get('seasons_covered', [])}")
    print(f"markets covered: {meta.get('markets_covered', [])}")
    print("diagnostics written: research/validation_v1/sportsbook_props_import_diagnostics.json")


if __name__ == "__main__":
    main()
