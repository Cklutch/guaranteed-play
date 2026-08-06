from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st


def get_db_path():
    candidates = [
        Path("data/players.db"),
        Path("players.db"),
        Path("data/fantasy.db"),
        Path("fantasy.db"),
        Path("data/draft.db"),
        Path("draft.db"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_csv_path():
    candidates = [
        Path("data/processed/master_players.csv"),
        Path("data/rankings.csv"),
        Path("rankings.csv"),
        Path("data/draft_board.csv"),
        Path("draft_board.csv"),
    ]
    # Intentionally excludes "data/players.csv" / "players.csv" -- that path is
    # a hand-authored sample file (see data/samples/), not a real data source.
    # It must never become the app's main dataset by silent fallback.

    for path in candidates:
        if path.exists():
            return path

    return None


def safe_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _calculate_column_coverage(df, column_candidates):
    column = safe_col(df, column_candidates)
    if df.empty or column is None:
        return {
            "column": column,
            "coverage": 0.0,
            "non_null_count": 0,
        }

    non_null_count = int(df[column].notna().sum())
    coverage = round((non_null_count / len(df)) * 100.0, 2) if len(df) else 0.0

    return {
        "column": column,
        "coverage": coverage,
        "non_null_count": non_null_count,
    }


def get_connection():
    db_path = get_db_path()
    if db_path is None:
        return None
    return sqlite3.connect(db_path)


@st.cache_data(show_spinner=False)
def get_database_debug_info():
    info = {
        "source_type": None,
        "db_path": None,
        "csv_path": None,
        "db_exists": False,
        "csv_exists": False,
        "tables": [],
        "selected_table": None,
        "selected_table_columns": [],
        "row_count": 0,
        "projection_coverage": 0.0,
        "projection_column": None,
        "projection_non_null_count": 0,
        "tier_coverage": 0.0,
        "tier_column": None,
        "tier_non_null_count": 0,
        "error": None,
    }

    db_path = get_db_path()
    csv_path = get_csv_path()

    if csv_path is not None:
        info["source_type"] = "csv"
        info["csv_path"] = str(csv_path)
        info["csv_exists"] = True

        try:
            df = pd.read_csv(csv_path)
            projection_coverage = _calculate_column_coverage(
                df,
                ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"],
            )
            tier_coverage = _calculate_column_coverage(
                df,
                ["tier", "Tier", "TIERS"],
            )
            info["selected_table_columns"] = list(df.columns)
            info["row_count"] = int(len(df))
            info["projection_coverage"] = projection_coverage["coverage"]
            info["projection_column"] = projection_coverage["column"]
            info["projection_non_null_count"] = projection_coverage["non_null_count"]
            info["tier_coverage"] = tier_coverage["coverage"]
            info["tier_column"] = tier_coverage["column"]
            info["tier_non_null_count"] = tier_coverage["non_null_count"]
            return info
        except Exception as e:
            info["error"] = str(e)
            return info

    if db_path is not None:
        info["source_type"] = "sqlite"
        info["db_path"] = str(db_path)
        info["db_exists"] = True

        try:
            conn = sqlite3.connect(db_path)

            tables_df = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                conn
            )
            tables = tables_df["name"].tolist()
            info["tables"] = tables

            if not tables:
                info["error"] = "Database file found, but it contains no tables."
                conn.close()
                return info

            preferred_tables = ["players", "rankings", "player_rankings", "draft_rankings"]
            selected_table = None

            for table_name in preferred_tables:
                if table_name in tables:
                    selected_table = table_name
                    break

            if selected_table is None:
                selected_table = tables[0]

            info["selected_table"] = selected_table

            preview_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 5", conn)
            info["selected_table_columns"] = list(preview_df.columns)

            count_df = pd.read_sql_query(
                f"SELECT COUNT(*) AS row_count FROM {selected_table}",
                conn
            )
            info["row_count"] = int(count_df.iloc[0]["row_count"])

            full_df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            projection_coverage = _calculate_column_coverage(
                full_df,
                ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"],
            )
            tier_coverage = _calculate_column_coverage(
                full_df,
                ["tier", "Tier", "TIERS"],
            )
            info["projection_coverage"] = projection_coverage["coverage"]
            info["projection_column"] = projection_coverage["column"]
            info["projection_non_null_count"] = projection_coverage["non_null_count"]
            info["tier_coverage"] = tier_coverage["coverage"]
            info["tier_column"] = tier_coverage["column"]
            info["tier_non_null_count"] = tier_coverage["non_null_count"]

            conn.close()
            return info

        except Exception as e:
            info["error"] = str(e)
            return info

    info["error"] = "No SQLite or CSV player data file found in expected locations."
    return info


@st.cache_data(show_spinner=False)
def _load_players_df_cached(source_type, source_path, source_mtime):
    path = Path(source_path)

    if source_type == "sqlite":
        try:
            conn = sqlite3.connect(path)

            tables_df = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                conn
            )
            tables = tables_df["name"].tolist()

            if not tables:
                conn.close()
                return pd.DataFrame()

            preferred_tables = ["players", "rankings", "player_rankings", "draft_rankings"]
            selected_table = None

            for table_name in preferred_tables:
                if table_name in tables:
                    selected_table = table_name
                    break

            if selected_table is None:
                selected_table = tables[0]

            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            conn.close()
            return df

        except Exception:
            return pd.DataFrame()

    if source_type == "csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def load_players_df():
    csv_path = get_csv_path()
    db_path = get_db_path()

    if csv_path is not None:
        return _load_players_df_cached(
            "csv",
            str(csv_path),
            csv_path.stat().st_mtime,
        )

    if db_path is not None:
        return _load_players_df_cached(
            "sqlite",
            str(db_path),
            db_path.stat().st_mtime,
        )

    return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=32)
def _filter_available_players_cached(players_df, player_col, drafted_tuple):
    if players_df.empty or player_col is None:
        return pd.DataFrame()

    df = players_df.copy()
    drafted = set(drafted_tuple)
    if drafted:
        df = df[~df[player_col].astype(str).isin(drafted)].copy()

    return df.reset_index(drop=True)


def get_available_players_df():
    df = load_players_df().copy()
    if df.empty:
        return df

    player_col = safe_col(df, ["player_name", "Player", "player", "name", "full_name"])
    if player_col is None:
        return pd.DataFrame()

    drafted = tuple(st.session_state.get("drafted_players", []))
    return _filter_available_players_cached(df, player_col, drafted)
