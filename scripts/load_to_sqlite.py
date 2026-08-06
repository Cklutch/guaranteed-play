# Dev/demo script only. Loads hand-authored SAMPLE data into a sqlite db that
# is not read by the live Streamlit app (see draftkit/data_access.py::get_db_path
# candidates, which do not include guaranteed_play.db). Do not treat this data
# as real projections/ADP.
from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "samples" / "players_cleaned_SAMPLE_DO_NOT_USE.csv"
DB_PATH = PROJECT_ROOT / "guaranteed_play.db"
TABLE_NAME = "players"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file at: {DATA_PATH}")

    print(f"Reading player data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from CSV.")
    print("Columns found:")
    print(list(df.columns))

    conn = sqlite3.connect(str(DB_PATH))

    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        verification_df = pd.read_sql_query(
            f"SELECT * FROM {TABLE_NAME} LIMIT 10",
            conn
        )

        row_count = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {TABLE_NAME}",
            conn
        ).iloc[0]["row_count"]

        print(f"\nSuccessfully wrote table '{TABLE_NAME}' to: {DB_PATH}")
        print(f"Row count in SQLite table: {row_count}")
        print("\nPreview of first 10 rows from SQLite:")
        print(verification_df)

    finally:
        conn.close()
        print("\nSQLite connection closed.")


if __name__ == "__main__":
    main()