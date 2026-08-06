# Dev/demo script only. Loads hand-authored SAMPLE data into a sqlite db that
# is not read by the live Streamlit app (see draftkit/data_access.py::get_db_path
# candidates, which do not include guaranteed_play.db). Do not treat this data
# as real projections/ADP.
from pathlib import Path
import sqlite3
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CSV_PATH = PROJECT_ROOT / "data" / "samples" / "players_expanded_sample_SAMPLE_DO_NOT_USE.csv"
DB_PATH = PROJECT_ROOT / "guaranteed_play.db"

df = pd.read_csv(CSV_PATH)
print("CSV rows:", len(df))
print(df.head())

conn = sqlite3.connect(DB_PATH)
df.to_sql("players", conn, if_exists="replace", index=False)

check_df = pd.read_sql_query("SELECT * FROM players", conn)
print("Rows now in SQLite:", len(check_df))
print(check_df.head())

conn.close()
print(f"Loaded into {DB_PATH}")