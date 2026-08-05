from pathlib import Path
import sqlite3
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CSV_PATH = PROJECT_ROOT / "data" / "players_expanded_sample.csv"
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