from pathlib import Path
import sqlite3

db_path = Path("guaranteed_play.db").resolve()
print("DB path:", db_path)
print("Exists:", db_path.exists())
print("Size:", db_path.stat().st_size if db_path.exists() else "missing")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = cursor.fetchall()
print("Tables:", tables)

conn.close()