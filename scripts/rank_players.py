# Dev/demo script only, over SAMPLE data (see scripts/load_to_sqlite.py). The
# value_score formula here (projection_points - adp * 10) is an arbitrary,
# unvalidated heuristic kept for historical reference -- do not port it into
# draftkit/draft_analysis.py or treat its output as a real ranking.
import sqlite3

conn = sqlite3.connect("../guaranteed_play.db")
cursor = conn.cursor()

cursor.execute("""
SELECT player_name, team, position, adp, projection_points,
       (projection_points - adp * 10) AS value_score
FROM players
ORDER BY value_score DESC
""")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()