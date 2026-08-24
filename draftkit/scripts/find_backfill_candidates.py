"""Real backfill-candidate scoping (rookie_data_backfill_plan.pdf's Task A,
Step 1 -- built fresh; the PDF referenced a find_backfill_candidates()
function that didn't exist anywhere in this repo, confirmed by a full-text
grep before writing this).

Cross-references the real 2024-2025 RB/WR draft classes against real
CONFIRMED archetype status (rb_archetypes.csv/wr_archetypes.csv) to find
who still needs backfilling -- cheaper than sourcing data for two entire
draft classes, since most players already clear the confirmed floor.

Real bug class avoided proactively: joins on
draftkit.data_pipeline.normalize_player_name() rather than exact string
match, per the real Marvin Harrison Jr./Tyrone Tracy Jr./Tre' Harris misses
already found this session from the same class of join.

Usage:
    python -m draftkit.scripts.find_backfill_candidates
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.data_pipeline import normalize_player_name  # noqa: E402

DRAFT_PICKS_CSV = REPO_ROOT / "research" / "validation_v1" / "data" / "raw_draft_picks.csv"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
RB_ARCHETYPES_CSV = REPO_ROOT / "data" / "processed" / "rb_archetypes.csv"
WR_ARCHETYPES_CSV = REPO_ROOT / "data" / "processed" / "wr_archetypes.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "backfill_candidates.csv"

BACKFILL_SEASONS = (2024, 2025)
SKILL_POSITIONS = ("RB", "WR")


def find_backfill_candidates() -> pd.DataFrame:
    draft = pd.read_csv(DRAFT_PICKS_CSV)
    draft["_key"] = draft["pfr_player_name"].apply(normalize_player_name)

    master = pd.read_csv(MASTER_CSV)
    master["_key"] = master["player_name"].apply(normalize_player_name)

    rb_arch = pd.read_csv(RB_ARCHETYPES_CSV)
    rb_arch["_key"] = rb_arch["player_name"].apply(normalize_player_name)
    wr_arch = pd.read_csv(WR_ARCHETYPES_CSV)
    wr_arch["_key"] = wr_arch["player_name"].apply(normalize_player_name)

    classes = draft[
        (draft["season"].isin(BACKFILL_SEASONS)) & (draft["position"].isin(SKILL_POSITIONS))
    ].copy()
    # Still-relevant filter -- same real convention as every other pool
    # this session (presence in master_players.csv), not every drafted
    # player who's since left the league/practice squad churn.
    classes = classes.merge(master[["_key", "player_name"]], on="_key", how="inner")
    classes = classes.rename(columns={"pick": "draft_pick", "team": "landing_team"})

    def confirmed_status(row) -> str:
        arch = rb_arch if row["position"] == "RB" else wr_arch
        primary_col = "archetype_primary" if row["position"] == "RB" else "wr_archetype_primary"
        match = arch[arch["_key"] == row["_key"]]
        if match.empty:
            return "no_row"
        return match.iloc[0][primary_col]

    classes["confirmed_status"] = classes.apply(confirmed_status, axis=1)
    candidates = classes[classes["confirmed_status"].isin(["unconfirmed", "no_row"])].copy()

    return candidates[[
        "season", "draft_pick", "player_name", "position", "college", "landing_team", "confirmed_status",
    ]].sort_values(["season", "draft_pick"]).reset_index(drop=True)


def main() -> int:
    df = find_backfill_candidates()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(df)} real backfill candidate(s)")
    print(df["confirmed_status"].value_counts().to_string())
    print(df.groupby("season")["position"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
