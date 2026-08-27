"""Overlay real Sleeper Half-PPR ADP onto master_players.csv (2026-08-27).

master_players.csv's `adp` column is Underdog-sourced (see draftkit/
data_pipeline.py's load_underdog_adp()). Per explicit user direction, this
overlays real Sleeper Half-PPR ADP -- from a multi-source sheet the user
supplied directly (JuiceBoxOne's 2026 rankings, copied to
data/raw/juicebox_2026_adp_sources.csv) -- onto every player the sheet
covers (~200 players across QB/RB/WR/TE, not just WR), so the WHOLE app's
ranking, tiers, and Base Value market component reflect Sleeper's own
market rather than Underdog's for those players. Anyone NOT in the sheet
keeps their existing Underdog adp untouched -- this is an overlay, not a
wholesale replacement, since the sheet doesn't cover the full player pool.

adp_rank is recomputed after the overlay (rank of the now-mixed-source adp
column across the whole pool) since Base Value's market component and the
board's ADP Rk column are rank-based, not raw-value-based -- changing some
players' adp shifts their neighbors' relative rank too even when the
neighbor's own number didn't change.

Also computes sleeper_value_gap = Average - Sleeper Half (the sheet's own
consensus-across-4-platforms column vs. Sleeper specifically): positive
means Sleeper drafts him EARLIER than the wider market (a Sleeper-specific
reach/bust risk), negative means Sleeper drafts him LATER (a Sleeper-
specific bargain). Feeds a new "Sleeper value" badge, separate from the
existing sportsbook-vs-ADP Market badge -- this compares one platform's
ADP against a cross-platform consensus, not the model's own projection
against ADP.

Idempotent: re-running always starts from a fresh backup of the CURRENT
master_players.csv (not a backup of a backup), and always recomputes from
the sheet rather than compounding a prior overlay.

Run: python draftkit/scripts/apply_sleeper_adp_overlay.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from draftkit.dads_scoring import _norm

MASTER_PLAYERS_PATH = PROJECT_ROOT / "data" / "processed" / "master_players.csv"
ADP_SHEET_PATH = PROJECT_ROOT / "data" / "raw" / "juicebox_2026_adp_sources.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "processed" / "master_players.pre_sleeper_overlay.csv"

POSITION_MAP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}


def main() -> int:
    master = pd.read_csv(MASTER_PLAYERS_PATH, low_memory=False)
    shutil.copy(MASTER_PLAYERS_PATH, BACKUP_PATH)

    sheet = pd.read_csv(ADP_SHEET_PATH)
    sheet = sheet[sheet["Pos"].isin(POSITION_MAP)].copy()
    sheet = sheet[sheet["Sleeper Half"].notna()].copy()
    sheet["_key"] = sheet["Pos"].map(POSITION_MAP) + "|" + sheet["Name"].apply(_norm)
    sheet = sheet.drop_duplicates("_key")

    master["_key"] = master["position"].astype(str).str.upper() + "|" + master["player_name"].apply(_norm)

    overlay = sheet.set_index("_key")[["Sleeper Half", "Average"]]
    matched = master["_key"].isin(overlay.index)
    print(f"Sheet rows (QB/RB/WR/TE, Sleeper Half present): {len(sheet)}")
    print(f"Matched into master_players.csv: {matched.sum()} of {len(sheet)}")
    unmatched = set(overlay.index) - set(master["_key"])
    if unmatched:
        print(f"Unmatched sheet entries ({len(unmatched)}): {sorted(k.split('|')[1] for k in unmatched)}")

    before = master.loc[matched, ["player_name", "adp"]].copy()

    master.loc[matched, "adp"] = master.loc[matched, "_key"].map(overlay["Sleeper Half"])
    master["sleeper_value_gap"] = master["_key"].map(overlay["Average"] - overlay["Sleeper Half"])

    # adp_rank is rank-based (Base Value's market component and the board's
    # ADP Rk column both consume it), so recompute across the WHOLE pool --
    # a few players' adp values changing shifts their neighbors' relative
    # rank too, even for players whose own number didn't move.
    master["adp_rank"] = pd.to_numeric(master["adp"], errors="coerce").rank(method="min")

    after = master.loc[matched, ["player_name", "adp"]].copy()
    diff = before.merge(after, on="player_name", suffixes=("_old", "_new"))
    diff["delta"] = diff["adp_new"] - diff["adp_old"]
    movers = diff.reindex(diff["delta"].abs().sort_values(ascending=False).index).head(15)
    print("\nBiggest ADP changes (old Underdog adp -> new Sleeper Half adp):")
    print(movers.to_string(index=False))

    master = master.drop(columns=["_key"])
    master.to_csv(MASTER_PLAYERS_PATH, index=False)
    print(f"\nBackup of pre-overlay master_players.csv: {BACKUP_PATH}")
    print(f"Written: {MASTER_PLAYERS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
