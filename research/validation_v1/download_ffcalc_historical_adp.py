from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from validation_utils import VALIDATION_DIR

SOURCE_NAME = "FantasyFootballCalculator"
BASE_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?position=all&teams={teams}&year={season}"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "GuaranteedPlayResearch/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def season_rows(payload: dict, season: int, scoring: str) -> list[dict]:
    meta = payload.get("meta", {}) or {}
    rows = []
    for player in payload.get("players", []) or []:
        position = str(player.get("position", "")).upper()
        if position not in {"WR", "RB", "QB", "TE"}:
            continue
        adp = player.get("adp")
        row = {
            "season": season,
            "player_name": player.get("name"),
            "position": position,
            "overall_adp": adp,
            "positional_adp": None,
            "adp_source": SOURCE_NAME,
            "source_scoring": scoring,
            "source_teams": meta.get("teams"),
            "source_rounds": meta.get("rounds"),
            "source_total_drafts": meta.get("total_drafts"),
            "source_start_date": meta.get("start_date"),
            "source_end_date": meta.get("end_date"),
            "source_player_id": player.get("player_id"),
            "source_team": player.get("team"),
            "source_times_drafted": player.get("times_drafted"),
            "source_high": player.get("high"),
            "source_low": player.get("low"),
            "source_stdev": player.get("stdev"),
        }
        if scoring == "ppr":
            row["ppr_adp"] = adp
        elif scoring == "half-ppr":
            row["half_ppr_adp"] = adp
        else:
            row["standard_adp"] = adp
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download archived Fantasy Football Calculator ADP for validation_v1.")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--scoring", choices=["ppr", "half-ppr", "standard"], default="ppr")
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--output-dir", default=str(VALIDATION_DIR / "source_adp_raw"))
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_rows = 0
    for season in range(args.start_season, args.end_season + 1):
        url = BASE_URL.format(scoring=args.scoring, teams=args.teams, season=season)
        payload = fetch_json(url)
        if payload.get("status") != "Success":
            raise RuntimeError(f"{season} request failed: {payload}")
        rows = season_rows(payload, season, args.scoring)
        out_path = output_dir / f"ffcalc_{args.scoring.replace('-', '_')}_{season}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        meta = payload.get("meta", {}) or {}
        manifest.append({
            "season": season,
            "rows": len(rows),
            "status": payload.get("status"),
            "source_url": url,
            "scoring": args.scoring,
            "teams": args.teams,
            "start_date": meta.get("start_date"),
            "end_date": meta.get("end_date"),
            "total_drafts": meta.get("total_drafts"),
            "output_file": str(out_path),
        })
        total_rows += len(rows)
        print(f"{season}: {len(rows)} WR/RB/QB/TE rows, window {meta.get('start_date')} to {meta.get('end_date')}")
        if args.sleep:
            time.sleep(args.sleep)

    manifest_path = output_dir / f"ffcalc_{args.scoring.replace('-', '_')}_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Manifest written: {manifest_path}")
    print(f"Total WR/RB/QB/TE rows: {total_rows}")


if __name__ == "__main__":
    main()
