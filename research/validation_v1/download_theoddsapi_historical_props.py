"""Download a limited The Odds API historical Week 1 player-prop pilot.

Research-only helper for sportsbook-implied fantasy validation. This script
does not recommend bets, scrape sportsbooks, connect to accounts, or use
undocumented endpoints.

Gate status:
- True preseason season-long props: not supported by the documented The Odds API
  NFL player-prop markets reviewed for this project.
- Pre-Week-1 game props: supported as a limited pilot for historical event odds
  snapshots after 2023-05-03T05:30:00Z, provided the requested snapshot is before
  each Week 1 game's commence_time.

The script writes raw API responses under source_sportsbook_raw/theoddsapi and
normalizes only documented event player-prop markets into historical_sportsbook_props.csv.
Rows include safety columns so downstream validation can exclude unsafe rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[1]
RAW_DIR = VALIDATION_DIR / "source_sportsbook_raw" / "theoddsapi"
OUTPUT_CSV = VALIDATION_DIR / "historical_sportsbook_props.csv"
API_BASE = "https://api.the-odds-api.com/v4"

DEFAULT_MARKETS = [
    "player_receptions",
    "player_reception_yds",
    "player_reception_tds",
    "player_rush_attempts",
    "player_rush_yds",
    "player_rush_tds",
    "player_rush_reception_yds",
    "player_rush_reception_tds",
    "player_anytime_td",
]

MARKET_MAP = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_reception_tds": "receiving_tds",
    "player_rush_attempts": "rushing_attempts",
    "player_rush_yds": "rushing_yards",
    "player_rush_tds": "rushing_tds",
    "player_rush_reception_yds": "rush_reception_yards",
    "player_rush_reception_tds": "rush_reception_tds",
    "player_anytime_td": "anytime_td",
}

OUTPUT_COLUMNS = [
    "season",
    "player_name",
    "position",
    "team",
    "opponent",
    "market",
    "market_type",
    "line",
    "sportsbook",
    "odds",
    "odds_format",
    "over_odds",
    "under_odds",
    "snapshot_date",
    "source",
    "source_url_or_file",
    "is_season_long",
    "is_preseason_snapshot",
    "commence_time",
    "week",
    "is_pre_week1_snapshot",
    "snapshot_before_commence_time",
    "sportsbook_data_safety_status",
]


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clean_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    keep = []
    for char in text:
        if char.isalnum() or char.isspace():
            keep.append(char)
    return " ".join("".join(keep).split())


def api_get(path: str, params: dict[str, str], api_key: str) -> tuple[dict[str, Any], str]:
    query = dict(params)
    query["apiKey"] = api_key
    url = f"{API_BASE}{path}?{urlencode(query)}"
    display_url = f"{API_BASE}{path}?{urlencode({**params, 'apiKey': 'REDACTED'})}"
    request = Request(url, headers={"User-Agent": "GuaranteedPlayResearch/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload, display_url


def save_raw(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_position_lookup() -> dict[tuple[int, str], str]:
    lookup: dict[tuple[int, str], str] = {}
    candidates = [
        VALIDATION_DIR / "predraft_validation_dataset_sportsbook.csv",
        VALIDATION_DIR / "predraft_validation_dataset_expanded.csv",
        VALIDATION_DIR / "predraft_validation_dataset.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                position = str(row.get("position", "")).upper()
                if position not in {"WR", "RB"}:
                    continue
                try:
                    season = int(float(str(row.get("season", ""))))
                except ValueError:
                    continue
                key = clean_name(row.get("player_name"))
                if key:
                    lookup[(season, key)] = position
    return lookup


def combine_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any], dict[str, Any]] = defaultdict(dict)
    for outcome in outcomes:
        player = outcome.get("description") or outcome.get("name")
        side = str(outcome.get("name", "")).lower()
        point = outcome.get("point")
        entry = grouped[(str(player or ""), point)]
        entry["player_name"] = player
        entry["line"] = point
        if side == "over" or side == "yes":
            entry["over_odds"] = outcome.get("price")
        elif side == "under" or side == "no":
            entry["under_odds"] = outcome.get("price")
        else:
            entry["odds"] = outcome.get("price")
    return list(grouped.values())


def normalize_event_odds(
    payload: dict[str, Any],
    *,
    season: int,
    week: int,
    source_url: str,
    position_lookup: dict[tuple[int, str], str],
) -> list[dict[str, Any]]:
    rows = []
    snapshot = payload.get("timestamp")
    data = payload.get("data") or {}
    snapshot_dt = parse_iso(snapshot)
    commence_time = data.get("commence_time")
    commence_dt = parse_iso(commence_time)
    snapshot_before_commence = bool(snapshot_dt and commence_dt and snapshot_dt < commence_dt)
    is_pre_week1 = bool(week == 1 and snapshot_before_commence)
    status = "pre_week1_game_prop" if is_pre_week1 else "unsafe_inseason_or_post_kickoff"
    home_team = data.get("home_team") or ""
    away_team = data.get("away_team") or ""

    for bookmaker in data.get("bookmakers", []) or []:
        sportsbook = bookmaker.get("title") or bookmaker.get("key") or ""
        for market in bookmaker.get("markets", []) or []:
            api_market = market.get("key")
            normalized_market = MARKET_MAP.get(api_market)
            if not normalized_market:
                continue
            for item in combine_outcomes(market.get("outcomes", []) or []):
                player_name = str(item.get("player_name") or "").strip()
                if not player_name:
                    continue
                position = position_lookup.get((season, clean_name(player_name)), "UNKNOWN")
                over_odds = item.get("over_odds")
                under_odds = item.get("under_odds")
                odds = over_odds if over_odds not in {None, ""} else item.get("odds")
                rows.append(
                    {
                        "season": season,
                        "player_name": player_name,
                        "position": position,
                        "team": "",
                        "opponent": f"{away_team} @ {home_team}",
                        "market": normalized_market,
                        "market_type": "player",
                        "line": item.get("line"),
                        "sportsbook": sportsbook,
                        "odds": odds,
                        "odds_format": "american",
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "snapshot_date": snapshot,
                        "source": "the_odds_api",
                        "source_url_or_file": source_url,
                        "is_season_long": "false",
                        "is_preseason_snapshot": str(is_pre_week1).lower(),
                        "commence_time": commence_time,
                        "week": week,
                        "is_pre_week1_snapshot": str(is_pre_week1).lower(),
                        "snapshot_before_commence_time": str(snapshot_before_commence).lower(),
                        "sportsbook_data_safety_status": status,
                    }
                )
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download The Odds API historical Week 1 player-prop pilot.")
    parser.add_argument("--season", type=int, required=True, help="NFL season, for example 2024.")
    parser.add_argument("--snapshot", required=True, help="Pre-kickoff ISO timestamp, for example 2024-09-04T12:00:00Z.")
    parser.add_argument("--commence-from", required=True, help="Week 1 start ISO timestamp.")
    parser.add_argument("--commence-to", required=True, help="Week 1 end ISO timestamp.")
    parser.add_argument("--regions", default="us", help="The Odds API regions parameter.")
    parser.add_argument("--bookmakers", default="", help="Optional comma-separated bookmaker keys.")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated player-prop market keys.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between event odds calls.")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Normalized CSV output path.")
    args = parser.parse_args()

    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("ERROR: Set THE_ODDS_API_KEY before running this downloader.", file=sys.stderr)
        return 2

    snapshot_dt = parse_iso(args.snapshot)
    if snapshot_dt is None:
        print("ERROR: --snapshot must be an ISO timestamp.", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    position_lookup = load_position_lookup()

    event_params = {
        "date": args.snapshot,
        "dateFormat": "iso",
        "commenceTimeFrom": args.commence_from,
        "commenceTimeTo": args.commence_to,
    }
    events_payload, events_url = api_get("/historical/sports/americanfootball_nfl/events", event_params, api_key)
    save_raw(events_payload, RAW_DIR / f"{args.season}_week1_events_{args.snapshot.replace(':', '')}.json")
    events = events_payload.get("data") or []

    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        odds_params = {
            "date": args.snapshot,
            "dateFormat": "iso",
            "oddsFormat": "american",
            "markets": args.markets,
        }
        if args.bookmakers:
            odds_params["bookmakers"] = args.bookmakers
        else:
            odds_params["regions"] = args.regions
        odds_payload, odds_url = api_get(
            f"/historical/sports/americanfootball_nfl/events/{event_id}/odds",
            odds_params,
            api_key,
        )
        save_raw(odds_payload, RAW_DIR / f"{args.season}_week1_event_{event_id}_{args.snapshot.replace(':', '')}.json")
        rows.extend(
            normalize_event_odds(
                odds_payload,
                season=args.season,
                week=1,
                source_url=odds_url,
                position_lookup=position_lookup,
            )
        )
        time.sleep(args.sleep)

    safe_rows = [row for row in rows if row["sportsbook_data_safety_status"] == "pre_week1_game_prop"]
    write_rows(safe_rows, Path(args.output))

    print("BUILD_STATUS: built_preseason_sportsbook_pilot")
    print("gate passed: Gate B - historical pre-Week-1 game props")
    print(f"events endpoint: {events_url}")
    print(f"events found: {len(events)}")
    print(f"safe rows written: {len(safe_rows)}")
    print(f"unsafe rows excluded: {len(rows) - len(safe_rows)}")
    print(f"output: {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
