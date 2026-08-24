"""One-time, manually-run puller for current-season NFL player props.

Research/analytics use only. This feeds Guaranteed Play's existing sportsbook
projection and market-disagreement signals (draftkit/projection_engine.py,
draftkit/market_disagreement.py) with real prop lines. It does not recommend
wagers, place bets, scrape sportsbook pages, or connect to any sportsbook
account -- it calls The Odds API's documented endpoints only.

There is no scheduler wired to this script. Run it by hand; re-run it by hand
whenever you want fresher lines. Every API call's response headers are
printed so remaining monthly credit quota is always visible.

Usage:
    python -m draftkit.scripts.pull_sportsbook_props --dry-run
    python -m draftkit.scripts.pull_sportsbook_props \
        --commence-from 2026-09-07T00:00:00Z --commence-to 2026-09-14T00:00:00Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(__file__).resolve().parent / "source_sportsbook_raw" / "theoddsapi"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"
TOKEN_FILE = REPO_ROOT / ".odds_api.token"
API_BASE = "https://api.the-odds-api.com/v4"

# Markets with a slot in draftkit's production loader vocabulary
# (draftkit/data_sources/*_source.py -> _normalize_market_type).
DEFAULT_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_tds",
    "player_reception_yds",
    "player_reception_tds",
    "player_receptions",
]

MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_rush_yds": "rushing_yards",
    "player_rush_tds": "rushing_tds",
    "player_reception_yds": "receiving_yards",
    "player_reception_tds": "receiving_tds",
    "player_receptions": "receptions",
}

UNSUPPORTED_MARKETS_NOTE = (
    "Not pulled -- no slot in draftkit's production market vocabulary: "
    "player_anytime_td, player_rush_attempts, player_rush_reception_yds, "
    "player_rush_reception_tds, player_pass_attempts, player_pass_completions, "
    "player_pass_interceptions."
)

DEFAULT_BOOKMAKERS = ["draftkings", "fanduel", "pinnacle"]

BOOKMAKER_KEY_TO_FILE = {
    "draftkings": "draftkings_props.csv",
    "fanduel": "fanduel_props.csv",
    "pinnacle": "pinnacle_props.csv",
}

BOOKMAKER_KEY_TO_LABEL = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "pinnacle": "Pinnacle",
}

PROP_COLUMNS = ["player_name", "market_type", "line_value", "sportsbook"]


def resolve_api_key() -> str | None:
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def api_get(path: str, params: dict, api_key: str):
    query = dict(params)
    query["apiKey"] = api_key
    url = f"{API_BASE}{path}?{urlencode(query)}"
    display_url = f"{API_BASE}{path}?{urlencode({**params, 'apiKey': 'REDACTED'})}"
    request = Request(url, headers={"User-Agent": "GuaranteedPlay/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = response.headers
    return payload, display_url, headers


def print_quota(headers, label: str) -> None:
    remaining = headers.get("x-requests-remaining")
    used = headers.get("x-requests-used")
    print(f"[quota] after {label}: remaining={remaining} used={used}")


def save_raw(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def check_quota(api_key: str):
    payload, url, headers = api_get("/sports", {}, api_key)
    print(f"[quota check] {url}")
    print_quota(headers, "quota check")
    return headers


def list_events(api_key: str, commence_from: str | None, commence_to: str | None):
    params = {}
    if commence_from:
        params["commenceTimeFrom"] = commence_from
    if commence_to:
        params["commenceTimeTo"] = commence_to
    payload, url, headers = api_get("/sports/americanfootball_nfl/events", params, api_key)
    print(f"[events] {url}")
    print_quota(headers, "events list")
    return payload or [], headers


def fetch_event_odds(api_key: str, event_id: str, markets: list[str], bookmakers: list[str]):
    params = {
        "markets": ",".join(markets),
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    return api_get(f"/sports/americanfootball_nfl/events/{event_id}/odds", params, api_key)


def combine_outcomes(outcomes: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = defaultdict(dict)
    for outcome in outcomes:
        player = outcome.get("description") or outcome.get("name")
        side = str(outcome.get("name", "")).lower()
        point = outcome.get("point")
        entry = grouped[(str(player or ""), point)]
        entry["player_name"] = player
        entry["line"] = point
        if side in {"over", "yes"}:
            entry["over_odds"] = outcome.get("price")
        elif side in {"under", "no"}:
            entry["under_odds"] = outcome.get("price")
        else:
            entry["odds"] = outcome.get("price")
    return list(grouped.values())


def normalize_event_odds(payload) -> list[dict]:
    rows = []
    data = payload or {}
    for bookmaker in data.get("bookmakers", []) or []:
        book_key = str(bookmaker.get("key") or "").lower()
        if book_key not in BOOKMAKER_KEY_TO_LABEL:
            continue
        for market in bookmaker.get("markets", []) or []:
            market_type = MARKET_MAP.get(market.get("key"))
            if not market_type:
                continue
            for item in combine_outcomes(market.get("outcomes", []) or []):
                player_name = str(item.get("player_name") or "").strip()
                line_value = item.get("line")
                if not player_name or line_value is None:
                    continue
                rows.append({
                    "player_name": player_name,
                    "market_type": market_type,
                    "line_value": line_value,
                    "bookmaker_key": book_key,
                })
    return rows


def write_book_csv(rows: list[dict], label: str, out_path: Path) -> None:
    if rows:
        df = pd.DataFrame(rows)[["player_name", "market_type", "line_value"]].copy()
        df["sportsbook"] = label
    else:
        df = pd.DataFrame(columns=PROP_COLUMNS)
    df = df[PROP_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def validate_pull(rows_by_book: dict) -> dict:
    total = sum(len(v) for v in rows_by_book.values())
    books_with_data = [BOOKMAKER_KEY_TO_LABEL[k] for k, v in rows_by_book.items() if v]
    messages = []
    if total == 0:
        messages.append("No prop rows were collected from any bookmaker.")
    return {
        "is_valid": total > 0,
        "row_count": total,
        "books_with_data": books_with_data,
        "messages": messages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull current NFL player props from The Odds API into draftkit's data/raw CSVs."
    )
    parser.add_argument("--commence-from", default=None, help="ISO lower bound for event commence_time.")
    parser.add_argument("--commence-to", default=None, help="ISO upper bound for event commence_time.")
    parser.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated Odds API market keys.")
    parser.add_argument("--bookmakers", default=",".join(DEFAULT_BOOKMAKERS), help="Comma-separated bookmaker keys.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between per-event calls.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for the three _props.csv files.")
    parser.add_argument("--dry-run", action="store_true", help="List events and estimated call count; write nothing.")
    args = parser.parse_args()

    api_key = resolve_api_key()
    if not api_key:
        print(
            "ERROR: Set THE_ODDS_API_KEY (env var) or create a gitignored .odds_api.token "
            "file at the repo root before running this script.",
            file=sys.stderr,
        )
        return 2

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    bookmakers = [b.strip().lower() for b in args.bookmakers.split(",") if b.strip()]

    unknown_markets = [m for m in markets if m not in MARKET_MAP]
    if unknown_markets:
        print(
            f"WARNING: these requested markets have no production mapping and will be "
            f"fetched (spending credits) but dropped during normalization: {unknown_markets}"
        )

    try:
        check_quota(api_key)
        events, _ = list_events(api_key, args.commence_from, args.commence_to)
    except HTTPError as exc:
        print(f"ERROR: quota/events check failed: HTTP {exc.code} ({exc.reason})", file=sys.stderr)
        return 2

    print(f"[events] found {len(events)} upcoming event(s) in window.")

    if args.dry_run:
        print(f"[dry-run] would spend up to {len(events)} additional odds-endpoint call(s) "
              f"(1 combined call per event covering all {len(markets)} market(s) and "
              f"{len(bookmakers)} bookmaker(s)).")
        print(f"[dry-run] {UNSUPPORTED_MARKETS_NOTE}")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_book: dict[str, list[dict]] = defaultdict(list)
    skipped_events = []

    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        try:
            payload, url, headers = fetch_event_odds(api_key, event_id, markets, bookmakers)
        except HTTPError as exc:
            print(f"[skip] event {event_id}: HTTP {exc.code} ({exc.reason})")
            skipped_events.append({"event_id": event_id, "error": f"HTTP {exc.code}"})
            time.sleep(args.sleep)
            continue
        except URLError as exc:
            print(f"[skip] event {event_id}: {exc.reason}")
            skipped_events.append({"event_id": event_id, "error": str(exc.reason)})
            time.sleep(args.sleep)
            continue

        print_quota(headers, f"event {event_id}")
        save_raw(payload, RAW_DIR / f"{event_id}.json")

        for row in normalize_event_odds(payload):
            rows_by_book[row["bookmaker_key"]].append(row)

        time.sleep(args.sleep)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for book_key, filename in BOOKMAKER_KEY_TO_FILE.items():
        label = BOOKMAKER_KEY_TO_LABEL[book_key]
        book_rows = rows_by_book.get(book_key, [])
        out_path = output_dir / filename
        write_book_csv(book_rows, label, out_path)
        total_rows += len(book_rows)
        market_counts = pd.Series([r["market_type"] for r in book_rows]).value_counts().to_dict() if book_rows else {}
        player_count = len({r["player_name"] for r in book_rows})
        print(f"[write] {out_path}: {len(book_rows)} row(s), {player_count} player(s), markets={market_counts}")

    validation = validate_pull(rows_by_book)

    print("BUILD_STATUS: sportsbook_props_pull_complete")
    print(f"events processed: {len(events)}")
    print(f"events skipped: {len(skipped_events)}")
    if skipped_events:
        print(f"skipped detail: {skipped_events}")
    print(f"total rows written: {total_rows}")
    print(f"validation: {validation}")
    print(f"[note] {UNSUPPORTED_MARKETS_NOTE}")

    return 0 if validation["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
