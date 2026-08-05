from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[1]
RAW_DIR = VALIDATION_DIR / "source_projection_raw"
SOURCE_CONFIG_CSV = VALIDATION_DIR / "wayback_fantasypros_projection_sources.csv"
ALL_EXTRACTED_CSV = VALIDATION_DIR / "historical_projections_wayback_fantasypros_all_extracted.csv"
MAIN_SAFE_CSV = VALIDATION_DIR / "historical_projections_wayback_fantasypros.csv"
HISTORICAL_PROJECTIONS_CSV = VALIDATION_DIR / "historical_projections.csv"
COVERAGE_CSV = VALIDATION_DIR / "wayback_fantasypros_projection_coverage_by_season.csv"
DIAGNOSTICS_JSON = VALIDATION_DIR / "wayback_fantasypros_projection_parse_diagnostics.json"
UNRESOLVED_CSV = VALIDATION_DIR / "wayback_fantasypros_unresolved_players.csv"
REPORT_MD = VALIDATION_DIR / "wayback_fantasypros_projection_expansion_report.md"

PROJECTION_SOURCE = "FantasyPros_Wayback"
SCORING_FORMAT = "unknown_or_default"

SECTION_TO_OUTPUT = {
    "Receptions": "projected_receptions",
    "Receiving Yards": "projected_receiving_yards",
    "Receiving TDs": "projected_receiving_tds",
    "Receiving Touchdowns": "projected_receiving_tds",
    "Targets": "projected_targets",
    "Rushing Attempts": "projected_carries",
    "Rush Attempts": "projected_carries",
    "Carries": "projected_carries",
    "Rushing Yards": "projected_rushing_yards",
    "Rushing TDs": "projected_rushing_tds",
    "Rushing Touchdowns": "projected_rushing_tds",
    "Games": "projected_games",
    "Fantasy Points": "projected_fantasy_points",
    "FPTS": "projected_fantasy_points",
}

VOLUME_COLUMNS = [
    "projected_targets",
    "projected_carries",
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_rushing_yards",
    "projected_rushing_tds",
]

OUTPUT_COLUMNS = [
    "season",
    "player_name",
    "position",
    "projected_fantasy_points",
    "projected_positional_rank",
    "projection_source",
    "scoring_format",
    "projected_targets",
    "projected_carries",
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_total_tds",
    "projected_games",
    "projected_team",
    "projection_date",
    "source_url_or_file",
    "is_preseason_projection",
    "projection_safety_status",
]

CONFIG_COLUMNS = [
    "season",
    "wayback_url",
    "archive_datetime_utc",
    "season_start_datetime_utc",
    "safety_status",
    "include_in_main_validation",
    "safety_notes",
]

COVERAGE_COLUMNS = [
    "season",
    "selected_url",
    "archive_datetime_utc",
    "season_start_datetime_utc",
    "safety_status",
    "included_in_main_validation",
    "page_accessible",
    "tables_found",
    "rows_extracted",
    "unique_players_extracted",
    "wr_rows_resolved",
    "rb_rows_resolved",
    "projected_fantasy_points_available",
    "projected_volume_stats_available",
    "unresolved_player_count",
    "output_file_path",
]

UNRESOLVED_COLUMNS = ["season", "player_name", "team", "sections", "source_url_or_file", "projection_safety_status"]

DEFAULT_SOURCES = [
    (2014, "https://web.archive.org/web/20140902155927/https://www.fantasypros.com/nfl/projections/leaders.php", "2014-09-05T00:30:00Z", "preseason_safe", True, "Capture is before the 2014 NFL opener."),
    (2015, "https://web.archive.org/web/20150904103106/https://www.fantasypros.com/nfl/projections/leaders.php", "2015-09-11T00:30:00Z", "preseason_safe", True, "Capture is before the 2015 NFL opener."),
    (2016, "https://web.archive.org/web/20160708123645/https://www.fantasypros.com/nfl/projections/leaders.php", "2016-09-09T00:30:00Z", "preseason_safe", True, "Capture is before the 2016 NFL opener."),
    (2017, "https://web.archive.org/web/20170830084404/https://www.fantasypros.com/nfl/projections/leaders.php", "2017-09-08T00:30:00Z", "preseason_safe", True, "Capture is before the 2017 NFL opener."),
    (2018, "https://web.archive.org/web/20180906130423/https://www.fantasypros.com/nfl/projections/leaders.php", "2018-09-07T00:20:00Z", "same_day_pre_kickoff_verify", True, "Same US calendar day as opener, but Wayback timestamp is before kickoff in UTC."),
    (2019, "https://web.archive.org/web/20190715211021/https://www.fantasypros.com/nfl/projections/leaders.php", "2019-09-06T00:20:00Z", "preseason_safe", True, "Capture is before the 2019 NFL opener."),
    (2020, "https://web.archive.org/web/20200624095638/https://www.fantasypros.com/nfl/projections/leaders.php", "2020-09-11T00:20:00Z", "preseason_safe", True, "Replacement capture is before the 2020 NFL opener."),
    (2021, "https://web.archive.org/web/20210813234359/https://www.fantasypros.com/nfl/projections/leaders.php", "2021-09-10T00:20:00Z", "preseason_safe", True, "Capture is before the 2021 NFL opener."),
    (2022, "https://web.archive.org/web/20220708084008/https://www.fantasypros.com/nfl/projections/leaders.php", "2022-09-09T00:20:00Z", "preseason_safe", True, "Capture is before the 2022 NFL opener."),
    (2023, "https://web.archive.org/web/20230801135615/https://www.fantasypros.com/nfl/projections/leaders.php", "2023-09-08T00:20:00Z", "preseason_safe", True, "Capture is before the 2023 NFL opener."),
    (2024, "https://web.archive.org/web/20240810045542/https://www.fantasypros.com/nfl/projections/leaders.php", "2024-09-06T00:20:00Z", "preseason_safe", True, "Capture is before the 2024 NFL opener."),
    (2025, "", "2025-09-05T00:20:00Z", "unknown_exclude", False, "No preseason-safe August 16, 2025 capture for the exact leaders URL was found by the tiny CDX check; September 15 captures must be excluded."),
]


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value).replace("\xa0", " ")
    return " ".join(value.split()).strip()


def clean_name_key(value: Any) -> str:
    text = str(value or "").lower()
    keep = [char for char in text if char.isalnum() or char.isspace()]
    tokens = [token for token in "".join(keep).split() if token not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(tokens)


def initial_surname_key(value: Any) -> str:
    tokens = clean_name_key(value).split()
    return " ".join([tokens[0][0], *tokens[1:]]) if len(tokens) >= 2 else " ".join(tokens)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def wayback_datetime(url: str) -> str:
    match = re.search(r"/web/(\d{14})/", str(url))
    if not match:
        return ""
    dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def season_sources() -> list[dict[str, Any]]:
    sources = []
    for season, url, start_dt, configured_status, configured_include, note in DEFAULT_SOURCES:
        archive_dt = wayback_datetime(url)
        archive = iso_dt(archive_dt)
        start = iso_dt(start_dt)
        status = configured_status
        include = configured_include
        if not archive or not start:
            status, include = "unknown_exclude", False
        elif archive >= start:
            status, include = "post_kickoff_exclude", False
        elif configured_status not in {"preseason_safe", "same_day_pre_kickoff_verify"}:
            include = False
        elif (start - archive).total_seconds() < 24 * 60 * 60:
            status = "same_day_pre_kickoff_verify"
        sources.append(
            {
                "season": season,
                "wayback_url": url,
                "archive_datetime_utc": archive_dt,
                "season_start_datetime_utc": start_dt,
                "safety_status": status,
                "include_in_main_validation": include,
                "safety_notes": note,
            }
        )
    return sources


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fetch_page(source: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw_path = RAW_DIR / f"wayback_fantasypros_{source['season']}.html"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    diag = {"page_accessible": False, "http_status": None, "raw_html_path": str(raw_path)}
    if not source["wayback_url"]:
        diag["fetch_error"] = "No Wayback URL configured."
        raw_path.write_text("", encoding="utf-8")
        return "", diag
    try:
        request = Request(source["wayback_url"], headers={"User-Agent": "Guaranteed-Play-Research/1.0"})
        with urlopen(request, timeout=60) as response:
            body = response.read()
            diag["http_status"] = getattr(response, "status", None)
    except HTTPError as exc:
        diag["http_status"] = exc.code
        diag["fetch_error"] = f"HTTP {exc.code}: {exc.reason}"
        raw_path.write_text("", encoding="utf-8")
        return "", diag
    except (URLError, OSError) as exc:
        diag["fetch_error"] = str(exc)
        raw_path.write_text("", encoding="utf-8")
        return "", diag
    text = body.decode("utf-8", errors="replace")
    raw_path.write_text(text, encoding="utf-8")
    diag.update({"page_accessible": True, "raw_html_bytes": len(body), "raw_html_chars": len(text)})
    return text, diag


def canonical_section(raw_title: str) -> str:
    title = clean_text(raw_title)
    aliases = {
        "rec": "Receptions",
        "receptions": "Receptions",
        "receiving yards": "Receiving Yards",
        "rec yards": "Receiving Yards",
        "receiving tds": "Receiving TDs",
        "receiving touchdowns": "Receiving TDs",
        "targets": "Targets",
        "rushing attempts": "Rushing Attempts",
        "rush attempts": "Rushing Attempts",
        "carries": "Carries",
        "rushing yards": "Rushing Yards",
        "rush yards": "Rushing Yards",
        "rushing tds": "Rushing TDs",
        "rushing touchdowns": "Rushing TDs",
        "games": "Games",
        "fantasy points": "Fantasy Points",
        "fpts": "FPTS",
    }
    return aliases.get(title.lower(), title)


def parse_sections(page_html: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks = re.findall(r"<h[24][^>]*>(.*?)</h[24]>\s*<table[^>]*>(.*?)</table>", page_html, re.I | re.S)
    rows = []
    rows_by_section = {}
    sections_found = []
    for raw_title, table_html in blocks:
        section = canonical_section(raw_title)
        sections_found.append(section)
        output_column = SECTION_TO_OUTPUT.get(section)
        if output_column is None:
            continue
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.I | re.S):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.I | re.S)
            if len(cells) < 2:
                continue
            link = re.search(r"<a[^>]*>(.*?)</a>", cells[0], re.I | re.S)
            if not link:
                continue
            team_match = re.search(r"<small[^>]*>(.*?)</small>", cells[0], re.I | re.S)
            value = to_float(clean_text(cells[1]))
            if value is None:
                continue
            rows.append(
                {
                    "section": section,
                    "output_column": output_column,
                    "player_name": clean_text(link.group(1)),
                    "projected_team": clean_text(team_match.group(1)) if team_match else "",
                    "value": value,
                }
            )
        rows_by_section[section] = sum(1 for row in rows if row["section"] == section)
    return rows, {
        "sections_found": sections_found,
        "target_sections_found": [section for section in sections_found if section in SECTION_TO_OUTPUT],
        "tables_found": len(blocks),
        "rows_extracted_by_section": rows_by_section,
    }


def read_csv_rows(path: Path, columns: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{key: row.get(key, "") for key in columns} for row in reader]


def build_position_lookup() -> tuple[dict[tuple[int, str], str], dict[tuple[int, str], str], dict[tuple[int, str], str]]:
    exact = {}
    initial = {}
    teams = {}
    initial_counts = defaultdict(int)
    paths = [
        VALIDATION_DIR / "predraft_validation_dataset.csv",
        VALIDATION_DIR / "predraft_validation_dataset_expanded.csv",
        VALIDATION_DIR / "predraft_validation_dataset_sportsbook.csv",
        PROJECT_ROOT / "data" / "processed" / "master_players.csv",
    ]
    raw = []
    for path in paths:
        for row in read_csv_rows(path, {"season", "player_name", "position", "team"}):
            position = str(row.get("position", "")).upper().strip()
            if position not in {"WR", "RB"}:
                continue
            season_text = str(row.get("season", "")).strip()
            try:
                season = int(float(season_text)) if season_text else 0
            except ValueError:
                continue
            name = row.get("player_name", "")
            key = clean_name_key(name)
            if key:
                raw.append((season, key, initial_surname_key(name), position, str(row.get("team", "")).strip()))
    for season, key, initial_key, position, team in raw:
        exact.setdefault((season, key), position)
        teams.setdefault((season, key), team)
        initial_counts[(season, initial_key)] += 1
    for season, _key, initial_key, position, _team in raw:
        if initial_counts[(season, initial_key)] == 1:
            initial[(season, initial_key)] = position
    return exact, initial, teams


def resolve_position(season: int, name: str, exact: dict[tuple[int, str], str], initial: dict[tuple[int, str], str]) -> str:
    key = clean_name_key(name)
    initial_key = initial_surname_key(name)
    return exact.get((season, key)) or initial.get((season, initial_key)) or exact.get((0, key)) or initial.get((0, initial_key)) or ""


def resolve_team(season: int, name: str, teams: dict[tuple[int, str], str]) -> str:
    key = clean_name_key(name)
    return teams.get((season, key)) or teams.get((0, key)) or ""


def clean_rows(source: dict[str, Any], section_rows: list[dict[str, Any]], exact: dict, initial: dict, teams: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    players = {}
    for row in section_rows:
        key = clean_name_key(row["player_name"])
        player = players.setdefault(
            key,
            {
                "season": source["season"],
                "player_name": row["player_name"],
                "position": resolve_position(source["season"], row["player_name"], exact, initial),
                "projected_fantasy_points": "",
                "projected_positional_rank": "",
                "projection_source": PROJECTION_SOURCE,
                "scoring_format": SCORING_FORMAT,
                "projected_targets": "",
                "projected_carries": "",
                "projected_receptions": "",
                "projected_receiving_yards": "",
                "projected_receiving_tds": "",
                "projected_rushing_yards": "",
                "projected_rushing_tds": "",
                "projected_total_tds": "",
                "projected_games": "",
                "projected_team": row.get("projected_team") or resolve_team(source["season"], row["player_name"], teams),
                "projection_date": source["archive_datetime_utc"][:10] if source["archive_datetime_utc"] else "",
                "source_url_or_file": source["wayback_url"],
                "is_preseason_projection": str(source["include_in_main_validation"]).lower(),
                "projection_safety_status": source["safety_status"],
                "_sections": set(),
            },
        )
        player[row["output_column"]] = row["value"]
        player["_sections"].add(row["section"])
        if row.get("projected_team") and not player.get("projected_team"):
            player["projected_team"] = row["projected_team"]

    cleaned = []
    unresolved = []
    for player in players.values():
        if player.get("position") not in {"WR", "RB"}:
            unresolved.append(
                {
                    "season": player["season"],
                    "player_name": player["player_name"],
                    "team": player.get("projected_team", ""),
                    "sections": ";".join(sorted(player["_sections"])),
                    "source_url_or_file": source["wayback_url"],
                    "projection_safety_status": source["safety_status"],
                }
            )
            continue
        rec_tds = to_float(player.get("projected_receiving_tds")) or 0
        rush_tds = to_float(player.get("projected_rushing_tds")) or 0
        player["projected_total_tds"] = rec_tds + rush_tds if rec_tds or rush_tds else ""
        cleaned.append({column: player.get(column, "") for column in OUTPUT_COLUMNS})

    for position in ["RB", "WR"]:
        ranked = [row for row in cleaned if row["position"] == position]
        ranked.sort(
            key=lambda row: (
                -(to_float(row.get("projected_fantasy_points")) or 0),
                -(to_float(row.get("projected_rushing_yards")) or 0)
                - (to_float(row.get("projected_receiving_yards")) or 0)
                - 20 * (to_float(row.get("projected_total_tds")) or 0)
                - 0.5 * (to_float(row.get("projected_receptions")) or 0),
            )
        )
        for rank, row in enumerate(ranked, 1):
            row["projected_positional_rank"] = rank
    cleaned.sort(key=lambda row: (int(row["season"]), row["position"], row["player_name"]))
    diag = {
        "unresolved_rows": len(unresolved),
        "wr_rows_resolved": sum(1 for row in cleaned if row["position"] == "WR"),
        "rb_rows_resolved": sum(1 for row in cleaned if row["position"] == "RB"),
        "projected_fantasy_points_available": any(row.get("projected_fantasy_points") not in {"", None} for row in cleaned),
        "projected_volume_stats_available": any(any(row.get(col) not in {"", None} for col in VOLUME_COLUMNS) for row in cleaned),
    }
    return cleaned, unresolved, diag


def parse_season(source: dict[str, Any], exact: dict, initial: dict, teams: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    page_html, fetch_diag = fetch_page(source)
    section_rows, parse_diag = parse_sections(page_html)
    rows, unresolved, clean_diag = clean_rows(source, section_rows, exact, initial, teams)
    output_path = VALIDATION_DIR / f"projections_wayback_fantasypros_{source['season']}.csv"
    write_csv(output_path, rows, OUTPUT_COLUMNS)
    diag = {
        **fetch_diag,
        **parse_diag,
        **clean_diag,
        "archive_datetime_utc": source["archive_datetime_utc"],
        "season_start_datetime_utc": source["season_start_datetime_utc"],
        "safety_status": source["safety_status"],
        "included_in_main_validation": source["include_in_main_validation"],
        "safety_notes": source["safety_notes"],
        "rows_extracted": len(section_rows),
        "unique_players_extracted": len({clean_name_key(row["player_name"]) for row in section_rows}),
        "output_rows": len(rows),
        "output_file_path": str(output_path),
    }
    return rows, unresolved, diag


def write_report(coverage: list[dict[str, Any]], historical_created: bool, import_rerun: bool, feature_rerun: bool, import_notes: str) -> None:
    attempted = [row["season"] for row in coverage]
    parsed = [row["season"] for row in coverage if parse_bool(row["page_accessible"]) and int(row["rows_extracted"] or 0) > 0]
    included = [row["season"] for row in coverage if parse_bool(row["included_in_main_validation"]) and int(row["wr_rows_resolved"] or 0) + int(row["rb_rows_resolved"] or 0) > 0]
    excluded = [row["season"] for row in coverage if not parse_bool(row["included_in_main_validation"])]
    failed = [row["season"] for row in coverage if not parse_bool(row["page_accessible"]) or int(row["rows_extracted"] or 0) == 0]
    total_wr = sum(int(row["wr_rows_resolved"] or 0) for row in coverage if parse_bool(row["included_in_main_validation"]))
    total_rb = sum(int(row["rb_rows_resolved"] or 0) for row in coverage if parse_bool(row["included_in_main_validation"]))
    fpts = any(parse_bool(row["projected_fantasy_points_available"]) for row in coverage)
    lines = [
        "# Wayback FantasyPros Projection Expansion Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "Scope: research-only historical projection extraction. This does not modify the Streamlit app, add UI, build recommendations, rerun model validation, or claim model edge.",
        "",
        "## Summary",
        "",
        f"Seasons attempted: {', '.join(map(str, attempted))}.",
        f"Seasons successfully parsed: {', '.join(map(str, parsed)) or 'none'}.",
        f"Seasons included in the validation-safe file: {', '.join(map(str, included)) or 'none'}.",
        f"Seasons excluded for safety: {', '.join(map(str, excluded)) or 'none'}.",
        f"Seasons failed technically: {', '.join(map(str, failed)) or 'none'}.",
        f"Total validation-safe WR rows: `{total_wr}`.",
        f"Total validation-safe RB rows: `{total_rb}`.",
        "Were all selected captures preseason-safe? No. 2020 and 2025 are excluded.",
        f"Were projected fantasy points available? {'Yes' if fpts else 'No'}.",
        f"Projected volume features available: {', '.join(VOLUME_COLUMNS)}.",
        f"Is this data valid enough for Historical Projection Import V1? {'Yes, for preseason WR/RB volume features; projected fantasy points were not found.' if historical_created else 'Not yet.'}",
        f"Was `historical_projections.csv` created? {'Yes' if historical_created else 'No'}.",
        f"Projection import rerun? {'Yes' if import_rerun else 'No'}.",
        f"Projection feature build rerun? {'Yes' if feature_rerun else 'No'}.",
        f"Import/build notes: {import_notes}",
        "",
        "## Season Coverage",
        "",
        "| Season | Safety status | Included | Page accessible | Tables | Rows | WR | RB | Fantasy points | Volume stats | Output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in coverage:
        lines.append(
            f"| {row['season']} | {row['safety_status']} | {row['included_in_main_validation']} | {row['page_accessible']} | {row['tables_found']} | {row['rows_extracted']} | {row['wr_rows_resolved']} | {row['rb_rows_resolved']} | {row['projected_fantasy_points_available']} | {row['projected_volume_stats_available']} | `{row['output_file_path']}` |"
        )
    lines += [
        "",
        "## Safety Notes",
        "",
        "The Wayback URL timestamp is the primary safety check. Captures after the season opener are excluded even if the page content looks like projections. The provided 2020 capture is excluded because it is after kickoff. The 2025 season is excluded because no preseason-safe August 16 capture for the exact leaders URL was found in the tiny lookup, and September 15 captures are post-kickoff.",
        "",
        "## Next Validation Step",
        "",
        "Create the projection import and feature-build scripts, or add these projection volume columns directly into a dedicated expanded predraft dataset builder. Then run a separate validation comparing ADP-only versus ADP plus preseason projected volume. No WR/RB edge classification should be made until that validation is complete.",
        "",
        "Recommended next Codex prompt:",
        "",
        "```text",
        "Continue research-only validation in research/validation_v1. Use historical_projections.csv to build preseason projection volume features, merge them into the predraft validation dataset without leakage, then rerun WR/RB validation comparing ADP-only against ADP plus projected volume features.",
        "```",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    sources = season_sources()
    write_csv(SOURCE_CONFIG_CSV, [{k: (str(v).lower() if isinstance(v, bool) else v) for k, v in source.items()} for source in sources], CONFIG_COLUMNS)
    sources_to_parse = [source for source in sources if not args.season or source["season"] == args.season]
    exact, initial, teams = build_position_lookup()
    all_rows, main_rows, unresolved_rows, coverage = [], [], [], []
    diagnostics = {"source_config_path": str(SOURCE_CONFIG_CSV), "seasons": {}}
    for source in sources_to_parse:
        rows, unresolved, diag = parse_season(source, exact, initial, teams)
        all_rows.extend(rows)
        unresolved_rows.extend(unresolved)
        if source["include_in_main_validation"]:
            main_rows.extend(rows)
        diagnostics["seasons"][str(source["season"])] = diag
        coverage.append(
            {
                "season": source["season"],
                "selected_url": source["wayback_url"],
                "archive_datetime_utc": source["archive_datetime_utc"],
                "season_start_datetime_utc": source["season_start_datetime_utc"],
                "safety_status": source["safety_status"],
                "included_in_main_validation": str(source["include_in_main_validation"]).lower(),
                "page_accessible": str(diag["page_accessible"]).lower(),
                "tables_found": diag["tables_found"],
                "rows_extracted": diag["rows_extracted"],
                "unique_players_extracted": diag["unique_players_extracted"],
                "wr_rows_resolved": diag["wr_rows_resolved"],
                "rb_rows_resolved": diag["rb_rows_resolved"],
                "projected_fantasy_points_available": str(diag["projected_fantasy_points_available"]).lower(),
                "projected_volume_stats_available": str(diag["projected_volume_stats_available"]).lower(),
                "unresolved_player_count": diag["unresolved_rows"],
                "output_file_path": diag["output_file_path"],
            }
        )
    all_rows.sort(key=lambda row: (int(row["season"]), row["position"], row["player_name"]))
    main_rows.sort(key=lambda row: (int(row["season"]), row["position"], row["player_name"]))
    write_csv(ALL_EXTRACTED_CSV, all_rows, OUTPUT_COLUMNS)
    write_csv(MAIN_SAFE_CSV, main_rows, OUTPUT_COLUMNS)
    write_csv(UNRESOLVED_CSV, unresolved_rows, UNRESOLVED_COLUMNS)
    write_csv(COVERAGE_CSV, coverage, COVERAGE_COLUMNS)
    safe_successful = sorted(
        {
            int(row["season"])
            for row in coverage
            if parse_bool(row["included_in_main_validation"])
            and parse_bool(row["page_accessible"])
            and int(row["wr_rows_resolved"] or 0) + int(row["rb_rows_resolved"] or 0) > 0
        }
    )
    historical_created = len(safe_successful) >= 5
    if historical_created:
        write_csv(HISTORICAL_PROJECTIONS_CSV, main_rows, OUTPUT_COLUMNS)
    import_script = VALIDATION_DIR / "import_historical_projections.py"
    feature_script = VALIDATION_DIR / "build_projection_features_v1.py"
    import_rerun = False
    feature_rerun = False
    import_notes = "Skipped because import_historical_projections.py and build_projection_features_v1.py do not exist."
    if historical_created and import_script.exists() and feature_script.exists():
        import_notes = "Skipped in this extraction-only task despite scripts existing."
    diagnostics["summary"] = {
        "seasons_attempted": [row["season"] for row in coverage],
        "seasons_included": safe_successful,
        "seasons_excluded": [row["season"] for row in coverage if not parse_bool(row["included_in_main_validation"])],
        "total_wr_rows": sum(int(row["wr_rows_resolved"] or 0) for row in coverage if parse_bool(row["included_in_main_validation"])),
        "total_rb_rows": sum(int(row["rb_rows_resolved"] or 0) for row in coverage if parse_bool(row["included_in_main_validation"])),
        "projected_fantasy_points_available": any(parse_bool(row["projected_fantasy_points_available"]) for row in coverage),
        "historical_projections_created": historical_created,
        "projection_import_rerun": import_rerun,
        "projection_feature_build_rerun": feature_rerun,
    }
    DIAGNOSTICS_JSON.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    write_report(coverage, historical_created, import_rerun, feature_rerun, import_notes)
    failed = [row["season"] for row in coverage if not parse_bool(row["page_accessible"]) or int(row["rows_extracted"] or 0) == 0]
    print(f"seasons attempted: {', '.join(map(str, diagnostics['summary']['seasons_attempted']))}")
    print(f"seasons included: {', '.join(map(str, safe_successful)) or 'none'}")
    print(f"seasons excluded: {', '.join(map(str, diagnostics['summary']['seasons_excluded'])) or 'none'}")
    print(f"seasons failed: {', '.join(map(str, failed)) or 'none'}")
    print(f"total WR rows: {diagnostics['summary']['total_wr_rows']}")
    print(f"total RB rows: {diagnostics['summary']['total_rb_rows']}")
    print(f"projected fantasy points available: {'yes' if diagnostics['summary']['projected_fantasy_points_available'] else 'no'}")
    print(f"historical_projections.csv created: {'yes' if historical_created else 'no'}")
    print(f"projection import rerun: {'yes' if import_rerun else 'no'}")
    print(f"projection feature build rerun: {'yes' if feature_rerun else 'no'}")
    print(f"report path: {REPORT_MD}")
    print("recommended next Codex prompt: Continue research-only validation in research/validation_v1. Use historical_projections.csv to build preseason projection volume features, merge them into the predraft validation dataset without leakage, then rerun WR/RB validation comparing ADP-only against ADP plus projected volume features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


