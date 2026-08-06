from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


CASE_STUDY_DATA_DIR = Path("case_studies/data")
NFLVERSE_RELEASE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
STATS_PLAYER_TAG = "stats_player"
ROSTERS_TAG = "rosters"
DEFAULT_SEASONS = tuple(range(1999, 2026))

POSITION_COLS = ["position", "pos", "Position", "POS"]
AGE_COLS = ["age", "Age"]
POSITIONAL_FINISH_COLS = [
    "positional_finish",
    "position_finish",
    "pos_rank",
    "position_rank",
    "Positional Finish",
    "Position Finish",
]
PRODUCTION_STAT_COLS = [
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]

SCORING_COLUMN_CANDIDATES = {
    "ppr": ["fantasy_points_ppr", "fantasy_points", "fantasy_points_half_ppr"],
    "half_ppr": ["fantasy_points_half_ppr", "fantasy_points_ppr", "fantasy_points"],
    "standard": ["fantasy_points", "fantasy_points_ppr", "fantasy_points_half_ppr"],
    "dad": [],
}

RB_YARDAGE_BONUSES = [
    (201, None, 10),
    (176, 200, 9),
    (151, 175, 8),
    (126, 150, 7),
    (100, 125, 6),
    (76, 99, 3),
    (60, 75, 2),
    (40, 59, 1),
]

QB_YARDAGE_BONUSES = [
    (351, None, 9),
    (301, 350, 8),
    (250, 300, 7),
    (226, 249, 4),
    (201, 225, 3),
    (176, 200, 2),
    (151, 175, 1),
]


def _safe_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _github_auth_headers() -> Dict[str, str]:
    """
    Optional GitHub token support to raise the unauthenticated 60-req/hour
    API rate limit to 5,000/hour. Looks for NFLVERSE_GITHUB_TOKEN/GITHUB_TOKEN
    env vars first, then a gitignored `.github_token` file at the repo root
    (never committed -- see .gitignore). No token means no auth header and
    the original unauthenticated behavior, unchanged.
    """
    import os

    token = os.environ.get("NFLVERSE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token_file = Path(__file__).resolve().parents[1] / ".github_token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        return {}
    return {"Authorization": f"token {token}"}


def _fetch_release_assets(tag: str) -> List[Dict[str, str]]:
    auth_headers = _github_auth_headers()

    def _get_json(url: str):
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "guaranteed-play-rb-age-study",
                **auth_headers,
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    request = Request(
        NFLVERSE_RELEASE_API.format(tag=tag),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "guaranteed-play-rb-age-study",
            **auth_headers,
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to pull nflverse release metadata for {tag}: {exc}") from exc

    assets_url = payload.get("assets_url")
    assets = []
    try:
        if assets_url:
            page = 1
            while True:
                page_assets = _get_json(f"{assets_url}?per_page=100&page={page}")
                if not page_assets:
                    break
                assets.extend(page_assets)
                page += 1
        else:
            assets = payload.get("assets", [])
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to pull nflverse release assets for {tag}: {exc}") from exc

    if not assets:
        raise RuntimeError(f"No nflverse release assets found for {tag}.")
    return assets


def _asset_year(name: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", name)
    return int(match.group(0)) if match else None


def _csv_assets_for_years(
    tag: str,
    seasons: Iterable[int],
    prefer_terms: Iterable[str],
    avoid_terms: Iterable[str] = (),
    fallback_to_all: bool = True,
) -> List[Dict[str, str]]:
    years = {int(season) for season in seasons}
    assets = [
        asset
        for asset in _fetch_release_assets(tag)
        if str(asset.get("name", "")).lower().endswith(".csv")
        and _asset_year(str(asset.get("name", ""))) in years
    ]

    preferred = [
        asset
        for asset in assets
        if any(term in str(asset.get("name", "")).lower() for term in prefer_terms)
        and not any(term in str(asset.get("name", "")).lower() for term in avoid_terms)
    ]
    if preferred:
        return sorted(preferred, key=lambda asset: str(asset.get("name", "")))
    if not fallback_to_all:
        return []
    return sorted(assets, key=lambda asset: str(asset.get("name", "")))


def _read_remote_csv(asset: Dict[str, str]) -> pd.DataFrame:
    url = asset.get("browser_download_url")
    if not url:
        return pd.DataFrame()
    df = pd.read_csv(url, low_memory=False)
    df.attrs["source_asset"] = asset.get("name", url)
    return df


def _normalize_position_column(df: pd.DataFrame) -> pd.DataFrame:
    position_col = _safe_col(df, ["position", "recent_position", "pos"])
    if position_col is None:
        df["position"] = ""
    elif position_col != "position":
        df["position"] = df[position_col]
    return df


def _choose_scoring_column(df: pd.DataFrame, scoring: str) -> Optional[str]:
    for column in SCORING_COLUMN_CANDIDATES.get(scoring, SCORING_COLUMN_CANDIDATES["ppr"]):
        if column in df.columns:
            return column
    return None


def _numeric_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _sum_available_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for column in columns:
        total = total + _numeric_col(df, column)
    return total


def _yardage_bonus(total_yards: pd.Series, bonuses=RB_YARDAGE_BONUSES) -> pd.Series:
    bonus = pd.Series(0.0, index=total_yards.index)
    for lower, upper, points in bonuses:
        if upper is None:
            mask = total_yards >= lower
        else:
            mask = (total_yards >= lower) & (total_yards <= upper)
        bonus = bonus.mask(mask, float(points))
    return bonus


def _dad_settings_points(df: pd.DataFrame) -> pd.Series:
    passing_yards = _numeric_col(df, "passing_yards")
    rushing_yards = _numeric_col(df, "rushing_yards")
    receiving_yards = _numeric_col(df, "receiving_yards")
    total_yards = passing_yards + rushing_yards + receiving_yards
    position = df["position"].astype(str).str.upper() if "position" in df.columns else pd.Series("", index=df.index)

    two_point_conversions = _sum_available_columns(
        df,
        [
            "passing_2pt_conversions",
            "rushing_2pt_conversions",
            "receiving_2pt_conversions",
            "two_point_conversions",
        ],
    )
    if "fumbles_lost" in df.columns:
        fumbles_lost = _numeric_col(df, "fumbles_lost")
    else:
        fumbles_lost = _sum_available_columns(
            df,
            [
                "passing_fumbles_lost",
                "rushing_fumbles_lost",
                "receiving_fumbles_lost",
                "sack_fumbles_lost",
            ],
        )

    points = (
        passing_yards * 0.04
        + rushing_yards * 0.1
        + receiving_yards * 0.1
        + _numeric_col(df, "passing_tds") * 4.0
        + _numeric_col(df, "rushing_tds") * 5.0
        + _numeric_col(df, "receiving_tds") * 4.0
        + _numeric_col(df, "interceptions") * -1.0
        + two_point_conversions * 2.0
        + fumbles_lost * -2.0
    )
    return points + _yardage_bonus(total_yards, RB_YARDAGE_BONUSES).where(
        ~position.eq("QB"),
        _yardage_bonus(total_yards, QB_YARDAGE_BONUSES),
    )


def _prepare_stats_frame(
    frames: List[pd.DataFrame],
    scoring: str,
    positions: Iterable[str] = ("RB",),
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    stats_df = pd.concat(frames, ignore_index=True, sort=False)
    if "season_type" in stats_df.columns:
        stats_df = stats_df[stats_df["season_type"].astype(str).str.upper().eq("REG")].copy()

    stats_df = _normalize_position_column(stats_df)
    id_col = _safe_col(stats_df, ["player_id", "gsis_id", "player_gsis_id", "nfl_id"])
    name_col = _safe_col(stats_df, ["player_name", "player_display_name", "display_name", "name"])
    season_col = _safe_col(stats_df, ["season", "year"])
    scoring_col = None if scoring == "dad" else _choose_scoring_column(stats_df, scoring)

    required = [id_col, season_col]
    if scoring != "dad":
        required.append(scoring_col)
    if any(column is None for column in required):
        return pd.DataFrame()

    working_df = stats_df.copy()
    working_df["_player_id"] = working_df[id_col].astype(str)
    working_df["_season"] = pd.to_numeric(working_df[season_col], errors="coerce")
    if scoring == "dad":
        working_df["_fantasy_points"] = _dad_settings_points(working_df)
    else:
        working_df["_fantasy_points"] = pd.to_numeric(working_df[scoring_col], errors="coerce").fillna(0.0)
    working_df["_position"] = working_df["position"].astype(str).str.upper()
    working_df["_player_name"] = (
        working_df[name_col].astype(str)
        if name_col
        else working_df["_player_id"]
    )
    allowed_positions = {str(position).upper() for position in positions}
    working_df = working_df[
        working_df["_season"].notna()
        & working_df["_player_id"].ne("")
        & working_df["_position"].isin(allowed_positions)
    ].copy()

    if working_df.empty:
        return pd.DataFrame()

    agg_map = {"fantasy_points": ("_fantasy_points", "sum")}
    for column in PRODUCTION_STAT_COLS:
        if column in working_df.columns:
            working_df[f"_{column}"] = _numeric_col(working_df, column)
            agg_map[column] = (f"_{column}", "sum")

    grouped = working_df.groupby(
        ["_season", "_player_id", "_player_name", "_position"],
        as_index=False,
    ).agg(**agg_map)
    grouped = grouped.rename(
        columns={
            "_season": "season",
            "_player_id": "player_id",
            "_player_name": "player_name",
            "_position": "position",
        }
    )
    grouped["season"] = grouped["season"].astype(int)
    grouped["positional_finish"] = grouped.groupby(["season", "position"])["fantasy_points"].rank(
        ascending=False,
        method="first",
    )
    return grouped


def _derive_age_from_birth_date(rosters_df: pd.DataFrame) -> Optional[pd.Series]:
    birth_col = _safe_col(rosters_df, ["birth_date", "birthdate", "dob"])
    season_col = _safe_col(rosters_df, ["season", "year"])
    if birth_col is None or season_col is None:
        return None

    birth_dates = pd.to_datetime(rosters_df[birth_col], errors="coerce")
    seasons = pd.to_numeric(rosters_df[season_col], errors="coerce")
    return seasons - birth_dates.dt.year


def _prepare_roster_ages(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    rosters_df = pd.concat(frames, ignore_index=True, sort=False)
    id_col = _safe_col(rosters_df, ["player_id", "gsis_id", "player_gsis_id"])
    season_col = _safe_col(rosters_df, ["season", "year"])
    age_col = _safe_col(rosters_df, ["age", "player_age"])
    if id_col is None or season_col is None:
        return pd.DataFrame()

    working_df = rosters_df.copy()
    working_df["_player_id"] = working_df[id_col].astype(str)
    working_df["_season"] = pd.to_numeric(working_df[season_col], errors="coerce")
    if age_col:
        working_df["_age"] = pd.to_numeric(working_df[age_col], errors="coerce")
    else:
        derived_age = _derive_age_from_birth_date(working_df)
        if derived_age is None:
            return pd.DataFrame()
        working_df["_age"] = pd.to_numeric(derived_age, errors="coerce")

    ages = working_df[
        working_df["_season"].notna()
        & working_df["_age"].notna()
        & working_df["_player_id"].ne("")
    ][["_season", "_player_id", "_age"]].copy()
    if ages.empty:
        return pd.DataFrame()

    ages = ages.groupby(["_season", "_player_id"], as_index=False).agg(age=("_age", "mean"))
    return ages.rename(columns={"_season": "season", "_player_id": "player_id"})


def pull_historical_seasons_df(
    force_refresh: bool = False,
    scoring: str = "ppr",
    seasons: Iterable[int] = DEFAULT_SEASONS,
    positions: Iterable[str] = ("RB",),
) -> pd.DataFrame:
    scoring_key = scoring if scoring in SCORING_COLUMN_CANDIDATES else "ppr"
    position_key = "_".join(sorted(str(position).lower() for position in positions))
    pulled_path = CASE_STUDY_DATA_DIR / f"{position_key}_elite_age_player_seasons_{scoring_key}.csv"
    if pulled_path.exists() and not force_refresh:
        df = pd.read_csv(pulled_path)
        if all(column in df.columns for column in PRODUCTION_STAT_COLS):
            df.attrs["source_path"] = str(pulled_path)
            df.attrs["source_type"] = "cached_pull"
            return df

    stats_assets = _csv_assets_for_years(
        STATS_PLAYER_TAG,
        seasons,
        prefer_terms=["season", "summary"],
        avoid_terms=["team", "week"],
        fallback_to_all=False,
    )
    if not stats_assets:
        stats_assets = _csv_assets_for_years(
            STATS_PLAYER_TAG,
            seasons,
            prefer_terms=["week"],
            avoid_terms=["team"],
            fallback_to_all=True,
        )
    roster_assets = _csv_assets_for_years(
        ROSTERS_TAG,
        seasons,
        prefer_terms=["roster"],
    )
    if not stats_assets:
        raise RuntimeError("No nflverse player stats CSV assets were found for the requested seasons.")

    stats_frames = [_read_remote_csv(asset) for asset in stats_assets]
    stats_df = _prepare_stats_frame(stats_frames, scoring=scoring_key, positions=positions)
    if stats_df.empty:
        raise RuntimeError("Pulled player stats did not contain usable RB fantasy scoring data.")

    roster_frames = [_read_remote_csv(asset) for asset in roster_assets] if roster_assets else []
    age_df = _prepare_roster_ages(roster_frames)
    if age_df.empty:
        raise RuntimeError("Pulled roster data did not contain usable player ages.")

    merged_df = stats_df.merge(age_df, on=["season", "player_id"], how="left")
    merged_df = merged_df[merged_df["age"].notna()].copy()
    if merged_df.empty:
        raise RuntimeError("No player-seasons had both RB fantasy finishes and age data.")

    merged_df["age"] = pd.to_numeric(merged_df["age"], errors="coerce").round(1)
    output_cols = [
        "season",
        "player_id",
        "player_name",
        "position",
        "age",
        "fantasy_points",
        "positional_finish",
    ]
    output_cols.extend([column for column in PRODUCTION_STAT_COLS if column in merged_df.columns])
    merged_df = merged_df[output_cols].sort_values(["season", "positional_finish"]).reset_index(drop=True)

    CASE_STUDY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(pulled_path, index=False)
    merged_df.attrs["source_path"] = str(pulled_path)
    merged_df.attrs["source_type"] = "nflverse_pull"
    return merged_df


def resolve_age_study_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if df is None or df.empty:
        return {"position": None, "age": None, "positional_finish": None}

    return {
        "position": _safe_col(df, POSITION_COLS),
        "age": _safe_col(df, AGE_COLS),
        "positional_finish": _safe_col(df, POSITIONAL_FINISH_COLS),
    }


def build_rb_elite_age_study(
    seasons_df: pd.DataFrame,
    min_sample_size: int = 25,
    position: str = "RB",
) -> Dict[str, object]:
    columns = resolve_age_study_columns(seasons_df)
    missing = [key for key, column in columns.items() if column is None]
    if seasons_df is None or seasons_df.empty or missing:
        return {
            "summary": pd.DataFrame(),
            "peaks": {
                "top36_rate": None,
                "top24_rate": None,
                "top20_rate": None,
                "top15_rate": None,
                "top12_rate": None,
                "top6_rate": None,
                "top5_rate": None,
                "top3_rate": None,
                "top1_rate": None,
            },
            "missing_columns": missing,
            "columns": columns,
            "raw_rb_rows": 0,
            "filtered_ages": 0,
        }

    working_df = seasons_df.copy()
    working_df["_position"] = working_df[columns["position"]].astype(str).str.upper()
    working_df["_age"] = pd.to_numeric(working_df[columns["age"]], errors="coerce")
    working_df["_positional_finish"] = pd.to_numeric(
        working_df[columns["positional_finish"]],
        errors="coerce",
    )

    position_key = str(position).upper()
    rb_df = working_df[
        (working_df["_position"] == position_key)
        & working_df["_age"].notna()
        & working_df["_positional_finish"].notna()
    ].copy()

    if rb_df.empty:
        return {
            "summary": pd.DataFrame(),
            "peaks": {
                "top36_rate": None,
                "top24_rate": None,
                "top20_rate": None,
                "top15_rate": None,
                "top12_rate": None,
                "top6_rate": None,
                "top5_rate": None,
                "top3_rate": None,
                "top1_rate": None,
            },
            "missing_columns": [],
            "columns": columns,
            "raw_rb_rows": 0,
            "filtered_ages": 0,
        }

    rb_df["_age"] = rb_df["_age"].round().astype(int)
    top36_df = rb_df[rb_df["_positional_finish"] <= 36].copy()
    if top36_df.empty:
        return {
            "summary": pd.DataFrame(),
            "peaks": {
                "top36_rate": None,
                "top24_rate": None,
                "top20_rate": None,
                "top15_rate": None,
                "top12_rate": None,
                "top6_rate": None,
                "top5_rate": None,
                "top3_rate": None,
                "top1_rate": None,
            },
            "missing_columns": [],
            "columns": columns,
            "raw_rb_rows": len(rb_df),
            "filtered_ages": 0,
        }

    grouped = pd.DataFrame({"Age": sorted(top36_df["_age"].unique().tolist())})
    age_season_counts = rb_df.groupby("_age").size()
    grouped["total_rb_seasons"] = grouped["Age"].map(age_season_counts).fillna(0).astype(int)

    thresholds = {
        "top36": 36,
        "top24": 24,
        "top20": 20,
        "top15": 15,
        "top12": 12,
        "top6": 6,
        "top5": 5,
        "top3": 3,
        "top1": 1,
    }
    for threshold, finish_cutoff in thresholds.items():
        threshold_df = rb_df[rb_df["_positional_finish"] <= finish_cutoff]
        total_slots = int(len(threshold_df))
        age_counts = threshold_df.groupby("_age").size()
        grouped[f"{threshold}_rb_seasons"] = grouped["Age"].map(age_counts).fillna(0).astype(int)
        grouped[f"{threshold}_total_slots"] = total_slots
        grouped[f"{threshold}_rate"] = (
            grouped[f"{threshold}_rb_seasons"] / total_slots if total_slots else 0.0
        )
        grouped[f"{threshold}_rate_pct"] = (grouped[f"{threshold}_rate"] * 100.0).round(2)

    if min_sample_size and int(min_sample_size) > 1:
        grouped = grouped[grouped["top36_rb_seasons"] >= int(min_sample_size)].copy()
    if grouped.empty:
        return {
            "summary": pd.DataFrame(),
            "peaks": {
                "top36_rate": None,
                "top24_rate": None,
                "top20_rate": None,
                "top15_rate": None,
                "top12_rate": None,
                "top6_rate": None,
                "top5_rate": None,
                "top3_rate": None,
                "top1_rate": None,
            },
            "missing_columns": [],
            "columns": columns,
            "raw_rb_rows": len(rb_df),
            "filtered_ages": 0,
        }

    grouped = grouped.sort_values("Age").reset_index(drop=True)

    peaks = {}
    for threshold in thresholds:
        rate_col = f"{threshold}_rate"
        peak_row = grouped.sort_values(
            [rate_col, f"{threshold}_rb_seasons", "Age"],
            ascending=[False, False, True],
        ).iloc[0]
        peaks[f"{threshold}_rate"] = {
            "age": int(peak_row["Age"]),
            "rate": float(peak_row[rate_col]),
            "rate_pct": float(peak_row[f"{threshold}_rate_pct"]),
            "total_rb_seasons": int(peak_row["total_rb_seasons"]),
            f"{threshold}_total_slots": int(peak_row[f"{threshold}_total_slots"]),
            f"{threshold}_rb_seasons": int(peak_row[f"{threshold}_rb_seasons"]),
        }

    return {
        "summary": grouped,
        "peaks": peaks,
        "missing_columns": [],
        "columns": columns,
        "raw_rb_rows": len(rb_df),
        "filtered_ages": int(len(grouped)),
    }
