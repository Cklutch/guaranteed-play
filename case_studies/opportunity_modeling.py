from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CASE_STUDIES_DIR = Path(__file__).resolve().parent
if str(CASE_STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_STUDIES_DIR))

from rb_elite_age_analysis import (  # noqa: E402
    CASE_STUDY_DATA_DIR,
    ROSTERS_TAG,
    STATS_PLAYER_TAG,
    _csv_assets_for_years,
    _prepare_roster_ages,
    _read_remote_csv,
    _safe_col,
)


OUTPUT_DIR = Path("case_studies/output")
TARGET_SHARE_ROOT_OUTPUT = Path("opportunity_wr_target_share.html")
TARGETS_ROOT_OUTPUT = Path("opportunity_wr_targets.html")

TARGET_SHARE_BUCKETS = [
    {"label": "<10%", "min": 0.00, "max": 0.10},
    {"label": "10-15%", "min": 0.10, "max": 0.15},
    {"label": "15-20%", "min": 0.15, "max": 0.20},
    {"label": "20-25%", "min": 0.20, "max": 0.25},
    {"label": "25-30%", "min": 0.25, "max": 0.30},
    {"label": "30%+", "min": 0.30, "max": None},
]

TARGET_BUCKETS = [
    {"label": "<50", "min": 0, "max": 50},
    {"label": "50-75", "min": 50, "max": 75},
    {"label": "75-100", "min": 75, "max": 100},
    {"label": "100-125", "min": 100, "max": 125},
    {"label": "125-150", "min": 125, "max": 150},
    {"label": "150+", "min": 150, "max": None},
]

THRESHOLDS = [
    ("Top 36 WR probability > 50%", "top36_rate", 50.0),
    ("Top 24 WR probability > 50%", "top24_rate", 50.0),
    ("Top 12 WR probability > 50%", "top12_rate", 50.0),
    ("Top 5 WR probability > 25%", "top5_rate", 25.0),
    ("WR1 overall probability > 10%", "wr1_rate", 10.0),
]


@dataclass(frozen=True)
class StudyConfig:
    slug: str
    study: str
    output_name: str
    metric: str
    metric_label: str
    metric_short: str
    metric_unit: str
    bucket_col: str
    buckets: Sequence[dict[str, object]]
    score_col: str
    score_label: str
    scarcity_bucket: str
    min_targets: int
    zones: Sequence[dict[str, object]]
    threshold_markers: dict[str, dict[str, object]]
    seasons: Sequence[int] = tuple(range(2016, 2026))
    scoring_label: str = "Half PPR"


TARGET_SHARE_CONFIG = StudyConfig(
    slug="opportunity_wr_target_share",
    study="WR Target Share",
    output_name="opportunity_wr_target_share.html",
    metric="target_share_pct",
    metric_label="Target Share",
    metric_short="TS",
    metric_unit="%",
    bucket_col="target_share_bucket",
    buckets=TARGET_SHARE_BUCKETS,
    score_col="target_share_score",
    score_label="Target Share Score",
    scarcity_bucket="30%+",
    min_targets=50,
    zones=[
        {"label": "Low Opportunity", "start": 0, "end": 2, "class": "zone-low"},
        {"label": "Starter Opportunity", "start": 2, "end": 3, "class": "zone-starter"},
        {"label": "High Opportunity", "start": 3, "end": 5, "class": "zone-high"},
        {"label": "Elite Opportunity", "start": 5, "end": 6, "class": "zone-elite"},
    ],
    threshold_markers={
        "top12_rate": {"bucket": "25-30%", "label": "Top 12 Threshold"},
        "wr1_rate": {"bucket": "30%+", "label": "League Winner Threshold"},
    },
)

TARGETS_CONFIG = StudyConfig(
    slug="opportunity_wr_targets",
    study="WR Targets",
    output_name="opportunity_wr_targets.html",
    metric="targets",
    metric_label="Targets",
    metric_short="Targets",
    metric_unit="",
    bucket_col="target_bucket",
    buckets=TARGET_BUCKETS,
    score_col="targets_score",
    score_label="Targets Score",
    scarcity_bucket="150+",
    min_targets=1,
    zones=[
        {"label": "Low Opportunity", "start": 0, "end": 3, "class": "zone-low"},
        {"label": "Starter Opportunity", "start": 3, "end": 4, "class": "zone-starter"},
        {"label": "High Opportunity", "start": 4, "end": 5, "class": "zone-high"},
        {"label": "Elite Opportunity", "start": 5, "end": 6, "class": "zone-elite"},
    ],
    threshold_markers={
        "top12_rate": {"bucket": "125-150", "label": "Top 12 Threshold"},
        "wr1_rate": {"bucket": "150+", "label": "League Winner Threshold"},
    },
)


def _numeric_col(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _choose_scoring_col(df: pd.DataFrame) -> str | None:
    return _safe_col(df, ["fantasy_points_half_ppr", "fantasy_points_ppr", "fantasy_points"])


def _bucket_for(value: float, buckets: Sequence[dict[str, object]]) -> str | None:
    if pd.isna(value):
        return None
    for bucket in buckets:
        upper = bucket["max"]
        if float(value) >= float(bucket["min"]) and (upper is None or float(value) < float(upper)):
            return str(bucket["label"])
    return None


def _bucket_sort(label: str, buckets: Sequence[dict[str, object]]) -> int:
    for index, bucket in enumerate(buckets):
        if str(bucket["label"]) == str(label):
            return index
    return len(buckets)


def _prepare_wr_frame(frames: list[pd.DataFrame], age_df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.concat(frames, ignore_index=True, sort=False)
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"].astype(str).str.upper().eq("REG")].copy()

    id_col = _safe_col(stats, ["player_id", "gsis_id", "player_gsis_id", "nfl_id"])
    name_col = _safe_col(stats, ["player_display_name", "player_name", "display_name", "name"])
    season_col = _safe_col(stats, ["season", "year"])
    team_col = _safe_col(stats, ["recent_team", "team", "posteam"])
    pos_col = _safe_col(stats, ["position", "recent_position", "pos"])
    scoring_col = _choose_scoring_col(stats)
    required = [id_col, name_col, season_col, team_col, pos_col, scoring_col]
    if any(column is None for column in required):
        raise RuntimeError("Pulled nflverse stats are missing required WR season columns.")

    working = stats.copy()
    working["_player_id"] = working[id_col].astype(str)
    working["_player_name"] = working[name_col].astype(str)
    working["_season"] = pd.to_numeric(working[season_col], errors="coerce")
    working["_team"] = working[team_col].astype(str)
    working["_position"] = working[pos_col].astype(str).str.upper()
    working["_targets"] = _numeric_col(working, "targets")
    working["_receiving_tds"] = _numeric_col(working, "receiving_tds")
    receptions = _numeric_col(working, "receptions")
    scoring = pd.to_numeric(working[scoring_col], errors="coerce").fillna(0.0)
    if scoring_col == "fantasy_points_half_ppr":
        working["_fantasy_points"] = scoring
    elif scoring_col == "fantasy_points_ppr":
        working["_fantasy_points"] = scoring - (0.5 * receptions)
    else:
        working["_fantasy_points"] = scoring + (0.5 * receptions)
    working["_games"] = 1.0 if "week" in working.columns else np.where(working["_targets"].gt(0), 1.0, 0.0)

    working = working[
        working["_season"].notna()
        & working["_player_id"].ne("")
        & working["_team"].ne("")
        & working["_team"].ne("nan")
    ].copy()
    working["_season"] = working["_season"].astype(int)

    team_targets = (
        working.groupby(["_season", "_team"], as_index=False)
        .agg(team_targets=("_targets", "sum"))
        .rename(columns={"_season": "season", "_team": "team"})
    )
    wr = working[working["_position"].eq("WR")].copy()
    player_team = (
        wr.groupby(["_season", "_player_id", "_team"], as_index=False)
        .agg(
            player_name=("_player_name", "last"),
            targets=("_targets", "sum"),
            receiving_tds=("_receiving_tds", "sum"),
            fantasy_points=("_fantasy_points", "sum"),
            games=("_games", "sum"),
        )
        .rename(columns={"_season": "season", "_player_id": "player_id", "_team": "team"})
    )
    player_team = player_team.merge(team_targets, on=["season", "team"], how="left")
    player_team = player_team[player_team["team_targets"].fillna(0).gt(0)].copy()
    if age_df is not None and not age_df.empty:
        player_team = player_team.merge(age_df, on=["season", "player_id"], how="left")
    else:
        player_team["age"] = np.nan

    def team_path(values: pd.Series) -> str:
        teams = [str(team) for team in values.tolist() if str(team) and str(team).lower() != "nan"]
        return "/".join(dict.fromkeys(teams))

    season_df = (
        player_team.groupby(["season", "player_id"], as_index=False)
        .agg(
            player_name=("player_name", "last"),
            team=("team", team_path),
            age=("age", "mean"),
            games=("games", "sum"),
            targets=("targets", "sum"),
            receiving_tds=("receiving_tds", "sum"),
            team_targets=("team_targets", "sum"),
            fantasy_points=("fantasy_points", "sum"),
        )
    )
    season_df = season_df[season_df["targets"].ge(1) & season_df["team_targets"].gt(0)].copy()
    season_df["target_share"] = season_df["targets"] / season_df["team_targets"]
    season_df["target_share_pct"] = season_df["target_share"] * 100.0
    season_df["fantasy_ppg"] = season_df["fantasy_points"] / season_df["games"].replace(0, np.nan)
    season_df["positional_finish"] = season_df.groupby("season")["fantasy_points"].rank(ascending=False, method="first")
    season_df["age"] = pd.to_numeric(season_df["age"], errors="coerce").round(1)
    season_df["target_share_bucket"] = season_df["target_share"].apply(lambda value: _bucket_for(value, TARGET_SHARE_BUCKETS))
    season_df["target_bucket"] = season_df["targets"].apply(lambda value: _bucket_for(value, TARGET_BUCKETS))
    return season_df.sort_values(["season", "positional_finish", "player_name"]).reset_index(drop=True)


def pull_wr_opportunity_df(force_refresh: bool = False, seasons: Iterable[int] = TARGET_SHARE_CONFIG.seasons) -> pd.DataFrame:
    seasons = tuple(int(season) for season in seasons)
    cache_path = CASE_STUDY_DATA_DIR / "opportunity_wr_target_share_player_seasons_half_ppr.csv"
    required = [
        "player_id",
        "player_name",
        "season",
        "team",
        "age",
        "games",
        "targets",
        "receiving_tds",
        "team_targets",
        "target_share",
        "target_share_pct",
        "fantasy_points",
        "fantasy_ppg",
        "positional_finish",
        "target_share_bucket",
        "target_bucket",
    ]
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
        if "target_share_pct" not in df.columns and "target_share" in df.columns:
            df["target_share_pct"] = pd.to_numeric(df["target_share"], errors="coerce") * 100.0
        if "target_bucket" not in df.columns and "targets" in df.columns:
            df["target_bucket"] = pd.to_numeric(df["targets"], errors="coerce").apply(lambda value: _bucket_for(value, TARGET_BUCKETS))
        if all(column in df.columns for column in required):
            df[required].to_csv(cache_path, index=False)
            return df[required]

    try:
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
    except RuntimeError:
        stats_assets = [
            {
                "name": f"stats_player_week_{season}.csv",
                "browser_download_url": (
                    "https://github.com/nflverse/nflverse-data/releases/download/"
                    f"stats_player/stats_player_week_{season}.csv"
                ),
            }
            for season in seasons
        ]
    try:
        roster_assets = _csv_assets_for_years(ROSTERS_TAG, seasons, prefer_terms=["roster"])
    except RuntimeError:
        roster_assets = []

    stats_frames = [_read_remote_csv(asset) for asset in stats_assets]
    roster_frames = [_read_remote_csv(asset) for asset in roster_assets] if roster_assets else []
    df = _prepare_wr_frame(stats_frames, _prepare_roster_ages(roster_frames))
    CASE_STUDY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df[required].to_csv(cache_path, index=False)
    return df[required]


def _rate(mask: pd.Series) -> float:
    return float(mask.mean() * 100.0) if len(mask) else 0.0


def _pearson(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return float(pair["x"].corr(pair["y"], method="pearson")) if len(pair) >= 2 else 0.0


def apply_min_targets(df: pd.DataFrame, floor: int) -> pd.DataFrame:
    working = df.copy()
    working["targets"] = pd.to_numeric(working["targets"], errors="coerce").fillna(0.0)
    return working[working["targets"].ge(int(floor))].copy().reset_index(drop=True)


def bucket_summary(df: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    for bucket in config.buckets:
        label = str(bucket["label"])
        group = df[df[config.bucket_col].eq(label)].copy()
        rows.append(
            {
                "bucket": label,
                "sample_size": int(len(group)),
                "average_fantasy_ppg": float(group["fantasy_ppg"].mean()) if len(group) else 0.0,
                "median_fantasy_ppg": float(group["fantasy_ppg"].median()) if len(group) else 0.0,
                "top36_rate": _rate(group["positional_finish"].le(36)),
                "top24_rate": _rate(group["positional_finish"].le(24)),
                "top12_rate": _rate(group["positional_finish"].le(12)),
                "top5_rate": _rate(group["positional_finish"].le(5)),
                "wr1_rate": _rate(group["positional_finish"].eq(1)),
            }
        )
    summary = pd.DataFrame(rows)
    pieces = []
    for column in ["average_fantasy_ppg", "top36_rate", "top24_rate", "top12_rate", "top5_rate", "wr1_rate"]:
        values = summary[column].astype(float)
        span = values.max() - values.min()
        pieces.append((values - values.min()) / span if span else values * 0.0)
    summary[config.score_col] = (pieces[0] * 0.25 + pieces[1] * 0.15 + pieces[2] * 0.2 + pieces[3] * 0.2 + pieces[4] * 0.12 + pieces[5] * 0.08) * 100.0
    return summary.sort_values("bucket", key=lambda col: col.map(lambda value: _bucket_sort(value, config.buckets))).reset_index(drop=True)


def _fmt(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _metric_value(df: pd.DataFrame, config: StudyConfig) -> pd.Series:
    return df["target_share_pct"] if config.metric == "target_share_pct" else df["targets"]


def _metric_display(value: object, config: StudyConfig) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%" if config.metric_unit == "%" else f"{float(value):.0f}"


def _summary_rows(summary: pd.DataFrame, config: StudyConfig) -> str:
    return "\n".join(
        "<tr>"
        f"<td>{escape(str(row['bucket']))}</td>"
        f"<td>{int(row['sample_size']):,}</td>"
        f"<td>{_fmt(row['average_fantasy_ppg'])}</td>"
        f"<td>{_fmt(row['median_fantasy_ppg'])}</td>"
        f"<td>{_fmt(row['top36_rate'])}%</td>"
        f"<td>{_fmt(row['top24_rate'])}%</td>"
        f"<td>{_fmt(row['top12_rate'])}%</td>"
        f"<td>{_fmt(row['top5_rate'])}%</td>"
        f"<td>{_fmt(row['wr1_rate'], 2)}%</td>"
        f"<td>{_fmt(row[config.score_col])}</td>"
        "</tr>"
        for row in summary.to_dict("records")
    )


def _threshold_rows(summary: pd.DataFrame) -> str:
    rows = []
    for label, column, target in THRESHOLDS:
        hit = summary[summary[column].gt(target)].head(1)
        if hit.empty:
            bucket = "Not reached"
            observed = f"Max {summary[column].max():.1f}%"
        else:
            bucket = str(hit.iloc[0]["bucket"])
            observed = f"{float(hit.iloc[0][column]):.1f}%"
        rows.append(f"<tr><td>{escape(label)}</td><td>{escape(bucket)}</td><td>{escape(observed)}</td></tr>")
    return "\n".join(rows)


def _cliff(summary: pd.DataFrame, config: StudyConfig) -> dict[str, object]:
    metrics = [
        ("top36_rate", "Top 36 Rate"),
        ("top24_rate", "Top 24 Rate"),
        ("top12_rate", "Top 12 Rate"),
        ("top5_rate", "Top 5 Rate"),
        ("wr1_rate", "WR1 Overall Rate"),
    ]
    best = {"source": "N/A", "dest": "N/A", "metric": "N/A", "source_rate": 0.0, "dest_rate": 0.0, "increase": 0.0}
    ordered = summary.sort_values("bucket", key=lambda col: col.map(lambda value: _bucket_sort(value, config.buckets)))
    records = ordered.to_dict("records")
    for current, nxt in zip(records, records[1:]):
        for column, label in metrics:
            increase = float(nxt[column]) - float(current[column])
            if increase > float(best["increase"]):
                best = {
                    "source": current["bucket"],
                    "dest": nxt["bucket"],
                    "metric": label,
                    "source_rate": float(current[column]),
                    "dest_rate": float(nxt[column]),
                    "increase": increase,
                }
    return best


def _inflection(summary: pd.DataFrame) -> dict[str, object]:
    hit = summary[summary["top12_rate"].gt(50.0)].head(1)
    row = (hit if len(hit) else summary.sort_values("top12_rate", ascending=False).head(1)).iloc[0]
    return {"bucket": str(row["bucket"]) if len(hit) else "Not reached", "rate": float(row["top12_rate"]), "sample": int(row["sample_size"])}


def _scarcity(df: pd.DataFrame, config: StudyConfig) -> dict[str, object]:
    total = int(len(df))
    count = int(df[config.bucket_col].eq(config.scarcity_bucket).sum())
    return {"bucket": config.scarcity_bucket, "count": count, "total": total, "share": count / total * 100.0 if total else 0.0}


def _spread(summary: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(summary[column], errors="coerce").dropna()
    return float(values.max() - values.min()) if len(values) else 0.0


def comparison_section(df: pd.DataFrame) -> tuple[str, str]:
    share_summary = bucket_summary(df, TARGET_SHARE_CONFIG)
    targets_summary = bucket_summary(df, TARGETS_CONFIG)
    share_r = _pearson(df["target_share"], df["fantasy_ppg"])
    target_r = _pearson(df["targets"], df["fantasy_ppg"])
    rows = []
    wins = {"Target Share": 0, "Targets": 0}
    for label, share_value, target_value, suffix in [
        ("Pearson r", abs(share_r), abs(target_r), ""),
        ("R2", share_r * share_r, target_r * target_r, ""),
        ("Top 12 Predictive Strength", _spread(share_summary, "top12_rate"), _spread(targets_summary, "top12_rate"), " pts"),
        ("Top 5 Predictive Strength", _spread(share_summary, "top5_rate"), _spread(targets_summary, "top5_rate"), " pts"),
        ("WR1 Predictive Strength", _spread(share_summary, "wr1_rate"), _spread(targets_summary, "wr1_rate"), " pts"),
    ]:
        winner = "Target Share" if share_value >= target_value else "Targets"
        wins[winner] += 1
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{share_value:.2f}{suffix}</td>"
            f"<td>{target_value:.2f}{suffix}</td><td>{winner}</td><td>{abs(share_value - target_value):.2f}{suffix}</td></tr>"
        )
    overall = "Target Share" if wins["Target Share"] >= wins["Targets"] else "Targets"
    html = (
        '<section class="table-wrap comparison"><h2>Opportunity Metric Comparison</h2>'
        '<p class="detail">Predictive-strength rows use the spread between the weakest and strongest buckets for that outcome.</p>'
        '<table><thead><tr><th>Metric</th><th>Target Share</th><th>Targets</th><th>Stronger</th><th>Gap</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table><div class=\"comparison-callout\">Standalone winner: <strong>{overall}</strong></div></section>"
    )
    return html, overall


def render_dashboard(config: StudyConfig, df: pd.DataFrame, comparison_html: str = "", comparison_winner: str = "") -> str:
    summary = bucket_summary(df, config)
    metric = _metric_value(df, config)
    corr_x = df["target_share"] if config.metric == "target_share_pct" else df["targets"]
    r = _pearson(corr_x, df["fantasy_ppg"])
    r2 = r * r
    slope, intercept = np.polyfit(metric.astype(float), df["fantasy_ppg"].astype(float), 1) if len(df) >= 2 else (0.0, 0.0)
    work = df.copy()
    work["metric_value"] = metric
    work["expected_ppg"] = intercept + slope * work["metric_value"]
    work["residual_ppg"] = work["fantasy_ppg"] - work["expected_ppg"]
    work["metric_display"] = work["metric_value"].apply(lambda value: _metric_display(value, config))
    outliers = work.assign(abs_residual=lambda x: x["residual_ppg"].abs()).sort_values("abs_residual", ascending=False).head(12)
    leaders = work.sort_values("residual_ppg", ascending=False).head(3)
    cliff = _cliff(summary, config)
    inflection = _inflection(summary)
    scarcity = _scarcity(df, config)
    seasons = sorted(df["season"].astype(int).unique().tolist())
    filter_label = "Min Share %" if config.metric == "target_share_pct" else "Min Targets"
    efficiency_key = "ppg_per_ts" if config.metric == "target_share_pct" else "ppg_per_target"
    efficiency_label = "PPG / TS%" if config.metric == "target_share_pct" else "PPG / Target"

    chart_records = summary[["bucket", "sample_size", "average_fantasy_ppg", "top24_rate", "top12_rate", "top5_rate", "wr1_rate", config.score_col]].to_dict("records")
    scatter_records = work[["player_name", "season", "team", "metric_value", "fantasy_ppg", "positional_finish"]].to_dict("records")
    outlier_records = outliers[["player_name", "season", "team", "games", "targets", "metric_display", "metric_value", "fantasy_ppg", "expected_ppg", "residual_ppg", "positional_finish"]].to_dict("records")
    table_records = []
    for row in work.sort_values(["season", "positional_finish", "player_name"]).to_dict("records"):
        table_records.append(
            {
                "player_name": str(row["player_name"]),
                "season": int(row["season"]),
                "team": str(row["team"]),
                "age": None if pd.isna(row["age"]) else float(row["age"]),
                "games": float(row["games"]),
                "targets": int(row["targets"]),
                "receiving_tds": int(round(float(row["receiving_tds"]))),
                "metric_value": float(row["metric_value"]),
                "metric_display": _metric_display(row["metric_value"], config),
                "ppg_per_ts": float(row["fantasy_ppg"]) / float(row["target_share_pct"]) if float(row["target_share_pct"]) else None,
                "ppg_per_target": float(row["fantasy_ppg"]) / float(row["targets"]) if float(row["targets"]) else None,
                "fantasy_points": float(row["fantasy_points"]),
                "fantasy_ppg": float(row["fantasy_ppg"]),
                "positional_finish": int(row["positional_finish"]),
                "bucket": str(row[config.bucket_col]),
            }
        )

    leader_items = "".join(
        f"<li><strong>{escape(str(row['player_name']))} {int(row['season'])}</strong>"
        f"<span>{escape(str(row['metric_display']))} {escape(config.metric_short)} | {float(row['fantasy_ppg']):.1f} actual vs {float(row['expected_ppg']):.1f} expected | {float(row['residual_ppg']):+.1f} PPG</span></li>"
        for row in leaders.to_dict("records")
    )
    outlier_rows = "".join(
        f"<tr><td>{escape(str(row['player_name']))}</td><td>{int(row['season'])}</td><td>{escape(str(row['team']))}</td>"
        f"<td>{int(round(float(row['games'])))}</td><td>{int(round(float(row['targets'])))}</td><td>{escape(str(row['metric_display']))}</td>"
        f"<td>{float(row['fantasy_ppg']):.1f}</td><td>{float(row['expected_ppg']):.1f}</td><td>{float(row['residual_ppg']):+.1f}</td><td>{int(row['positional_finish'])}</td></tr>"
        for row in outlier_records
    )
    score_pills = "".join(
        f"<div class=\"score-pill\"><span>{escape(str(row['bucket']))}</span><strong>{float(row[config.score_col]):.1f}</strong></div>"
        for row in summary.to_dict("records")
    )
    metric_specs = [
        ("average_fantasy_ppg", "Average Fantasy PPG", "", "avg PPG"),
        ("top24_rate", "Top 24 Rate", "%", "Top 24 rate"),
        ("top12_rate", "Top 12 Rate", "%", "Top 12 probability"),
        ("top5_rate", "Top 5 Rate", "%", "Top 5 probability"),
        ("wr1_rate", "WR1 Overall Rate", "%", "WR1 overall rate"),
    ]
    chart_insights: dict[str, str] = {}
    for column, title, suffix, label in metric_specs:
        ordered = summary.sort_values("bucket", key=lambda col: col.map(lambda value: _bucket_sort(value, config.buckets))).to_dict("records")
        best = max(ordered, key=lambda row: float(row[column])) if ordered else None
        biggest = {"source": "", "dest": "", "increase": 0.0}
        for current, nxt in zip(ordered, ordered[1:]):
            increase = float(nxt[column]) - float(current[column])
            if increase > biggest["increase"]:
                biggest = {"source": str(current["bucket"]), "dest": str(nxt["bucket"]), "increase": increase}
        if best and suffix == "%":
            chart_insights[column] = (
                f"WRs in the {best['bucket']} {config.metric_label.lower()} bucket historically reach {label} "
                f"{float(best[column]):.1f}% of the time (n={int(best['sample_size']):,}). "
                f"Largest jump: {biggest['source']} to {biggest['dest']} (+{biggest['increase']:.1f} pts)."
            )
        elif best:
            chart_insights[column] = (
                f"The {best['bucket']} {config.metric_label.lower()} bucket posts the strongest average output at "
                f"{float(best[column]):.1f} PPG (n={int(best['sample_size']):,}). "
                f"Largest jump: {biggest['source']} to {biggest['dest']} (+{biggest['increase']:.1f} PPG)."
            )
    if config.slug == "opportunity_wr_targets":
        conclusion = [
            "Targets are strongly directional for WR fantasy success, especially once raw volume moves into the 100+ and 125+ target buckets.",
            f"Compared with Target Share, this study's standalone comparison favors {comparison_winner or 'the stronger metric'} across correlation and elite-outcome separation.",
            "Targets should be included as a future WR Opportunity Score component, but paired with Target Share so the model captures both absolute volume and team-context-adjusted opportunity.",
        ]
    else:
        conclusion = [
            "Target share is strongly directional for WR fantasy success: average PPG and elite-finish rates climb sharply as usage moves from sub-15% into the 20%+ buckets.",
            "The major opportunity threshold is around 20-25% target share, with 25%+ concentrating more Top 12 and Top 5 outcomes.",
            "Target Share is suitable as a core Opportunity Model feature because it captures team-context-adjusted opportunity.",
        ]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(config.study)} | Opportunity Modeling</title>
<style>
:root{{--ink:#121826;--muted:#5f6b7a;--line:#dbe2ea;--panel:#fff;--page:#f5f7fa;--navy:#1f3a5f;--teal:#16817a;--green:#247a43;--gold:#a66f16;--red:#b33a3a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--page);color:var(--ink);font-family:Arial,Helvetica,sans-serif}} main{{max-width:1220px;margin:0 auto;padding:30px 22px 56px}}
h1{{margin:0;font-size:32px;line-height:1.15}} h2{{margin:0 0 14px;font-size:20px}} p{{line-height:1.5}} .subtitle{{margin:10px 0 22px;color:var(--muted);max-width:900px}}
.meta,.kpis,.chart-grid,.threshold-grid{{display:grid;gap:12px}} .meta{{grid-template-columns:repeat(5,minmax(0,1fr));margin-bottom:20px}} .kpis{{grid-template-columns:repeat(3,minmax(0,1fr));margin-bottom:22px}} .chart-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}} .threshold-grid{{grid-template-columns:1fr 1fr;align-items:start}}
.card,.chart,.table-wrap,.conclusion{{background:var(--panel);border:1px solid var(--line);border-radius:8px}} .card{{padding:15px 16px}} .chart{{min-height:250px;padding:0;overflow:hidden}} .table-wrap,.conclusion{{margin-top:18px;padding:18px;overflow-x:auto}} .comparison{{margin:0 0 22px}}
.chart-header{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;padding:14px 16px 4px;border-bottom:1px solid #edf1f5}} .chart-title{{margin:0;font-size:14px;font-weight:900;color:#172033}} .chart-subtitle{{margin-top:3px;color:var(--muted);font-size:12px}} .chart-badge{{border:1px solid #c8e5e1;background:#eef8f6;color:#0d6f69;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:900;white-space:nowrap}} .chart-body{{padding:8px 14px 0}} .chart-footer{{padding:8px 16px 14px;color:#4d5a68;font-size:12px;line-height:1.35;border-top:1px solid #edf1f5;background:#fbfcfe}}
.label{{color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase}} .value{{margin-top:6px;font-size:17px;font-weight:800;line-height:1.3}} .detail{{margin-top:5px;color:var(--muted);font-size:13px;line-height:1.4}} .research-note{{border-left:4px solid var(--teal)}} .kpi-main{{margin-top:7px;font-size:21px;font-weight:900;line-height:1.2}} .kpi-sub{{margin-top:7px;color:var(--muted);font-size:13px;font-weight:700;line-height:1.4}}
.efficiency-list{{list-style:none;margin:8px 0 0;padding:0}} .efficiency-list li{{margin-top:8px}} .efficiency-list strong{{display:block;font-size:13px}} .efficiency-list span{{display:block;color:var(--muted);font-size:12px;line-height:1.35}} .comparison-callout{{margin-top:12px;font-size:16px;font-weight:800;color:var(--navy)}}
.score-row{{display:grid;grid-template-columns:repeat(6,minmax(92px,1fr));gap:8px;margin-top:12px}} .score-pill{{background:#eef6f5;border:1px solid #c9e5e1;border-radius:8px;padding:10px}} .score-pill strong{{display:block;font-size:20px;color:var(--teal)}}
svg{{width:100%;height:auto;display:block}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid #e7ebf0;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{background:#f8fafc;font-weight:800;color:#263548}} .interactive-table th{{cursor:pointer;user-select:none}} .interactive-table th.sort-active{{color:var(--teal)}}
.table-controls{{display:grid;grid-template-columns:1.5fr 1fr repeat(5,minmax(116px,1fr)) auto;gap:10px;margin:4px 0 14px;align-items:end}} .control label{{display:block;color:var(--muted);font-size:11px;font-weight:800;margin-bottom:5px;text-transform:uppercase}} .control input,.control select{{width:100%;min-height:34px;border:1px solid var(--line);border-radius:6px;padding:7px 9px;color:var(--ink);background:#fff;font:inherit}} .icon-button{{min-width:36px;min-height:34px;border:1px solid var(--line);border-radius:6px;background:#f8fafc;color:var(--ink);cursor:pointer;font-weight:900}} .table-status{{margin:-4px 0 10px;color:var(--muted);font-size:12px;font-weight:700}}
.axis{{stroke:#263548;stroke-width:1.15}} .grid{{stroke:#e8edf3;stroke-width:1}} .curve-line{{fill:none;stroke:var(--teal);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}} .curve-area{{fill:#16817a;opacity:.07}} .marker-dot{{fill:#fff;stroke:var(--teal);stroke-width:3}} .marker-dot.top12{{stroke:var(--green)}} .marker-dot.top5{{stroke:var(--gold)}} .marker-dot.wr1{{stroke:#7057c7}} .threshold-line{{stroke:#7057c7;stroke-width:1.3;stroke-dasharray:4 5;opacity:.55}} .cliff-line{{stroke:#b33a3a;stroke-width:1.2;stroke-dasharray:3 5;opacity:.55}} .zone-low{{fill:#d8e8f4;opacity:.16}} .zone-starter{{fill:#dff3eb;opacity:.18}} .zone-high{{fill:#f6edc8;opacity:.18}} .zone-elite{{fill:#eadcf8;opacity:.20}} .chart-wide{{margin:0 0 12px}} .scatter-point{{fill:var(--teal);opacity:.34}} .scatter-outlier{{fill:#fff;stroke:var(--red);stroke-width:2;opacity:1}} .trend-line{{stroke:var(--red);stroke-width:2.2}} .outlier-label{{fill:#121826;font-size:11px;font-weight:800}} .tick,.svg-label{{fill:#526071;font-size:10px}} .svg-title{{fill:#121826;font-size:13px;font-weight:900}} .value-label{{fill:#111827;font-size:10px;font-weight:900}} .value-label-bg{{fill:#fff;stroke:#d7e1ea;stroke-width:1;opacity:.96}} .sample-label{{fill:#7a8491;font-size:10px;font-weight:700}}
@media(max-width:920px){{.meta,.kpis,.chart-grid,.threshold-grid{{grid-template-columns:1fr}}.score-row{{grid-template-columns:repeat(2,minmax(0,1fr))}}.table-controls{{grid-template-columns:repeat(2,minmax(0,1fr))}}main{{padding-inline:14px}}}}
</style>
</head>
<body>
<main>
<h1>Opportunity Modeling: {escape(config.study)}</h1>
<p class="subtitle">Research question: {'How predictive are raw targets of fantasy WR success, and how does predictive power compare against Target Share?' if config.slug == 'opportunity_wr_targets' else 'How predictive is target share of fantasy WR success?'} This is a research module for future Opportunity Model components, not a breakout study and not a ranking system.</p>
{comparison_html}
<section class="meta">
<div class="card"><div class="label">Data Source</div><div class="value">nflverse stats_player + rosters</div></div>
<div class="card"><div class="label">Timeline</div><div class="value">{seasons[0]}-{seasons[-1]}</div></div>
<div class="card"><div class="label">Scoring</div><div class="value">{escape(config.scoring_label)}</div></div>
<div class="card"><div class="label">Unit</div><div class="value">WR player seasons</div></div>
<div class="card"><div class="label">Sample</div><div class="value">{len(df):,}</div><div class="detail">Minimum {config.min_targets} season targets</div></div>
</section>
<section class="kpis">
<div class="card research-note"><div class="label">Biggest Opportunity Cliff</div><div class="kpi-main">{escape(str(cliff['source']))} -> {escape(str(cliff['dest']))}</div><div class="kpi-sub">{escape(str(cliff['metric']))}: {float(cliff['source_rate']):.1f}% -> {float(cliff['dest_rate']):.1f}%</div><div class="detail">+{float(cliff['increase']):.1f} percentage points between adjacent {escape(config.metric_label.lower())} buckets.</div></div>
<div class="card"><div class="label">Inflection Point</div><div class="kpi-main">{escape(inflection['bucket'])}</div><div class="kpi-sub">{float(inflection['rate']):.1f}% Top 12 Rate</div><div class="detail">{int(inflection['sample']):,} seasons in bucket.</div></div>
<div class="card"><div class="label">Opportunity Elasticity</div><div class="kpi-main">+{float(slope):.2f} PPG</div><div class="detail">Expected fantasy PPG gained for every {'+1% target share' if config.metric_unit == '%' else '+1 target'}.</div></div>
<div class="card"><div class="label">Opportunity Scarcity</div><div class="kpi-main">{escape(scarcity['bucket'])} {escape(config.metric_label)}</div><div class="kpi-sub">{float(scarcity['share']):.1f}% of WR seasons</div><div class="detail">{int(scarcity['count']):,} of {int(scarcity['total']):,} seasons.</div></div>
<div class="card"><div class="label">Predictive Power</div><div class="kpi-main">r = {r:.2f} | R2 = {r2:.2f}</div><div class="detail">{r2 * 100:.0f}% of fantasy scoring variance explained by {escape(config.metric_label.lower())} alone.</div></div>
<div class="card"><div class="label">Efficiency Leaders</div><ul class="efficiency-list">{leader_items}</ul></div>
</section>
<section class="chart chart-wide"><div class="chart-header"><div><h2 class="chart-title">{escape(config.metric_label)} vs Fantasy PPG Scatter</h2><div class="chart-subtitle">Player-season distribution with least-squares trend</div></div><div class="chart-badge">r = {r:.2f}</div></div><div class="chart-body"><svg id="chart-correlation" viewBox="0 0 1160 330" role="img" aria-label="{escape(config.metric_label)} vs Fantasy PPG scatter"></svg></div><div class="chart-footer">{r2 * 100:.0f}% of fantasy scoring variance is explained by {escape(config.metric_label.lower())} alone.</div></section>
<section class="table-wrap"><h2>Severe Pearson Scatter Outliers</h2><table><thead><tr><th>Player</th><th>Season</th><th>Team</th><th>Games</th><th>Targets</th><th>{escape(config.metric_label)}</th><th>Actual PPG</th><th>Trend PPG</th><th>Residual</th><th>WR Finish</th></tr></thead><tbody>{outlier_rows}</tbody></table></section>
<section class="card"><div class="label">Normalized {escape(config.score_label)}</div><div class="score-row">{score_pills}</div><p class="detail">Score is a 0-100 historical outcome index using average PPG and Top 36/24/12/5/WR1 rates. It is designed as one future Opportunity Model feature, not a fantasy ranking.</p></section>
<section class="chart-grid">
<div class="chart"><div class="chart-header"><div><h2 class="chart-title">Fantasy Production by Opportunity</h2><div class="chart-subtitle">Average half-PPR points per game across increasing opportunity buckets</div></div><div class="chart-badge">PPG</div></div><div class="chart-body"><svg id="chart-ppg" viewBox="0 0 560 240"></svg></div><div class="chart-footer" id="footer-ppg">{escape(chart_insights['average_fantasy_ppg'])}</div></div>
<div class="chart"><div class="chart-header"><div><h2 class="chart-title">Starter Probability Curve</h2><div class="chart-subtitle">Chance of finishing as a Top 24 fantasy wide receiver</div></div><div class="chart-badge">Top 24</div></div><div class="chart-body"><svg id="chart-top24" viewBox="0 0 560 240"></svg></div><div class="chart-footer" id="footer-top24">{escape(chart_insights['top24_rate'])}</div></div>
<div class="chart"><div class="chart-header"><div><h2 class="chart-title">WR1 Probability Curve</h2><div class="chart-subtitle">Chance of finishing inside the season-long Top 12</div></div><div class="chart-badge">Top 12</div></div><div class="chart-body"><svg id="chart-top12" viewBox="0 0 560 240"></svg></div><div class="chart-footer" id="footer-top12">{escape(chart_insights['top12_rate'])}</div></div>
<div class="chart"><div class="chart-header"><div><h2 class="chart-title">Elite Outcome Curve</h2><div class="chart-subtitle">Chance of reaching a Top 5 positional finish</div></div><div class="chart-badge">Top 5</div></div><div class="chart-body"><svg id="chart-top5" viewBox="0 0 560 240"></svg></div><div class="chart-footer" id="footer-top5">{escape(chart_insights['top5_rate'])}</div></div>
<div class="chart"><div class="chart-header"><div><h2 class="chart-title">League Winner Probability</h2><div class="chart-subtitle">Chance of finishing as the overall WR1</div></div><div class="chart-badge">WR1</div></div><div class="chart-body"><svg id="chart-wr1" viewBox="0 0 560 240"></svg></div><div class="chart-footer" id="footer-wr1">{escape(chart_insights['wr1_rate'])}</div></div>
</section>
<section class="threshold-grid"><div class="table-wrap"><h2>Opportunity Thresholds</h2><table><thead><tr><th>Outcome</th><th>Minimum Bucket</th><th>Observed Rate</th></tr></thead><tbody>{_threshold_rows(summary)}</tbody></table></div><div class="conclusion"><h2>Conclusion</h2><ul>{''.join(f'<li>{escape(item)}</li>' for item in conclusion)}</ul></div></section>
<section class="table-wrap"><h2>Bucket Outcome Metrics</h2><table><thead><tr><th>Bucket</th><th>Sample</th><th>Avg PPG</th><th>Median PPG</th><th>Top 36</th><th>Top 24</th><th>Top 12</th><th>Top 5</th><th>WR1 Overall</th><th>{escape(config.score_label)}</th></tr></thead><tbody>{_summary_rows(summary, config)}</tbody></table></section>
<section class="table-wrap"><h2>Calculated WR Season Fields</h2><div class="table-controls"><div class="control"><label for="filter-player">Player / Team</label><input id="filter-player" type="search"></div><div class="control"><label for="filter-bucket">Bucket</label><select id="filter-bucket"><option value="">All</option></select></div><div class="control"><label for="filter-season-min">Season From</label><input id="filter-season-min" type="number"></div><div class="control"><label for="filter-season-max">Season To</label><input id="filter-season-max" type="number"></div><div class="control"><label for="filter-targets-min">Target Floor</label><input id="filter-targets-min" type="number"></div><div class="control"><label for="filter-metric-min">{escape(filter_label)}</label><input id="filter-metric-min" type="number" step="0.1"></div><div class="control"><label for="filter-finish-max">Max Finish</label><input id="filter-finish-max" type="number"></div><button class="icon-button" id="reset-season-table" type="button" title="Reset filters">x</button></div><div class="table-status" id="season-table-status"></div><table class="interactive-table"><thead><tr><th data-sort="player_name">Player</th><th data-sort="season">Season</th><th data-sort="team">Team</th><th data-sort="age">Age</th><th data-sort="games">Games</th><th data-sort="targets">Targets</th><th data-sort="receiving_tds">Rec TD</th><th data-sort="metric_value">{escape(config.metric_label)}</th><th data-sort="{efficiency_key}">{escape(efficiency_label)}</th><th data-sort="fantasy_points">Half PPR Points</th><th data-sort="fantasy_ppg">PPG</th><th data-sort="positional_finish">WR Finish</th></tr></thead><tbody id="season-table-body"></tbody></table></section>
</main>
<script>
const bucketData = {json.dumps(chart_records, allow_nan=False)};
const scatterData = {json.dumps(scatter_records, allow_nan=False)};
const outlierData = {json.dumps(outlier_records, allow_nan=False)};
const seasonTableData = {json.dumps(table_records, allow_nan=False)};
const pearsonR = {r:.6f};
const metricLabel = {json.dumps(config.metric_label)};
const metricUnit = {json.dumps(config.metric_unit)};
const opportunityZones = {json.dumps(list(config.zones), allow_nan=False)};
const thresholdMarkers = {json.dumps(config.threshold_markers, allow_nan=False)};
const efficiencyKey = {json.dumps(efficiency_key)};
const efficiencyDigits = efficiencyKey === "ppg_per_target" ? 3 : 2;
function node(svg,name,attrs,text){{const el=document.createElementNS("http://www.w3.org/2000/svg",name); Object.entries(attrs||{{}}).forEach(([k,v])=>el.setAttribute(k,v)); if(text!==undefined)el.textContent=text; svg.appendChild(el); return el;}}
function drawOpportunityCurve(id,metric,suffix,className){{const svg=document.getElementById(id),w=560,h=240,m={{top:28,right:30,bottom:58,left:48}},iw=w-m.left-m.right,ih=h-m.top-m.bottom,values=bucketData.map(d=>Number(d[metric])||0),max=Math.max(...values,1),yMax=metric==="average_fantasy_ppg"?Math.ceil(max/5)*5:Math.max(10,Math.ceil(max/10)*10),x=i=>m.left+(bucketData.length===1?iw/2:(i/(bucketData.length-1))*iw),y=v=>m.top+ih-(v/yMax)*ih; svg.innerHTML=""; opportunityZones.forEach(z=>{{const zx1=x(Math.max(0,Number(z.start))),zx2=x(Math.min(bucketData.length-1,Number(z.end)-1)),pad=iw/(bucketData.length-1)/2; node(svg,"rect",{{x:Math.max(m.left,zx1-pad),y:m.top,width:Math.min(w-m.right,zx2+pad)-Math.max(m.left,zx1-pad),height:ih,class:z.class}});}}); for(let i=0;i<=4;i++){{const v=yMax*i/4,yy=y(v); node(svg,"line",{{x1:m.left,x2:m.left+iw,y1:yy,y2:yy,class:"grid"}}); node(svg,"text",{{x:m.left-8,y:yy+4,class:"tick","text-anchor":"end"}},suffix==="%"?v.toFixed(0)+"%":v.toFixed(1));}} let biggest={{index:0,increase:-Infinity}}; for(let i=0;i<values.length-1;i++){{const inc=values[i+1]-values[i]; if(inc>biggest.increase)biggest={{index:i,increase:inc}};}} const line=values.map((v,i)=>`${{i===0?"M":"L"}} ${{x(i)}} ${{y(v)}}`).join(" "),area=`M ${{x(0)}} ${{y(0)}} `+values.map((v,i)=>`L ${{x(i)}} ${{y(v)}}`).join(" ")+` L ${{x(values.length-1)}} ${{y(0)}} Z`; node(svg,"path",{{d:area,class:"curve-area"}}); node(svg,"path",{{d:line,class:"curve-line"}}); const threshold=thresholdMarkers[metric]; if(threshold){{const idx=bucketData.findIndex(d=>d.bucket===threshold.bucket); if(idx>=0){{const xx=x(idx); node(svg,"line",{{x1:xx,x2:xx,y1:m.top,y2:m.top+ih,class:"threshold-line"}});}}}} if(biggest.increase>0){{const x1=x(biggest.index),x2=x(biggest.index+1); node(svg,"line",{{x1,x2,y1:m.top+8,y2:m.top+8,class:"cliff-line"}});}} bucketData.forEach((d,i)=>{{const v=values[i],xx=x(i),yy=y(v),dot=node(svg,"circle",{{cx:xx,cy:yy,r:5.5,class:"marker-dot "+className}}); const t=document.createElementNS("http://www.w3.org/2000/svg","title"); t.textContent=`Bucket: ${{d.bucket}}\\nSample Size: ${{Number(d.sample_size).toLocaleString()}}\\nAverage PPG: ${{Number(d.average_fantasy_ppg).toFixed(1)}}\\nTop 24 Rate: ${{Number(d.top24_rate).toFixed(1)}}%\\nTop 12 Rate: ${{Number(d.top12_rate).toFixed(1)}}%\\nTop 5 Rate: ${{Number(d.top5_rate).toFixed(1)}}%\\nWR1 Rate: ${{Number(d.wr1_rate).toFixed(2)}}%`; dot.appendChild(t); const labelText=v.toFixed(metric==="wr1_rate"?2:1)+suffix,labelW=Math.max(28,labelText.length*6+10),labelX=Math.min(w-m.right-labelW/2,Math.max(m.left+labelW/2,xx)),above=yy>m.top+30,labelY=above?yy-18:yy+18; node(svg,"rect",{{x:labelX-labelW/2,y:labelY-11,width:labelW,height:15,rx:4,class:"value-label-bg"}}); node(svg,"text",{{x:labelX,y:labelY,class:"value-label","text-anchor":"middle"}},labelText); node(svg,"text",{{x:xx,y:m.top+ih+23,class:"tick","text-anchor":"middle"}},d.bucket); node(svg,"text",{{x:xx,y:m.top+ih+38,class:"sample-label","text-anchor":"middle"}},`n=${{Number(d.sample_size).toLocaleString()}}`);}}); node(svg,"line",{{x1:m.left,x2:m.left,y1:m.top,y2:m.top+ih,class:"axis"}}); node(svg,"line",{{x1:m.left,x2:m.left+iw,y1:m.top+ih,y2:m.top+ih,class:"axis"}});}}
function drawScatter(){{const svg=document.getElementById("chart-correlation"),w=1160,h=330,m={{top:20,right:34,bottom:50,left:66}},iw=w-m.left-m.right,ih=h-m.top-m.bottom,points=scatterData.map(d=>({{...d,x:Number(d.metric_value),y:Number(d.fantasy_ppg),finish:Number(d.positional_finish)}})).filter(d=>Number.isFinite(d.x)&&Number.isFinite(d.y)); svg.innerHTML=""; const step=metricUnit==="%"?5:25,minMax=metricUnit==="%"?35:150,xMax=Math.max(minMax,Math.ceil(Math.max(...points.map(d=>d.x),0)/step)*step),yMax=Math.max(30,Math.ceil(Math.max(...points.map(d=>d.y),0)/5)*5),x=v=>m.left+(v/xMax)*iw,y=v=>m.top+ih-(v/yMax)*ih; for(let i=0;i<=7;i++){{const v=xMax*i/7,xx=x(v); node(svg,"line",{{x1:xx,x2:xx,y1:m.top,y2:m.top+ih,class:"grid"}}); node(svg,"text",{{x:xx,y:m.top+ih+24,class:"tick","text-anchor":"middle"}},metricUnit==="%"?v.toFixed(0)+"%":v.toFixed(0));}} for(let i=0;i<=5;i++){{const v=yMax*i/5,yy=y(v); node(svg,"line",{{x1:m.left,x2:m.left+iw,y1:yy,y2:yy,class:"grid"}}); node(svg,"text",{{x:m.left-10,y:yy+4,class:"tick","text-anchor":"end"}},v.toFixed(0));}} const n=points.length||1,meanX=points.reduce((s,d)=>s+d.x,0)/n,meanY=points.reduce((s,d)=>s+d.y,0)/n,varX=points.reduce((s,d)=>s+Math.pow(d.x-meanX,2),0),cov=points.reduce((s,d)=>s+(d.x-meanX)*(d.y-meanY),0),sl=varX?cov/varX:0,int=meanY-sl*meanX; node(svg,"line",{{x1:x(0),y1:y(Math.max(0,Math.min(yMax,int))),x2:x(xMax),y2:y(Math.max(0,Math.min(yMax,int+sl*xMax))),class:"trend-line"}}); points.forEach(d=>{{const c=node(svg,"circle",{{cx:x(d.x),cy:y(d.y),r:d.finish<=12?3.2:d.finish<=24?2.7:2.1,class:"scatter-point"}}); const t=document.createElementNS("http://www.w3.org/2000/svg","title"); t.textContent=`${{d.player_name}} (${{d.season}}, ${{d.team}}): ${{metricUnit==="%"?d.x.toFixed(1)+"%":d.x.toFixed(0)}} ${{metricLabel.toLowerCase()}}, ${{d.y.toFixed(1)}} PPG, WR${{d.finish.toFixed(0)}}`; c.appendChild(t);}}); outlierData.forEach((d,i)=>{{const sx=x(Number(d.metric_value)),sy=y(Number(d.fantasy_ppg)),lx=Math.max(m.left+4,Math.min(m.left+iw-138,sx+(sx<m.left+iw*.66?12:-138))),ly=Math.max(m.top+8,Math.min(m.top+ih-18,sy+(i%2===0?-22:18))); node(svg,"circle",{{cx:sx,cy:sy,r:5,class:"scatter-outlier"}}); node(svg,"text",{{x:lx,y:ly+2,class:"outlier-label"}},`${{d.player_name}} ${{d.season}}`);}}); node(svg,"line",{{x1:m.left,x2:m.left,y1:m.top,y2:m.top+ih,class:"axis"}}); node(svg,"line",{{x1:m.left,x2:m.left+iw,y1:m.top+ih,y2:m.top+ih,class:"axis"}}); node(svg,"text",{{x:m.left+iw/2,y:h-10,class:"svg-label","text-anchor":"middle"}},metricLabel); node(svg,"text",{{x:20,y:m.top+ih/2,class:"svg-label","text-anchor":"middle",transform:`rotate(-90 20 ${{m.top+ih/2}})`}},"Fantasy PPG");}}
function setupTable(){{const body=document.getElementById("season-table-body"),status=document.getElementById("season-table-status"),controls={{search:document.getElementById("filter-player"),bucket:document.getElementById("filter-bucket"),seasonMin:document.getElementById("filter-season-min"),seasonMax:document.getElementById("filter-season-max"),targetsMin:document.getElementById("filter-targets-min"),metricMin:document.getElementById("filter-metric-min"),finishMax:document.getElementById("filter-finish-max"),reset:document.getElementById("reset-season-table")}},sortState=[{{key:"season",direction:"desc"}},{{key:"positional_finish",direction:"asc"}}],numberKeys=new Set(["season","age","games","targets","receiving_tds","metric_value","ppg_per_ts","ppg_per_target","fantasy_points","fantasy_ppg","positional_finish"]),bucketOrder=new Map(bucketData.map((r,i)=>[r.bucket,i])); [...new Set(seasonTableData.map(r=>r.bucket))].sort((a,b)=>(bucketOrder.get(a)??99)-(bucketOrder.get(b)??99)).forEach(b=>{{const o=document.createElement("option"); o.value=b; o.textContent=b; controls.bucket.appendChild(o);}}); const seasons=seasonTableData.map(r=>Number(r.season)); controls.seasonMin.placeholder=String(Math.min(...seasons)); controls.seasonMax.placeholder=String(Math.max(...seasons)); const fmt=(v,d=1)=>{{const n=Number(v); return Number.isFinite(n)?n.toFixed(d):"";}}; function matches(r){{const q=controls.search.value.trim().toLowerCase(); if(q&&!`${{r.player_name}} ${{r.team}}`.toLowerCase().includes(q))return false; if(controls.bucket.value&&r.bucket!==controls.bucket.value)return false; const s=Number(r.season),smin=Number(controls.seasonMin.value),smax=Number(controls.seasonMax.value),tmin=Number(controls.targetsMin.value),mmin=Number(controls.metricMin.value),fmax=Number(controls.finishMax.value); if(Number.isFinite(smin)&&controls.seasonMin.value&&s<smin)return false; if(Number.isFinite(smax)&&controls.seasonMax.value&&s>smax)return false; if(Number.isFinite(tmin)&&controls.targetsMin.value&&Number(r.targets)<tmin)return false; if(Number.isFinite(mmin)&&controls.metricMin.value&&Number(r.metric_value)<mmin)return false; if(Number.isFinite(fmax)&&controls.finishMax.value&&Number(r.positional_finish)>fmax)return false; return true;}} function cmp(a,b){{for(const s of sortState){{let av=a[s.key],bv=b[s.key]; if(s.key==="bucket"){{av=bucketOrder.get(av)??99; bv=bucketOrder.get(bv)??99;}} else if(numberKeys.has(s.key)){{av=Number(av); bv=Number(bv);}} else {{av=String(av).toLowerCase(); bv=String(bv).toLowerCase();}} if(av<bv)return s.direction==="asc"?-1:1; if(av>bv)return s.direction==="asc"?1:-1;}} return 0;}} function render(){{const rows=seasonTableData.filter(matches).sort(cmp); body.innerHTML=rows.map(r=>`<tr><td>${{r.player_name}}</td><td>${{r.season}}</td><td>${{r.team}}</td><td>${{fmt(r.age,1)}}</td><td>${{fmt(r.games,0)}}</td><td>${{Number(r.targets).toLocaleString()}}</td><td>${{Number(r.receiving_tds).toLocaleString()}}</td><td>${{r.metric_display}}</td><td>${{fmt(r[efficiencyKey],efficiencyDigits)}}</td><td>${{fmt(r.fantasy_points,1)}}</td><td>${{fmt(r.fantasy_ppg,1)}}</td><td>${{r.positional_finish}}</td></tr>`).join(""); status.textContent=`${{rows.length.toLocaleString()}} of ${{seasonTableData.length.toLocaleString()}} player seasons`; document.querySelectorAll(".interactive-table th").forEach(th=>{{const idx=sortState.findIndex(s=>s.key===th.dataset.sort); th.classList.toggle("sort-active",idx>=0); th.textContent=th.textContent.replace(/ [▲▼][0-9]?$/,""); if(idx>=0) th.textContent+=` ${{sortState[idx].direction==="asc"?"▲":"▼"}}${{sortState.length>1?idx+1:""}}`;}});}} Object.values(controls).forEach(c=>{{if(c&&c!==controls.reset)c.addEventListener("input",render);}}); controls.reset.addEventListener("click",()=>{{Object.values(controls).forEach(c=>{{if(c&&c!==controls.reset)c.value="";}}); sortState.splice(0,sortState.length,{{key:"season",direction:"desc"}},{{key:"positional_finish",direction:"asc"}}); render();}}); document.querySelectorAll(".interactive-table th").forEach(th=>th.addEventListener("click",e=>{{const key=th.dataset.sort,idx=sortState.findIndex(s=>s.key===key); if(!e.shiftKey){{const dir=idx===0&&sortState[0].direction==="asc"?"desc":"asc"; sortState.splice(0,sortState.length,{{key,direction:dir}});}} else if(idx>=0) sortState[idx].direction=sortState[idx].direction==="asc"?"desc":"asc"; else sortState.push({{key,direction:numberKeys.has(key)?"desc":"asc"}}); render();}})); render();}}
drawScatter(); setupTable(); drawOpportunityCurve("chart-ppg","average_fantasy_ppg","",""); drawOpportunityCurve("chart-top24","top24_rate","%",""); drawOpportunityCurve("chart-top12","top12_rate","%","top12"); drawOpportunityCurve("chart-top5","top5_rate","%","top5"); drawOpportunityCurve("chart-wr1","wr1_rate","%","wr1");
</script>
</body>
</html>
"""


def write_dashboard(config: StudyConfig, base_df: pd.DataFrame, min_targets: int | None = None) -> Path:
    floor = config.min_targets if min_targets is None else int(min_targets)
    df = apply_min_targets(base_df, floor)
    config = StudyConfig(**{**config.__dict__, "min_targets": floor})
    comparison_html = ""
    winner = ""
    if config.slug == "opportunity_wr_targets":
        comparison_html, winner = comparison_section(df)
    html = render_dashboard(config, df, comparison_html, winner)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / config.output_name
    output_path.write_text(html, encoding="utf-8")
    root_path = TARGETS_ROOT_OUTPUT if config.slug == "opportunity_wr_targets" else TARGET_SHARE_ROOT_OUTPUT
    root_path.write_text(html, encoding="utf-8")
    return output_path


def build_all(force_refresh: bool = False, min_targets: int | None = None, study: str = "all") -> list[Path]:
    base_df = pull_wr_opportunity_df(force_refresh=force_refresh)
    paths = []
    if study in {"target_share", "all"}:
        paths.append(write_dashboard(TARGET_SHARE_CONFIG, base_df, min_targets))
    if study in {"targets", "all"}:
        targets_min = min_targets if min_targets is not None else TARGETS_CONFIG.min_targets
        paths.append(write_dashboard(TARGETS_CONFIG, base_df, targets_min))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Opportunity Modeling research dashboards.")
    parser.add_argument("--force-refresh", action="store_true", help="Redownload nflverse source data.")
    parser.add_argument("--min-targets", type=int, default=None, help="Optional minimum season target floor.")
    parser.add_argument("--study", choices=["target_share", "targets", "all"], default="all")
    args = parser.parse_args()
    for path in build_all(force_refresh=args.force_refresh, min_targets=args.min_targets, study=args.study):
        print(path)
    if args.study in {"target_share", "all"}:
        print(TARGET_SHARE_ROOT_OUTPUT)
    if args.study in {"targets", "all"}:
        print(TARGETS_ROOT_OUTPUT)


if __name__ == "__main__":
    main()
