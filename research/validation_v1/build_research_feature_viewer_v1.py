from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALIDATION_DIR = Path(__file__).resolve().parent
INPUT = VALIDATION_DIR / "predraft_validation_dataset_market_disagreement_score.csv"
OUTPUT = VALIDATION_DIR / "research_feature_viewer_v1.html"

DISPLAY_SEASON_MIN = 2014
DISPLAY_SEASON_MAX = 2024

MAIN_TARGETS = [
    "WR_Underpriced_Top24",
    "RB_Underpriced_Top24",
    "WR_Beat_ADP_By_12",
    "RB_Beat_ADP_By_12",
    "WR_Underpriced_Top12",
    "RB_Underpriced_Top12",
]

ADP_CANDIDATES = ["overall_adp", "adp", "preseason_adp", "adp_rank"]
POSITIONAL_ADP_CANDIDATES = ["positional_adp", "pos_adp", "position_adp", "positional_adp_rank"]
PLAYER_CANDIDATES = ["player_name", "player", "name", "player_display_name"]
POSITION_CANDIDATES = ["position", "pos", "fantasy_position"]
TEAM_CANDIDATES = ["team", "projected_team", "adp_team", "nfl_team"]
SEASON_CANDIDATES = ["season", "year"]


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return str(value)


def pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def clean_position(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().strip()


def table_html(df: pd.DataFrame, classes: str = "data-table") -> str:
    if df is None or df.empty:
        return '<p class="muted">No rows available.</p>'

    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: fmt(x))
        else:
            out[col] = out[col].fillna("").astype(str)

    return out.to_html(index=False, escape=True, classes=classes, border=0)


def detect_scores(df: pd.DataFrame) -> list[str]:
    score_cols = []

    for col in df.columns:
        lower = col.lower()
        if lower.endswith("_score") or "score" in lower:
            if safe_numeric(df[col]).notna().any():
                score_cols.append(col)

    preferred = [
        "market_disagreement_score",
        "age_curve_edge_score",
        "market_disagreement_component_count",
    ]

    ordered = [col for col in preferred if col in score_cols]
    ordered.extend([col for col in score_cols if col not in ordered])

    return ordered


def detect_targets(df: pd.DataFrame) -> list[str]:
    found = [target for target in MAIN_TARGETS if target in df.columns]

    for col in df.columns:
        if col in found:
            continue

        if re.search(r"Beat_ADP|Underpriced|Breakout|Tier_Jump", col, re.I):
            lower = col.lower()
            if not lower.startswith(("age_curve_", "market_disagreement_", "late_")):
                found.append(col)

    return found


def detect_component_columns(df: pd.DataFrame) -> list[str]:
    patterns = [
        "component",
        "gap",
        "vs_adp",
        "disagreement",
        "projection_rank",
        "rank_minus",
        "residual",
        "projected_",
    ]

    cols = []

    for col in df.columns:
        lower = col.lower()

        if any(p in lower for p in patterns) and safe_numeric(df[col]).notna().any():
            if not re.search(r"beat_adp|underpriced|top12|top24|final_|finish", lower):
                cols.append(col)

    preferred = [
        "market_disagreement_score",
        "market_disagreement_component_count",
        "projected_volume_score",
        "projected_touch_score",
        "projected_receiving_role_score",
    ]

    ordered = [col for col in preferred if col in cols]
    ordered.extend([col for col in cols if col not in ordered])

    return ordered


def classify_component(col: str) -> str:
    lower = col.lower()

    if any(x in lower for x in ["td", "touchdown"]):
        return "TD-related"
    if any(x in lower for x in ["rec", "receiving", "target"]):
        return "Receiving-related"
    if any(x in lower for x in ["rush", "carry", "carries"]):
        return "Rushing-related"
    if any(x in lower for x in ["volume", "touch", "opportunity"]):
        return "Volume-related"
    if any(x in lower for x in ["role", "route", "snap"]):
        return "Role-related"
    if any(x in lower for x in ["adp", "rank", "gap", "disagreement"]):
        return "Market-gap-related"

    return "Unknown"


def summarize_score(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    s = safe_numeric(df[score_col])
    valid = s.dropna()

    return {
        "score": score_col,
        "valid_rows": int(valid.shape[0]),
        "missing_rate": 1 - (valid.shape[0] / len(df)) if len(df) else np.nan,
        "mean": valid.mean() if not valid.empty else np.nan,
        "median": valid.median() if not valid.empty else np.nan,
        "min": valid.min() if not valid.empty else np.nan,
        "max": valid.max() if not valid.empty else np.nan,
        "p90": valid.quantile(0.90) if not valid.empty else np.nan,
    }


def score_cards_html(score_summary: pd.DataFrame) -> str:
    if score_summary.empty:
        return '<p class="muted">No score columns detected.</p>'

    cards = []

    for _, row in score_summary.iterrows():
        cards.append(
            f"""
            <div class="card">
                <h3>{html.escape(str(row["score"]))}</h3>
                <div class="metric-row"><span>Valid rows</span><strong>{fmt(row["valid_rows"], 0)}</strong></div>
                <div class="metric-row"><span>Missing rate</span><strong>{pct(row["missing_rate"])}</strong></div>
                <div class="metric-row"><span>Mean</span><strong>{fmt(row["mean"])}</strong></div>
                <div class="metric-row"><span>Median</span><strong>{fmt(row["median"])}</strong></div>
                <div class="metric-row"><span>Min</span><strong>{fmt(row["min"])}</strong></div>
                <div class="metric-row"><span>Max</span><strong>{fmt(row["max"])}</strong></div>
                <div class="metric-row"><span>90th percentile</span><strong>{fmt(row["p90"])}</strong></div>
            </div>
            """
        )

    return "\n".join(cards)


def make_top_table(
    df: pd.DataFrame,
    score_col: str,
    position_col: str | None,
    position: str | None,
    player_col: str | None,
    season_col: str | None,
    team_col: str | None,
    adp_col: str | None,
    pos_adp_col: str | None,
    targets: list[str],
    n: int = 25,
) -> pd.DataFrame:
    if score_col not in df.columns:
        return pd.DataFrame()

    view = df.copy()

    if position and position_col:
        view = view[view[position_col].map(clean_position) == position]

    view["_score_sort"] = safe_numeric(view[score_col])
    view = view[view["_score_sort"].notna()]
    view = view.sort_values("_score_sort", ascending=False).head(n)

    cols = []

    for col in [player_col, season_col, position_col, team_col, adp_col, pos_adp_col]:
        if col and col in view.columns and col not in cols:
            cols.append(col)

    for col in [
        "market_disagreement_score",
        "market_disagreement_bucket",
        "market_disagreement_note",
        "age_curve_edge_score",
        "age_curve_edge_bucket",
    ]:
        if col in view.columns and col not in cols:
            cols.append(col)

    for target in targets:
        if target in view.columns and target not in cols:
            cols.append(target)

    if not cols:
        return pd.DataFrame()

    return view[cols].copy()


def make_bucket_summary(
    df: pd.DataFrame,
    position_col: str | None,
    bucket_col: str,
    score_col: str,
    adp_col: str | None,
    targets: list[str],
) -> pd.DataFrame:
    if bucket_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    group_cols = []

    if position_col and position_col in temp.columns:
        temp[position_col] = temp[position_col].map(clean_position)
        group_cols.append(position_col)

    group_cols.append(bucket_col)

    agg = {
        score_col: ["count", "mean"],
    }

    if adp_col and adp_col in temp.columns:
        temp[adp_col] = safe_numeric(temp[adp_col])
        agg[adp_col] = ["mean"]

    for target in targets:
        if target in temp.columns:
            temp[target] = safe_numeric(temp[target])
            agg[target] = ["mean"]

    summary = temp.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns]
    summary = summary.reset_index()

    rename = {
        f"{score_col}_count": "rows",
        f"{score_col}_mean": f"avg_{score_col}",
    }

    if adp_col:
        rename[f"{adp_col}_mean"] = "avg_adp"

    for target in targets:
        rename[f"{target}_mean"] = f"{target}_hit_rate"

    summary = summary.rename(columns=rename)

    if "rows" in summary.columns:
        summary = summary.sort_values(["rows"], ascending=False)

    return summary


def make_component_summary(
    df: pd.DataFrame,
    component_cols: list[str],
    market_score_col: str = "market_disagreement_score",
) -> pd.DataFrame:
    rows = []

    market_score = safe_numeric(df[market_score_col]) if market_score_col in df.columns else None

    for col in component_cols:
        s = safe_numeric(df[col])
        valid = s.notna()

        corr = np.nan
        if market_score is not None:
            both = valid & market_score.notna()
            if both.sum() >= 5 and s[both].nunique() > 1 and market_score[both].nunique() > 1:
                corr = s[both].corr(market_score[both])

        rows.append(
            {
                "component": col,
                "type": classify_component(col),
                "valid_rows": int(valid.sum()),
                "missing_rate": 1 - (valid.sum() / len(df)) if len(df) else np.nan,
                "mean": s.mean(),
                "median": s.median(),
                "corr_with_market_disagreement": corr,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["corr_with_market_disagreement", "valid_rows"],
            ascending=[False, False],
            na_position="last",
        )

    return out


def make_hit_miss_summary(
    df: pd.DataFrame,
    targets: list[str],
    compare_cols: list[str],
    position_col: str | None,
) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}

    for target in targets:
        if target not in df.columns:
            continue

        temp = df.copy()
        temp[target] = safe_numeric(temp[target])

        temp = temp[temp[target].isin([0, 1])]
        if temp.empty:
            continue

        rows = []

        for pos in ["WR", "RB", "TE", "QB", "ALL"]:
            if pos != "ALL" and position_col:
                sub = temp[temp[position_col].map(clean_position) == pos]
            else:
                sub = temp

            if sub.empty:
                continue

            for col in compare_cols:
                if col not in sub.columns:
                    continue

                values = safe_numeric(sub[col])
                hits = values[sub[target] == 1]
                misses = values[sub[target] == 0]

                rows.append(
                    {
                        "target": target,
                        "position": pos,
                        "metric": col,
                        "rows": int(values.notna().sum()),
                        "hit_count": int((sub[target] == 1).sum()),
                        "miss_count": int((sub[target] == 0).sum()),
                        "hit_avg": hits.mean(),
                        "miss_avg": misses.mean(),
                        "hit_minus_miss": hits.mean() - misses.mean(),
                    }
                )

        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary.sort_values(
                ["position", "hit_minus_miss"],
                ascending=[True, False],
                na_position="last",
            )
            summaries[target] = summary

    return summaries


def make_quality_flags(
    df: pd.DataFrame,
    player_col: str | None,
    season_col: str | None,
    position_col: str | None,
    adp_col: str | None,
    score_cols: list[str],
    component_cols: list[str],
) -> pd.DataFrame:
    rows = []

    checks = {
        "missing_player_name": player_col,
        "missing_season": season_col,
        "missing_position": position_col,
        "missing_adp": adp_col,
    }

    for label, col in checks.items():
        if col and col in df.columns:
            missing = df[col].isna().sum()
            rows.append(
                {
                    "check": label,
                    "column": col,
                    "rows_affected": int(missing),
                    "rate": missing / len(df) if len(df) else np.nan,
                }
            )
        else:
            rows.append(
                {
                    "check": label,
                    "column": "not detected",
                    "rows_affected": len(df),
                    "rate": 1.0,
                }
            )

    for col in score_cols + component_cols:
        if col not in df.columns:
            continue

        missing = safe_numeric(df[col]).isna().sum()
        rows.append(
            {
                "check": "numeric_missingness",
                "column": col,
                "rows_affected": int(missing),
                "rate": missing / len(df) if len(df) else np.nan,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["rate", "rows_affected"], ascending=[False, False])

    return out


def apply_display_season_filter(
    df: pd.DataFrame,
    season_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    original_rows = len(df)

    info = {
        "original_rows": original_rows,
        "filtered_rows": original_rows,
        "season_filter_applied": False,
        "season_filter_warning": "",
        "seasons_displayed": [],
    }

    if not season_col or season_col not in df.columns:
        info["season_filter_warning"] = "Season column not detected; viewer could not filter to 2014-2024."
        return df.copy(), info

    filtered = df.copy()
    filtered["_season_numeric_for_viewer"] = safe_numeric(filtered[season_col])

    filtered = filtered[
        (filtered["_season_numeric_for_viewer"] >= DISPLAY_SEASON_MIN)
        & (filtered["_season_numeric_for_viewer"] <= DISPLAY_SEASON_MAX)
    ].copy()

    if "_season_numeric_for_viewer" in filtered.columns:
        seasons = (
            filtered["_season_numeric_for_viewer"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        filtered = filtered.drop(columns=["_season_numeric_for_viewer"])

    else:
        seasons = []

    info["filtered_rows"] = len(filtered)
    info["season_filter_applied"] = True
    info["seasons_displayed"] = seasons

    return filtered, info


def build_html(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    filter_info: dict[str, Any],
    player_col: str | None,
    season_col: str | None,
    position_col: str | None,
    team_col: str | None,
    adp_col: str | None,
    pos_adp_col: str | None,
    score_cols: list[str],
    targets: list[str],
    component_cols: list[str],
) -> str:
    seasons = filter_info.get("seasons_displayed", [])

    positions = []
    if position_col and position_col in df.columns:
        positions = sorted(df[position_col].map(clean_position).replace("", np.nan).dropna().unique().tolist())

    score_summary = pd.DataFrame([summarize_score(df, col) for col in score_cols])
    component_summary = make_component_summary(df, component_cols)

    market_bucket_summary = make_bucket_summary(
        df=df,
        position_col=position_col,
        bucket_col="market_disagreement_bucket",
        score_col="market_disagreement_score",
        adp_col=adp_col,
        targets=targets,
    )

    age_bucket_summary = make_bucket_summary(
        df=df,
        position_col=position_col,
        bucket_col="age_curve_edge_bucket",
        score_col="age_curve_edge_score",
        adp_col=adp_col,
        targets=targets,
    )

    top_wr_market = make_top_table(
        df, "market_disagreement_score", position_col, "WR", player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )
    top_rb_market = make_top_table(
        df, "market_disagreement_score", position_col, "RB", player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )
    top_overall_market = make_top_table(
        df, "market_disagreement_score", position_col, None, player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )

    top_wr_age = make_top_table(
        df, "age_curve_edge_score", position_col, "WR", player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )
    top_rb_age = make_top_table(
        df, "age_curve_edge_score", position_col, "RB", player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )
    top_overall_age = make_top_table(
        df, "age_curve_edge_score", position_col, None, player_col, season_col,
        team_col, adp_col, pos_adp_col, targets
    )

    compare_cols = [
        "market_disagreement_score",
        "age_curve_edge_score",
    ]

    if adp_col:
        compare_cols.append(adp_col)

    compare_cols.extend(component_cols[:20])
    compare_cols = [c for c in dict.fromkeys(compare_cols) if c in df.columns]

    hit_miss = make_hit_miss_summary(df, targets, compare_cols, position_col)
    quality_flags = make_quality_flags(
        df=df,
        player_col=player_col,
        season_col=season_col,
        position_col=position_col,
        adp_col=adp_col,
        score_cols=score_cols,
        component_cols=component_cols,
    )

    if filter_info.get("season_filter_warning"):
        season_note = f"""
        <div class="warning">
            {html.escape(filter_info["season_filter_warning"])}
        </div>
        """
    else:
        season_note = f"""
        <div class="note">
            Viewer filtered to {DISPLAY_SEASON_MIN}–{DISPLAY_SEASON_MAX} because FantasyPros Wayback projection data is only available for those seasons.
        </div>
        """

    hit_miss_sections = []

    for target, summary in hit_miss.items():
        hit_miss_sections.append(
            f"""
            <h3>{html.escape(target)}</h3>
            {table_html(summary.head(50))}
            """
        )

    hit_miss_html = "\n".join(hit_miss_sections) if hit_miss_sections else '<p class="muted">No target hit/miss summaries available.</p>'

    html_doc = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Guaranteed Play Research Feature Viewer V1</title>
        <style>
            :root {{
                --bg: #f4f6f8;
                --card: #ffffff;
                --text: #17202a;
                --muted: #667085;
                --line: #d9dee7;
                --accent: #1f5eff;
                --warning-bg: #fff3cd;
                --warning-text: #7a5200;
                --note-bg: #eaf2ff;
                --note-text: #123c7c;
            }}

            body {{
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: var(--bg);
                color: var(--text);
            }}

            header {{
                background: #101828;
                color: white;
                padding: 28px 36px;
            }}

            header h1 {{
                margin: 0 0 8px 0;
                font-size: 30px;
            }}

            header p {{
                margin: 0;
                color: #d0d5dd;
            }}

            main {{
                max-width: 1500px;
                margin: 0 auto;
                padding: 28px 36px 60px 36px;
            }}

            section {{
                margin-bottom: 34px;
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 22px;
                box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
            }}

            h2 {{
                margin-top: 0;
                border-bottom: 1px solid var(--line);
                padding-bottom: 10px;
            }}

            h3 {{
                margin-top: 22px;
            }}

            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 14px;
            }}

            .summary-box {{
                background: #f9fafb;
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 14px;
            }}

            .summary-box span {{
                display: block;
                color: var(--muted);
                font-size: 13px;
                margin-bottom: 6px;
            }}

            .summary-box strong {{
                font-size: 19px;
            }}

            .card-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
            }}

            .card {{
                background: #f9fafb;
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 16px;
            }}

            .card h3 {{
                margin: 0 0 12px 0;
                font-size: 17px;
                color: var(--accent);
            }}

            .metric-row {{
                display: flex;
                justify-content: space-between;
                gap: 14px;
                border-bottom: 1px solid #eaecf0;
                padding: 6px 0;
                font-size: 14px;
            }}

            .metric-row:last-child {{
                border-bottom: 0;
            }}

            .metric-row span {{
                color: var(--muted);
            }}

            .table-wrap {{
                overflow-x: auto;
                margin-top: 12px;
            }}

            table.data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}

            table.data-table th {{
                background: #f2f4f7;
                color: #344054;
                text-align: left;
                padding: 9px;
                border-bottom: 1px solid var(--line);
                position: sticky;
                top: 0;
            }}

            table.data-table td {{
                padding: 8px 9px;
                border-bottom: 1px solid #eaecf0;
                vertical-align: top;
            }}

            table.data-table tr:nth-child(even) {{
                background: #fbfcfd;
            }}

            .muted {{
                color: var(--muted);
            }}

            .note {{
                background: var(--note-bg);
                color: var(--note-text);
                border: 1px solid #b8d4ff;
                padding: 12px 14px;
                border-radius: 10px;
                margin: 14px 0;
            }}

            .warning {{
                background: var(--warning-bg);
                color: var(--warning-text);
                border: 1px solid #ffdf7e;
                padding: 12px 14px;
                border-radius: 10px;
                margin: 14px 0;
            }}

            .small {{
                font-size: 13px;
            }}

            code {{
                background: #eef2f6;
                padding: 2px 5px;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Guaranteed Play Research Feature Viewer V1</h1>
            <p>Research-only inspection dashboard for market disagreement, age curve edge, projection components, and target sanity checks.</p>
        </header>

        <main>
            <section>
                <h2>Executive Summary</h2>

                {season_note}

                <div class="summary-grid">
                    <div class="summary-box">
                        <span>Original rows loaded</span>
                        <strong>{fmt(filter_info.get("original_rows"), 0)}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Rows displayed after season filter</span>
                        <strong>{fmt(filter_info.get("filtered_rows"), 0)}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Seasons displayed</span>
                        <strong>{html.escape(", ".join(map(str, seasons)) if seasons else "Unknown")}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Positions displayed</span>
                        <strong>{html.escape(", ".join(positions) if positions else "Unknown")}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Score columns detected</span>
                        <strong>{len(score_cols)}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Target columns detected</span>
                        <strong>{len(targets)}</strong>
                    </div>
                    <div class="summary-box">
                        <span>Component columns detected</span>
                        <strong>{len(component_cols)}</strong>
                    </div>
                    <div class="summary-box">
                        <span>ADP column used</span>
                        <strong>{html.escape(adp_col or "Not detected")}</strong>
                    </div>
                </div>

                <p class="small muted">
                    Market Disagreement Score means: <strong>projection profile likes the player more than ADP does.</strong>
                    High score means potentially underpriced by the market. Low score means fairly priced or overpriced.
                    This viewer is research-only and does not imply app readiness.
                </p>
            </section>

            <section>
                <h2>Score Overview Cards</h2>
                <div class="card-grid">
                    {score_cards_html(score_summary)}
                </div>
            </section>

            <section>
                <h2>Top Players by Market Disagreement</h2>

                <h3>Top 25 WR</h3>
                <div class="table-wrap">{table_html(top_wr_market)}</div>

                <h3>Top 25 RB</h3>
                <div class="table-wrap">{table_html(top_rb_market)}</div>

                <h3>Top 25 Overall</h3>
                <div class="table-wrap">{table_html(top_overall_market)}</div>
            </section>

            <section>
                <h2>Top Players by Age Curve Edge</h2>

                <h3>Top 25 WR</h3>
                <div class="table-wrap">{table_html(top_wr_age)}</div>

                <h3>Top 25 RB</h3>
                <div class="table-wrap">{table_html(top_rb_age)}</div>

                <h3>Top 25 Overall</h3>
                <div class="table-wrap">{table_html(top_overall_age)}</div>
            </section>

            <section>
                <h2>Market Disagreement Bucket Summary</h2>
                <div class="table-wrap">{table_html(market_bucket_summary)}</div>
            </section>

            <section>
                <h2>Age Curve Edge Bucket Summary</h2>
                <div class="table-wrap">{table_html(age_bucket_summary)}</div>
            </section>

            <section>
                <h2>Component Breakdown</h2>
                <p class="muted">
                    This section shows which numeric projection/gap/component columns are present and how strongly they line up with the total market disagreement score.
                </p>
                <div class="table-wrap">{table_html(component_summary)}</div>
            </section>

            <section>
                <h2>Hit vs Miss Quick Check</h2>
                <p class="muted">
                    Simple averages only. No model is trained here.
                </p>
                {hit_miss_html}
            </section>

            <section>
                <h2>Data Quality Flags</h2>
                <div class="table-wrap">{table_html(quality_flags.head(80))}</div>
            </section>

            <section>
                <h2>Notes / Interpretation</h2>
                <p>
                    This viewer only displays <strong>{DISPLAY_SEASON_MIN}–{DISPLAY_SEASON_MAX}</strong>, because the FantasyPros Wayback projection data used for Market Disagreement exists for that era.
                    Earlier rows may exist in the source dataset, but they are excluded from this viewer because they are not meaningful for this research layer.
                </p>
                <p>
                    Market Disagreement Score should be treated as a research feature, not a final ranking.
                    It tells us where projection-based role or volume appears stronger than ADP price.
                    The next useful inspection is to determine whether the strongest rows are driven by stable volume components or noisy TD components.
                </p>
            </section>
        </main>
    </body>
    </html>
    """

    return html_doc


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    original_df = df.copy()

    player_col = detect_column(df, PLAYER_CANDIDATES)
    season_col = detect_column(df, SEASON_CANDIDATES)
    position_col = detect_column(df, POSITION_CANDIDATES)
    team_col = detect_column(df, TEAM_CANDIDATES)
    adp_col = detect_column(df, ADP_CANDIDATES)
    pos_adp_col = detect_column(df, POSITIONAL_ADP_CANDIDATES)

    df, filter_info = apply_display_season_filter(df, season_col)

    score_cols = detect_scores(df)
    targets = detect_targets(df)
    component_cols = detect_component_columns(df)

    html_doc = build_html(
        df=df,
        original_df=original_df,
        filter_info=filter_info,
        player_col=player_col,
        season_col=season_col,
        position_col=position_col,
        team_col=team_col,
        adp_col=adp_col,
        pos_adp_col=pos_adp_col,
        score_cols=score_cols,
        targets=targets,
        component_cols=component_cols,
    )

    OUTPUT.write_text(html_doc, encoding="utf-8")

    print("Research feature viewer generated.")
    print(f"Input: {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Original rows loaded: {filter_info.get('original_rows')}")
    print(f"Rows after 2014-2024 filter: {filter_info.get('filtered_rows')}")
    print(f"Season column: {season_col}")
    print(f"Seasons displayed: {filter_info.get('seasons_displayed')}")
    print(f"Player column: {player_col}")
    print(f"Position column: {position_col}")
    print(f"ADP column: {adp_col}")
    print(f"Positional ADP column: {pos_adp_col}")
    print(f"Score columns detected: {len(score_cols)}")
    print(f"Target columns detected: {len(targets)}")
    print(f"Component columns detected: {len(component_cols)}")

    if filter_info.get("season_filter_warning"):
        print(f"WARNING: {filter_info['season_filter_warning']}")


if __name__ == "__main__":
    main()