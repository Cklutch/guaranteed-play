from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import webbrowser

import pandas as pd

from rb_elite_age_analysis import build_rb_elite_age_study, pull_historical_seasons_df

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


OUTPUT_DIR = Path("case_studies/output/rb_elite_age_last10")
LAST_N_SEASONS = 10
SCORING_OPTIONS = {
    "1": ("ppr", "Full PPR"),
    "2": ("half_ppr", "Half PPR"),
    "ppr": ("ppr", "Full PPR"),
    "full": ("ppr", "Full PPR"),
    "full_ppr": ("ppr", "Full PPR"),
    "half": ("half_ppr", "Half PPR"),
    "half_ppr": ("half_ppr", "Half PPR"),
    "3": ("dad", "Dad's Settings"),
    "dad": ("dad", "Dad's Settings"),
    "dads": ("dad", "Dad's Settings"),
    "dad_settings": ("dad", "Dad's Settings"),
}

RATE_SPECS = [
    ("top36", "Top 36"),
    ("top24", "Top 24"),
    ("top12", "Top 12"),
    ("top5", "Top 5"),
    ("top3", "Top 3"),
    ("top1", "#1 Overall"),
]
RATE_COLORS = {
    "top36": "#2563eb",
    "top24": "#0891b2",
    "top12": "#16a34a",
    "top5": "#ca8a04",
    "top3": "#ea580c",
    "top1": "#dc2626",
}
RATE_GROUPS = {
    "starter_depth": {
        "title": "RB Age Share Within Top 36 / Top 24 / Top 12",
        "thresholds": ["top36", "top24", "top12"],
        "svg_name": "rb_starter_depth_rates_by_age.svg",
        "png_name": "rb_starter_depth_rates_by_age.png",
    },
    "elite": {
        "title": "RB Age Share Within Top 5 / Top 3 / #1 Overall",
        "thresholds": ["top5", "top3", "top1"],
        "svg_name": "rb_elite_rates_by_age.svg",
        "png_name": "rb_elite_rates_by_age.png",
    },
}


def _fmt_pct(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _normalize_scoring_choice(value: str | None) -> tuple[str, str]:
    if not value:
        return SCORING_OPTIONS["1"]
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return SCORING_OPTIONS.get(key, SCORING_OPTIONS["1"])


def _choose_scoring(cli_value: str | None, no_prompt: bool) -> tuple[str, str]:
    if cli_value:
        return _normalize_scoring_choice(cli_value)
    if no_prompt:
        return SCORING_OPTIONS["1"]

    print("")
    print("SCORING FORMAT")
    print("--------------")
    print("1. Full PPR")
    print("2. Half PPR")
    print("3. Dad's Settings")
    choice = input("Choose scoring format [1]: ").strip()
    return _normalize_scoring_choice(choice or "1")


def _build_study_metadata(seasons_df: pd.DataFrame, scoring: str) -> Dict[str, object]:
    season_min = int(seasons_df["season"].min()) if "season" in seasons_df.columns and not seasons_df.empty else None
    season_max = int(seasons_df["season"].max()) if "season" in seasons_df.columns and not seasons_df.empty else None
    timeline = f"{season_min}-{season_max}" if season_min and season_max else "Unknown"
    source_path = seasons_df.attrs.get("source_path", "case_studies/data/rb_elite_age_player_seasons_<scoring>.csv")
    source_type = seasons_df.attrs.get("source_type", "nflverse_pull")
    return {
        "data_source": "nflverse-data GitHub releases: stats_player + rosters",
        "source_releases": "stats_player, rosters",
        "source_type": source_type,
        "cache_file": source_path,
        "timeline": timeline,
        "first_season": season_min,
        "last_season": season_max,
        "scoring": scoring,
        "scoring_label": _normalize_scoring_choice(scoring)[1],
        "player_seasons": int(len(seasons_df)),
    }


def _filter_last_n_seasons(seasons_df: pd.DataFrame, season_count: int = LAST_N_SEASONS) -> pd.DataFrame:
    if "season" not in seasons_df.columns or seasons_df.empty:
        return seasons_df.copy()

    working_df = seasons_df.copy()
    working_df["season"] = pd.to_numeric(working_df["season"], errors="coerce")
    available_seasons = sorted(working_df["season"].dropna().astype(int).unique().tolist())
    if not available_seasons:
        return working_df

    selected_seasons = set(available_seasons[-season_count:])
    filtered_df = working_df[working_df["season"].astype("Int64").isin(selected_seasons)].copy()
    filtered_df.attrs.update(seasons_df.attrs)
    filtered_df.attrs["timeline_filter"] = f"Last {season_count} seasons"
    return filtered_df


def _write_svg_line_chart(
    summary_df: pd.DataFrame,
    rate_col: str,
    count_col: str,
    title: str,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1000
    height = 560
    margin_left = 78
    margin_right = 42
    margin_top = 74
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    chart_df = summary_df[["Age", rate_col, "total_rb_seasons", count_col]].copy()
    chart_df = chart_df.dropna(subset=["Age", rate_col]).sort_values("Age")
    if chart_df.empty:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    x_min = int(chart_df["Age"].min())
    x_max = int(chart_df["Age"].max())
    y_max = max(5.0, float(chart_df[rate_col].max()) * 1.15)

    def x_pos(age: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + ((age - x_min) / (x_max - x_min)) * plot_width

    def y_pos(rate: float) -> float:
        return margin_top + plot_height - (rate / y_max) * plot_height

    points = [
        (x_pos(float(row["Age"])), y_pos(float(row[rate_col])), row)
        for _, row in chart_df.iterrows()
    ]
    path_data = " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y, _) in enumerate(points)
    )

    y_ticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    x_ticks = list(range(x_min, x_max + 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 28px; font-weight: 700; }",
        ".label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.4; }",
        ".line { fill: none; stroke: #2563eb; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { fill: #10b981; stroke: #064e3b; stroke-width: 1.5; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="42">{title}</text>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.1f}%</text>')

    for age in x_ticks:
        x = x_pos(age)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{height - 42}" text-anchor="middle">{age}</text>')

    lines.extend([
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
        f'<text class="label" x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle">Age</text>',
        f'<text class="label" transform="translate(22 {height / 2:.2f}) rotate(-90)" text-anchor="middle">Share of Top-N Slots</text>',
    ])

    mean_rate = float(chart_df[rate_col].mean())
    mean_y = y_pos(mean_rate)
    lines.extend([
        f'<line x1="{margin_left}" y1="{mean_y:.2f}" x2="{width - margin_right}" y2="{mean_y:.2f}" stroke="#9333ea" stroke-width="2" stroke-dasharray="8 6"/>',
        f'<text class="tick" x="{width - margin_right - 4}" y="{mean_y - 8:.2f}" text-anchor="end">Mean {mean_rate:.2f}%</text>',
        f'<path class="line" d="{path_data}"/>',
    ])

    for x, y, row in points:
        tooltip = (
            f"Age: {int(row['Age'])}\n"
            f"RB seasons at age: {int(row['total_rb_seasons'])}\n"
            f"Age count in group: {int(row[count_col])}\n"
            f"Share: {float(row[rate_col]):.2f}%"
        )
        lines.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="6"><title>{tooltip}</title></circle>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_report(study: Dict[str, object], metadata: Dict[str, object], output_path: Path) -> None:
    peaks = study["peaks"]
    summary_df = study["summary"]
    lines = [
        "# RB Elite Season Age Study - Last 10 Seasons",
        "",
        "This study measures each age's share of elite RB finishes. It does not use average PPG by age.",
        "",
        "Rate % = players at that age inside Top N RB slots / all Top N RB slots.",
        "",
        "## Data Source",
        "",
        f"- Source: {metadata['data_source']}",
        f"- Releases: {metadata['source_releases']}",
        f"- Timeline: {metadata['timeline']}",
        f"- Scoring: {metadata['scoring_label']}",
        f"- Player-seasons analyzed: {metadata['player_seasons']:,}",
        f"- Cache file: `{metadata['cache_file']}`",
        "",
        "## Peaks",
        "",
    ]

    for threshold, label in RATE_SPECS:
        key = f"{threshold}_rate"
        peak = peaks.get(key)
        if peak:
            lines.append(f"- Peak Age for {label} Rate: Age {peak['age']} ({peak['rate_pct']:.2f}%)")
        else:
            lines.append(f"- Peak Age for {label} Rate: N/A")

    table_df = summary_df.rename(columns={
        "total_rb_seasons": "rb_seasons_at_age",
        "top36_rb_seasons": "top36_age_count",
        "top24_rb_seasons": "top24_age_count",
        "top12_rb_seasons": "top12_age_count",
        "top5_rb_seasons": "top5_age_count",
        "top3_rb_seasons": "top3_age_count",
        "top1_rb_seasons": "top1_age_count",
    })
    lines.extend(["", "## Table", "", _dataframe_to_markdown(table_df), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_multi_line_chart(
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
    thresholds=None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    height = 640
    margin_left = 78
    margin_right = 190
    margin_top = 74
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    chart_df = summary_df.copy().sort_values("Age")
    if chart_df.empty:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    x_min = int(chart_df["Age"].min())
    x_max = int(chart_df["Age"].max())
    chart_specs = [(threshold, label) for threshold, label in RATE_SPECS if thresholds is None or threshold in thresholds]
    y_max = max(
        5.0,
        max(float(chart_df[f"{threshold}_rate_pct"].max()) for threshold, _ in chart_specs) * 1.12,
    )

    def x_pos(age: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + ((age - x_min) / (x_max - x_min)) * plot_width

    def y_pos(rate: float) -> float:
        return margin_top + plot_height - (rate / y_max) * plot_height

    y_ticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    x_ticks = list(range(x_min, x_max + 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 28px; font-weight: 700; }",
        ".label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".legend { font-size: 14px; font-weight: 700; }",
        ".mean-label { font-size: 12px; font-weight: 700; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.4; }",
        ".rate-line { fill: none; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { stroke: #ffffff; stroke-width: 1.4; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="42">{title}</text>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.1f}%</text>')

    for age in x_ticks:
        x = x_pos(age)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{height - 42}" text-anchor="middle">{age}</text>')

    lines.extend([
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
        f'<text class="label" x="{(width - margin_right + margin_left) / 2:.2f}" y="{height - 12}" text-anchor="middle">Age</text>',
        f'<text class="label" transform="translate(22 {height / 2:.2f}) rotate(-90)" text-anchor="middle">Rate (%)</text>',
    ])

    for threshold, label in chart_specs:
        rate_col = f"{threshold}_rate_pct"
        count_col = f"{threshold}_rb_seasons"
        color = RATE_COLORS[threshold]
        mean_rate = float(chart_df[rate_col].mean())
        mean_y = y_pos(mean_rate)
        lines.append(
            f'<line x1="{margin_left}" y1="{mean_y:.2f}" x2="{width - margin_right}" y2="{mean_y:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="7 6" opacity="0.7"/>'
        )
        lines.append(
            f'<text class="mean-label" fill="{color}" x="{width - margin_right - 6}" y="{mean_y - 5:.2f}" '
            f'text-anchor="end">{label} mean {mean_rate:.2f}%</text>'
        )
        points = [
            (x_pos(float(row["Age"])), y_pos(float(row[rate_col])), row)
            for _, row in chart_df.iterrows()
        ]
        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
            for index, (x, y, _) in enumerate(points)
        )
        lines.append(f'<path class="rate-line" stroke="{color}" d="{path_data}"/>')
        for x, y, row in points:
            tooltip = (
                f"{label}\n"
                f"Age: {int(row['Age'])}\n"
                f"RB seasons at age: {int(row['total_rb_seasons'])}\n"
                f"Age count in group: {int(row[count_col])}\n"
                f"Share: {float(row[rate_col]):.2f}%"
            )
            lines.append(f'<circle class="dot" fill="{color}" cx="{x:.2f}" cy="{y:.2f}" r="4.5"><title>{tooltip}</title></circle>')

    legend_x = width - margin_right + 34
    legend_y = margin_top + 8
    for index, (threshold, label) in enumerate(chart_specs):
        y = legend_y + index * 30
        color = RATE_COLORS[threshold]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        lines.append(f'<text class="legend" x="{legend_x + 38}" y="{y + 5}">{label}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[column]) for column in df.columns])

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    ) + " |"
    divider_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *row_lines])


def _write_matplotlib_chart(
    summary_df: pd.DataFrame,
    rate_col: str,
    count_col: str,
    title: str,
    output_path: Path,
) -> bool:
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(summary_df["Age"], summary_df[rate_col], marker="o", linewidth=2.5)
    mean_rate = float(summary_df[rate_col].mean())
    ax.axhline(mean_rate, color="#9333ea", linestyle="--", linewidth=1.8, label=f"Mean {mean_rate:.2f}%")
    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Top-N RB Slots (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{row[rate_col]:.1f}%",
            (row["Age"], row[rate_col]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _write_matplotlib_multi_chart(
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
    thresholds=None,
) -> bool:
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    chart_specs = [(threshold, label) for threshold, label in RATE_SPECS if thresholds is None or threshold in thresholds]
    for threshold, label in chart_specs:
        ax.plot(
            summary_df["Age"],
            summary_df[f"{threshold}_rate_pct"],
            marker="o",
            linewidth=2.4,
            label=label,
            color=RATE_COLORS[threshold],
        )
        mean_rate = float(summary_df[f"{threshold}_rate_pct"].mean())
        ax.axhline(
            mean_rate,
            color=RATE_COLORS[threshold],
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
            label=f"{label} mean {mean_rate:.2f}%",
        )

    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Top-N RB Slots (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _write_html_report(
    study: Dict[str, object],
    metadata: Dict[str, object],
    chart_paths: Dict[str, Path],
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peaks = study["peaks"]
    summary_df = study["summary"]

    def svg_body(path: Path) -> str:
        path = Path(path)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    peak_cards = []
    for threshold, label in RATE_SPECS:
        key = f"{threshold}_rate"
        peak = peaks.get(key)
        value = f"Age {peak['age']}" if peak else "N/A"
        detail = f"{peak['rate_pct']:.2f}%" if peak else ""
        peak_cards.append(
            f"""
            <section class="card">
                <div class="label">{label} Peak</div>
                <div class="value">{value}</div>
                <div class="detail">{detail}</div>
            </section>
            """
        )

    table_html = summary_df.rename(columns={
        "total_rb_seasons": "rb_seasons_at_age",
        "top36_rb_seasons": "top36_age_count",
        "top24_rb_seasons": "top24_age_count",
        "top12_rb_seasons": "top12_age_count",
        "top5_rb_seasons": "top5_age_count",
        "top3_rb_seasons": "top3_age_count",
        "top1_rb_seasons": "top1_age_count",
    }).to_html(index=False, classes="rate-table", border=0)
    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>RB Elite Season Age Study - Last 10 Seasons</title>
    <style>
        body {{
            margin: 0;
            background: #f4f7fb;
            color: #111827;
            font-family: Arial, sans-serif;
        }}
        main {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 34px 28px 60px;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 34px;
        }}
        .subtitle {{
            margin: 0 0 28px;
            color: #4b5563;
        }}
        .meta {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 24px;
        }}
        .meta-card {{
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .meta-label {{
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
        }}
        .meta-value {{
            margin-top: 6px;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.35;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 26px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 18px;
        }}
        .label {{
            color: #64748b;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .value {{
            margin-top: 8px;
            font-size: 32px;
            font-weight: 800;
        }}
        .detail {{
            margin-top: 4px;
            color: #047857;
            font-size: 18px;
            font-weight: 700;
        }}
        .chart {{
            margin: 22px 0;
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 10px;
        }}
        .chart svg {{
            width: 100%;
            height: auto;
        }}
        .table-wrap {{
            margin-top: 24px;
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 18px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e5e7eb;
            text-align: right;
        }}
        th:first-child, td:first-child {{
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            font-weight: 800;
        }}
    </style>
</head>
<body>
    <main>
        <h1>RB Elite Season Age Study - Last 10 Seasons</h1>
        <p class="subtitle">Age share inside elite RB finish groups. No average PPG by age.</p>
        <p class="subtitle">Rate % = players at that age inside Top N RB slots / all Top N RB slots.</p>
        <section class="meta">
            <div class="meta-card">
                <div class="meta-label">Data Source</div>
                <div class="meta-value">{metadata['data_source']}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Timeline</div>
                <div class="meta-value">{metadata['timeline']}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Scoring</div>
                <div class="meta-value">{metadata['scoring_label']}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Player-Seasons</div>
                <div class="meta-value">{metadata['player_seasons']:,}</div>
            </div>
        </section>
        <p class="subtitle">Source releases: {metadata['source_releases']} | Cache: {metadata['cache_file']}</p>
        <div class="cards">
            {''.join(peak_cards)}
        </div>
        <section class="chart">{svg_body(chart_paths["starter_depth"])}</section>
        <section class="chart">{svg_body(chart_paths["elite"])}</section>
        {''.join(f'<section class="chart">{svg_body(path)}</section>' for key, path in chart_paths.items() if key not in {"starter_depth", "elite"})}
        <section class="table-wrap">
            <h2>Age Share Table</h2>
            {table_html}
        </section>
    </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone RB elite-season age study for the last 10 seasons.")
    parser.add_argument(
        "--scoring",
        choices=["ppr", "full_ppr", "half_ppr", "full", "half", "dad", "dads", "dad_settings"],
        default=None,
        help="Optional scoring override. If omitted, the script asks in the console.",
    )
    parser.add_argument("--no-prompt", action="store_true", help="Use Full PPR without asking for a scoring choice.")
    parser.add_argument("--refresh", action="store_true", help="Pull source data again instead of using cached data.")
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=1,
        help="Minimum Top 36 slots an age must have to appear in the table.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML report after the run.")
    args = parser.parse_args()
    scoring, scoring_label = _choose_scoring(args.scoring, args.no_prompt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling/caching historical RB player-season data ({scoring_label})...")
    seasons_df = pull_historical_seasons_df(scoring=scoring, force_refresh=args.refresh)
    seasons_df = _filter_last_n_seasons(seasons_df, LAST_N_SEASONS)
    metadata = _build_study_metadata(seasons_df, scoring)

    print(f"Calculating RB age share inside top-N finish groups for the last {LAST_N_SEASONS} seasons...")
    study = build_rb_elite_age_study(seasons_df, min_sample_size=args.min_sample_size)
    summary_df = study["summary"]
    if summary_df.empty:
        raise SystemExit("No RB ages met the minimum Top 36 slot filter.")

    table_path = output_dir / "rb_elite_age_rates_last10.csv"
    report_path = output_dir / "rb_elite_age_study_last10.md"
    html_path = output_dir / "rb_elite_age_study_last10.html"
    svg_paths = {
        threshold: output_dir / f"rb_{threshold}_rate_by_age.svg"
        for threshold, _ in RATE_SPECS
    }
    png_paths = {
        threshold: output_dir / f"rb_{threshold}_rate_by_age.png"
        for threshold, _ in RATE_SPECS
    }
    grouped_svg_paths = {
        group_key: output_dir / spec["svg_name"]
        for group_key, spec in RATE_GROUPS.items()
    }
    grouped_png_paths = {
        group_key: output_dir / spec["png_name"]
        for group_key, spec in RATE_GROUPS.items()
    }

    summary_df.to_csv(table_path, index=False)
    _write_markdown_report(study, metadata, report_path)
    wrote_matplotlib = []
    for group_key, spec in RATE_GROUPS.items():
        _write_svg_multi_line_chart(
            summary_df,
            spec["title"],
            grouped_svg_paths[group_key],
            thresholds=spec["thresholds"],
        )
        wrote_matplotlib.append(
            _write_matplotlib_multi_chart(
                summary_df,
                spec["title"],
                grouped_png_paths[group_key],
                thresholds=spec["thresholds"],
            )
        )
    for threshold, label in RATE_SPECS:
        _write_svg_line_chart(
            summary_df,
            f"{threshold}_rate_pct",
            f"{threshold}_rb_seasons",
            f"RB {label} Age Share",
            svg_paths[threshold],
        )
        wrote_matplotlib.append(
            _write_matplotlib_chart(
                summary_df,
                f"{threshold}_rate_pct",
                f"{threshold}_rb_seasons",
                f"RB {label} Age Share",
                png_paths[threshold],
            )
        )
    _write_html_report(study, metadata, {**grouped_svg_paths, **svg_paths}, html_path)

    print("")
    print("DATA SOURCE")
    print("-----------")
    print(f"Source: {metadata['data_source']}")
    print(f"Releases: {metadata['source_releases']}")
    print(f"Timeline: {metadata['timeline']}")
    print(f"Scoring: {metadata['scoring_label']}")
    print(f"Player-seasons analyzed: {metadata['player_seasons']:,}")
    print(f"Cache file: {metadata['cache_file']}")
    print("")
    print("DEFINITION")
    print("----------")
    print("Rate % = players at that age inside Top N RB slots / all Top N RB slots.")
    print("Example: Top 3 ages 24, 25, 26 -> each age is 1/3 = 33.33%.")
    print("")
    print("PEAK AGES")
    print("---------")
    for threshold, label in RATE_SPECS:
        key = f"{threshold}_rate"
        peak = study["peaks"].get(key)
        if peak:
            print(f"Peak Age for {label} Rate: Age {peak['age']} ({_fmt_pct(peak['rate_pct'])})")

    print("")
    print("AGE SHARE TABLE")
    print("---------------")
    table_df = summary_df[
        [
            "Age",
            "total_rb_seasons",
            "top36_rb_seasons",
            "top36_rate_pct",
            "top24_rb_seasons",
            "top24_rate_pct",
            "top12_rb_seasons",
            "top12_rate_pct",
            "top5_rb_seasons",
            "top5_rate_pct",
            "top3_rb_seasons",
            "top3_rate_pct",
            "top1_rb_seasons",
            "top1_rate_pct",
        ]
    ].rename(columns={
        "total_rb_seasons": "rb_seasons_at_age",
        "top36_rb_seasons": "top36_age_count",
        "top24_rb_seasons": "top24_age_count",
        "top12_rb_seasons": "top12_age_count",
        "top5_rb_seasons": "top5_age_count",
        "top3_rb_seasons": "top3_age_count",
        "top1_rb_seasons": "top1_age_count",
    })
    print(table_df.to_string(index=False))

    print("")
    print(f"Wrote table: {table_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote HTML report: {html_path}")
    for path in grouped_svg_paths.values():
        print(f"Wrote grouped chart: {path}")
    for path in svg_paths.values():
        print(f"Wrote chart: {path}")
    if any(wrote_matplotlib):
        for path in grouped_png_paths.values():
            print(f"Wrote Matplotlib grouped chart: {path}")
        for path in png_paths.values():
            print(f"Wrote Matplotlib chart: {path}")
    else:
        print("Matplotlib is not installed, so PNG charts were skipped. SVG charts were still created.")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
