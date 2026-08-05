from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
import webbrowser

import pandas as pd
import numpy as np

from rb_elite_age_analysis import build_rb_elite_age_study, pull_historical_seasons_df

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle
except ModuleNotFoundError:
    plt = None


OUTPUT_DIR = Path("case_studies/output")
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
    ("top20", "Top 20"),
    ("top15", "Top 15"),
    ("top12", "Top 12"),
    ("top6", "Top 6"),
    ("top5", "Top 5"),
    ("top3", "Top 3"),
    ("top1", "#1 Overall"),
]
POSITION_RATE_SPECS = {
    "RB": [("top36", "Top 36"), ("top24", "Top 24"), ("top12", "Top 12"), ("top5", "Top 5"), ("top3", "Top 3")],
    "WR": [("top36", "Top 36"), ("top24", "Top 24"), ("top12", "Top 12"), ("top5", "Top 5"), ("top3", "Top 3")],
    "QB": [("top20", "Top 20"), ("top15", "Top 15"), ("top12", "Top 12"), ("top6", "Top 6"), ("top3", "Top 3")],
    "TE": [("top20", "Top 20"), ("top15", "Top 15"), ("top12", "Top 12"), ("top6", "Top 6"), ("top3", "Top 3")],
}
POSITIONS = ["RB", "WR", "QB", "TE"]
RATE_COLORS = {
    "top36": "#2563eb",
    "top24": "#0891b2",
    "top20": "#2563eb",
    "top15": "#0891b2",
    "top12": "#16a34a",
    "top6": "#ca8a04",
    "top5": "#ca8a04",
    "top3": "#ea580c",
    "top1": "#dc2626",
}
RATE_GROUPS = {
    "consolidated": {
        "title": "{position} Age Share Within Key Positional Finish Tiers",
        "thresholds": None,
        "svg_name": "{position_key}_consolidated_rates_by_age.svg",
        "png_name": "{position_key}_consolidated_rates_by_age.png",
    },
}
COMPARISON_RATE_SPECS = [
    ("top36", "Top 36"),
    ("top24", "Top 24"),
    ("top12", "Top 12"),
    ("top5", "Top 5"),
    ("top3", "Top 3"),
    ("top1", "#1 Overall"),
]
CAREER_ARCS = {
    "RB": [("ASCENT", 21, 23, "#dbeafe"), ("PEAK YEARS", 24, 27, "#dcfce7"), ("DECLINE", 28, None, "#f1f5f9")],
    "WR": [("ASCENT", 21, 23, "#dbeafe"), ("PEAK YEARS", 24, 29, "#dcfce7"), ("DECLINE", 30, None, "#f1f5f9")],
    "QB": [("DEVELOPMENT", 21, 25, "#dbeafe"), ("PRIME", 26, 33, "#dcfce7"), ("VETERAN YEARS", 34, None, "#f1f5f9")],
    "TE": [("DEVELOPMENT", 21, 24, "#dbeafe"), ("PRIME", 25, 31, "#dcfce7"), ("VETERAN YEARS", 32, None, "#f1f5f9")],
}
MOBILE_QB_QUALIFIER_NAMES = {
    "L.Jackson",
    "C.Newton",
    "R.Wilson",
    "J.Allen",
    "J.Hurts",
    "K.Murray",
    "J.Fields",
    "P.Mahomes",
    "M.Mariota",
    "T.Taylor",
    "D.Jones",
    "C.Kaepernick",
    "A.Smith",
    "D.Watson",
    "D.Prescott",
    "R.Griffin",
    "B.Bortles",
    "J.Herbert",
    "A.Luck",
    "T.Hill",
    "T.Lawrence",
    "J.Daniels",
    "D.Maye",
    "B.Nix",
    "A.Richardson",
    "C.Williams",
    "B.Young",
    "J.Dart",
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


def _filter_recent_seasons(seasons_df: pd.DataFrame, season_count: int | None) -> pd.DataFrame:
    if season_count is None or "season" not in seasons_df.columns or seasons_df.empty:
        return seasons_df.copy()

    working_df = seasons_df.copy()
    working_df["season"] = pd.to_numeric(working_df["season"], errors="coerce")
    available_seasons = sorted(working_df["season"].dropna().astype(int).unique().tolist())
    if not available_seasons:
        return working_df

    clamped_count = max(1, min(int(season_count), len(available_seasons)))
    selected_seasons = set(available_seasons[-clamped_count:])
    filtered_df = working_df[working_df["season"].astype("Int64").isin(selected_seasons)].copy()
    filtered_df.attrs.update(seasons_df.attrs)
    filtered_df.attrs["requested_season_count"] = int(season_count)
    filtered_df.attrs["season_count_used"] = clamped_count
    filtered_df.attrs["available_season_count"] = len(available_seasons)
    return filtered_df


def _choose_season_count(seasons_df: pd.DataFrame, cli_value: int | None, no_prompt: bool) -> int:
    if "season" not in seasons_df.columns or seasons_df.empty:
        return 0

    available_seasons = sorted(pd.to_numeric(seasons_df["season"], errors="coerce").dropna().astype(int).unique().tolist())
    max_count = len(available_seasons)
    if max_count == 0:
        return 0

    if cli_value is not None:
        return max(1, min(int(cli_value), max_count))
    if no_prompt:
        return max_count

    print("")
    print("YEARS OF DATA")
    print("-------------")
    print(f"Available seasons: {available_seasons[0]}-{available_seasons[-1]} ({max_count} seasons)")
    choice = input(f"How many most-recent seasons should be used? [{max_count}]: ").strip()
    if not choice:
        return max_count
    try:
        requested_count = int(choice)
    except ValueError:
        return max_count
    return max(1, min(requested_count, max_count))


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
        "season_count_used": seasons_df.attrs.get("season_count_used"),
        "available_season_count": seasons_df.attrs.get("available_season_count"),
        "requested_season_count": seasons_df.attrs.get("requested_season_count"),
    }


def _position_rate_specs(position: str):
    return POSITION_RATE_SPECS.get(position, POSITION_RATE_SPECS["RB"])


def _position_table_df(summary_df: pd.DataFrame, position: str) -> pd.DataFrame:
    columns = ["Age", "total_rb_seasons"]
    rename_map = {"total_rb_seasons": "player_seasons_at_age"}
    for threshold, _label in _position_rate_specs(position):
        columns.extend([f"{threshold}_rb_seasons", f"{threshold}_rate_pct"])
        rename_map[f"{threshold}_rb_seasons"] = f"{threshold}_age_count"
    return summary_df[columns].rename(columns=rename_map)


def _average_top1_age(summary_df: pd.DataFrame) -> float | None:
    total = float(summary_df["top1_rb_seasons"].sum()) if "top1_rb_seasons" in summary_df.columns else 0.0
    if not total:
        return None
    return float((summary_df["Age"] * summary_df["top1_rb_seasons"]).sum() / total)


def _average_age_for_threshold(summary_df: pd.DataFrame, threshold: str) -> float | None:
    count_col = f"{threshold}_rb_seasons"
    if count_col not in summary_df.columns:
        return None
    total = float(summary_df[count_col].sum())
    if not total:
        return None
    return float((summary_df["Age"] * summary_df[count_col]).sum() / total)


def _peak_for_threshold(study: Dict[str, object], threshold: str) -> Dict[str, object] | None:
    return study.get("peaks", {}).get(f"{threshold}_rate")


def _threshold_label(position: str, threshold: str) -> str:
    return dict(_position_rate_specs(position)).get(threshold, threshold.replace("top", "Top "))


def render_peak_summary_table(study: Dict[str, object], position: str) -> str:
    rows = []
    for threshold, label in _position_rate_specs(position):
        peak = _peak_for_threshold(study, threshold)
        rows.append(
            {
                "Metric": label,
                "Peak Age": int(peak["age"]) if peak else "N/A",
                "Peak Share": _fmt_pct(peak["rate_pct"]) if peak else "N/A",
                "Prime Window": _prime_window_for_threshold(study["summary"], threshold),
            }
        )
    return pd.DataFrame(rows).to_html(index=False, classes="summary-table", border=0)


def calculate_prime_window(summary_df: pd.DataFrame, threshold: str, threshold_pct: float = 0.75) -> str:
    rate_col = f"{threshold}_rate_pct"
    if rate_col not in summary_df.columns or summary_df.empty:
        return "N/A"
    peak_rate = float(summary_df[rate_col].max())
    if peak_rate <= 0:
        return "N/A"
    prime_df = summary_df[summary_df[rate_col] >= peak_rate * threshold_pct]
    if prime_df.empty:
        return "N/A"
    return f"{int(prime_df['Age'].min())}-{int(prime_df['Age'].max())}"


def _prime_window_for_threshold(summary_df: pd.DataFrame, threshold: str) -> str:
    return calculate_prime_window(summary_df, threshold)


def _top_threshold_for_position(position: str) -> str:
    return "top3"


def _broad_threshold_for_position(position: str) -> str:
    return _position_rate_specs(position)[0][0]


def _peak_age(study: Dict[str, object], threshold: str) -> int | None:
    peak = _peak_for_threshold(study, threshold)
    return int(peak["age"]) if peak else None


def _peak_ages(summary_df: pd.DataFrame, threshold: str) -> list[int]:
    rate_col = f"{threshold}_rate_pct"
    if rate_col not in summary_df.columns or summary_df.empty:
        return []
    peak_rate = float(summary_df[rate_col].max())
    if peak_rate <= 0:
        return []
    return summary_df.loc[summary_df[rate_col] == peak_rate, "Age"].astype(int).tolist()


def _format_ages(ages: list[int]) -> str:
    if not ages:
        return "N/A"
    if len(ages) == 1:
        return str(ages[0])
    if len(ages) == 2:
        return f"{ages[0]} and {ages[1]}"
    return ", ".join(str(age) for age in ages[:-1]) + f", and {ages[-1]}"


def _age_noun(ages: list[int]) -> str:
    return "ages" if len(ages) != 1 else "age"


def _decline_age(summary_df: pd.DataFrame, threshold: str, drop_pct: float = 0.55) -> int | None:
    rate_col = f"{threshold}_rate_pct"
    if rate_col not in summary_df.columns or summary_df.empty:
        return None
    peak_rate = float(summary_df[rate_col].max())
    if peak_rate <= 0:
        return None
    peak_age = int(summary_df.loc[summary_df[rate_col].idxmax(), "Age"])
    later_df = summary_df[(summary_df["Age"] > peak_age) & (summary_df[rate_col] <= peak_rate * drop_pct)]
    if later_df.empty:
        return None
    return int(later_df.iloc[0]["Age"])


def _sentence_join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _build_executive_insight(study: Dict[str, object], position: str) -> str:
    summary_df = study["summary"]
    broad_threshold = _broad_threshold_for_position(position)
    anchor_threshold = "top12"
    elite_threshold = "top3"
    broad_label = _threshold_label(position, broad_threshold)
    broad_ages = _peak_ages(summary_df, broad_threshold)
    anchor_ages = _peak_ages(summary_df, anchor_threshold)
    elite_ages = _peak_ages(summary_df, elite_threshold)
    prime_window = calculate_prime_window(summary_df, anchor_threshold)

    if not elite_ages and not anchor_ages and not broad_ages:
        return f"{position} aging curve is inconclusive with the current sample."

    elite_midpoint = float(np.mean(elite_ages)) if elite_ages else None
    broad_midpoint = float(np.mean(broad_ages)) if broad_ages else None

    if position in {"RB", "WR"} and elite_midpoint is not None and broad_midpoint is not None:
        if elite_midpoint < broad_midpoint:
            return (
                f"Elite {position} production peaks earlier at {_age_noun(elite_ages)} {_format_ages(elite_ages)}, "
                f"while broader {broad_label} relevance peaks at {_age_noun(broad_ages)} {_format_ages(broad_ages)}. "
                f"Prime window: {prime_window}."
            )
        if elite_midpoint > broad_midpoint:
            return (
                f"{position}s maintain elite outcomes later than broad relevance, with Top 3 finishes peaking "
                f"at {_age_noun(elite_ages)} {_format_ages(elite_ages)}. Prime window: {prime_window}."
            )
        return (
            f"{position} elite and broad relevance both concentrate around {_age_noun(elite_ages)} {_format_ages(elite_ages)}. "
            f"Prime window: {prime_window}."
        )

    available_peaks = []
    if anchor_ages:
        available_peaks.append(f"Top 12 finishes peak at {_age_noun(anchor_ages)} {_format_ages(anchor_ages)}")
    if elite_ages:
        available_peaks.append(f"Top 3 finishes peak at {_age_noun(elite_ages)} {_format_ages(elite_ages)}")
    if broad_ages:
        available_peaks.append(f"{broad_label} relevance peaks at {_age_noun(broad_ages)} {_format_ages(broad_ages)}")
    return f"{position} curve: {_sentence_join(available_peaks)}. Prime window: {prime_window}."


def generate_age_study_insights(study: Dict[str, object], position: str) -> Dict[str, object]:
    summary_df = study["summary"]
    specs = _position_rate_specs(position)
    peak_rows = []
    for threshold, label in specs:
        peak = _peak_for_threshold(study, threshold)
        if not peak:
            continue
        peak_rows.append((threshold, label, int(peak["age"]), float(peak["rate_pct"])))

    anchor_threshold = "top12" if "top12_rate_pct" in summary_df.columns else specs[0][0]
    decline_age = _decline_age(summary_df, anchor_threshold)
    avg_top1 = _average_top1_age(summary_df)
    peak_age = _peak_for_threshold(study, anchor_threshold)
    prime_window = _prime_window_for_threshold(summary_df, anchor_threshold)
    findings = []
    for threshold, label, age, _rate in peak_rows:
        if threshold in {specs[0][0], "top12", "top5", "top3"}:
            ages = _peak_ages(summary_df, threshold)
            findings.append(f"{label} finishes peak at {_age_noun(ages)} {_format_ages(ages)}.")
    if decline_age:
        findings.append(f"Top 12 production falls below 55% of peak by age {decline_age}.")
    return {
        "findings": findings[:5],
        "executive": _build_executive_insight(study, position),
        "avg_top1_age": avg_top1,
        "prime_window": prime_window,
        "anchor_peak_age": int(peak_age["age"]) if peak_age else "N/A",
    }


def render_executive_insight(study: Dict[str, object], position: str) -> str:
    insights = generate_age_study_insights(study, position)
    return f"""
    <section class="executive-insight">
        <div class="eyebrow">Key Insight</div>
        <p>{insights["executive"]}</p>
    </section>
    """


def render_position_takeaways(study: Dict[str, object], position: str) -> str:
    insights = generate_age_study_insights(study, position)
    items = "".join(f"<li>{finding}</li>" for finding in insights["findings"])
    items += f"<li>Prime window spans ages {insights['prime_window']}.</li>"
    return f"""
    <section class="insight-panel">
        <h2>{position} Takeaways</h2>
        <ul>{items}</ul>
    </section>
    """


def render_key_findings_panel(study: Dict[str, object], position: str) -> str:
    return render_position_takeaways(study, position)


def _position_insight_summary_table(studies: Dict[str, Dict[str, object]]) -> str:
    rows = []
    for position, study in studies.items():
        insights = generate_age_study_insights(study, position)
        avg_age = _average_age_for_threshold(study["summary"], "top12")
        rows.append(
            {
                "Position": position,
                "Peak Age": insights["anchor_peak_age"],
                "Average Age": f"{avg_age:.1f}" if avg_age is not None else "N/A",
                "Prime Window": insights["prime_window"],
            }
        )
    return pd.DataFrame(rows).to_html(index=False, classes="summary-table", border=0)


def render_advanced_stats_section(study: Dict[str, object], position: str) -> str:
    summary_df = study["summary"]
    rows = []
    for threshold, label in _position_rate_specs(position):
        count_col = f"{threshold}_rb_seasons"
        if count_col not in summary_df.columns:
            continue
        counts = summary_df[count_col].fillna(0).astype(float)
        total = float(counts.sum())
        peak = _peak_for_threshold(study, threshold)
        if total:
            average_age = float((summary_df["Age"] * counts).sum() / total)
            expanded_ages = [
                int(row["Age"])
                for _, row in summary_df.iterrows()
                for _ in range(int(row[count_col]))
            ]
            median_age = float(np.median(expanded_ages)) if expanded_ages else np.nan
            max_count = counts.max()
            peak_ages = summary_df.loc[counts == max_count, "Age"].astype(float).tolist()
            weighted_peak_age = float(np.mean(peak_ages)) if peak_ages else np.nan
        else:
            average_age = np.nan
            median_age = np.nan
            weighted_peak_age = np.nan
        rows.append(
            {
                "Metric": label,
                "Peak Age": int(peak["age"]) if peak else "N/A",
                "Prime Window": _prime_window_for_threshold(summary_df, threshold),
                "Average Age": f"{average_age:.1f}" if pd.notna(average_age) else "N/A",
                "Median Age": f"{median_age:.1f}" if pd.notna(median_age) else "N/A",
                "Weighted Peak Age": f"{weighted_peak_age:.1f}" if pd.notna(weighted_peak_age) else "N/A",
                "Sample Size": int(total),
            }
        )
    table_html = pd.DataFrame(rows).to_html(index=False, classes="summary-table", border=0)
    return f"""
    <details class="advanced-stats">
        <summary>Advanced Statistics</summary>
        {table_html}
    </details>
    """


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
            f"Player seasons at age: {int(row['total_rb_seasons'])}\n"
            f"Age count in group: {int(row[count_col])}\n"
            f"Share: {float(row[rate_col]):.2f}%"
        )
        lines.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="6"><title>{tooltip}</title></circle>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_report(
    study: Dict[str, object],
    metadata: Dict[str, object],
    output_path: Path,
    position: str = "RB",
) -> None:
    peaks = study["peaks"]
    summary_df = study["summary"]
    lines = [
        f"# {position} Elite Season Age Study",
        "",
        f"This study measures each age's share of elite {position} finishes. It does not use average PPG by age.",
        "",
        f"Rate % = players at that age inside Top N {position} slots / all Top N {position} slots.",
        "",
        "## Data Source",
        "",
        f"- Source: {metadata['data_source']}",
        f"- Releases: {metadata['source_releases']}",
        f"- Timeline: {metadata['timeline']}",
        f"- Seasons used: {metadata['season_count_used'] or 'All available'}",
        f"- Scoring: {metadata['scoring_label']}",
        f"- Player-seasons analyzed: {metadata['player_seasons']:,}",
        f"- Cache file: `{metadata['cache_file']}`",
        "",
        "## Peaks",
        "",
    ]

    for threshold, label in _position_rate_specs(position):
        key = f"{threshold}_rate"
        peak = peaks.get(key)
        if peak:
            lines.append(f"- Peak Age for {label} Rate: Age {peak['age']} ({peak['rate_pct']:.2f}%)")
        else:
            lines.append(f"- Peak Age for {label} Rate: N/A")

    table_df = _position_table_df(summary_df, position)
    lines.extend(["", "## Table", "", _dataframe_to_markdown(table_df), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_multi_line_chart(
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
    thresholds=None,
    subtitle: str | None = None,
    position: str | None = None,
    show_career_arc: bool = False,
    avg_top1_age: float | None = None,
    show_peak_labels: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    height = 700
    margin_left = 78
    margin_right = 190
    margin_top = 126
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
        ".subtitle { font-size: 14px; fill: #64748b; }",
        ".label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".legend { font-size: 14px; font-weight: 700; }",
        ".marker-label { font-size: 11px; font-weight: 800; }",
        ".arc-label { font-size: 12px; font-weight: 800; fill: #475569; text-transform: uppercase; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.4; }",
        ".rate-line { fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { stroke: #ffffff; stroke-width: 1.4; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="38">{title}</text>',
    ]
    if subtitle:
        lines.append(f'<text class="subtitle" x="{margin_left}" y="62">{subtitle}</text>')

    if show_career_arc:
        arc_regions = CAREER_ARCS.get(position or "RB", CAREER_ARCS["RB"])
        for label, start_age, end_age, fill in arc_regions:
            end_age = x_max if end_age is None else end_age
            if end_age < x_min or start_age > x_max:
                continue
            left_age = max(x_min, start_age - 0.5)
            right_age = min(x_max, end_age + 0.5)
            x1 = x_pos(left_age)
            x2 = x_pos(right_age)
            lines.append(
                f'<rect x="{x1:.2f}" y="{margin_top}" width="{max(0, x2 - x1):.2f}" height="{plot_height}" '
                f'fill="{fill}" opacity="0.36"/>'
            )
            lines.append(
                f'<text class="arc-label" x="{(x1 + x2) / 2:.2f}" y="{margin_top - 14}" text-anchor="middle">{label}</text>'
            )

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
        f'<text class="label" transform="translate(22 {height / 2:.2f}) rotate(-90)" text-anchor="middle">Share of Finishes (%)</text>',
    ])

    for index, (threshold, label) in enumerate(chart_specs):
        rate_col = f"{threshold}_rate_pct"
        color = RATE_COLORS[threshold]
        peak_row = chart_df.loc[chart_df[rate_col].idxmax()]
        peak_age = int(peak_row["Age"])
        x = x_pos(peak_age)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" '
            f'stroke="{color}" stroke-width="1.35" stroke-dasharray="5 6" opacity="0.42"/>'
        )
        if show_peak_labels:
            label_y = margin_top - 56 + index * 13
            lines.append(
                f'<text class="marker-label" x="{width - margin_right + 8:.2f}" y="{label_y:.2f}" '
                f'fill="{color}">{label} Peak: {peak_age}</text>'
            )

    if avg_top1_age is not None and x_min <= avg_top1_age <= x_max:
        x = x_pos(avg_top1_age)
        label_y = margin_top + 18 + len(chart_specs) * 15
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" '
            f'stroke="#111827" stroke-width="1.8" stroke-dasharray="8 5" opacity="0.78"/>'
        )
        text_x = min(x + 8, width - margin_right - 105)
        lines.append(f'<text class="marker-label" x="{text_x:.2f}" y="{label_y:.2f}" fill="#111827">Average #1 Age</text>')
        lines.append(f'<text class="marker-label" x="{text_x:.2f}" y="{label_y + 14:.2f}" fill="#111827">{avg_top1_age:.1f}</text>')

    for threshold, label in chart_specs:
        rate_col = f"{threshold}_rate_pct"
        count_col = f"{threshold}_rb_seasons"
        color = RATE_COLORS[threshold]
        mean_rate = float(chart_df[rate_col].mean())
        mean_y = y_pos(mean_rate)
        lines.append(
            f'<line x1="{margin_left}" y1="{mean_y:.2f}" x2="{width - margin_right}" y2="{mean_y:.2f}" '
            f'stroke="{color}" stroke-width="1.2" stroke-dasharray="7 6" opacity="0.55"/>'
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
                f"Player seasons at age: {int(row['total_rb_seasons'])}\n"
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
    ax.set_ylabel("Share of Top-N Slots (%)")
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
            linewidth=1.7,
            label=label,
            color=RATE_COLORS[threshold],
        )
        mean_rate = float(summary_df[f"{threshold}_rate_pct"].mean())
        ax.axhline(
            mean_rate,
            color=RATE_COLORS[threshold],
            linestyle="--",
            linewidth=1.0,
            alpha=0.45,
        )

    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Share of Top-N Slots (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def render_age_curve_chart(
    study: Dict[str, object],
    position: str,
    output_path: Path,
    show_career_arc: bool = True,
    show_peak_labels: bool = False,
) -> None:
    labels = [label for _threshold, label in _position_rate_specs(position)]
    title = f"{position} Peak Performance Curve by Age"
    subtitle = f"Historical distribution of {', '.join(labels[:-1])}, and {labels[-1]} {position} finishes by age"
    _write_svg_multi_line_chart(
        study["summary"],
        title,
        output_path,
        thresholds=[threshold for threshold, _label in _position_rate_specs(position)],
        subtitle=subtitle,
        position=position,
        show_career_arc=show_career_arc,
        avg_top1_age=None,
        show_peak_labels=show_peak_labels,
    )


def _write_position_comparison_chart(
    studies: Dict[str, Dict[str, object]],
    threshold: str,
    label: str,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, study in studies.items():
        summary_df = study["summary"].copy()
        rate_col = f"{threshold}_rate_pct"
        if rate_col not in summary_df.columns:
            continue
        for _, row in summary_df.iterrows():
            rows.append({"Position": position, "Age": int(row["Age"]), "Rate": float(row[rate_col])})
    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    width = 1100
    height = 620
    margin_left = 78
    margin_right = 160
    margin_top = 94
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = int(chart_df["Age"].min())
    x_max = int(chart_df["Age"].max())
    y_max = max(5.0, float(chart_df["Rate"].max()) * 1.15)
    position_colors = {"RB": "#2563eb", "WR": "#16a34a", "QB": "#ca8a04", "TE": "#ea580c"}

    def x_pos(age: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + ((age - x_min) / (x_max - x_min)) * plot_width

    def y_pos(rate: float) -> float:
        return margin_top + plot_height - (rate / y_max) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 27px; font-weight: 800; }",
        ".subtitle { font-size: 14px; fill: #64748b; }",
        ".label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".legend { font-size: 14px; font-weight: 800; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.4; }",
        ".rate-line { fill: none; stroke-width: 1.9; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { stroke: #ffffff; stroke-width: 1.3; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="38">{label} Comparison by Position</text>',
        f'<text class="subtitle" x="{margin_left}" y="62">Overlay of {label} age-share curves across RB, WR, QB, and TE</text>',
    ]
    for tick in [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.1f}%</text>')
    for age in range(x_min, x_max + 1):
        x = x_pos(age)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{height - 42}" text-anchor="middle">{age}</text>')
    lines.extend([
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
        f'<text class="label" x="{(width - margin_right + margin_left) / 2:.2f}" y="{height - 12}" text-anchor="middle">Age</text>',
        f'<text class="label" transform="translate(22 {height / 2:.2f}) rotate(-90)" text-anchor="middle">Share of Finishes (%)</text>',
    ])

    for position in POSITIONS:
        position_df = chart_df[chart_df["Position"] == position].sort_values("Age")
        if position_df.empty:
            continue
        color = position_colors[position]
        points = [(x_pos(row["Age"]), y_pos(row["Rate"]), row) for _, row in position_df.iterrows()]
        path_data = " ".join(f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y, _row) in enumerate(points))
        lines.append(f'<path class="rate-line" stroke="{color}" d="{path_data}"/>')
        for x, y, row in points:
            lines.append(f'<circle class="dot" fill="{color}" cx="{x:.2f}" cy="{y:.2f}" r="3.9"><title>{position} Age {int(row["Age"])}: {float(row["Rate"]):.2f}%</title></circle>')

    legend_x = width - margin_right + 34
    legend_y = margin_top + 8
    for index, position in enumerate(POSITIONS):
        y = legend_y + index * 30
        color = position_colors[position]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        lines.append(f'<text class="legend" x="{legend_x + 38}" y="{y + 5}">{position}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_position_comparison_view(studies: Dict[str, Dict[str, object]], output_dir: Path) -> Dict[str, Path]:
    comparison_paths = {}
    for threshold, label in COMPARISON_RATE_SPECS:
        path = output_dir / f"position_comparison_{threshold}.svg"
        _write_position_comparison_chart(studies, threshold, label, path)
        comparison_paths[threshold] = path
    return comparison_paths


def _mobile_qb_rushing_summary_by_age(seasons_df: pd.DataFrame, min_rush_ypg: float = 15.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = {"player_id", "player_name", "position", "age", "rushing_yards"}
    if seasons_df.empty or not required_cols.issubset(seasons_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    qb_df = seasons_df[seasons_df["position"].astype(str).str.upper().eq("QB")].copy()
    qb_df["_rushing_yards"] = pd.to_numeric(qb_df["rushing_yards"], errors="coerce").fillna(0.0)
    qb_df = qb_df[qb_df["player_name"].astype(str).isin(MOBILE_QB_QUALIFIER_NAMES)].copy()
    if qb_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    qualifier_df = (
        qb_df.groupby(["player_name"], as_index=False)
        .agg(seasons=("season", "nunique"), rushing_yards=("_rushing_yards", "sum"))
    )
    qualifier_df["qualifier"] = f"Provided 15+ Rush YPG List"
    if qualifier_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    qualified_names = set(qualifier_df["player_name"].astype(str))
    mobile_df = qb_df[qb_df["player_name"].astype(str).isin(qualified_names)].copy()
    mobile_df["_age"] = pd.to_numeric(mobile_df["age"], errors="coerce").round().astype("Int64")
    mobile_df = mobile_df[mobile_df["_age"].notna()].copy()
    if mobile_df.empty:
        return pd.DataFrame(), qualifier_df

    summary_df = (
        mobile_df.groupby("_age", as_index=False)
        .agg(
            average_rushing_yards=("_rushing_yards", "mean"),
            median_rushing_yards=("_rushing_yards", "median"),
            player_seasons=("_rushing_yards", "size"),
            qualified_qbs=("player_id", "nunique"),
        )
        .rename(columns={"_age": "Age"})
        .sort_values("Age")
    )
    qualifier_df = qualifier_df.sort_values("rushing_yards", ascending=False)
    return summary_df, qualifier_df


def _write_svg_mobile_qb_rushing_chart(
    summary_df: pd.DataFrame,
    qualifier_df: pd.DataFrame,
    output_path: Path,
    min_rush_ypg: float = 15.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    width = 1100
    height = 590
    margin_left = 78
    margin_right = 190
    margin_top = 86
    margin_bottom = 68
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_min = int(summary_df["Age"].min())
    x_max = int(summary_df["Age"].max())
    y_max = max(1.0, float(summary_df["average_rushing_yards"].max()) * 1.15)
    qualified_count = int(len(qualifier_df))

    def x_pos(age: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + ((age - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + plot_height - (value / y_max) * plot_height

    points = [
        (x_pos(float(row["Age"])), y_pos(float(row["average_rushing_yards"])), row)
        for _, row in summary_df.iterrows()
    ]
    path_data = " ".join(f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y, _row) in enumerate(points))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 27px; font-weight: 800; }",
        ".subtitle { font-size: 14px; fill: #64748b; }",
        ".label { font-size: 15px; font-weight: 700; }",
        ".tick { font-size: 12px; fill: #4b5563; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis { stroke: #111827; stroke-width: 1.4; }",
        ".rush-line { fill: none; stroke: #7c3aed; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { fill: #7c3aed; stroke: #ffffff; stroke-width: 1.4; }",
        ".note { font-size: 12px; fill: #475569; font-weight: 700; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{margin_left}" y="38">Mobile QB Rushing Yards by Age</text>',
        f'<text class="subtitle" x="{margin_left}" y="62">Qualified QBs come from the provided 15+ rushing yards per game mobile-QB list</text>',
    ]
    for tick in [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]:
        y = y_pos(tick)
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end">{tick:.0f}</text>')
    for age in range(x_min, x_max + 1):
        x = x_pos(age)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{height - 40}" text-anchor="middle">{age}</text>')
    lines.extend([
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>',
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>',
        f'<text class="label" x="{(width - margin_right + margin_left) / 2:.2f}" y="{height - 12}" text-anchor="middle">Age</text>',
        f'<text class="label" transform="translate(22 {height / 2:.2f}) rotate(-90)" text-anchor="middle">Avg Rush Yards / Season</text>',
        f'<path class="rush-line" d="{path_data}"/>',
    ])
    for x, y, row in points:
        tooltip = (
            "Mobile QB rushing yards by age\n"
            f"Age: {int(row['Age'])}\n"
            f"Average rush yards: {float(row['average_rushing_yards']):.1f}\n"
            f"Median rush yards: {float(row['median_rushing_yards']):.1f}\n"
            f"Player-seasons: {int(row['player_seasons'])}\n"
            f"Qualified QBs: {int(row['qualified_qbs'])}"
        )
        lines.append(f'<circle class="dot" cx="{x:.2f}" cy="{y:.2f}" r="4.8"><title>{tooltip}</title></circle>')

    legend_x = width - margin_right + 30
    lines.append(f'<text class="note" x="{legend_x}" y="{margin_top + 12}">Qualified QBs: {qualified_count}</text>')
    lines.append(f'<text class="note" x="{legend_x}" y="{margin_top + 32}">Source: provided qualifier list</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_mobile_qb_rushing_chart(seasons_df: pd.DataFrame, output_dir: Path) -> tuple[Path | None, pd.DataFrame]:
    summary_df, qualifier_df = _mobile_qb_rushing_summary_by_age(seasons_df)
    if summary_df.empty:
        return None, qualifier_df
    path = output_dir / "qb_mobile_rushing_yards_by_age.svg"
    _write_svg_mobile_qb_rushing_chart(summary_df, qualifier_df, path)
    return path, qualifier_df


def plot_age_heatmap(
    age_matrix,
    percentage_matrix,
    peak_summary,
    stats_summary,
    show_counts_only=False,
    show_trend=False,
    save_path=None,
    transparent=False,
):
    if plt is None:
        return None

    matrix_df = pd.DataFrame(age_matrix).copy()
    pct_df = pd.DataFrame(percentage_matrix).reindex(index=matrix_df.index, columns=matrix_df.columns).fillna(0.0)
    positions = list(matrix_df.index)
    ages = [int(age) for age in matrix_df.columns]
    values = matrix_df.to_numpy(dtype=float)
    masked_values = np.ma.masked_where(values == 0, values)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "guaranteed_play_blues",
        ["#f8fafc", "#e8f1fb", "#b9d8f4", "#5aa2d6", "#0f5f9f", "#083b6f"],
    )
    cmap = cmap.copy()
    cmap.set_bad("#f8fafc")
    max_value = max(1.0, float(np.nanmax(values)))
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, max(4.5, max_value + 0.5)]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    fig_width = max(12.5, len(ages) * 0.62)
    fig_height = 10.4
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="none" if transparent else "white")
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[4.9, 1.35, 1.45], hspace=0.48)
    ax = fig.add_subplot(gs[0])
    peak_table_ax = fig.add_subplot(gs[1])
    stats_table_ax = fig.add_subplot(gs[2])

    image = ax.imshow(masked_values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(
        "#1 Overall Fantasy Finishes by Age",
        fontsize=24,
        fontweight="bold",
        loc="left",
        pad=42,
    )
    ax.text(
        0,
        1.055,
        "Historical peak-age analysis of positional #1 overall fantasy finishes",
        transform=ax.transAxes,
        fontsize=13.5,
        color="#475569",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Age", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_ylabel("Position", fontsize=14, fontweight="bold", labelpad=12)
    ax.set_xticks(range(len(ages)))
    ax.set_xticklabels(ages, fontsize=11)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=13, fontweight="bold")

    for row_index, position in enumerate(positions):
        for col_index, age in enumerate(ages):
            count = int(matrix_df.loc[position, age])
            if count == 0:
                continue
            pct = float(pct_df.loc[position, age])
            text_color = "white" if count >= 4 else "#0f172a"
            ax.text(
                col_index,
                row_index - (0.08 if not show_counts_only else 0),
                f"{count}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=text_color,
            )
            if not show_counts_only:
                ax.text(
                    col_index,
                    row_index + 0.20,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="normal",
                    color=text_color,
                )

    for _, row in peak_summary.iterrows():
        position = row["Position"]
        peak_ages = row["Peak Age"]
        if not isinstance(peak_ages, list):
            peak_ages = [peak_ages]
        if position not in positions:
            continue
        row_index = positions.index(position)
        for age in peak_ages:
            if int(age) not in ages:
                continue
            col_index = ages.index(int(age))
            ax.add_patch(
                Rectangle(
                    (col_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#f59e0b",
                    linewidth=3.6,
                )
            )

    if show_trend and not stats_summary.empty:
        for _, row in stats_summary.iterrows():
            position = row["Position"]
            if position not in positions:
                continue
            mean_age = float(row["Mean Age"])
            if mean_age < min(ages) or mean_age > max(ages):
                continue
            row_index = positions.index(position)
            trend_x = mean_age - min(ages)
            ax.scatter(
                [trend_x],
                [row_index - 0.39],
                marker="v",
                s=72,
                color="#111827",
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )

    ax.set_xticks([x - 0.5 for x in range(1, len(ages))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(positions))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.018, boundaries=boundaries)
    cbar.set_label("#1 Overall Finishes", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    peak_table_ax.axis("off")
    display_summary = peak_summary.copy()
    display_summary["Peak Age(s)"] = display_summary["Peak Age"].apply(
        lambda value: ", ".join(str(age) for age in value) if isinstance(value, list) else str(value)
    )
    peak_table_ax.text(
        0,
        1.08,
        "Peak Summary",
        transform=peak_table_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
    )
    table = peak_table_ax.table(
        cellText=display_summary[["Position", "Peak Age(s)", "Peak Count", "Peak Rate"]].values,
        colLabels=["Position", "Peak Age(s)", "Peak Count", "Peak Rate"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.45)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        else:
            cell.set_facecolor("#ffffff")

    stats_table_ax.axis("off")
    stats_display = stats_summary.copy()
    for column in ["Mean Age", "Median Age", "Weighted Peak Age"]:
        stats_display[column] = stats_display[column].map(lambda value: f"{float(value):.1f}" if pd.notna(value) else "N/A")
    stats_table_ax.text(
        0,
        1.08,
        "Statistical Summary",
        transform=stats_table_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
    )
    stats_table = stats_table_ax.table(
        cellText=stats_display[
            ["Position", "Mean Age", "Median Age", "Weighted Peak Age", "Total #1 Seasons"]
        ].values,
        colLabels=["Position", "Mean Age", "Median Age", "Weighted Peak Age", "Total #1 Seasons"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(11)
    stats_table.scale(1, 1.45)
    for (row, _col), cell in stats_table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        else:
            cell.set_facecolor("#ffffff")

    fig.text(
        0.01,
        0.012,
        "Source: Guaranteed Play Historical Database",
        fontsize=9.5,
        color="#64748b",
        ha="left",
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=transparent)

    return fig


def _number_one_heatmap_inputs(studies: Dict[str, Dict[str, object]]):
    nonzero_ages = [
        int(age)
        for study in studies.values()
        for age in study["summary"].loc[study["summary"]["top1_rb_seasons"] > 0, "Age"].tolist()
    ]
    if not nonzero_ages:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    min_age = min(nonzero_ages)
    max_age = max(nonzero_ages)
    ages = list(range(min_age, max_age + 1))
    counts = pd.DataFrame(0, index=POSITIONS, columns=ages)
    percentages = pd.DataFrame(0.0, index=POSITIONS, columns=ages)
    peak_rows = []
    stats_rows = []

    for position, study in studies.items():
        summary_df = study["summary"].set_index("Age")
        for age in ages:
            if age in summary_df.index:
                counts.loc[position, age] = int(summary_df.loc[age, "top1_rb_seasons"])
                percentages.loc[position, age] = float(summary_df.loc[age, "top1_rate_pct"])

        max_count = int(counts.loc[position].max())
        peak_ages = [int(age) for age, value in counts.loc[position].items() if int(value) == max_count and max_count > 0]
        peak_rate = max(float(percentages.loc[position, age]) for age in peak_ages) if peak_ages else 0.0
        total_finishes = int(counts.loc[position].sum())
        if total_finishes:
            expanded_ages = [
                int(age)
                for age, count in counts.loc[position].items()
                for _ in range(int(count))
            ]
            mean_age = float(np.average(ages, weights=counts.loc[position].to_numpy(dtype=float)))
            median_age = float(np.median(expanded_ages))
            weighted_peak_age = float(np.average(peak_ages)) if peak_ages else np.nan
        else:
            mean_age = np.nan
            median_age = np.nan
            weighted_peak_age = np.nan
        peak_rows.append(
            {
                "Position": position,
                "Peak Age": peak_ages if len(peak_ages) != 1 else peak_ages[0],
                "Peak Count": max_count,
                "Peak Rate": f"{peak_rate:.1f}%",
            }
        )
        stats_rows.append(
            {
                "Position": position,
                "Mean Age": mean_age,
                "Median Age": median_age,
                "Weighted Peak Age": weighted_peak_age,
                "Total #1 Seasons": total_finishes,
            }
        )

    return counts, percentages, pd.DataFrame(peak_rows), pd.DataFrame(stats_rows)


def _write_number_one_comparison_chart(
    studies: Dict[str, Dict[str, object]],
    output_path: Path,
    show_counts_only: bool = False,
) -> bool:
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_path.with_suffix(".svg")

    counts, percentages, peak_summary, stats_summary = _number_one_heatmap_inputs(studies)
    if counts.empty:
        return False

    fig = plot_age_heatmap(
        counts,
        percentages,
        peak_summary,
        stats_summary,
        show_counts_only=show_counts_only,
        show_trend=False,
        save_path=output_path,
    )
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_number_one_comparison_report(
    studies: Dict[str, Dict[str, object]],
    metadata: Dict[str, object],
    chart_path: Path,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peak_items = []
    _counts, _percentages, peak_summary, stats_summary = _number_one_heatmap_inputs(studies)
    for position, study in studies.items():
        peak = study["peaks"].get("top1_rate")
        if peak:
            peak_items.append(
                f"<li>{position}: Age {peak['age']} - {peak['top1_rb_seasons']} finishes ({peak['rate_pct']:.2f}%)</li>"
            )
    display_peak_summary = peak_summary.copy()
    if not display_peak_summary.empty:
        display_peak_summary["Peak Age(s)"] = display_peak_summary["Peak Age"].apply(
            lambda value: ", ".join(str(age) for age in value) if isinstance(value, list) else str(value)
        )
        display_peak_summary = display_peak_summary[["Position", "Peak Age(s)", "Peak Count", "Peak Rate"]]
    display_stats_summary = stats_summary.copy()
    if not display_stats_summary.empty:
        for column in ["Mean Age", "Median Age", "Weighted Peak Age"]:
            display_stats_summary[column] = display_stats_summary[column].map(
                lambda value: f"{float(value):.1f}" if pd.notna(value) else "N/A"
            )
    summary_table_html = display_peak_summary.to_html(index=False, classes="summary-table", border=0) if not display_peak_summary.empty else ""
    stats_table_html = display_stats_summary.to_html(index=False, classes="summary-table", border=0) if not display_stats_summary.empty else ""

    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>#1 Overall Age Share</title>
    <style>
        body {{ margin: 0; background: #f4f7fb; color: #111827; font-family: Arial, sans-serif; }}
        main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 60px; }}
        h1 {{ margin: 0 0 8px; font-size: 34px; }}
        .subtitle {{ color: #4b5563; line-height: 1.5; }}
        .card, .chart {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 18px; margin: 22px 0; }}
        .chart img {{ width: 100%; height: auto; display: block; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px 12px; text-align: center; }}
        th {{ background: #f1f5f9; font-weight: 800; }}
    </style>
</head>
<body>
    <main>
        <h1>#1 Overall Finishes by Age</h1>
        <p class="subtitle">Separate comparison page for #1 overall fantasy seasons by position. The heatmap uses finish counts, with percent share shown in each nonzero cell.</p>
        <p class="subtitle">Timeline: {metadata['timeline']} | Scoring: {metadata['scoring_label']}</p>
        <section class="card">
            <h2>Peak #1 Ages</h2>
            <ul>{''.join(peak_items)}</ul>
        </section>
        <section class="chart"><img src="{chart_path.name}" alt="#1 overall age share by position"></section>
        <section class="card">
            <h2>Peak Summary</h2>
            {summary_table_html}
        </section>
        <section class="card">
            <h2>Statistical Summary</h2>
            {stats_table_html}
        </section>
        <p><a href="fantasy_age_study_dashboard.html">Back to positional age-share report</a></p>
    </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _write_multi_position_html_report(
    studies: Dict[str, Dict[str, object]],
    metadata: Dict[str, object],
    chart_paths: Dict[str, Dict[str, Path]],
    comparison_paths: Dict[str, Path],
    mobile_qb_chart_path: Path | None,
    mobile_qb_qualifiers: pd.DataFrame,
    number_one_report_path: Path,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def svg_body(path: Path) -> str:
        path = Path(path)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    tabs = []
    panels = []
    for index, position in enumerate(POSITIONS):
        study = studies[position]
        summary_df = study["summary"]
        checked = " checked" if index == 0 else ""
        tabs.append(f'<input type="radio" id="tab-{position}" name="position-tabs"{checked}>')
        tabs.append(f'<label class="tab-label" for="tab-{position}">{position}</label>')

        table_df = _position_table_df(summary_df, position)
        table_html = table_df.to_html(index=False, classes="rate-table", border=0)
        summary_table = render_peak_summary_table(study, position)
        executive_insight = render_executive_insight(study, position)
        takeaways_panel = render_position_takeaways(study, position)
        advanced_stats = render_advanced_stats_section(study, position)
        mobile_qb_section = ""
        if position == "QB":
            if mobile_qb_chart_path and Path(mobile_qb_chart_path).exists():
                qualifier_table = ""
                if not mobile_qb_qualifiers.empty:
                    display_qualifiers = mobile_qb_qualifiers[["player_name", "seasons", "rushing_yards", "qualifier"]].copy()
                    display_qualifiers["seasons"] = display_qualifiers["seasons"].map(lambda value: f"{float(value):.0f}")
                    display_qualifiers["rushing_yards"] = display_qualifiers["rushing_yards"].map(lambda value: f"{float(value):.0f}")
                    display_qualifiers = display_qualifiers.rename(
                        columns={
                            "player_name": "Player",
                            "seasons": "Seasons",
                            "rushing_yards": "Rush Yards",
                            "qualifier": "Qualifier",
                        }
                    )
                    qualifier_table = display_qualifiers.head(25).to_html(index=False, classes="summary-table", border=0)
                mobile_qb_section = f"""
                <section class="table-wrap summary-wrap">
                    <h2>Mobile QB Rushing Curve</h2>
                    <p class="subtitle">QB rushing yards by age for quarterbacks from the provided 15+ rushing yards per game mobile-QB qualifier list.</p>
                    <section class="chart">{svg_body(mobile_qb_chart_path)}</section>
                    <details class="advanced-stats">
                        <summary>Qualified Mobile QBs</summary>
                        {qualifier_table}
                    </details>
                </section>
                """

        panels.append(
            f"""
            <section class="tab-panel panel-{position}">
                <section class="table-wrap summary-wrap">
                    <h2>{position} Peak Summary</h2>
                    {summary_table}
                </section>
                {executive_insight}
                {''.join(f'<section class="chart">{svg_body(path)}</section>' for path in chart_paths[position].values())}
                {takeaways_panel}
                {mobile_qb_section}
                {advanced_stats}
                <details class="advanced-stats">
                    <summary>{position} Raw Age Share Table</summary>
                    {table_html}
                </details>
            </section>
            """
        )

    comparison_tabs = []
    comparison_sections = []
    for index, (threshold, label) in enumerate(COMPARISON_RATE_SPECS):
        checked = " checked" if index == 2 else ""
        comparison_tabs.append(f'<input type="radio" id="compare-{threshold}" name="comparison-metric"{checked}>')
        comparison_tabs.append(f'<label class="metric-label" for="compare-{threshold}">{label}</label>')
        comparison_sections.append(
            f'<section class="comparison-panel comparison-{threshold} chart">{svg_body(comparison_paths[threshold])}</section>'
        )
    position_summary_table = _position_insight_summary_table(studies)

    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>RB/WR/QB/TE Elite Season Age Study</title>
    <style>
        body {{ margin: 0; background: #f4f7fb; color: #111827; font-family: Arial, sans-serif; }}
        main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 60px; }}
        h1 {{ margin: 0 0 8px; font-size: 34px; }}
        .subtitle {{ margin: 0 0 22px; color: #4b5563; line-height: 1.5; }}
        .meta {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }}
        .meta-card, .card, .chart, .table-wrap, .insight-panel, .executive-insight, .advanced-stats {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; }}
        .meta-card {{ padding: 14px 16px; }}
        .meta-label, .label {{ color: #64748b; font-size: 12px; font-weight: 800; text-transform: uppercase; }}
        .meta-value {{ margin-top: 6px; font-size: 15px; font-weight: 700; line-height: 1.35; }}
        .view-label {{ display: inline-block; padding: 12px 22px; margin: 0 8px 22px 0; border: 1px solid #cbd5e1; border-radius: 999px; background: #e2e8f0; font-weight: 800; cursor: pointer; }}
        input[name="dashboard-view"] {{ display: none; }}
        .position-view, .comparison-view {{ display: none; }}
        #view-position:checked ~ .position-view, #view-comparison:checked ~ .comparison-view {{ display: block; }}
        #view-position:checked + label, #view-comparison:checked + label {{ background: #0f172a; border-color: #0f172a; color: #ffffff; }}
        .tab-label {{ display: inline-block; padding: 12px 22px; margin-right: 8px; border: 1px solid #cbd5e1; border-radius: 8px 8px 0 0; background: #e2e8f0; font-weight: 800; cursor: pointer; }}
        input[name="position-tabs"] {{ display: none; }}
        .tab-panel {{ display: none; border-top: 2px solid #cbd5e1; padding-top: 22px; }}
        #tab-RB:checked ~ .panel-RB, #tab-WR:checked ~ .panel-WR, #tab-QB:checked ~ .panel-QB, #tab-TE:checked ~ .panel-TE {{ display: block; }}
        #tab-RB:checked + label, #tab-WR:checked + label, #tab-QB:checked + label, #tab-TE:checked + label {{ background: #ffffff; border-bottom-color: #ffffff; }}
        .metric-label {{ display: inline-block; padding: 10px 16px; margin: 0 8px 16px 0; border: 1px solid #cbd5e1; border-radius: 999px; background: #f1f5f9; font-weight: 800; cursor: pointer; }}
        input[name="comparison-metric"] {{ display: none; }}
        .comparison-panel {{ display: none; }}
        #compare-top36:checked ~ .comparison-top36,
        #compare-top24:checked ~ .comparison-top24,
        #compare-top12:checked ~ .comparison-top12,
        #compare-top5:checked ~ .comparison-top5,
        #compare-top3:checked ~ .comparison-top3,
        #compare-top1:checked ~ .comparison-top1 {{ display: block; }}
        #compare-top36:checked + label,
        #compare-top24:checked + label,
        #compare-top12:checked + label,
        #compare-top5:checked + label,
        #compare-top3:checked + label,
        #compare-top1:checked + label {{ background: #0f172a; border-color: #0f172a; color: #ffffff; }}
        .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 26px; }}
        .card {{ padding: 18px; }}
        .value {{ margin-top: 8px; font-size: 32px; font-weight: 800; }}
        .detail {{ margin-top: 4px; color: #047857; font-size: 18px; font-weight: 700; }}
        .chart {{ margin: 22px 0; padding: 10px; }}
        .chart svg {{ width: 100%; height: auto; }}
        .table-wrap {{ margin-top: 24px; padding: 18px; overflow-x: auto; }}
        .summary-wrap {{ margin-bottom: 18px; }}
        .insight-panel, .executive-insight, .advanced-stats {{ margin: 22px 0; padding: 18px 22px; }}
        .executive-insight .eyebrow {{ color: #047857; font-size: 12px; font-weight: 900; letter-spacing: .04em; text-transform: uppercase; }}
        .executive-insight p {{ margin: 8px 0 0; font-size: 20px; font-weight: 800; line-height: 1.35; }}
        .insight-panel h2, .table-wrap h2 {{ margin-top: 0; }}
        .insight-panel li {{ margin: 8px 0; line-height: 1.45; }}
        .advanced-stats summary {{ cursor: pointer; font-size: 18px; font-weight: 800; }}
        .advanced-stats table {{ margin-top: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background: #f8fafc; font-weight: 800; }}
    </style>
</head>
<body>
    <main>
        <h1>RB/WR/QB/TE Elite Season Age Study</h1>
        <p class="subtitle">Age share inside elite finish groups. Rate % = players at that age inside Top N slots / all Top N slots.</p>
        <p class="subtitle"><a href="{number_one_report_path.name}">Open separate #1 overall comparison page</a></p>
        <section class="meta">
            <div class="meta-card"><div class="meta-label">Data Source</div><div class="meta-value">{metadata['data_source']}</div></div>
            <div class="meta-card"><div class="meta-label">Timeline</div><div class="meta-value">{metadata['timeline']}</div></div>
            <div class="meta-card"><div class="meta-label">Seasons Used</div><div class="meta-value">{metadata['season_count_used'] or 'All available'}</div></div>
            <div class="meta-card"><div class="meta-label">Scoring</div><div class="meta-value">{metadata['scoring_label']}</div></div>
            <div class="meta-card"><div class="meta-label">Player-Seasons</div><div class="meta-value">{metadata['player_seasons']:,}</div></div>
        </section>
        <input type="radio" id="view-position" name="dashboard-view" checked>
        <label class="view-label" for="view-position">Position View</label>
        <input type="radio" id="view-comparison" name="dashboard-view">
        <label class="view-label" for="view-comparison">Comparison View</label>
        <section class="position-view">
            {''.join(tabs)}
            {''.join(panels)}
        </section>
        <section class="comparison-view">
            <section class="table-wrap summary-wrap">
                <h2>Position Insight Summary</h2>
                {position_summary_table}
            </section>
            <section class="table-wrap summary-wrap">
                <h2>Comparison Metric</h2>
                <p class="subtitle">Select one finish tier to compare aging curves across positions.</p>
                {''.join(comparison_tabs)}
                {''.join(comparison_sections)}
            </section>
        </section>
    </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


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
    <title>RB Elite Season Age Study</title>
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
            grid-template-columns: repeat(5, 1fr);
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
        <h1>RB Elite Season Age Study</h1>
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
                <div class="meta-label">Seasons Used</div>
                <div class="meta-value">{metadata['season_count_used'] or 'All available'}</div>
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
    parser = argparse.ArgumentParser(description="Run the standalone fantasy football age study.")
    parser.add_argument(
        "--scoring",
        choices=["ppr", "full_ppr", "half_ppr", "full", "half", "dad", "dads", "dad_settings"],
        default=None,
        help="Optional scoring override. If omitted, the script asks in the console.",
    )
    parser.add_argument("--no-prompt", action="store_true", help="Use Full PPR without asking for a scoring choice.")
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Number of most-recent seasons to use. Values above the available data use the max.",
    )
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

    print(f"Pulling/caching historical RB/WR/QB/TE player-season data ({scoring_label})...")
    seasons_df = pull_historical_seasons_df(scoring=scoring, force_refresh=args.refresh, positions=POSITIONS)
    season_count = _choose_season_count(seasons_df, args.years, args.no_prompt)
    seasons_df = _filter_recent_seasons(seasons_df, season_count)
    metadata = _build_study_metadata(seasons_df, scoring)

    print(f"Calculating RB/WR/QB/TE age share inside top-N finish groups for {metadata['timeline']}...")
    studies = {
        position: build_rb_elite_age_study(
            seasons_df,
            min_sample_size=args.min_sample_size,
            position=position,
        )
        for position in POSITIONS
    }
    if any(study["summary"].empty for study in studies.values()):
        raise SystemExit("One or more positions had no ages after filtering.")

    html_path = output_dir / "fantasy_age_study_dashboard.html"
    number_one_html_path = output_dir / "number_one_overall_age_share.html"
    number_one_chart_path = output_dir / "number_one_overall_age_share.png"
    chart_paths = {}
    grouped_png_paths = {}
    wrote_matplotlib = []

    for position, study in studies.items():
        position_key = position.lower()
        summary_df = study["summary"]
        table_path = output_dir / f"{position_key}_elite_age_rates.csv"
        report_path = output_dir / f"{position_key}_elite_age_study.md"
        summary_df.to_csv(table_path, index=False)
        _write_markdown_report(study, metadata, report_path, position=position)

        chart_paths[position] = {}
        grouped_png_paths[position] = {}
        for group_key, spec in RATE_GROUPS.items():
            svg_path = output_dir / spec["svg_name"].format(position_key=position_key)
            png_path = output_dir / spec["png_name"].format(position_key=position_key)
            title = f"{position} Peak Performance Curve by Age"
            chart_paths[position][group_key] = svg_path
            grouped_png_paths[position][group_key] = png_path
            render_age_curve_chart(study, position, svg_path)
            wrote_matplotlib.append(
                _write_matplotlib_multi_chart(
                    summary_df,
                    title,
                    png_path,
                    thresholds=[threshold for threshold, _label in _position_rate_specs(position)],
                )
            )

    comparison_paths = render_position_comparison_view(studies, output_dir)
    mobile_qb_chart_path, mobile_qb_qualifiers = render_mobile_qb_rushing_chart(seasons_df, output_dir)
    _write_number_one_comparison_chart(studies, number_one_chart_path)
    _write_number_one_comparison_report(studies, metadata, number_one_chart_path, number_one_html_path)
    _write_multi_position_html_report(
        studies,
        metadata,
        chart_paths,
        comparison_paths,
        mobile_qb_chart_path,
        mobile_qb_qualifiers,
        number_one_html_path,
        html_path,
    )

    # Individual per-threshold charts were intentionally removed from the main report.
    if False:
        for threshold, label in RATE_SPECS:
            _write_svg_line_chart(
                summary_df,
                f"{threshold}_rate_pct",
                f"{threshold}_rb_seasons",
                f"{label} Age Share",
                output_dir / f"{threshold}_rate_by_age.svg",
            )

    print("")
    print("DATA SOURCE")
    print("-----------")
    print(f"Source: {metadata['data_source']}")
    print(f"Releases: {metadata['source_releases']}")
    print(f"Timeline: {metadata['timeline']}")
    if metadata.get("season_count_used"):
        print(
            f"Seasons used: {metadata['season_count_used']} "
            f"of {metadata.get('available_season_count') or metadata['season_count_used']} available"
        )
    print(f"Scoring: {metadata['scoring_label']}")
    print(f"Player-seasons analyzed: {metadata['player_seasons']:,}")
    print(f"Cache file: {metadata['cache_file']}")
    print("")
    print("DEFINITION")
    print("----------")
    print("Rate % = players at that age inside Top N position slots / all Top N position slots.")
    print("Example: Top 3 ages 24, 25, 26 -> each age is 1/3 = 33.33%.")
    print("")
    print("PEAK AGES")
    print("---------")
    for position, study in studies.items():
        print(position)
        for threshold, label in _position_rate_specs(position):
            key = f"{threshold}_rate"
            peak = study["peaks"].get(key)
            if peak:
                print(f"  Peak Age for {label} Rate: Age {peak['age']} ({_fmt_pct(peak['rate_pct'])})")

    print("")
    print("AGE SHARE TABLES")
    print("----------------")
    for position, study in studies.items():
        print(position)
        table_df = _position_table_df(study["summary"], position)
        print(table_df.to_string(index=False))

    print("")
    print(f"Wrote HTML report: {html_path}")
    print(f"Wrote #1 overall comparison: {number_one_html_path}")
    for position, paths in chart_paths.items():
        for path in paths.values():
            print(f"Wrote {position} grouped chart: {path}")
    if any(wrote_matplotlib):
        for position, paths in grouped_png_paths.items():
            for path in paths.values():
                print(f"Wrote Matplotlib {position} grouped chart: {path}")
        print(f"Wrote Matplotlib #1 overall chart: {number_one_chart_path}")
    else:
        print("Matplotlib is not installed, so PNG charts were skipped. SVG charts were still created.")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
