from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable
import webbrowser

import pandas as pd

from rb_elite_age_analysis import pull_historical_seasons_df, resolve_age_study_columns

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


OUTPUT_DIR = Path("case_studies/output/rb_age_share_study")
THRESHOLDS = [
    ("top36", 36, "Top 36"),
    ("top24", 24, "Top 24"),
    ("top12", 12, "Top 12"),
    ("top5", 5, "Top 5"),
    ("top3", 3, "Top 3"),
    ("top1", 1, "#1 Overall"),
]
GROUPS = {
    "starter_depth": {
        "title": "RB Age Share Within Top 36 / Top 24 / Top 12",
        "thresholds": ["top36", "top24", "top12"],
        "file": "rb_age_share_starter_depth.png",
    },
    "elite": {
        "title": "RB Age Share Within Top 5 / Top 3 / #1 Overall",
        "thresholds": ["top5", "top3", "top1"],
        "file": "rb_age_share_elite.png",
    },
}
COLORS = {
    "top36": "#2563eb",
    "top24": "#0891b2",
    "top12": "#16a34a",
    "top5": "#ca8a04",
    "top3": "#ea580c",
    "top1": "#dc2626",
}
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


def _metadata(seasons_df: pd.DataFrame, scoring: str, scoring_label: str) -> Dict[str, object]:
    season_min = int(seasons_df["season"].min()) if "season" in seasons_df.columns and not seasons_df.empty else None
    season_max = int(seasons_df["season"].max()) if "season" in seasons_df.columns and not seasons_df.empty else None
    return {
        "data_source": "nflverse-data GitHub releases: stats_player + rosters",
        "source_releases": "stats_player, rosters",
        "cache_file": seasons_df.attrs.get("source_path", "case_studies/data/rb_elite_age_player_seasons_<scoring>.csv"),
        "timeline": f"{season_min}-{season_max}" if season_min and season_max else "Unknown",
        "scoring": scoring,
        "scoring_label": scoring_label,
        "player_seasons": int(len(seasons_df)),
    }


def build_rb_age_share_study(seasons_df: pd.DataFrame) -> Dict[str, object]:
    columns = resolve_age_study_columns(seasons_df)
    missing = [key for key, column in columns.items() if column is None]
    if seasons_df is None or seasons_df.empty or missing:
        return {"summary": pd.DataFrame(), "peaks": {}, "missing_columns": missing}

    working_df = seasons_df.copy()
    working_df["_position"] = working_df[columns["position"]].astype(str).str.upper()
    working_df["_age"] = pd.to_numeric(working_df[columns["age"]], errors="coerce").round()
    working_df["_positional_finish"] = pd.to_numeric(working_df[columns["positional_finish"]], errors="coerce")
    rb_df = working_df[
        working_df["_position"].eq("RB")
        & working_df["_age"].notna()
        & working_df["_positional_finish"].notna()
    ].copy()
    rb_df["_age"] = rb_df["_age"].astype(int)

    top36_df = rb_df[rb_df["_positional_finish"] <= 36]
    ages = sorted(top36_df["_age"].unique().tolist())
    summary_df = pd.DataFrame({"Age": ages})
    summary_df["rb_seasons_at_age"] = summary_df["Age"].map(rb_df.groupby("_age").size()).fillna(0).astype(int)

    peaks = {}
    for key, threshold, label in THRESHOLDS:
        threshold_df = rb_df[rb_df["_positional_finish"] <= threshold]
        total_slots = int(len(threshold_df))
        counts = threshold_df.groupby("_age").size()
        summary_df[f"{key}_age_count"] = summary_df["Age"].map(counts).fillna(0).astype(int)
        summary_df[f"{key}_total_slots"] = total_slots
        summary_df[f"{key}_rate"] = (
            summary_df[f"{key}_age_count"] / total_slots if total_slots else 0.0
        )
        summary_df[f"{key}_rate_pct"] = (summary_df[f"{key}_rate"] * 100.0).round(2)

        if total_slots and not summary_df.empty:
            peak_row = summary_df.sort_values(
                [f"{key}_rate", f"{key}_age_count", "Age"],
                ascending=[False, False, True],
            ).iloc[0]
            peaks[f"{key}_rate"] = {
                "label": label,
                "age": int(peak_row["Age"]),
                "rate_pct": float(peak_row[f"{key}_rate_pct"]),
                "age_count": int(peak_row[f"{key}_age_count"]),
                "total_slots": total_slots,
            }
        else:
            peaks[f"{key}_rate"] = None

    return {
        "summary": summary_df.sort_values("Age").reset_index(drop=True),
        "peaks": peaks,
        "missing_columns": [],
        "raw_rb_rows": int(len(rb_df)),
    }


def _write_group_chart(summary_df: pd.DataFrame, group: Dict[str, object], output_path: Path) -> bool:
    if plt is None:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    for key, _, label in THRESHOLDS:
        if key not in group["thresholds"]:
            continue
        rate_col = f"{key}_rate_pct"
        ax.plot(
            summary_df["Age"],
            summary_df[rate_col],
            marker="o",
            linewidth=2.5,
            color=COLORS[key],
            label=label,
        )
        mean_rate = float(summary_df[rate_col].mean())
        ax.axhline(
            mean_rate,
            color=COLORS[key],
            linestyle="--",
            linewidth=1.4,
            alpha=0.65,
            label=f"{label} mean {mean_rate:.2f}%",
        )

    ax.set_title(group["title"])
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
    summary_df = study["summary"]
    peaks = study["peaks"]

    peak_items = []
    for key, _, label in THRESHOLDS:
        peak = peaks.get(f"{key}_rate")
        if peak:
            peak_items.append(
                f"<li>{label}: Age {peak['age']} ({peak['rate_pct']:.2f}% of {peak['total_slots']:,} slots)</li>"
            )

    chart_html = "\n".join(
        f'<section class="chart"><img src="{path.name}" alt="{group["title"]}"></section>'
        for group_key, group in GROUPS.items()
        for path in [chart_paths[group_key]]
    )
    table_html = summary_df.to_html(index=False, classes="rate-table", border=0)

    html = f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>RB Age Share Study</title>
    <style>
        body {{ margin: 0; background: #f4f7fb; color: #111827; font-family: Arial, sans-serif; }}
        main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 60px; }}
        h1 {{ margin: 0 0 8px; font-size: 34px; }}
        .subtitle {{ margin: 0 0 20px; color: #4b5563; line-height: 1.5; }}
        .meta {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
        .card, .chart, .table-wrap {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 16px; }}
        .label {{ color: #64748b; font-size: 12px; font-weight: 800; text-transform: uppercase; }}
        .value {{ margin-top: 6px; font-size: 15px; font-weight: 700; line-height: 1.35; }}
        .chart {{ margin: 22px 0; }}
        .chart img {{ width: 100%; height: auto; display: block; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background: #f8fafc; font-weight: 800; }}
    </style>
</head>
<body>
    <main>
        <h1>RB Age Share Study</h1>
        <p class="subtitle">
            This study measures age share inside each elite RB group. Example: if the Top 3 RBs are ages
            24, 25, and 26, then each age receives 1/3, or 33.3%, of the Top 3 age share.
        </p>
        <section class="meta">
            <div class="card"><div class="label">Data Source</div><div class="value">{metadata['data_source']}</div></div>
            <div class="card"><div class="label">Timeline</div><div class="value">{metadata['timeline']}</div></div>
            <div class="card"><div class="label">Scoring</div><div class="value">{metadata['scoring_label']}</div></div>
            <div class="card"><div class="label">Player-Seasons</div><div class="value">{metadata['player_seasons']:,}</div></div>
        </section>
        <p class="subtitle">Source releases: {metadata['source_releases']} | Cache: {metadata['cache_file']}</p>
        <section class="card">
            <h2>Peak Age Shares</h2>
            <ul>{''.join(peak_items)}</ul>
        </section>
        {chart_html}
        <section class="table-wrap">
            <h2>Age Share Table</h2>
            {table_html}
        </section>
    </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _print_table(summary_df: pd.DataFrame) -> None:
    cols = ["Age", "rb_seasons_at_age"]
    for key, _, _ in THRESHOLDS:
        cols.extend([f"{key}_age_count", f"{key}_rate_pct"])
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RB age share within top-N fantasy finishes.")
    parser.add_argument(
        "--scoring",
        choices=["ppr", "full_ppr", "half_ppr", "full", "half", "dad", "dads", "dad_settings"],
        default=None,
    )
    parser.add_argument("--no-prompt", action="store_true", help="Use Full PPR without asking for a scoring choice.")
    parser.add_argument("--refresh", action="store_true", help="Pull source data again instead of using cached data.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML report after the run.")
    args = parser.parse_args()

    scoring, scoring_label = _choose_scoring(args.scoring, args.no_prompt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pulling/caching historical RB player-season data ({scoring_label})...")
    seasons_df = pull_historical_seasons_df(scoring=scoring, force_refresh=args.refresh)
    metadata = _metadata(seasons_df, scoring, scoring_label)

    print("Calculating RB age share within top-N groups...")
    study = build_rb_age_share_study(seasons_df)
    summary_df = study["summary"]
    if summary_df.empty:
        raise SystemExit("No age-share results were generated.")

    table_path = output_dir / "rb_age_share_rates.csv"
    html_path = output_dir / "rb_age_share_study.html"
    chart_paths = {
        group_key: output_dir / group["file"]
        for group_key, group in GROUPS.items()
    }

    summary_df.to_csv(table_path, index=False)
    wrote_charts = {
        group_key: _write_group_chart(summary_df, group, chart_paths[group_key])
        for group_key, group in GROUPS.items()
    }
    _write_html_report(study, metadata, chart_paths, html_path)

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
    print("Example: Top 3 ages 24, 25, 26 -> each age is 1/3 = 33.3%.")

    print("")
    print("PEAK AGE SHARES")
    print("---------------")
    for key, _, label in THRESHOLDS:
        peak = study["peaks"].get(f"{key}_rate")
        if peak:
            print(
                f"{label}: Age {peak['age']} "
                f"({peak['rate_pct']:.2f}% of {peak['total_slots']:,} slots)"
            )

    print("")
    print("AGE SHARE TABLE")
    print("---------------")
    _print_table(summary_df)

    print("")
    print(f"Wrote table: {table_path}")
    print(f"Wrote HTML report: {html_path}")
    for group_key, path in chart_paths.items():
        if wrote_charts[group_key]:
            print(f"Wrote chart: {path}")
        else:
            print(f"Matplotlib is unavailable, chart skipped: {path}")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
